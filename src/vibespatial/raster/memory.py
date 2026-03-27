"""Tiered GPU memory pool manager (ADR-0040).

Provides a three-tier RMM-based memory pool with CuPy fallback:

| Tier | Activation                           | Allocator Stack                              |
|------|--------------------------------------|----------------------------------------------|
| A    | VIBESPATIAL_GPU_POOL_ONLY=1          | PoolMemoryResource -> CudaMemoryResource     |
| B    | RMM installed (default)              | FailureCallback -> Pool -> Cuda              |
| C    | VIBESPATIAL_GPU_MANAGED_MEMORY=1     | Bare ManagedMemoryResource                   |
| Fall | RMM not installed                    | CuPy MemoryPool (default)                    |

Tier B is the default because its OOM callback provides robust recovery
from pool fragmentation.  The callback first tries ``gc.collect()`` to
free unreferenced CuPy arrays; if that fails, it tears down and rebuilds
the pool to release all held blocks back to the CUDA driver, then retries.

Tier A (pool-only, no OOM callback) is opt-in via ``VIBESPATIAL_GPU_POOL_ONLY=1``
for users who want maximum performance and are willing to manage memory
pressure themselves.

All imports are lazy -- this module does not import CuPy or RMM at module
level.  The pool is configured exactly once on the first GPU operation via
``_ensure_memory_pool()``.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time

__all__ = [
    "configure_memory_pool",
    "memory_pool_stats",
    "free_pool_memory",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_MANAGED_MEMORY_THRESHOLD = 0.50
"""When less than 50 % of total VRAM is free at init time, auto-select
Tier C (managed memory) to avoid OOM on shared GPUs.  Set to 0.0 to
disable auto-detection."""

_POOL_MAX_SIZE_FRACTION = 0.80
"""Tier A maximum pool size as a fraction of total VRAM.  Prevents the
pool from consuming all device memory, leaving headroom for non-pool
allocations (driver overhead, CUDA contexts, etc.)."""

# ---------------------------------------------------------------------------
# Module-level state (guarded by a lock for thread safety)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_configured = False
_active_tier: str = ""
_stats_adaptor: object | None = None  # StatisticsResourceAdaptor when Tier A/B


# ---------------------------------------------------------------------------
# Tier B: OOM callback with gc.collect + pool rebuild + retry
# ---------------------------------------------------------------------------

_OOM_GC_RETRIES = 3
"""Number of gc.collect()-only retries before escalating to pool rebuild."""

_OOM_MAX_RETRIES = 5
"""Total retry budget including pool-rebuild attempts."""

_OOM_COOLDOWN_SECONDS = 1.0
_oom_last_retry_time: float = 0.0
_oom_retry_count: int = 0


def _oom_callback(nbytes: int) -> bool:
    """Failure callback for Tier B.

    Called by RMM's ``FailureCallbackResourceAdaptor`` when an allocation
    fails.  The recovery strategy escalates:

    1. Retries 1-3: ``gc.collect()`` to release unreferenced CuPy arrays
       (which frees their underlying RMM allocations), then retry.
    2. Retries 4-5: Tear down and rebuild the entire RMM pool via
       ``free_pool_memory()``, releasing all held blocks back to the
       CUDA driver.  This is heavier but handles pool fragmentation
       where ``gc.collect()`` alone cannot coalesce free blocks into
       the contiguous region needed.

    The callback returns True to retry the allocation or False to raise
    ``MemoryError``.

    Time-based reset: if >1 s has elapsed since the last retry burst,
    the retry counter resets so transient spikes get fresh retries.
    """
    global _oom_last_retry_time, _oom_retry_count

    now = time.monotonic()

    # Reset counter if enough time has passed since the last retry
    if now - _oom_last_retry_time > _OOM_COOLDOWN_SECONDS:
        _oom_retry_count = 0

    if _oom_retry_count >= _OOM_MAX_RETRIES:
        logger.warning(
            "OOM callback: exhausted %d retries for %d-byte allocation",
            _OOM_MAX_RETRIES,
            nbytes,
        )
        return False

    _oom_retry_count += 1
    _oom_last_retry_time = now

    if _oom_retry_count <= _OOM_GC_RETRIES:
        # Phase 1: lightweight -- gc.collect() releases unreferenced arrays
        logger.debug(
            "OOM callback: gc retry %d/%d for %d bytes",
            _oom_retry_count,
            _OOM_GC_RETRIES,
            nbytes,
        )
        gc.collect()
    else:
        # Phase 2: heavy -- rebuild the pool to defragment
        logger.warning(
            "OOM callback: pool rebuild retry %d/%d for %d bytes "
            "(gc.collect retries exhausted, rebuilding pool)",
            _oom_retry_count,
            _OOM_MAX_RETRIES,
            nbytes,
        )
        gc.collect()
        _rebuild_pool()

    return True


def _rebuild_pool() -> None:
    """Tear down and rebuild the Tier B pool to release fragmented blocks.

    This is called from the OOM callback when gc.collect() alone cannot
    free enough contiguous memory.  The PoolMemoryResource holds onto
    freed blocks and cannot always coalesce them; rebuilding forces all
    free blocks back to the CUDA driver.

    IMPORTANT: This rebuilds the pool *in place* by replacing the
    upstream PoolMemoryResource inside the existing adaptor chain.
    The FailureCallbackResourceAdaptor and StatisticsResourceAdaptor
    wrappers remain valid because RMM re-wires them on
    set_current_device_resource().
    """
    try:
        from rmm.allocators.cupy import rmm_cupy_allocator

        logger.debug("_rebuild_pool: tearing down and rebuilding Tier B pool")
        _configure_tier_b(rmm_cupy_allocator)
    except Exception:
        logger.debug("_rebuild_pool: pool rebuild failed", exc_info=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_memory_pool() -> str:
    """Configure the GPU memory pool.  Returns the tier name.

    This function is idempotent: calling it multiple times returns the
    previously-configured tier without reconfiguring.

    Tier selection priority:
      1. ``VIBESPATIAL_GPU_MANAGED_MEMORY=1`` -> Tier C
      2. ``VIBESPATIAL_GPU_POOL_ONLY=1``      -> Tier A (no OOM recovery)
      3. RMM importable                       -> Tier B (default, with OOM recovery)
      4. Fallback                             -> CuPy default pool
    """
    global _configured, _active_tier, _stats_adaptor

    with _lock:
        if _configured:
            return _active_tier

        tier = _configure_unlocked()
        _configured = True
        _active_tier = tier
        logger.info("GPU memory pool configured: tier %s", tier)
        return tier


def _configure_unlocked() -> str:
    """Internal configuration logic (caller holds ``_lock``)."""
    global _stats_adaptor

    env_managed = os.environ.get("VIBESPATIAL_GPU_MANAGED_MEMORY", "0") == "1"
    env_pool_only = os.environ.get("VIBESPATIAL_GPU_POOL_ONLY", "0") == "1"

    # Try RMM first
    try:
        import rmm.mr
        from rmm.allocators.cupy import rmm_cupy_allocator
    except ImportError:
        # RMM not available -- fall back to CuPy default pool
        return _configure_fallback()

    # Tier C: bare ManagedMemoryResource (cudaMallocManaged)
    if env_managed:
        return _configure_tier_c(rmm_cupy_allocator)

    # Tier A: Pool-only (no OOM callback) -- opt-in for max performance
    if env_pool_only:
        return _configure_tier_a(rmm_cupy_allocator)

    # Auto-detect: if less than 50% of total VRAM is free, use Tier C
    # (managed memory) to avoid OOM on shared GPUs.  When the GPU is
    # heavily loaded by other processes, pool-based allocation will fail
    # on large raster operations whose working set exceeds free VRAM.
    # Managed memory uses CUDA Unified Memory which can oversubscribe
    # physical VRAM via OS page migration.
    try:
        free, total = rmm.mr.available_device_memory()
        if total > 0 and (free / total) < _MANAGED_MEMORY_THRESHOLD:
            logger.info(
                "Auto-selecting Tier C (managed memory): only %.0f%% of "
                "VRAM is free (%.1f GB / %.1f GB)",
                100.0 * free / total,
                free / 1e9,
                total / 1e9,
            )
            return _configure_tier_c(rmm_cupy_allocator)
    except Exception:
        pass

    # Tier B: Pool with OOM callback (default -- safe for all workloads)
    return _configure_tier_b(rmm_cupy_allocator)


def _configure_tier_a(rmm_cupy_allocator) -> str:
    """Tier A: PoolMemoryResource -> CudaMemoryResource.

    Uses ``initial_pool_size=0`` so the pool grows on demand without
    starving other processes.  Sets ``maximum_pool_size`` to 80% of
    total VRAM to leave headroom for non-pool allocations.
    """
    global _stats_adaptor

    import rmm.mr

    # Cap the pool at 80% of total VRAM to prevent the pool from
    # consuming everything and leaving no headroom for CUDA driver
    # overhead, contexts, or non-pool allocations.
    max_pool_size = 0
    try:
        _free, total = rmm.mr.available_device_memory()
        max_pool_size = int(total * _POOL_MAX_SIZE_FRACTION)
    except Exception:
        pass  # Fall through with 0 (unlimited) if we can't query

    cuda_mr = rmm.mr.CudaMemoryResource()
    pool_kwargs: dict = {"initial_pool_size": 0}
    if max_pool_size > 0:
        pool_kwargs["maximum_pool_size"] = max_pool_size
    pool_mr = rmm.mr.PoolMemoryResource(cuda_mr, **pool_kwargs)
    stats_mr = rmm.mr.StatisticsResourceAdaptor(pool_mr)

    rmm.mr.set_current_device_resource(stats_mr)
    _stats_adaptor = stats_mr

    _set_cupy_allocator(rmm_cupy_allocator)
    return "A"


def _configure_tier_b(rmm_cupy_allocator) -> str:
    """Tier B: FailureCallbackResourceAdaptor -> Pool -> Cuda.

    The OOM callback runs ``gc.collect()`` to free unreferenced CuPy
    arrays (which returns memory to the pool), then signals a retry.
    After exhausting retries, the allocation fails with ``MemoryError``.
    """
    global _stats_adaptor

    import rmm.mr

    cuda_mr = rmm.mr.CudaMemoryResource()
    pool_mr = rmm.mr.PoolMemoryResource(cuda_mr, initial_pool_size=0)
    fail_mr = rmm.mr.FailureCallbackResourceAdaptor(pool_mr, _oom_callback)
    stats_mr = rmm.mr.StatisticsResourceAdaptor(fail_mr)

    rmm.mr.set_current_device_resource(stats_mr)
    _stats_adaptor = stats_mr

    _set_cupy_allocator(rmm_cupy_allocator)
    return "B"


def _configure_tier_c(rmm_cupy_allocator) -> str:
    """Tier C: bare ManagedMemoryResource (no pool wrapping)."""
    global _stats_adaptor

    import rmm.mr

    managed_mr = rmm.mr.ManagedMemoryResource()
    stats_mr = rmm.mr.StatisticsResourceAdaptor(managed_mr)

    rmm.mr.set_current_device_resource(stats_mr)
    _stats_adaptor = stats_mr

    _set_cupy_allocator(rmm_cupy_allocator)
    return "C"


def _configure_fallback() -> str:
    """Fallback: CuPy default MemoryPool (no RMM)."""
    # CuPy's default pool is already active -- nothing to configure.
    return "fallback"


def _set_cupy_allocator(rmm_cupy_allocator) -> None:
    """Point CuPy at the RMM allocator."""
    import cupy as cp

    cp.cuda.set_allocator(rmm_cupy_allocator)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def memory_pool_stats() -> dict:
    """Return current pool statistics.

    Keys always present:
      - ``tier``:  active tier name ("A", "B", "C", "fallback", or "" if
        not yet configured)
      - ``configured``:  whether the pool has been initialized

    When RMM is active (tiers A/B/C):
      - ``current_bytes``:  bytes currently allocated
      - ``current_count``:  number of live allocations
      - ``peak_bytes``:     high-water-mark bytes
      - ``peak_count``:     high-water-mark allocation count
      - ``total_bytes``:    cumulative bytes allocated (lifetime)
      - ``total_count``:    cumulative allocation count (lifetime)
      - ``pool_size``:      current RMM pool size (tier A/B only)

    When using the CuPy fallback:
      - ``used_bytes``:     bytes in use by CuPy pool
      - ``free_bytes``:     bytes held by CuPy pool but currently free
      - ``total_bytes``:    used + free
    """
    result: dict = {
        "tier": _active_tier,
        "configured": _configured,
    }

    if not _configured:
        return result

    if _stats_adaptor is not None:
        # RMM tiers (A, B, C)
        counts = _stats_adaptor.allocation_counts  # type: ignore[union-attr]
        result["current_bytes"] = counts.current_bytes
        result["current_count"] = counts.current_count
        result["peak_bytes"] = counts.peak_bytes
        result["peak_count"] = counts.peak_count
        result["total_bytes"] = counts.total_bytes
        result["total_count"] = counts.total_count

        # Pool size is only meaningful for tiers with a pool underneath
        if _active_tier in ("A", "B"):
            try:
                import rmm.mr

                resource = rmm.mr.get_current_device_resource()
                # Walk up the adaptor chain to find the PoolMemoryResource
                upstream = resource
                while hasattr(upstream, "get_upstream"):
                    upstream = upstream.get_upstream()
                    if hasattr(upstream, "pool_size"):
                        result["pool_size"] = upstream.pool_size()
                        break
            except Exception:
                pass
    else:
        # CuPy fallback
        try:
            import cupy as cp

            pool = cp.get_default_memory_pool()
            result["used_bytes"] = pool.used_bytes()
            result["free_bytes"] = pool.free_bytes()
            result["total_bytes"] = pool.used_bytes() + pool.free_bytes()
        except Exception:
            pass

    return result


def free_pool_memory() -> None:
    """Release unused memory back to the driver.

    For RMM pool tiers (A/B), this tears down the current pool and
    rebuilds it so that all unused blocks are returned to the CUDA
    driver.  ``PoolMemoryResource`` does not support incremental shrink,
    so a full rebuild is the only way to reclaim fragmented pool memory
    on a shared GPU.

    For Tier C (managed memory) this is a no-op since the OS manages
    page migration.

    For the CuPy fallback, calls ``free_all_blocks()`` on CuPy's pool.

    This function is safe to call between independent GPU operations to
    reduce memory pressure.  It must NOT be called while device arrays
    are still live -- all CuPy/RMM allocations must be freed first.
    """
    global _stats_adaptor

    if not _configured:
        return

    if _active_tier == "fallback":
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return

    if _active_tier in ("A", "B"):
        # Rebuild the pool to release all free blocks back to the driver.
        try:
            from rmm.allocators.cupy import rmm_cupy_allocator

            if _active_tier == "A":
                _configure_tier_a(rmm_cupy_allocator)
            else:
                _configure_tier_b(rmm_cupy_allocator)
            logger.debug("free_pool_memory: rebuilt tier %s pool", _active_tier)
        except Exception:
            logger.debug("free_pool_memory: pool rebuild failed", exc_info=True)


# ---------------------------------------------------------------------------
# Deferred initialization hook
# ---------------------------------------------------------------------------


def _ensure_memory_pool() -> None:
    """Ensure the memory pool is configured.

    This is a fast-path check: if already configured, it returns
    immediately without acquiring the lock.  Called from
    ``OwnedRasterArray._ensure_device_state`` on every H->D transfer.
    """
    if _configured:
        return
    configure_memory_pool()
