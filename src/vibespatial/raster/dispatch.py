"""VRAM budget functions and band dispatch executors for multiband GPU processing.

Provides utilities to query available GPU memory, compute how many raster
bands can be processed in a single GPU pass, and dispatch per-band operations
across multiband rasters on both GPU and CPU paths.

When CuPy is unavailable, ``available_vram_bytes()`` returns 0 gracefully and
``dispatch_per_band_gpu`` will fail with a clear error at call time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vibespatial.raster.buffers import OwnedRasterArray, RasterMetadata, RasterPlan

__all__ = [
    "available_vram_bytes",
    "max_bands_for_budget",
    "analyze_raster_plan",
    "plan_from_metadata",
    "dispatch_per_band_gpu",
    "dispatch_per_band_cpu",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VRAM_HEADROOM_FRACTION = 0.15
"""Reserve 15 % of effective VRAM as headroom for driver allocations,
fragmentation, and concurrent kernel launches."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_vram_bytes() -> int:
    """Return the effective available VRAM in bytes after headroom.

    When RMM is the active allocator (tiers A/B/C), the function queries
    ``rmm.mr.available_device_memory`` which accounts for pool-managed
    blocks.  Otherwise it falls back to the CuPy pool query.

    A 15 % headroom fraction is subtracted from the effective free memory
    to leave breathing room for the CUDA driver, fragmentation, and any
    concurrent allocations.

    Returns 0 when CuPy is not importable or no CUDA device is available,
    making the function safe to call unconditionally on CPU-only machines.
    """
    try:
        import cupy as cp
    except ImportError:
        return 0

    try:
        # Check if RMM is managing the pool (tiers A/B/C).
        from vibespatial.raster.memory import _active_tier, _configured

        if _configured and _active_tier in ("A", "B", "C"):
            import rmm.mr

            free, _total = rmm.mr.available_device_memory()
            usable = int(free * (1.0 - _VRAM_HEADROOM_FRACTION))
            return max(0, usable)

        # Fallback: CuPy pool query (original logic).
        free, _ = cp.cuda.runtime.memGetInfo()
        pool_free = cp.get_default_memory_pool().free_bytes()
        effective = free + pool_free
        usable = int(effective * (1.0 - _VRAM_HEADROOM_FRACTION))
        return max(0, usable)
    except Exception:
        # Any CUDA runtime failure (no device, driver mismatch, etc.)
        return 0


def max_bands_for_budget(
    height: int,
    width: int,
    dtype: np.dtype,
    buffers_per_band: int = 2,
    scratch_bytes: int = 0,
) -> int:
    """Compute how many raster bands fit in available VRAM.

    Parameters
    ----------
    height, width:
        Spatial dimensions of each band.
    dtype:
        NumPy dtype of the raster (e.g. ``np.float32``).  Used to determine
        per-element byte width via ``dtype.itemsize``.
    buffers_per_band:
        Number of device buffers required per band (default 2 — one input and
        one output buffer).
    scratch_bytes:
        Additional fixed scratch memory consumed by the operation, subtracted
        from the VRAM budget before dividing by per-band cost.

    Returns
    -------
    int
        Maximum number of bands that fit, but always at least 1 so that a
        single-band fallback is always possible.
    """
    dtype = np.dtype(dtype)
    per_band = height * width * dtype.itemsize * buffers_per_band
    if per_band <= 0:
        return 1

    budget = available_vram_bytes() - scratch_bytes
    if budget <= 0:
        return 1

    return max(1, budget // per_band)


# ---------------------------------------------------------------------------
# Raster plan analysis (vibeSpatial-fx3.1)
# ---------------------------------------------------------------------------

# Budget safety factor: reserve 30% of the headroom-adjusted VRAM for runtime
# variance (kernel temporaries, driver bookkeeping, concurrent allocations).
_BUDGET_SAFETY_FACTOR = 0.7

# Default tile target: 4096x4096 is warp-friendly (~64 MB for float32).
_DEFAULT_TILE_DIM = 4096

# Tile dimensions must be multiples of this value for GPU warp alignment.
_TILE_ALIGNMENT = 256

# Minimum tile dimension after alignment (must be >= _TILE_ALIGNMENT).
_MIN_TILE_DIM = _TILE_ALIGNMENT


def analyze_raster_plan(
    height: int,
    width: int,
    dtype: np.dtype,
    *,
    band_count: int = 1,
    buffers_per_band: int = 2,
    scratch_bytes: int = 0,
    halo: int = 0,
    vram_budget: int | None = None,
) -> RasterPlan:
    """Decide the processing strategy for a raster given VRAM constraints.

    Parameters
    ----------
    height, width:
        Spatial dimensions of the raster.
    dtype:
        NumPy dtype (e.g. ``np.float32``).
    band_count:
        Number of bands.
    buffers_per_band:
        Device buffers required per band (default 2: input + output).
    scratch_bytes:
        Additional fixed scratch memory consumed by the operation.
    halo:
        Overlap pixels for stencil/focal operations.  Each tile's effective
        data area is ``(tile_H - 2*halo, tile_W - 2*halo)``.
    vram_budget:
        Available VRAM in bytes.  ``None`` triggers auto-detection via
        :func:`available_vram_bytes`.

    Returns
    -------
    RasterPlan
        Frozen dataclass describing the strategy, tile dimensions, and
        estimated VRAM per tile.
    """
    from vibespatial.raster.buffers import RasterPlan, TilingStrategy

    dtype = np.dtype(dtype)
    pixel_bytes = dtype.itemsize

    # Total bytes for the full raster (all bands, all buffers).
    full_raster_bytes = height * width * pixel_bytes * buffers_per_band * band_count

    # Auto-detect VRAM when caller does not supply an explicit budget.
    if vram_budget is None:
        vram_budget = available_vram_bytes()

    # When VRAM budget is 0 (no GPU), strategy is WHOLE -- tiling is a
    # GPU-side concern.  CPU operations process the raster in one pass.
    if vram_budget <= 0:
        return RasterPlan(
            strategy=TilingStrategy.WHOLE,
            tile_shape=None,
            halo=halo,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )

    usable_budget = int(vram_budget * _BUDGET_SAFETY_FACTOR)

    # Does the full raster (+ scratch) fit within the usable budget?
    if full_raster_bytes + scratch_bytes <= usable_budget:
        return RasterPlan(
            strategy=TilingStrategy.WHOLE,
            tile_shape=None,
            halo=halo,
            n_tiles=0,
            estimated_vram_per_tile=full_raster_bytes + scratch_bytes,
        )

    # --- Tiled strategy ---
    # Start from the default tile dimension and shrink until the tile fits.
    tile_h = min(_DEFAULT_TILE_DIM, height)
    tile_w = min(_DEFAULT_TILE_DIM, width)

    # Align to _TILE_ALIGNMENT (round down, but at least _MIN_TILE_DIM).
    tile_h = max(_MIN_TILE_DIM, (tile_h // _TILE_ALIGNMENT) * _TILE_ALIGNMENT)
    tile_w = max(_MIN_TILE_DIM, (tile_w // _TILE_ALIGNMENT) * _TILE_ALIGNMENT)

    # Shrink tile dims until per-tile VRAM fits in the usable budget.
    # Each tile includes the halo border: physical tile is
    # (tile_h + 2*halo) x (tile_w + 2*halo) pixels.  tile_shape stores
    # the *effective* tile dimensions (without halo); the physical tile
    # read from disk is (tile_h + 2*halo, tile_w + 2*halo).
    while True:
        physical_h = tile_h + 2 * halo
        physical_w = tile_w + 2 * halo
        tile_bytes = (
            physical_h * physical_w * pixel_bytes * buffers_per_band * band_count + scratch_bytes
        )
        if tile_bytes <= usable_budget:
            break
        # Both dimensions at minimum — cannot shrink further.
        if tile_h <= _MIN_TILE_DIM and tile_w <= _MIN_TILE_DIM:
            break
        # Halve the larger dimension (aligned).
        if tile_h >= tile_w:
            tile_h = max(_MIN_TILE_DIM, ((tile_h // 2) // _TILE_ALIGNMENT) * _TILE_ALIGNMENT)
        else:
            tile_w = max(_MIN_TILE_DIM, ((tile_w // 2) // _TILE_ALIGNMENT) * _TILE_ALIGNMENT)

    # Recompute final physical tile dimensions and bytes.
    physical_h = tile_h + 2 * halo
    physical_w = tile_w + 2 * halo
    tile_bytes = (
        physical_h * physical_w * pixel_bytes * buffers_per_band * band_count + scratch_bytes
    )

    # Compute tile count using the effective (non-halo) area.
    effective_h = tile_h
    effective_w = tile_w
    rows_of_tiles = (height + effective_h - 1) // effective_h
    cols_of_tiles = (width + effective_w - 1) // effective_w
    n_tiles = rows_of_tiles * cols_of_tiles

    return RasterPlan(
        strategy=TilingStrategy.TILED,
        tile_shape=(tile_h, tile_w),
        halo=halo,
        n_tiles=n_tiles,
        estimated_vram_per_tile=tile_bytes,
    )


def plan_from_metadata(
    metadata: RasterMetadata,
    *,
    buffers_per_band: int = 2,
    scratch_bytes: int = 0,
    halo: int = 0,
    vram_budget: int | None = None,
) -> RasterPlan:
    """Convenience wrapper: extract dimensions from :class:`RasterMetadata`.

    Parameters
    ----------
    metadata:
        Raster metadata (from :func:`read_raster_metadata` or
        ``raster.metadata``).
    buffers_per_band, scratch_bytes, halo, vram_budget:
        Forwarded to :func:`analyze_raster_plan`.

    Returns
    -------
    RasterPlan
    """
    return analyze_raster_plan(
        height=metadata.height,
        width=metadata.width,
        dtype=metadata.dtype,
        band_count=metadata.band_count,
        buffers_per_band=buffers_per_band,
        scratch_bytes=scratch_bytes,
        halo=halo,
        vram_budget=vram_budget,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _single_band_view_gpu(
    raster: OwnedRasterArray,
    band_index: int,
) -> OwnedRasterArray:
    """Create a single-band OwnedRasterArray sharing device memory (zero-copy).

    The caller is responsible for ensuring the full raster is already on-device
    before calling this helper. The returned raster wraps a 2D CuPy slice of
    the band -- no new H->D transfer is triggered.

    Parameters
    ----------
    raster : OwnedRasterArray
        Multiband raster that is already DEVICE-resident.
    band_index : int
        0-indexed band to extract.

    Returns
    -------
    OwnedRasterArray
        A single-band raster with ``band_count == 1`` and ``ndim == 2``,
        sharing the device buffer of *raster*.
    """
    from vibespatial.raster.buffers import (
        OwnedRasterArray as _ORA,
    )
    from vibespatial.raster.buffers import (
        RasterDeviceState,
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
    )
    from vibespatial.residency import Residency

    # Zero-copy slice on device -- no H->D transfer
    band_device = raster.device_band(band_index)  # 2D CuPy view

    # Construct a lightweight host placeholder (never read -- device is authoritative)
    host_placeholder = np.empty(
        (band_device.shape[0], band_device.shape[1]),
        dtype=raster.dtype,
    )

    return _ORA(
        data=host_placeholder,
        nodata=raster.nodata,
        dtype=raster.dtype,
        affine=raster.affine,
        crs=raster.crs,
        residency=Residency.DEVICE,
        device_state=RasterDeviceState(data=band_device),
        _host_materialized=False,
        diagnostics=[
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.CREATED,
                detail=f"_single_band_view_gpu band={band_index}",
                residency=Residency.DEVICE,
            )
        ],
    )


def _single_band_view_cpu(
    raster: OwnedRasterArray,
    band_index: int,
) -> OwnedRasterArray:
    """Create a single-band OwnedRasterArray from host data.

    Parameters
    ----------
    raster : OwnedRasterArray
        Multiband raster with host-resident data.
    band_index : int
        0-indexed band to extract.

    Returns
    -------
    OwnedRasterArray
        A single-band raster with ``band_count == 1`` and ``ndim == 2``.
    """
    from vibespatial.raster.buffers import from_numpy

    host = raster.to_numpy()
    if host.ndim == 2:
        if band_index != 0:
            raise IndexError(f"single-band raster, got band_index={band_index}")
        band_data = host
    else:
        if band_index < 0 or band_index >= host.shape[0]:
            raise IndexError(
                f"band_index={band_index} out of range for {host.shape[0]}-band raster"
            )
        band_data = host[band_index]

    return from_numpy(
        band_data,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )


# ---------------------------------------------------------------------------
# Band dispatch executors
# ---------------------------------------------------------------------------


def dispatch_per_band_gpu(
    raster: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
    *,
    buffers_per_band: int = 2,
    scratch_bytes: int = 0,
) -> OwnedRasterArray:
    """Apply *op_fn* to each band of *raster* on the GPU, then reassemble.

    For single-band rasters this is a zero-overhead passthrough: *op_fn* is
    called once and its result is returned directly.

    For multiband rasters the full raster is transferred to device once, then
    each band is sliced as a zero-copy 2D view and passed to *op_fn*.  The
    per-band results are assembled via
    :meth:`OwnedRasterArray.from_band_stack`.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single- or multi-band).
    op_fn : Callable[[OwnedRasterArray], OwnedRasterArray]
        Operation to apply per band.  Receives a single-band
        ``OwnedRasterArray`` and must return a single-band
        ``OwnedRasterArray``.
    buffers_per_band : int
        Number of device buffers the operation needs per band (used by
        ``max_bands_for_budget`` for callers that want to pre-plan chunking;
        not consumed directly by this executor).
    scratch_bytes : int
        Fixed scratch memory consumed by the operation (same caveat as
        *buffers_per_band*).

    Returns
    -------
    OwnedRasterArray
        Result raster with the same band count, affine, CRS, and nodata as
        the input (metadata propagation is handled by *from_band_stack*).
    """
    from vibespatial.raster.buffers import (
        OwnedRasterArray as _ORA,
    )
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
    )
    from vibespatial.residency import Residency, TransferTrigger

    t0 = time.perf_counter()

    # -- Single-band fast path: zero overhead --
    if raster.band_count == 1:
        result = op_fn(raster)
        elapsed = time.perf_counter() - t0
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=(f"dispatch_per_band_gpu single-band passthrough elapsed={elapsed:.4f}s"),
                residency=result.residency,
            )
        )
        return result

    # -- Multiband: transfer once, iterate bands --
    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="dispatch_per_band_gpu: transfer full multiband raster to device",
    )

    band_results: list[_ORA] = []
    for band_idx in range(raster.band_count):
        band_view = _single_band_view_gpu(raster, band_idx)
        band_result = op_fn(band_view)
        band_results.append(band_result)

    result = _ORA.from_band_stack(band_results, source=raster)
    elapsed = time.perf_counter() - t0
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_per_band_gpu bands={raster.band_count} "
                f"shape=({raster.band_count},{raster.height},{raster.width}) "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=result.residency,
        )
    )
    return result


def dispatch_per_band_cpu(
    raster: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
) -> OwnedRasterArray:
    """Apply *op_fn* to each band of *raster* on the CPU, then reassemble.

    For single-band rasters this is a zero-overhead passthrough.

    For multiband rasters each band is sliced from the host numpy array,
    wrapped as a single-band ``OwnedRasterArray``, passed to *op_fn*, and
    the per-band results are assembled via
    :meth:`OwnedRasterArray.from_band_stack`.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single- or multi-band).
    op_fn : Callable[[OwnedRasterArray], OwnedRasterArray]
        Operation to apply per band.  Receives a single-band
        ``OwnedRasterArray`` and must return a single-band
        ``OwnedRasterArray``.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same band count, affine, CRS, and nodata as
        the input.
    """
    from vibespatial.raster.buffers import (
        OwnedRasterArray as _ORA,
    )
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
    )

    t0 = time.perf_counter()

    # -- Single-band fast path: zero overhead --
    if raster.band_count == 1:
        result = op_fn(raster)
        elapsed = time.perf_counter() - t0
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=(f"dispatch_per_band_cpu single-band passthrough elapsed={elapsed:.4f}s"),
                residency=result.residency,
            )
        )
        return result

    # -- Multiband: iterate bands on host --
    band_results: list[_ORA] = []
    for band_idx in range(raster.band_count):
        band_view = _single_band_view_cpu(raster, band_idx)
        band_result = op_fn(band_view)
        band_results.append(band_result)

    result = _ORA.from_band_stack(band_results, source=raster)
    elapsed = time.perf_counter() - t0
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_per_band_cpu bands={raster.band_count} "
                f"shape=({raster.band_count},{raster.height},{raster.width}) "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=result.residency,
        )
    )
    return result
