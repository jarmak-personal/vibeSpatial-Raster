"""Hydrological DEM conditioning: sink/depression filling.

CPU baseline uses iterative numpy. GPU path uses custom NVRTC kernels
(kernels/hydrology.py) with shared-memory tiling and convergence detection.

The priority-flood algorithm fills depressions to their spill elevation:
1. Initialize border pixels to own elevation, interior to +infinity.
2. Iteratively propagate: fill[i] = max(elevation[i], min(neighbor_fills)).
3. Converge when no pixel changes.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_device,
    from_numpy,
)
from vibespatial.residency import Residency, TransferTrigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    """Return True if CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu(raster: OwnedRasterArray, threshold: int = 100_000) -> bool:
    """Auto-dispatch heuristic: use GPU when available and image is large enough."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= threshold
    except (ImportError, RuntimeError):
        return False


def _numpy_dtype_to_cuda(dtype: np.dtype) -> str:
    """Map numpy dtype to CUDA type name for kernel templating."""
    mapping = {
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
    }
    return mapping.get(dtype, "double")


# ---------------------------------------------------------------------------
# CPU baseline: iterative sink fill
# ---------------------------------------------------------------------------


def _fill_sinks_cpu(
    raster: OwnedRasterArray,
    *,
    _max_iterations: int | None = None,
) -> OwnedRasterArray:
    """CPU sink fill via iterative numpy propagation.

    Implements the same algorithm as the GPU path:
    border pixels keep their elevation, interior pixels start at +inf,
    then iteratively lowered to max(own_elev, min(neighbor_fills)).

    Parameters
    ----------
    _max_iterations : int or None
        Override for max iteration count (testing only).
    """
    t0 = time.perf_counter()

    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("sink fill requires a single-band raster")
        data = data[0]

    height, width = data.shape
    elev = data.astype(np.float64)

    # Build nodata mask
    nodata_mask = np.zeros((height, width), dtype=bool)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            nodata_mask = np.isnan(data)
        else:
            nodata_mask = data == raster.nodata

    # Initialize fill surface
    fill = np.full_like(elev, np.inf)

    # Border pixels: set to own elevation
    fill[0, :] = elev[0, :]
    fill[-1, :] = elev[-1, :]
    fill[:, 0] = elev[:, 0]
    fill[:, -1] = elev[:, -1]

    # Nodata pixels: set to -inf (barriers)
    fill[nodata_mask] = -np.inf

    # Iterative propagation
    max_iterations = _max_iterations if _max_iterations is not None else max(height + width, 1000)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        changed = False

        # Pad fill with +inf for boundary handling
        padded = np.pad(fill, 1, mode="constant", constant_values=np.inf)

        # Compute min of all 8 neighbors from the padded array
        min_neighbor = np.full_like(fill, np.inf)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                neighbor_slice = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
                # Only consider non-barrier neighbors
                valid_neighbor = np.where(neighbor_slice > -np.inf, neighbor_slice, np.inf)
                min_neighbor = np.minimum(min_neighbor, valid_neighbor)

        # new_fill = max(elevation, min_neighbor)
        new_fill = np.maximum(elev, min_neighbor)

        # Only update interior, non-nodata pixels that can improve
        interior = np.ones((height, width), dtype=bool)
        interior[0, :] = False
        interior[-1, :] = False
        interior[:, 0] = False
        interior[:, -1] = False
        interior[nodata_mask] = False

        update_mask = interior & (new_fill < fill)
        if update_mask.any():
            fill[update_mask] = new_fill[update_mask]
            changed = True

        if not changed:
            converged = True
            break

    if not converged:
        logger.warning(
            "fill_sinks_cpu did NOT converge after %d iterations "
            "(pixels=%d). Result may contain unfilled depressions.",
            iterations,
            height * width,
        )

    # Restore nodata pixels to original values
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            fill[nodata_mask] = np.nan
        else:
            fill[nodata_mask] = raster.nodata

    # Cast back to original dtype
    result_data = fill.astype(raster.dtype)

    elapsed = time.perf_counter() - t0
    result = from_numpy(
        result_data,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"fill_sinks_cpu iterations={iterations} "
                f"converged={converged} "
                f"pixels={height * width} elapsed={elapsed:.3f}s"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    if not converged:
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=(
                    f"WARNING: fill_sinks_cpu did not converge after "
                    f"{iterations} iterations (max_iterations={max_iterations}). "
                    f"Result may contain unfilled depressions."
                ),
                residency=result.residency,
                visible_to_user=True,
            )
        )
    logger.debug(
        "fill_sinks_cpu iterations=%d converged=%s pixels=%d elapsed=%.4fs",
        iterations,
        converged,
        height * width,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# GPU: sink fill (iterative priority-flood)
# ---------------------------------------------------------------------------


def _fill_sinks_gpu(
    raster: OwnedRasterArray,
    *,
    _max_iterations: int | None = None,
) -> OwnedRasterArray:
    """GPU sink fill using iterative NVRTC propagation kernels.

    Uses shared-memory tiled 3x3 stencil kernels with convergence detection.
    Same algorithm as CCL: init -> iterate propagation until fixpoint.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input DEM raster (single-band, float32 or float64).
    _max_iterations : int or None
        Override for max iteration count (testing only).

    Returns
    -------
    OwnedRasterArray
        HOST-resident filled DEM raster.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.hydrology import (
        FILL_INIT_NAMES,
        FILL_PROPAGATE_NAMES,
        get_fill_init_source,
        get_fill_propagate_source,
    )

    t0 = time.perf_counter()
    runtime = get_cuda_runtime()

    # --- Validate band count before any transfer ---
    if raster.band_count != 1:
        raise ValueError("sink fill requires a single-band raster")

    height, width = raster.height, raster.width

    # --- Move data to device (zero-copy: no D->H->D ping-pong) ---
    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="fill_sinks_gpu requires device-resident data",
    )
    d_data = raster.device_data()

    # Squeeze band dimension if 3D -> 2D view
    if d_data.ndim == 3:
        d_data = d_data[0]

    # Determine working dtype for CUDA kernel
    if raster.dtype in (np.dtype("float32"), np.dtype("float64")):
        work_dtype = raster.dtype
    else:
        # Integer DEMs: work in float32
        work_dtype = np.dtype("float32")

    cuda_dtype = _numpy_dtype_to_cuda(work_dtype)

    # Cast to working dtype on device (copy=False avoids allocation when already matching)
    d_elevation = d_data.astype(work_dtype, copy=False).ravel()
    d_fill = cp.empty_like(d_elevation)

    # Build nodata mask entirely on device
    d_nodata_mask = None
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            d_nodata_mask = cp.isnan(d_data).astype(cp.uint8).ravel()
        else:
            d_nodata_mask = (d_data == raster.nodata).astype(cp.uint8).ravel()
        # Always pass the mask — kernel handles per-pixel nodata check via nullptr

    d_changed = cp.zeros(1, dtype=np.int32)

    # Device pointer for nodata mask (0 if no nodata)
    nodata_ptr = d_nodata_mask.data.ptr if d_nodata_mask is not None else 0

    # --- Compile kernels ---
    init_source = get_fill_init_source(cuda_dtype)
    init_key = make_kernel_cache_key(f"fill_init_{cuda_dtype}", init_source)
    init_kernels = runtime.compile_kernels(
        cache_key=init_key,
        source=init_source,
        kernel_names=FILL_INIT_NAMES,
    )

    prop_source = get_fill_propagate_source(cuda_dtype)
    prop_key = make_kernel_cache_key(f"fill_propagate_{cuda_dtype}", prop_source)
    prop_kernels = runtime.compile_kernels(
        cache_key=prop_key,
        source=prop_source,
        kernel_names=FILL_PROPAGATE_NAMES,
    )

    # --- Launch configuration ---
    block_2d = (16, 16, 1)
    grid_2d = ((width + 15) // 16, (height + 15) // 16, 1)

    # --- Phase 1: Initialize fill surface ---
    runtime.launch(
        kernel=init_kernels["fill_init"],
        grid=grid_2d,
        block=block_2d,
        params=(
            (d_elevation.data.ptr, d_fill.data.ptr, nodata_ptr, width, height),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # --- Phase 2: Iterative propagation with batched convergence checks ---
    # Amortize D->H sync cost by checking convergence every BATCH_SIZE iterations.
    # The kernel only writes changed=1 (never resets), so changes accumulate
    # across the batch. We reset d_changed only at batch boundaries.
    CONVERGENCE_BATCH_SIZE = 32
    max_iterations = _max_iterations if _max_iterations is not None else max(height + width, 1000)
    iterations = 0

    prop_params = (
        (
            d_elevation.data.ptr,
            d_fill.data.ptr,
            nodata_ptr,
            width,
            height,
            d_changed.data.ptr,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        ),
    )

    converged = False
    d_changed.fill(0)
    for iterations in range(1, max_iterations + 1):
        runtime.launch(
            kernel=prop_kernels["fill_propagate"],
            grid=grid_2d,
            block=block_2d,
            params=prop_params,
        )

        # Check convergence only at batch boundaries
        if iterations % CONVERGENCE_BATCH_SIZE == 0:
            if int(d_changed.item()) == 0:
                converged = True
                break
            d_changed.fill(0)
    else:
        # Loop exhausted without break — perform a final convergence check
        # for iterations since the last batch boundary.
        if int(d_changed.item()) == 0:
            converged = True

    if not converged:
        logger.warning(
            "fill_sinks_gpu did NOT converge after %d iterations "
            "(pixels=%d). Result may contain unfilled depressions.",
            iterations,
            height * width,
        )

    # --- Restore nodata in fill array ---
    if d_nodata_mask is not None:
        # Use CuPy vectorized op to restore nodata values on device
        d_nodata_bool = d_nodata_mask.astype(bool)
        if raster.nodata is not None and np.isnan(raster.nodata):
            d_fill[d_nodata_bool] = cp.nan
        elif raster.nodata is not None:
            nodata_val = work_dtype.type(raster.nodata)
            d_fill[d_nodata_bool] = nodata_val

    # --- Keep result on device (zero-copy) ---
    d_fill_2d = d_fill.reshape(height, width).astype(raster.dtype)

    elapsed = time.perf_counter() - t0
    result = from_device(
        d_fill_2d,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"fill_sinks_gpu iterations={iterations} "
                f"converged={converged} "
                f"pixels={height * width} dtype={cuda_dtype} "
                f"grid={grid_2d} elapsed={elapsed:.3f}s"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    if not converged:
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=(
                    f"WARNING: fill_sinks_gpu did not converge after "
                    f"{iterations} iterations (max_iterations={max_iterations}). "
                    f"Result may contain unfilled depressions."
                ),
                residency=result.residency,
                visible_to_user=True,
            )
        )
    logger.debug(
        "fill_sinks_gpu iterations=%d converged=%s pixels=%d dtype=%s elapsed=%.4fs",
        iterations,
        converged,
        height * width,
        cuda_dtype,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Public API: dispatcher
# ---------------------------------------------------------------------------


def raster_fill_sinks(
    raster: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Fill sinks/depressions in a DEM raster to their spill elevation.

    Uses a priority-flood algorithm: border pixels anchor the fill surface,
    and interior depressions are iteratively raised to the level at which
    water would spill out. This is essential preprocessing for hydrological
    analysis (flow direction, flow accumulation, watershed delineation).

    Parameters
    ----------
    raster : OwnedRasterArray
        Input DEM raster (single-band). Supports float32, float64, and
        integer dtypes (integers are promoted to float32 internally).
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when available and pixel count exceeds threshold.

    Returns
    -------
    OwnedRasterArray
        HOST-resident DEM with all depressions filled to spill elevation.
        Nodata pixels are preserved unchanged.

    Notes
    -----
    The GPU implementation uses shared-memory tiled NVRTC kernels with
    iterative convergence detection (same pattern as connected component
    labeling). For large rasters, this provides significant speedup over
    the CPU baseline.

    Examples
    --------
    >>> dem = from_numpy(elevation_data, nodata=-9999.0)
    >>> filled = raster_fill_sinks(dem)
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        return _fill_sinks_gpu(raster)
    else:
        return _fill_sinks_cpu(raster)
