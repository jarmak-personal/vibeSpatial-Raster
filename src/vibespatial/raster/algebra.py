"""GPU raster algebra: local and focal operations.

Local operations use CuPy element-wise broadcasting.
Focal operations use custom NVRTC shared-memory tiled stencil kernels.

ADR-0039: GPU Raster Algebra Dispatch
"""

from __future__ import annotations

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
)
from vibespatial.residency import Residency, TransferTrigger

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


def _to_device_data(raster: OwnedRasterArray):
    """Ensure raster is device-resident and return device data."""
    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="raster algebra requires device-resident data",
    )
    return raster.device_data()


def _binary_op(a: OwnedRasterArray, b: OwnedRasterArray, op_name: str, op_func):
    """Apply a binary element-wise operation on two rasters."""
    if a.shape != b.shape:
        raise ValueError(f"raster shapes must match for {op_name}: {a.shape} vs {b.shape}")

    import cupy as cp

    da = _to_device_data(a)
    db = _to_device_data(b)

    result_device = op_func(da, db)

    # Nodata propagation: if either input is nodata, output is nodata
    nodata = a.nodata if a.nodata is not None else b.nodata
    if nodata is not None:
        mask_a = a.device_nodata_mask()
        mask_b = b.device_nodata_mask()
        combined_mask = cp.logical_or(mask_a, mask_b)
        if combined_mask.any():
            result_device = cp.where(combined_mask, nodata, result_device)

    # Build result as HOST with device state already populated
    host_data = cp.asnumpy(result_device)
    result = from_numpy(host_data, nodata=nodata, affine=a.affine, crs=a.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_{op_name} shape={a.shape} dtype={a.dtype}",
            residency=Residency.HOST,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Local raster algebra (element-wise via CuPy)
# ---------------------------------------------------------------------------


def raster_add(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise addition of two rasters."""
    import cupy as cp

    return _binary_op(a, b, "add", cp.add)


def raster_subtract(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise subtraction of two rasters."""
    import cupy as cp

    return _binary_op(a, b, "subtract", cp.subtract)


def raster_multiply(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise multiplication of two rasters."""
    import cupy as cp

    return _binary_op(a, b, "multiply", cp.multiply)


def raster_divide(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise division of two rasters. Division by zero yields nodata."""
    import cupy as cp

    def safe_divide(da, db):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = cp.true_divide(da, db)
        # Replace inf/nan from div-by-zero with nodata
        nodata_val = (
            a.nodata if a.nodata is not None else (b.nodata if b.nodata is not None else 0.0)
        )
        bad = cp.logical_or(cp.isinf(result), cp.isnan(result))
        result = cp.where(bad, nodata_val, result)
        return result

    return _binary_op(a, b, "divide", safe_divide)


def raster_apply(
    raster: OwnedRasterArray,
    func,
    *,
    nodata: float | int | None = None,
) -> OwnedRasterArray:
    """Apply an arbitrary element-wise function to a raster on GPU.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    func : callable
        Function that accepts a CuPy array and returns a CuPy array.
    nodata : float | int | None
        Nodata value for the output. If None, inherits from input.
    """
    import cupy as cp

    d = _to_device_data(raster)
    result_device = func(d)

    out_nodata = nodata if nodata is not None else raster.nodata
    if out_nodata is not None and raster.nodata is not None:
        mask = raster.device_nodata_mask()
        if mask.any():
            result_device = cp.where(mask, out_nodata, result_device)

    host_data = cp.asnumpy(result_device)
    return from_numpy(host_data, nodata=out_nodata, affine=raster.affine, crs=raster.crs)


def raster_where(
    condition: OwnedRasterArray,
    true_val: OwnedRasterArray | float | int,
    false_val: OwnedRasterArray | float | int,
) -> OwnedRasterArray:
    """Element-wise conditional selection.

    Parameters
    ----------
    condition : OwnedRasterArray
        Boolean-like raster (nonzero = True).
    true_val, false_val : OwnedRasterArray or scalar
        Values to use where condition is True/False.
    """
    import cupy as cp

    cond_d = _to_device_data(condition)
    cond_bool = cond_d.astype(cp.bool_)

    if isinstance(true_val, OwnedRasterArray):
        tv = _to_device_data(true_val)
    else:
        tv = true_val

    if isinstance(false_val, OwnedRasterArray):
        fv = _to_device_data(false_val)
    else:
        fv = false_val

    result_device = cp.where(cond_bool, tv, fv)
    host_data = cp.asnumpy(result_device)

    nodata = condition.nodata
    return from_numpy(host_data, nodata=nodata, affine=condition.affine, crs=condition.crs)


def raster_classify(
    raster: OwnedRasterArray,
    bins: list[float],
    labels: list[int | float],
) -> OwnedRasterArray:
    """Reclassify raster values into discrete classes.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    bins : list[float]
        Bin edges (N edges define N-1 bins). Values below bins[0] get labels[0],
        values in [bins[i], bins[i+1]) get labels[i+1], etc.
    labels : list[int | float]
        Class labels. Must have len(bins) + 1 elements.
    """
    import cupy as cp

    if len(labels) != len(bins) + 1:
        raise ValueError(
            f"labels must have len(bins)+1={len(bins) + 1} elements, got {len(labels)}"
        )

    d = _to_device_data(raster)
    bins_d = cp.asarray(bins, dtype=d.dtype)
    labels_d = cp.asarray(labels, dtype=cp.float64)

    indices = cp.digitize(d.ravel(), bins_d).reshape(d.shape)
    result_device = labels_d[indices]

    # Preserve nodata
    if raster.nodata is not None:
        mask = raster.device_nodata_mask()
        if mask.any():
            result_device = cp.where(mask, raster.nodata, result_device)

    host_data = cp.asnumpy(result_device)
    return from_numpy(
        host_data.astype(np.float64),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )


# ---------------------------------------------------------------------------
# Focal raster operations (NVRTC stencil kernels)
# ---------------------------------------------------------------------------

# Tile dimensions must match the #define TILE_W/TILE_H in kernel sources
_TILE_W = 16
_TILE_H = 16


def _convolve_shared_mem_bytes(kw: int, kh: int) -> int:
    """Calculate shared memory bytes needed for the tiled convolution kernel.

    Layout (contiguous in ``extern __shared__ char _smem[]``):
      data tile   : (TILE_H + 2*pad_y) * (TILE_W + 2*pad_x + 1) doubles  (+1 bank padding)
      kweights    : kh * kw doubles
      nodata tile : (TILE_H + 2*pad_y) * (TILE_W + 2*pad_x) uint8s
    """
    pad_x = kw // 2
    pad_y = kh // 2
    smem_cols = _TILE_W + 2 * pad_x + 1  # +1 for bank conflict avoidance
    tile_doubles = (_TILE_H + 2 * pad_y) * smem_cols
    kweights_doubles = kh * kw
    # nodata tile: 1 byte per element, no bank padding needed
    nodata_tile_bytes = (_TILE_H + 2 * pad_y) * (_TILE_W + 2 * pad_x)
    return (tile_doubles + kweights_doubles) * 8 + nodata_tile_bytes


def _gpu_convolve(raster: OwnedRasterArray, kernel_weights: np.ndarray) -> OwnedRasterArray:
    """Run a 2D convolution on GPU via shared-memory tiled NVRTC kernel.

    The kernel uses shared memory for both the input tile (with halo) and the
    kernel weights. Each thread block loads a TILE_WxTILE_H region plus
    pad_x/pad_y halo cells, then all threads read from shared memory for the
    convolution, achieving O(1) global memory reads per output pixel regardless
    of kernel size.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import CONVOLVE_NORMALIZED_KERNEL_SOURCE

    # Move to device and cast to float64 for computation
    d_data = _to_device_data(raster).astype(cp.float64)
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape
    kh, kw = kernel_weights.shape
    pad_y, pad_x = kh // 2, kw // 2

    d_input = d_data
    d_output = cp.zeros_like(d_input)
    d_kernel = cp.asarray(kernel_weights.astype(np.float64))

    nodata_val = float(raster.nodata) if raster.nodata is not None else 0.0

    if raster.nodata is not None:
        d_nodata = raster.device_nodata_mask().astype(cp.uint8)
        nodata_ptr = d_nodata.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("convolve_normalized", CONVOLVE_NORMALIZED_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=CONVOLVE_NORMALIZED_KERNEL_SOURCE,
        kernel_names=("convolve_normalized",),
    )

    # Block size is fixed at (_TILE_W, _TILE_H) because the kernel source
    # uses #define TILE_W/TILE_H and indexes shared memory relative to
    # threadIdx.  Validate via occupancy API that the hardware can schedule
    # this block size; fall back to _TILE_W * _TILE_H if the API is
    # unavailable.
    shared_mem_bytes = _convolve_shared_mem_bytes(kw, kh)
    optimal = runtime.optimal_block_size(
        kernels["convolve_normalized"], shared_mem_bytes=shared_mem_bytes
    )
    required = _TILE_W * _TILE_H
    if optimal < required:
        raise RuntimeError(
            f"Kernel requires block size {required} ({_TILE_W}x{_TILE_H}) but "
            f"occupancy API reports max {optimal} threads/block for "
            f"{shared_mem_bytes} bytes shared memory.  Reduce TILE_W/TILE_H "
            f"or kernel size (current: {kw}x{kh})."
        )

    block = (_TILE_W, _TILE_H, 1)
    grid = (
        (width + _TILE_W - 1) // _TILE_W,
        (height + _TILE_H - 1) // _TILE_H,
        1,
    )

    params = (
        (
            d_input.data.ptr,
            d_output.data.ptr,
            d_kernel.data.ptr,
            nodata_ptr,
            width,
            height,
            kw,
            kh,
            pad_x,
            pad_y,
            nodata_val,
        ),
        (
            KERNEL_PARAM_PTR,  # input
            KERNEL_PARAM_PTR,  # output
            KERNEL_PARAM_PTR,  # kernel_weights
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
            KERNEL_PARAM_I32,  # kw
            KERNEL_PARAM_I32,  # kh
            KERNEL_PARAM_I32,  # pad_x
            KERNEL_PARAM_I32,  # pad_y
            KERNEL_PARAM_F64,  # nodata_val
        ),
    )

    runtime.launch(
        kernel=kernels["convolve_normalized"],
        grid=grid,
        block=block,
        params=params,
        shared_mem_bytes=shared_mem_bytes,
    )

    host_result = cp.asnumpy(d_output)
    result = from_numpy(
        host_result,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(f"gpu_convolve {width}x{height} kernel={kw}x{kh} smem={shared_mem_bytes}B"),
            residency=Residency.HOST,
        )
    )
    return result


def raster_convolve(
    raster: OwnedRasterArray,
    kernel: np.ndarray,
) -> OwnedRasterArray:
    """Apply a 2D convolution kernel to a raster on GPU.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    kernel : np.ndarray
        2D convolution kernel (e.g., 3x3, 5x5).
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D, got {kernel.ndim}D")
    return _gpu_convolve(raster, kernel)


def raster_gaussian_filter(
    raster: OwnedRasterArray,
    sigma: float,
    *,
    kernel_size: int | None = None,
) -> OwnedRasterArray:
    """Apply a Gaussian filter to a raster on GPU.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    sigma : float
        Standard deviation of the Gaussian.
    kernel_size : int or None
        Size of the kernel. Default: 2 * ceil(3*sigma) + 1.
    """
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    ax = np.arange(kernel_size) - kernel_size // 2
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_2d = np.outer(gauss_1d, gauss_1d)
    kernel_2d /= kernel_2d.sum()

    return _gpu_convolve(raster, kernel_2d)


# ---------------------------------------------------------------------------
# CPU slope/aspect via numpy Horn method (fallback)
# ---------------------------------------------------------------------------


def _cpu_slope_aspect(
    dem: OwnedRasterArray,
    *,
    compute_slope: bool,
    compute_aspect: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute slope and/or aspect on CPU using the Horn method with numpy.

    Uses np.gradient for central-difference computation, equivalent to
    a 3x3 Horn stencil. Nodata pixels are propagated: any pixel whose
    3x3 neighbourhood contains nodata receives nodata in the output.

    Returns (slope_host, aspect_host) numpy arrays, either may be None.
    """
    data = dem.to_numpy()
    if data.ndim == 3:
        data = data[0]

    data = data.astype(np.float64)
    height, width = data.shape

    cell_x = abs(dem.affine[0]) if dem.affine[0] != 0 else 1.0
    cell_y = abs(dem.affine[4]) if dem.affine[4] != 0 else 1.0

    nodata_val = float(dem.nodata) if dem.nodata is not None else None

    # Build nodata mask (True where nodata)
    if nodata_val is not None:
        nodata_mask = data == nodata_val
    else:
        nodata_mask = np.zeros_like(data, dtype=bool)

    # Pad with edge values for gradient computation
    padded = np.pad(data, 1, mode="edge")
    # Propagate nodata into padded array's mask
    nodata_padded = np.pad(nodata_mask, 1, mode="edge")

    # Horn method: compute dz/dx and dz/dy using 3x3 neighbourhood
    # dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_x)
    # dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_y)
    # where the 3x3 window is:
    #   a b c
    #   d e f
    #   g h i
    a = padded[0:-2, 0:-2]
    b = padded[0:-2, 1:-1]
    c = padded[0:-2, 2:]
    d = padded[1:-1, 0:-2]
    # e = padded[1:-1, 1:-1]  # center, not used in Horn gradients
    f = padded[1:-1, 2:]
    g = padded[2:, 0:-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_x)
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell_y)

    # Any pixel whose 3x3 window touches nodata gets nodata in output
    nd_a = nodata_padded[0:-2, 0:-2]
    nd_b = nodata_padded[0:-2, 1:-1]
    nd_c = nodata_padded[0:-2, 2:]
    nd_d = nodata_padded[1:-1, 0:-2]
    nd_e = nodata_padded[1:-1, 1:-1]
    nd_f = nodata_padded[1:-1, 2:]
    nd_g = nodata_padded[2:, 0:-2]
    nd_h = nodata_padded[2:, 1:-1]
    nd_i = nodata_padded[2:, 2:]
    neighbourhood_nodata = nd_a | nd_b | nd_c | nd_d | nd_e | nd_f | nd_g | nd_h | nd_i

    slope_host = None
    aspect_host = None

    if compute_slope:
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        if nodata_val is not None:
            slope_deg[neighbourhood_nodata] = nodata_val
        slope_host = slope_deg

    if compute_aspect:
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_deg = np.degrees(aspect_rad)
        # Convert from math convention (0=east, CCW) to geographic (0=north, CW)
        aspect_deg = (90.0 - aspect_deg) % 360.0
        if nodata_val is not None:
            aspect_deg[neighbourhood_nodata] = nodata_val
        aspect_host = aspect_deg

    return slope_host, aspect_host


# ---------------------------------------------------------------------------
# Fused slope/aspect via NVRTC kernel (zero-copy, single-pass)
# ---------------------------------------------------------------------------


def _gpu_slope_aspect(
    dem: OwnedRasterArray,
    *,
    compute_slope: bool,
    compute_aspect: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run fused slope+aspect NVRTC kernel on device-resident DEM data.

    Computes Horn method gradient in a single pass using shared-memory tiling.
    No D->H->D round-trips. No cp.pad allocation. Nodata handled on device.

    Returns (slope_host, aspect_host) numpy arrays, either may be None if
    not requested.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import SLOPE_ASPECT_KERNEL_SOURCE

    # Keep data on device -- no to_numpy() round-trip
    d_data = _to_device_data(dem).astype(cp.float64)
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape

    # Allocate output buffers on device
    d_slope = cp.zeros_like(d_data) if compute_slope else cp.empty(1, dtype=cp.float64)
    d_aspect = cp.zeros_like(d_data) if compute_aspect else cp.empty(1, dtype=cp.float64)

    # Nodata mask on device (no host round-trip)
    nodata_val = float(dem.nodata) if dem.nodata is not None else 0.0
    if dem.nodata is not None:
        d_nodata = dem.device_nodata_mask().astype(cp.uint8)
        nodata_ptr = d_nodata.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    # Cell size from affine transform
    cell_x = abs(dem.affine[0]) if dem.affine[0] != 0 else 1.0
    cell_y = abs(dem.affine[4]) if dem.affine[4] != 0 else 1.0

    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("slope_aspect", SLOPE_ASPECT_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=SLOPE_ASPECT_KERNEL_SOURCE,
        kernel_names=("slope_aspect",),
    )

    # Block size is fixed at (_TILE_W, _TILE_H) because the kernel source
    # uses #define TILE_W/TILE_H and indexes shared memory relative to
    # threadIdx.  The slope/aspect kernel uses statically-sized shared
    # memory: __shared__ double tile[TILE_H+2][TILE_W+2+1].
    # Validate via occupancy API that the hardware can schedule this block.
    slope_smem = ((_TILE_H + 2) * (_TILE_W + 2 + 1)) * 8  # doubles
    optimal = runtime.optimal_block_size(kernels["slope_aspect"], shared_mem_bytes=slope_smem)
    required = _TILE_W * _TILE_H
    if optimal < required:
        raise RuntimeError(
            f"slope_aspect kernel requires block size {required} "
            f"({_TILE_W}x{_TILE_H}) but occupancy API reports max "
            f"{optimal} threads/block.  Reduce TILE_W/TILE_H."
        )

    block = (_TILE_W, _TILE_H, 1)
    grid = (
        (width + _TILE_W - 1) // _TILE_W,
        (height + _TILE_H - 1) // _TILE_H,
        1,
    )

    slope_ptr = d_slope.data.ptr if compute_slope else 0
    aspect_ptr = d_aspect.data.ptr if compute_aspect else 0

    params = (
        (
            d_data.data.ptr,
            slope_ptr,
            aspect_ptr,
            nodata_ptr,
            width,
            height,
            cell_x,
            cell_y,
            nodata_val,
            1 if compute_slope else 0,
            1 if compute_aspect else 0,
        ),
        (
            KERNEL_PARAM_PTR,  # dem
            KERNEL_PARAM_PTR,  # slope_out
            KERNEL_PARAM_PTR,  # aspect_out
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
            KERNEL_PARAM_F64,  # cell_x
            KERNEL_PARAM_F64,  # cell_y
            KERNEL_PARAM_F64,  # nodata_val
            KERNEL_PARAM_I32,  # compute_slope
            KERNEL_PARAM_I32,  # compute_aspect
        ),
    )

    runtime.launch(
        kernel=kernels["slope_aspect"],
        grid=grid,
        block=block,
        params=params,
    )

    # Transfer results to host only at the end
    slope_host = cp.asnumpy(d_slope) if compute_slope else None
    aspect_host = cp.asnumpy(d_aspect) if compute_aspect else None

    return slope_host, aspect_host


def raster_slope(
    dem: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute slope (degrees) from a DEM raster.

    Uses a fused NVRTC kernel with shared-memory tiled 3x3 Horn method
    when GPU is available, or a numpy CPU fallback otherwise.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when CuPy is available.
    """
    if use_gpu is None:
        use_gpu = _has_cupy()

    orig_dtype = dem.dtype

    if use_gpu:
        slope_host, _ = _gpu_slope_aspect(dem, compute_slope=True, compute_aspect=False)
        backend = "gpu"
    else:
        slope_host, _ = _cpu_slope_aspect(dem, compute_slope=True, compute_aspect=False)
        backend = "cpu"

    # Restore original dtype for float inputs (float32 in -> float32 out)
    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        slope_host = slope_host.astype(orig_dtype)

    result = from_numpy(
        slope_host,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"{backend}_slope_fused {slope_host.shape[1]}x{slope_host.shape[0]}",
            residency=Residency.HOST,
        )
    )
    return result


def raster_aspect(
    dem: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute aspect (degrees, 0=north, clockwise) from a DEM raster.

    Uses a fused NVRTC kernel with shared-memory tiled 3x3 Horn method
    when GPU is available, or a numpy CPU fallback otherwise.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when CuPy is available.
    """
    if use_gpu is None:
        use_gpu = _has_cupy()

    orig_dtype = dem.dtype

    if use_gpu:
        _, aspect_host = _gpu_slope_aspect(dem, compute_slope=False, compute_aspect=True)
        backend = "gpu"
    else:
        _, aspect_host = _cpu_slope_aspect(dem, compute_slope=False, compute_aspect=True)
        backend = "cpu"

    # Restore original dtype for float inputs (float32 in -> float32 out)
    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        aspect_host = aspect_host.astype(orig_dtype)

    result = from_numpy(
        aspect_host,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"{backend}_aspect_fused {aspect_host.shape[1]}x{aspect_host.shape[0]}",
            residency=Residency.HOST,
        )
    )
    return result
