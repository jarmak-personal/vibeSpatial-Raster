"""Vector-to-raster conversion: rasterize geometries onto a grid.

GPU path uses an NVRTC per-pixel point-in-polygon kernel adapted from
kernels/predicates/point_in_polygon.py. CPU fallback uses rasterio.features.

Bead o17.8.8 (GridSpec + CPU baseline) and o17.8.9 (GPU rasterize).
"""

from __future__ import annotations

import time

import numpy as np

from vibespatial.raster.buffers import (
    GridSpec,
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
)

# ---------------------------------------------------------------------------
# NVRTC kernel source for per-pixel polygon rasterize
# ---------------------------------------------------------------------------

_RASTERIZE_KERNEL_SOURCE = r"""
// -----------------------------------------------------------------------
// Shared-memory tile size for polygon bounding-box caching.
// Each block cooperatively loads TILE_POLYS bounding boxes (4 doubles
// each = 32 bytes) into shared memory so that the AABB pre-filter reads
// from SMEM instead of global memory.  This eliminates redundant global
// loads when many threads in the same block test the same polygons.
// -----------------------------------------------------------------------
#define TILE_POLYS 64

extern "C" __device__ inline bool ring_contains_point(
    const double px, const double py,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int coord_start, const int coord_end
) {
    bool inside = false;
    if ((coord_end - coord_start) < 2) return false;
    for (int c = coord_start + 1; c < coord_end; ++c) {
        const double ax = x[c - 1], ay = y[c - 1];
        const double bx = x[c],     by = y[c];
        if (((ay > py) != (by > py)) &&
            (px <= (((bx - ax) * (py - ay)) / (by - ay)) + ax)) {
            inside = !inside;
        }
    }
    return inside;
}

extern "C" __global__ void rasterize_polygons(
    double* __restrict__       out,
    const double* __restrict__ poly_x,
    const double* __restrict__ poly_y,
    const int* __restrict__    geom_offsets,
    const int* __restrict__    ring_offsets,
    const double* __restrict__ values,
    const double* __restrict__ bounds,    // (n_poly, 4): minx, miny, maxx, maxy
    const int                  n_poly,
    const int                  n_pixels,
    const double               affine_a,
    const double               affine_b,
    const double               affine_c,
    const double               affine_d,
    const double               affine_e,
    const double               affine_f,
    const double               fill_value,
    const int                  width,
    const int                  height
) {
    // --- shared-memory tile for polygon AABBs (4 doubles per polygon) ---
    __shared__ double s_bounds[TILE_POLYS * 4];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (idx < n_pixels);

    // Compute pixel coordinates only for active threads (avoid garbage reads)
    double wx = 0.0, wy = 0.0;
    double result = fill_value;
    if (active) {
        const int row = idx / width;
        const int col = idx - row * width;  // faster than idx % width

        // Pixel center world coordinates from affine (computed once)
        const double half_col = col + 0.5;
        const double half_row = row + 0.5;
        wx = affine_a * half_col + affine_b * half_row + affine_c;
        wy = affine_d * half_col + affine_e * half_row + affine_f;
    }

    // Process polygons in tiles of TILE_POLYS for shared-memory AABB caching.
    // ALL threads in the block participate in the cooperative load and
    // __syncthreads() regardless of whether they are active, which avoids
    // deadlocks.
    for (int tile_start = 0; tile_start < n_poly; tile_start += TILE_POLYS) {
        const int tile_end = min(tile_start + TILE_POLYS, n_poly);
        const int tile_count = tile_end - tile_start;

        // Cooperative load: each thread loads one or more AABB entries
        for (int t = threadIdx.x; t < tile_count * 4; t += blockDim.x) {
            s_bounds[t] = bounds[(tile_start * 4) + t];
        }
        __syncthreads();

        // Test each polygon in this tile (only active threads)
        if (active) {
            for (int tp = 0; tp < tile_count; ++tp) {
                const int sb = tp * 4;
                // AABB pre-filter from shared memory
                if (wx < s_bounds[sb]     || wx > s_bounds[sb + 2] ||
                    wy < s_bounds[sb + 1] || wy > s_bounds[sb + 3]) {
                    continue;
                }

                // Full ring-based PIP test (global memory for geometry data)
                const int p = tile_start + tp;
                const int ring_start = geom_offsets[p];
                const int ring_end   = geom_offsets[p + 1];
                bool inside = false;
                for (int ring = ring_start; ring < ring_end; ++ring) {
                    const int cs = ring_offsets[ring];
                    const int ce = ring_offsets[ring + 1];
                    if (ring_contains_point(wx, wy, poly_x, poly_y, cs, ce)) {
                        inside = !inside;
                    }
                }
                if (inside) {
                    result = values[p];
                }
            }
        }

        __syncthreads();  // ensure tile is consumed before next load
    }

    if (active) {
        out[idx] = result;
    }
}
"""


# ---------------------------------------------------------------------------
# CPU baseline via rasterio.features
# ---------------------------------------------------------------------------


def rasterize_cpu(
    geometries,
    values: np.ndarray,
    grid_spec: GridSpec,
) -> OwnedRasterArray:
    """Rasterize geometries onto a grid using rasterio (CPU).

    Parameters
    ----------
    geometries : sequence of Shapely geometries
        Geometries to rasterize.
    values : np.ndarray
        Value for each geometry (same length as geometries).
    grid_spec : GridSpec
        Target grid specification.

    Returns
    -------
    OwnedRasterArray
    """
    try:
        from rasterio.features import rasterize as rio_rasterize
        from rasterio.transform import Affine
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for CPU rasterize. "
            "Install with: uv sync --group upstream-optional"
        ) from exc

    transform = Affine(
        grid_spec.affine[0],
        grid_spec.affine[1],
        grid_spec.affine[2],
        grid_spec.affine[3],
        grid_spec.affine[4],
        grid_spec.affine[5],
    )

    shapes = list(zip(geometries, values))
    result = rio_rasterize(
        shapes,
        out_shape=(grid_spec.height, grid_spec.width),
        transform=transform,
        fill=grid_spec.fill_value,
        dtype=str(grid_spec.dtype),
    )

    return from_numpy(
        result,
        nodata=grid_spec.fill_value,
        affine=grid_spec.affine,
    )


# ---------------------------------------------------------------------------
# GPU rasterize via NVRTC kernel
# ---------------------------------------------------------------------------


def rasterize_gpu(
    geometries,
    values: np.ndarray,
    grid_spec: GridSpec,
) -> OwnedRasterArray:
    """Rasterize polygons onto a grid using GPU per-pixel PIP kernel.

    Parameters
    ----------
    geometries : sequence of Shapely geometries
        Polygon/MultiPolygon geometries to rasterize.
    values : np.ndarray
        Value for each geometry.
    grid_spec : GridSpec
        Target grid specification.

    Returns
    -------
    OwnedRasterArray
    """
    import cupy as cp
    from shapely import get_parts
    from shapely.geometry import MultiPolygon

    from vibespatial.cuda_runtime import get_cuda_runtime, make_kernel_cache_key

    # Decompose all geometries into simple polygons with values
    all_x = []
    all_y = []
    all_geom_offsets = [0]
    all_ring_offsets = [0]
    all_values = []
    all_bounds = []

    coord_cursor = 0
    ring_cursor = 0

    for geom, val in zip(geometries, values):
        if geom is None or geom.is_empty:
            continue

        # Decompose MultiPolygon into parts
        if isinstance(geom, MultiPolygon):
            parts = get_parts(geom)
        else:
            parts = [geom]

        for poly in parts:
            ext = poly.exterior
            coords = np.array(ext.coords)
            n_ext = len(coords)
            all_x.append(coords[:, 0])
            all_y.append(coords[:, 1])
            coord_cursor += n_ext

            rings_in_poly = [coord_cursor]
            # Holes
            for hole in poly.interiors:
                hcoords = np.array(hole.coords)
                n_h = len(hcoords)
                all_x.append(hcoords[:, 0])
                all_y.append(hcoords[:, 1])
                coord_cursor += n_h
                rings_in_poly.append(coord_cursor)

            all_ring_offsets.extend(rings_in_poly)
            ring_cursor += len(rings_in_poly)

            all_geom_offsets.append(ring_cursor)
            all_values.append(float(val))

            b = poly.bounds  # (minx, miny, maxx, maxy)
            all_bounds.extend([b[0], b[1], b[2], b[3]])

    if not all_values:
        result_data = np.full(
            (grid_spec.height, grid_spec.width),
            grid_spec.fill_value,
            dtype=np.float64,
        )
        return from_numpy(result_data, nodata=grid_spec.fill_value, affine=grid_spec.affine)

    n_poly = len(all_values)

    # Flatten coordinate arrays
    poly_x = np.concatenate(all_x).astype(np.float64)
    poly_y = np.concatenate(all_y).astype(np.float64)
    geom_offsets = np.array(all_geom_offsets, dtype=np.int32)
    ring_offsets = np.array(all_ring_offsets, dtype=np.int32)
    poly_values = np.array(all_values, dtype=np.float64)
    bounds = np.array(all_bounds, dtype=np.float64)

    # Transfer to device
    d_poly_x = cp.asarray(poly_x)
    d_poly_y = cp.asarray(poly_y)
    d_geom_offsets = cp.asarray(geom_offsets)
    d_ring_offsets = cp.asarray(ring_offsets)
    d_values = cp.asarray(poly_values)
    d_bounds = cp.asarray(bounds)

    total_pixels = grid_spec.width * grid_spec.height
    d_out = cp.full(total_pixels, grid_spec.fill_value, dtype=np.float64)

    # Compile kernel with shared-memory AABB tiling
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("rasterize_polygons", _RASTERIZE_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=_RASTERIZE_KERNEL_SOURCE,
        kernel_names=("rasterize_polygons",),
    )

    # Shared memory: TILE_POLYS(64) * 4 doubles * 8 bytes = 2048 bytes
    smem_bytes = 64 * 4 * 8

    # Occupancy-based launch config (never hardcode block size)
    kernel = kernels["rasterize_polygons"]
    grid_dim, block_dim = runtime.launch_config(kernel, total_pixels, shared_mem_bytes=smem_bytes)

    af_a, af_b, af_c, af_d, af_e, af_f = grid_spec.affine

    from vibespatial.cuda_runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32, KERNEL_PARAM_PTR

    params = (
        (
            d_out.data.ptr,
            d_poly_x.data.ptr,
            d_poly_y.data.ptr,
            d_geom_offsets.data.ptr,
            d_ring_offsets.data.ptr,
            d_values.data.ptr,
            d_bounds.data.ptr,
            n_poly,
            total_pixels,
            af_a,
            af_b,
            af_c,
            af_d,
            af_e,
            af_f,
            float(grid_spec.fill_value),
            grid_spec.width,
            grid_spec.height,
        ),
        (
            KERNEL_PARAM_PTR,  # out
            KERNEL_PARAM_PTR,  # poly_x
            KERNEL_PARAM_PTR,  # poly_y
            KERNEL_PARAM_PTR,  # geom_offsets
            KERNEL_PARAM_PTR,  # ring_offsets
            KERNEL_PARAM_PTR,  # values
            KERNEL_PARAM_PTR,  # bounds
            KERNEL_PARAM_I32,  # n_poly
            KERNEL_PARAM_I32,  # n_pixels
            KERNEL_PARAM_F64,  # affine_a
            KERNEL_PARAM_F64,  # affine_b
            KERNEL_PARAM_F64,  # affine_c
            KERNEL_PARAM_F64,  # affine_d
            KERNEL_PARAM_F64,  # affine_e
            KERNEL_PARAM_F64,  # affine_f
            KERNEL_PARAM_F64,  # fill_value
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
        ),
    )

    runtime.launch(
        kernel=kernel,
        grid=grid_dim,
        block=block_dim,
        params=params,
        shared_mem_bytes=smem_bytes,
    )

    result_data = cp.asnumpy(d_out).reshape(grid_spec.height, grid_spec.width)
    return from_numpy(
        result_data.astype(grid_spec.dtype),
        nodata=grid_spec.fill_value,
        affine=grid_spec.affine,
    )


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------


def rasterize_owned(
    geometries,
    values: np.ndarray,
    grid_spec: GridSpec,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Rasterize geometries onto a grid with automatic GPU/CPU dispatch.

    Parameters
    ----------
    geometries : sequence of Shapely geometries
        Geometries to rasterize.
    values : np.ndarray
        Value per geometry.
    grid_spec : GridSpec
        Target grid specification.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when available and pixel count exceeds threshold.

    Returns
    -------
    OwnedRasterArray
    """
    values = np.asarray(values, dtype=np.float64)
    total_pixels = grid_spec.width * grid_spec.height
    gpu_threshold = 100_000  # CONSTRUCTIVE class threshold

    if use_gpu is None:
        try:
            from vibespatial.runtime import has_gpu_runtime

            use_gpu = has_gpu_runtime() and total_pixels >= gpu_threshold
        except Exception:
            use_gpu = False

    t0 = time.perf_counter()
    if use_gpu:
        result = rasterize_gpu(geometries, values, grid_spec)
    else:
        result = rasterize_cpu(geometries, values, grid_spec)
    elapsed = time.perf_counter() - t0

    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"rasterize {'gpu' if use_gpu else 'cpu'} {grid_spec.width}x{grid_spec.height} "
            f"geoms={len(values)} elapsed={elapsed:.3f}s",
            residency=result.residency,
        )
    )
    return result
