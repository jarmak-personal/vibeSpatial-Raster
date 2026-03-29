"""Tiling execution engine for pointwise (trivial) raster operations.

Phase 1 of the tiling infrastructure (vibeSpatial-fx3.2): processes rasters
in spatial chunks that fit in VRAM, enabling large-raster GPU processing
without OOM.  Each tile undergoes a HOST->DEVICE->HOST round trip
independently so that the full raster never needs to reside on the GPU at
once.

For pointwise (zero-overlap) operations the tiles are non-overlapping
rectangles that partition the raster.  The results are stitched by direct
array assignment -- no blending or overlap reconciliation is required.

ADR: vibeSpatial-fx3.2  Phase 1: Trivial tiling for pointwise operations
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vibespatial.raster.buffers import OwnedRasterArray, RasterPlan

__all__ = [
    "dispatch_tiled",
    "dispatch_tiled_binary",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tile_bounds(
    tile_row: int,
    tile_col: int,
    tile_h: int,
    tile_w: int,
    raster_h: int,
    raster_w: int,
) -> tuple[int, int, int, int]:
    """Return (row_start, row_end, col_start, col_end) clamped to raster bounds.

    Parameters
    ----------
    tile_row, tile_col:
        0-indexed tile position in the tile grid.
    tile_h, tile_w:
        Nominal tile dimensions (height, width).
    raster_h, raster_w:
        Full raster spatial dimensions.

    Returns
    -------
    tuple[int, int, int, int]
        ``(row_start, row_end, col_start, col_end)`` where slicing with
        ``data[..., row_start:row_end, col_start:col_end]`` extracts the
        tile, correctly clamped for edge tiles.
    """
    row_start = tile_row * tile_h
    row_end = min(row_start + tile_h, raster_h)
    col_start = tile_col * tile_w
    col_end = min(col_start + tile_w, raster_w)
    return row_start, row_end, col_start, col_end


def _adjust_affine(
    affine: tuple[float, float, float, float, float, float],
    row_offset: int,
    col_offset: int,
) -> tuple[float, float, float, float, float, float]:
    """Shift affine transform origin to account for tile position.

    The affine is ``(a, b, c, d, e, f)`` where::

        world_x = a * col + b * row + c
        world_y = d * col + e * row + f

    For a tile starting at pixel ``(row_offset, col_offset)`` the new
    origin is::

        new_c = c + col_offset * a + row_offset * b
        new_f = f + col_offset * d + row_offset * e

    Parameters
    ----------
    affine:
        6-element GDAL-style affine of the full raster.
    row_offset, col_offset:
        Pixel offset of the tile's upper-left corner within the full raster.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Adjusted affine for the tile.
    """
    a, b, c, d, e, f = affine
    new_c = c + col_offset * a + row_offset * b
    new_f = f + col_offset * d + row_offset * e
    return (a, b, new_c, d, e, new_f)


def _ensure_host_resident(raster: OwnedRasterArray, *, label: str) -> np.ndarray:
    """Return the host numpy array, raising if the raster is DEVICE-resident.

    Phase 1 tiling requires HOST-resident input because the whole point of
    tiling is that the full raster does not fit in VRAM.  A DEVICE-resident
    raster would require a full D->H transfer via ``to_numpy()``, defeating
    the OOM-avoidance goal.

    Raises
    ------
    ValueError
        If the raster is DEVICE-resident.
    """
    from vibespatial.residency import Residency

    if raster.residency == Residency.DEVICE:
        raise ValueError(
            f"{label} requires HOST-resident input; a DEVICE-resident raster "
            "cannot be tiled because the full D->H transfer would defeat the "
            "OOM-avoidance goal.  Call raster.move_to(Residency.HOST) first, "
            "or use WHOLE strategy for device-resident data."
        )
    return raster.to_numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dispatch_tiled(
    raster: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
    plan: RasterPlan,
) -> OwnedRasterArray:
    """Execute a unary pointwise operation using spatial tiling.

    Parameters
    ----------
    raster:
        Input raster (single- or multi-band).  Must be HOST-resident for
        the TILED path; the WHOLE fast path accepts any residency.
    op_fn:
        The operation to apply per tile.  Receives a tile-sized
        ``OwnedRasterArray`` and returns a result of the same spatial
        dimensions.  The operation may internally transfer the tile to
        device and back -- this is the expected pattern for tiled GPU
        processing.
    plan:
        A frozen ``RasterPlan`` produced by ``analyze_raster_plan()``.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same shape, affine, CRS, and nodata as the
        input.  The dtype is determined by ``op_fn``'s output.

    Raises
    ------
    ValueError
        If ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is None,
        or if the raster is DEVICE-resident on the TILED path.
    """
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
        TilingStrategy,
        from_numpy,
    )
    from vibespatial.residency import Residency

    # -- WHOLE fast path: no tiling overhead --
    if plan.strategy == TilingStrategy.WHOLE:
        return op_fn(raster)

    # -- TILED path --
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    t0 = time.perf_counter()

    tile_h, tile_w = plan.tile_shape
    host = _ensure_host_resident(raster, label="dispatch_tiled")
    raster_h = raster.height
    raster_w = raster.width

    # Lazy output allocation: deferred until first tile result so that the
    # output dtype matches op_fn's actual return dtype (which may differ
    # from the input dtype, e.g. classify returning uint8 from float32).
    output: np.ndarray | None = None

    # Compute tile grid dimensions.
    rows_of_tiles = (raster_h + tile_h - 1) // tile_h
    cols_of_tiles = (raster_w + tile_w - 1) // tile_w

    # TODO(fx3-perf): Overlap H->D transfer of tile N+1 with GPU compute
    # on tile N using two CUDA streams (double-buffering).  Current serial
    # execution leaves the GPU idle during host-side tile preparation.
    tiles_processed = 0
    result_nodata: float | int | None = raster.nodata
    for tr in range(rows_of_tiles):
        for tc in range(cols_of_tiles):
            rs, re, cs, ce = _tile_bounds(tr, tc, tile_h, tile_w, raster_h, raster_w)

            # Slice tile from host array.  ascontiguousarray() is a no-op
            # for already-contiguous slices (e.g. full-row tiles) and
            # ensures contiguity for interior tiles where the slice has
            # non-unit stride along the column axis.
            if host.ndim == 3:
                tile_data = np.ascontiguousarray(host[:, rs:re, cs:ce])
            else:
                tile_data = np.ascontiguousarray(host[rs:re, cs:ce])

            # Adjust affine for this tile's spatial position.
            tile_affine = _adjust_affine(raster.affine, row_offset=rs, col_offset=cs)

            # Wrap tile as OwnedRasterArray (HOST-resident).
            tile_raster = from_numpy(
                tile_data,
                nodata=raster.nodata,
                affine=tile_affine,
                crs=raster.crs,
            )

            # Apply the operation (op may H->D->H internally).
            tile_result = op_fn(tile_raster)

            # Retrieve result to host.
            tile_host = tile_result.to_numpy()

            # Lazy allocation on first tile result.
            if output is None:
                if host.ndim == 3:
                    output = np.empty(
                        (host.shape[0], raster_h, raster_w),
                        dtype=tile_host.dtype,
                    )
                else:
                    output = np.empty((raster_h, raster_w), dtype=tile_host.dtype)
                result_nodata = tile_result.nodata

            if output.ndim == 3:
                output[:, rs:re, cs:ce] = tile_host
            else:
                output[rs:re, cs:ce] = tile_host

            # Release tile references.  CPython's reference counting
            # triggers immediate deallocation, and both the RMM pool and
            # CuPy default pool immediately reclaim freed device blocks
            # for reuse by the next tile's allocation.
            del tile_raster, tile_result, tile_host, tile_data

            tiles_processed += 1

    elapsed = time.perf_counter() - t0

    # Guard against zero-tile degenerate rasters.
    if output is None:
        raise ValueError("Zero tiles processed; input raster has degenerate dimensions")

    result = from_numpy(
        output,
        nodata=result_nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_tiled unary tiles={tiles_processed} "
                f"tile_shape=({tile_h},{tile_w}) "
                f"raster_shape={raster.shape} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.HOST,
            elapsed_seconds=elapsed,
        )
    )
    return result


def dispatch_tiled_binary(
    a: OwnedRasterArray,
    b: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray, OwnedRasterArray], OwnedRasterArray],
    plan: RasterPlan,
) -> OwnedRasterArray:
    """Execute a binary pointwise operation using spatial tiling.

    Same tiling strategy as :func:`dispatch_tiled` but extracts the same
    tile region from both input rasters and passes them to a binary
    ``op_fn``.

    Parameters
    ----------
    a, b:
        Input rasters.  Must have the same spatial dimensions.  Must be
        HOST-resident for the TILED path.
    op_fn:
        Binary operation.  Receives two tile-sized ``OwnedRasterArray``
        objects and returns a result of the same spatial dimensions.
    plan:
        A frozen ``RasterPlan`` produced by ``analyze_raster_plan()``.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same shape, affine, CRS, and nodata as ``a``.
        The dtype is determined by ``op_fn``'s output.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different spatial dimensions, if
        ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is None,
        or if either input is DEVICE-resident on the TILED path.
    """
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
        TilingStrategy,
        from_numpy,
    )
    from vibespatial.residency import Residency

    if a.height != b.height or a.width != b.width:
        raise ValueError(
            f"Spatial dimension mismatch: a=({a.height},{a.width}), b=({b.height},{b.width})"
        )

    # -- WHOLE fast path: no tiling overhead --
    if plan.strategy == TilingStrategy.WHOLE:
        return op_fn(a, b)

    # -- TILED path --
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    t0 = time.perf_counter()

    tile_h, tile_w = plan.tile_shape
    host_a = _ensure_host_resident(a, label="dispatch_tiled_binary (input a)")
    host_b = _ensure_host_resident(b, label="dispatch_tiled_binary (input b)")
    raster_h = a.height
    raster_w = a.width

    # Lazy output allocation: deferred until first tile result.
    output: np.ndarray | None = None

    rows_of_tiles = (raster_h + tile_h - 1) // tile_h
    cols_of_tiles = (raster_w + tile_w - 1) // tile_w

    # TODO(fx3-perf): Double-buffer with two CUDA streams for overlap.
    tiles_processed = 0
    result_nodata: float | int | None = a.nodata

    for tr in range(rows_of_tiles):
        for tc in range(cols_of_tiles):
            rs, re, cs, ce = _tile_bounds(tr, tc, tile_h, tile_w, raster_h, raster_w)

            # Slice tiles from host arrays (contiguous for DMA).
            if host_a.ndim == 3:
                tile_a_data = np.ascontiguousarray(host_a[:, rs:re, cs:ce])
            else:
                tile_a_data = np.ascontiguousarray(host_a[rs:re, cs:ce])

            if host_b.ndim == 3:
                tile_b_data = np.ascontiguousarray(host_b[:, rs:re, cs:ce])
            else:
                tile_b_data = np.ascontiguousarray(host_b[rs:re, cs:ce])

            tile_affine = _adjust_affine(a.affine, row_offset=rs, col_offset=cs)

            tile_a = from_numpy(
                tile_a_data,
                nodata=a.nodata,
                affine=tile_affine,
                crs=a.crs,
            )
            tile_b = from_numpy(
                tile_b_data,
                nodata=b.nodata,
                affine=tile_affine,
                crs=b.crs,
            )

            tile_result = op_fn(tile_a, tile_b)
            tile_host = tile_result.to_numpy()

            # Lazy output allocation: use first tile's result dtype.
            if output is None:
                if host_a.ndim == 3:
                    output = np.empty(
                        (host_a.shape[0], raster_h, raster_w),
                        dtype=tile_host.dtype,
                    )
                else:
                    output = np.empty((raster_h, raster_w), dtype=tile_host.dtype)
                result_nodata = tile_result.nodata

            if output.ndim == 3:
                output[:, rs:re, cs:ce] = tile_host
            else:
                output[rs:re, cs:ce] = tile_host

            del tile_a, tile_b, tile_result, tile_host, tile_a_data, tile_b_data

            tiles_processed += 1

    elapsed = time.perf_counter() - t0

    # Guard against zero-tile degenerate rasters.
    if output is None:
        raise ValueError("Zero tiles processed; input rasters have degenerate dimensions")

    result = from_numpy(
        output,
        nodata=result_nodata,
        affine=a.affine,
        crs=a.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_tiled_binary tiles={tiles_processed} "
                f"tile_shape=({tile_h},{tile_w}) "
                f"raster_shape={a.shape} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.HOST,
            elapsed_seconds=elapsed,
        )
    )
    return result
