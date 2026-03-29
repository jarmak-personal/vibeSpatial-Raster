"""Raster IO: GeoTIFF and COG read/write via rasterio or nvImageCodec.

Host-side decode via rasterio, or GPU-native decode via nvImageCodec.
The ``read_raster`` dispatcher tries the GPU path first (when available),
falling back to rasterio transparently.

ADR-0037: Raster IO Support and Read Paths
ADR: vibeSpatial-fx3.6  Phase 5: Streamed windowed IO
"""

from __future__ import annotations

import time
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    RasterMetadata,
    RasterWindow,
    TilingStrategy,
    from_device,
    from_numpy,
)
from vibespatial.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from pyproj import CRS

    from vibespatial.raster.buffers import RasterPlan


def has_rasterio_support() -> bool:
    """Check whether rasterio is available."""
    try:
        return find_spec("rasterio") is not None
    except ModuleNotFoundError:
        return False


def _require_rasterio():
    if not has_rasterio_support():
        raise ImportError(
            "rasterio is required for raster IO. Install it with: uv sync --extra upstream-optional"
        )


def _affine_to_tuple(transform) -> tuple[float, float, float, float, float, float]:
    """Convert a rasterio Affine to a 6-element tuple (a, b, c, d, e, f)."""
    return (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)


def _tuple_to_affine(t: tuple[float, float, float, float, float, float]):
    """Convert a 6-element tuple to a rasterio Affine."""
    import rasterio.transform

    return rasterio.transform.Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def _extract_crs(src) -> CRS | None:
    """Extract a pyproj CRS from a rasterio dataset, or None."""
    if src.crs is None:
        return None
    try:
        from pyproj import CRS

        return CRS.from_user_input(src.crs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metadata-only read (no pixel data)
# ---------------------------------------------------------------------------


def read_raster_metadata(path: str | Path) -> RasterMetadata:
    """Read raster metadata without loading pixel data.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF, COG, or other rasterio-supported raster file.

    Returns
    -------
    RasterMetadata
        Shape, dtype, nodata, affine, CRS, and driver information.
    """
    _require_rasterio()
    import rasterio

    with rasterio.open(path) as src:
        nodata = src.nodata
        return RasterMetadata(
            height=src.height,
            width=src.width,
            band_count=src.count,
            dtype=np.dtype(src.dtypes[0]),
            nodata=np.dtype(src.dtypes[0]).type(nodata) if nodata is not None else None,
            affine=_affine_to_tuple(src.transform),
            crs=_extract_crs(src),
            driver=src.driver,
        )


# ---------------------------------------------------------------------------
# nvImageCodec availability check
# ---------------------------------------------------------------------------


def has_nvimgcodec_support() -> bool:
    """Check whether nvImageCodec GPU decode is available."""
    try:
        from vibespatial.raster.nvimgcodec_io import has_nvimgcodec_support as _check

        return _check()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# nvImageCodec dispatch helper
# ---------------------------------------------------------------------------


def _try_nvimgcodec_read(path, *, bands, window, overview_level):
    """Attempt nvImageCodec GPU decode. Returns (device_data, metadata) or None."""
    try:
        from vibespatial.raster.nvimgcodec_io import (
            has_nvimgcodec_support as _check,
        )
        from vibespatial.raster.nvimgcodec_io import (
            nvimgcodec_read,
        )

        if not _check():
            return None
        return nvimgcodec_read(path, bands=bands, window=window, overview_level=overview_level)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# rasterio read helper
# ---------------------------------------------------------------------------


def _read_raster_rasterio(path, *, bands, window, overview_level, residency):
    """Read raster via rasterio (HYBRID path)."""
    _require_rasterio()
    import rasterio
    from rasterio.windows import Window

    open_kwargs = {}
    if overview_level is not None:
        open_kwargs["overview_level"] = overview_level

    with rasterio.open(path, **open_kwargs) as src:
        # Determine bands to read
        if bands is None:
            band_indices = list(range(1, src.count + 1))
        else:
            band_indices = bands

        # Build rasterio Window
        rio_window = None
        if window is not None:
            rio_window = Window(
                col_off=window.col_off,
                row_off=window.row_off,
                width=window.width,
                height=window.height,
            )

        # Read pixel data
        data = src.read(band_indices, window=rio_window)
        # data shape: (bands, height, width)

        # Squeeze single-band to 2D
        if data.shape[0] == 1:
            data = data[0]

        # Extract metadata
        nodata = src.nodata
        if rio_window is not None:
            transform = src.window_transform(rio_window)
        else:
            transform = src.transform

        affine = _affine_to_tuple(transform)
        crs = _extract_crs(src)

    result = from_numpy(
        data,
        nodata=data.dtype.type(nodata) if nodata is not None else None,
        affine=affine,
        crs=crs,
        residency=residency,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"read_raster backend=rasterio path={Path(path).name}",
            residency=result.residency,
            visible_to_user=True,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Full read
# ---------------------------------------------------------------------------


def read_raster(
    path: str | Path,
    *,
    bands: list[int] | None = None,
    window: RasterWindow | None = None,
    overview_level: int | None = None,
    residency: Residency = Residency.HOST,
    decode_backend: str = "auto",
) -> OwnedRasterArray:
    """Read a raster file into an OwnedRasterArray.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF, COG, or other rasterio-supported raster file.
    bands : list[int] or None
        1-based band indices to read. None reads all bands.
    window : RasterWindow or None
        Sub-window to read. None reads the full raster.
    overview_level : int or None
        Overview (pyramid) level to read. None reads full resolution.
    residency : Residency
        Target residency for the output array.
    decode_backend : str
        Decode backend selection: ``"auto"`` (try GPU first, fall back to
        rasterio), ``"nvimgcodec"`` (GPU-only, raises on failure), or
        ``"rasterio"`` (CPU-only, skip GPU path).

    Returns
    -------
    OwnedRasterArray
        The raster data with metadata, in the requested residency.
    """
    path = str(path)  # normalize

    # --- Try GPU-native decode first ---
    if decode_backend in ("auto", "nvimgcodec"):
        gpu_result = _try_nvimgcodec_read(
            path,
            bands=bands,
            window=window,
            overview_level=overview_level,
        )
        if gpu_result is not None:
            device_data, meta = gpu_result
            # Supplement nodata from rasterio if nvimgcodec didn't provide it
            if meta.nodata is None and has_rasterio_support():
                try:
                    rio_meta = read_raster_metadata(path)
                    nodata = rio_meta.nodata
                except Exception:
                    nodata = None
            else:
                nodata = meta.nodata

            result = from_device(
                device_data,
                nodata=nodata,
                affine=meta.affine if meta.affine else (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                crs=meta.crs,
            )
            result.diagnostics.append(
                RasterDiagnosticEvent(
                    kind=RasterDiagnosticKind.RUNTIME,
                    detail=f"read_raster backend=nvimgcodec path={Path(path).name}",
                    residency=result.residency,
                    visible_to_user=True,
                )
            )
            # If user wants HOST residency, transfer from device
            if residency is Residency.HOST:
                result.move_to(
                    Residency.HOST,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="read_raster requested HOST residency",
                )
            return result

        # If nvimgcodec was explicitly requested but failed, raise
        if decode_backend == "nvimgcodec":
            raise RuntimeError(
                f"nvimgcodec decode failed for {path}. "
                "Ensure nvidia-nvimgcodec-cu12[all] is installed and the format is supported."
            )

    # --- Fallback to rasterio ---
    return _read_raster_rasterio(
        path,
        bands=bands,
        window=window,
        overview_level=overview_level,
        residency=residency,
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_raster(
    path: str | Path,
    raster: OwnedRasterArray,
    *,
    driver: str = "GTiff",
    compress: str | None = "deflate",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> None:
    """Write an OwnedRasterArray to a raster file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    raster : OwnedRasterArray
        The raster to write.
    driver : str
        GDAL driver name (default "GTiff").
    compress : str or None
        Compression algorithm (default "deflate"). None for no compression.
    tiled : bool
        Whether to write tiled (default True for COG compatibility).
    blockxsize, blockysize : int
        Tile dimensions when tiled=True.
    """
    _require_rasterio()
    import rasterio

    host_data = raster.to_numpy()

    # Ensure 3D (bands, height, width)
    if host_data.ndim == 2:
        host_data = host_data[np.newaxis, :, :]

    band_count = host_data.shape[0]
    height = host_data.shape[1]
    width = host_data.shape[2]

    profile = {
        "driver": driver,
        "dtype": str(raster.dtype),
        "width": width,
        "height": height,
        "count": band_count,
        "transform": _tuple_to_affine(raster.affine),
        "tiled": tiled,
    }

    if raster.nodata is not None:
        profile["nodata"] = raster.nodata
    if raster.crs is not None:
        profile["crs"] = raster.crs.to_epsg() or raster.crs.to_wkt()
    if compress is not None:
        profile["compress"] = compress
    if tiled:
        profile["blockxsize"] = blockxsize
        profile["blockysize"] = blockysize

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(host_data)

    raster.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.MATERIALIZATION,
            detail=f"wrote {height}x{width}x{band_count} to {path} driver={driver}",
            residency=raster.residency,
            visible_to_user=True,
        )
    )


# ---------------------------------------------------------------------------
# Streamed windowed IO (vibeSpatial-fx3.6)
# ---------------------------------------------------------------------------


def process_raster_streamed(
    input_path: str | Path,
    output_path: str | Path,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
    plan: RasterPlan,
    *,
    halo: int = 0,
    compress: str | None = "deflate",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> RasterMetadata:
    """Process a raster file tile-by-tile without loading the full raster.

    Reads tiles from the input file via rasterio windowed reads, applies
    ``op_fn`` to each tile, and writes the result to the output file via
    rasterio windowed writes.  At no point is the full raster held in host
    or device memory — only one tile's worth of data is live at a time.

    Parameters
    ----------
    input_path : str or Path
        Path to the input raster file (GeoTIFF, COG, or any rasterio-
        supported format).
    output_path : str or Path
        Path to write the output raster file.
    op_fn : Callable[[OwnedRasterArray], OwnedRasterArray]
        The operation to apply to each tile.  Receives a tile-sized
        ``OwnedRasterArray`` and returns a result of the same spatial
        dimensions as the effective (non-halo) region.  When ``halo > 0``
        the input includes halo border pixels and the dispatcher trims the
        result to the effective region.
    plan : RasterPlan
        A frozen ``RasterPlan`` that controls the tiling strategy.  When
        ``plan.strategy == WHOLE`` the entire raster is read, processed,
        and written in one shot (the fast path).  When ``plan.strategy ==
        TILED``, ``plan.tile_shape`` specifies the effective tile dimensions
        ``(tile_H, tile_W)`` and the raster is streamed tile-by-tile.
    halo : int
        Number of overlap pixels to include around each tile for stencil
        operations.  The result is trimmed back to the effective region
        before writing.  Default 0 (no overlap).
    compress : str or None
        Compression algorithm for the output file.  Default ``"deflate"``.
        Pass ``None`` for no compression.
    tiled : bool
        Whether to write tiled GeoTIFF blocks.  Default ``True``.
    blockxsize, blockysize : int
        GeoTIFF internal tile dimensions when ``tiled=True``.

    Returns
    -------
    RasterMetadata
        Metadata of the output file.

    Raises
    ------
    ValueError
        If ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is ``None``.
    ImportError
        If rasterio is not installed.
    """
    _require_rasterio()
    import rasterio
    from rasterio.windows import Window

    from vibespatial.raster.tiling import _adjust_affine, _tile_bounds

    input_path = Path(input_path)
    output_path = Path(output_path)

    # -- WHOLE fast path: read → process → write --------------------------
    if plan.strategy == TilingStrategy.WHOLE:
        t0 = time.perf_counter()
        raster = read_raster(str(input_path))
        result = op_fn(raster)
        write_raster(
            output_path,
            result,
            compress=compress,
            tiled=tiled,
            blockxsize=blockxsize,
            blockysize=blockysize,
        )
        elapsed = time.perf_counter() - t0
        _whole_diag = RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"process_raster_streamed strategy=WHOLE "
                f"path={input_path.name} elapsed={elapsed:.4f}s"
            ),
            residency=result.residency,
            elapsed_seconds=elapsed,
        )
        result.diagnostics.append(_whole_diag)
        process_raster_streamed._last_diagnostic = _whole_diag  # type: ignore[attr-defined]
        return read_raster_metadata(str(output_path))

    # -- TILED path: streamed windowed IO ---------------------------------
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    t0 = time.perf_counter()
    tile_h, tile_w = plan.tile_shape

    with rasterio.open(str(input_path)) as src:
        raster_h = src.height
        raster_w = src.width
        band_count = src.count
        src_dtype = src.dtypes[0]
        src_nodata = src.nodata
        src_transform = src.transform
        src_crs = _extract_crs(src)
        affine = _affine_to_tuple(src_transform)

        # Build output profile from input metadata.
        profile = {
            "driver": "GTiff",
            "dtype": src_dtype,
            "width": raster_w,
            "height": raster_h,
            "count": band_count,
            "transform": src_transform,
            "tiled": tiled,
        }
        if src_nodata is not None:
            profile["nodata"] = src_nodata
        if src.crs is not None:
            profile["crs"] = src.crs
        if compress is not None:
            profile["compress"] = compress
        if tiled:
            profile["blockxsize"] = blockxsize
            profile["blockysize"] = blockysize

        nodata_typed = np.dtype(src_dtype).type(src_nodata) if src_nodata is not None else None

        # Compute tile grid.
        rows_of_tiles = (raster_h + tile_h - 1) // tile_h
        cols_of_tiles = (raster_w + tile_w - 1) // tile_w

        tiles_processed = 0

        with rasterio.open(str(output_path), "w", **profile) as dst:
            for tr in range(rows_of_tiles):
                for tc in range(cols_of_tiles):
                    # Effective bounds — the non-overlapping output region.
                    eff_rs, eff_re, eff_cs, eff_ce = _tile_bounds(
                        tr,
                        tc,
                        tile_h,
                        tile_w,
                        raster_h,
                        raster_w,
                    )

                    if halo > 0:
                        # Physical bounds: expand by halo, clamped.
                        phys_rs = max(0, eff_rs - halo)
                        phys_re = min(raster_h, eff_re + halo)
                        phys_cs = max(0, eff_cs - halo)
                        phys_ce = min(raster_w, eff_ce + halo)
                    else:
                        phys_rs, phys_re = eff_rs, eff_re
                        phys_cs, phys_ce = eff_cs, eff_ce

                    phys_height = phys_re - phys_rs
                    phys_width = phys_ce - phys_cs

                    # Read the physical tile via windowed read.
                    read_window = Window(
                        col_off=phys_cs,
                        row_off=phys_rs,
                        width=phys_width,
                        height=phys_height,
                    )
                    tile_data = src.read(window=read_window)
                    # tile_data shape: (bands, phys_height, phys_width)

                    # Squeeze single-band to 2D for consistency with
                    # OwnedRasterArray conventions.
                    if tile_data.shape[0] == 1:
                        tile_data = tile_data[0]

                    # Adjust affine to the physical tile origin.
                    tile_affine = _adjust_affine(
                        affine,
                        row_offset=phys_rs,
                        col_offset=phys_cs,
                    )

                    # Wrap as OwnedRasterArray (HOST-resident).
                    tile_raster = from_numpy(
                        tile_data,
                        nodata=nodata_typed,
                        affine=tile_affine,
                        crs=src_crs,
                    )

                    # Apply the user operation.
                    tile_result = op_fn(tile_raster)

                    # Materialize result to host.
                    tile_host = tile_result.to_numpy()

                    # Trim halo from result if present.
                    if halo > 0:
                        top_trim = eff_rs - phys_rs
                        left_trim = eff_cs - phys_cs
                        eff_height = eff_re - eff_rs
                        eff_width = eff_ce - eff_cs

                        if tile_host.ndim == 3:
                            tile_host = tile_host[
                                :,
                                top_trim : top_trim + eff_height,
                                left_trim : left_trim + eff_width,
                            ]
                        else:
                            tile_host = tile_host[
                                top_trim : top_trim + eff_height,
                                left_trim : left_trim + eff_width,
                            ]

                    # Ensure 3D for rasterio write (bands, H, W).
                    write_data = tile_host
                    if write_data.ndim == 2:
                        write_data = write_data[np.newaxis, :, :]

                    # Write to the effective window in the output file.
                    eff_height = eff_re - eff_rs
                    eff_width = eff_ce - eff_cs
                    write_window = Window(
                        col_off=eff_cs,
                        row_off=eff_rs,
                        width=eff_width,
                        height=eff_height,
                    )
                    dst.write(write_data, window=write_window)

                    # Release tile references.
                    del tile_raster, tile_result, tile_host, tile_data, write_data

                    tiles_processed += 1

    elapsed = time.perf_counter() - t0

    out_meta = read_raster_metadata(str(output_path))

    # Attach a diagnostic event to the returned metadata object.  Since
    # RasterMetadata is a frozen dataclass without a diagnostics list we
    # log to stderr-style detail string in the returned metadata and rely
    # on the caller to inspect the elapsed time in the metadata itself.
    # For visibility we create a standalone event that callers who capture
    # the return value can introspect.
    _streamed_last_diagnostic = RasterDiagnosticEvent(
        kind=RasterDiagnosticKind.RUNTIME,
        detail=(
            f"process_raster_streamed strategy=TILED "
            f"tiles={tiles_processed} tile_shape=({tile_h},{tile_w}) "
            f"halo={halo} raster_shape=({raster_h},{raster_w}) "
            f"path={input_path.name} elapsed={elapsed:.4f}s"
        ),
        residency=Residency.HOST,
        elapsed_seconds=elapsed,
    )
    # Store as module-level for diagnostic introspection.
    process_raster_streamed._last_diagnostic = _streamed_last_diagnostic  # type: ignore[attr-defined]

    return out_meta
