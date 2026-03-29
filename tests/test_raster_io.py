"""Tests for raster IO: read/write GeoTIFF via rasterio."""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.io import has_nvimgcodec_support, has_rasterio_support

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
requires_nvimgcodec = pytest.mark.skipif(
    not has_nvimgcodec_support() or not HAS_GPU,
    reason="nvImageCodec or CuPy not available",
)

pytestmark = pytest.mark.skipif(
    not has_rasterio_support(),
    reason="rasterio not installed",
)


@pytest.fixture
def tmp_geotiff(tmp_path):
    """Create a small GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "test.tif"
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    transform = from_bounds(0.0, 0.0, 5.0, 4.0, 5, 4)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        width=5,
        height=4,
        count=1,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return path, data, transform


@pytest.fixture
def tmp_multiband_geotiff(tmp_path):
    """Create a multi-band GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "multiband.tif"
    data = np.random.default_rng(42).random((3, 10, 15)).astype(np.float64)
    transform = from_bounds(100.0, 200.0, 115.0, 210.0, 15, 10)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float64",
        width=15,
        height=10,
        count=3,
        transform=transform,
    ) as dst:
        dst.write(data)

    return path, data, transform


class TestHasRasterioSupport:
    def test_available(self):
        assert has_rasterio_support()


class TestReadRasterMetadata:
    def test_metadata(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster_metadata

        path, data, transform = tmp_geotiff
        meta = read_raster_metadata(path)
        assert meta.height == 4
        assert meta.width == 5
        assert meta.band_count == 1
        assert meta.dtype == np.float32
        assert meta.nodata == -9999.0
        assert meta.driver == "GTiff"
        assert meta.pixel_count == 20

    def test_multiband_metadata(self, tmp_multiband_geotiff):
        from vibespatial.raster.io import read_raster_metadata

        path, data, transform = tmp_multiband_geotiff
        meta = read_raster_metadata(path)
        assert meta.band_count == 3
        assert meta.height == 10
        assert meta.width == 15


class TestReadRaster:
    def test_full_read(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, expected_data, _ = tmp_geotiff
        raster = read_raster(path)
        assert raster.residency is Residency.HOST
        assert raster.height == 4
        assert raster.width == 5
        assert raster.band_count == 1
        assert raster.dtype == np.float32
        assert raster.nodata == -9999.0
        np.testing.assert_array_equal(raster.to_numpy(), expected_data)

    def test_multiband_read(self, tmp_multiband_geotiff):
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_multiband_geotiff
        raster = read_raster(path)
        assert raster.band_count == 3
        assert raster.shape == (3, 10, 15)
        np.testing.assert_array_almost_equal(raster.to_numpy(), expected_data)

    def test_select_bands(self, tmp_multiband_geotiff):
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_multiband_geotiff
        raster = read_raster(path, bands=[2])
        assert raster.band_count == 1
        assert raster.shape == (10, 15)  # single band squeezed to 2D
        np.testing.assert_array_almost_equal(raster.to_numpy(), expected_data[1])

    def test_windowed_read(self, tmp_geotiff):
        from vibespatial.raster.buffers import RasterWindow
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_geotiff
        window = RasterWindow(col_off=1, row_off=1, width=3, height=2)
        raster = read_raster(path, window=window)
        assert raster.height == 2
        assert raster.width == 3
        np.testing.assert_array_equal(raster.to_numpy(), expected_data[1:3, 1:4])

    def test_affine_preserved(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster

        path, _, transform = tmp_geotiff
        raster = read_raster(path)
        assert raster.affine[0] == pytest.approx(transform.a)
        assert raster.affine[4] == pytest.approx(transform.e)

    @gpu
    def test_read_to_device(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, _, _ = tmp_geotiff
        raster = read_raster(path, residency=Residency.DEVICE)
        assert raster.residency is Residency.DEVICE
        assert raster.device_state is not None


class TestWriteRaster:
    def test_roundtrip(self, tmp_path, tmp_geotiff):
        from vibespatial.raster.io import read_raster, write_raster

        src_path, expected_data, _ = tmp_geotiff
        raster = read_raster(src_path)

        out_path = tmp_path / "output.tif"
        write_raster(out_path, raster)

        roundtrip = read_raster(out_path)
        np.testing.assert_array_equal(roundtrip.to_numpy(), expected_data)
        assert roundtrip.nodata == raster.nodata
        assert roundtrip.dtype == raster.dtype

    def test_write_records_diagnostic(self, tmp_path, tmp_geotiff):
        from vibespatial.raster.buffers import RasterDiagnosticKind
        from vibespatial.raster.io import read_raster, write_raster

        src_path, _, _ = tmp_geotiff
        raster = read_raster(src_path)
        initial_count = len(raster.diagnostics)

        out_path = tmp_path / "output.tif"
        write_raster(out_path, raster)

        assert len(raster.diagnostics) == initial_count + 1
        assert raster.diagnostics[-1].kind == RasterDiagnosticKind.MATERIALIZATION


class TestDecodeBackendDispatch:
    """Tests for the decode_backend parameter and GPU/rasterio dispatch."""

    def test_read_raster_explicit_rasterio_backend(self, tmp_geotiff):
        """decode_backend='rasterio' works and skips nvimgcodec entirely."""
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, expected_data, _ = tmp_geotiff
        raster = read_raster(path, decode_backend="rasterio")
        assert raster.residency is Residency.HOST
        assert raster.height == 4
        assert raster.width == 5
        np.testing.assert_array_equal(raster.to_numpy(), expected_data)

    def test_read_raster_auto_falls_back_to_rasterio(self, tmp_geotiff):
        """With nvimgcodec unavailable, auto backend still works via rasterio."""
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_geotiff
        # auto is the default -- on CI without nvimgcodec this must still work
        raster = read_raster(path, decode_backend="auto")
        np.testing.assert_array_equal(raster.to_numpy(), expected_data)

    def test_read_raster_backend_diagnostic(self, tmp_geotiff):
        """Verify diagnostic event says 'backend=rasterio'."""
        from vibespatial.raster.buffers import RasterDiagnosticKind
        from vibespatial.raster.io import read_raster

        path, _, _ = tmp_geotiff
        raster = read_raster(path, decode_backend="rasterio")
        runtime_events = [e for e in raster.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "backend=rasterio" in runtime_events[-1].detail

    def test_has_nvimgcodec_support_returns_bool(self):
        """Verify the public API function returns a bool."""
        result = has_nvimgcodec_support()
        assert isinstance(result, bool)


class TestNvimgcodecBackend:
    """GPU-dependent tests for nvImageCodec decode path."""

    @requires_nvimgcodec
    def test_read_raster_nvimgcodec_device(self, tmp_geotiff):
        """Verify device-resident result when nvimgcodec available."""
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, _, _ = tmp_geotiff
        raster = read_raster(path, residency=Residency.DEVICE, decode_backend="nvimgcodec")
        assert raster.residency is Residency.DEVICE
        assert raster.device_state is not None
        runtime_events = [e for e in raster.diagnostics if "backend=nvimgcodec" in e.detail]
        assert len(runtime_events) >= 1

    @requires_nvimgcodec
    def test_read_raster_nvimgcodec_to_host(self, tmp_geotiff):
        """Verify HOST residency with nvimgcodec (auto D->H transfer)."""
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, expected_data, _ = tmp_geotiff
        raster = read_raster(path, residency=Residency.HOST, decode_backend="nvimgcodec")
        assert raster.residency is Residency.HOST
        # Data should be accessible as numpy
        host_data = raster.to_numpy()
        assert host_data.shape == expected_data.shape


# ---------------------------------------------------------------------------
# Streamed windowed IO (vibeSpatial-fx3.6)
# ---------------------------------------------------------------------------


def _make_geotiff(path, data, *, nodata=None, crs_epsg=4326):
    """Helper: write a numpy array to a GeoTIFF at *path*."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    if data.ndim == 2:
        band_count = 1
        height, width = data.shape
    else:
        band_count, height, width = data.shape

    transform = from_bounds(0.0, 0.0, float(width), float(height), width, height)

    profile = {
        "driver": "GTiff",
        "dtype": str(data.dtype),
        "width": width,
        "height": height,
        "count": band_count,
        "transform": transform,
        "crs": CRS.from_epsg(crs_epsg),
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(str(path), "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)

    return transform


class TestProcessRasterStreamed:
    """Tests for process_raster_streamed (vibeSpatial-fx3.6)."""

    # -- Fixtures local to this class --

    @pytest.fixture
    def tiff_64x64(self, tmp_path):
        """64x64 float32 raster with gradient data and nodata=-9999."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        path = tmp_path / "grad_64x64.tif"
        transform = _make_geotiff(path, data, nodata=-9999.0)
        return path, data, transform

    @pytest.fixture
    def tiff_multiband(self, tmp_path):
        """3-band 32x32 float64 raster."""
        rng = np.random.default_rng(42)
        data = rng.random((3, 32, 32)).astype(np.float64)
        path = tmp_path / "multi_32x32.tif"
        transform = _make_geotiff(path, data, nodata=-9999.0)
        return path, data, transform

    @pytest.fixture
    def tiff_non_square(self, tmp_path):
        """47x31 raster (not divisible by common tile sizes)."""
        data = np.arange(47 * 31, dtype=np.float32).reshape(47, 31)
        path = tmp_path / "nonsquare_47x31.tif"
        transform = _make_geotiff(path, data, nodata=-9999.0)
        return path, data, transform

    # -- Helper to build plans --

    @staticmethod
    def _make_plan(strategy, tile_shape=None, halo=0, n_tiles=0):
        from vibespatial.raster.buffers import RasterPlan, TilingStrategy

        return RasterPlan(
            strategy=TilingStrategy(strategy),
            tile_shape=tile_shape,
            halo=halo,
            n_tiles=n_tiles,
            estimated_vram_per_tile=0,
        )

    # -- Tests --

    def test_streamed_identity(self, tiff_64x64, tmp_path):
        """Identity op: output should exactly match input."""
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, expected_data, _ = tiff_64x64
        output_path = tmp_path / "identity.tif"

        plan = self._make_plan("tiled", tile_shape=(32, 32), n_tiles=4)

        def identity(r):
            return r

        meta = process_raster_streamed(input_path, output_path, identity, plan)

        assert meta.height == 64
        assert meta.width == 64

        result = read_raster(str(output_path))
        np.testing.assert_array_equal(result.to_numpy(), expected_data)

    def test_streamed_double(self, tiff_64x64, tmp_path):
        """Double op: output = input * 2."""
        from vibespatial.raster.buffers import from_numpy
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, expected_data, _ = tiff_64x64
        output_path = tmp_path / "doubled.tif"

        plan = self._make_plan("tiled", tile_shape=(32, 32), n_tiles=4)

        def double_op(r):
            data = r.to_numpy() * 2
            return from_numpy(data, nodata=r.nodata, affine=r.affine, crs=r.crs)

        process_raster_streamed(input_path, output_path, double_op, plan)

        result = read_raster(str(output_path))
        np.testing.assert_array_equal(result.to_numpy(), expected_data * 2)

    def test_streamed_tiled_matches_whole(self, tiff_64x64, tmp_path):
        """Tiled and WHOLE fast path produce identical output."""
        from vibespatial.raster.buffers import from_numpy
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, _, _ = tiff_64x64

        def double_op(r):
            data = r.to_numpy() * 2
            return from_numpy(data, nodata=r.nodata, affine=r.affine, crs=r.crs)

        out_whole = tmp_path / "whole.tif"
        plan_whole = self._make_plan("whole")
        process_raster_streamed(input_path, out_whole, double_op, plan_whole)

        out_tiled = tmp_path / "tiled.tif"
        plan_tiled = self._make_plan("tiled", tile_shape=(16, 16), n_tiles=16)
        process_raster_streamed(input_path, out_tiled, double_op, plan_tiled)

        r_whole = read_raster(str(out_whole))
        r_tiled = read_raster(str(out_tiled))
        np.testing.assert_array_equal(r_whole.to_numpy(), r_tiled.to_numpy())

    def test_streamed_with_halo(self, tiff_64x64, tmp_path):
        """Halo > 0: tile includes overlap, result is trimmed correctly."""
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, expected_data, _ = tiff_64x64
        output_path = tmp_path / "halo.tif"

        plan = self._make_plan("tiled", tile_shape=(32, 32), halo=4, n_tiles=4)

        # Identity op — if halo trimming works the output matches input.
        def identity(r):
            return r

        process_raster_streamed(
            input_path,
            output_path,
            identity,
            plan,
            halo=4,
        )

        result = read_raster(str(output_path))
        np.testing.assert_array_equal(result.to_numpy(), expected_data)

    def test_streamed_metadata_preservation(self, tiff_64x64, tmp_path):
        """CRS, nodata, dtype, and affine are preserved through streamed IO."""
        from vibespatial.raster.io import process_raster_streamed, read_raster_metadata

        input_path, _, transform = tiff_64x64
        output_path = tmp_path / "meta_check.tif"

        plan = self._make_plan("tiled", tile_shape=(32, 32), n_tiles=4)

        def identity(r):
            return r

        out_meta = process_raster_streamed(input_path, output_path, identity, plan)

        in_meta = read_raster_metadata(str(input_path))

        assert out_meta.dtype == in_meta.dtype
        assert out_meta.nodata == in_meta.nodata
        assert out_meta.height == in_meta.height
        assert out_meta.width == in_meta.width
        assert out_meta.band_count == in_meta.band_count
        # CRS should both be EPSG:4326.
        assert out_meta.crs is not None
        assert out_meta.crs.to_epsg() == 4326
        # Affine: pixel size and origin should match.
        assert out_meta.affine[0] == pytest.approx(in_meta.affine[0])
        assert out_meta.affine[4] == pytest.approx(in_meta.affine[4])
        assert out_meta.affine[2] == pytest.approx(in_meta.affine[2])
        assert out_meta.affine[5] == pytest.approx(in_meta.affine[5])

    def test_streamed_multiband(self, tiff_multiband, tmp_path):
        """3-band raster is streamed correctly."""
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, expected_data, _ = tiff_multiband
        output_path = tmp_path / "multiband.tif"

        plan = self._make_plan("tiled", tile_shape=(16, 16), n_tiles=4)

        def identity(r):
            return r

        meta = process_raster_streamed(input_path, output_path, identity, plan)

        assert meta.band_count == 3
        result = read_raster(str(output_path))
        np.testing.assert_array_almost_equal(result.to_numpy(), expected_data)

    def test_whole_fast_path(self, tiff_64x64, tmp_path):
        """WHOLE plan reads/writes in one shot without tiling."""
        from vibespatial.raster.buffers import from_numpy
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, expected_data, _ = tiff_64x64
        output_path = tmp_path / "whole_fast.tif"

        plan = self._make_plan("whole")

        def double_op(r):
            data = r.to_numpy() * 2
            return from_numpy(data, nodata=r.nodata, affine=r.affine, crs=r.crs)

        meta = process_raster_streamed(input_path, output_path, double_op, plan)

        assert meta.height == 64
        assert meta.width == 64

        result = read_raster(str(output_path))
        np.testing.assert_array_equal(result.to_numpy(), expected_data * 2)

    def test_edge_tiles(self, tiff_non_square, tmp_path):
        """Raster not evenly divisible by tile size produces correct output."""
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, expected_data, _ = tiff_non_square
        output_path = tmp_path / "edge_tiles.tif"

        # 47x31 with 16x16 tiles: partial tiles on right and bottom edges.
        plan = self._make_plan("tiled", tile_shape=(16, 16), n_tiles=6)

        def identity(r):
            return r

        meta = process_raster_streamed(input_path, output_path, identity, plan)

        assert meta.height == 47
        assert meta.width == 31

        result = read_raster(str(output_path))
        np.testing.assert_array_equal(result.to_numpy(), expected_data)

    def test_malformed_plan_raises(self, tiff_64x64, tmp_path):
        """TILED plan with tile_shape=None raises ValueError."""
        from vibespatial.raster.io import process_raster_streamed

        input_path, _, _ = tiff_64x64
        output_path = tmp_path / "bad_plan.tif"

        plan = self._make_plan("tiled", tile_shape=None, n_tiles=0)

        with pytest.raises(ValueError, match="tile_shape is None"):
            process_raster_streamed(input_path, output_path, lambda r: r, plan)

    def test_diagnostic_event_whole(self, tiff_64x64, tmp_path):
        """WHOLE path records a diagnostic event."""
        from vibespatial.raster.io import process_raster_streamed

        input_path, _, _ = tiff_64x64
        output_path = tmp_path / "diag_whole.tif"

        plan = self._make_plan("whole")
        process_raster_streamed(input_path, output_path, lambda r: r, plan)

        # The WHOLE path appends a diagnostic to the result raster's list,
        # but since we only return RasterMetadata we check the module-level
        # diagnostic stored on the function.
        diag = process_raster_streamed._last_diagnostic  # type: ignore[attr-defined]
        assert "strategy=WHOLE" in diag.detail

    def test_diagnostic_event_tiled(self, tiff_64x64, tmp_path):
        """TILED path records a diagnostic event with tile count."""
        from vibespatial.raster.io import process_raster_streamed

        input_path, _, _ = tiff_64x64
        output_path = tmp_path / "diag_tiled.tif"

        plan = self._make_plan("tiled", tile_shape=(32, 32), n_tiles=4)
        process_raster_streamed(input_path, output_path, lambda r: r, plan)

        diag = process_raster_streamed._last_diagnostic  # type: ignore[attr-defined]
        assert "strategy=TILED" in diag.detail
        assert "tiles=4" in diag.detail

    def test_streamed_halo_stencil_op(self, tiff_64x64, tmp_path):
        """Verify halo provides sufficient context for a stencil operation.

        Uses a 3x3 average filter as a simple stencil. Compares the streamed
        result against a non-streamed (whole) result to verify that halo
        overlap prevents seam artifacts at tile boundaries.
        """
        from vibespatial.raster.buffers import from_numpy
        from vibespatial.raster.io import process_raster_streamed, read_raster

        input_path, _, _ = tiff_64x64

        def avg_3x3(r):
            """3x3 neighbourhood mean (numpy, zero-padded)."""
            from scipy.ndimage import uniform_filter

            data = r.to_numpy().astype(np.float64)
            filtered = uniform_filter(data, size=3, mode="constant", cval=0.0)
            return from_numpy(
                filtered.astype(np.float32),
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        # WHOLE: ground-truth
        out_whole = tmp_path / "stencil_whole.tif"
        plan_whole = self._make_plan("whole")
        process_raster_streamed(input_path, out_whole, avg_3x3, plan_whole)

        # TILED with halo=1 (radius of 3x3 kernel)
        out_tiled = tmp_path / "stencil_tiled.tif"
        plan_tiled = self._make_plan("tiled", tile_shape=(32, 32), halo=1, n_tiles=4)
        process_raster_streamed(
            input_path,
            out_tiled,
            avg_3x3,
            plan_tiled,
            halo=1,
        )

        r_whole = read_raster(str(out_whole)).to_numpy()
        r_tiled = read_raster(str(out_tiled)).to_numpy()

        # Interior pixels (away from raster boundary) must match exactly.
        # Edge pixels may differ due to zero-padding at raster boundary.
        np.testing.assert_allclose(r_tiled[2:-2, 2:-2], r_whole[2:-2, 2:-2], atol=1e-5)

    def test_streamed_nodata_propagation(self, tmp_path):
        """Nodata pixels survive streaming: input nodata positions preserved."""
        from vibespatial.raster.io import process_raster_streamed, read_raster

        data = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        data[5, 10] = -9999.0
        data[20, 25] = -9999.0

        path = tmp_path / "nodata.tif"
        _make_geotiff(path, data, nodata=-9999.0)

        output_path = tmp_path / "nodata_out.tif"
        plan = self._make_plan("tiled", tile_shape=(16, 16), n_tiles=4)

        def identity(r):
            return r

        process_raster_streamed(path, output_path, identity, plan)

        result = read_raster(str(output_path))
        result_data = result.to_numpy()

        assert result_data[5, 10] == -9999.0
        assert result_data[20, 25] == -9999.0
        assert result.nodata == -9999.0
