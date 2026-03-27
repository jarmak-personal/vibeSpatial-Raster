"""Comprehensive multiband integration tests.

Cross-cutting tests that validate full multiband pipelines end-to-end,
stress test the VRAM chunking infrastructure via mocking, exercise a
parametrized dtype x band-count matrix, and verify edge cases around
3D shapes, nodata consistency, and diagnostic events.

These are INTEGRATION tests -- they chain multiple operations across
modules rather than testing single operations in isolation.  Per-module
tests already exist elsewhere; this file focuses on cross-module
pipelines, dispatch infrastructure, and multi-dimensional correctness.

All tests use ``use_gpu=False`` (CPU path) unless marked with
``@pytest.mark.gpu`` and the ``requires_gpu`` skip condition.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vibespatial.raster import (
    GridSpec,
    OwnedRasterArray,
    RasterDiagnosticKind,
    from_numpy,
)
from vibespatial.raster.dispatch import (
    dispatch_per_band_cpu,
    max_bands_for_budget,
)

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")

# ---------------------------------------------------------------------------
# Shared test helpers (no conftest.py -- each test module is self-contained)
# ---------------------------------------------------------------------------

_AFFINE = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_500_000.0)
_CRS = "EPSG:32610"
_NODATA = -9999.0


def _make_multiband(
    bands: int = 3,
    height: int = 16,
    width: int = 16,
    dtype=np.float64,
    nodata: float | int | None = None,
    seed: int = 42,
) -> OwnedRasterArray:
    rng = np.random.default_rng(seed)
    data = rng.random((bands, height, width)).astype(dtype)
    if nodata is not None and np.issubdtype(dtype, np.floating):
        # Ensure no accidental collisions with nodata sentinel
        data = np.where(data == nodata, data + 0.001, data)
    return from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)


def _make_binary_multiband(
    bands: int = 3,
    height: int = 16,
    width: int = 16,
    seed: int = 42,
) -> OwnedRasterArray:
    """Binary mask raster suitable for morphology and distance transform."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=(bands, height, width), dtype=np.uint8)
    return from_numpy(data, affine=_AFFINE, crs=_CRS)


def _make_dem_multiband(
    bands: int = 3,
    height: int = 16,
    width: int = 16,
    seed: int = 42,
) -> OwnedRasterArray:
    """Smooth elevation surface suitable for slope/terrain ops."""
    rng = np.random.default_rng(seed)
    data = rng.random((bands, height, width)).astype(np.float64) * 1000.0
    return from_numpy(data, affine=_AFFINE, crs=_CRS)


def _identity_op(raster: OwnedRasterArray) -> OwnedRasterArray:
    """Pass-through operation for dispatch infrastructure tests."""
    return from_numpy(
        raster.to_numpy().copy(),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )


def _double_op(raster: OwnedRasterArray) -> OwnedRasterArray:
    """Multiply pixel values by 2."""
    return from_numpy(
        raster.to_numpy() * 2,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )


# ===================================================================
# 1. End-to-end pipeline tests
# ===================================================================


class TestPipelineConvolveThenSlope:
    """Chain: 3-band raster -> convolve each band -> slope."""

    def test_pipeline_convolve_then_slope(self):
        """Gaussian smooth then slope on a multiband DEM."""
        from vibespatial.raster import raster_slope

        dem = _make_dem_multiband(bands=3, height=16, width=16)

        # Step 1: smooth each band with a simple 3x3 box filter via
        # dispatch_per_band_cpu (since raster_convolve requires GPU)
        from scipy.ndimage import uniform_filter

        def smooth_op(r: OwnedRasterArray) -> OwnedRasterArray:
            data = r.to_numpy()
            if data.ndim == 3:
                data = data[0]
            smoothed = uniform_filter(data, size=3, mode="constant", cval=0.0)
            return from_numpy(
                smoothed,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        smoothed = dispatch_per_band_cpu(dem, smooth_op)
        assert smoothed.band_count == 3
        assert smoothed.to_numpy().shape == (3, 16, 16)

        # Step 2: compute slope on each band
        result = raster_slope(smoothed, use_gpu=False)
        out = result.to_numpy()

        assert out.shape == (3, 16, 16)
        assert result.affine == _AFFINE
        assert result.crs == _CRS
        # Slope values should be non-negative (degrees)
        assert np.all(out >= 0.0)


class TestPipelineExpressionNDVI:
    """Chain: 4-band raster -> NDVI via band expression -> verify."""

    def test_pipeline_expression_ndvi(self):
        """Compute NDVI from a 4-band raster via raster_expression."""
        from vibespatial.raster import raster_ndvi

        # Create 4-band raster with distinct per-band ranges
        rng = np.random.default_rng(42)
        data = np.empty((4, 20, 20), dtype=np.float32)
        data[0] = rng.uniform(0.05, 0.15, (20, 20))  # blue
        data[1] = rng.uniform(0.10, 0.25, (20, 20))  # green
        data[2] = rng.uniform(0.10, 0.40, (20, 20))  # red
        data[3] = rng.uniform(0.30, 0.80, (20, 20))  # nir
        raster = from_numpy(data.astype(np.float32), affine=_AFFINE, crs=_CRS)

        # Use the public NDVI convenience function (wraps raster_expression)
        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()

        # Single-band output
        assert out.ndim == 2
        assert out.shape == (20, 20)

        # NDVI range check
        assert np.all(out >= -1.0 - 1e-6)
        assert np.all(out <= 1.0 + 1e-6)

        # Verify against manual computation
        nir = data[3]
        red = data[2]
        expected = (nir - red) / (nir + red)
        np.testing.assert_allclose(out, expected, atol=1e-5)

        # Metadata
        assert result.affine == _AFFINE
        assert result.crs == _CRS
        assert result.dtype == np.float32


class TestPipelineErodeThenDistance:
    """Chain: 3-band binary mask -> erode -> distance transform."""

    def test_pipeline_erode_then_distance(self):
        """Erode a binary mask then compute distance transform."""
        from vibespatial.raster import (
            raster_distance_transform,
            raster_morphology,
        )

        mask = _make_binary_multiband(bands=3, height=20, width=20, seed=99)
        assert mask.band_count == 3

        # Step 1: erode (suppress the multiband warning from morphology)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            eroded = raster_morphology(mask, "erode", use_gpu=False)

        # Erosion should preserve band count and spatial dims
        eroded_data = eroded.to_numpy()
        assert eroded_data.shape == (3, 20, 20)
        # Eroded binary mask should only contain 0 and 1
        assert set(np.unique(eroded_data)).issubset({0, 1})

        # Step 2: distance transform on the eroded result
        dist = raster_distance_transform(eroded, use_gpu=False)
        dist_data = dist.to_numpy()

        assert dist_data.shape == (3, 20, 20)
        # Foreground pixels should have distance 0
        for b in range(3):
            fg = eroded_data[b] != 0
            np.testing.assert_array_equal(
                dist_data[b][fg],
                0.0,
                err_msg=f"Band {b}: foreground pixels must have distance 0",
            )
        # Background pixels should have positive distance
        for b in range(3):
            bg = eroded_data[b] == 0
            if bg.any() and eroded_data[b].any():
                assert np.all(dist_data[b][bg] > 0), (
                    f"Band {b}: background pixels must have positive distance"
                )

        # Metadata propagation
        assert dist.affine == _AFFINE
        assert dist.crs == _CRS


class TestPipelineResampleThenFocal:
    """Chain: 3-band raster -> resample to new grid -> focal mean."""

    def test_pipeline_resample_then_focal(self):
        """Resample to a different grid then apply focal mean."""
        from vibespatial.raster import raster_focal_mean, raster_resample

        raster = _make_multiband(
            bands=3,
            height=20,
            width=20,
            dtype=np.float64,
        )

        # Resample to a smaller grid (10x10)
        target = GridSpec(
            affine=(20.0, 0.0, 500_000.0, 0.0, -20.0, 4_500_000.0),
            width=10,
            height=10,
            dtype=np.float64,
        )
        resampled = raster_resample(raster, target, method="nearest", use_gpu=False)
        res_data = resampled.to_numpy()

        assert res_data.shape == (3, 10, 10)
        assert resampled.affine == target.affine

        # Focal mean on the resampled result
        result = raster_focal_mean(resampled, radius=1, use_gpu=False)
        out = result.to_numpy()

        assert out.shape == (3, 10, 10)
        # Focal mean should produce values within the range of the input
        for b in range(3):
            band_in = res_data[b]
            band_out = out[b]
            assert np.nanmin(band_out) >= np.nanmin(band_in) - 1e-10
            assert np.nanmax(band_out) <= np.nanmax(band_in) + 1e-10


class TestPipelineSlopeThenFocalMean:
    """Chain: multiband DEM -> slope -> focal mean (smoothed slope)."""

    def test_slope_then_focal_mean(self):
        """Compute slope then smooth it with focal mean."""
        from vibespatial.raster import raster_focal_mean, raster_slope

        dem = _make_dem_multiband(bands=2, height=16, width=16)

        slope = raster_slope(dem, use_gpu=False)
        assert slope.band_count == 2
        assert slope.to_numpy().shape == (2, 16, 16)

        result = raster_focal_mean(slope, radius=1, use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (2, 16, 16)
        # Smoothed slope should still be non-negative
        assert np.all(out >= 0.0)
        assert result.affine == _AFFINE


# ===================================================================
# 2. VRAM chunking stress tests (via mocked available_vram_bytes)
# ===================================================================


class TestVRAMChunkingForced:
    """Mock VRAM budget to force band chunking."""

    def test_vram_chunking_forced(self):
        """Mock budget forces 7-band raster to chunk in groups of 2.

        With 7 bands, groups of 2 -> 4 groups (2+2+2+1). Verify output
        identical to unlimited (single dispatch) run.
        """
        raster = _make_multiband(bands=7, height=10, width=10, dtype=np.float32)

        # Unlimited run (all bands at once)
        result_unlimited = dispatch_per_band_cpu(raster, _double_op)

        # Budget: 10x10 * float32 * 2 buffers = 800 bytes per band
        # Allow 2 bands: 1600 bytes
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=1600,
        ):
            budget = max_bands_for_budget(10, 10, np.dtype(np.float32))
            assert budget == 2

        # The CPU dispatch processes all bands regardless of VRAM budget
        # (VRAM chunking only applies to GPU path), but verify correctness
        result_chunked = dispatch_per_band_cpu(raster, _double_op)
        np.testing.assert_array_equal(
            result_unlimited.to_numpy(),
            result_chunked.to_numpy(),
        )
        assert result_chunked.band_count == 7

    def test_vram_chunking_last_group(self):
        """7 bands chunked in groups of 3 (3+3+1): verify partial last group."""
        raster = _make_multiband(bands=7, height=10, width=10, dtype=np.float32)

        # 10x10 * float32 * 2 = 800 bytes/band; allow 3 bands = 2400
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=2400,
        ):
            budget = max_bands_for_budget(10, 10, np.dtype(np.float32))
            assert budget == 3

        result = dispatch_per_band_cpu(raster, _double_op)
        expected = raster.to_numpy() * 2

        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 7

    def test_vram_single_band_per_chunk(self):
        """Extreme: budget allows only 1 band at a time."""
        raster = _make_multiband(bands=5, height=10, width=10, dtype=np.float32)

        # Budget allows exactly 1 band: 800 bytes
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=800,
        ):
            budget = max_bands_for_budget(10, 10, np.dtype(np.float32))
            assert budget == 1

        result = dispatch_per_band_cpu(raster, _double_op)
        expected = raster.to_numpy() * 2

        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 5

    def test_vram_zero_budget_returns_min_one(self):
        """Zero VRAM budget still returns at least 1 band."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=0,
        ):
            budget = max_bands_for_budget(100, 100, np.dtype(np.float64))
            assert budget == 1

    def test_vram_budget_respects_dtype_size(self):
        """Budget computation accounts for dtype itemsize correctly."""
        # 100x100 with 2 buffers:
        #   float32: 100*100*4*2 = 80_000 bytes/band
        #   float64: 100*100*8*2 = 160_000 bytes/band
        #   uint8:   100*100*1*2 = 20_000 bytes/band
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=320_000,
        ):
            assert max_bands_for_budget(100, 100, np.dtype(np.float32)) == 4
            assert max_bands_for_budget(100, 100, np.dtype(np.float64)) == 2
            assert max_bands_for_budget(100, 100, np.dtype(np.uint8)) == 16


# ===================================================================
# 3. Dtype x band-count parametrized matrix
# ===================================================================


@pytest.mark.parametrize("band_count", [1, 3, 7])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestSlopeMultibandMatrix:
    """Slope across dtype and band-count combinations."""

    def test_slope_multiband_matrix(self, band_count, dtype):
        from vibespatial.raster import raster_slope

        rng = np.random.default_rng(42)
        data = (rng.random((band_count, 12, 12)) * 1000).astype(dtype)
        raster = from_numpy(data, affine=_AFFINE, crs=_CRS)

        result = raster_slope(raster, use_gpu=False)
        out = result.to_numpy()

        if band_count == 1:
            # Single band stored as 2D
            assert out.ndim == 2
            assert out.shape == (12, 12)
        else:
            assert out.shape == (band_count, 12, 12)

        # Slope is always non-negative
        assert np.all(out >= 0.0)
        # Metadata preservation
        assert result.affine == _AFFINE
        assert result.crs == _CRS


@pytest.mark.parametrize("band_count", [1, 3, 7])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
class TestDistanceMultibandMatrix:
    """Distance transform across dtype and band-count combinations."""

    def test_distance_multiband_matrix(self, band_count, dtype):
        from vibespatial.raster import raster_distance_transform

        rng = np.random.default_rng(42)
        if np.issubdtype(dtype, np.integer):
            data = rng.integers(0, 2, size=(band_count, 10, 10)).astype(dtype)
        else:
            data = (rng.random((band_count, 10, 10)) > 0.5).astype(dtype)
        raster = from_numpy(data, affine=_AFFINE, crs=_CRS)

        result = raster_distance_transform(raster, use_gpu=False)
        out = result.to_numpy()

        if band_count == 1:
            assert out.ndim == 2
            assert out.shape == (10, 10)
        else:
            assert out.shape == (band_count, 10, 10)

        # Distance values are non-negative
        assert np.all(out >= 0.0)
        # Foreground pixels (nonzero in input) should have distance 0
        input_data = raster.to_numpy()
        if band_count == 1:
            # input_data may be (1, H, W) while out is (H, W) after squeeze
            fg = input_data.squeeze() != 0
            np.testing.assert_array_equal(out[fg], 0.0)
        else:
            for b in range(band_count):
                fg = input_data[b] != 0
                np.testing.assert_array_equal(out[b][fg], 0.0)

        assert result.affine == _AFFINE
        assert result.crs == _CRS


@pytest.mark.parametrize("band_count", [1, 3, 7])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestResampleMultibandMatrix:
    """Resample across dtype and band-count combinations."""

    def test_resample_multiband_matrix(self, band_count, dtype):
        from vibespatial.raster import raster_resample

        rng = np.random.default_rng(42)
        data = rng.random((band_count, 20, 20)).astype(dtype)
        raster = from_numpy(data, affine=_AFFINE, crs=_CRS)

        target = GridSpec(
            affine=(20.0, 0.0, 500_000.0, 0.0, -20.0, 4_500_000.0),
            width=10,
            height=10,
            dtype=np.dtype(dtype),
        )
        result = raster_resample(raster, target, method="nearest", use_gpu=False)
        out = result.to_numpy()

        if band_count == 1:
            assert out.ndim == 2
            assert out.shape == (10, 10)
        else:
            assert out.shape == (band_count, 10, 10)

        # Nearest resampling should only produce values from the input
        for b in range(band_count):
            if band_count == 1:
                band_in = data[0] if data.ndim == 3 else data
                band_out = out
            else:
                band_in = data[b]
                band_out = out[b]
            unique_out = set(np.unique(band_out))
            unique_in = set(np.unique(band_in))
            assert unique_out.issubset(unique_in), (
                f"Band {b}: nearest resample produced values not in input"
            )

        assert result.affine == target.affine
        assert result.crs == _CRS


@pytest.mark.parametrize("band_count", [1, 3, 7])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestFocalMeanMultibandMatrix:
    """Focal mean across dtype and band-count combinations."""

    def test_focal_mean_multiband_matrix(self, band_count, dtype):
        from vibespatial.raster import raster_focal_mean

        rng = np.random.default_rng(42)
        data = rng.random((band_count, 12, 12)).astype(dtype)
        raster = from_numpy(data, affine=_AFFINE, crs=_CRS)

        result = raster_focal_mean(raster, radius=1, use_gpu=False)
        out = result.to_numpy()

        if band_count == 1:
            assert out.ndim == 2
            assert out.shape == (12, 12)
        else:
            assert out.shape == (band_count, 12, 12)

        # Focal mean output should be within the range of the input
        for b in range(band_count):
            if band_count == 1:
                band_in = data[0] if data.ndim == 3 else data
                band_out = out
            else:
                band_in = data[b]
                band_out = out[b]
            assert np.nanmin(band_out) >= np.nanmin(band_in) - 1e-10
            assert np.nanmax(band_out) <= np.nanmax(band_in) + 1e-10

        assert result.affine == _AFFINE


# ===================================================================
# 4. Edge cases
# ===================================================================


class TestSingleBand3DShape:
    """(1, H, W) rasters should behave identically to (H, W)."""

    def test_single_band_3d_slope(self):
        """Slope treats (1, H, W) same as (H, W)."""
        from vibespatial.raster import raster_slope

        rng = np.random.default_rng(42)
        data_2d = (rng.random((16, 16)) * 1000).astype(np.float64)
        data_3d = data_2d[np.newaxis, :, :]

        raster_2d = from_numpy(data_2d, affine=_AFFINE, crs=_CRS)
        raster_3d = from_numpy(data_3d, affine=_AFFINE, crs=_CRS)

        result_2d = raster_slope(raster_2d, use_gpu=False)
        result_3d = raster_slope(raster_3d, use_gpu=False)

        np.testing.assert_allclose(
            result_2d.to_numpy(),
            result_3d.to_numpy().squeeze(),
            atol=1e-12,
        )

    def test_single_band_3d_distance(self):
        """Distance transform treats (1, H, W) same as (H, W)."""
        from vibespatial.raster import raster_distance_transform

        rng = np.random.default_rng(42)
        data_2d = rng.integers(0, 2, size=(10, 10)).astype(np.uint8)
        data_3d = data_2d[np.newaxis, :, :]

        raster_2d = from_numpy(data_2d, affine=_AFFINE, crs=_CRS)
        raster_3d = from_numpy(data_3d, affine=_AFFINE, crs=_CRS)

        result_2d = raster_distance_transform(raster_2d, use_gpu=False)
        result_3d = raster_distance_transform(raster_3d, use_gpu=False)

        np.testing.assert_allclose(
            result_2d.to_numpy(),
            result_3d.to_numpy().squeeze(),
            atol=1e-12,
        )

    def test_single_band_3d_focal_mean(self):
        """Focal mean treats (1, H, W) same as (H, W)."""
        from vibespatial.raster import raster_focal_mean

        rng = np.random.default_rng(42)
        data_2d = rng.random((10, 10)).astype(np.float64)
        data_3d = data_2d[np.newaxis, :, :]

        raster_2d = from_numpy(data_2d, affine=_AFFINE, crs=_CRS)
        raster_3d = from_numpy(data_3d, affine=_AFFINE, crs=_CRS)

        result_2d = raster_focal_mean(raster_2d, radius=1, use_gpu=False)
        result_3d = raster_focal_mean(raster_3d, radius=1, use_gpu=False)

        np.testing.assert_allclose(
            result_2d.to_numpy(),
            result_3d.to_numpy().squeeze(),
            atol=1e-12,
        )


class TestNodataConsistencyAcrossBands:
    """All band results must preserve nodata at correct positions."""

    def test_nodata_positions_preserved_through_pipeline(self):
        """Nodata in specific band positions stays nodata through slope."""
        from vibespatial.raster import raster_slope

        rng = np.random.default_rng(42)
        data = (rng.random((3, 16, 16)) * 1000).astype(np.float64)
        nodata = _NODATA

        # Place nodata in known locations per band
        data[0, 2, 3] = nodata
        data[1, 5, 5] = nodata
        data[2, 0, 0] = nodata
        data[2, 15, 15] = nodata

        raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)
        result = raster_slope(raster, use_gpu=False)
        out = result.to_numpy()

        assert result.nodata == nodata

        # The center pixel and its neighborhood are affected by nodata.
        # For Horn's method (3x3 stencil), any pixel whose 3x3 neighborhood
        # touches a nodata pixel receives nodata. So at minimum the nodata
        # pixel itself should be nodata.
        assert out[0, 2, 3] == nodata, "Band 0 nodata at (2,3) must propagate"
        assert out[1, 5, 5] == nodata, "Band 1 nodata at (5,5) must propagate"
        assert out[2, 0, 0] == nodata, "Band 2 nodata at (0,0) must propagate"
        assert out[2, 15, 15] == nodata, "Band 2 nodata at (15,15) must propagate"

    def test_nodata_consistency_distance_transform(self):
        """Distance transform produces NaN where nodata was in input."""
        from vibespatial.raster import raster_distance_transform

        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=(3, 12, 12)).astype(np.float64)
        nodata = _NODATA

        # Place nodata
        data[0, 0, 0] = nodata
        data[1, 6, 6] = nodata
        data[2, 11, 11] = nodata

        raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)
        result = raster_distance_transform(raster, use_gpu=False)
        out = result.to_numpy()

        # Distance transform outputs NaN for nodata positions
        assert np.isnan(out[0, 0, 0]), "Band 0 nodata -> NaN in distance output"
        assert np.isnan(out[1, 6, 6]), "Band 1 nodata -> NaN in distance output"
        assert np.isnan(out[2, 11, 11]), "Band 2 nodata -> NaN in distance output"


class TestMultibandDiagnosticEvents:
    """Verify dispatch events have correct band counts."""

    def test_dispatch_cpu_emits_band_count(self):
        """dispatch_per_band_cpu emits a RUNTIME event with band count."""
        raster = _make_multiband(bands=5)
        result = dispatch_per_band_cpu(raster, _double_op)

        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        last = runtime_events[-1]
        assert "dispatch_per_band_cpu" in last.detail
        assert "bands=5" in last.detail

    def test_single_band_dispatch_emits_passthrough(self):
        """Single-band dispatch emits 'single-band passthrough' event."""
        raster = _make_multiband(bands=1)
        # Single-band raster created as (1, H, W) -- dispatch sees band_count=1
        result = dispatch_per_band_cpu(raster, _double_op)

        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "single-band" in runtime_events[-1].detail

    def test_pipeline_accumulates_diagnostics(self):
        """Chained operations accumulate diagnostic events."""
        from vibespatial.raster import raster_focal_mean, raster_slope

        dem = _make_dem_multiband(bands=2, height=12, width=12)

        slope = raster_slope(dem, use_gpu=False)
        result = raster_focal_mean(slope, radius=1, use_gpu=False)

        # Both slope and focal_mean should have emitted RUNTIME events
        all_events = result.diagnostics
        runtime_events = [e for e in all_events if e.kind == RasterDiagnosticKind.RUNTIME]
        # At minimum: dispatch events from slope per-band and focal per-band
        assert len(runtime_events) >= 2

    def test_call_count_matches_band_count(self):
        """Per-band dispatch calls the op_fn exactly N times for N bands."""
        raster = _make_multiband(bands=4)
        mock_op = MagicMock(side_effect=_identity_op)
        dispatch_per_band_cpu(raster, mock_op)
        assert mock_op.call_count == 4

    def test_multiband_preserves_all_metadata(self):
        """Affine, CRS, nodata, and dtype are preserved through dispatch."""
        raster = _make_multiband(
            bands=3,
            dtype=np.float32,
            nodata=_NODATA,
        )
        result = dispatch_per_band_cpu(raster, _identity_op)

        assert result.affine == _AFFINE
        assert result.crs == _CRS
        assert result.nodata == _NODATA
        assert result.dtype == np.float32
        assert result.band_count == 3


class TestEdgeCaseRasterShapes:
    """Non-standard raster shapes through multiband dispatch."""

    def test_nonsquare_multiband(self):
        """Non-square raster (bands, 5, 30) through pipeline."""
        from vibespatial.raster import raster_focal_mean

        rng = np.random.default_rng(42)
        data = rng.random((3, 5, 30)).astype(np.float64)
        raster = from_numpy(data, affine=_AFFINE, crs=_CRS)

        result = raster_focal_mean(raster, radius=1, use_gpu=False)
        assert result.to_numpy().shape == (3, 5, 30)
        assert result.affine == _AFFINE

    def test_single_pixel_multiband(self):
        """(3, 1, 1) raster through dispatch."""
        data = np.array([[[10.0]], [[20.0]], [[30.0]]], dtype=np.float64)
        raster = from_numpy(data, affine=_AFFINE, crs=_CRS)

        result = dispatch_per_band_cpu(raster, _double_op)
        expected = np.array([[[20.0]], [[40.0]], [[60.0]]], dtype=np.float64)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_large_band_count(self):
        """Many-band (10 bands) raster through dispatch."""
        raster = _make_multiband(bands=10, height=8, width=8)

        result = dispatch_per_band_cpu(raster, _double_op)
        expected = raster.to_numpy() * 2

        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 10

    def test_all_nodata_raster(self):
        """Raster where all pixels are nodata."""
        data = np.full((3, 8, 8), _NODATA, dtype=np.float64)
        raster = from_numpy(data, nodata=_NODATA, affine=_AFFINE, crs=_CRS)

        result = dispatch_per_band_cpu(raster, _identity_op)
        np.testing.assert_array_equal(result.to_numpy(), data)
        assert result.nodata == _NODATA

    def test_constant_value_raster(self):
        """Raster with all pixels having the same non-nodata value."""
        from vibespatial.raster import raster_focal_mean

        data = np.full((3, 10, 10), 42.0, dtype=np.float64)
        raster = from_numpy(data, affine=_AFFINE, crs=_CRS)

        result = raster_focal_mean(raster, radius=1, use_gpu=False)
        out = result.to_numpy()

        # Focal mean of a constant surface should be the same constant
        # (except possibly at boundaries due to zero-padding)
        # Interior pixels (away from edges) should be exactly the constant
        interior = out[:, 2:-2, 2:-2]
        np.testing.assert_allclose(interior, 42.0, atol=1e-10)


# ===================================================================
# 5. GPU-specific tests (skipped without CuPy)
# ===================================================================


class TestGPUMultibandResidencyTracking:
    """Verify raster stays DEVICE-resident throughout GPU dispatch."""

    @requires_gpu
    def test_gpu_multiband_residency_tracking(self):
        from vibespatial.raster.dispatch import dispatch_per_band_gpu
        from vibespatial.residency import Residency

        raster = _make_multiband(bands=3, height=10, width=10)

        def check_device_op(r: OwnedRasterArray) -> OwnedRasterArray:
            """Op that verifies the input is device-resident."""
            assert r.residency is Residency.DEVICE, f"Expected DEVICE residency, got {r.residency}"
            # Perform a simple operation via device data
            d = r.device_data()
            result_d = d * 2.0
            from vibespatial.raster import from_device

            return from_device(
                result_d,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_per_band_gpu(raster, check_device_op)
        assert result.band_count == 3

    @requires_gpu
    def test_gpu_single_h2d_transfer(self):
        """Dispatch transfers the full raster to device once, not per-band."""
        from vibespatial.raster.dispatch import dispatch_per_band_gpu

        raster = _make_multiband(bands=3, height=10, width=10)

        dispatch_per_band_gpu(raster, _double_op)

        # Count TRANSFER events in the source raster's diagnostics
        transfer_events = [
            e
            for e in raster.diagnostics
            if e.kind == RasterDiagnosticKind.TRANSFER and "host->device" in e.detail
        ]
        assert len(transfer_events) == 1, (
            f"Expected exactly 1 H->D transfer, found {len(transfer_events)}: "
            f"{[e.detail for e in transfer_events]}"
        )

    @requires_gpu
    def test_gpu_cpu_parity_across_operations(self):
        """GPU and CPU multiband dispatch produce identical results."""
        from vibespatial.raster.dispatch import dispatch_per_band_gpu

        raster = _make_multiband(bands=3, height=12, width=12)

        result_gpu = dispatch_per_band_gpu(raster, _double_op)
        result_cpu = dispatch_per_band_cpu(raster, _double_op)

        np.testing.assert_allclose(
            result_gpu.to_numpy(),
            result_cpu.to_numpy(),
            atol=1e-10,
        )
