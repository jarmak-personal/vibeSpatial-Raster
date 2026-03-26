"""Tests for histogram, CDF, equalization, and percentile operations."""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import from_numpy

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raster_uniform():
    """Uniform data [0..255] as uint8."""
    data = np.arange(256, dtype=np.uint8).reshape(16, 16)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 16.0))


@pytest.fixture
def raster_float():
    """Float raster with known distribution."""
    rng = np.random.RandomState(42)
    data = rng.normal(100.0, 20.0, size=(50, 50)).astype(np.float64)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 50.0))


@pytest.fixture
def raster_with_nodata():
    """Raster with nodata pixels."""
    data = np.array(
        [[10.0, 20.0, 30.0], [40.0, -9999.0, 60.0], [70.0, 80.0, 90.0]],
        dtype=np.float64,
    )
    return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))


@pytest.fixture
def raster_uint8_nodata():
    """uint8 raster with nodata sentinel."""
    data = np.array([[10, 20, 30], [40, 0, 60], [70, 80, 90]], dtype=np.uint8)
    return from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))


# ---------------------------------------------------------------------------
# CPU tests — raster_histogram
# ---------------------------------------------------------------------------


class TestRasterHistogramCPU:
    def test_basic_counts(self, raster_uniform):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(raster_uniform, bins=256, use_gpu=False)
        assert counts.shape == (256,)
        assert edges.shape == (257,)
        # Each value 0..255 appears exactly once
        assert counts.sum() == 256

    def test_total_count_matches_pixels(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(raster_float, bins=50, use_gpu=False)
        assert counts.sum() == raster_float.pixel_count

    def test_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_histogram

        counts, _edges = raster_histogram(raster_with_nodata, bins=10, use_gpu=False)
        # 9 pixels, 1 nodata -> 8 valid
        assert counts.sum() == 8

    def test_custom_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(
            raster_float, bins=10, range_min=50.0, range_max=150.0, use_gpu=False
        )
        assert edges[0] == pytest.approx(50.0)
        assert edges[-1] == pytest.approx(150.0)
        assert counts.shape == (10,)

    def test_bin_edges_monotonic(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        _counts, edges = raster_histogram(raster_float, bins=100, use_gpu=False)
        assert np.all(np.diff(edges) > 0), "Bin edges must be strictly increasing"


# ---------------------------------------------------------------------------
# CPU tests — raster_cumulative_distribution
# ---------------------------------------------------------------------------


class TestRasterCDFCPU:
    def test_monotonically_increasing(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=False)
        assert np.all(np.diff(cdf) >= 0), "CDF must be monotonically non-decreasing"

    def test_final_value_equals_pixel_count(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=False)
        assert cdf[-1] == raster_float.pixel_count

    def test_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_with_nodata, bins=10, use_gpu=False)
        assert cdf[-1] == 8  # 9 - 1 nodata


# ---------------------------------------------------------------------------
# CPU tests — raster_histogram_equalize
# ---------------------------------------------------------------------------


class TestRasterHistogramEqualizeCPU:
    def test_output_dtype_uint8(self, raster_uniform):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_uniform, use_gpu=False)
        assert result.dtype == np.uint8

    def test_output_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        data = result.to_numpy()
        assert data.min() >= 0
        assert data.max() <= 255

    def test_roughly_uniform_distribution(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        data = result.to_numpy().ravel()
        counts, _ = np.histogram(data, bins=16, range=(0, 256))
        # After equalization, bins should be more evenly distributed
        # Allow generous tolerance since input may not be perfectly equalized
        # Check that no single bin has more than 50% of pixels
        assert counts.max() < raster_float.pixel_count * 0.5

    def test_preserves_shape(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        assert result.shape == raster_float.shape

    def test_preserves_affine(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        assert result.affine == raster_float.affine

    def test_nan_nodata_float64(self):
        """Regression test for bug #7: NaN nodata corrupts min/max normalization.

        When nodata=NaN, the old code used `data[data != nodata]` which includes
        NaN values (IEEE 754: NaN != NaN is True). NaN then corrupts np.min/np.max,
        producing NaN for dmin/dmax and garbage output.
        """
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, np.nan, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=np.nan, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        result_data = result.to_numpy()
        assert result.dtype == np.uint8
        assert result.shape == (3, 3)
        # All valid pixels must be finite (no NaN leaking into uint8 output)
        # NaN nodata pixel at [1, 1] maps to nodata_u8=0 in the equalized output
        valid_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        for r, c in valid_positions:
            assert 0 <= result_data[r, c] <= 255, f"pixel [{r},{c}] out of range"
        # Output should span a wide range (equalization spreads values)
        valid_vals = np.array([result_data[r, c] for r, c in valid_positions])
        assert valid_vals.max() > valid_vals.min(), "equalization should spread values"

    def test_nan_nodata_float32(self):
        """NaN nodata with float32 dtype."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [np.nan, 8.0, 9.0]],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=np.nan, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        result_data = result.to_numpy()
        assert result.dtype == np.uint8
        assert result.shape == (3, 3)
        # 7 valid pixels, 2 NaN nodata -- valid pixels should all be in [0, 255]
        nodata_mask = np.isnan(data)
        assert np.all(result_data[~nodata_mask] <= 255)
        assert np.all(result_data[~nodata_mask] >= 0)

    def test_nan_nodata_all_nan(self):
        """All-NaN raster with NaN nodata should produce all-zero output."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.full((3, 3), np.nan, dtype=np.float64)
        raster = from_numpy(data, nodata=np.nan, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        result_data = result.to_numpy()
        assert result.dtype == np.uint8
        # All pixels are nodata -> output should be all zeros
        np.testing.assert_array_equal(result_data, 0)
        # NaN nodata in float input maps to nodata=0 in uint8 output
        assert result.nodata == 0

    def test_nan_nodata_preserves_metadata(self):
        """Metadata (affine, crs, shape) preserved through NaN nodata equalization."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, np.nan], [30.0, 40.0]],
            dtype=np.float64,
        )
        affine = (2.0, 0.0, 100.0, 0.0, -2.0, 200.0)
        raster = from_numpy(data, nodata=np.nan, affine=affine, crs="EPSG:4326")
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.shape == raster.shape
        assert result.affine == affine
        assert result.crs == raster.crs

    def test_nan_nodata_diagnostic_event(self):
        """Equalization with NaN nodata should emit a diagnostic event."""
        from vibespatial.raster.buffers import RasterDiagnosticKind
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
        raster = from_numpy(data, nodata=np.nan, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "raster_histogram_equalize" in runtime_events[0].detail
        assert "cpu" in runtime_events[0].detail


# ---------------------------------------------------------------------------
# CPU tests — nodata preservation for non-uint8 equalization (bug #8)
# ---------------------------------------------------------------------------


class TestEqualizeNodataPreservationCPU:
    """Regression tests for bug #8: non-uint8 rasters with nodata lose nodata info.

    Before the fix, equalizing a float/int raster with nodata would map nodata
    pixels to 0 but declare nodata=None in the output.  Downstream consumers had
    no way to identify which pixels were originally nodata.
    """

    def test_float64_nodata_sentinel_declared(self):
        """Float64 input with nodata=-9999 must declare nodata=0 in uint8 output."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, -9999.0, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.dtype == np.uint8
        assert result.nodata == 0, "non-uint8 input with nodata must declare nodata=0 in output"
        # The nodata pixel at [1, 1] must hold the sentinel value
        assert result.to_numpy()[1, 1] == 0

    def test_float32_nodata_sentinel_declared(self):
        """Float32 input with nodata=-1.0 must declare nodata=0 in uint8 output."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, -1.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=-1.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.dtype == np.uint8
        assert result.nodata == 0
        assert result.to_numpy()[1, 1] == 0

    def test_int16_nodata_sentinel_declared(self):
        """Int16 input with nodata=-32768 must declare nodata=0 in uint8 output."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[100, 200, 300], [400, -32768, 600], [700, 800, 900]],
            dtype=np.int16,
        )
        raster = from_numpy(data, nodata=-32768, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.dtype == np.uint8
        assert result.nodata == 0
        assert result.to_numpy()[1, 1] == 0

    def test_nan_nodata_sentinel_declared(self):
        """Float64 input with nodata=NaN must declare nodata=0 in uint8 output."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, np.nan, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=np.nan, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.dtype == np.uint8
        assert result.nodata == 0
        assert result.to_numpy()[1, 1] == 0

    def test_nodata_mask_roundtrips_through_equalization(self):
        """The output nodata_mask must match the input nodata positions."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, -9999.0, 30.0], [-9999.0, 50.0, -9999.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        input_nodata_positions = raster.nodata_mask
        output_nodata_positions = result.nodata_mask
        np.testing.assert_array_equal(
            input_nodata_positions,
            output_nodata_positions,
            err_msg="nodata pixel positions must be preserved through equalization",
        )

    def test_uint8_nodata_still_preserved(self):
        """Uint8 input: original nodata sentinel must be preserved (not changed to 0)."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array([[10, 20, 30], [40, 255, 60], [70, 80, 90]], dtype=np.uint8)
        raster = from_numpy(data, nodata=255, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.dtype == np.uint8
        assert result.nodata == 255
        assert result.to_numpy()[1, 1] == 255

    def test_no_nodata_stays_none(self):
        """Input without nodata: output nodata must remain None."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.nodata is None

    def test_all_nodata_float_declares_sentinel(self):
        """All-nodata float raster must still declare nodata=0 in output."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.full((3, 3), -9999.0, dtype=np.float64)
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        assert result.dtype == np.uint8
        assert result.nodata == 0
        np.testing.assert_array_equal(result.to_numpy(), 0)

    def test_valid_pixels_nonzero_with_nodata_sentinel_zero(self):
        """Valid pixels must have nonzero equalized values when nodata sentinel is 0.

        This verifies that the equalization spreads valid pixels across the
        [1, 255] range when 0 is reserved for nodata.
        """
        from vibespatial.raster.histogram import raster_histogram_equalize

        # Large enough range to produce spread-out equalization
        rng = np.random.default_rng(42)
        data = rng.normal(100.0, 20.0, size=(20, 20)).astype(np.float64)
        data[5, 5] = -9999.0
        data[10, 15] = -9999.0
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 20.0))
        result = raster_histogram_equalize(raster, use_gpu=False)

        result_data = result.to_numpy()
        assert result.nodata == 0
        # Nodata pixels must be 0
        assert result_data[5, 5] == 0
        assert result_data[10, 15] == 0
        # Valid pixels should have non-trivial equalized values (not all 0)
        nodata_positions = np.array(raster.nodata_mask)
        valid_values = result_data[~nodata_positions]
        assert valid_values.max() > 0, "valid pixels should have nonzero equalized values"


# ---------------------------------------------------------------------------
# CPU tests — NaN nodata for histogram and percentile
# ---------------------------------------------------------------------------


class TestNaNNodataHistogramCPU:
    """Verify that histogram and percentile correctly exclude NaN nodata pixels."""

    def test_histogram_nan_nodata_excluded(self):
        from vibespatial.raster.histogram import raster_histogram

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, np.nan, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=np.nan, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        counts, _edges = raster_histogram(raster, bins=10, use_gpu=False)
        # 9 pixels total, 1 NaN nodata -> 8 valid
        assert counts.sum() == 8

    def test_percentile_nan_nodata_excluded(self):
        from vibespatial.raster.histogram import raster_percentile

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, np.nan, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=np.nan, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_percentile(raster, 50.0, bins=100, use_gpu=False)
        assert not np.isnan(result[0]), "NaN nodata must not corrupt percentile result"
        # Median of [10, 20, 30, 40, 60, 70, 80, 90] is ~50
        assert 30.0 <= result[0] <= 70.0


# ---------------------------------------------------------------------------
# CPU tests — raster_percentile
# ---------------------------------------------------------------------------


class TestRasterPercentileCPU:
    def test_median(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, 50.0, use_gpu=False)
        assert result.shape == (1,)
        # The histogram-based median should be close to the true median
        true_median = float(np.median(raster_float.to_numpy()))
        assert abs(result[0] - true_median) < 5.0  # generous tolerance

    def test_multiple_percentiles(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, [25.0, 50.0, 75.0], use_gpu=False)
        assert result.shape == (3,)
        # Percentiles must be ordered
        assert result[0] <= result[1] <= result[2]

    def test_0_and_100_percentiles(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, [0.0, 100.0], bins=1000, use_gpu=False)
        data = raster_float.to_numpy()
        # 0th percentile should be close to min
        assert abs(result[0] - float(np.min(data))) < 1.0
        # 100th percentile should be close to max (within bin width)
        assert abs(result[1] - float(np.max(data))) < 1.0

    def test_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_with_nodata, 50.0, bins=100, use_gpu=False)
        assert not np.isnan(result[0])
        # Median of [10, 20, 30, 40, 60, 70, 80, 90] is ~50
        assert 30.0 <= result[0] <= 70.0

    def test_invalid_percentile_raises(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        with pytest.raises(ValueError, match="percentile must be in"):
            raster_percentile(raster_float, 101.0, use_gpu=False)

        with pytest.raises(ValueError, match="percentile must be in"):
            raster_percentile(raster_float, -1.0, use_gpu=False)

    def test_scalar_percentile(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, 50.0, use_gpu=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# GPU tests — raster_histogram
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterHistogramGPU:
    def test_gpu_matches_cpu(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        cpu_counts, cpu_edges = raster_histogram(raster_float, bins=50, use_gpu=False)
        gpu_counts, gpu_edges = raster_histogram(raster_float, bins=50, use_gpu=True)

        np.testing.assert_array_equal(cpu_counts, gpu_counts)
        np.testing.assert_array_almost_equal(cpu_edges, gpu_edges)

    def test_gpu_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_histogram

        counts, _edges = raster_histogram(raster_with_nodata, bins=10, use_gpu=True)
        assert counts.sum() == 8

    def test_gpu_custom_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(
            raster_float, bins=10, range_min=50.0, range_max=150.0, use_gpu=True
        )
        assert edges[0] == pytest.approx(50.0)
        assert edges[-1] == pytest.approx(150.0)

    def test_gpu_total_count(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, _edges = raster_histogram(raster_float, bins=50, use_gpu=True)
        assert counts.sum() == raster_float.pixel_count


# ---------------------------------------------------------------------------
# GPU tests — raster_cumulative_distribution
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterCDFGPU:
    def test_gpu_monotonic(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=True)
        assert np.all(np.diff(cdf) >= 0)

    def test_gpu_final_value(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=True)
        assert cdf[-1] == raster_float.pixel_count


# ---------------------------------------------------------------------------
# GPU tests — raster_histogram_equalize
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterHistogramEqualizeGPU:
    def test_gpu_output_dtype(self, raster_uniform):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_uniform, use_gpu=True)
        assert result.dtype == np.uint8

    def test_gpu_output_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=True)
        data = result.to_numpy()
        assert data.min() >= 0
        assert data.max() <= 255

    def test_gpu_preserves_shape(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=True)
        assert result.shape == raster_float.shape

    def test_gpu_roughly_uniform(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=True)
        data = result.to_numpy().ravel()
        counts, _ = np.histogram(data, bins=16, range=(0, 256))
        # No single bin should dominate
        assert counts.max() < raster_float.pixel_count * 0.5

    def test_gpu_nodata_preserved_uint8(self, raster_uint8_nodata):
        """Regression test for kernel param type mismatch (bug #4).

        The histogram remap kernel declares nodata_val as const int and
        casts to unsigned char internally.  Before the fix, the host
        passed KERNEL_PARAM_I32 (4 bytes) for a kernel parameter declared
        as unsigned char (1 byte), corrupting the nodata value written
        to output pixels.  This test verifies nodata pixels get the
        correct sentinel value after equalization.
        """
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_uint8_nodata, use_gpu=True)
        data = result.to_numpy()
        # The input nodata sentinel is 0.  The nodata pixel is at [1,1].
        # After equalization, nodata pixels must still hold the sentinel.
        assert data[1, 1] == 0, f"nodata pixel should be 0, got {data[1, 1]}"
        # Valid pixels should have been remapped (not all zeros)
        valid_mask = np.array([[True, True, True], [True, False, True], [True, True, True]])
        assert data[valid_mask].sum() > 0, "valid pixels should have nonzero equalized values"

    def test_gpu_equalize_matches_cpu_with_nodata(self, raster_uint8_nodata):
        """GPU and CPU equalization should produce identical output for uint8 with nodata."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        cpu_result = raster_histogram_equalize(raster_uint8_nodata, use_gpu=False)
        gpu_result = raster_histogram_equalize(raster_uint8_nodata, use_gpu=True)
        np.testing.assert_array_equal(
            cpu_result.to_numpy(),
            gpu_result.to_numpy(),
            err_msg="GPU and CPU histogram equalize should match for uint8 with nodata",
        )


# ---------------------------------------------------------------------------
# GPU tests — nodata preservation for non-uint8 equalization (bug #8)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestEqualizeNodataPreservationGPU:
    """GPU regression tests for bug #8: non-uint8 nodata information loss."""

    def test_float64_nodata_sentinel_declared(self):
        """GPU: float64 with nodata must declare nodata=0 in uint8 output."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, -9999.0, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=True)

        assert result.dtype == np.uint8
        assert result.nodata == 0
        assert result.to_numpy()[1, 1] == 0

    def test_float64_nodata_gpu_matches_cpu(self):
        """GPU and CPU must agree on nodata metadata for float64 input."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, 20.0, 30.0], [40.0, -9999.0, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        cpu_result = raster_histogram_equalize(raster, use_gpu=False)
        gpu_result = raster_histogram_equalize(raster, use_gpu=True)

        assert cpu_result.nodata == gpu_result.nodata == 0
        # Both must have nodata at the same positions
        np.testing.assert_array_equal(cpu_result.nodata_mask, gpu_result.nodata_mask)

    def test_nodata_mask_roundtrips_gpu(self):
        """GPU: nodata pixel positions must survive equalization."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        data = np.array(
            [[10.0, -9999.0, 30.0], [-9999.0, 50.0, -9999.0], [70.0, 80.0, 90.0]],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        result = raster_histogram_equalize(raster, use_gpu=True)

        np.testing.assert_array_equal(
            raster.nodata_mask,
            result.nodata_mask,
            err_msg="nodata positions must be preserved through GPU equalization",
        )


# ---------------------------------------------------------------------------
# GPU tests — raster_percentile
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterPercentileGPU:
    def test_gpu_matches_cpu(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        cpu_result = raster_percentile(raster_float, [25.0, 50.0, 75.0], bins=256, use_gpu=False)
        gpu_result = raster_percentile(raster_float, [25.0, 50.0, 75.0], bins=256, use_gpu=True)
        # Both GPU and CPU percentile paths use the same algorithm:
        # histogram -> cumulative sum -> searchsorted on host.  The GPU
        # uses CCCL histogram_even + exclusive_sum, the CPU uses
        # np.histogram + np.cumsum.  Histogram counts match exactly
        # (verified by TestRasterHistogramGPU.test_gpu_matches_cpu), and
        # edges are both computed via np.linspace from the same [lo, hi].
        # Therefore percentile values should match within floating-point
        # rounding of the CDF cumsum.  Use decimal=5 (atol ~1e-5) to
        # account for float64 accumulation order differences.
        np.testing.assert_array_almost_equal(cpu_result, gpu_result, decimal=5)

    def test_gpu_ordered_percentiles(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, [10.0, 25.0, 50.0, 75.0, 90.0], use_gpu=True)
        assert np.all(np.diff(result) >= 0)

    def test_gpu_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_with_nodata, 50.0, bins=100, use_gpu=True)
        assert not np.isnan(result[0])


# ---------------------------------------------------------------------------
# Multiband validation tests
# ---------------------------------------------------------------------------


class TestMultibandPerBandDispatch:
    """Verify that histogram functions dispatch per-band for multiband rasters."""

    def test_histogram_multiband_per_band(self):
        """3-band raster: returns list of (counts, edges), one per band."""
        from vibespatial.raster.histogram import raster_histogram

        rng = np.random.default_rng(42)
        # Create 3 bands with distinct distributions so histograms differ
        band0 = rng.integers(0, 64, size=(16, 16), dtype=np.uint8)
        band1 = rng.integers(64, 128, size=(16, 16), dtype=np.uint8)
        band2 = rng.integers(192, 256, size=(16, 16), dtype=np.uint8)
        data = np.stack([band0, band1, band2], axis=0)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 16.0))
        assert raster.band_count == 3

        results = raster_histogram(raster, bins=256, use_gpu=False)
        assert isinstance(results, list)
        assert len(results) == 3

        for i, (counts, edges) in enumerate(results):
            assert counts.shape == (256,), f"band {i}: counts shape mismatch"
            assert edges.shape == (257,), f"band {i}: edges shape mismatch"
            assert counts.sum() == 16 * 16, f"band {i}: total count mismatch"

        # Verify distributions are distinct: per-band ranges should differ.
        # Each band's edges are auto-computed from its own data range.
        _, edges0 = results[0]
        _, edges2 = results[2]
        # Band 0 values are in [0, 64), band 2 values are in [192, 256)
        assert edges0[0] < 64, "band 0 range should start below 64"
        assert edges2[0] >= 192, "band 2 range should start at 192 or above"

    def test_histogram_multiband_matches_individual_bands(self):
        """Multiband histogram results must match single-band histograms."""
        from vibespatial.raster.histogram import raster_histogram

        rng = np.random.default_rng(99)
        data = rng.integers(0, 256, size=(3, 16, 16), dtype=np.uint8)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 16.0))

        # Multiband result
        multi_results = raster_histogram(raster, bins=128, use_gpu=False)

        # Compare against individual single-band rasters
        for band_idx in range(3):
            single = from_numpy(data[band_idx], affine=(1.0, 0.0, 0.0, 0.0, -1.0, 16.0))
            single_counts, single_edges = raster_histogram(single, bins=128, use_gpu=False)
            multi_counts, multi_edges = multi_results[band_idx]
            np.testing.assert_array_equal(single_counts, multi_counts)
            np.testing.assert_array_equal(single_edges, multi_edges)

    def test_equalize_multiband(self):
        """3-band equalization: output shape (3, H, W), dtype uint8."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(3, 16, 16), dtype=np.uint8)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 16.0))
        assert raster.band_count == 3

        result = raster_histogram_equalize(raster, use_gpu=False)
        assert result.dtype == np.uint8
        assert result.shape == (3, 16, 16)
        assert result.band_count == 3
        result_data = result.to_numpy()
        assert result_data.min() >= 0
        assert result_data.max() <= 255
        # Metadata preserved
        assert result.affine == raster.affine

    def test_equalize_multiband_preserves_nodata(self):
        """Multiband equalization preserves nodata metadata from source."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        rng = np.random.default_rng(42)
        data = rng.normal(100.0, 20.0, size=(3, 8, 8)).astype(np.float64)
        # Place nodata in different positions per band
        data[0, 2, 3] = -9999.0
        data[1, 5, 1] = -9999.0
        data[2, 0, 7] = -9999.0
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 8.0))

        result = raster_histogram_equalize(raster, use_gpu=False)
        assert result.dtype == np.uint8
        assert result.shape == (3, 8, 8)
        # from_band_stack propagates the source's nodata metadata
        assert result.nodata == -9999.0

    def test_percentile_multiband(self):
        """3-band percentiles: returns list of per-band percentile arrays."""
        from vibespatial.raster.histogram import raster_percentile

        rng = np.random.default_rng(42)
        # Three bands with clearly different ranges
        band0 = rng.uniform(0.0, 10.0, size=(20, 20)).astype(np.float64)
        band1 = rng.uniform(50.0, 60.0, size=(20, 20)).astype(np.float64)
        band2 = rng.uniform(100.0, 110.0, size=(20, 20)).astype(np.float64)
        data = np.stack([band0, band1, band2], axis=0)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 20.0))
        assert raster.band_count == 3

        results = raster_percentile(raster, [25, 50, 75], use_gpu=False)
        assert isinstance(results, list)
        assert len(results) == 3

        for i, pcts in enumerate(results):
            assert isinstance(pcts, np.ndarray)
            assert pcts.shape == (3,), f"band {i}: shape mismatch"
            # Percentiles must be ordered
            assert pcts[0] <= pcts[1] <= pcts[2], f"band {i}: not ordered"

        # Band medians should be in their respective ranges
        assert 0.0 <= results[0][1] <= 10.0, "band 0 median out of range"
        assert 50.0 <= results[1][1] <= 60.0, "band 1 median out of range"
        assert 100.0 <= results[2][1] <= 110.0, "band 2 median out of range"

    def test_percentile_multiband_matches_individual_bands(self):
        """Multiband percentile results must match single-band percentiles."""
        from vibespatial.raster.histogram import raster_percentile

        rng = np.random.default_rng(99)
        data = rng.normal(50.0, 15.0, size=(3, 20, 20)).astype(np.float64)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 20.0))

        multi_results = raster_percentile(raster, [10, 50, 90], bins=256, use_gpu=False)

        for band_idx in range(3):
            single = from_numpy(data[band_idx], affine=(1.0, 0.0, 0.0, 0.0, -1.0, 20.0))
            single_pcts = raster_percentile(single, [10, 50, 90], bins=256, use_gpu=False)
            np.testing.assert_array_equal(single_pcts, multi_results[band_idx])

    def test_histogram_multiband_diagnostic_event(self):
        """Multiband histogram should emit a diagnostic event with 'multiband'."""
        from vibespatial.raster.buffers import RasterDiagnosticKind
        from vibespatial.raster.histogram import raster_histogram

        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 8.0))

        raster_histogram(raster, use_gpu=False)
        runtime_events = [e for e in raster.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert any("multiband" in e.detail for e in runtime_events)

    def test_single_band_return_types_unchanged(self):
        """Single-band operations must return their original types, not lists."""
        from vibespatial.raster.histogram import (
            raster_histogram,
            raster_histogram_equalize,
            raster_percentile,
        )

        data = np.arange(256, dtype=np.uint8).reshape(16, 16)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 16.0))

        # raster_histogram returns a tuple, not a list
        hist_result = raster_histogram(raster, use_gpu=False)
        assert isinstance(hist_result, tuple)
        assert len(hist_result) == 2

        # raster_histogram_equalize returns OwnedRasterArray
        eq_result = raster_histogram_equalize(raster, use_gpu=False)
        assert eq_result.dtype == np.uint8
        assert eq_result.band_count == 1

        # raster_percentile returns np.ndarray
        pct_result = raster_percentile(raster, 50.0, use_gpu=False)
        assert isinstance(pct_result, np.ndarray)
        assert pct_result.shape == (1,)


# ---------------------------------------------------------------------------
# Auto-dispatch tests
# ---------------------------------------------------------------------------


class TestAutoDispatch:
    def test_histogram_auto_dispatch(self, raster_float):
        """Auto-dispatch should work regardless of GPU availability."""
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(raster_float)
        assert counts.shape[0] == 256
        assert edges.shape[0] == 257

    def test_cdf_auto_dispatch(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, edges = raster_cumulative_distribution(raster_float)
        assert cdf.shape[0] == 256

    def test_equalize_auto_dispatch(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float)
        assert result.dtype == np.uint8

    def test_percentile_auto_dispatch(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, 50.0)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_histogram_importable(self):
        from vibespatial.raster import raster_histogram

        assert callable(raster_histogram)

    def test_cdf_importable(self):
        from vibespatial.raster import raster_cumulative_distribution

        assert callable(raster_cumulative_distribution)

    def test_equalize_importable(self):
        from vibespatial.raster import raster_histogram_equalize

        assert callable(raster_histogram_equalize)

    def test_percentile_importable(self):
        from vibespatial.raster import raster_percentile

        assert callable(raster_percentile)
