"""Tests for spectral index convenience functions.

Exercises raster_ndvi, raster_band_ratio, and raster_band_math — thin
wrappers around raster_expression() that provide ergonomic access to
common remote-sensing indices.

All tests use ``use_gpu=False`` (CPU path) so they run without a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.algebra import (
    raster_band_math,
    raster_band_ratio,
    raster_ndvi,
)
from vibespatial.raster.buffers import (
    RasterDiagnosticKind,
    from_numpy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AFFINE = (1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
_CRS = "EPSG:32610"


def _make_4band(
    height: int = 20,
    width: int = 20,
    *,
    nodata: float | None = None,
) -> tuple:
    """Create a 4-band float32 raster with known per-band values.

    Band layout (0-indexed):
      0 (blue):  uniform 0.1
      1 (green): uniform 0.2
      2 (red):   uniform 0.3
      3 (nir):   uniform 0.5

    Returns (raster, raw_data_array).
    """
    data = np.empty((4, height, width), dtype=np.float32)
    data[0] = 0.1
    data[1] = 0.2
    data[2] = 0.3
    data[3] = 0.5
    raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)
    return raster, data


def _make_4band_gradient(
    height: int = 20,
    width: int = 20,
    *,
    nodata: float | None = None,
):
    """Create a 4-band raster with per-pixel variation for non-trivial tests."""
    rng = np.random.default_rng(42)
    data = np.empty((4, height, width), dtype=np.float32)
    for b in range(4):
        data[b] = rng.uniform(0.05, 0.95, (height, width)).astype(np.float32)
    raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)
    return raster, data


# ---------------------------------------------------------------------------
# NDVI tests
# ---------------------------------------------------------------------------


class TestNDVI:
    """Tests for raster_ndvi()."""

    def test_ndvi_basic(self):
        """Known constant NIR=0.5 RED=0.3 -> NDVI = (0.5-0.3)/(0.5+0.3) = 0.25."""
        raster, data = _make_4band()
        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()

        expected_ndvi = (0.5 - 0.3) / (0.5 + 0.3)  # 0.25
        np.testing.assert_allclose(out, expected_ndvi, atol=1e-6)

        # NDVI is in [-1, 1]
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)

        # Result should be 2D single-band
        assert out.ndim == 2
        assert out.shape == (20, 20)

    def test_ndvi_metadata_preservation(self):
        """Affine, CRS, and float32 dtype are preserved."""
        raster, _ = _make_4band()
        result = raster_ndvi(raster, use_gpu=False)

        assert result.affine == _AFFINE
        assert result.crs == _CRS
        assert result.dtype == np.float32

    def test_ndvi_nodata(self):
        """Nodata in NIR or RED band -> nodata in output."""
        nodata = -9999.0
        raster, data = _make_4band(nodata=nodata)

        # Inject nodata into RED (band 2) at specific positions
        data[2, 0, 0:3] = nodata
        # Inject nodata into NIR (band 3) at other positions
        data[3, 5, 5] = nodata
        raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)

        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()

        # Positions where RED was nodata should be nodata in output
        assert out[0, 0] == nodata
        assert out[0, 1] == nodata
        assert out[0, 2] == nodata

        # Position where NIR was nodata should be nodata in output
        assert out[5, 5] == nodata

        # Other positions should have valid NDVI
        assert out[10, 10] != nodata

    def test_ndvi_custom_bands(self):
        """Use non-default band indices for NIR and RED."""
        raster, data = _make_4band()

        # Swap: use band 1 as "NIR", band 2 as "RED" (1-indexed)
        result = raster_ndvi(raster, nir_band=1, red_band=2, use_gpu=False)
        out = result.to_numpy()

        # band 0 = 0.1 (nir), band 1 = 0.2 (red)
        expected = (0.1 - 0.2) / (0.1 + 0.2)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_ndvi_gradient_range(self):
        """NDVI from random gradient data is in [-1, 1]."""
        raster, _ = _make_4band_gradient()
        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()

        assert np.all(out >= -1.0 - 1e-6)
        assert np.all(out <= 1.0 + 1e-6)

    def test_ndvi_diagnostic_event(self):
        """Diagnostic event is recorded."""
        raster, _ = _make_4band()
        result = raster_ndvi(raster, use_gpu=False)

        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "band_expression" in runtime_events[0].detail
        assert "CPU" in runtime_events[0].detail

    def test_ndvi_invalid_band_zero(self):
        """Band index 0 (< 1) raises ValueError."""
        raster, _ = _make_4band()
        with pytest.raises(ValueError, match="nir_band must be >= 1"):
            raster_ndvi(raster, nir_band=0, use_gpu=False)

    def test_ndvi_invalid_band_too_high(self):
        """Band index exceeding band count raises IndexError."""
        raster, _ = _make_4band()
        with pytest.raises(IndexError, match="exceeds raster band count"):
            raster_ndvi(raster, nir_band=5, use_gpu=False)

    def test_ndvi_red_band_out_of_range(self):
        """Red band out of range raises IndexError."""
        raster, _ = _make_4band()
        with pytest.raises(IndexError, match="red_band"):
            raster_ndvi(raster, red_band=10, use_gpu=False)


# ---------------------------------------------------------------------------
# Band ratio tests
# ---------------------------------------------------------------------------


class TestBandRatio:
    """Tests for raster_band_ratio()."""

    def test_band_ratio(self):
        """Basic band ratio: band_a / band_b with known values."""
        raster, data = _make_4band()
        # band 4 (nir=0.5) / band 3 (red=0.3)
        result = raster_band_ratio(raster, band_a=4, band_b=3, use_gpu=False)
        out = result.to_numpy()

        expected = 0.5 / 0.3
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_band_ratio_metadata(self):
        """Metadata is preserved through band ratio."""
        raster, _ = _make_4band()
        result = raster_band_ratio(raster, band_a=4, band_b=3, use_gpu=False)

        assert result.affine == _AFFINE
        assert result.crs == _CRS

    def test_band_ratio_div_zero(self):
        """Division by zero produces nodata."""
        nodata = -9999.0
        height, width = 10, 10
        data = np.ones((4, height, width), dtype=np.float32)
        # Make band 2 (0-indexed 1) all zeros at specific locations
        data[1, 0, :] = 0.0
        raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)

        result = raster_band_ratio(raster, band_a=1, band_b=2, use_gpu=False)
        out = result.to_numpy()

        # Row 0 had denominator = 0 -> should be nodata
        assert np.all(out[0, :] == nodata)
        # Other rows had denominator = 1.0 -> should be 1.0
        np.testing.assert_allclose(out[1:, :], 1.0, atol=1e-6)

    def test_band_ratio_nodata_propagation(self):
        """Nodata in either band -> nodata in output."""
        nodata = -9999.0
        raster, data = _make_4band(nodata=nodata)
        data[0, 3, 3] = nodata  # nodata in numerator band
        data[1, 7, 7] = nodata  # nodata in denominator band
        raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)

        result = raster_band_ratio(raster, band_a=1, band_b=2, use_gpu=False)
        out = result.to_numpy()

        assert out[3, 3] == nodata
        assert out[7, 7] == nodata

    def test_band_ratio_invalid_band(self):
        """Invalid band indices raise appropriate errors."""
        raster, _ = _make_4band()
        with pytest.raises(ValueError, match="band_a must be >= 1"):
            raster_band_ratio(raster, band_a=0, band_b=1, use_gpu=False)
        with pytest.raises(ValueError, match="band_b must be >= 1"):
            raster_band_ratio(raster, band_a=1, band_b=0, use_gpu=False)
        with pytest.raises(IndexError, match="exceeds raster band count"):
            raster_band_ratio(raster, band_a=5, band_b=1, use_gpu=False)

    def test_band_ratio_diagnostic(self):
        """Diagnostic event is recorded for band ratio."""
        raster, _ = _make_4band()
        result = raster_band_ratio(raster, band_a=4, band_b=3, use_gpu=False)

        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1


# ---------------------------------------------------------------------------
# Band math tests
# ---------------------------------------------------------------------------


class TestBandMath:
    """Tests for raster_band_math()."""

    def test_band_math_simple(self):
        """Simple band subtraction."""
        raster, data = _make_4band()
        result = raster_band_math(raster, "b[3] - b[2]", use_gpu=False)
        out = result.to_numpy()

        expected = 0.5 - 0.3  # nir - red
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_band_math_complex(self):
        """Complex multi-band expression: (b[3] - b[2]) / (b[3] + b[2] + 0.5 * b[1])."""
        raster, data = _make_4band_gradient()
        result = raster_band_math(
            raster,
            "(b[3] - b[2]) / (b[3] + b[2] + 0.5 * b[1])",
            use_gpu=False,
        )
        out = result.to_numpy()

        # Compute expected manually
        nir = data[3].astype(np.float32)
        red = data[2].astype(np.float32)
        green = data[1].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (nir - red) / (nir + red + 0.5 * green)

        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_band_math_with_functions(self):
        """Expression using built-in functions like sqrt and abs."""
        raster, data = _make_4band()
        result = raster_band_math(raster, "sqrt(abs(b[3] - b[2]))", use_gpu=False)
        out = result.to_numpy()

        expected = np.sqrt(np.abs(0.5 - 0.3))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_band_math_nodata_propagation(self):
        """Nodata propagates through band math expressions."""
        nodata = -9999.0
        raster, data = _make_4band(nodata=nodata)
        data[2, 2, 2] = nodata
        raster = from_numpy(data, nodata=nodata, affine=_AFFINE, crs=_CRS)

        result = raster_band_math(raster, "b[3] + b[2]", use_gpu=False)
        out = result.to_numpy()

        assert out[2, 2] == nodata
        # Non-nodata pixel is valid
        np.testing.assert_allclose(out[10, 10], 0.5 + 0.3, atol=1e-6)

    def test_band_math_invalid_expression_empty(self):
        """Empty expression raises ValueError."""
        raster, _ = _make_4band()
        with pytest.raises(ValueError, match="expression must not be empty"):
            raster_band_math(raster, "", use_gpu=False)
        with pytest.raises(ValueError, match="expression must not be empty"):
            raster_band_math(raster, "   ", use_gpu=False)

    def test_band_math_invalid_band_index(self):
        """Band index out of range raises IndexError."""
        raster, _ = _make_4band()  # 4 bands, indices 0-3
        with pytest.raises(IndexError, match="out of range"):
            raster_band_math(raster, "b[4]", use_gpu=False)

    def test_band_math_metadata(self):
        """Metadata is preserved through band math."""
        raster, _ = _make_4band()
        result = raster_band_math(raster, "b[0] + b[1]", use_gpu=False)

        assert result.affine == _AFFINE
        assert result.crs == _CRS
        assert result.dtype == np.float32

    def test_band_math_diagnostic(self):
        """Diagnostic event is recorded."""
        raster, _ = _make_4band()
        result = raster_band_math(raster, "b[0] + b[1]", use_gpu=False)

        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1

    def test_band_math_single_band_reference(self):
        """Expression referencing a single band works."""
        raster, data = _make_4band()
        result = raster_band_math(raster, "b[0] * 2.0", use_gpu=False)
        out = result.to_numpy()

        expected = 0.1 * 2.0
        np.testing.assert_allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSpectralEdgeCases:
    """Edge cases across all spectral functions."""

    def test_ndvi_equal_bands(self):
        """When NIR == RED, NDVI denominator is nonzero but result is 0."""
        data = np.full((4, 10, 10), 0.5, dtype=np.float32)
        raster = from_numpy(data, affine=_AFFINE)
        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()

        # (0.5 - 0.5) / (0.5 + 0.5) = 0.0
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_ndvi_both_zero(self):
        """When NIR and RED are both zero, result should be nodata (0/0)."""
        nodata = -9999.0
        data = np.zeros((4, 10, 10), dtype=np.float32)
        raster = from_numpy(data, nodata=nodata, affine=_AFFINE)
        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()

        # 0/0 -> inf/nan -> nodata
        assert np.all(out == nodata)

    def test_small_raster(self):
        """Single-pixel raster works correctly."""
        data = np.array([[[0.1]], [[0.2]], [[0.3]], [[0.5]]], dtype=np.float32)
        raster = from_numpy(data, affine=_AFFINE)
        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()

        expected = (0.5 - 0.3) / (0.5 + 0.3)
        np.testing.assert_allclose(out, expected, atol=1e-6)
        assert out.shape == (1, 1)

    def test_nonsquare_raster(self):
        """Non-square raster produces correct shape."""
        raster, _ = _make_4band(height=5, width=30)
        result = raster_ndvi(raster, use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (5, 30)
