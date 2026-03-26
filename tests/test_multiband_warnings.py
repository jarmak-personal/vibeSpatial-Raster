"""Tests for multiband squeeze warnings (vibeSpatial-2to.1.4).

Verify that UserWarning is emitted when multiband rasters are silently
squeezed to band 0 in algebra, label, and hydrology operations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibespatial.raster.buffers import from_numpy

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AFFINE = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)


@pytest.fixture
def multiband_raster():
    """3-band float64 raster (3, 4, 4)."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 4, 4), dtype=np.float64)
    return from_numpy(data, affine=_AFFINE)


@pytest.fixture
def multiband_raster_nodata():
    """3-band float64 raster with nodata."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 4, 4), dtype=np.float64)
    data[0, 0, 0] = -9999.0
    return from_numpy(data, nodata=-9999.0, affine=_AFFINE)


@pytest.fixture
def multiband_binary_raster():
    """3-band uint8 binary raster (3, 4, 4) for label/morphology ops."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2, size=(3, 4, 4), dtype=np.uint8)
    return from_numpy(data, affine=_AFFINE)


@pytest.fixture
def multiband_dem():
    """3-band float64 DEM raster for terrain ops."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 8, 8), dtype=np.float64) * 100
    return from_numpy(data, affine=_AFFINE)


# ---------------------------------------------------------------------------
# Algebra: raster_expression
# ---------------------------------------------------------------------------


@requires_gpu
def test_expression_multiband_warns_gpu(multiband_raster):
    """raster_expression GPU path warns on multiband input."""
    from vibespatial.raster.algebra import raster_expression

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_expression("a + 1.0", a=multiband_raster, use_gpu=True)


def test_expression_multiband_warns_cpu(multiband_raster):
    """raster_expression CPU path warns on multiband input."""
    from vibespatial.raster.algebra import raster_expression

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_expression("a + 1.0", a=multiband_raster, use_gpu=False)


# ---------------------------------------------------------------------------
# Algebra: raster_convolve / raster_gaussian_filter
# ---------------------------------------------------------------------------


@requires_gpu
def test_convolve_multiband_warns(multiband_raster):
    """raster_convolve warns on multiband input."""
    from vibespatial.raster.algebra import raster_convolve

    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_convolve(multiband_raster, kernel)


@requires_gpu
def test_gaussian_filter_multiband_warns(multiband_raster):
    """raster_gaussian_filter warns on multiband input."""
    from vibespatial.raster.algebra import raster_gaussian_filter

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_gaussian_filter(multiband_raster, sigma=1.0)


# ---------------------------------------------------------------------------
# Algebra: slope / aspect
# ---------------------------------------------------------------------------


@requires_gpu
def test_slope_multiband_warns_gpu(multiband_dem):
    """raster_slope GPU path warns on multiband DEM."""
    from vibespatial.raster.algebra import raster_slope

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_slope(multiband_dem, use_gpu=True)


def test_slope_multiband_warns_cpu(multiband_dem):
    """raster_slope CPU path warns on multiband DEM."""
    from vibespatial.raster.algebra import raster_slope

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_slope(multiband_dem, use_gpu=False)


# ---------------------------------------------------------------------------
# Algebra: hillshade
# ---------------------------------------------------------------------------


@requires_gpu
def test_hillshade_multiband_warns_gpu(multiband_dem):
    """raster_hillshade GPU path warns on multiband DEM."""
    from vibespatial.raster.algebra import raster_hillshade

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_hillshade(multiband_dem, use_gpu=True)


def test_hillshade_multiband_warns_cpu(multiband_dem):
    """raster_hillshade CPU path warns on multiband DEM."""
    from vibespatial.raster.algebra import raster_hillshade

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_hillshade(multiband_dem, use_gpu=False)


# ---------------------------------------------------------------------------
# Algebra: TRI / TPI / curvature
# ---------------------------------------------------------------------------


@requires_gpu
def test_tri_multiband_warns_gpu(multiband_dem):
    """raster_tri GPU path warns on multiband DEM."""
    from vibespatial.raster.algebra import raster_tri

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_tri(multiband_dem, use_gpu=True)


def test_tri_multiband_warns_cpu(multiband_dem):
    """raster_tri CPU path warns on multiband DEM."""
    from vibespatial.raster.algebra import raster_tri

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_tri(multiband_dem, use_gpu=False)


# ---------------------------------------------------------------------------
# Algebra: focal statistics
# ---------------------------------------------------------------------------


@requires_gpu
def test_focal_min_multiband_warns_gpu(multiband_raster):
    """raster_focal_min GPU path warns on multiband input."""
    from vibespatial.raster.algebra import raster_focal_min

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_focal_min(multiband_raster, radius=1, use_gpu=True)


def test_focal_min_multiband_warns_cpu(multiband_raster):
    """raster_focal_min CPU path warns on multiband input."""
    from vibespatial.raster.algebra import raster_focal_min

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_focal_min(multiband_raster, radius=1, use_gpu=False)


# ---------------------------------------------------------------------------
# Label: morphology CPU
# ---------------------------------------------------------------------------


def test_morphology_cpu_multiband_warns(multiband_binary_raster):
    """CPU morphology path warns on multiband input."""
    from vibespatial.raster.label import raster_morphology

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_morphology(multiband_binary_raster, "erode", use_gpu=False)


# ---------------------------------------------------------------------------
# Verify warning message content
# ---------------------------------------------------------------------------


def test_warning_message_includes_band_count():
    """Warning message should include the actual number of bands."""
    rng = np.random.default_rng(42)
    data = rng.random((5, 4, 4), dtype=np.float64)
    five_band = from_numpy(data, affine=_AFFINE)

    from vibespatial.raster.algebra import raster_expression

    with pytest.warns(UserWarning, match=r"Multiband raster with 5 bands"):
        raster_expression("a + 1.0", a=five_band, use_gpu=False)


def test_singleband_no_warning():
    """Single-band rasters should not emit any multiband warning."""
    data = np.ones((4, 4), dtype=np.float64)
    single_band = from_numpy(data, affine=_AFFINE)

    from vibespatial.raster.algebra import raster_expression

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Should NOT raise -- no multiband warning expected
        raster_expression("a + 1.0", a=single_band, use_gpu=False)
