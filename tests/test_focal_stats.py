"""Tests for focal statistics: min, max, mean, std, range, variety."""

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
def raster_5x5():
    """5x5 raster with sequential values 1..25."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


@pytest.fixture
def raster_uniform():
    """5x5 raster with all values = 7."""
    data = np.full((5, 5), 7.0, dtype=np.float64)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


@pytest.fixture
def raster_with_nodata():
    """5x5 raster with a nodata hole at (1,1)."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    data[1, 1] = -9999.0
    return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


@pytest.fixture
def raster_categorical():
    """5x5 raster with integer-like categories for variety testing."""
    data = np.array(
        [
            [1, 1, 2, 2, 3],
            [1, 1, 2, 3, 3],
            [4, 4, 5, 5, 5],
            [4, 4, 5, 6, 6],
            [7, 7, 7, 6, 6],
        ],
        dtype=np.float64,
    )
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


# ---------------------------------------------------------------------------
# CPU tests for each statistic
# ---------------------------------------------------------------------------


class TestFocalMinCPU:
    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center pixel (2,2): neighborhood is 3x3 around value 13
        # min of [7,8,9,12,13,14,17,18,19] = 7
        assert out[2, 2] == 7.0
        # Corner (0,0): neighborhood is [1,2,6,7] -> min = 1
        assert out[0, 0] == 1.0

    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_uniform, radius=1, use_gpu=False)
        np.testing.assert_array_almost_equal(result.to_numpy(), 7.0)

    def test_nodata(self, raster_with_nodata):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_with_nodata, radius=1, use_gpu=False)
        out = result.to_numpy()
        # The nodata pixel itself should remain nodata
        assert out[1, 1] == -9999.0

    def test_preserves_affine(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=False)
        assert result.affine == raster_5x5.affine


class TestFocalMaxCPU:
    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_max

        result = raster_focal_max(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): max of [7,8,9,12,13,14,17,18,19] = 19
        assert out[2, 2] == 19.0
        # Corner (4,4): neighborhood is [19,20,24,25] -> max = 25
        assert out[4, 4] == 25.0


class TestFocalMeanCPU:
    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_mean

        result = raster_focal_mean(raster_uniform, radius=1, use_gpu=False)
        np.testing.assert_array_almost_equal(result.to_numpy(), 7.0)

    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_mean

        result = raster_focal_mean(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): mean of [7,8,9,12,13,14,17,18,19] = 13
        np.testing.assert_almost_equal(out[2, 2], 13.0)


class TestFocalStdCPU:
    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_uniform, radius=1, use_gpu=False)
        # Std of uniform values = 0
        np.testing.assert_array_almost_equal(result.to_numpy(), 0.0)

    def test_nonzero(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): population std([7,8,9,12,13,14,17,18,19]) = 3.944... (ddof=0)
        expected = np.std([7, 8, 9, 12, 13, 14, 17, 18, 19], ddof=0)
        np.testing.assert_almost_equal(out[2, 2], expected, decimal=5)


class TestFocalRangeCPU:
    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_range

        result = raster_focal_range(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): range of [7..19] = 12
        assert out[2, 2] == 12.0

    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_range

        result = raster_focal_range(raster_uniform, radius=1, use_gpu=False)
        np.testing.assert_array_almost_equal(result.to_numpy(), 0.0)


class TestFocalVarietyCPU:
    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_variety

        result = raster_focal_variety(raster_uniform, radius=1, use_gpu=False)
        # All values the same -> variety = 1
        np.testing.assert_array_almost_equal(result.to_numpy(), 1.0)

    def test_categorical(self, raster_categorical):
        from vibespatial.raster.algebra import raster_focal_variety

        result = raster_focal_variety(raster_categorical, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2) = value 5, neighborhood: [1,2,2,3,4,5,5,5,5] -> unique: {1,2,3,4,5} = 5
        assert out[2, 2] >= 3.0  # at least several unique values


class TestRadiusParsing:
    def test_int_radius(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=2, use_gpu=False)
        assert result.shape == raster_5x5.shape

    def test_tuple_radius(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=(1, 2), use_gpu=False)
        assert result.shape == raster_5x5.shape

    def test_bad_tuple(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        with pytest.raises(ValueError, match="2 elements"):
            raster_focal_min(raster_5x5, radius=(1, 2, 3), use_gpu=False)


class TestDiagnostics:
    def test_cpu_diagnostics(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=False)
        assert any("cpu_focal_min" in d.detail for d in result.diagnostics)


# ---------------------------------------------------------------------------
# GPU tests (compare GPU output to CPU baseline)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestFocalStatsGPU:
    """GPU focal statistics tests. Validate that GPU matches CPU for all stats."""

    def test_focal_min_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        cpu = raster_focal_min(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_min(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_focal_max_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_max

        cpu = raster_focal_max(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_max(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_focal_mean_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_mean

        cpu = raster_focal_mean(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_mean(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu, decimal=10)

    def test_focal_std_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_std

        cpu = raster_focal_std(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_std(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu, decimal=5)

    def test_focal_range_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_range

        cpu = raster_focal_range(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_range(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_focal_variety_gpu_vs_cpu(self, raster_categorical):
        from vibespatial.raster.algebra import raster_focal_variety

        cpu = raster_focal_variety(raster_categorical, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_variety(raster_categorical, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_gpu_nodata_handling(self, raster_with_nodata):
        from vibespatial.raster.algebra import raster_focal_min

        cpu = raster_focal_min(raster_with_nodata, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_min(raster_with_nodata, radius=1, use_gpu=True).to_numpy()
        # Nodata pixel should be nodata in both
        assert gpu[1, 1] == -9999.0
        assert cpu[1, 1] == -9999.0
        # Non-nodata pixels should match
        mask = cpu != -9999.0
        np.testing.assert_array_almost_equal(gpu[mask], cpu[mask])

    def test_gpu_asymmetric_radius(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_mean

        cpu = raster_focal_mean(raster_5x5, radius=(1, 2), use_gpu=False).to_numpy()
        gpu = raster_focal_mean(raster_5x5, radius=(1, 2), use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu, decimal=10)

    def test_gpu_diagnostics(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=True)
        assert any("gpu_focal_min" in d.detail for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Export / import tests
# ---------------------------------------------------------------------------


class TestExports:
    def test_importable_from_init(self):
        from vibespatial.raster import (  # noqa: F401
            raster_focal_max,
            raster_focal_mean,
            raster_focal_min,
            raster_focal_range,
            raster_focal_std,
            raster_focal_variety,
        )

    def test_in_all(self):
        from vibespatial.raster import __all__

        for name in [
            "raster_focal_min",
            "raster_focal_max",
            "raster_focal_mean",
            "raster_focal_std",
            "raster_focal_range",
            "raster_focal_variety",
        ]:
            assert name in __all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# Multi-band raster tests (Bug #6: 3D/2D nodata_mask shape mismatch)
# ---------------------------------------------------------------------------


class TestMultiBandFocalStats:
    """Verify focal stats handle multi-band (3D) rasters without crashing.

    Bug #6: _focal_stat_cpu squeezes data from 3D to 2D but leaves
    nodata_mask as 3D, causing a shape mismatch crash.
    """

    @pytest.fixture
    def raster_3d_no_nodata(self):
        """Single-band raster stored as (1, 5, 5) shape — no nodata."""
        data = np.arange(1, 26, dtype=np.float64).reshape(1, 5, 5)
        return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))

    @pytest.fixture
    def raster_3d_with_nodata(self):
        """Single-band raster stored as (1, 5, 5) shape — with nodata at (0,1,1)."""
        data = np.arange(1, 26, dtype=np.float64).reshape(1, 5, 5)
        data[0, 1, 1] = -9999.0
        return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))

    @pytest.fixture
    def raster_2d_with_nodata(self):
        """Equivalent 2D raster for comparison."""
        data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
        data[1, 1] = -9999.0
        return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))

    def test_focal_min_3d_no_nodata(self, raster_3d_no_nodata):
        """3D raster without nodata should not crash."""
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_3d_no_nodata, radius=1, use_gpu=False)
        out = result.to_numpy()
        assert out.ndim == 2
        assert out.shape == (5, 5)
        # Center pixel (2,2): min of [7,8,9,12,13,14,17,18,19] = 7
        assert out[2, 2] == 7.0

    def test_focal_min_3d_with_nodata(self, raster_3d_with_nodata):
        """3D raster with nodata must not crash (the core Bug #6 scenario)."""
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_3d_with_nodata, radius=1, use_gpu=False)
        out = result.to_numpy()
        assert out.ndim == 2
        assert out.shape == (5, 5)
        # Nodata pixel itself must remain nodata
        assert out[1, 1] == -9999.0

    def test_focal_min_3d_matches_2d(self, raster_3d_with_nodata, raster_2d_with_nodata):
        """3D and equivalent 2D rasters must produce identical results."""
        from vibespatial.raster.algebra import raster_focal_min

        out_3d = raster_focal_min(raster_3d_with_nodata, radius=1, use_gpu=False).to_numpy()
        out_2d = raster_focal_min(raster_2d_with_nodata, radius=1, use_gpu=False).to_numpy()
        np.testing.assert_array_equal(out_3d, out_2d)

    def test_all_stats_3d_with_nodata(self, raster_3d_with_nodata):
        """All focal stat functions must handle 3D+nodata without crashing."""
        from vibespatial.raster.algebra import (
            raster_focal_max,
            raster_focal_mean,
            raster_focal_min,
            raster_focal_range,
            raster_focal_std,
            raster_focal_variety,
        )

        for fn in [
            raster_focal_min,
            raster_focal_max,
            raster_focal_mean,
            raster_focal_std,
            raster_focal_range,
            raster_focal_variety,
        ]:
            result = fn(raster_3d_with_nodata, radius=1, use_gpu=False)
            out = result.to_numpy()
            assert out.ndim == 2, f"{fn.__name__} returned {out.ndim}D"
            assert out.shape == (5, 5), f"{fn.__name__} shape {out.shape}"
            # Nodata pixel must be preserved in all stats
            assert out[1, 1] == -9999.0, f"{fn.__name__} lost nodata"

    def test_metadata_preserved_3d(self, raster_3d_with_nodata):
        """Affine, CRS, and nodata must survive through 3D focal stat."""
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_3d_with_nodata, radius=1, use_gpu=False)
        assert result.affine == raster_3d_with_nodata.affine
        assert result.crs == raster_3d_with_nodata.crs
        assert result.nodata == raster_3d_with_nodata.nodata

    def test_diagnostics_3d(self, raster_3d_with_nodata):
        """Diagnostic event must be emitted for 3D raster path."""
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_3d_with_nodata, radius=1, use_gpu=False)
        assert any("cpu_focal_min" in d.detail for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Population std (ddof=0) consistency tests — Bug #12
# ---------------------------------------------------------------------------


class TestFocalStdPopulation:
    """Verify focal std uses population std (ddof=0) matching GIS convention."""

    def test_cpu_focal_std_is_population(self, raster_5x5):
        """CPU focal std must match np.std(ddof=0) for every interior pixel."""
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        data = raster_5x5.to_numpy()

        # Check all fully interior pixels (rows 1-3, cols 1-3)
        for r in range(1, 4):
            for c in range(1, 4):
                window = data[r - 1 : r + 2, c - 1 : c + 2].ravel()
                expected_pop = np.std(window, ddof=0)
                np.testing.assert_almost_equal(
                    out[r, c],
                    expected_pop,
                    decimal=10,
                    err_msg=f"CPU focal std at ({r},{c}) does not match population std",
                )

    def test_cpu_focal_std_not_sample(self, raster_5x5):
        """CPU focal std must NOT match np.std(ddof=1) — regression guard."""
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        data = raster_5x5.to_numpy()

        # Center (2,2) has a full 3x3 window
        window = data[1:4, 1:4].ravel()
        sample_std = np.std(window, ddof=1)
        pop_std = np.std(window, ddof=0)
        # These differ for n=9: sample / pop = sqrt(9/8)
        assert sample_std != pop_std, "Test data does not distinguish ddof=0 from ddof=1"
        # Result must match population, not sample
        np.testing.assert_almost_equal(out[2, 2], pop_std, decimal=10)
        assert abs(out[2, 2] - sample_std) > 1e-6, (
            "Focal std appears to use sample std (ddof=1) instead of population std (ddof=0)"
        )

    def test_single_valid_pixel_returns_zero(self):
        """A window with exactly one valid pixel must return std=0.0."""
        from vibespatial.raster.algebra import raster_focal_std

        # 3x3 raster: center=5.0, all others=nodata
        data = np.full((3, 3), -9999.0, dtype=np.float64)
        data[1, 1] = 5.0
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))

        result = raster_focal_std(raster, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center pixel: only 1 valid neighbor (itself) -> std = 0.0
        assert out[1, 1] == 0.0

    def test_uniform_window_returns_zero(self):
        """A window with all identical values must return std=0.0."""
        from vibespatial.raster.algebra import raster_focal_std

        data = np.full((5, 5), 42.0, dtype=np.float64)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        result = raster_focal_std(raster, radius=1, use_gpu=False)
        np.testing.assert_array_almost_equal(result.to_numpy(), 0.0)


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestFocalStdPopulationGPU:
    """GPU focal std must also use population std and match CPU exactly."""

    def test_gpu_focal_std_is_population(self, raster_5x5):
        """GPU focal std must match np.std(ddof=0) for interior pixels."""
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_5x5, radius=1, use_gpu=True)
        out = result.to_numpy()
        data = raster_5x5.to_numpy()

        for r in range(1, 4):
            for c in range(1, 4):
                window = data[r - 1 : r + 2, c - 1 : c + 2].ravel()
                expected_pop = np.std(window, ddof=0)
                np.testing.assert_almost_equal(
                    out[r, c],
                    expected_pop,
                    decimal=5,
                    err_msg=f"GPU focal std at ({r},{c}) does not match population std",
                )

    def test_gpu_cpu_focal_std_match(self, raster_5x5):
        """GPU and CPU focal std must produce identical results (both ddof=0)."""
        from vibespatial.raster.algebra import raster_focal_std

        cpu = raster_focal_std(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_std(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu, decimal=5)

    def test_gpu_focal_std_not_sample(self, raster_5x5):
        """GPU focal std must NOT match np.std(ddof=1) — regression guard."""
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_5x5, radius=1, use_gpu=True)
        out = result.to_numpy()
        data = raster_5x5.to_numpy()

        window = data[1:4, 1:4].ravel()
        sample_std = np.std(window, ddof=1)
        pop_std = np.std(window, ddof=0)
        np.testing.assert_almost_equal(out[2, 2], pop_std, decimal=5)
        assert abs(out[2, 2] - sample_std) > 1e-6, (
            "GPU focal std appears to use sample std (ddof=1) instead of population std"
        )
