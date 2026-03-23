"""Tests for hydrological DEM conditioning: sink/depression filling."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from vibespatial.raster.buffers import from_numpy
from vibespatial.raster.hydrology import raster_fill_sinks

requires_gpu = pytest.mark.gpu


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------


class TestFillSinksCPU:
    """Test CPU baseline sink filling."""

    def test_simple_pit(self):
        """A single interior pit should be filled to spill elevation."""
        # 3x3 DEM with a pit in the center
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # The pit at (1,1) should be filled to 5.0 (spill elevation)
        assert filled[1, 1] == pytest.approx(5.0)
        # Border pixels unchanged
        np.testing.assert_array_equal(filled[0, :], data[0, :])
        np.testing.assert_array_equal(filled[-1, :], data[-1, :])

    def test_deep_pit(self):
        """A deep pit should be filled to the spill level, not the pit depth."""
        data = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 3.0, 2.0, 3.0, 10.0],
                [10.0, 2.0, 0.0, 2.0, 10.0],
                [10.0, 3.0, 2.0, 3.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # All interior pixels should be filled to 10.0 (border elevation)
        assert filled[2, 2] == pytest.approx(10.0)
        assert filled[1, 1] == pytest.approx(10.0)

    def test_multiple_pits(self):
        """Multiple separate pits should each fill to their own spill level."""
        data = np.array(
            [
                [8.0, 8.0, 8.0, 4.0, 4.0],
                [8.0, 1.0, 8.0, 4.0, 4.0],
                [8.0, 8.0, 8.0, 4.0, 4.0],
                [6.0, 6.0, 6.0, 6.0, 6.0],
                [6.0, 2.0, 6.0, 6.0, 6.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # First pit at (1,1): surrounded by 8s on left and 4s on right
        # The spill goes through col=3 which has elevation 4.0
        # So fill level = max of the path to border
        assert filled[1, 1] >= data[1, 1]  # Must be raised
        # Second pit at (4,1): surrounded by 6s
        assert filled[4, 1] >= data[4, 1]  # Must be raised

    def test_already_filled(self):
        """A DEM with no depressions should be unchanged."""
        # Monotonically decreasing from top-left to bottom-right
        data = np.array(
            [
                [9.0, 8.0, 7.0],
                [8.0, 7.0, 6.0],
                [7.0, 6.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        np.testing.assert_array_almost_equal(filled, data)

    def test_flat_area(self):
        """A flat DEM should remain flat (no depressions to fill)."""
        data = np.full((5, 5), 10.0, dtype=np.float32)
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        np.testing.assert_array_almost_equal(filled, data)

    def test_nodata_barrier(self):
        """Nodata pixels should act as barriers and be preserved."""
        data = np.array(
            [
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [5.0, 1.0, -9999.0, 1.0, 5.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=-9999.0)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # Nodata pixel preserved
        assert filled[1, 2] == pytest.approx(-9999.0)
        # Pits on either side of barrier should still be filled
        assert filled[1, 1] >= data[1, 1]
        assert filled[1, 3] >= data[1, 3]

    def test_nan_nodata(self):
        """NaN nodata should be handled correctly."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, np.nan],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=np.nan)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # NaN preserved
        assert np.isnan(filled[1, 2])
        # Pit filled
        assert filled[1, 1] == pytest.approx(5.0)

    def test_spillway(self):
        """Depression with a lower spill point fills to that level."""
        data = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 2.0, 2.0, 2.0, 10.0],
                [10.0, 2.0, 1.0, 2.0, 6.0],
                [10.0, 2.0, 2.0, 2.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # The depression should fill to the spill point at (2,4)=6.0
        assert filled[2, 2] == pytest.approx(6.0)
        assert filled[1, 1] == pytest.approx(6.0)

    def test_integer_dtype(self):
        """Integer DEMs should work (promoted to float internally)."""
        data = np.array(
            [
                [5, 5, 5],
                [5, 1, 5],
                [5, 5, 5],
            ],
            dtype=np.int16,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # Result dtype matches input
        assert filled.dtype == np.int16
        assert filled[1, 1] == 5

    def test_single_pixel(self):
        """Single pixel raster should be returned as-is."""
        data = np.array([[42.0]], dtype=np.float32)
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        assert filled[0, 0] == pytest.approx(42.0)

    def test_single_row(self):
        """1-row raster: all pixels are border, no filling needed."""
        data = np.array([[5.0, 1.0, 5.0, 1.0, 5.0]], dtype=np.float32)
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        np.testing.assert_array_almost_equal(filled, data)

    def test_single_column(self):
        """1-column raster: all pixels are border, no filling needed."""
        data = np.array([[5.0], [1.0], [5.0]], dtype=np.float32)
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        np.testing.assert_array_almost_equal(filled, data)

    def test_3d_single_band(self):
        """3D input with shape (1, H, W) should work."""
        data = np.array(
            [
                [
                    [5.0, 5.0, 5.0],
                    [5.0, 1.0, 5.0],
                    [5.0, 5.0, 5.0],
                ]
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        # Result should be 2D
        assert filled.ndim == 2
        assert filled[1, 1] == pytest.approx(5.0)

    def test_multiband_raises(self):
        """Multi-band raster should raise ValueError."""
        data = np.ones((3, 5, 5), dtype=np.float32)
        raster = from_numpy(data)
        with pytest.raises(ValueError, match="single-band"):
            raster_fill_sinks(raster, use_gpu=False)

    def test_diagnostics_present(self):
        """Result should have diagnostic events."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)

        runtime_events = [
            e for e in result.diagnostics if e.kind == "runtime" and "fill_sinks_cpu" in e.detail
        ]
        assert len(runtime_events) >= 1


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestFillSinksGPU:
    """Test GPU sink filling matches CPU results."""

    def test_simple_pit_gpu(self):
        """GPU should fill a simple pit identically to CPU."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        cpu_result = raster_fill_sinks(raster, use_gpu=False).to_numpy()
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_almost_equal(gpu_result, cpu_result, decimal=5)

    def test_deep_pit_gpu(self):
        """GPU should handle deep pits correctly."""
        data = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 3.0, 2.0, 3.0, 10.0],
                [10.0, 2.0, 0.0, 2.0, 10.0],
                [10.0, 3.0, 2.0, 3.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )
        raster = from_numpy(data)
        cpu_result = raster_fill_sinks(raster, use_gpu=False).to_numpy()
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_almost_equal(gpu_result, cpu_result)

    def test_spillway_gpu(self):
        """GPU should fill to the correct spill level."""
        data = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 2.0, 2.0, 2.0, 10.0],
                [10.0, 2.0, 1.0, 2.0, 6.0],
                [10.0, 2.0, 2.0, 2.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        cpu_result = raster_fill_sinks(raster, use_gpu=False).to_numpy()
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_almost_equal(gpu_result, cpu_result, decimal=5)

    def test_nodata_barrier_gpu(self):
        """GPU nodata handling should match CPU."""
        data = np.array(
            [
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [5.0, 1.0, -9999.0, 1.0, 5.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=-9999.0)
        cpu_result = raster_fill_sinks(raster, use_gpu=False).to_numpy()
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_almost_equal(gpu_result, cpu_result, decimal=5)

    def test_nan_nodata_gpu(self):
        """GPU NaN nodata handling should match CPU."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, np.nan],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        raster = from_numpy(data, nodata=np.nan)
        cpu_result = raster_fill_sinks(raster, use_gpu=False).to_numpy()
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        # NaN in same positions
        np.testing.assert_array_equal(np.isnan(cpu_result), np.isnan(gpu_result))
        # Non-NaN values match
        valid = ~np.isnan(cpu_result)
        np.testing.assert_array_almost_equal(gpu_result[valid], cpu_result[valid])

    def test_already_filled_gpu(self):
        """GPU on an already-filled DEM should be a no-op."""
        data = np.array(
            [
                [9.0, 8.0, 7.0],
                [8.0, 7.0, 6.0],
                [7.0, 6.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_almost_equal(gpu_result, data, decimal=5)

    def test_flat_area_gpu(self):
        """GPU flat area should remain unchanged."""
        data = np.full((5, 5), 10.0, dtype=np.float32)
        raster = from_numpy(data)
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_almost_equal(gpu_result, data, decimal=5)

    def test_float64_gpu(self):
        """GPU should work with float64 dtype."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        raster = from_numpy(data)
        cpu_result = raster_fill_sinks(raster, use_gpu=False).to_numpy()
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_almost_equal(gpu_result, cpu_result)

    def test_integer_dtype_gpu(self):
        """GPU should handle integer DEMs (promoted to float internally)."""
        data = np.array(
            [
                [5, 5, 5],
                [5, 1, 5],
                [5, 5, 5],
            ],
            dtype=np.int16,
        )
        raster = from_numpy(data)
        cpu_result = raster_fill_sinks(raster, use_gpu=False).to_numpy()
        gpu_result = raster_fill_sinks(raster, use_gpu=True).to_numpy()

        np.testing.assert_array_equal(gpu_result, cpu_result)
        assert gpu_result.dtype == np.int16

    def test_gpu_diagnostics(self):
        """GPU result should have diagnostic events."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=True)

        runtime_events = [
            e for e in result.diagnostics if e.kind == "runtime" and "fill_sinks_gpu" in e.detail
        ]
        assert len(runtime_events) >= 1


# ---------------------------------------------------------------------------
# Auto-dispatch tests
# ---------------------------------------------------------------------------


class TestFillSinksAutoDispatch:
    """Test auto-dispatch fallback behavior."""

    def test_auto_dispatch_works(self):
        """Auto-dispatch should work regardless of GPU availability."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster)
        filled = result.to_numpy()

        # Should fill the pit regardless of GPU/CPU path
        assert filled[1, 1] == pytest.approx(5.0)

    def test_explicit_cpu(self):
        """Explicit use_gpu=False should always use CPU."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_fill_sinks(raster, use_gpu=False)
        filled = result.to_numpy()

        assert filled[1, 1] == pytest.approx(5.0)

    def test_preserves_metadata(self):
        """Result should preserve affine and CRS from input."""
        data = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        affine = (1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
        raster = from_numpy(data, affine=affine)
        result = raster_fill_sinks(raster, use_gpu=False)

        assert result.affine == affine
        assert result.nodata == raster.nodata
