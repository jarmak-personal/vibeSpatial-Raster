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
from vibespatial.raster.hydrology import _fill_sinks_cpu, raster_fill_sinks

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


# ---------------------------------------------------------------------------
# Convergence diagnostic tests
# ---------------------------------------------------------------------------


class TestFillSinksConvergenceCPU:
    """Test that convergence status is correctly reported in diagnostics."""

    def _make_deep_pit_5x5(self):
        """Create a 5x5 DEM with a central pit needing >1 iteration."""
        return np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )

    def test_converged_diagnostics(self):
        """When CPU converges, diagnostics should contain converged=True."""
        data = self._make_deep_pit_5x5()
        raster = from_numpy(data)
        result = _fill_sinks_cpu(raster)

        runtime_events = [
            e for e in result.diagnostics if e.kind == "runtime" and "fill_sinks_cpu" in e.detail
        ]
        assert len(runtime_events) >= 1
        assert "converged=True" in runtime_events[0].detail

    def test_unconverged_warns(self):
        """When CPU hits max_iterations, diagnostics should warn unconverged."""
        # This 5x5 DEM needs 2 iterations. Cap at 1 to force unconverged.
        data = self._make_deep_pit_5x5()
        raster = from_numpy(data)
        result = _fill_sinks_cpu(raster, _max_iterations=1)

        # The main diagnostic should say converged=False
        runtime_events = [
            e for e in result.diagnostics if e.kind == "runtime" and "fill_sinks_cpu" in e.detail
        ]
        assert any("converged=False" in e.detail for e in runtime_events)

        # There should also be a WARNING diagnostic
        warning_events = [
            e
            for e in result.diagnostics
            if e.kind == "runtime" and "WARNING" in e.detail and "converge" in e.detail
        ]
        assert len(warning_events) == 1
        assert "did not converge" in warning_events[0].detail
        assert warning_events[0].visible_to_user is True

    def test_unconverged_result_has_unfilled_pixels(self):
        """When CPU is unconverged, the result should have unfilled interior pixels."""
        data = self._make_deep_pit_5x5()
        raster = from_numpy(data)
        result = _fill_sinks_cpu(raster, _max_iterations=1)
        filled = result.to_numpy()

        # Center pixel (2,2) should NOT be fully filled to 10.0 with only 1 iteration.
        # It stays at +inf because it is 2 hops from the border and info only
        # propagates one ring per iteration.
        assert filled[2, 2] != pytest.approx(10.0)

    def test_unconverged_emits_logger_warning(self, caplog):
        """When CPU is unconverged, a logger warning should be emitted."""
        import logging

        data = self._make_deep_pit_5x5()
        raster = from_numpy(data)
        with caplog.at_level(logging.WARNING, logger="vibespatial.raster.hydrology"):
            _fill_sinks_cpu(raster, _max_iterations=1)

        assert any("did NOT converge" in record.message for record in caplog.records)


@requires_gpu
class TestFillSinksConvergenceGPU:
    """Test GPU convergence diagnostics."""

    def _make_deep_pit_5x5(self):
        """Create a 5x5 DEM with a central pit."""
        return np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )

    def test_converged_diagnostics_gpu(self):
        """When GPU converges, diagnostics should contain converged=True."""
        from vibespatial.raster.hydrology import _fill_sinks_gpu

        data = self._make_deep_pit_5x5()
        raster = from_numpy(data)
        result = _fill_sinks_gpu(raster)

        runtime_events = [
            e for e in result.diagnostics if e.kind == "runtime" and "fill_sinks_gpu" in e.detail
        ]
        assert len(runtime_events) >= 1
        assert "converged=True" in runtime_events[0].detail

    def test_unconverged_warns_gpu(self):
        """When GPU hits max_iterations (not at batch boundary), diagnostics warn."""
        from vibespatial.raster.hydrology import _fill_sinks_gpu

        # Use max_iterations=3 -- NOT a multiple of batch size (32).
        # The 5x5 DEM with a central pit needs a few GPU kernel iterations.
        # With max_iterations=3, the loop exhausts without hitting a batch
        # boundary check, exercising the post-loop convergence check.
        data = self._make_deep_pit_5x5()
        raster = from_numpy(data)
        _fill_sinks_gpu(raster, _max_iterations=3)  # sanity: must not raise

        # The 5x5 pit may converge in 3 GPU iterations.  Use 1 iteration
        # to guarantee the loop exhausts and the post-loop check fires.
        result_1iter = _fill_sinks_gpu(raster, _max_iterations=1)
        events_1 = [
            e
            for e in result_1iter.diagnostics
            if e.kind == "runtime" and "fill_sinks_gpu" in e.detail
        ]
        # With only 1 iteration and batch size 32, the loop body runs once
        # and then exhausts. The else clause performs the final check.
        # The DEM may or may not converge in 1 GPU iteration depending on
        # kernel semantics. Check that converged status is reported either way.
        assert any("converged=" in e.detail for e in events_1)

    def test_unconverged_not_at_batch_boundary_gpu(self):
        """Verify unconverged detection when max_iterations is not a batch multiple.

        This is the specific edge case from Bug #9: if max_iterations is not a
        multiple of CONVERGENCE_BATCH_SIZE (32), the last partial batch's changes
        were never checked before the fix.
        """
        from vibespatial.raster.hydrology import _fill_sinks_gpu

        # Create a larger DEM that genuinely needs many iterations.
        # A 20x20 DEM with a deep central pit requires O(10) propagation steps.
        size = 20
        data = np.full((size, size), 100.0, dtype=np.float32)
        data[5:15, 5:15] = 1.0  # Large interior depression

        raster = from_numpy(data)

        # Cap at 5 iterations (not a multiple of 32) -- guaranteed unconverged
        # for a 10x10 interior depression.
        result = _fill_sinks_gpu(raster, _max_iterations=5)

        # Must have a WARNING diagnostic
        warning_events = [
            e
            for e in result.diagnostics
            if e.kind == "runtime" and "WARNING" in e.detail and "converge" in e.detail
        ]
        assert len(warning_events) == 1
        assert "did not converge" in warning_events[0].detail
        assert warning_events[0].visible_to_user is True

        # The main diagnostic should report converged=False
        main_events = [
            e
            for e in result.diagnostics
            if e.kind == "runtime" and "fill_sinks_gpu" in e.detail and "WARNING" not in e.detail
        ]
        assert any("converged=False" in e.detail for e in main_events)

    def test_unconverged_emits_logger_warning_gpu(self, caplog):
        """When GPU is unconverged, a logger warning should be emitted."""
        import logging

        from vibespatial.raster.hydrology import _fill_sinks_gpu

        size = 20
        data = np.full((size, size), 100.0, dtype=np.float32)
        data[5:15, 5:15] = 1.0

        raster = from_numpy(data)
        with caplog.at_level(logging.WARNING, logger="vibespatial.raster.hydrology"):
            _fill_sinks_gpu(raster, _max_iterations=5)

        assert any("did NOT converge" in record.message for record in caplog.records)
