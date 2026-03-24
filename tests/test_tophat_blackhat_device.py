"""Tests for Bug #21 fix: tophat/blackhat device-side difference computation.

Verifies that raster_morphology_tophat and raster_morphology_blackhat compute
their binary difference on device when use_gpu=True, avoiding unnecessary D->H
transfers. CPU tests always run; GPU tests require CuPy and are marked
@pytest.mark.gpu.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from scipy.ndimage import label as _scipy_label  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from vibespatial.raster.buffers import (
    RasterDiagnosticKind,
    from_numpy,
)
from vibespatial.raster.label import (
    raster_morphology_blackhat,
    raster_morphology_tophat,
)
from vibespatial.residency import Residency

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# ---------------------------------------------------------------------------
# CPU path: correctness, diagnostics, nodata, metadata
# ---------------------------------------------------------------------------


class TestTopHatBlackHatCPUDiff:
    """Verify the CPU difference helpers produce correct results."""

    def test_tophat_cpu_correctness(self):
        """Top-hat CPU path extracts small bright features."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1  # large block
        data[2, 2] = 1  # small bright feature
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=False)
        tophat = result.to_numpy()
        # Isolated pixel should survive top-hat
        assert tophat[2, 2] == 1
        # Bulk of large block should not
        assert tophat[10, 10] == 0

    def test_blackhat_cpu_correctness(self):
        """Black-hat CPU path extracts small dark features (holes)."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[9, 9] = 0  # small hole
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, use_gpu=False)
        blackhat = result.to_numpy()
        # The hole should appear in black-hat
        assert blackhat[9, 9] == 1
        # Bulk area should not
        assert blackhat[7, 7] == 0

    def test_tophat_cpu_diagnostics(self):
        """CPU top-hat path appends a RUNTIME diagnostic with 'cpu' indicator."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        detail = runtime_events[-1].detail
        assert "tophat_cpu" in detail

    def test_blackhat_cpu_diagnostics(self):
        """CPU black-hat path appends a RUNTIME diagnostic with 'cpu' indicator."""
        data = np.ones((10, 10), dtype=np.uint8)
        data[5, 5] = 0
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        detail = runtime_events[-1].detail
        assert "blackhat_cpu" in detail

    def test_tophat_cpu_nodata_propagation(self):
        """Nodata pixels should not appear as foreground in top-hat result."""
        data = np.zeros((10, 10), dtype=np.float32)
        data[3:7, 3:7] = 1.0
        data[5, 5] = -9999.0  # nodata pixel inside foreground
        raster = from_numpy(data, nodata=-9999.0)
        result = raster_morphology_tophat(raster, use_gpu=False)
        tophat = result.to_numpy()
        # Nodata pixel should not be marked as foreground in the difference
        assert tophat[5, 5] == 0

    def test_blackhat_cpu_nodata_propagation(self):
        """Nodata pixels treated as background in morphology; black-hat shows holes filled by closing."""
        data = np.ones((10, 10), dtype=np.float32)
        data[5, 5] = -9999.0  # nodata treated as background
        raster = from_numpy(data, nodata=-9999.0)
        result = raster_morphology_blackhat(raster, use_gpu=False)
        blackhat = result.to_numpy()
        # The nodata pixel was treated as a hole in foreground.
        # Closing fills the hole, so blackhat correctly marks (5,5) as 1:
        # closed_bin=1 & ~orig_bin=1 -> 1.
        assert blackhat[5, 5] == 1
        # A pixel that was foreground in both original and closed should be 0
        assert blackhat[3, 3] == 0

    def test_tophat_cpu_metadata_preservation(self):
        """Top-hat preserves affine, CRS, and dtype."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1
        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        raster = from_numpy(data, affine=affine)
        result = raster_morphology_tophat(raster, use_gpu=False)
        assert result.affine == affine
        assert result.crs == raster.crs
        assert result.dtype == np.dtype(np.uint8)

    def test_blackhat_cpu_metadata_preservation(self):
        """Black-hat preserves affine, CRS, and dtype."""
        data = np.ones((10, 10), dtype=np.uint8)
        data[5, 5] = 0
        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        raster = from_numpy(data, affine=affine)
        result = raster_morphology_blackhat(raster, use_gpu=False)
        assert result.affine == affine
        assert result.crs == raster.crs
        assert result.dtype == np.dtype(np.uint8)

    def test_tophat_cpu_all_zeros(self):
        """Top-hat of all-zero raster is all zeros."""
        data = np.zeros((10, 10), dtype=np.uint8)
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=False)
        np.testing.assert_array_equal(result.to_numpy(), 0)

    def test_blackhat_cpu_all_ones(self):
        """Black-hat of all-one raster is all zeros."""
        data = np.ones((10, 10), dtype=np.uint8)
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, use_gpu=False)
        np.testing.assert_array_equal(result.to_numpy(), 0)

    def test_tophat_cpu_result_is_host_resident(self):
        """CPU path result should be HOST-resident."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=False)
        assert result.residency is Residency.HOST

    def test_blackhat_cpu_result_is_host_resident(self):
        """CPU path result should be HOST-resident."""
        data = np.ones((10, 10), dtype=np.uint8)
        data[5, 5] = 0
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, use_gpu=False)
        assert result.residency is Residency.HOST


# ---------------------------------------------------------------------------
# GPU path: device residency, diagnostics, correctness vs CPU
# ---------------------------------------------------------------------------


@requires_gpu
class TestTopHatBlackHatGPUDiff:
    """Verify the GPU difference helpers stay on device."""

    def test_tophat_gpu_result_is_device_resident(self):
        """GPU top-hat should return a DEVICE-resident raster."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[2, 2] = 1
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=True)
        assert result.residency is Residency.DEVICE

    def test_blackhat_gpu_result_is_device_resident(self):
        """GPU black-hat should return a DEVICE-resident raster."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[9, 9] = 0
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, use_gpu=True)
        assert result.residency is Residency.DEVICE

    def test_tophat_gpu_matches_cpu(self):
        """GPU top-hat should produce identical results to CPU path."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[2, 2] = 1  # small bright feature
        raster = from_numpy(data)
        gpu_result = raster_morphology_tophat(raster, use_gpu=True).to_numpy()
        cpu_result = raster_morphology_tophat(raster, use_gpu=False).to_numpy()
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_blackhat_gpu_matches_cpu(self):
        """GPU black-hat should produce identical results to CPU path."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[9, 9] = 0  # small hole
        raster = from_numpy(data)
        gpu_result = raster_morphology_blackhat(raster, use_gpu=True).to_numpy()
        cpu_result = raster_morphology_blackhat(raster, use_gpu=False).to_numpy()
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_tophat_gpu_diagnostics(self):
        """GPU top-hat path appends a RUNTIME diagnostic with 'gpu' indicator."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=True)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        detail = runtime_events[-1].detail
        assert "tophat_gpu" in detail

    def test_blackhat_gpu_diagnostics(self):
        """GPU black-hat path appends a RUNTIME diagnostic with 'gpu' indicator."""
        data = np.ones((10, 10), dtype=np.uint8)
        data[5, 5] = 0
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, use_gpu=True)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        detail = runtime_events[-1].detail
        assert "blackhat_gpu" in detail

    def test_tophat_gpu_nodata_propagation(self):
        """Nodata pixels excluded from foreground on GPU path."""
        data = np.zeros((10, 10), dtype=np.float32)
        data[3:7, 3:7] = 1.0
        data[5, 5] = -9999.0
        raster = from_numpy(data, nodata=-9999.0)
        result = raster_morphology_tophat(raster, use_gpu=True)
        tophat = result.to_numpy()
        assert tophat[5, 5] == 0

    def test_blackhat_gpu_nodata_propagation(self):
        """Nodata pixels treated as background; GPU matches CPU semantics."""
        data = np.ones((10, 10), dtype=np.float32)
        data[5, 5] = -9999.0
        raster = from_numpy(data, nodata=-9999.0)
        result = raster_morphology_blackhat(raster, use_gpu=True)
        blackhat = result.to_numpy()
        # Nodata treated as hole; closing fills it, so blackhat marks it as 1
        assert blackhat[5, 5] == 1
        assert blackhat[3, 3] == 0

    def test_tophat_gpu_metadata_preservation(self):
        """GPU top-hat preserves affine, CRS, and dtype."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1
        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        raster = from_numpy(data, affine=affine)
        result = raster_morphology_tophat(raster, use_gpu=True)
        assert result.affine == affine
        assert result.crs == raster.crs
        assert result.dtype == np.dtype(np.uint8)

    def test_blackhat_gpu_metadata_preservation(self):
        """GPU black-hat preserves affine, CRS, and dtype."""
        data = np.ones((10, 10), dtype=np.uint8)
        data[5, 5] = 0
        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        raster = from_numpy(data, affine=affine)
        result = raster_morphology_blackhat(raster, use_gpu=True)
        assert result.affine == affine
        assert result.crs == raster.crs
        assert result.dtype == np.dtype(np.uint8)
