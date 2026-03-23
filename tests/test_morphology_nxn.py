"""Tests for NxN binary morphology with arbitrary structuring elements.

Covers: make_structuring_element, raster_morphology with custom SE,
separable decomposition for rectangular SEs, disk SE, top-hat, black-hat.
CPU tests always run; GPU tests require CuPy and are marked @pytest.mark.gpu.
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

from vibespatial.raster.buffers import from_numpy
from vibespatial.raster.label import (
    make_structuring_element,
    raster_morphology,
    raster_morphology_blackhat,
    raster_morphology_tophat,
)

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")

requires_gpu = pytest.mark.gpu


def _cpu_morph_with_se(data, operation, se, nodata=None, iterations=1):
    """CPU reference morphology using scipy with an arbitrary SE."""
    raster = from_numpy(data, nodata=nodata)
    return raster_morphology(
        raster,
        operation,
        structuring_element=se,
        iterations=iterations,
        use_gpu=False,
    ).to_numpy()


# ---------------------------------------------------------------------------
# Tests: make_structuring_element
# ---------------------------------------------------------------------------


class TestMakeStructuringElement:
    def test_rect_3x3(self):
        se = make_structuring_element("rect", 3)
        expected = np.ones((3, 3), dtype=np.uint8)
        np.testing.assert_array_equal(se, expected)

    def test_rect_5x5(self):
        se = make_structuring_element("rect", 5)
        assert se.shape == (5, 5)
        assert se.all()

    def test_rect_hw(self):
        se = make_structuring_element("rect", (3, 7))
        assert se.shape == (3, 7)
        assert se.all()

    def test_cross_3x3(self):
        se = make_structuring_element("cross", 3)
        expected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(se, expected)

    def test_cross_5x5(self):
        se = make_structuring_element("cross", 5)
        assert se.shape == (5, 5)
        # Center row and center column should be all 1
        assert se[2, :].all()
        assert se[:, 2].all()
        # Corners should be 0
        assert se[0, 0] == 0
        assert se[4, 4] == 0

    def test_disk_1(self):
        se = make_structuring_element("disk", 1)
        # radius=1 -> 3x3, pixels within distance 1 of center
        assert se.shape == (3, 3)
        expected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(se, expected)

    def test_disk_2(self):
        se = make_structuring_element("disk", 2)
        assert se.shape == (5, 5)
        # Center and cardinal neighbors should be active
        assert se[2, 2] == 1
        assert se[0, 2] == 1
        assert se[2, 0] == 1
        # Corners at distance sqrt(8) > 2 should be 0
        assert se[0, 0] == 0
        assert se[4, 4] == 0

    def test_disk_3(self):
        se = make_structuring_element("disk", 3)
        assert se.shape == (7, 7)
        # Center must be 1
        assert se[3, 3] == 1

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="shape must be"):
            make_structuring_element("triangle", 3)

    def test_even_size_rect(self):
        with pytest.raises(ValueError, match="odd"):
            make_structuring_element("rect", 4)

    def test_disk_tuple_rejected(self):
        with pytest.raises(ValueError, match="single integer radius"):
            make_structuring_element("disk", (3, 5))


# ---------------------------------------------------------------------------
# CPU tests: NxN morphology with custom SE
# ---------------------------------------------------------------------------


class TestMorphologyNxNCPU:
    def test_erode_5x5_rect(self):
        """Erosion with 5x5 rect SE should shrink foreground by 2 pixels."""
        data = np.zeros((15, 15), dtype=np.uint8)
        data[4:11, 4:11] = 1  # 7x7 block
        se = make_structuring_element("rect", 5)
        result = _cpu_morph_with_se(data, "erode", se)
        # After erosion with radius 2, only 3x3 center should remain
        assert result[7, 7] == 1  # center survives
        assert result[4, 4] == 0  # edge removed

    def test_dilate_5x5_rect(self):
        """Dilation with 5x5 rect SE should grow foreground by 2 pixels."""
        data = np.zeros((15, 15), dtype=np.uint8)
        data[7, 7] = 1  # single pixel
        se = make_structuring_element("rect", 5)
        result = _cpu_morph_with_se(data, "dilate", se)
        # Should expand to 5x5 block
        assert result[5, 5] == 1
        assert result[9, 9] == 1
        assert result[4, 4] == 0  # outside radius

    def test_erode_disk(self):
        """Erosion with disk SE preserves circular features."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        se = make_structuring_element("disk", 2)
        result = _cpu_morph_with_se(data, "erode", se)
        assert result[10, 10] == 1  # center survives
        assert result[5, 5] == 0  # edge eroded

    def test_dilate_cross_5x5(self):
        """Dilation with 5x5 cross SE."""
        data = np.zeros((11, 11), dtype=np.uint8)
        data[5, 5] = 1
        se = make_structuring_element("cross", 5)
        result = _cpu_morph_with_se(data, "dilate", se)
        # Cross dilation: center row and column expand by 2
        assert result[5, 3] == 1  # horizontal reach
        assert result[3, 5] == 1  # vertical reach
        assert result[3, 3] == 0  # diagonal not reached by cross

    def test_open_large_rect(self):
        """Opening with large rect removes small protrusions."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1  # 10x10 block
        data[5, 3] = 1  # small protrusion
        se = make_structuring_element("rect", 5)
        result = _cpu_morph_with_se(data, "open", se)
        # Protrusion should be removed, main block preserved
        assert result[5, 3] == 0
        assert result[10, 10] == 1

    def test_close_fills_small_hole(self):
        """Closing with large rect fills small holes."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[9, 9] = 0  # small hole
        se = make_structuring_element("rect", 3)
        result = _cpu_morph_with_se(data, "close", se)
        assert result[9, 9] == 1  # hole filled

    def test_structuring_element_none_uses_connectivity(self):
        """Passing None for SE falls back to connectivity-based 3x3."""
        data = np.ones((5, 5), dtype=np.uint8)
        raster = from_numpy(data)
        # Should not raise; uses default SE from connectivity
        result = raster_morphology(raster, "erode", connectivity=4, use_gpu=False)
        assert result is not None

    def test_iterations_with_custom_se(self):
        """Multiple iterations with custom SE."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        se = make_structuring_element("rect", 3)
        result = _cpu_morph_with_se(data, "erode", se, iterations=2)
        # 2 iterations of radius-1 erosion = radius-2 total
        assert result[7, 7] == 1
        assert result[5, 5] == 0


# ---------------------------------------------------------------------------
# CPU tests: top-hat and black-hat
# ---------------------------------------------------------------------------


class TestTopHatBlackHatCPU:
    def test_tophat_extracts_small_bright_features(self):
        """Top-hat should extract features smaller than SE."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1  # large block
        data[2, 2] = 1  # small bright feature
        se = make_structuring_element("rect", 5)
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, se, use_gpu=False)
        tophat = result.to_numpy()
        # The isolated pixel should be in the top-hat result
        assert tophat[2, 2] == 1
        # The bulk of the large block should NOT be in top-hat
        assert tophat[10, 10] == 0

    def test_blackhat_extracts_small_dark_features(self):
        """Black-hat should extract holes smaller than SE."""
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[9, 9] = 0  # small hole
        se = make_structuring_element("rect", 3)
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, se, use_gpu=False)
        blackhat = result.to_numpy()
        # The hole should appear in black-hat
        assert blackhat[9, 9] == 1
        # Bulk area should NOT appear
        assert blackhat[7, 7] == 0

    def test_tophat_all_zeros(self):
        """Top-hat of all-zero raster is all zeros."""
        data = np.zeros((10, 10), dtype=np.uint8)
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=False)
        np.testing.assert_array_equal(result.to_numpy(), 0)

    def test_blackhat_all_ones(self):
        """Black-hat of all-one raster is all zeros."""
        data = np.ones((10, 10), dtype=np.uint8)
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster, use_gpu=False)
        np.testing.assert_array_equal(result.to_numpy(), 0)

    def test_tophat_default_se(self):
        """Top-hat with default SE (None) should work."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster, use_gpu=False)
        assert result is not None


# ---------------------------------------------------------------------------
# GPU tests: NxN morphology
# ---------------------------------------------------------------------------


@requires_gpu
class TestMorphologyNxNGPU:
    """GPU NxN morphology tests -- compare GPU kernels against CPU baseline."""

    def test_erode_5x5_rect_gpu_matches_cpu(self):
        data = np.zeros((15, 15), dtype=np.uint8)
        data[4:11, 4:11] = 1
        se = make_structuring_element("rect", 5)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "erode", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "erode", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_dilate_5x5_rect_gpu_matches_cpu(self):
        data = np.zeros((15, 15), dtype=np.uint8)
        data[7, 7] = 1
        se = make_structuring_element("rect", 5)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "dilate", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "dilate", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_erode_disk_gpu_matches_cpu(self):
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        se = make_structuring_element("disk", 2)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "erode", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "erode", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_dilate_disk_gpu_matches_cpu(self):
        data = np.zeros((20, 20), dtype=np.uint8)
        data[10, 10] = 1
        se = make_structuring_element("disk", 3)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "dilate", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "dilate", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_open_cross_gpu_matches_cpu(self):
        data = np.zeros((15, 15), dtype=np.uint8)
        data[4:11, 4:11] = 1
        data[4, 2] = 1  # protrusion
        se = make_structuring_element("cross", 5)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "open", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "open", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_close_disk_gpu_matches_cpu(self):
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[9, 9] = 0  # hole
        se = make_structuring_element("disk", 2)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "close", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "close", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_7x7_rect_separable_gpu_matches_cpu(self):
        """Large rectangular SE should use separable decomposition."""
        data = np.zeros((25, 25), dtype=np.uint8)
        data[8:17, 8:17] = 1
        se = make_structuring_element("rect", 7)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "erode", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "erode", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_dilate_7x7_rect_separable_gpu_matches_cpu(self):
        data = np.zeros((25, 25), dtype=np.uint8)
        data[12, 12] = 1
        se = make_structuring_element("rect", 7)
        raster = from_numpy(data)
        gpu_result = raster_morphology(
            raster, "dilate", structuring_element=se, use_gpu=True
        ).to_numpy()
        cpu_result = _cpu_morph_with_se(data, "dilate", se)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_iterations_2_nxn_gpu_matches_cpu(self):
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        se = make_structuring_element("disk", 2)
        raster = from_numpy(data)
        for op in ("erode", "dilate", "open", "close"):
            gpu_result = raster_morphology(
                raster, op, structuring_element=se, iterations=2, use_gpu=True
            ).to_numpy()
            cpu_result = _cpu_morph_with_se(data, op, se, iterations=2)
            np.testing.assert_array_equal(
                gpu_result, cpu_result, err_msg=f"Mismatch for op={op} iter=2"
            )

    def test_backward_compat_3x3_default(self):
        """Passing no structuring_element should use legacy 3x3 fast path."""
        data = np.ones((5, 5), dtype=np.uint8)
        raster = from_numpy(data)
        # Should dispatch to the fast 3x3 kernel, not NxN
        result = raster_morphology(raster, "erode", connectivity=4, use_gpu=True)
        cpu_result = raster_morphology(raster, "erode", connectivity=4, use_gpu=False)
        np.testing.assert_array_equal(result.to_numpy(), cpu_result.to_numpy())

    def test_all_zeros_nxn(self):
        data = np.zeros((10, 10), dtype=np.uint8)
        se = make_structuring_element("rect", 5)
        raster = from_numpy(data)
        for op in ("erode", "dilate"):
            result = raster_morphology(raster, op, structuring_element=se, use_gpu=True).to_numpy()
            np.testing.assert_array_equal(result, 0)

    def test_all_ones_dilate_nxn(self):
        data = np.ones((10, 10), dtype=np.uint8)
        se = make_structuring_element("rect", 5)
        raster = from_numpy(data)
        result = raster_morphology(
            raster, "dilate", structuring_element=se, use_gpu=True
        ).to_numpy()
        np.testing.assert_array_equal(result, 1)

    def test_gpu_nxn_diagnostics(self):
        """NxN GPU path should record diagnostic events."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[3:7, 3:7] = 1
        se = make_structuring_element("disk", 2)
        raster = from_numpy(data)
        result = raster_morphology(raster, "erode", structuring_element=se, use_gpu=True)
        runtime_events = [
            e
            for e in result.diagnostics
            if e.kind == "runtime" and "morphology_nxn_gpu" in e.detail
        ]
        assert len(runtime_events) == 1
        assert runtime_events[0].elapsed_seconds > 0


# ---------------------------------------------------------------------------
# GPU tests: top-hat and black-hat
# ---------------------------------------------------------------------------


@requires_gpu
class TestTopHatBlackHatGPU:
    def test_tophat_gpu_matches_cpu(self):
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[2, 2] = 1  # small bright feature
        se = make_structuring_element("rect", 5)
        raster = from_numpy(data)

        gpu_result = raster_morphology_tophat(raster, se, use_gpu=True).to_numpy()
        cpu_result = raster_morphology_tophat(raster, se, use_gpu=False).to_numpy()
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_blackhat_gpu_matches_cpu(self):
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        data[9, 9] = 0  # small hole
        se = make_structuring_element("rect", 3)
        raster = from_numpy(data)

        gpu_result = raster_morphology_blackhat(raster, se, use_gpu=True).to_numpy()
        cpu_result = raster_morphology_blackhat(raster, se, use_gpu=False).to_numpy()
        np.testing.assert_array_equal(gpu_result, cpu_result)


# ---------------------------------------------------------------------------
# Dispatcher / integration tests
# ---------------------------------------------------------------------------


class TestDispatcherNxN:
    def test_auto_dispatch_with_custom_se(self):
        """Auto-dispatch should work with custom SE."""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[3:7, 3:7] = 1
        se = make_structuring_element("rect", 5)
        raster = from_numpy(data)
        result = raster_morphology(raster, "erode", structuring_element=se)
        assert result is not None

    def test_tophat_auto_dispatch(self):
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1
        raster = from_numpy(data)
        result = raster_morphology_tophat(raster)
        assert result is not None

    def test_blackhat_auto_dispatch(self):
        data = np.ones((10, 10), dtype=np.uint8)
        data[5, 5] = 0
        raster = from_numpy(data)
        result = raster_morphology_blackhat(raster)
        assert result is not None
