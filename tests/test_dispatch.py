"""Tests for VRAM budget functions and band dispatch executors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticKind,
    from_numpy,
)
from vibespatial.raster.dispatch import (
    _VRAM_HEADROOM_FRACTION,
    available_vram_bytes,
    dispatch_per_band_cpu,
    dispatch_per_band_gpu,
    max_bands_for_budget,
)

try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# ---------------------------------------------------------------------------
# available_vram_bytes
# ---------------------------------------------------------------------------


class TestAvailableVramBytes:
    def test_returns_int(self):
        result = available_vram_bytes()
        assert isinstance(result, int)

    def test_returns_non_negative(self):
        assert available_vram_bytes() >= 0

    def test_returns_zero_when_cupy_unavailable(self):
        """Simulate CuPy not installed by making the import raise."""
        with patch.dict("sys.modules", {"cupy": None}):
            # Force re-import failure inside the function
            with patch(
                "builtins.__import__",
                side_effect=_make_import_blocker("cupy"),
            ):
                assert available_vram_bytes() == 0

    @gpu
    def test_returns_positive_with_gpu(self):
        result = available_vram_bytes()
        assert result > 0, "Expected positive VRAM on a machine with a GPU"

    @gpu
    def test_headroom_applied(self):
        """Verify the returned value is strictly less than raw free + pool."""
        import cupy as cp

        free, _ = cp.cuda.runtime.memGetInfo()
        pool_free = cp.get_default_memory_pool().free_bytes()
        raw_total = free + pool_free
        result = available_vram_bytes()
        # Result should be at most (1 - headroom) * raw_total
        assert result <= int(raw_total * (1.0 - _VRAM_HEADROOM_FRACTION))


# ---------------------------------------------------------------------------
# max_bands_for_budget — deterministic tests via mocking
# ---------------------------------------------------------------------------


class TestMaxBandsForBudget:
    """Use mocked available_vram_bytes to get deterministic answers."""

    def test_basic_float32(self):
        # 1000x1000 float32, 2 buffers => 8 MB per band
        # 80 MB budget => 10 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32)
            assert result == 10

    def test_basic_uint8(self):
        # 1000x1000 uint8, 2 buffers => 2 MB per band
        # 80 MB budget => 40 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.uint8)
            assert result == 40

    def test_basic_float64(self):
        # 1000x1000 float64, 2 buffers => 16 MB per band
        # 80 MB budget => 5 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float64)
            assert result == 5

    def test_scratch_bytes_subtracted(self):
        # 80 MB total, 30 MB scratch => 50 MB usable
        # 1000x1000 float32 x 2 buffers = 8 MB/band => 6 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32, scratch_bytes=30_000_000)
            assert result == 6

    def test_scratch_exceeds_available_returns_one(self):
        """When scratch_bytes > available VRAM, must still return >= 1."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=1_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32, scratch_bytes=999_000_000)
            assert result == 1

    def test_zero_vram_returns_one(self):
        """No GPU memory at all => still return 1 for CPU fallback."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=0,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32)
            assert result == 1

    def test_custom_buffers_per_band(self):
        # 1000x1000 float32, 4 buffers => 16 MB per band
        # 80 MB budget => 5 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32, buffers_per_band=4)
            assert result == 5

    def test_small_raster_many_bands(self):
        # 10x10 float32, 2 buffers => 800 bytes per band
        # 80 MB budget => 100_000 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(10, 10, np.float32)
            assert result == 100_000

    def test_accepts_dtype_instance_and_type(self):
        """Both np.float32 (type) and np.dtype(np.float32) (instance) work."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result_type = max_bands_for_budget(1000, 1000, np.float32)
            result_inst = max_bands_for_budget(1000, 1000, np.dtype(np.float32))
            assert result_type == result_inst

    def test_int16_dtype(self):
        # 1000x1000 int16, 2 buffers => 4 MB per band
        # 80 MB budget => 20 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.int16)
            assert result == 20


# ---------------------------------------------------------------------------
# dispatch_per_band_cpu — CPU band dispatch
# ---------------------------------------------------------------------------

# Shared test fixtures (inline, no conftest.py)

_TEST_AFFINE = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
_TEST_NODATA = -9999.0


def _make_single_band_raster(
    height: int = 8,
    width: int = 10,
    dtype: np.dtype = np.float32,
    nodata: float | int | None = _TEST_NODATA,
    seed: int = 42,
) -> OwnedRasterArray:
    rng = np.random.default_rng(seed)
    data = rng.random((height, width)).astype(dtype)
    return from_numpy(data, nodata=nodata, affine=_TEST_AFFINE, crs=None)


def _make_multiband_raster(
    bands: int = 3,
    height: int = 8,
    width: int = 10,
    dtype: np.dtype = np.float32,
    nodata: float | int | None = _TEST_NODATA,
    seed: int = 42,
) -> OwnedRasterArray:
    rng = np.random.default_rng(seed)
    data = rng.random((bands, height, width)).astype(dtype)
    return from_numpy(data, nodata=nodata, affine=_TEST_AFFINE, crs=None)


def _double_op(raster: OwnedRasterArray) -> OwnedRasterArray:
    """Simple test op: multiply pixel values by 2."""
    return from_numpy(
        raster.to_numpy() * 2,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )


class TestDispatchPerBandCpuSingleBand:
    def test_single_band_passthrough(self):
        """Single-band raster: op_fn called once, result returned directly."""
        raster = _make_single_band_raster()
        result = dispatch_per_band_cpu(raster, _double_op)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 1

    def test_single_band_call_count(self):
        """op_fn called exactly once for a single-band raster."""
        raster = _make_single_band_raster()
        mock_op = MagicMock(side_effect=_double_op)
        dispatch_per_band_cpu(raster, mock_op)
        assert mock_op.call_count == 1

    def test_single_band_diagnostic_event(self):
        """Single-band passthrough appends a RUNTIME diagnostic event."""
        raster = _make_single_band_raster()
        result = dispatch_per_band_cpu(raster, _double_op)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "dispatch_per_band_cpu" in runtime_events[-1].detail
        assert "single-band" in runtime_events[-1].detail


class TestDispatchPerBandCpuMultiband:
    def test_multiband_3_bands(self):
        """All 3 bands processed correctly, output shape (3, H, W)."""
        raster = _make_multiband_raster(bands=3)
        result = dispatch_per_band_cpu(raster, _double_op)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 3
        assert result.to_numpy().shape == (3, 8, 10)

    def test_multiband_call_count(self):
        """op_fn called exactly N times for N bands."""
        n_bands = 5
        raster = _make_multiband_raster(bands=n_bands)
        mock_op = MagicMock(side_effect=_double_op)
        dispatch_per_band_cpu(raster, mock_op)
        assert mock_op.call_count == n_bands

    def test_multiband_preserves_metadata(self):
        """Affine, CRS, nodata, and dtype propagated to result."""
        raster = _make_multiband_raster(bands=3, dtype=np.float32)
        result = dispatch_per_band_cpu(raster, _double_op)
        assert result.affine == _TEST_AFFINE
        assert result.crs is None
        assert result.nodata == _TEST_NODATA
        assert result.dtype == np.float32

    def test_multiband_diagnostic_event(self):
        """Multiband dispatch appends a RUNTIME diagnostic with band count."""
        raster = _make_multiband_raster(bands=3)
        result = dispatch_per_band_cpu(raster, _double_op)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        last = runtime_events[-1]
        assert "dispatch_per_band_cpu" in last.detail
        assert "bands=3" in last.detail

    def test_multiband_2_bands(self):
        """Edge case: 2-band raster."""
        raster = _make_multiband_raster(bands=2)
        result = dispatch_per_band_cpu(raster, _double_op)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 2

    def test_multiband_nodata_preserved(self):
        """Nodata pixels remain nodata after per-band dispatch."""
        raster = _make_multiband_raster(bands=2)
        # Inject nodata into known positions
        data = raster.to_numpy().copy()
        data[0, 0, 0] = _TEST_NODATA
        data[1, 3, 4] = _TEST_NODATA
        raster_with_nd = from_numpy(data, nodata=_TEST_NODATA, affine=_TEST_AFFINE, crs=None)

        def identity_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(r.to_numpy().copy(), nodata=r.nodata, affine=r.affine, crs=r.crs)

        result = dispatch_per_band_cpu(raster_with_nd, identity_op)
        result_data = result.to_numpy()
        assert result_data[0, 0, 0] == _TEST_NODATA
        assert result_data[1, 3, 4] == _TEST_NODATA


# ---------------------------------------------------------------------------
# dispatch_per_band_gpu — GPU band dispatch (CPU-backed tests where possible)
# ---------------------------------------------------------------------------


class TestDispatchPerBandGpuSingleBand:
    @gpu
    def test_single_band_passthrough(self):
        """Single-band raster: op_fn called once on GPU path."""
        raster = _make_single_band_raster()
        result = dispatch_per_band_gpu(raster, _double_op)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 1

    @gpu
    def test_single_band_call_count(self):
        """op_fn called exactly once for a single-band raster."""
        raster = _make_single_band_raster()
        mock_op = MagicMock(side_effect=_double_op)
        dispatch_per_band_gpu(raster, mock_op)
        assert mock_op.call_count == 1

    @gpu
    def test_single_band_diagnostic_event(self):
        """Single-band GPU passthrough appends a RUNTIME diagnostic."""
        raster = _make_single_band_raster()
        result = dispatch_per_band_gpu(raster, _double_op)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "dispatch_per_band_gpu" in runtime_events[-1].detail
        assert "single-band" in runtime_events[-1].detail


class TestDispatchPerBandGpuMultiband:
    @gpu
    def test_multiband_3_bands(self):
        """All 3 bands processed, correct output shape (3, H, W)."""
        raster = _make_multiband_raster(bands=3)
        result = dispatch_per_band_gpu(raster, _double_op)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 3
        assert result.to_numpy().shape == (3, 8, 10)

    @gpu
    def test_multiband_call_count(self):
        """op_fn called exactly N times for N bands."""
        n_bands = 4
        raster = _make_multiband_raster(bands=n_bands)
        mock_op = MagicMock(side_effect=_double_op)
        dispatch_per_band_gpu(raster, mock_op)
        assert mock_op.call_count == n_bands

    @gpu
    def test_multiband_preserves_metadata(self):
        """Affine, CRS, nodata, and dtype propagated to GPU result."""
        raster = _make_multiband_raster(bands=3, dtype=np.float32)
        result = dispatch_per_band_gpu(raster, _double_op)
        assert result.affine == _TEST_AFFINE
        assert result.crs is None
        assert result.nodata == _TEST_NODATA
        assert result.dtype == np.float32

    @gpu
    def test_multiband_diagnostic_event(self):
        """Multiband GPU dispatch appends a RUNTIME diagnostic."""
        raster = _make_multiband_raster(bands=3)
        result = dispatch_per_band_gpu(raster, _double_op)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        last = runtime_events[-1]
        assert "dispatch_per_band_gpu" in last.detail
        assert "bands=3" in last.detail

    @gpu
    def test_gpu_cpu_parity(self):
        """GPU and CPU per-band dispatch produce the same result."""
        raster = _make_multiband_raster(bands=3)
        result_gpu = dispatch_per_band_gpu(raster, _double_op)
        result_cpu = dispatch_per_band_cpu(raster, _double_op)
        np.testing.assert_allclose(result_gpu.to_numpy(), result_cpu.to_numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_import_blocker(blocked_name: str):
    """Return a side_effect for builtins.__import__ that blocks *blocked_name*."""
    import builtins

    _real_import = builtins.__import__

    def _blocker(name, *args, **kwargs):
        if name == blocked_name or name.startswith(blocked_name + "."):
            raise ImportError(f"mocked: {name} is not installed")
        return _real_import(name, *args, **kwargs)

    return _blocker
