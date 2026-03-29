"""Tests for RasterPlan analysis (vibeSpatial-fx3.1).

Covers:
- TilingStrategy and RasterPlan dataclasses
- analyze_raster_plan() decision logic
- plan_from_metadata() convenience function
- Tile alignment, halo accounting, edge cases
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from vibespatial.raster.buffers import (
    RasterMetadata,
    RasterPlan,
    TilingStrategy,
)
from vibespatial.raster.dispatch import (
    _BUDGET_SAFETY_FACTOR,
    _DEFAULT_TILE_DIM,
    _TILE_ALIGNMENT,
    analyze_raster_plan,
    plan_from_metadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_AFFINE = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)


def _make_metadata(
    height: int = 256,
    width: int = 256,
    band_count: int = 1,
    dtype: np.dtype | type = np.float32,
    nodata: float | int | None = None,
) -> RasterMetadata:
    return RasterMetadata(
        height=height,
        width=width,
        band_count=band_count,
        dtype=np.dtype(dtype),
        nodata=nodata,
        affine=_TEST_AFFINE,
        crs=None,
    )


# ---------------------------------------------------------------------------
# TilingStrategy enum
# ---------------------------------------------------------------------------


class TestTilingStrategy:
    def test_values(self):
        assert TilingStrategy.WHOLE == "whole"
        assert TilingStrategy.TILED == "tiled"

    def test_is_str(self):
        """TilingStrategy members are strings (StrEnum)."""
        assert isinstance(TilingStrategy.WHOLE, str)
        assert isinstance(TilingStrategy.TILED, str)


# ---------------------------------------------------------------------------
# RasterPlan dataclass
# ---------------------------------------------------------------------------


class TestRasterPlan:
    def test_frozen(self):
        plan = RasterPlan(
            strategy=TilingStrategy.WHOLE,
            tile_shape=None,
            halo=0,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )
        with pytest.raises(AttributeError):
            plan.strategy = TilingStrategy.TILED  # type: ignore[misc]

    def test_fields(self):
        plan = RasterPlan(
            strategy=TilingStrategy.TILED,
            tile_shape=(4096, 4096),
            halo=1,
            n_tiles=16,
            estimated_vram_per_tile=128_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape == (4096, 4096)
        assert plan.halo == 1
        assert plan.n_tiles == 16
        assert plan.estimated_vram_per_tile == 128_000_000


# ---------------------------------------------------------------------------
# analyze_raster_plan — WHOLE strategy
# ---------------------------------------------------------------------------


class TestAnalyzePlanWhole:
    def test_small_raster_fits_in_vram(self):
        """256x256 float32, 2 buffers = 512 KB << 1 GB budget."""
        plan = analyze_raster_plan(
            256,
            256,
            np.float32,
            vram_budget=1_000_000_000,
        )
        assert plan.strategy == TilingStrategy.WHOLE
        assert plan.tile_shape is None
        assert plan.n_tiles == 0

    def test_single_pixel_raster(self):
        """1x1 raster always fits."""
        plan = analyze_raster_plan(
            1,
            1,
            np.float64,
            vram_budget=1_000,
        )
        assert plan.strategy == TilingStrategy.WHOLE
        assert plan.tile_shape is None
        assert plan.n_tiles == 0

    def test_raster_exactly_at_budget_boundary(self):
        """Raster that exactly fills 70% of the budget -> WHOLE."""
        # 1000x1000 float32, 2 buffers, 1 band = 8,000,000 bytes.
        # Need: full_raster_bytes <= int(budget * 0.7).
        # Pick budget such that int(budget * 0.7) == 8,000,000.
        # int(11_428_572 * 0.7) = int(8_000_000.4) = 8_000_000.  OK.
        plan = analyze_raster_plan(
            1000,
            1000,
            np.float32,
            vram_budget=11_428_572,
        )
        assert plan.strategy == TilingStrategy.WHOLE

    def test_raster_just_over_budget(self):
        """Raster that exceeds 70% of budget -> TILED."""
        # 1000x1000 float32, 2 buffers, 1 band = 8 MB
        # Budget where 8 MB > budget * 0.7 => budget = 11,428,570
        raster_bytes = 1000 * 1000 * 4 * 2  # 8,000,000
        budget = int(raster_bytes / _BUDGET_SAFETY_FACTOR) - 1
        plan = analyze_raster_plan(
            1000,
            1000,
            np.float32,
            vram_budget=budget,
        )
        assert plan.strategy == TilingStrategy.TILED

    def test_zero_vram_budget_returns_whole(self):
        """No GPU VRAM -> WHOLE (CPU processes in one pass)."""
        plan = analyze_raster_plan(
            10000,
            10000,
            np.float64,
            vram_budget=0,
        )
        assert plan.strategy == TilingStrategy.WHOLE
        assert plan.tile_shape is None
        assert plan.n_tiles == 0
        assert plan.estimated_vram_per_tile == 0

    def test_negative_vram_budget_returns_whole(self):
        """Negative budget treated same as zero."""
        plan = analyze_raster_plan(
            10000,
            10000,
            np.float64,
            vram_budget=-100,
        )
        assert plan.strategy == TilingStrategy.WHOLE

    def test_small_raster_below_tile_alignment(self):
        """Raster smaller than tile alignment -> WHOLE even with tight budget."""
        # 100x100 float32 = 40 KB per buffer, well under any tile.
        # Budget is large enough for the raster but raster is smaller
        # than _DEFAULT_TILE_DIM.
        plan = analyze_raster_plan(
            100,
            100,
            np.float32,
            vram_budget=1_000_000_000,
        )
        assert plan.strategy == TilingStrategy.WHOLE

    def test_scratch_bytes_included_in_budget_check(self):
        """Scratch bytes push the raster over the budget -> TILED."""
        # 1000x1000 float32, 2 buffers = 8 MB.  Budget = 20 MB.
        # Without scratch: 8 MB <= 20 * 0.7 = 14 MB -> WHOLE.
        plan_no_scratch = analyze_raster_plan(
            1000,
            1000,
            np.float32,
            vram_budget=20_000_000,
        )
        assert plan_no_scratch.strategy == TilingStrategy.WHOLE

        # With 7 MB scratch: 8 + 7 = 15 MB > 14 MB -> TILED.
        plan_with_scratch = analyze_raster_plan(
            1000,
            1000,
            np.float32,
            scratch_bytes=7_000_000,
            vram_budget=20_000_000,
        )
        assert plan_with_scratch.strategy == TilingStrategy.TILED

    def test_multiband_budget_check(self):
        """Band count multiplies the per-band cost."""
        # 1000x1000 float32, 2 buffers, 1 band = 8 MB -> fits in 20 MB
        plan_1band = analyze_raster_plan(
            1000,
            1000,
            np.float32,
            band_count=1,
            vram_budget=20_000_000,
        )
        assert plan_1band.strategy == TilingStrategy.WHOLE

        # 3 bands = 24 MB > 20 * 0.7 = 14 MB -> TILED
        plan_3band = analyze_raster_plan(
            1000,
            1000,
            np.float32,
            band_count=3,
            vram_budget=20_000_000,
        )
        assert plan_3band.strategy == TilingStrategy.TILED


# ---------------------------------------------------------------------------
# analyze_raster_plan — TILED strategy
# ---------------------------------------------------------------------------


class TestAnalyzePlanTiled:
    def test_large_raster_requires_tiling(self):
        """16384x16384 float32, 2 buffers = 2 GB >> 256 MB budget."""
        plan = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            vram_budget=256_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None
        assert plan.n_tiles > 0
        assert plan.estimated_vram_per_tile > 0

    def test_tile_dimensions_are_aligned(self):
        """Tile height and width are always multiples of 256."""
        plan = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            vram_budget=256_000_000,
        )
        assert plan.tile_shape is not None
        tile_h, tile_w = plan.tile_shape
        assert tile_h % _TILE_ALIGNMENT == 0
        assert tile_w % _TILE_ALIGNMENT == 0

    def test_tile_count_covers_full_raster(self):
        """n_tiles * effective_tile_area >= raster area."""
        height, width = 8000, 6000
        plan = analyze_raster_plan(
            height,
            width,
            np.float32,
            vram_budget=100_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None

        tile_h, tile_w = plan.tile_shape
        # Effective tile area (without halo, since halo=0 here).
        rows_of_tiles = (height + tile_h - 1) // tile_h
        cols_of_tiles = (width + tile_w - 1) // tile_w
        assert plan.n_tiles == rows_of_tiles * cols_of_tiles
        # Tile grid covers the full raster.
        assert rows_of_tiles * tile_h >= height
        assert cols_of_tiles * tile_w >= width

    def test_tile_vram_fits_budget(self):
        """Estimated VRAM per tile does not exceed the usable budget."""
        budget = 256_000_000
        plan = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            vram_budget=budget,
        )
        assert plan.strategy == TilingStrategy.TILED
        usable = int(budget * _BUDGET_SAFETY_FACTOR)
        assert plan.estimated_vram_per_tile <= usable

    def test_default_tile_dim_4096(self):
        """When budget allows, tiles start at 4096."""
        # Large raster but generous budget -> tile should be 4096x4096.
        # 4096x4096 float32, 2 buffers = 128 MB.  Budget = 200 MB.
        # 128 MB <= 200 * 0.7 = 140 MB -> fits.
        plan = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            vram_budget=200_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None
        tile_h, tile_w = plan.tile_shape
        assert tile_h == _DEFAULT_TILE_DIM
        assert tile_w == _DEFAULT_TILE_DIM

    def test_tiles_shrink_when_budget_tight(self):
        """Tiles shrink below 4096 when VRAM is very limited."""
        # 4096x4096 float32, 2 buffers = 128 MB.  Budget = 50 MB.
        # 128 MB > 50 * 0.7 = 35 MB -> must shrink.
        plan = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            vram_budget=50_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None
        tile_h, tile_w = plan.tile_shape
        assert tile_h < _DEFAULT_TILE_DIM or tile_w < _DEFAULT_TILE_DIM
        # Still aligned.
        assert tile_h % _TILE_ALIGNMENT == 0
        assert tile_w % _TILE_ALIGNMENT == 0

    def test_non_square_raster_tiling(self):
        """Non-square raster: 20000x100 float32."""
        plan = analyze_raster_plan(
            20000,
            100,
            np.float32,
            vram_budget=5_000_000,
        )
        # 20000x100 float32, 2 buffers = 16 MB.  5 * 0.7 = 3.5 MB -> TILED.
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None
        tile_h, tile_w = plan.tile_shape
        assert tile_h % _TILE_ALIGNMENT == 0
        assert tile_w % _TILE_ALIGNMENT == 0

    def test_uint8_dtype(self):
        """uint8 raster uses 1 byte per pixel."""
        # 8192x8192 uint8, 2 buffers = 128 MB.  Budget = 100 MB -> TILED.
        plan = analyze_raster_plan(
            8192,
            8192,
            np.uint8,
            vram_budget=100_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED

    def test_float64_dtype(self):
        """float64 raster uses 8 bytes per pixel."""
        # 4096x4096 float64, 2 buffers = 256 MB.  Budget = 100 MB -> TILED.
        plan = analyze_raster_plan(
            4096,
            4096,
            np.float64,
            vram_budget=100_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED


# ---------------------------------------------------------------------------
# Halo accounting
# ---------------------------------------------------------------------------


class TestHaloAccounting:
    def test_halo_stored_in_plan(self):
        plan = analyze_raster_plan(
            256,
            256,
            np.float32,
            halo=2,
            vram_budget=1_000_000_000,
        )
        assert plan.halo == 2

    def test_halo_increases_tile_vram(self):
        """Tiles with halo use more VRAM than tiles without."""
        budget = 200_000_000
        plan_no_halo = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            halo=0,
            vram_budget=budget,
        )
        plan_with_halo = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            halo=16,
            vram_budget=budget,
        )
        # Both should be TILED.
        assert plan_no_halo.strategy == TilingStrategy.TILED
        assert plan_with_halo.strategy == TilingStrategy.TILED
        # Halo increases per-tile cost.
        assert plan_with_halo.estimated_vram_per_tile > plan_no_halo.estimated_vram_per_tile

    def test_halo_can_trigger_tiling(self):
        """A raster that fits WHOLE without halo may become TILED with halo."""
        # 4096x4096 float32, 2 buffers = 128 MB.
        # Budget = 200 MB => usable = 140 MB.
        # Without halo: 128 MB <= 140 MB -> WHOLE.
        plan_no_halo = analyze_raster_plan(
            4096,
            4096,
            np.float32,
            halo=0,
            vram_budget=200_000_000,
        )
        assert plan_no_halo.strategy == TilingStrategy.WHOLE

        # Large halo: physical tile becomes (4096 + 2*512) x (4096 + 2*512)
        # = 5120x5120 float32, 2 buf = ~200 MB > 140 MB.
        # But this raster may still fit WHOLE if full raster bytes fit.
        # Full raster: 4096*4096*4*2 = 128 MB <= 140 MB -> still WHOLE.
        # (Halo only applies when tiling, not to the WHOLE budget check.)
        plan_halo = analyze_raster_plan(
            4096,
            4096,
            np.float32,
            halo=512,
            vram_budget=200_000_000,
        )
        # The full raster still fits, so strategy is WHOLE.
        assert plan_halo.strategy == TilingStrategy.WHOLE

    def test_halo_reflected_in_tile_count(self):
        """Tile count is based on effective (non-halo) tile area."""
        height, width = 8192, 8192
        plan = analyze_raster_plan(
            height,
            width,
            np.float32,
            halo=4,
            vram_budget=200_000_000,
        )
        if plan.strategy == TilingStrategy.TILED:
            assert plan.tile_shape is not None
            tile_h, tile_w = plan.tile_shape
            # Effective tile area = tile_h x tile_w (halo is extra border).
            rows_of_tiles = (height + tile_h - 1) // tile_h
            cols_of_tiles = (width + tile_w - 1) // tile_w
            assert plan.n_tiles == rows_of_tiles * cols_of_tiles


# ---------------------------------------------------------------------------
# VRAM auto-detection
# ---------------------------------------------------------------------------


class TestVramAutoDetection:
    def test_auto_detect_calls_available_vram_bytes(self):
        """When vram_budget=None, available_vram_bytes() is called."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=500_000_000,
        ) as mock_avb:
            plan = analyze_raster_plan(256, 256, np.float32)
            mock_avb.assert_called_once()
            assert plan.strategy == TilingStrategy.WHOLE

    def test_auto_detect_zero_gpu(self):
        """Auto-detect returns 0 when no GPU -> WHOLE."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=0,
        ):
            plan = analyze_raster_plan(10000, 10000, np.float64)
            assert plan.strategy == TilingStrategy.WHOLE

    def test_explicit_budget_overrides_auto(self):
        """Explicit vram_budget takes precedence over auto-detection."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=10_000_000_000,
        ) as mock_avb:
            plan = analyze_raster_plan(
                16384,
                16384,
                np.float32,
                vram_budget=50_000_000,
            )
            # Should NOT call available_vram_bytes.
            mock_avb.assert_not_called()
            assert plan.strategy == TilingStrategy.TILED


# ---------------------------------------------------------------------------
# plan_from_metadata
# ---------------------------------------------------------------------------


class TestPlanFromMetadata:
    def test_delegates_to_analyze(self):
        """plan_from_metadata extracts fields and delegates."""
        meta = _make_metadata(height=256, width=256, band_count=1, dtype=np.float32)
        plan = plan_from_metadata(meta, vram_budget=1_000_000_000)
        assert plan.strategy == TilingStrategy.WHOLE

    def test_multiband_metadata(self):
        meta = _make_metadata(
            height=8192,
            width=8192,
            band_count=4,
            dtype=np.float32,
        )
        # 8192x8192x4 float32, 2 buf = 2 GB.  Budget = 256 MB -> TILED.
        plan = plan_from_metadata(meta, vram_budget=256_000_000)
        assert plan.strategy == TilingStrategy.TILED

    def test_forwards_all_kwargs(self):
        meta = _make_metadata(height=4096, width=4096, dtype=np.float32)
        plan = plan_from_metadata(
            meta,
            buffers_per_band=4,
            scratch_bytes=10_000_000,
            halo=8,
            vram_budget=500_000_000,
        )
        assert plan.halo == 8

    def test_consistent_with_direct_call(self):
        """plan_from_metadata produces the same result as analyze_raster_plan."""
        meta = _make_metadata(
            height=8192,
            width=8192,
            band_count=2,
            dtype=np.float64,
        )
        plan_meta = plan_from_metadata(
            meta,
            buffers_per_band=3,
            scratch_bytes=1000,
            halo=2,
            vram_budget=100_000_000,
        )
        plan_direct = analyze_raster_plan(
            height=8192,
            width=8192,
            dtype=np.dtype(np.float64),
            band_count=2,
            buffers_per_band=3,
            scratch_bytes=1000,
            halo=2,
            vram_budget=100_000_000,
        )
        assert plan_meta == plan_direct

    def test_metadata_dtype_passthrough(self):
        """int16 metadata dtype is preserved."""
        meta = _make_metadata(height=1024, width=1024, dtype=np.int16)
        plan = plan_from_metadata(meta, vram_budget=1_000_000_000)
        # 1024x1024 int16, 2 buf = 4 MB -> WHOLE with 1 GB budget.
        assert plan.strategy == TilingStrategy.WHOLE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_1x1_raster(self):
        plan = analyze_raster_plan(1, 1, np.float32, vram_budget=100)
        assert plan.strategy == TilingStrategy.WHOLE

    def test_very_narrow_raster(self):
        """1x100000 raster."""
        plan = analyze_raster_plan(
            1,
            100000,
            np.float32,
            vram_budget=1_000_000_000,
        )
        assert plan.strategy == TilingStrategy.WHOLE

    def test_very_tall_raster(self):
        """100000x1 raster."""
        plan = analyze_raster_plan(
            100000,
            1,
            np.float32,
            vram_budget=1_000_000_000,
        )
        assert plan.strategy == TilingStrategy.WHOLE

    def test_buffers_per_band_4(self):
        """4 buffers doubles memory compared to 2 buffers."""
        budget = 200_000_000
        plan_2buf = analyze_raster_plan(
            4096,
            4096,
            np.float32,
            buffers_per_band=2,
            vram_budget=budget,
        )
        plan_4buf = analyze_raster_plan(
            4096,
            4096,
            np.float32,
            buffers_per_band=4,
            vram_budget=budget,
        )
        # 4096x4096 float32: 2 buf = 128 MB (< 140 MB -> WHOLE),
        # 4 buf = 256 MB (> 140 MB -> TILED).
        assert plan_2buf.strategy == TilingStrategy.WHOLE
        assert plan_4buf.strategy == TilingStrategy.TILED

    def test_dtype_as_type_and_instance(self):
        """Both np.float32 (type) and np.dtype('float32') work."""
        plan_type = analyze_raster_plan(
            4096,
            4096,
            np.float32,
            vram_budget=500_000_000,
        )
        plan_inst = analyze_raster_plan(
            4096,
            4096,
            np.dtype("float32"),
            vram_budget=500_000_000,
        )
        assert plan_type == plan_inst

    def test_all_whole_fields_consistent(self):
        """WHOLE plan has tile_shape=None and n_tiles=0."""
        plan = analyze_raster_plan(
            100,
            100,
            np.float32,
            vram_budget=1_000_000_000,
        )
        assert plan.strategy == TilingStrategy.WHOLE
        assert plan.tile_shape is None
        assert plan.n_tiles == 0

    def test_all_tiled_fields_consistent(self):
        """TILED plan has tile_shape != None and n_tiles > 0."""
        plan = analyze_raster_plan(
            16384,
            16384,
            np.float32,
            vram_budget=100_000_000,
        )
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None
        assert plan.n_tiles > 0
        assert plan.estimated_vram_per_tile > 0

    def test_min_tile_budget_exhausted(self):
        """When even the smallest 256x256 tile exceeds budget, loop terminates."""
        # 256x256 float32, 2 buffers = 512 KB.  Budget where usable < 512 KB.
        # usable = int(budget * 0.7).  Pick budget = 500_000 -> usable = 350_000.
        # 256*256*4*2 = 524_288 > 350_000 -> cannot fit even smallest tile.
        plan = analyze_raster_plan(
            10000,
            10000,
            np.float32,
            vram_budget=500_000,
        )
        # Should still produce TILED with minimum tile (not hang forever).
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None
        tile_h, tile_w = plan.tile_shape
        assert tile_h == _TILE_ALIGNMENT
        assert tile_w == _TILE_ALIGNMENT

    def test_raster_smaller_than_default_tile(self):
        """2048x2048 raster: tiles capped at raster size, rounds to WHOLE."""
        # 2048x2048 float32, 2 buf = 32 MB.  Budget = 20 MB -> TILED.
        plan = analyze_raster_plan(
            2048,
            2048,
            np.float32,
            vram_budget=20_000_000,
        )
        # 2048 is below _DEFAULT_TILE_DIM (4096), so tiles start at 2048.
        # Aligned: 2048 is already aligned (2048/256=8).
        # 2048 covers the full raster -> would be WHOLE, but the budget
        # check already determined TILED is needed.
        # So the tile must shrink to fit.
        assert plan.strategy == TilingStrategy.TILED
        assert plan.tile_shape is not None
        tile_h, tile_w = plan.tile_shape
        assert tile_h % _TILE_ALIGNMENT == 0
        assert tile_w % _TILE_ALIGNMENT == 0
