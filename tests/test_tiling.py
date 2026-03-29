"""Tests for the Phase 1 tiling execution engine (vibeSpatial-fx3.2).

Covers:
- WHOLE fast-path (op_fn called directly, no tiling overhead)
- TILED unary: small raster with small tile size, correct stitching
- TILED binary: same for binary operations
- Edge tile handling: raster not divisible by tile size
- Multiband tiling: 3D (bands, H, W) rasters tile on spatial dims
- Affine adjustment: each tile gets correct shifted affine
- Nodata propagation: nodata sentinel preserved through tiling
- Identity operation: tiling with identity op_fn produces exact input
- Metadata preservation: CRS, nodata, dtype preserved through tiling
- Diagnostic events: RUNTIME events appended for tiled dispatches
- Malformed plan: TILED with tile_shape=None raises ValueError
- Binary spatial mismatch: a and b with different shapes raises ValueError

All tests use explicit RasterPlan construction -- no GPU dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticKind,
    RasterPlan,
    TilingStrategy,
    from_numpy,
)
from vibespatial.raster.tiling import (
    _adjust_affine,
    _tile_bounds,
    dispatch_tiled,
    dispatch_tiled_binary,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_TEST_AFFINE: tuple[float, float, float, float, float, float] = (
    10.0,
    0.0,
    500.0,
    0.0,
    -10.0,
    1000.0,
)
"""(a=10, b=0, c=500, d=0, e=-10, f=1000): 10m resolution, north-up."""

_WHOLE_PLAN = RasterPlan(
    strategy=TilingStrategy.WHOLE,
    tile_shape=None,
    halo=0,
    n_tiles=0,
    estimated_vram_per_tile=0,
)

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_raster(
    height: int = 64,
    width: int = 64,
    *,
    bands: int = 1,
    dtype: np.dtype | type = np.float32,
    nodata: float | int | None = None,
    affine: tuple[float, float, float, float, float, float] = _TEST_AFFINE,
    fill: float | None = None,
) -> OwnedRasterArray:
    """Create a synthetic HOST-resident OwnedRasterArray for testing."""
    dtype = np.dtype(dtype)
    shape = (bands, height, width) if bands > 1 else (height, width)
    if fill is not None:
        data = np.full(shape, fill, dtype=dtype)
    else:
        if np.issubdtype(dtype, np.floating):
            data = _RNG.random(shape).astype(dtype)
        else:
            data = _RNG.integers(0, 255, size=shape, dtype=dtype)
    return from_numpy(data, nodata=nodata, affine=affine, crs=None)


def _make_tiled_plan(
    tile_h: int = 32,
    tile_w: int = 32,
) -> RasterPlan:
    """Create a TILED RasterPlan with given tile dimensions."""
    return RasterPlan(
        strategy=TilingStrategy.TILED,
        tile_shape=(tile_h, tile_w),
        halo=0,
        n_tiles=0,  # n_tiles is informational; tiling computes it from shape
        estimated_vram_per_tile=0,
    )


# ---------------------------------------------------------------------------
# _tile_bounds helper
# ---------------------------------------------------------------------------


class TestTileBounds:
    def test_first_tile(self):
        rs, re, cs, ce = _tile_bounds(0, 0, 32, 32, 64, 64)
        assert (rs, re, cs, ce) == (0, 32, 0, 32)

    def test_last_tile_exact(self):
        """Raster exactly divisible by tile size."""
        rs, re, cs, ce = _tile_bounds(1, 1, 32, 32, 64, 64)
        assert (rs, re, cs, ce) == (32, 64, 32, 64)

    def test_last_tile_clamped(self):
        """Raster not divisible by tile: last tile is smaller."""
        rs, re, cs, ce = _tile_bounds(1, 1, 32, 32, 50, 50)
        assert (rs, re) == (32, 50)
        assert (cs, ce) == (32, 50)

    def test_single_tile_covers_raster(self):
        rs, re, cs, ce = _tile_bounds(0, 0, 100, 100, 50, 50)
        assert (rs, re, cs, ce) == (0, 50, 0, 50)

    def test_asymmetric_tiles(self):
        rs, re, cs, ce = _tile_bounds(0, 1, 16, 24, 64, 64)
        assert (rs, re) == (0, 16)
        assert (cs, ce) == (24, 48)


# ---------------------------------------------------------------------------
# _adjust_affine helper
# ---------------------------------------------------------------------------


class TestAdjustAffine:
    def test_no_offset(self):
        result = _adjust_affine(_TEST_AFFINE, 0, 0)
        assert result == _TEST_AFFINE

    def test_col_offset(self):
        """Shifting by col_offset adjusts x-origin by col_offset * a."""
        a, b, c, d, e, f = _TEST_AFFINE
        result = _adjust_affine(_TEST_AFFINE, row_offset=0, col_offset=10)
        expected_c = c + 10 * a
        assert result == (a, b, expected_c, d, e, f)

    def test_row_offset(self):
        """Shifting by row_offset adjusts y-origin by row_offset * e."""
        a, b, c, d, e, f = _TEST_AFFINE
        result = _adjust_affine(_TEST_AFFINE, row_offset=5, col_offset=0)
        expected_f = f + 5 * e
        assert result == (a, b, c, d, e, expected_f)

    def test_both_offsets(self):
        a, b, c, d, e, f = _TEST_AFFINE
        result = _adjust_affine(_TEST_AFFINE, row_offset=3, col_offset=7)
        expected_c = c + 7 * a + 3 * b
        expected_f = f + 7 * d + 3 * e
        assert result == (a, b, expected_c, d, e, expected_f)

    def test_rotated_affine(self):
        """Non-zero b and d (rotated grid) are handled correctly."""
        rotated = (10.0, 2.0, 500.0, 3.0, -10.0, 1000.0)
        a, b, c, d, e, f = rotated
        result = _adjust_affine(rotated, row_offset=4, col_offset=6)
        expected_c = c + 6 * a + 4 * b
        expected_f = f + 6 * d + 4 * e
        assert result == (a, b, expected_c, d, e, expected_f)


# ---------------------------------------------------------------------------
# dispatch_tiled — WHOLE fast path
# ---------------------------------------------------------------------------


class TestDispatchTiledWhole:
    def test_whole_calls_op_fn_directly(self):
        """WHOLE plan passes the raster through op_fn without tiling."""
        raster = _make_raster(64, 64)
        calls: list[OwnedRasterArray] = []

        def spy_op(r: OwnedRasterArray) -> OwnedRasterArray:
            calls.append(r)
            return r

        result = dispatch_tiled(raster, spy_op, _WHOLE_PLAN)
        assert len(calls) == 1
        assert calls[0] is raster
        assert result is raster

    def test_whole_returns_op_fn_result(self):
        raster = _make_raster(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, double_op, _WHOLE_PLAN)
        np.testing.assert_allclose(result.to_numpy(), raster.to_numpy() * 2)


# ---------------------------------------------------------------------------
# dispatch_tiled — TILED unary
# ---------------------------------------------------------------------------


class TestDispatchTiledUnary:
    def test_identity_op_exact_reconstruction(self):
        """Tiling an identity function reconstructs the original raster."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def identity(r: OwnedRasterArray) -> OwnedRasterArray:
            return r

        result = dispatch_tiled(raster, identity, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_double_op_tiled(self):
        """Doubling values through tiling matches direct doubling."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, double_op, plan)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_tiled_vs_whole_match(self):
        """Tiled and whole paths produce identical results for pointwise op."""
        raster = _make_raster(64, 64, dtype=np.float64)

        def negate_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(-r.to_numpy(), nodata=r.nodata, affine=r.affine, crs=r.crs)

        result_whole = dispatch_tiled(raster, negate_op, _WHOLE_PLAN)
        result_tiled = dispatch_tiled(raster, negate_op, _make_tiled_plan(16, 16))
        np.testing.assert_array_equal(result_tiled.to_numpy(), result_whole.to_numpy())

    def test_tile_count_matches_calls(self):
        """Op function is called exactly once per tile."""
        raster = _make_raster(64, 48)  # 64/32=2 rows, 48/32=2 cols -> 4 tiles
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def counting_op(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return r

        dispatch_tiled(raster, counting_op, plan)
        assert call_count == 4  # 2 x 2 tile grid

    def test_single_pixel_raster(self):
        """1x1 raster with tiling still works."""
        raster = _make_raster(1, 1, dtype=np.float32, fill=42.0)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.to_numpy().item() == pytest.approx(42.0)

    def test_very_small_tiles(self):
        """1x1 tiles process every pixel independently."""
        raster = _make_raster(4, 4, dtype=np.float32)
        plan = _make_tiled_plan(1, 1)
        call_count = 0

        def counting_identity(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return r

        result = dispatch_tiled(raster, counting_identity, plan)
        assert call_count == 16  # 4 x 4
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())


# ---------------------------------------------------------------------------
# Edge tile handling
# ---------------------------------------------------------------------------


class TestEdgeTiles:
    def test_raster_not_divisible_by_tile(self):
        """50x50 raster with 32x32 tiles: edge tiles are 18 pixels wide/tall."""
        raster = _make_raster(50, 50, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)
        tile_shapes: list[tuple[int, ...]] = []

        def shape_spy(r: OwnedRasterArray) -> OwnedRasterArray:
            tile_shapes.append(r.shape)
            return r

        result = dispatch_tiled(raster, shape_spy, plan)

        # 2x2 tile grid
        assert len(tile_shapes) == 4
        assert tile_shapes[0] == (32, 32)  # top-left
        assert tile_shapes[1] == (32, 18)  # top-right (50-32=18)
        assert tile_shapes[2] == (18, 32)  # bottom-left
        assert tile_shapes[3] == (18, 18)  # bottom-right

        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_raster_smaller_than_tile(self):
        """Raster smaller than tile -> one tile covers everything."""
        raster = _make_raster(10, 15, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def counting_identity(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            assert r.height == 10
            assert r.width == 15
            return r

        result = dispatch_tiled(raster, counting_identity, plan)
        assert call_count == 1
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_asymmetric_tile_and_raster(self):
        """Non-square raster with non-square tiles."""
        raster = _make_raster(100, 60, dtype=np.int16)
        plan = _make_tiled_plan(40, 25)  # 3 row tiles, 3 col tiles (100/40, 60/25)
        call_count = 0

        def counting_identity(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return r

        result = dispatch_tiled(raster, counting_identity, plan)
        # rows: ceil(100/40)=3, cols: ceil(60/25)=3 -> 9 tiles
        assert call_count == 9
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())


# ---------------------------------------------------------------------------
# Multiband tiling
# ---------------------------------------------------------------------------


class TestMultibandTiling:
    def test_3band_identity(self):
        """3-band raster tiled correctly on spatial dims."""
        raster = _make_raster(64, 64, bands=3, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())
        assert result.band_count == 3
        assert result.shape == (3, 64, 64)

    def test_3band_edge_tiles(self):
        """Multiband raster with edge tiles."""
        raster = _make_raster(50, 50, bands=3, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, double_op, plan)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.shape == (3, 50, 50)

    def test_tile_slices_all_bands(self):
        """Each tile receives all bands for its spatial extent."""
        raster = _make_raster(32, 32, bands=4, dtype=np.uint8)
        plan = _make_tiled_plan(16, 16)

        def check_bands(r: OwnedRasterArray) -> OwnedRasterArray:
            assert r.band_count == 4
            assert r.to_numpy().ndim == 3
            assert r.to_numpy().shape[0] == 4
            return r

        dispatch_tiled(raster, check_bands, plan)


# ---------------------------------------------------------------------------
# Affine adjustment per tile
# ---------------------------------------------------------------------------


class TestAffinePerTile:
    def test_tile_affine_is_shifted(self):
        """Each tile receives an affine shifted to its spatial position."""
        raster = _make_raster(
            64,
            64,
            affine=(10.0, 0.0, 500.0, 0.0, -10.0, 1000.0),
        )
        plan = _make_tiled_plan(32, 32)
        tile_affines: list[tuple[float, ...]] = []

        def affine_spy(r: OwnedRasterArray) -> OwnedRasterArray:
            tile_affines.append(r.affine)
            return r

        dispatch_tiled(raster, affine_spy, plan)

        # 2x2 grid -> 4 tiles
        assert len(tile_affines) == 4

        a, b, c, d, e, f = 10.0, 0.0, 500.0, 0.0, -10.0, 1000.0

        # Tile (0,0): no offset
        assert tile_affines[0] == (a, b, c, d, e, f)
        # Tile (0,1): col_offset=32 -> new_c = 500 + 32*10 = 820
        assert tile_affines[1] == (a, b, c + 32 * a, d, e, f)
        # Tile (1,0): row_offset=32 -> new_f = 1000 + 32*(-10) = 680
        assert tile_affines[2] == (a, b, c, d, e, f + 32 * e)
        # Tile (1,1): both offsets
        assert tile_affines[3] == (a, b, c + 32 * a, d, e, f + 32 * e)

    def test_output_has_original_affine(self):
        """The assembled output raster uses the full (un-shifted) affine."""
        raster = _make_raster(64, 64, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.affine == _TEST_AFFINE


# ---------------------------------------------------------------------------
# Nodata propagation
# ---------------------------------------------------------------------------


class TestNodataPropagation:
    def test_nodata_preserved_through_identity(self):
        """Nodata sentinel is propagated to the output raster."""
        raster = _make_raster(64, 64, dtype=np.float32, nodata=-9999.0)
        # Place nodata in known positions
        data = raster.to_numpy().copy()
        data[0, 0] = -9999.0
        data[31, 31] = -9999.0
        data[63, 63] = -9999.0
        raster = from_numpy(data, nodata=-9999.0, affine=_TEST_AFFINE)

        plan = _make_tiled_plan(32, 32)
        result = dispatch_tiled(raster, lambda r: r, plan)

        assert result.nodata == -9999.0
        result_data = result.to_numpy()
        assert result_data[0, 0] == -9999.0
        assert result_data[31, 31] == -9999.0
        assert result_data[63, 63] == -9999.0

    def test_nodata_none_preserved(self):
        """Raster with nodata=None stays nodata=None after tiling."""
        raster = _make_raster(32, 32, nodata=None)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.nodata is None

    def test_integer_nodata(self):
        """Integer nodata (e.g., 255 for uint8) is preserved."""
        raster = _make_raster(32, 32, dtype=np.uint8, nodata=255)
        data = raster.to_numpy().copy()
        data[0, 0] = 255
        data[15, 15] = 255
        raster = from_numpy(data, nodata=255, affine=_TEST_AFFINE)

        plan = _make_tiled_plan(16, 16)
        result = dispatch_tiled(raster, lambda r: r, plan)

        assert result.nodata == 255
        assert result.to_numpy()[0, 0] == 255
        assert result.to_numpy()[15, 15] == 255

    def test_nan_nodata_float(self):
        """NaN nodata in float rasters is preserved through tiling."""
        data = _RNG.random((32, 32)).astype(np.float32)
        data[5, 5] = np.nan
        data[20, 20] = np.nan
        raster = from_numpy(data, nodata=float("nan"), affine=_TEST_AFFINE)

        plan = _make_tiled_plan(16, 16)
        result = dispatch_tiled(raster, lambda r: r, plan)

        assert result.nodata is not None and np.isnan(result.nodata)
        assert np.isnan(result.to_numpy()[5, 5])
        assert np.isnan(result.to_numpy()[20, 20])


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------


class TestDtypePreservation:
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32, np.float64])
    def test_dtype_roundtrip(self, dtype):
        raster = _make_raster(32, 32, dtype=dtype)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_dtype_changing_op_unary(self):
        """Op that changes dtype (float32 -> uint8) works via lazy allocation."""
        raster = _make_raster(32, 32, dtype=np.float32, fill=128.5)
        plan = _make_tiled_plan(16, 16)

        def to_uint8(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy().astype(np.uint8),
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, to_uint8, plan)
        assert result.dtype == np.dtype(np.uint8)
        np.testing.assert_array_equal(result.to_numpy(), 128)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    def test_affine_preserved(self):
        custom_affine = (5.0, 0.0, 100.0, 0.0, -5.0, 200.0)
        raster = _make_raster(32, 32, affine=custom_affine)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.affine == custom_affine

    def test_nodata_preserved(self):
        raster = _make_raster(32, 32, nodata=-9999.0)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.nodata == -9999.0

    def test_crs_preserved(self):
        """CRS is propagated (None in tests, but slot preserved)."""
        raster = _make_raster(32, 32)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.crs == raster.crs


# ---------------------------------------------------------------------------
# Diagnostic events
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_tiled_unary_has_runtime_event(self):
        raster = _make_raster(64, 64)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        runtime_events = [
            ev for ev in result.diagnostics if ev.kind == RasterDiagnosticKind.RUNTIME
        ]
        assert len(runtime_events) >= 1
        evt = runtime_events[-1]
        assert "dispatch_tiled" in evt.detail
        assert "unary" in evt.detail
        assert "tiles=" in evt.detail

    def test_tiled_binary_has_runtime_event(self):
        a = _make_raster(64, 64)
        b = _make_raster(64, 64)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        runtime_events = [
            ev for ev in result.diagnostics if ev.kind == RasterDiagnosticKind.RUNTIME
        ]
        assert len(runtime_events) >= 1
        evt = runtime_events[-1]
        assert "dispatch_tiled_binary" in evt.detail
        assert "tiles=" in evt.detail

    def test_whole_path_no_tiling_diagnostic(self):
        """WHOLE path delegates to op_fn -- no tiling diagnostic appended."""
        raster = _make_raster(32, 32)

        result = dispatch_tiled(raster, lambda r: r, _WHOLE_PLAN)
        # The WHOLE path returns op_fn's result directly; dispatch_tiled
        # does not wrap it with an extra diagnostic.
        tiling_events = [
            ev
            for ev in result.diagnostics
            if ev.kind == RasterDiagnosticKind.RUNTIME and "dispatch_tiled" in ev.detail
        ]
        assert len(tiling_events) == 0


# ---------------------------------------------------------------------------
# dispatch_tiled_binary — WHOLE fast path
# ---------------------------------------------------------------------------


class TestDispatchTiledBinaryWhole:
    def test_whole_calls_op_fn_directly(self):
        a = _make_raster(32, 32, fill=3.0)
        b = _make_raster(32, 32, fill=7.0)
        calls: list[tuple[OwnedRasterArray, OwnedRasterArray]] = []

        def spy_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            calls.append((x, y))
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, spy_op, _WHOLE_PLAN)
        assert len(calls) == 1
        assert calls[0][0] is a
        assert calls[0][1] is b
        np.testing.assert_allclose(result.to_numpy(), 10.0)


# ---------------------------------------------------------------------------
# dispatch_tiled_binary — TILED path
# ---------------------------------------------------------------------------


class TestDispatchTiledBinary:
    def test_add_tiled(self):
        """Binary add through tiling matches direct addition."""
        a = _make_raster(64, 64, dtype=np.float32)
        b = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result_tiled = dispatch_tiled_binary(a, b, add_op, plan)
        expected = a.to_numpy() + b.to_numpy()
        np.testing.assert_allclose(result_tiled.to_numpy(), expected)

    def test_tiled_vs_whole_binary(self):
        """Tiled and whole binary paths produce identical results."""
        a = _make_raster(64, 64, dtype=np.float64)
        b = _make_raster(64, 64, dtype=np.float64)

        def mul_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() * y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result_whole = dispatch_tiled_binary(a, b, mul_op, _WHOLE_PLAN)
        result_tiled = dispatch_tiled_binary(a, b, mul_op, _make_tiled_plan(16, 16))
        np.testing.assert_array_equal(result_tiled.to_numpy(), result_whole.to_numpy())

    def test_binary_tile_count(self):
        """Binary tiling calls op_fn once per tile."""
        a = _make_raster(48, 64)
        b = _make_raster(48, 64)
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def counting_add(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        dispatch_tiled_binary(a, b, counting_add, plan)
        # ceil(48/32)=2 rows, ceil(64/32)=2 cols -> 4 tiles
        assert call_count == 4

    def test_binary_edge_tiles(self):
        """Binary tiling with non-divisible raster dimensions."""
        a = _make_raster(50, 35, dtype=np.float32)
        b = _make_raster(50, 35, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        expected = a.to_numpy() + b.to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_binary_multiband(self):
        """3-band binary tiling."""
        a = _make_raster(64, 64, bands=3, dtype=np.float32)
        b = _make_raster(64, 64, bands=3, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        expected = a.to_numpy() + b.to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 3

    def test_binary_nodata_propagation(self):
        """Nodata from either input is preserved in binary tiling output."""
        data_a = np.ones((32, 32), dtype=np.float32)
        data_b = np.ones((32, 32), dtype=np.float32) * 2
        data_a[0, 0] = -9999.0
        data_b[15, 15] = -9999.0

        a = from_numpy(data_a, nodata=-9999.0, affine=_TEST_AFFINE)
        b = from_numpy(data_b, nodata=-9999.0, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(16, 16)

        def add_with_nodata(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            xd = x.to_numpy()
            yd = y.to_numpy()
            result = xd + yd
            # Propagate nodata: if either is nodata, output is nodata
            mask = (xd == -9999.0) | (yd == -9999.0)
            result[mask] = -9999.0
            return from_numpy(result, nodata=-9999.0, affine=x.affine, crs=x.crs)

        result = dispatch_tiled_binary(a, b, add_with_nodata, plan)
        rd = result.to_numpy()
        assert rd[0, 0] == -9999.0
        assert rd[15, 15] == -9999.0
        # Non-nodata pixels have correct sum
        assert rd[1, 1] == pytest.approx(3.0)

    def test_binary_metadata_preserved(self):
        """Output preserves a's affine and CRS."""
        custom_affine = (5.0, 0.0, 100.0, 0.0, -5.0, 200.0)
        a = _make_raster(32, 32, affine=custom_affine, nodata=-1.0)
        b = _make_raster(32, 32, affine=custom_affine, nodata=-1.0)
        plan = _make_tiled_plan(16, 16)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        assert result.affine == custom_affine
        assert result.crs is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_tiled_plan_without_tile_shape_raises(self):
        """TILED strategy with tile_shape=None is a malformed plan."""
        bad_plan = RasterPlan(
            strategy=TilingStrategy.TILED,
            tile_shape=None,
            halo=0,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )
        raster = _make_raster(32, 32)

        with pytest.raises(ValueError, match="tile_shape is None"):
            dispatch_tiled(raster, lambda r: r, bad_plan)

    def test_binary_tiled_plan_without_tile_shape_raises(self):
        bad_plan = RasterPlan(
            strategy=TilingStrategy.TILED,
            tile_shape=None,
            halo=0,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )
        a = _make_raster(32, 32)
        b = _make_raster(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        with pytest.raises(ValueError, match="tile_shape is None"):
            dispatch_tiled_binary(a, b, add_op, bad_plan)

    def test_device_resident_unary_raises(self):
        """DEVICE-resident input on TILED path raises ValueError."""
        from vibespatial.residency import Residency

        raster = _make_raster(32, 32)
        raster.residency = Residency.DEVICE  # simulate device-resident
        plan = _make_tiled_plan(16, 16)

        with pytest.raises(ValueError, match="HOST-resident"):
            dispatch_tiled(raster, lambda r: r, plan)

    def test_device_resident_binary_raises(self):
        """DEVICE-resident input on binary TILED path raises ValueError."""
        from vibespatial.residency import Residency

        a = _make_raster(32, 32)
        b = _make_raster(32, 32)
        a.residency = Residency.DEVICE  # simulate device-resident
        plan = _make_tiled_plan(16, 16)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        with pytest.raises(ValueError, match="HOST-resident"):
            dispatch_tiled_binary(a, b, add_op, plan)

    def test_binary_spatial_mismatch_raises(self):
        """Mismatched spatial dimensions raise ValueError."""
        a = _make_raster(32, 32)
        b = _make_raster(32, 64)  # different width
        plan = _make_tiled_plan(16, 16)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        with pytest.raises(ValueError, match="Spatial dimension mismatch"):
            dispatch_tiled_binary(a, b, add_op, plan)


# ---------------------------------------------------------------------------
# Constant-value and all-nodata edge cases
# ---------------------------------------------------------------------------


class TestConstantAndAllNodata:
    def test_constant_value_raster(self):
        raster = _make_raster(64, 64, fill=42.0, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), 42.0)

    def test_all_nodata_raster(self):
        """Raster where every pixel is nodata."""
        data = np.full((32, 32), -9999.0, dtype=np.float32)
        raster = from_numpy(data, nodata=-9999.0, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.nodata == -9999.0
        np.testing.assert_array_equal(result.to_numpy(), -9999.0)


# ---------------------------------------------------------------------------
# Lazy import via __init__.py
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_dispatch_tiled_importable(self):
        from vibespatial.raster import dispatch_tiled as dt

        assert callable(dt)

    def test_dispatch_tiled_binary_importable(self):
        from vibespatial.raster import dispatch_tiled_binary as dtb

        assert callable(dtb)
