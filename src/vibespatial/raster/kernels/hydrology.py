"""NVRTC kernel sources for GPU sink/depression filling (priority-flood).

Algorithm: iterative parallel flood fill for DEM conditioning.
1. init_fill: Border pixels keep their elevation, interior pixels set to +infinity.
   Nodata pixels set to a sentinel (they are barriers).
2. propagate_fill: Each pixel takes max(own_original_elevation, min(neighbor_fill_levels)).
   Iterates until convergence (no pixel changes).

The kernel is templated on {dtype} (float or double) for native raster dtype support.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 1: Initialize fill surface
# ---------------------------------------------------------------------------
# Border pixels get their own elevation as the initial fill level.
# Interior pixels get +infinity (will be lowered by propagation).
# Nodata pixels get a special sentinel so they act as barriers.

FILL_INIT_SOURCE_TEMPLATE = r"""
extern "C" __global__
void fill_init(
    const {dtype}* __restrict__ elevation,
    {dtype}* __restrict__ fill,
    const unsigned char* __restrict__ nodata_mask,
    const int width,
    const int height
) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;

    /* Nodata pixels are barriers: set fill to -infinity so they never
       participate in min-neighbor propagation (neighbors ignore them). */
    if (nodata_mask != nullptr && nodata_mask[idx]) {{
        fill[idx] = ({dtype}){neg_inf};
        return;
    }}

    /* Border pixels: initialize to own elevation (known spill level) */
    if (row == 0 || row == height - 1 || col == 0 || col == width - 1) {{
        fill[idx] = elevation[idx];
    }} else {{
        /* Interior pixels: initialize to +infinity (will be lowered) */
        fill[idx] = ({dtype}){pos_inf};
    }}
}}
"""

# ---------------------------------------------------------------------------
# Phase 2: Propagation kernel (iterative)
# ---------------------------------------------------------------------------
# Each pixel's fill level becomes max(own_elevation, min(neighbor_fill_levels)).
# Uses shared memory with 1-pixel halo for the 3x3 neighborhood.

FILL_PROPAGATE_SOURCE_TEMPLATE = r"""
#define TILE_W 16
#define TILE_H 16

extern "C" __global__
void fill_propagate(
    const {dtype}* __restrict__ elevation,
    {dtype}* __restrict__ fill,
    const unsigned char* __restrict__ nodata_mask,
    const int width,
    const int height,
    int* __restrict__ changed
) {{
    __shared__ {dtype} tile[TILE_H + 2][TILE_W + 3];  /* +3 width for bank-conflict padding */

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    const {dtype} POS_INF = ({dtype}){pos_inf};
    const {dtype} NEG_INF = ({dtype}){neg_inf};

    /* Helper: load fill value, treat out-of-bounds as +infinity (no contribution) */
    #define LOAD_FILL(r, c) \
        (((r) >= 0 && (r) < height && (c) >= 0 && (c) < width) \
            ? fill[(r) * width + (c)] : POS_INF)

    /* Load center of tile */
    {dtype} center_fill = (gx < width && gy < height)
                              ? fill[gy * width + gx] : POS_INF;
    tile[ty + 1][tx + 1] = center_fill;

    /* Load left halo */
    if (tx == 0) {{
        tile[ty + 1][0] = LOAD_FILL(gy, gx - 1);
    }}
    /* Load right halo */
    if (tx == TILE_W - 1 || gx == width - 1) {{
        tile[ty + 1][tx + 2] = LOAD_FILL(gy, gx + 1);
    }}
    /* Load top halo */
    if (ty == 0) {{
        tile[0][tx + 1] = LOAD_FILL(gy - 1, gx);
    }}
    /* Load bottom halo */
    if (ty == TILE_H - 1 || gy == height - 1) {{
        tile[ty + 2][tx + 1] = LOAD_FILL(gy + 1, gx);
    }}
    /* Load corner halos */
    if (tx == 0 && ty == 0) {{
        tile[0][0] = LOAD_FILL(gy - 1, gx - 1);
    }}
    if ((tx == TILE_W - 1 || gx == width - 1) && ty == 0) {{
        tile[0][tx + 2] = LOAD_FILL(gy - 1, gx + 1);
    }}
    if (tx == 0 && (ty == TILE_H - 1 || gy == height - 1)) {{
        tile[ty + 2][0] = LOAD_FILL(gy + 1, gx - 1);
    }}
    if ((tx == TILE_W - 1 || gx == width - 1) &&
        (ty == TILE_H - 1 || gy == height - 1)) {{
        tile[ty + 2][tx + 2] = LOAD_FILL(gy + 1, gx + 1);
    }}

    __syncthreads();

    if (gx >= width || gy >= height) return;

    int idx = gy * width + gx;

    /* Nodata pixels are barriers — never update them */
    if (nodata_mask != nullptr && nodata_mask[idx]) return;

    /* Border pixels are already at their correct level — skip */
    if (gy == 0 || gy == height - 1 || gx == 0 || gx == width - 1) return;

    {dtype} my_elev = elevation[idx];
    {dtype} my_fill = tile[ty + 1][tx + 1];

    /* Already at own elevation — can't go lower */
    if (my_fill <= my_elev) return;

    /* Find minimum fill level among 8-connected neighbors,
       ignoring nodata barriers (NEG_INF values). */
    {dtype} min_neighbor = POS_INF;

    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {{
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {{
            if (dx == 0 && dy == 0) continue;
            {dtype} nval = tile[ty + 1 + dy][tx + 1 + dx];
            /* Skip nodata barriers (NEG_INF) */
            if (nval > NEG_INF) {{
                if (nval < min_neighbor) min_neighbor = nval;
            }}
        }}
    }}

    /* New fill = max(own_elevation, min_neighbor_fill) */
    {dtype} new_fill = (my_elev > min_neighbor) ? my_elev : min_neighbor;

    if (new_fill < my_fill) {{
        fill[idx] = new_fill;
        *changed = 1;
    }}

    #undef LOAD_FILL
}}
"""


def get_fill_init_source(dtype_name: str) -> str:
    """Return fill_init kernel source for the given dtype."""
    if dtype_name == "float":
        pos_inf = "3.402823466e+38f"
        neg_inf = "-3.402823466e+38f"
    else:
        pos_inf = "1.7976931348623157e+308"
        neg_inf = "-1.7976931348623157e+308"
    return FILL_INIT_SOURCE_TEMPLATE.format(
        dtype=dtype_name,
        pos_inf=pos_inf,
        neg_inf=neg_inf,
    )


def get_fill_propagate_source(dtype_name: str) -> str:
    """Return fill_propagate kernel source for the given dtype."""
    if dtype_name == "float":
        pos_inf = "3.402823466e+38f"
        neg_inf = "-3.402823466e+38f"
    else:
        pos_inf = "1.7976931348623157e+308"
        neg_inf = "-1.7976931348623157e+308"
    return FILL_PROPAGATE_SOURCE_TEMPLATE.format(
        dtype=dtype_name,
        pos_inf=pos_inf,
        neg_inf=neg_inf,
    )


FILL_INIT_NAMES = ("fill_init",)
FILL_PROPAGATE_NAMES = ("fill_propagate",)
