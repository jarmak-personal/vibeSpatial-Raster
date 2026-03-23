"""NVRTC shared-memory tiled morphology kernels for binary erode/dilate.

Bead: GPU morphology stencil kernels (erode/dilate) with shared memory halo.
Uses 3x3 structuring elements with (16,16) thread blocks and 1-pixel halo.

Halo loading uses the standard tile-edge pattern: border threads (tx==0,
tx==TILE_W-1, ty==0, ty==TILE_H-1) load halo cells at *fixed* tile-edge
shared-memory positions using clamped global coordinates.  This eliminates
the fragile ``gx == width - 1`` dual-condition pattern that caused
overlapping writes and incorrect results at tile boundaries.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tile dimensions — must match the #define TILE_W / TILE_H in kernel sources.
# Dispatch code imports these to size thread blocks and grids.
# ---------------------------------------------------------------------------
MORPH_TILE_W: int = 16
MORPH_TILE_H: int = 16

# ---------------------------------------------------------------------------
# Shared helper: safe global-to-shared halo load with clamping
# ---------------------------------------------------------------------------
#
# The halo loading strategy:
#   - Each thread loads its center cell at tile[ty+1][tx+1].
#   - The 4 border thread-rows/columns of the tile load the 1-pixel halo
#     ring at FIXED tile-edge positions (column 0, column TILE_W+1,
#     row 0, row TILE_H+1).
#   - Global coordinates for halo cells are computed from the tile origin
#     (blockIdx * TILE), NOT from the thread's own (gx, gy).
#   - Out-of-bounds global coordinates produce 0 (correct for morphology
#     where out-of-bounds pixels are treated as background).
#   - 4 corner cells are loaded by the 4 corner threads of the tile.
#
# This pattern has NO overlapping shared-memory writes (each shared-memory
# cell is written by exactly one thread) and no dual-condition branches.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Binary erosion kernel -- shared memory tile with 1-pixel halo
# ---------------------------------------------------------------------------

BINARY_ERODE_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16

/* Safe global memory fetch: returns 0 for out-of-bounds coordinates. */
__device__ __forceinline__
unsigned char safe_fetch(
    const unsigned char* __restrict__ data,
    int x, int y, int width, int height
) {
    if (x >= 0 && x < width && y >= 0 && y < height)
        return data[y * width + x];
    return 0;
}

extern "C" __global__
void binary_erode(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ selem,  /* 9 elements, row-major 3x3 */
    const int width,
    const int height
) {
    __shared__ unsigned char tile[TILE_H + 2][TILE_W + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    /* Global coords for the tile origin (top-left corner of the center region). */
    const int tile_ox = blockIdx.x * TILE_W;
    const int tile_oy = blockIdx.y * TILE_H;
    /* Global coords for this thread's pixel. */
    const int gx = tile_ox + tx;
    const int gy = tile_oy + ty;

    /* ---- Load center cell ---- */
    tile[ty + 1][tx + 1] = safe_fetch(input, gx, gy, width, height);

    /* ---- Load halo edges (4 sides) ---- */
    /* Left halo column (shared col 0): loaded by tx == 0 */
    if (tx == 0) {
        tile[ty + 1][0] = safe_fetch(input, tile_ox - 1, gy, width, height);
    }
    /* Right halo column (shared col TILE_W + 1): loaded by tx == TILE_W - 1 */
    if (tx == TILE_W - 1) {
        tile[ty + 1][TILE_W + 1] = safe_fetch(
            input, tile_ox + TILE_W, gy, width, height);
    }
    /* Top halo row (shared row 0): loaded by ty == 0 */
    if (ty == 0) {
        tile[0][tx + 1] = safe_fetch(input, gx, tile_oy - 1, width, height);
    }
    /* Bottom halo row (shared row TILE_H + 1): loaded by ty == TILE_H - 1 */
    if (ty == TILE_H - 1) {
        tile[TILE_H + 1][tx + 1] = safe_fetch(
            input, gx, tile_oy + TILE_H, width, height);
    }

    /* ---- Load 4 corner halo cells ---- */
    if (tx == 0 && ty == 0) {
        tile[0][0] = safe_fetch(
            input, tile_ox - 1, tile_oy - 1, width, height);
    }
    if (tx == TILE_W - 1 && ty == 0) {
        tile[0][TILE_W + 1] = safe_fetch(
            input, tile_ox + TILE_W, tile_oy - 1, width, height);
    }
    if (tx == 0 && ty == TILE_H - 1) {
        tile[TILE_H + 1][0] = safe_fetch(
            input, tile_ox - 1, tile_oy + TILE_H, width, height);
    }
    if (tx == TILE_W - 1 && ty == TILE_H - 1) {
        tile[TILE_H + 1][TILE_W + 1] = safe_fetch(
            input, tile_ox + TILE_W, tile_oy + TILE_H, width, height);
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    /* Erode: output 1 only if ALL structuring-element neighbors are 1 */
    unsigned char result = 1;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (selem[(dy + 1) * 3 + (dx + 1)]) {
                if (!tile[ty + 1 + dy][tx + 1 + dx]) {
                    result = 0;
                }
            }
        }
    }
    output[gy * width + gx] = result;
}
"""

# ---------------------------------------------------------------------------
# Binary dilation kernel -- shared memory tile with 1-pixel halo
# ---------------------------------------------------------------------------

BINARY_DILATE_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16

/* Safe global memory fetch: returns 0 for out-of-bounds coordinates. */
__device__ __forceinline__
unsigned char safe_fetch(
    const unsigned char* __restrict__ data,
    int x, int y, int width, int height
) {
    if (x >= 0 && x < width && y >= 0 && y < height)
        return data[y * width + x];
    return 0;
}

extern "C" __global__
void binary_dilate(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ selem,  /* 9 elements, row-major 3x3 */
    const int width,
    const int height
) {
    __shared__ unsigned char tile[TILE_H + 2][TILE_W + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    /* Global coords for the tile origin (top-left corner of the center region). */
    const int tile_ox = blockIdx.x * TILE_W;
    const int tile_oy = blockIdx.y * TILE_H;
    /* Global coords for this thread's pixel. */
    const int gx = tile_ox + tx;
    const int gy = tile_oy + ty;

    /* ---- Load center cell ---- */
    tile[ty + 1][tx + 1] = safe_fetch(input, gx, gy, width, height);

    /* ---- Load halo edges (4 sides) ---- */
    /* Left halo column (shared col 0): loaded by tx == 0 */
    if (tx == 0) {
        tile[ty + 1][0] = safe_fetch(input, tile_ox - 1, gy, width, height);
    }
    /* Right halo column (shared col TILE_W + 1): loaded by tx == TILE_W - 1 */
    if (tx == TILE_W - 1) {
        tile[ty + 1][TILE_W + 1] = safe_fetch(
            input, tile_ox + TILE_W, gy, width, height);
    }
    /* Top halo row (shared row 0): loaded by ty == 0 */
    if (ty == 0) {
        tile[0][tx + 1] = safe_fetch(input, gx, tile_oy - 1, width, height);
    }
    /* Bottom halo row (shared row TILE_H + 1): loaded by ty == TILE_H - 1 */
    if (ty == TILE_H - 1) {
        tile[TILE_H + 1][tx + 1] = safe_fetch(
            input, gx, tile_oy + TILE_H, width, height);
    }

    /* ---- Load 4 corner halo cells ---- */
    if (tx == 0 && ty == 0) {
        tile[0][0] = safe_fetch(
            input, tile_ox - 1, tile_oy - 1, width, height);
    }
    if (tx == TILE_W - 1 && ty == 0) {
        tile[0][TILE_W + 1] = safe_fetch(
            input, tile_ox + TILE_W, tile_oy - 1, width, height);
    }
    if (tx == 0 && ty == TILE_H - 1) {
        tile[TILE_H + 1][0] = safe_fetch(
            input, tile_ox - 1, tile_oy + TILE_H, width, height);
    }
    if (tx == TILE_W - 1 && ty == TILE_H - 1) {
        tile[TILE_H + 1][TILE_W + 1] = safe_fetch(
            input, tile_ox + TILE_W, tile_oy + TILE_H, width, height);
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    /* Dilate: output 1 if ANY structuring-element neighbor is 1 */
    unsigned char result = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (selem[(dy + 1) * 3 + (dx + 1)]) {
                if (tile[ty + 1 + dy][tx + 1 + dx]) {
                    result = 1;
                }
            }
        }
    }
    output[gy * width + gx] = result;
}
"""
