"""NVRTC shared-memory tiled morphology kernels for binary erode/dilate.

Bead: GPU morphology stencil kernels (erode/dilate) with shared memory halo.
Uses (16,16) thread blocks with shared-memory halo tiling.

Halo loading uses the standard tile-edge pattern: border threads (tx==0,
tx==TILE_W-1, ty==0, ty==TILE_H-1) load halo cells at *fixed* tile-edge
shared-memory positions using clamped global coordinates.

Kernel variants:
- BINARY_ERODE/DILATE_KERNEL_SOURCE: Fixed 3x3 SE, 1-pixel halo, 16x16 tiles.
- BINARY_ERODE/DILATE_NXN_KERNEL_SOURCE: Arbitrary NxN SE, dynamic halo via
  compile-time defines (SE_RADIUS_X, SE_RADIUS_Y, SE_W, SE_H).
  Shared-memory tile is (TILE_H + 2*SE_RADIUS_Y) x (TILE_W + 2*SE_RADIUS_X).
- BINARY_ERODE/DILATE_SEP_KERNEL_SOURCE: 1D separable pass (horizontal or
  vertical) for rectangular SEs. O(N) per pixel instead of O(N^2).
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

# ---------------------------------------------------------------------------
# NxN binary erosion kernel -- arbitrary structuring element
# ---------------------------------------------------------------------------
# Compile-time defines required:
#   SE_RADIUS_X, SE_RADIUS_Y, SE_W, SE_H, TILE_W, TILE_H

BINARY_ERODE_NXN_KERNEL_SOURCE = r"""
extern "C" __global__
void binary_erode_nxn(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ selem,  /* SE_H * SE_W elements, row-major */
    const int width,
    const int height
) {
    /* Shared tile with halo on all sides.
       Pad columns by +1 to avoid shared-memory bank conflicts. */
    __shared__ unsigned char tile[TILE_H + 2 * SE_RADIUS_Y][TILE_W + 2 * SE_RADIUS_X + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_W + tx;
    const int gy = blockIdx.y * TILE_H + ty;

    /* Cooperative halo load: each thread loads multiple cells to fill
       the (TILE_H + 2*SE_RADIUS_Y) x (TILE_W + 2*SE_RADIUS_X) tile. */
    const int smem_w = TILE_W + 2 * SE_RADIUS_X;
    const int smem_h = TILE_H + 2 * SE_RADIUS_Y;
    const int smem_total = smem_w * smem_h;
    const int tid = ty * TILE_W + tx;
    const int block_threads = TILE_W * TILE_H;

    for (int i = tid; i < smem_total; i += block_threads) {
        int sy = i / smem_w;
        int sx = i % smem_w;
        /* Map shared tile coords back to global coords */
        int img_x = (blockIdx.x * TILE_W) + sx - SE_RADIUS_X;
        int img_y = (blockIdx.y * TILE_H) + sy - SE_RADIUS_Y;
        unsigned char val = 0;
        if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
            val = input[img_y * width + img_x];
        }
        tile[sy][sx] = val;
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    /* Erode: output 1 only if ALL SE-active neighbors are 1 */
    unsigned char result = 1;
    for (int dy = 0; dy < SE_H; dy++) {
        for (int dx = 0; dx < SE_W; dx++) {
            if (selem[dy * SE_W + dx]) {
                if (!tile[ty + dy][tx + dx]) {
                    result = 0;
                    /* Early exit via outer loop break */
                    dy = SE_H;
                    break;
                }
            }
        }
    }
    output[gy * width + gx] = result;
}
"""

# ---------------------------------------------------------------------------
# NxN binary dilation kernel -- arbitrary structuring element
# ---------------------------------------------------------------------------

BINARY_DILATE_NXN_KERNEL_SOURCE = r"""
extern "C" __global__
void binary_dilate_nxn(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ selem,  /* SE_H * SE_W elements, row-major */
    const int width,
    const int height
) {
    __shared__ unsigned char tile[TILE_H + 2 * SE_RADIUS_Y][TILE_W + 2 * SE_RADIUS_X + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_W + tx;
    const int gy = blockIdx.y * TILE_H + ty;

    const int smem_w = TILE_W + 2 * SE_RADIUS_X;
    const int smem_h = TILE_H + 2 * SE_RADIUS_Y;
    const int smem_total = smem_w * smem_h;
    const int tid = ty * TILE_W + tx;
    const int block_threads = TILE_W * TILE_H;

    for (int i = tid; i < smem_total; i += block_threads) {
        int sy = i / smem_w;
        int sx = i % smem_w;
        int img_x = (blockIdx.x * TILE_W) + sx - SE_RADIUS_X;
        int img_y = (blockIdx.y * TILE_H) + sy - SE_RADIUS_Y;
        unsigned char val = 0;
        if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
            val = input[img_y * width + img_x];
        }
        tile[sy][sx] = val;
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    /* Dilate: output 1 if ANY SE-active neighbor is 1 */
    unsigned char result = 0;
    for (int dy = 0; dy < SE_H; dy++) {
        for (int dx = 0; dx < SE_W; dx++) {
            if (selem[dy * SE_W + dx]) {
                if (tile[ty + dy][tx + dx]) {
                    result = 1;
                    dy = SE_H;
                    break;
                }
            }
        }
    }
    output[gy * width + gx] = result;
}
"""

# ---------------------------------------------------------------------------
# Separable 1D erosion/dilation kernels for rectangular SE decomposition
# ---------------------------------------------------------------------------
# Compile-time defines required: RADIUS, TILE_SIZE
# Direction is selected by kernel name: _h (horizontal), _v (vertical)

BINARY_ERODE_SEP_H_KERNEL_SOURCE = r"""
extern "C" __global__
void binary_erode_sep_h(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const int width,
    const int height
) {
    /* 1D horizontal pass: shared memory row with halo */
    __shared__ unsigned char row[TILE_SIZE + 2 * RADIUS];

    const int gy = blockIdx.y;
    const int tx = threadIdx.x;
    const int gx = blockIdx.x * TILE_SIZE + tx;

    if (gy >= height) return;

    /* Load center + halo cooperatively */
    const int smem_w = TILE_SIZE + 2 * RADIUS;
    for (int i = tx; i < smem_w; i += TILE_SIZE) {
        int img_x = blockIdx.x * TILE_SIZE + i - RADIUS;
        unsigned char val = 0;
        if (img_x >= 0 && img_x < width) {
            val = input[gy * width + img_x];
        }
        row[i] = val;
    }

    __syncthreads();

    if (gx >= width) return;

    unsigned char result = 1;
    for (int dx = 0; dx <= 2 * RADIUS; dx++) {
        if (!row[tx + dx]) {
            result = 0;
            break;
        }
    }
    output[gy * width + gx] = result;
}
"""

BINARY_ERODE_SEP_V_KERNEL_SOURCE = r"""
extern "C" __global__
void binary_erode_sep_v(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const int width,
    const int height
) {
    /* 1D vertical pass: shared memory column with halo */
    __shared__ unsigned char col[TILE_SIZE + 2 * RADIUS];

    const int gx = blockIdx.x;
    const int ty = threadIdx.x;
    const int gy = blockIdx.y * TILE_SIZE + ty;

    if (gx >= width) return;

    const int smem_h = TILE_SIZE + 2 * RADIUS;
    for (int i = ty; i < smem_h; i += TILE_SIZE) {
        int img_y = blockIdx.y * TILE_SIZE + i - RADIUS;
        unsigned char val = 0;
        if (img_y >= 0 && img_y < height) {
            val = input[img_y * width + gx];
        }
        col[i] = val;
    }

    __syncthreads();

    if (gy >= height) return;

    unsigned char result = 1;
    for (int dy = 0; dy <= 2 * RADIUS; dy++) {
        if (!col[ty + dy]) {
            result = 0;
            break;
        }
    }
    output[gy * width + gx] = result;
}
"""

BINARY_DILATE_SEP_H_KERNEL_SOURCE = r"""
extern "C" __global__
void binary_dilate_sep_h(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const int width,
    const int height
) {
    __shared__ unsigned char row[TILE_SIZE + 2 * RADIUS];

    const int gy = blockIdx.y;
    const int tx = threadIdx.x;
    const int gx = blockIdx.x * TILE_SIZE + tx;

    if (gy >= height) return;

    const int smem_w = TILE_SIZE + 2 * RADIUS;
    for (int i = tx; i < smem_w; i += TILE_SIZE) {
        int img_x = blockIdx.x * TILE_SIZE + i - RADIUS;
        unsigned char val = 0;
        if (img_x >= 0 && img_x < width) {
            val = input[gy * width + img_x];
        }
        row[i] = val;
    }

    __syncthreads();

    if (gx >= width) return;

    unsigned char result = 0;
    for (int dx = 0; dx <= 2 * RADIUS; dx++) {
        if (row[tx + dx]) {
            result = 1;
            break;
        }
    }
    output[gy * width + gx] = result;
}
"""

BINARY_DILATE_SEP_V_KERNEL_SOURCE = r"""
extern "C" __global__
void binary_dilate_sep_v(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const int width,
    const int height
) {
    __shared__ unsigned char col[TILE_SIZE + 2 * RADIUS];

    const int gx = blockIdx.x;
    const int ty = threadIdx.x;
    const int gy = blockIdx.y * TILE_SIZE + ty;

    if (gx >= width) return;

    const int smem_h = TILE_SIZE + 2 * RADIUS;
    for (int i = ty; i < smem_h; i += TILE_SIZE) {
        int img_y = blockIdx.y * TILE_SIZE + i - RADIUS;
        unsigned char val = 0;
        if (img_y >= 0 && img_y < height) {
            val = input[img_y * width + gx];
        }
        col[i] = val;
    }

    __syncthreads();

    if (gy >= height) return;

    unsigned char result = 0;
    for (int dy = 0; dy <= 2 * RADIUS; dy++) {
        if (col[ty + dy]) {
            result = 1;
            break;
        }
    }
    output[gy * width + gx] = result;
}
"""
