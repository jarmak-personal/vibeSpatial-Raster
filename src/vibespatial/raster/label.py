"""Connected component labeling, sieve filtering, and morphology.

CPU baseline uses scipy.ndimage. GPU path uses custom NVRTC union-find
kernels (kernels/ccl.py) and morphology stencil kernels (kernels/morphology.py).

ADR-0040: CCCL Connected Component Labeling
"""

from __future__ import annotations

import logging
import time

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structure_for_connectivity(connectivity: int) -> np.ndarray:
    """Return a structuring element for the given connectivity."""
    if connectivity == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    elif connectivity == 8:
        return np.ones((3, 3), dtype=np.int32)
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _has_cupy() -> bool:
    """Return True if CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def make_structuring_element(
    shape: str,
    size: int | tuple[int, int],
) -> np.ndarray:
    """Build a binary structuring element (SE) for morphological operations.

    Parameters
    ----------
    shape : str
        One of ``'rect'``, ``'cross'``, ``'disk'``.
    size : int or (int, int)
        For ``'rect'`` and ``'cross'``: side length (int) or (height, width).
        For ``'disk'``: radius as int (the SE will be ``(2*r+1, 2*r+1)``).
        Sizes must be odd.

    Returns
    -------
    np.ndarray
        2-D uint8 array with 1s for active SE positions.

    Raises
    ------
    ValueError
        If *shape* is unknown or *size* is even.
    """
    if isinstance(size, int):
        h = w = size
    else:
        h, w = size

    valid_shapes = ("rect", "cross", "disk")
    if shape not in valid_shapes:
        raise ValueError(f"shape must be one of {list(valid_shapes)}, got {shape!r}")

    if shape == "disk":
        if isinstance(size, tuple):
            raise ValueError("disk shape requires a single integer radius, not (h, w)")
        radius = size
        d = 2 * radius + 1
        se = np.zeros((d, d), dtype=np.uint8)
        cy = cx = radius
        for y in range(d):
            for x in range(d):
                if (y - cy) ** 2 + (x - cx) ** 2 <= radius * radius:
                    se[y, x] = 1
        return se

    # rect and cross require odd dimensions
    if h % 2 == 0 or w % 2 == 0:
        raise ValueError(f"structuring element dimensions must be odd, got ({h}, {w})")

    if shape == "rect":
        return np.ones((h, w), dtype=np.uint8)

    # cross: only center row and center column are active
    se = np.zeros((h, w), dtype=np.uint8)
    se[h // 2, :] = 1
    se[:, w // 2] = 1
    return se


def _resolve_structuring_element(
    structuring_element: np.ndarray | str | None,
    connectivity: int,
) -> np.ndarray:
    """Resolve an SE specification to a concrete numpy array.

    If *structuring_element* is None, fall back to the legacy 3x3 SE
    determined by *connectivity*.
    """
    if structuring_element is None:
        return _structure_for_connectivity(connectivity).astype(np.uint8)

    if isinstance(structuring_element, str):
        # Preset strings: 'rect3', 'cross5', 'disk2', etc. are NOT supported
        # here — callers should use make_structuring_element() explicitly.
        raise TypeError(
            "structuring_element must be a numpy array or None. "
            "Use make_structuring_element() to build presets."
        )

    se = np.asarray(structuring_element, dtype=np.uint8)
    if se.ndim != 2:
        raise ValueError(f"structuring element must be 2-D, got {se.ndim}-D")
    if se.shape[0] % 2 == 0 or se.shape[1] % 2 == 0:
        raise ValueError(f"structuring element dimensions must be odd, got {se.shape}")
    return se


def _is_full_rect(se: np.ndarray) -> bool:
    """Return True if *se* is a full rectangular SE (all ones) — separable."""
    return bool(se.all())


def _se_is_default_3x3(se: np.ndarray) -> bool:
    """Return True if *se* is a 3x3 element matching the legacy kernels."""
    return se.shape == (3, 3)


def _should_use_gpu(raster: OwnedRasterArray, threshold: int = 100_000) -> bool:
    """Auto-dispatch heuristic: use GPU when available and image is large enough."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= threshold
    except (ImportError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# CPU baseline: connected component labeling
# ---------------------------------------------------------------------------


def _label_cpu(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
) -> OwnedRasterArray:
    """CPU connected component labeling via scipy.ndimage.label."""
    from scipy.ndimage import label as scipy_label

    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("connected component labeling requires a single-band raster")
        data = data[0]

    # Build foreground mask
    foreground = data != 0
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            foreground &= ~np.isnan(data)
        else:
            foreground &= data != raster.nodata

    structure = _structure_for_connectivity(connectivity)
    labeled, num_features = scipy_label(foreground.astype(np.int32), structure=structure)

    result = from_numpy(
        labeled.astype(np.int32),
        nodata=0,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"label_cpu components={num_features} connectivity={connectivity}",
            residency=result.residency,
        )
    )
    return result


# ---------------------------------------------------------------------------
# GPU: connected component labeling (iterative union-find)
# ---------------------------------------------------------------------------


def label_gpu(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
) -> OwnedRasterArray:
    """GPU connected component labeling using iterative union-find.

    Uses NVRTC kernels: init_labels -> local_merge -> pointer_jump (iterate)
    -> relabel to compact sequential labels 1..N.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster. Nonzero values are foreground.
    connectivity : int
        4 or 8 neighbor connectivity.

    Returns
    -------
    OwnedRasterArray
        HOST-resident integer-labeled raster (int32, nodata=0).
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.ccl import (
        INIT_LABELS_SOURCE,
        LOCAL_MERGE_4C_SOURCE,
        LOCAL_MERGE_8C_SOURCE,
        POINTER_JUMP_SOURCE,
        RELABEL_SOURCE,
    )

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    t0 = time.perf_counter()
    runtime = get_cuda_runtime()

    # --- Prepare data on device ---
    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("connected component labeling requires a single-band raster")
        data = data[0]

    height, width = data.shape
    n = height * width

    # Build foreground mask on host, then transfer once
    foreground_host = (data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            foreground_host &= (~np.isnan(data)).astype(np.uint8)
        else:
            foreground_host &= (data != raster.nodata).astype(np.uint8)

    # H->D transfer (only transfer at start)
    d_foreground = cp.asarray(np.ascontiguousarray(foreground_host.ravel()))

    # Allocate device buffers
    d_labels = cp.empty(n, dtype=np.int32)
    d_changed = cp.zeros(1, dtype=np.int32)

    # --- Compile kernels ---
    # Init labels kernel
    init_key = make_kernel_cache_key("init_labels", INIT_LABELS_SOURCE)
    init_kernels = runtime.compile_kernels(
        cache_key=init_key,
        source=INIT_LABELS_SOURCE,
        kernel_names=("init_labels",),
    )

    # Merge kernel (4c or 8c)
    if connectivity == 4:
        merge_source = LOCAL_MERGE_4C_SOURCE
        merge_name = "local_merge_4c"
    else:
        merge_source = LOCAL_MERGE_8C_SOURCE
        merge_name = "local_merge_8c"

    merge_key = make_kernel_cache_key(merge_name, merge_source)
    merge_kernels = runtime.compile_kernels(
        cache_key=merge_key,
        source=merge_source,
        kernel_names=(merge_name,),
    )

    # Pointer jump kernel
    pj_key = make_kernel_cache_key("pointer_jump", POINTER_JUMP_SOURCE)
    pj_kernels = runtime.compile_kernels(
        cache_key=pj_key,
        source=POINTER_JUMP_SOURCE,
        kernel_names=("pointer_jump",),
    )

    # Relabel kernel
    relabel_key = make_kernel_cache_key("relabel", RELABEL_SOURCE)
    relabel_kernels = runtime.compile_kernels(
        cache_key=relabel_key,
        source=RELABEL_SOURCE,
        kernel_names=("relabel",),
    )

    # --- Phase 1: Init labels ---
    block_1d = (256, 1, 1)
    grid_1d = ((n + 255) // 256, 1, 1)

    runtime.launch(
        kernel=init_kernels["init_labels"],
        grid=grid_1d,
        block=block_1d,
        params=(
            (d_labels.data.ptr, d_foreground.data.ptr, n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # --- Phases 2-3: Iterate merge + pointer jump until convergence ---
    block_2d = (16, 16, 1)
    grid_2d = ((width + 15) // 16, (height + 15) // 16, 1)

    max_iterations = 1000
    for _iteration in range(max_iterations):
        # Reset changed flag
        d_changed.fill(0)

        # Phase 2: Local merge
        runtime.launch(
            kernel=merge_kernels[merge_name],
            grid=grid_2d,
            block=block_2d,
            params=(
                (d_labels.data.ptr, d_foreground.data.ptr, width, height, d_changed.data.ptr),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                ),
            ),
        )

        # Phase 3: Pointer jumping (multiple rounds per iteration)
        for _pj in range(10):
            d_pj_changed = cp.zeros(1, dtype=np.int32)
            runtime.launch(
                kernel=pj_kernels["pointer_jump"],
                grid=grid_1d,
                block=block_1d,
                params=(
                    (d_labels.data.ptr, n, d_pj_changed.data.ptr),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_PTR),
                ),
            )
            if int(d_pj_changed.item()) == 0:
                break

        if int(d_changed.item()) == 0:
            break

    iterations_done = _iteration + 1

    # --- Phase 4: Compact relabel using CuPy sort + unique ---
    # Extract only foreground labels
    h_labels = cp.asnumpy(d_labels)
    fg_mask = h_labels >= 0
    if not fg_mask.any():
        # No foreground pixels at all
        result = from_numpy(
            np.zeros((height, width), dtype=np.int32),
            nodata=0,
            affine=raster.affine,
            crs=raster.crs,
        )
        elapsed = time.perf_counter() - t0
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=f"label_gpu components=0 connectivity={connectivity} iterations={iterations_done} elapsed={elapsed:.3f}s",
                residency=result.residency,
                visible_to_user=True,
                elapsed_seconds=elapsed,
            )
        )
        return result

    # Find unique root labels and build compact mapping
    unique_roots = np.unique(h_labels[fg_mask])
    num_components = len(unique_roots)

    # Build compact IDs 1..N
    compact_ids = np.arange(1, num_components + 1, dtype=np.int32)

    # Transfer mapping to device
    d_unique_roots = cp.asarray(unique_roots.astype(np.int32))
    d_compact_ids = cp.asarray(compact_ids)

    # Launch relabel kernel
    runtime.launch(
        kernel=relabel_kernels["relabel"],
        grid=grid_1d,
        block=block_1d,
        params=(
            (d_labels.data.ptr, d_compact_ids.data.ptr, d_unique_roots.data.ptr, num_components, n),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # --- D->H transfer (final) ---
    host_labels = cp.asnumpy(d_labels).reshape(height, width)

    elapsed = time.perf_counter() - t0
    result = from_numpy(
        host_labels.astype(np.int32),
        nodata=0,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"label_gpu components={num_components} connectivity={connectivity} "
                f"iterations={iterations_done} pixels={n} elapsed={elapsed:.3f}s"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


# ---------------------------------------------------------------------------
# GPU: morphology (erosion, dilation, open, close)
# ---------------------------------------------------------------------------


def morphology_gpu(
    raster: OwnedRasterArray,
    operation: str,
    *,
    connectivity: int = 4,
    iterations: int = 1,
) -> OwnedRasterArray:
    """GPU binary morphology using NVRTC 3x3 stencil kernels.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input binary raster (nonzero = foreground).
    operation : str
        One of "erode", "dilate", "open", "close".
    connectivity : int
        4 or 8 neighbor connectivity for the structuring element.
    iterations : int
        Number of times to apply the operation.

    Returns
    -------
    OwnedRasterArray
        HOST-resident result raster.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.morphology import (
        BINARY_DILATE_KERNEL_SOURCE,
        BINARY_ERODE_KERNEL_SOURCE,
    )

    valid_ops = ("erode", "dilate", "open", "close")
    if operation not in valid_ops:
        raise ValueError(f"operation must be one of {list(valid_ops)}, got {operation!r}")

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    t0 = time.perf_counter()
    runtime = get_cuda_runtime()

    # --- Prepare data on device ---
    data = raster.to_numpy()
    if data.ndim == 3:
        data = data[0]

    height, width = data.shape

    # Build binary foreground mask
    binary = (data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            binary &= (~np.isnan(data)).astype(np.uint8)
        else:
            binary &= (data != raster.nodata).astype(np.uint8)

    # H->D transfer
    d_input = cp.asarray(np.ascontiguousarray(binary.ravel()))

    # Build structuring element
    structure = _structure_for_connectivity(connectivity).astype(np.uint8)
    d_structure = cp.asarray(np.ascontiguousarray(structure.ravel()))

    # Allocate output
    n = height * width
    d_output = cp.empty(n, dtype=np.uint8)

    # --- Compile kernels (shared-memory tiled versions) ---
    erode_key = make_kernel_cache_key("binary_erode", BINARY_ERODE_KERNEL_SOURCE)
    erode_kernels = runtime.compile_kernels(
        cache_key=erode_key,
        source=BINARY_ERODE_KERNEL_SOURCE,
        kernel_names=("binary_erode",),
    )

    dilate_key = make_kernel_cache_key("binary_dilate", BINARY_DILATE_KERNEL_SOURCE)
    dilate_kernels = runtime.compile_kernels(
        cache_key=dilate_key,
        source=BINARY_DILATE_KERNEL_SOURCE,
        kernel_names=("binary_dilate",),
    )

    block_2d = (16, 16, 1)
    grid_2d = ((width + 15) // 16, (height + 15) // 16, 1)

    def _run_erode(d_in: object, d_out: object) -> None:
        runtime.launch(
            kernel=erode_kernels["binary_erode"],
            grid=grid_2d,
            block=block_2d,
            params=(
                (d_in.data.ptr, d_out.data.ptr, d_structure.data.ptr, width, height),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

    def _run_dilate(d_in: object, d_out: object) -> None:
        runtime.launch(
            kernel=dilate_kernels["binary_dilate"],
            grid=grid_2d,
            block=block_2d,
            params=(
                (d_in.data.ptr, d_out.data.ptr, d_structure.data.ptr, width, height),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

    # Determine sequence of operations
    if operation == "erode":
        ops_sequence = ["erode"] * iterations
    elif operation == "dilate":
        ops_sequence = ["dilate"] * iterations
    elif operation == "open":
        # Open = erode then dilate, repeated
        ops_sequence = (["erode"] * iterations) + (["dilate"] * iterations)
    elif operation == "close":
        # Close = dilate then erode, repeated
        ops_sequence = (["dilate"] * iterations) + (["erode"] * iterations)
    else:
        raise ValueError(f"Unknown operation: {operation!r}")

    # Run the operation sequence, ping-ponging between buffers
    current_in = d_input
    current_out = d_output
    for op in ops_sequence:
        if op == "erode":
            _run_erode(current_in, current_out)
        else:
            _run_dilate(current_in, current_out)
        # Swap buffers for next iteration
        current_in, current_out = current_out, current_in

    # After the loop, the result is in current_in (last swap put it there)
    d_result = current_in

    # --- D->H transfer ---
    host_result = cp.asnumpy(d_result).reshape(height, width)
    elapsed = time.perf_counter() - t0
    kernel_launches = len(ops_sequence)

    result = from_numpy(
        host_result.astype(np.uint8),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"morphology_gpu op={operation} connectivity={connectivity} "
                f"iterations={iterations} pixels={n} "
                f"kernel_launches={kernel_launches} grid={grid_2d}"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    logger.debug(
        "morphology_gpu op=%s conn=%d iter=%d pixels=%d elapsed=%.4fs",
        operation,
        connectivity,
        iterations,
        n,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# GPU: NxN morphology with arbitrary structuring elements
# ---------------------------------------------------------------------------


def _morphology_nxn_gpu(
    raster: OwnedRasterArray,
    operation: str,
    *,
    structuring_element: np.ndarray,
    iterations: int = 1,
) -> OwnedRasterArray:
    """GPU binary morphology with arbitrary NxN structuring elements.

    Dispatches to separable 1-D passes for full-rectangle SEs (O(N) per pixel)
    or to the general NxN 2-D kernel otherwise.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input binary raster (nonzero = foreground).
    operation : str
        One of "erode", "dilate", "open", "close".
    structuring_element : np.ndarray
        2-D uint8 array with odd dimensions.
    iterations : int
        Number of times to apply the operation.

    Returns
    -------
    OwnedRasterArray
        HOST-resident result raster.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import get_cuda_runtime

    valid_ops = ("erode", "dilate", "open", "close")
    if operation not in valid_ops:
        raise ValueError(f"operation must be one of {list(valid_ops)}, got {operation!r}")

    t0 = time.perf_counter()
    runtime = get_cuda_runtime()

    # --- Prepare data on device ---
    data = raster.to_numpy()
    if data.ndim == 3:
        data = data[0]

    height, width = data.shape

    # Build binary foreground mask
    binary = (data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            binary &= (~np.isnan(data)).astype(np.uint8)
        else:
            binary &= (data != raster.nodata).astype(np.uint8)

    # H->D transfer (once)
    d_input = cp.asarray(np.ascontiguousarray(binary.ravel()))

    se = structuring_element
    se_h, se_w = se.shape
    n = height * width
    d_output = cp.empty(n, dtype=np.uint8)

    use_separable = _is_full_rect(se) and se_h == se_w

    # Determine operation sequence
    if operation == "erode":
        ops_sequence = ["erode"] * iterations
    elif operation == "dilate":
        ops_sequence = ["dilate"] * iterations
    elif operation == "open":
        ops_sequence = (["erode"] * iterations) + (["dilate"] * iterations)
    elif operation == "close":
        ops_sequence = (["dilate"] * iterations) + (["erode"] * iterations)
    else:
        raise ValueError(f"Unknown operation: {operation!r}")

    if use_separable:
        # --- Separable path: two 1-D passes per operation ---
        _run_separable_morph(runtime, d_input, d_output, width, height, n, se, ops_sequence)
        # After _run_separable_morph, result is in d_input (ping-pong ended there)
        d_result = d_input
    else:
        # --- General NxN 2-D kernel ---
        d_result = _run_nxn_morph(runtime, d_input, d_output, width, height, n, se, ops_sequence)

    # --- D->H transfer ---
    host_result = cp.asnumpy(d_result).reshape(height, width)
    elapsed = time.perf_counter() - t0

    result = from_numpy(
        host_result.astype(np.uint8),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    se_desc = f"{se_h}x{se_w}" + (" separable" if use_separable else "")
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"morphology_nxn_gpu op={operation} se={se_desc} iterations={iterations} pixels={n}"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    logger.debug(
        "morphology_nxn_gpu op=%s se=%s iter=%d pixels=%d elapsed=%.4fs",
        operation,
        se_desc,
        iterations,
        n,
        elapsed,
    )
    return result


def _run_nxn_morph(
    runtime: object,
    d_input: object,
    d_output: object,
    width: int,
    height: int,
    n: int,
    se: np.ndarray,
    ops_sequence: list[str],
) -> object:
    """Execute NxN 2-D morphology kernels for each operation in the sequence.

    Returns the device buffer holding the final result.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.morphology import (
        BINARY_DILATE_NXN_KERNEL_SOURCE,
        BINARY_ERODE_NXN_KERNEL_SOURCE,
    )

    se_h, se_w = se.shape
    se_ry = se_h // 2
    se_rx = se_w // 2

    # Choose tile size: shrink tiles when halo is large to keep smem reasonable
    tile_w = 16
    tile_h = 16
    smem_bytes = (tile_h + 2 * se_ry) * (tile_w + 2 * se_rx + 1)
    # Cap shared memory at 48 KB (conservative; most GPUs have >=48 KB)
    while smem_bytes > 48 * 1024 and tile_w > 4:
        tile_w = max(4, tile_w // 2)
        tile_h = max(4, tile_h // 2)
        smem_bytes = (tile_h + 2 * se_ry) * (tile_w + 2 * se_rx + 1)

    defines = (
        f"-DSE_RADIUS_X={se_rx}",
        f"-DSE_RADIUS_Y={se_ry}",
        f"-DSE_W={se_w}",
        f"-DSE_H={se_h}",
        f"-DTILE_W={tile_w}",
        f"-DTILE_H={tile_h}",
    )

    # Compile NxN erode kernel
    erode_key = make_kernel_cache_key(
        f"binary_erode_nxn_{se_h}x{se_w}_t{tile_h}x{tile_w}",
        BINARY_ERODE_NXN_KERNEL_SOURCE,
    )
    erode_kernels = runtime.compile_kernels(
        cache_key=erode_key,
        source=BINARY_ERODE_NXN_KERNEL_SOURCE,
        kernel_names=("binary_erode_nxn",),
        options=defines,
    )

    # Compile NxN dilate kernel
    dilate_key = make_kernel_cache_key(
        f"binary_dilate_nxn_{se_h}x{se_w}_t{tile_h}x{tile_w}",
        BINARY_DILATE_NXN_KERNEL_SOURCE,
    )
    dilate_kernels = runtime.compile_kernels(
        cache_key=dilate_key,
        source=BINARY_DILATE_NXN_KERNEL_SOURCE,
        kernel_names=("binary_dilate_nxn",),
        options=defines,
    )

    # Transfer SE to device (once)
    d_se = cp.asarray(np.ascontiguousarray(se.ravel().astype(np.uint8)))

    block_2d = (tile_w, tile_h, 1)
    grid_2d = (
        (width + tile_w - 1) // tile_w,
        (height + tile_h - 1) // tile_h,
        1,
    )

    param_types = (
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
    )

    current_in = d_input
    current_out = d_output
    for op in ops_sequence:
        kernel = (
            erode_kernels["binary_erode_nxn"]
            if op == "erode"
            else dilate_kernels["binary_dilate_nxn"]
        )
        runtime.launch(
            kernel=kernel,
            grid=grid_2d,
            block=block_2d,
            params=(
                (current_in.data.ptr, current_out.data.ptr, d_se.data.ptr, width, height),
                param_types,
            ),
        )
        current_in, current_out = current_out, current_in

    return current_in


def _run_separable_morph(
    runtime: object,
    d_input: object,
    d_output: object,
    width: int,
    height: int,
    n: int,
    se: np.ndarray,
    ops_sequence: list[str],
) -> None:
    """Execute separable 1-D morphology passes (horizontal then vertical).

    Modifies d_input/d_output in-place via ping-pong. After return the final
    result is in d_input (due to an even number of swaps per operation: H then V).
    """
    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.morphology import (
        BINARY_DILATE_SEP_H_KERNEL_SOURCE,
        BINARY_DILATE_SEP_V_KERNEL_SOURCE,
        BINARY_ERODE_SEP_H_KERNEL_SOURCE,
        BINARY_ERODE_SEP_V_KERNEL_SOURCE,
    )

    se_h, se_w = se.shape
    radius_y = se_h // 2
    radius_x = se_w // 2

    tile_size = 256  # 1-D tile for separable passes

    h_defines = (f"-DRADIUS={radius_x}", f"-DTILE_SIZE={tile_size}")
    v_defines = (f"-DRADIUS={radius_y}", f"-DTILE_SIZE={tile_size}")

    # Compile separable kernels
    erode_h_key = make_kernel_cache_key(
        f"binary_erode_sep_h_r{radius_x}_t{tile_size}",
        BINARY_ERODE_SEP_H_KERNEL_SOURCE,
    )
    erode_h_kernels = runtime.compile_kernels(
        cache_key=erode_h_key,
        source=BINARY_ERODE_SEP_H_KERNEL_SOURCE,
        kernel_names=("binary_erode_sep_h",),
        options=h_defines,
    )

    erode_v_key = make_kernel_cache_key(
        f"binary_erode_sep_v_r{radius_y}_t{tile_size}",
        BINARY_ERODE_SEP_V_KERNEL_SOURCE,
    )
    erode_v_kernels = runtime.compile_kernels(
        cache_key=erode_v_key,
        source=BINARY_ERODE_SEP_V_KERNEL_SOURCE,
        kernel_names=("binary_erode_sep_v",),
        options=v_defines,
    )

    dilate_h_key = make_kernel_cache_key(
        f"binary_dilate_sep_h_r{radius_x}_t{tile_size}",
        BINARY_DILATE_SEP_H_KERNEL_SOURCE,
    )
    dilate_h_kernels = runtime.compile_kernels(
        cache_key=dilate_h_key,
        source=BINARY_DILATE_SEP_H_KERNEL_SOURCE,
        kernel_names=("binary_dilate_sep_h",),
        options=h_defines,
    )

    dilate_v_key = make_kernel_cache_key(
        f"binary_dilate_sep_v_r{radius_y}_t{tile_size}",
        BINARY_DILATE_SEP_V_KERNEL_SOURCE,
    )
    dilate_v_kernels = runtime.compile_kernels(
        cache_key=dilate_v_key,
        source=BINARY_DILATE_SEP_V_KERNEL_SOURCE,
        kernel_names=("binary_dilate_sep_v",),
        options=v_defines,
    )

    # Grid configs for H and V passes
    grid_h = ((width + tile_size - 1) // tile_size, height, 1)
    block_h = (tile_size, 1, 1)

    grid_v = (width, (height + tile_size - 1) // tile_size, 1)
    block_v = (tile_size, 1, 1)

    param_types = (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32)

    current_in = d_input
    current_out = d_output
    for op in ops_sequence:
        # Horizontal pass: current_in -> current_out
        if op == "erode":
            kernel_h = erode_h_kernels["binary_erode_sep_h"]
            kernel_v = erode_v_kernels["binary_erode_sep_v"]
        else:
            kernel_h = dilate_h_kernels["binary_dilate_sep_h"]
            kernel_v = dilate_v_kernels["binary_dilate_sep_v"]

        runtime.launch(
            kernel=kernel_h,
            grid=grid_h,
            block=block_h,
            params=(
                (current_in.data.ptr, current_out.data.ptr, width, height),
                param_types,
            ),
        )
        # Vertical pass: current_out -> current_in (ping-pong back)
        runtime.launch(
            kernel=kernel_v,
            grid=grid_v,
            block=block_v,
            params=(
                (current_out.data.ptr, current_in.data.ptr, width, height),
                param_types,
            ),
        )
    # After each op: H swaps in->out, V swaps out->in. Result is in current_in.


# ---------------------------------------------------------------------------
# Public API: dispatchers (GPU/CPU auto-selection)
# ---------------------------------------------------------------------------


def label_connected_components(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Label connected components in a raster.

    Each group of connected nonzero (and non-nodata) pixels receives a unique
    integer label. Background (zero or nodata) pixels get label 0.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster. Nonzero values are foreground.
    connectivity : int
        4 or 8 neighbor connectivity.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when available and pixel count exceeds threshold.

    Returns
    -------
    OwnedRasterArray
        Integer-labeled raster where each connected component has a unique label.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        return label_gpu(raster, connectivity=connectivity)
    else:
        return _label_cpu(raster, connectivity=connectivity)


def sieve_filter(
    labeled: OwnedRasterArray,
    min_size: int,
    *,
    connectivity: int = 4,
    replace_value: int = 0,
) -> OwnedRasterArray:
    """Remove small connected components from a labeled raster.

    Parameters
    ----------
    labeled : OwnedRasterArray
        Integer-labeled raster (e.g., from label_connected_components).
    min_size : int
        Minimum pixel count to keep a component.
    connectivity : int
        4 or 8 neighbor connectivity (used for counting).
    replace_value : int
        Value to assign to removed components (default 0 = background).

    Returns
    -------
    OwnedRasterArray
        Sieved raster with small components replaced.
    """
    data = labeled.to_numpy().copy()
    if data.ndim == 3:
        data = data[0]

    unique_labels, counts = np.unique(data, return_counts=True)
    small_labels = unique_labels[counts < min_size]

    # Don't remove the background label
    nodata = labeled.nodata
    if nodata is not None:
        small_labels = small_labels[small_labels != int(nodata)]

    mask = np.isin(data, small_labels)
    data[mask] = replace_value

    removed_count = len(small_labels)
    result = from_numpy(data, nodata=nodata, affine=labeled.affine, crs=labeled.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"sieve_filter removed={removed_count} min_size={min_size}",
            residency=result.residency,
        )
    )
    return result


def raster_morphology(
    raster: OwnedRasterArray,
    operation: str,
    *,
    connectivity: int = 4,
    iterations: int = 1,
    structuring_element: np.ndarray | None = None,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Apply binary morphological operation to a raster.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input binary raster (nonzero = foreground).
    operation : str
        One of "erode", "dilate", "open", "close".
    connectivity : int
        4 or 8 neighbor connectivity for the structuring element.
        Ignored when *structuring_element* is provided.
    iterations : int
        Number of times to apply the operation.
    structuring_element : np.ndarray or None
        Custom structuring element (2-D uint8 array with odd dimensions).
        If None, falls back to the legacy 3x3 SE based on *connectivity*.
        Use :func:`make_structuring_element` to build presets.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    OwnedRasterArray
        Result raster.
    """
    valid_ops = ("erode", "dilate", "open", "close")
    if operation not in valid_ops:
        raise ValueError(f"operation must be one of {list(valid_ops)}, got {operation!r}")

    se = _resolve_structuring_element(structuring_element, connectivity)

    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        # Use legacy 3x3 fast path when SE matches
        if _se_is_default_3x3(se):
            return morphology_gpu(
                raster, operation, connectivity=connectivity, iterations=iterations
            )
        return _morphology_nxn_gpu(raster, operation, structuring_element=se, iterations=iterations)
    else:
        return _morphology_cpu(
            raster,
            operation,
            connectivity=connectivity,
            iterations=iterations,
            structuring_element=se,
        )


# ---------------------------------------------------------------------------
# CPU baseline: morphology
# ---------------------------------------------------------------------------


def _morphology_cpu(
    raster: OwnedRasterArray,
    operation: str,
    *,
    connectivity: int = 4,
    iterations: int = 1,
    structuring_element: np.ndarray | None = None,
) -> OwnedRasterArray:
    """CPU binary morphology via scipy.ndimage."""
    from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

    ops = {
        "erode": binary_erosion,
        "dilate": binary_dilation,
        "open": binary_opening,
        "close": binary_closing,
    }

    data = raster.to_numpy()
    if data.ndim == 3:
        data = data[0]

    binary = data != 0
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            binary &= ~np.isnan(data)
        else:
            binary &= data != raster.nodata

    if structuring_element is not None:
        structure = structuring_element
    else:
        structure = _structure_for_connectivity(connectivity)
    result_data = ops[operation](binary, structure=structure, iterations=iterations)

    se_desc = f"{structure.shape[0]}x{structure.shape[1]}"
    result = from_numpy(
        result_data.astype(raster.dtype),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"morphology_cpu op={operation} se={se_desc} "
                f"connectivity={connectivity} iterations={iterations}"
            ),
            residency=result.residency,
        )
    )
    return result


def raster_morphology_tophat(
    raster: OwnedRasterArray,
    structuring_element: np.ndarray | None = None,
    *,
    connectivity: int = 4,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """White top-hat transform: original minus morphological opening.

    Extracts bright features smaller than the structuring element.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input binary raster (nonzero = foreground).
    structuring_element : np.ndarray or None
        Custom SE (2-D uint8, odd dimensions). Defaults to 3x3 from connectivity.
    connectivity : int
        4 or 8 (used only if *structuring_element* is None).
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    OwnedRasterArray
        Pixels that were removed by opening (bright detail).
    """
    opened = raster_morphology(
        raster,
        "open",
        connectivity=connectivity,
        structuring_element=structuring_element,
        use_gpu=use_gpu,
    )
    # Top-hat = original - opened (binary: foreground in original but not in opened)
    original_data = raster.to_numpy()
    if original_data.ndim == 3:
        original_data = original_data[0]
    opened_data = opened.to_numpy()
    if opened_data.ndim == 3:
        opened_data = opened_data[0]

    # Binary: original foreground minus opened foreground
    orig_bin = (original_data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            orig_bin &= (~np.isnan(original_data)).astype(np.uint8)
        else:
            orig_bin &= (original_data != raster.nodata).astype(np.uint8)

    opened_bin = (opened_data != 0).astype(np.uint8)
    tophat_data = (orig_bin & ~opened_bin).astype(np.uint8)

    result = from_numpy(
        tophat_data,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail="morphology_tophat (original - opening)",
            residency=result.residency,
        )
    )
    return result


def raster_morphology_blackhat(
    raster: OwnedRasterArray,
    structuring_element: np.ndarray | None = None,
    *,
    connectivity: int = 4,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Black top-hat transform: morphological closing minus original.

    Extracts dark features (holes) smaller than the structuring element.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input binary raster (nonzero = foreground).
    structuring_element : np.ndarray or None
        Custom SE (2-D uint8, odd dimensions). Defaults to 3x3 from connectivity.
    connectivity : int
        4 or 8 (used only if *structuring_element* is None).
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    OwnedRasterArray
        Pixels that were added by closing (dark detail / holes filled).
    """
    closed = raster_morphology(
        raster,
        "close",
        connectivity=connectivity,
        structuring_element=structuring_element,
        use_gpu=use_gpu,
    )
    # Black-hat = closed - original
    original_data = raster.to_numpy()
    if original_data.ndim == 3:
        original_data = original_data[0]
    closed_data = closed.to_numpy()
    if closed_data.ndim == 3:
        closed_data = closed_data[0]

    orig_bin = (original_data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            orig_bin &= (~np.isnan(original_data)).astype(np.uint8)
        else:
            orig_bin &= (original_data != raster.nodata).astype(np.uint8)

    closed_bin = (closed_data != 0).astype(np.uint8)
    blackhat_data = (closed_bin & ~orig_bin).astype(np.uint8)

    result = from_numpy(
        blackhat_data,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail="morphology_blackhat (closing - original)",
            residency=result.residency,
        )
    )
    return result
