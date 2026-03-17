# Installation

## Requirements

- Python 3.12+
- vibespatial (core package)
- scipy 1.14+

For raster IO:

- rasterio 1.4+

For GPU acceleration (pick one):

- **CUDA 12**: CuPy 13+, cuda-python 12.x, cuda-cccl, nvImageCodec
- **CUDA 13**: CuPy 14+, cuda-python 13.x, cuda-cccl, nvImageCodec

## Install with pip

```bash
# Core (CPU-only, scipy fallback for all ops)
pip install vibespatial-raster

# With rasterio for GeoTIFF/COG IO
pip install vibespatial-raster[io]

# With GPU support (CUDA 12)
pip install vibespatial-raster[cu12]

# With GPU support (CUDA 13)
pip install vibespatial-raster[cu13]
```

The `cu12` and `cu13` extras are mutually exclusive — install one or the other.
Both include CuPy, cuda-python, CCCL, and nvImageCodec for GPU-native IO.

## Install with uv

```bash
# CPU-only
uv sync

# With dev tools
uv sync --group dev
```

## Verifying GPU support

```python
from vibespatial.runtime import has_gpu_runtime

print(has_gpu_runtime())  # True if CuPy + GPU detected
```

When CuPy is not installed or no GPU is available, all operations automatically
fall back to CPU (scipy, rasterio). No code changes required.

## Verifying IO support

```python
from vibespatial.raster import has_rasterio_support, has_nvimgcodec_support

print(has_rasterio_support())    # True if rasterio is installed
print(has_nvimgcodec_support())  # True if nvImageCodec is available
```
