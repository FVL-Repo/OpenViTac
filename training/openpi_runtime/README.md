# OpenPi runtime environment

This is the self-contained OpenPi-derived training runtime used by OpenViTac.
The `pyproject.toml`, `uv.lock`, Python version, and local `openpi-client`
workspace are included in this directory. The other benchmark repositories are
not required.

On Linux x86-64 with NVIDIA GPUs and CUDA 12.8, install from this directory:

```bash
cd training/openpi_runtime
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

Installation needs network access for dependencies, including the pinned
LeRobot source. It does not download model weights or datasets. See the
[training guide](../README.md) for weight and data locations, preprocessing,
training commands, and details of the pinned CUDA nightly dependencies.
