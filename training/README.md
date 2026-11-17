# Simulation policy training

This directory provides one simulation data pipeline for full fine-tuning of
vision-only π0.5 and OpenVTLA. OpenVTLA adds shared AnyTouch2 tactile encoding
and token concatenation to π0.5. Training uses the bundled OpenPI-derived
runtime; the neighboring benchmark repositories are not required.

## Environment

Use Linux x86-64, NVIDIA GPUs, CUDA 12.8, Python 3.11 and `uv`. The locked
runtime uses the PyTorch CUDA nightly stack described in
[`openpi_runtime/pyproject.toml`](openpi_runtime/pyproject.toml). Its exact
nightly wheels may become unavailable; retain a wheel cache for reproducibility.
The [runtime environment guide](openpi_runtime/README.md) also describes how to
install from inside the runtime directory.
From the OpenViTac root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --project training/openpi_runtime --frozen
cp -r training/openpi_runtime/src/openpi/models_pytorch/transformers_replace/* \
  training/openpi_runtime/.venv/lib/python3.11/site-packages/transformers/
```

Repeat the Transformers patch if the environment is reinstalled. OpenPI also
uses the PaliGemma tokenizer: on first use it may download
`gs://big_vision/paligemma_tokenizer.model` to `~/.cache/openpi/big_vision/`.
For an offline machine, put that tokenizer in the same cache path beforehand.
No pretrained model weights or datasets are downloaded by the commands below.

## Files and tasks

Place the files yourself; paths are relative to the OpenViTac root:

```text
data/training/hdf5/<task>/*.hdf5             # one episode per file
data/training/lerobot/<task>/                # converted dataset
data/training/stats/<model>/<task>/          # normalization statistics
weights/pi05_base_pytorch/model.safetensors  # required for both models
weights/anytouch2/checkpoint-4frames.pth    # required for OpenVTLA
outputs/training/<model>/<task>/<run>/      # model and optimizer checkpoints
```

The π0.5 weight must be an OpenPI-compatible **PyTorch** checkpoint; it is a
directory containing `model.safetensors`. AnyTouch2 needs the pretrained
four-frame checkpoint. These files, datasets, statistics and outputs are not
committed to Git.

| Task | Name | Steps |
| --- | --- | ---: |
| `wc` | Weight Classification | 10,000 |
| `hc` | Hardness Classification | 10,000 |
| `rc` | Roughness Classification | 10,000 |
| `rr` | Roughness-Guided Regrasp | 10,000 |
| `ecs` | Empty-Can Selection | 10,000 |
| `gc` | Grasp Chip | 10,000 |
| `ga` | Gear Assembly | 10,000 |
| `pd` | Pull Drawer | 10,000 |
| `in_usb` | Insert USB | 10,000 |
| `in_b_v1` | Insert Block v1 | 10,000 |
| `in_b_v2` | Insert Block v2 | 20,000 |

All commands run from the repository root. Replace `ga` with any task above.
Inspect the resolved defaults without installing the runtime:

```bash
python training/run.py show --task ga --model openvtla
```

## Convert and compute statistics

```bash
uv run --project training/openpi_runtime python training/run.py convert --task ga
uv run --project training/openpi_runtime python training/run.py stats --task ga --model pi05
uv run --project training/openpi_runtime python training/run.py stats --task ga --model openvtla
```

The converter reads OpenViTac simulation HDF5: `step`,
`embodiment/joint`, head and wrist RGB JPEG streams, both
`tactile/*/rgb_marker` streams, and the episode `instruction` attribute (or
`episode_context_json`). Use `--task-prompt "..."` on `convert` only if an
episode lacks an instruction or all instructions should be overridden. It
validates the schema and refuses to overwrite an existing output directory.
Inspect example images when using a different HDF5 source; this converter
preserves the existing OpenViTac/OpenCV JPEG channel convention.

The output stores 224×224 images, the seven arm joints and left gripper joint,
and the next recorded frame as the absolute target (the last frame is clamped).
At training time each target arm joint is made relative to the current state;
the gripper target remains absolute. Statistics use the same transformation.
Recompute statistics after replacing or changing a dataset. Full statistics need
at least 64 frames; `stats --max-frames 64` is only a quick debugging check.

## Train

For example, use four GPUs, with a global batch size of 64 (16 per GPU):

```bash
# Vision-only π0.5
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --project training/openpi_runtime torchrun \
  --standalone --nnodes=1 --nproc-per-node=4 training/run.py train \
  --task ga --model pi05 --exp-name run1

# OpenVTLA
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --project training/openpi_runtime torchrun \
  --standalone --nnodes=1 --nproc-per-node=4 training/run.py train \
  --task ga --model openvtla --exp-name run1
```

To resume, repeat the same command with `--resume`; the latest saved model and
optimizer are restored. New runs refuse to overwrite an existing experiment.
A single GPU can use `uv run --project training/openpi_runtime python
training/run.py train ...`; enough GPU memory is required for batch size 64.
The GPU count must divide 64. Checkpoints are saved every 5,000 steps and at
the final step. `--num-workers` controls data-loader workers per process.

Both policies use PyTorch full fine-tuning, action horizon 50, 8-dimensional
state/action (padded internally), AdamW with gradient clipping at 1.0, 5,000
warmup steps and a 5e-5 peak learning rate. OpenVTLA trains AnyTouch2 and its
projector along with the π0.5 backbone. It reads each side's frames at offsets
`[-6, -4, -2, 0]`, pads at episode starts, and prepends 36 projected tokens per
side to the visual token sequence. The conversion stores individual frames;
histories are assembled by the loader. The two models use separate statistics
and checkpoint directories.

## Checks and provenance

Run CPU pipeline checks with `python -m pytest training/tests`. With the runtime
environment installed, additional upstream regression tests can be run from
`training/openpi_runtime` using `uv run pytest scripts/anytouch2_test.py`.
The embedded runtime derives from `openpi_benchmark` branch `anytouch2-clean`
(commit `38f8bc1`), with the vision-only configuration informed by
`full-code` (commit `0f6ea6b`). Its Apache and Gemma license texts are retained
in `openpi_runtime/`. Full GPU training and real-data support are outside these
CPU checks.
