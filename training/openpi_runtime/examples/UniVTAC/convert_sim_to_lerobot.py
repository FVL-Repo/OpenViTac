#!/usr/bin/env python3
"""
Convert UniVTAC simulated HDF5 episodes under /home/tione/notebook/data/UniVTAC/sim
to LeRobot datasets.

Recommended usage from the OpenPI repo root:

    cd /home/tione/notebook/openpi_benchmark
    export HF_LEROBOT_HOME=/run/ti/pi-training-data/lerobot

    uv run examples/UniVTAC/convert_sim_to_lerobot.py \
        --hdf5-dir /home/tione/notebook/data/UniVTAC/sim/insert_block \
        --output-dir /run/ti/pi-training-data/lerobot/simulation/insert_block_gelsight_delta_eef \
        --control-mode delta_eef \
        --overwrite

Pull Drawer delta EEF form:

    uv run examples/UniVTAC/convert_sim_to_lerobot.py \
        --hdf5-dir /home/tione/notebook/data/UniVTAC/sim/pull_drawer_GelSight \
        --output-dir /run/ti/pi-training-data/lerobot/simulation/pull_drawer_gelsight_delta_eef \
        --control-mode delta_eef \
        --overwrite

Current implementation:

    abs_joint:
        state   = [franka_joint(7), gripper_q(1)]
        actions = next-frame absolute joint target with the same layout

    delta_joint:
        state   = [franka_joint(7), gripper_q(1)]
        actions = [target_joint(7) - current_joint(7), target_gripper_q(1)]

    delta_eef:
        state   = [eef_pos(3), eef_quat_xyzw(4), gripper_q(1)]
        actions = [delta_xyz(3), delta_rotvec(3), target_gripper_q(1)]

The terminal frame is kept. Its action target is clamped to the final frame,
matching the current Insert USB converter behavior.

LeRobot feature names are kept close to the real Insert USB converter:

    image                <- observation/head/rgb
    wrist_image          <- observation/wrist/rgb
    left_tactile_image   <- tactile/left_tactile/rgb_marker
    right_tactile_image  <- tactile/right_tactile/rgb_marker

Simulation-specific notes:

* HDF5 image datasets are JPEG byte streams. For these UniVTAC sim files,
  cv2.imdecode returns the visually correct channel order; do not apply
  BGR->RGB unless a new dataset is explicitly verified to need it.
* embodiment/joint is expected to be [7 arm joints, left finger, right finger].
  Use --gripper-mode left/right/mean to choose the scalar gripper value.
* embodiment/ee is expected to be [x, y, z, qw, qx, qy, qz].
* Each file must provide HDF5 attr "instruction" unless --task-prompt is passed.
  The per-episode instruction is written to the LeRobot task field.
* delta_eef uses world-frame delta rotation:
  R_delta = R_target @ R_current.T.
* delta_joint uses the current selected frame as the joint delta base.
* abs_eef is intentionally left as a future mode.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import io
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_OUTPUT_ROOT = Path(os.environ.get("HF_LEROBOT_HOME", "/run/ti/pi-training-data/lerobot"))
DEFAULT_CONTROL_MODE = "delta_eef"
DEFAULT_FPS = 60
DEFAULT_IMAGE_SIZE = (224, 224)
PROMPT_ATTR = "instruction"
PROMPT_CONTEXT_ATTR = "episode_context_json"


IMAGE_STREAMS = {
    "image": "observation/head/rgb",
    "wrist_image": "observation/wrist/rgb",
    "left_tactile_image": "tactile/left_tactile/rgb_marker",
    "right_tactile_image": "tactile/right_tactile/rgb_marker",
}

BASE_REQUIRED_DATASETS = (
    "step",
    "embodiment/joint",
    *IMAGE_STREAMS.values(),
)

CONTROL_MODE_DIMS = {
    "abs_joint": (8, 8),
    "delta_joint": (8, 8),
    "delta_eef": (8, 7),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UniVTAC simulated HDF5 episodes to LeRobot format.")
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        required=True,
        help="Input directory containing one .hdf5 file per episode.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo id stored in metadata. Default is derived from --output-dir or --hdf5-dir.",
    )
    parser.add_argument(
        "--control-mode",
        choices=tuple(CONTROL_MODE_DIMS),
        default=DEFAULT_CONTROL_MODE,
        help=(
            "State/action representation to write. "
            "abs_joint writes state/action [joint7, gripper]. "
            "delta_joint writes state [joint7, gripper] and action [delta_joint7, target_gripper]. "
            "delta_eef writes state [xyz, quat_xyzw, gripper] and action [dxyz, drotvec, target_gripper]. "
            f"Default: {DEFAULT_CONTROL_MODE}."
        ),
    )
    parser.add_argument(
        "--gripper-mode",
        choices=("left", "right", "mean"),
        default="left",
        help=(
            "Scalar gripper value used in state/action. 'left' uses embodiment/joint[:, 7], "
            "'right' uses column 8, and 'mean' averages both fingers. Default: left."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Base LeRobot output directory used when --output-dir is omitted and for deriving "
            f"repo id from --output-dir. Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Exact local LeRobot dataset directory to write, e.g. "
            "/run/ti/pi-training-data/lerobot/simulation/pull_drawer_gelsight_delta_eef."
        ),
    )
    parser.add_argument(
        "--task-prompt",
        default=None,
        help=(
            "Override task string stored in every episode. If omitted, each HDF5 file must provide "
            f"'{PROMPT_ATTR}' attr, or '{PROMPT_CONTEXT_ATTR}' containing that field."
        ),
    )
    parser.add_argument(
        "--episode-ids",
        nargs="*",
        default=None,
        help="Optional HDF5 episode ids to convert, e.g. 2 3 4. The .hdf5 suffix is optional.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N sorted episodes.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth saved frame. Default: 1.",
    )
    parser.add_argument(
        "--target-offset",
        type=int,
        default=1,
        help=(
            "Absolute action target offset in raw frames. "
            "Default 1 means actions at raw frame t are state at raw frame t + 1."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "FPS recorded in LeRobot metadata. Default: infer from saved step "
            f"deltas and frame stride, fallback {DEFAULT_FPS}."
        ),
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_IMAGE_SIZE,
        help=f"Resize all image streams before writing. Default: {DEFAULT_IMAGE_SIZE[0]} {DEFAULT_IMAGE_SIZE[1]}.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=5,
        help="LeRobot image writer processes. Default: 5.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="LeRobot image writer threads. Default: 10.",
    )
    parser.add_argument(
        "--read-retries",
        type=int,
        default=3,
        help="Number of retries for transient HDF5 image read errors. Default: 3.",
    )
    parser.add_argument(
        "--read-retry-delay",
        type=float,
        default=0.5,
        help="Seconds to wait between HDF5 image read retries. Default: 0.5.",
    )
    parser.add_argument(
        "--episode-retries",
        type=int,
        default=3,
        help="Number of times to retry an episode from scratch after a conversion failure. Default: 3.",
    )
    parser.add_argument(
        "--episode-retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before retrying a failed episode. Default: 2.0.",
    )
    parser.add_argument(
        "--bad-image-policy",
        choices=("error", "previous", "zero"),
        default="error",
        help=(
            "What to do when a JPEG frame cannot be read or decoded after retries. "
            "'previous' reuses the previous image from the same stream, falling back to zeros for the first frame. "
            "Default: error."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output dataset directory before conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned dataset without writing.",
    )
    return parser.parse_args()


def progress(items: Iterable, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def repo_id_to_path(repo_id: str) -> Path:
    return Path(*repo_id.split("/"))


def resolve_input_args(args: argparse.Namespace) -> tuple[Path, str | None, str | None]:
    return args.hdf5_dir.expanduser().resolve(), args.repo_id, args.task_prompt


def default_repo_id_from_hdf5_dir(hdf5_dir: Path, control_mode: str) -> str:
    return f"simulation/{hdf5_dir.name}_sim_{control_mode}"


def repo_id_from_output_dir(output_dir: Path, output_root: Path) -> str:
    try:
        relative_output_dir = output_dir.relative_to(output_root)
    except ValueError:
        if output_dir.parent.name:
            return f"{output_dir.parent.name}/{output_dir.name}"
        return f"local/{output_dir.name}"

    if not relative_output_dir.parts:
        raise ValueError("--output-dir must point to a dataset directory, not exactly --output-root.")
    if len(relative_output_dir.parts) == 1:
        return f"local/{relative_output_dir.parts[0]}"
    return relative_output_dir.as_posix()


def resolve_output_args(
    args: argparse.Namespace,
    hdf5_dir: Path,
    repo_id: str | None,
) -> tuple[Path, str]:
    output_root = args.output_root.expanduser().resolve()
    if args.output_dir is not None:
        dataset_root = args.output_dir.expanduser().resolve()
        resolved_repo_id = repo_id or repo_id_from_output_dir(dataset_root, output_root)
        return dataset_root, resolved_repo_id

    resolved_repo_id = repo_id or default_repo_id_from_hdf5_dir(hdf5_dir, args.control_mode)
    return output_root / repo_id_to_path(resolved_repo_id), resolved_repo_id


def repo_id_for_control_mode(repo_id: str, control_mode: str) -> str:
    if control_mode == "abs_joint":
        return repo_id
    if repo_id.endswith("_abs_joint"):
        return f"{repo_id.removesuffix('_abs_joint')}_{control_mode}"
    return f"{repo_id}_{control_mode}"


def natural_hdf5_path_key(path: Path, root: Path) -> tuple:
    rel_path = path.relative_to(root)
    key = []
    for part in rel_path.parts:
        stem = Path(part).stem if part.endswith(".hdf5") else part
        if stem.isdigit():
            key.append((0, int(stem)))
        else:
            key.append((1, part))
    return tuple(key)


def resolve_episode_id_path(hdf5_dir: Path, episode_id: str) -> Path:
    episode_path = Path(episode_id)
    if episode_path.suffix != ".hdf5":
        episode_path = episode_path.with_suffix(".hdf5")

    direct_path = hdf5_dir / episode_path
    if direct_path.is_file():
        return direct_path

    matches = sorted(
        hdf5_dir.rglob(episode_path.name),
        key=lambda path: natural_hdf5_path_key(path, hdf5_dir),
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        relative_matches = [str(path.relative_to(hdf5_dir)) for path in matches[:20]]
        raise ValueError(
            f"Episode id {episode_id!r} matched multiple files under {hdf5_dir}: {relative_matches}. "
            "Use a relative path from --hdf5-dir, such as GelSight/cube/0."
        )

    return direct_path


def sorted_hdf5_paths(hdf5_dir: Path, episode_ids: list[str] | None) -> list[Path]:
    if episode_ids:
        paths = [resolve_episode_id_path(hdf5_dir, episode_id) for episode_id in episode_ids]
    else:
        paths = sorted(hdf5_dir.rglob("*.hdf5"), key=lambda path: natural_hdf5_path_key(path, hdf5_dir))

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing HDF5 episode files: {missing}")
    if not paths:
        raise FileNotFoundError(f"No .hdf5 episodes found in {hdf5_dir}")
    return paths


def selected_indices(num_frames: int, frame_stride: int) -> np.ndarray:
    if frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {frame_stride}")

    indices = np.arange(0, num_frames, frame_stride, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError(
            f"Need at least 2 selected frames, got {len(indices)} from {num_frames} frames with stride {frame_stride}"
        )
    return indices


def decode_sim_rgb_jpeg(
    dataset: h5py.Dataset,
    frame_index: int,
    read_retries: int = 3,
    read_retry_delay: float = 0.5,
) -> np.ndarray:
    """Decode one simulated RGB image."""

    last_error: Exception | None = None
    last_buffer_info = ""
    for attempt in range(max(1, read_retries + 1)):
        try:
            buffer = np.frombuffer(dataset[frame_index], dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is not None:
                return image

            last_buffer_info = f"buffer length={len(buffer)}, first bytes={buffer[:16].tolist()}"
            try:
                pil_image = Image.open(io.BytesIO(buffer.tobytes())).convert("RGB")
                # Match cv2.imdecode output for these sim files. The source data
                # was encoded through OpenCV from arrays that should be kept as-is.
                return np.asarray(pil_image)[:, :, ::-1].copy()
            except Exception as pil_exc:
                last_error = ValueError(
                    f"OpenCV and PIL failed to decode image frame {frame_index} from {dataset.name}; "
                    f"{last_buffer_info}; PIL error: {pil_exc}"
                )
        except OSError as exc:
            last_error = exc

        if attempt < read_retries:
            time.sleep(read_retry_delay)

    raise ValueError(
        f"Failed to read/decode {dataset.name}[{frame_index}] after {read_retries + 1} attempts; {last_buffer_info}"
    ) from last_error


def resize_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}")
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def required_datasets(control_mode: str) -> tuple[str, ...]:
    if control_mode in {"abs_joint", "delta_joint"}:
        return BASE_REQUIRED_DATASETS
    if control_mode == "delta_eef":
        return (*BASE_REQUIRED_DATASETS, "embodiment/ee")
    raise ValueError(f"Unsupported control mode: {control_mode}")


def validate_hdf5_episode(h5_file: h5py.File, path: Path, control_mode: str) -> int:
    required = required_datasets(control_mode)
    missing = [key for key in required if key not in h5_file]
    if missing:
        raise KeyError(f"{path}: missing required datasets {missing}")

    num_frames = int(h5_file["step"].shape[0])
    if num_frames < 2:
        raise ValueError(f"{path}: expected at least 2 frames, got {num_frames}")

    for key in required:
        length = int(h5_file[key].shape[0])
        if length != num_frames:
            raise ValueError(f"{path}: {key} length {length} does not match step length {num_frames}")

    joint_shape = h5_file["embodiment/joint"].shape
    if len(joint_shape) != 2 or joint_shape[1] < 9:
        raise ValueError(f"{path}: expected embodiment/joint shape [T, >=9], got {joint_shape}")
    if control_mode == "delta_eef":
        ee_shape = h5_file["embodiment/ee"].shape
        if len(ee_shape) != 2 or ee_shape[1] != 7:
            raise ValueError(f"{path}: expected embodiment/ee shape [T, 7], got {ee_shape}")

    return num_frames


def image_shape(h5_file: h5py.File, h5_key: str, image_size: tuple[int, int]) -> tuple[int, int, int]:
    image = resize_image(decode_sim_rgb_jpeg(h5_file[h5_key], 0), image_size)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image at {h5_key}, got shape {image.shape}")
    return tuple(image.shape)


def make_gripper_q(joint: np.ndarray, gripper_mode: str) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[1] < 9:
        raise ValueError(f"Expected joint shape [T, >=9], got {joint.shape}")

    left_gripper_q = joint[:, 7:8]
    right_gripper_q = joint[:, 8:9]

    if gripper_mode == "left":
        gripper_q = left_gripper_q
    elif gripper_mode == "right":
        gripper_q = right_gripper_q
    elif gripper_mode == "mean":
        gripper_q = 0.5 * (left_gripper_q + right_gripper_q)
    else:
        raise ValueError(f"Unsupported --gripper-mode: {gripper_mode}")

    return gripper_q.astype(np.float32)


def make_abs_joint_state(joint: np.ndarray, gripper_mode: str) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[1] < 9:
        raise ValueError(f"Expected joint shape [T, >=9], got {joint.shape}")

    arm_q = joint[:, :7]
    gripper_q = make_gripper_q(joint, gripper_mode)
    return np.concatenate([arm_q, gripper_q], axis=1).astype(np.float32)


def make_delta_joint_actions(
    joint: np.ndarray,
    source_indices: np.ndarray,
    target_offset: int,
    gripper_mode: str,
) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"--target-offset must be positive, got {target_offset}")

    joint = np.asarray(joint, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[1] < 9:
        raise ValueError(f"Expected joint shape [T, >=9], got {joint.shape}")

    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.minimum(source_indices + target_offset, len(joint) - 1)

    delta_q = joint[target_indices, :7] - joint[source_indices, :7]
    target_gripper_q = make_gripper_q(joint, gripper_mode)[target_indices]
    return np.concatenate([delta_q, target_gripper_q], axis=1).astype(np.float32)


def rotation_from_quat_wxyz(quat_wxyz: np.ndarray) -> Rotation:
    from scipy.spatial.transform import Rotation
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"Expected quat_wxyz shape [T, 4], got {quat_wxyz.shape}")
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw)


def quat_xyzw_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"Expected quat_wxyz shape [T, 4], got {quat_wxyz.shape}")
    return quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)


def make_eef_state(joint: np.ndarray, ee: np.ndarray, gripper_mode: str) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float32)
    ee = np.asarray(ee, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[1] < 9:
        raise ValueError(f"Expected joint shape [T, >=9], got {joint.shape}")
    if ee.ndim != 2 or ee.shape[1] != 7:
        raise ValueError(f"Expected ee shape [T, 7], got {ee.shape}")
    if joint.shape[0] != ee.shape[0]:
        raise ValueError(f"joint length {joint.shape[0]} does not match ee length {ee.shape[0]}")

    eef_pos = ee[:, :3]
    eef_quat_xyzw = quat_xyzw_from_quat_wxyz(ee[:, 3:7])
    gripper_q = make_gripper_q(joint, gripper_mode)
    return np.concatenate([eef_pos, eef_quat_xyzw, gripper_q], axis=1).astype(np.float32)


def make_delta_eef_actions(
    joint: np.ndarray,
    ee: np.ndarray,
    source_indices: np.ndarray,
    target_offset: int,
    gripper_mode: str,
) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"--target-offset must be positive, got {target_offset}")

    joint = np.asarray(joint, dtype=np.float32)
    ee = np.asarray(ee, dtype=np.float32)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.minimum(source_indices + target_offset, len(ee) - 1)

    source_ee = ee[source_indices]
    target_ee = ee[target_indices]
    delta_xyz = target_ee[:, :3] - source_ee[:, :3]

    source_rot = rotation_from_quat_wxyz(source_ee[:, 3:7])
    target_rot = rotation_from_quat_wxyz(target_ee[:, 3:7])
    delta_rotvec = (target_rot * source_rot.inv()).as_rotvec().astype(np.float32)

    gripper_q = make_gripper_q(joint, gripper_mode)
    target_gripper_q = gripper_q[target_indices]
    return np.concatenate([delta_xyz, delta_rotvec, target_gripper_q], axis=1).astype(np.float32)


def make_state_targets(state: np.ndarray, source_indices: np.ndarray, target_offset: int) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"--target-offset must be positive, got {target_offset}")
    indices = np.asarray(source_indices, dtype=np.int64) + target_offset
    indices = np.minimum(indices, len(state) - 1)
    return state[indices].astype(np.float32)


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def read_episode_prompt(h5_file: h5py.File, prompt_override: str | None, path: Path) -> str:
    if prompt_override is not None:
        return prompt_override

    instruction = h5_file.attrs.get(PROMPT_ATTR)
    if instruction is not None:
        prompt = decode_h5_string(instruction).strip()
        if prompt:
            return prompt

    context_json = h5_file.attrs.get(PROMPT_CONTEXT_ATTR)
    if context_json is not None:
        context = json.loads(decode_h5_string(context_json))
        prompt = str(context.get(PROMPT_ATTR, "")).strip()
        if prompt:
            return prompt

    raise KeyError(
        f"{path}: missing task prompt. Expected HDF5 attr {PROMPT_ATTR!r}, "
        f"or attr {PROMPT_CONTEXT_ATTR!r} containing {PROMPT_ATTR!r}, or pass --task-prompt."
    )


def inspect_episode(path: Path, image_size: tuple[int, int], control_mode: str, task_prompt: str | None) -> dict:
    with h5py.File(path, "r") as h5_file:
        num_frames = validate_hdf5_episode(h5_file, path, control_mode)
        episode_prompt = read_episode_prompt(h5_file, task_prompt, path)
        image_shapes = {
            feature_name: image_shape(h5_file, h5_key, image_size) for feature_name, h5_key in IMAGE_STREAMS.items()
        }
        step = h5_file["step"][()]

    return {
        "num_frames": num_frames,
        "task_prompt": episode_prompt,
        "image_shapes": image_shapes,
        "step": step,
    }


def infer_fps(step: np.ndarray, frame_stride: int, fallback: int) -> int:
    if len(step) < 2:
        return fallback
    deltas = np.diff(step)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return fallback

    # Base sim dt is 1/120. A saved-step delta of 2 therefore means 60 FPS.
    raw_fps = round(120.0 / float(np.median(deltas)))
    return max(1, raw_fps // max(1, frame_stride))


def build_features(image_shapes: dict[str, tuple[int, int, int]], control_mode: str) -> dict:
    features = {}
    for feature_name, shape in image_shapes.items():
        features[feature_name] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    state_dim, action_dim = CONTROL_MODE_DIMS[control_mode]
    features["state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["actions"],
    }
    return features


def make_states_and_actions_for_mode(
    h5_file: h5py.File,
    control_mode: str,
    indices: np.ndarray,
    target_offset: int,
    gripper_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    joint = h5_file["embodiment/joint"][()]
    if control_mode == "abs_joint":
        state_all = make_abs_joint_state(joint, gripper_mode)
        return state_all[indices], make_state_targets(state_all, indices, target_offset)
    if control_mode == "delta_joint":
        state_all = make_abs_joint_state(joint, gripper_mode)
        actions = make_delta_joint_actions(joint, indices, target_offset, gripper_mode)
        return state_all[indices], actions
    if control_mode == "delta_eef":
        ee = h5_file["embodiment/ee"][()]
        state_all = make_eef_state(joint, ee, gripper_mode)
        actions = make_delta_eef_actions(joint, ee, indices, target_offset, gripper_mode)
        return state_all[indices], actions
    raise ValueError(f"Unsupported control mode: {control_mode}")


def convert_episode(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    task_prompt: str | None,
    frame_stride: int,
    target_offset: int,
    image_size: tuple[int, int],
    control_mode: str,
    gripper_mode: str,
    read_retries: int,
    read_retry_delay: float,
    bad_image_policy: str,
) -> int:
    with h5py.File(hdf5_path, "r") as h5_file:
        num_frames = validate_hdf5_episode(h5_file, hdf5_path, control_mode)
        episode_prompt = read_episode_prompt(h5_file, task_prompt, hdf5_path)
        indices = selected_indices(num_frames, frame_stride)

        states, actions = make_states_and_actions_for_mode(h5_file, control_mode, indices, target_offset, gripper_mode)
        last_images: dict[str, np.ndarray] = {}
        bad_image_count = 0

        for out_idx, raw_idx in enumerate(indices):
            frame = {
                "state": states[out_idx],
                "actions": actions[out_idx],
                "task": episode_prompt,
            }
            for feature_name, h5_key in IMAGE_STREAMS.items():
                try:
                    image = resize_image(
                        decode_sim_rgb_jpeg(
                            h5_file[h5_key],
                            int(raw_idx),
                            read_retries=read_retries,
                            read_retry_delay=read_retry_delay,
                        ),
                        image_size,
                    )
                except Exception as exc:
                    if bad_image_policy == "error":
                        raise RuntimeError(f"{hdf5_path}: failed to decode {h5_key}[{int(raw_idx)}]") from exc

                    bad_image_count += 1
                    if bad_image_policy == "previous" and feature_name in last_images:
                        image = last_images[feature_name].copy()
                        replacement = "previous"
                    else:
                        width, height = image_size
                        image = np.zeros((height, width, 3), dtype=np.uint8)
                        replacement = "zero"

                    if bad_image_count <= 20 or bad_image_count % 100 == 0:
                        print(
                            f"Warning: {hdf5_path.name} {h5_key}[{int(raw_idx)}] failed "
                            f"({type(exc).__name__}: {exc}); using {replacement} image."
                        )

                frame[feature_name] = image
                last_images[feature_name] = image
            dataset.add_frame(frame)

        if bad_image_count:
            print(f"Warning: replaced {bad_image_count} bad image frame(s) in {hdf5_path}")

    dataset.save_episode()
    return len(indices)


def clear_partial_episode(dataset: LeRobotDataset) -> None:
    if getattr(dataset, "episode_buffer", None) is None:
        return

    try:
        wait_image_writer = getattr(dataset, "_wait_image_writer", None)
        if wait_image_writer is not None:
            wait_image_writer()
        dataset.clear_episode_buffer()
    except Exception as exc:
        print(f"Warning: failed to clear partial episode buffer after conversion error: {exc}")
        dataset.episode_buffer = dataset.create_episode_buffer()


def convert_episode_with_retries(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    episode_retries: int,
    episode_retry_delay: float,
    **convert_kwargs,
) -> int:
    if episode_retries < 0:
        raise ValueError(f"--episode-retries must be non-negative, got {episode_retries}")

    attempts = episode_retries + 1
    for attempt in range(1, attempts + 1):
        episodes_before = dataset.meta.total_episodes
        try:
            return convert_episode(dataset, hdf5_path, **convert_kwargs)
        except Exception as exc:
            if dataset.meta.total_episodes != episodes_before:
                raise RuntimeError(
                    f"{hdf5_path}: conversion failed after the episode may have been saved; "
                    "not retrying to avoid duplicating an episode."
                ) from exc

            clear_partial_episode(dataset)
            if attempt >= attempts:
                raise

            print(
                f"Warning: {hdf5_path.name} failed on attempt {attempt}/{attempts} "
                f"({type(exc).__name__}: {exc}); retrying after {episode_retry_delay}s."
            )
            time.sleep(episode_retry_delay)

    raise RuntimeError(f"{hdf5_path}: failed after {attempts} attempts")


def main() -> None:
    args = parse_args()
    hdf5_dir, repo_id, task_prompt = resolve_input_args(args)
    dataset_root, repo_id = resolve_output_args(args, hdf5_dir, repo_id)
    image_size = tuple(args.image_size)

    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Missing HDF5 directory: {hdf5_dir}")

    episodes = sorted_hdf5_paths(hdf5_dir, args.episode_ids)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    first_episode = inspect_episode(episodes[0], image_size, args.control_mode, task_prompt)
    indices = selected_indices(first_episode["num_frames"], args.frame_stride)
    fps = args.fps or infer_fps(first_episode["step"], args.frame_stride, DEFAULT_FPS)
    features = build_features(first_episode["image_shapes"], args.control_mode)

    print("Task preset: explicit hdf5-dir/output-dir")
    if task_prompt is None:
        print(f"Task prompt source: HDF5 attr {PROMPT_ATTR!r} (fallback {PROMPT_CONTEXT_ATTR!r})")
        print(f"First episode task prompt: {first_episode['task_prompt']!r}")
    else:
        print(f"Task prompt override/fallback: {task_prompt!r}")
    print(f"HDF5 dir: {hdf5_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Repo id: {repo_id}")
    print(f"Output dataset root: {dataset_root}")
    print(f"Control mode: {args.control_mode}")
    print(f"Gripper mode: {args.gripper_mode}")
    print(f"FPS: {fps}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Selected frames in first episode: {len(indices)} / {first_episode['num_frames']}")
    print(f"Target offset: {args.target_offset} raw frame(s)")
    if args.control_mode == "delta_eef":
        print("Action semantics: [delta xyz, world-frame delta rotvec, target gripper]")
        print("Terminal frame action: zero xyz/rot delta with final gripper target")
    elif args.control_mode == "delta_joint":
        print("Action semantics: [target joint - current joint, target gripper]")
        print("Terminal frame action: zero joint delta with final gripper target")
    else:
        print("Terminal frame action: clamped to final state")
    print(f"Episode retries: {args.episode_retries}")
    print(f"Bad image policy: {args.bad_image_policy}")
    print(f"Image size: {image_size[0]}x{image_size[1]} (width x height)")
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: dtype={value['dtype']}, shape={value['shape']}")

    if args.dry_run:
        print("Dry run only; no LeRobot dataset was written.")
        return

    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {dataset_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type="franka",
        fps=fps,
        features=features,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    for hdf5_path in progress(episodes, desc="Converting episodes"):
        total_frames += convert_episode_with_retries(
            dataset,
            hdf5_path,
            episode_retries=args.episode_retries,
            episode_retry_delay=args.episode_retry_delay,
            task_prompt=task_prompt,
            frame_stride=args.frame_stride,
            target_offset=args.target_offset,
            image_size=image_size,
            control_mode=args.control_mode,
            gripper_mode=args.gripper_mode,
            read_retries=args.read_retries,
            read_retry_delay=args.read_retry_delay,
            bad_image_policy=args.bad_image_policy,
        )

    print(f"Done. Wrote {len(episodes)} episodes and {total_frames} frames to {dataset_root}")


if __name__ == "__main__":
    main()
