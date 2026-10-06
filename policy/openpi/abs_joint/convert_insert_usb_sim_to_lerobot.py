#!/usr/bin/env python3
"""
Convert OpenViTac simulated Insert USB HDF5 episodes to a LeRobot dataset.

Recommended usage from the OpenPI repo root:

    cd /shareNFS_40/wyf/workplace/openpi
    export HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot

    uv run examples/insert_usb/convert_insert_usb_sim_to_lerobot.py \
        --hdf5-dir /shareNFS_40/wyf/data/OpenViTac/sim/Insert_USB_Tactile \
        --repo-id wyf/insert_usb_sim_abs_eef \
        --control-mode abs_eef \
        --output-root /shareNFS_40/wyf/data/lerobot \
        --image-size 224 224 \
        --overwrite

To convert the GelSight simulation data to absolute joint targets:

    uv run examples/insert_usb/convert_insert_usb_sim_to_lerobot.py \
        --hdf5-dir /shareNFS_40/wyf/data/OpenViTac/sim/insert_USB_GelSight \
        --repo-id wyf/insert_usb_sim_abs_joint \
        --control-mode abs_joint \
        --output-root /shareNFS_40/wyf/data/lerobot \
        --image-size 224 224 \
        --drop-terminal-target-frame \
        --overwrite

Default output:

    /shareNFS_40/wyf/data/lerobot/wyf/insert_usb_sim_abs_eef

This converter keeps the same LeRobot feature names as the real Insert USB
converter:

    image                <- observation/head/rgb
    wrist_image          <- observation/wrist/rgb
    left_tactile_image   <- tactile/left_tactile/rgb_marker
    right_tactile_image  <- tactile/right_tactile/rgb_marker

    abs_eef:
        state/actions = [eef_pos(3), eef_rot6d(6), gripper_q(1)]      # shape [10]

    abs_joint:
        state/actions = [franka_joint(7), left_gripper_q(1)]          # shape [8]

Important simulation-specific notes:

* HDF5 image datasets are JPEG byte streams. `cv2.imdecode` returns the
  visually correct channel order for this dataset; do not apply BGR->RGB.
* `embodiment/ee` stores quaternions as [qw, qx, qy, qz]. Convert to xyzw
  before calling scipy Rotation.
* `embodiment/joint` is [7 arm joints, left finger, right finger]. For abs_eef
  this script keeps the right finger to match the existing EEF dataset. For
  abs_joint it keeps the first finger value, matching the ACT preprocessing
  convention of `embodiment/joint[:, 0:8]`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import shutil
import time

import cv2
import h5py
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError as exc:
    if exc.name not in ("lerobot.common", "lerobot.common.datasets", "lerobot.common.datasets.lerobot_dataset"):
        raise
    # LeRobot 0.3.x retains dataset format v2.1 under its new module layout.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_HDF5_DIR = Path("/shareNFS_40/wyf/data/OpenViTac/sim/Insert_USB_Tactile")
DEFAULT_OUTPUT_ROOT = Path("/shareNFS_40/wyf/data/lerobot")
DEFAULT_REPO_ID = "wyf/insert_usb_sim_abs_eef"
DEFAULT_TASK_PROMPT = "pick and insert the USB"
DEFAULT_FPS = 60
DEFAULT_IMAGE_SIZE = (224, 224)

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
    "abs_eef": 10,
    "abs_joint": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenViTac simulated Insert USB HDF5 episodes to LeRobot format."
    )
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=DEFAULT_HDF5_DIR,
        help=f"Input directory containing one .hdf5 file per episode. Default: {DEFAULT_HDF5_DIR}",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"LeRobot repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--control-mode",
        choices=tuple(CONTROL_MODE_DIMS),
        default="abs_eef",
        help="State/action representation to write. Default: abs_eef.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Base LeRobot output directory. The dataset is written to "
            "<output-root>/<repo-id>. Default: /shareNFS_40/wyf/data/lerobot"
        ),
    )
    parser.add_argument(
        "--task-prompt",
        default=DEFAULT_TASK_PROMPT,
        help=f"Task string stored in every episode. Default: {DEFAULT_TASK_PROMPT!r}",
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
        "--drop-terminal-target-frame",
        action="store_true",
        help=(
            "Drop selected frames whose target would be clamped to the final frame. "
            "Use this for exact ACT-style state=joint[:-1], action=joint[1:] pairs."
        ),
    )
    parser.add_argument(
        "--joint-gripper-index",
        type=int,
        default=7,
        help=(
            "Column in embodiment/joint to use as the abs_joint gripper value. "
            "Default 7 means the first finger after the 7 Franka joints."
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


def sorted_hdf5_paths(hdf5_dir: Path, episode_ids: list[str] | None) -> list[Path]:
    if episode_ids:
        paths = []
        for episode_id in episode_ids:
            episode_path = hdf5_dir / episode_id
            if episode_path.suffix != ".hdf5":
                episode_path = episode_path.with_suffix(".hdf5")
            paths.append(episode_path)
    else:
        paths = sorted(
            hdf5_dir.glob("*.hdf5"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing HDF5 episode files: {missing}")
    if not paths:
        raise FileNotFoundError(f"No .hdf5 episodes found in {hdf5_dir}")
    return paths


def selected_indices(
    num_frames: int,
    frame_stride: int,
    target_offset: int = 1,
    drop_terminal_target_frame: bool = False,
) -> np.ndarray:
    if frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {frame_stride}")
    if target_offset <= 0:
        raise ValueError(f"--target-offset must be positive, got {target_offset}")

    stop = num_frames - target_offset if drop_terminal_target_frame else num_frames
    indices = np.arange(0, stop, frame_stride, dtype=np.int64)
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
    """Decode one simulated RGB image.

    The simulated data was encoded with OpenCV from arrays that already looked
    correct in RGB order. Applying BGR->RGB here visibly swaps red and blue.
    """

    last_error: OSError | None = None
    for attempt in range(max(1, read_retries + 1)):
        try:
            buffer = np.frombuffer(dataset[frame_index], dtype=np.uint8)
            break
        except OSError as exc:
            last_error = exc
            if attempt >= read_retries:
                raise OSError(
                    f"Failed to read {dataset.name}[{frame_index}] after {read_retries + 1} attempts"
                ) from exc
            time.sleep(read_retry_delay)
    else:  # pragma: no cover
        raise OSError(f"Failed to read {dataset.name}[{frame_index}]") from last_error

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        first_bytes = buffer[:16].tolist()
        raise ValueError(
            f"OpenCV failed to decode image frame {frame_index} from {dataset.name}; "
            f"buffer length={len(buffer)}, first bytes={first_bytes}"
        )
    return image


def resize_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}")
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def required_datasets(control_mode: str) -> tuple[str, ...]:
    if control_mode == "abs_eef":
        return (*BASE_REQUIRED_DATASETS, "embodiment/ee")
    if control_mode == "abs_joint":
        return BASE_REQUIRED_DATASETS
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
    if control_mode == "abs_eef":
        ee_shape = h5_file["embodiment/ee"].shape
        if len(ee_shape) != 2 or ee_shape[1] != 7:
            raise ValueError(f"{path}: expected embodiment/ee shape [T, 7], got {ee_shape}")

    return num_frames


def image_shape(h5_file: h5py.File, h5_key: str, image_size: tuple[int, int]) -> tuple[int, int, int]:
    image = resize_image(decode_sim_rgb_jpeg(h5_file[h5_key], 0), image_size)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image at {h5_key}, got shape {image.shape}")
    return tuple(image.shape)


def rot6d_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"Expected quat_wxyz shape [T, 4], got {quat_wxyz.shape}")
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    matrices = Rotation.from_quat(quat_xyzw).as_matrix()
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1).astype(np.float32)


def make_abs_eef_state(joint: np.ndarray, ee: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float32)
    ee = np.asarray(ee, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[1] < 9:
        raise ValueError(f"Expected joint shape [T, >=9], got {joint.shape}")
    if ee.ndim != 2 or ee.shape[1] != 7:
        raise ValueError(f"Expected ee shape [T, 7], got {ee.shape}")
    if joint.shape[0] != ee.shape[0]:
        raise ValueError(f"joint length {joint.shape[0]} does not match ee length {ee.shape[0]}")

    eef_pos = ee[:, :3]
    eef_quat_wxyz = ee[:, 3:7]
    eef_rot6d = rot6d_from_quat_wxyz(eef_quat_wxyz)
    gripper_q = joint[:, 8:9]
    return np.concatenate([eef_pos, eef_rot6d, gripper_q], axis=1).astype(np.float32)


def make_abs_joint_state(joint: np.ndarray, gripper_index: int) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[1] < 8:
        raise ValueError(f"Expected joint shape [T, >=8], got {joint.shape}")
    if not 7 <= gripper_index < joint.shape[1]:
        raise ValueError(
            f"--joint-gripper-index must select a gripper column in [7, {joint.shape[1] - 1}], got {gripper_index}"
        )

    arm_q = joint[:, :7]
    gripper_q = joint[:, gripper_index : gripper_index + 1]
    return np.concatenate([arm_q, gripper_q], axis=1).astype(np.float32)


def make_state_targets(state: np.ndarray, source_indices: np.ndarray, target_offset: int) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"--target-offset must be positive, got {target_offset}")
    indices = np.asarray(source_indices, dtype=np.int64) + target_offset
    indices = np.minimum(indices, len(state) - 1)
    return state[indices].astype(np.float32)


def inspect_episode(path: Path, image_size: tuple[int, int], control_mode: str) -> dict:
    with h5py.File(path, "r") as h5_file:
        num_frames = validate_hdf5_episode(h5_file, path, control_mode)
        image_shapes = {
            feature_name: image_shape(h5_file, h5_key, image_size)
            for feature_name, h5_key in IMAGE_STREAMS.items()
        }
        step = h5_file["step"][()]

    return {
        "num_frames": num_frames,
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
    raw_fps = int(round(120.0 / float(np.median(deltas))))
    return max(1, raw_fps // max(1, frame_stride))


def build_features(image_shapes: dict[str, tuple[int, int, int]], control_mode: str) -> dict:
    features = {}
    for feature_name, shape in image_shapes.items():
        features[feature_name] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    state_dim = CONTROL_MODE_DIMS[control_mode]
    features["state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["actions"],
    }
    return features


def force_task_context(h5_file):
    # Old datasets did not need this optional metadata to be valid JSON.
    import json
    try:
        context = json.loads(h5_file.attrs.get("episode_context_json", "{}"))
    except (ValueError, TypeError):
        return None
    if isinstance(context, dict) and context.get("task") in {
            "grasp_fragile_chip", "bulb_tightening", "tension_strap", "wipe_vase"}:
        return context
    return None


def episode_indices(h5_file, frame_stride, target_offset, drop_terminal_target_frame,
                    control_mode, joint_gripper_index):
    """Keep legacy sampling; force tasks learn only consecutive POLICY actions."""
    context = force_task_context(h5_file)
    if context is None:
        return selected_indices(len(h5_file["step"]), frame_stride, target_offset,
                                drop_terminal_target_frame)
    if (control_mode, joint_gripper_index, frame_stride, target_offset) != ("abs_joint", 7, 1, 1):
        raise ValueError("Force-task conversion requires abs_joint, gripper index 7, frame stride 1, target offset 1")
    timing = context.get("timing_contract", {})
    if not np.isclose(timing.get("policy_observation_dt_s", -1), 1 / 60):
        raise ValueError("Force-task HDF5 must declare 60 Hz observations")
    phase = np.asarray(h5_file["phase/id"]).reshape(-1)
    steps = np.asarray(h5_file["step"]).reshape(-1)
    terminal = np.flatnonzero(phase == 2)
    policy = np.flatnonzero(phase == 1)
    if (not len(terminal) or str(h5_file["phase"].attrs.get("terminal_reason", "")) != "success"
            or len(policy) < 2 or not np.all(np.diff(steps[policy]) == 2)):
        raise ValueError("Force-task training requires a successful episode with a uniform POLICY clock")
    # The first terminal state can be the final action target. Presentation
    # motions after that state must never become training observations/actions.
    indices = policy[(policy + 1 < len(steps)) & (policy < terminal[0])]
    indices = indices[(indices + 1 <= terminal[0]) & (steps[indices + 1] - steps[indices] == 2)]
    if len(indices) < 2:
        raise ValueError("Force-task episode has fewer than two valid 60 Hz state/action pairs")
    return indices


def make_states_for_mode(h5_file: h5py.File, control_mode: str, joint_gripper_index: int) -> np.ndarray:
    joint = h5_file["embodiment/joint"][()]
    if control_mode == "abs_eef":
        return make_abs_eef_state(joint, h5_file["embodiment/ee"][()])
    if control_mode == "abs_joint":
        return make_abs_joint_state(joint, joint_gripper_index)
    raise ValueError(f"Unsupported control mode: {control_mode}")


def convert_episode(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    task_prompt: str,
    frame_stride: int,
    target_offset: int,
    image_size: tuple[int, int],
    control_mode: str,
    joint_gripper_index: int,
    drop_terminal_target_frame: bool,
    read_retries: int,
    read_retry_delay: float,
    bad_image_policy: str,
) -> int:
    import inspect
    separate_task_argument = "task" in inspect.signature(dataset.add_frame).parameters
    with h5py.File(hdf5_path, "r") as h5_file:
        num_frames = validate_hdf5_episode(h5_file, hdf5_path, control_mode)
        indices = episode_indices(h5_file, frame_stride, target_offset, drop_terminal_target_frame,
                                  control_mode, joint_gripper_index)

        state_all = make_states_for_mode(h5_file, control_mode, joint_gripper_index)
        states = state_all[indices]
        actions = make_state_targets(state_all, indices, target_offset)
        last_images: dict[str, np.ndarray] = {}
        bad_image_count = 0

        for out_idx, raw_idx in enumerate(indices):
            frame = {
                "state": states[out_idx],
                "actions": actions[out_idx],
                "task": task_prompt,
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
            if separate_task_argument:
                frame.pop("task")
                dataset.add_frame(frame, task=task_prompt)
            else:
                dataset.add_frame(frame)

        if bad_image_count:
            print(f"Warning: replaced {bad_image_count} bad image frame(s) in {hdf5_path}")

    dataset.save_episode()
    return len(indices)


def main() -> None:
    args = parse_args()
    hdf5_dir = args.hdf5_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_root = output_root / repo_id_to_path(args.repo_id)
    image_size = tuple(args.image_size)

    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Missing HDF5 directory: {hdf5_dir}")

    episodes = sorted_hdf5_paths(hdf5_dir, args.episode_ids)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    first_episode = inspect_episode(episodes[0], image_size, args.control_mode)
    with h5py.File(episodes[0], "r") as first_h5:
        indices = episode_indices(first_h5, args.frame_stride, args.target_offset,
                                  args.drop_terminal_target_frame, args.control_mode,
                                  args.joint_gripper_index)
        if force_task_context(first_h5) is not None:
            if args.fps not in (None, 60):
                raise ValueError("Force-task datasets require --fps 60")
            args.fps = 60
    fps = args.fps or infer_fps(first_episode["step"], args.frame_stride, DEFAULT_FPS)
    features = build_features(first_episode["image_shapes"], args.control_mode)

    print(f"HDF5 dir: {hdf5_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Repo id: {args.repo_id}")
    print(f"Output dataset root: {dataset_root}")
    print(f"Control mode: {args.control_mode}")
    print(f"FPS: {fps}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Selected frames in first episode: {len(indices)} / {first_episode['num_frames']}")
    print(f"Target offset: {args.target_offset} raw frame(s)")
    print(f"Drop terminal target frame: {args.drop_terminal_target_frame}")
    if args.control_mode == "abs_joint":
        print(f"Joint gripper index: {args.joint_gripper_index}")
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
        repo_id=args.repo_id,
        root=dataset_root,
        robot_type="franka",
        fps=fps,
        features=features,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    for hdf5_path in progress(episodes, desc="Converting episodes"):
        total_frames += convert_episode(
            dataset,
            hdf5_path,
            task_prompt=args.task_prompt,
            frame_stride=args.frame_stride,
            target_offset=args.target_offset,
            image_size=image_size,
            control_mode=args.control_mode,
            joint_gripper_index=args.joint_gripper_index,
            drop_terminal_target_frame=args.drop_terminal_target_frame,
            read_retries=args.read_retries,
            read_retry_delay=args.read_retry_delay,
            bad_image_policy=args.bad_image_policy,
        )

    print(f"Done. Wrote {len(episodes)} episodes and {total_frames} frames to {dataset_root}")


if __name__ == "__main__":
    main()
