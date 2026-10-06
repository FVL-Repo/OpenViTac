#!/usr/bin/env python3
"""
Convert OpenViTac real Insert USB raw episodes to a LeRobot dataset.

Recommended usage from the OpenPI repo root:

    cd /shareNFS_40/wyf/workplace/openpi
    export HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot

    uv run examples/insert_usb/convert_insert_usb_real_to_lerobot.py \
        --raw-dir /shareNFS_40/wyf/OpenViTac/data/real/raw/insert_USB_real \
        --repo-id wyf/insert_usb_new_real_abs_eef \
        --output-root /shareNFS_40/wyf/data/lerobot \
        --image-size 224 224 \
        --overwrite

Default output:

    /shareNFS_40/wyf/data/lerobot/wyf/insert_usb_new_real_abs_eef

The LeRobot dataset stores one action per frame. OpenPI later builds action
chunks automatically with `delta_timestamps` according to the training config's
`action_horizon` and the dataset FPS.

This first version uses absolute EEF targets:

    state   = [eef_pos(3), eef_rot6d(6), gripper_q(1)]      # shape [10]
    actions = [target_eef_pos(3), target_eef_rot6d(6), target_gripper_q(1)]

By default, `actions[t]` is the next selected frame's absolute EEF state. The
last selected frame repeats its final target.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_RAW_DIR = Path("/shareNFS_40/wyf/OpenViTac/data/real/raw/insert_USB_real")
DEFAULT_OUTPUT_ROOT = Path("/shareNFS_40/wyf/data/lerobot")
DEFAULT_REPO_ID = "wyf/insert_usb_new_real_abs_eef"
DEFAULT_TASK_PROMPT = "pick and insert the USB"
DEFAULT_EPISODE_START = 1
DEFAULT_EPISODE_END = 80
DEFAULT_FPS = 20
DEFAULT_IMAGE_SIZE = (224, 224)

IMAGE_STREAMS = {
    "image": Path("main/color"),
    "wrist_image": Path("wrist/color"),
    "left_tactile_image": Path("xense/left/rectify"),
    "right_tactile_image": Path("xense/right/rectify"),
}

REQUIRED_STATE_KEYS = ("eef_pos", "eef_quat", "gripper_q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenViTac real Insert USB raw episodes to LeRobot format."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Input raw task directory. Default: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"LeRobot repo id. Default: {DEFAULT_REPO_ID}",
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
        help="Optional raw episode ids to convert, e.g. 000000 000001.",
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=DEFAULT_EPISODE_START,
        help=(
            "First raw numeric episode id to convert, inclusive. Ignored when --episode-ids is set. "
            f"Default: {DEFAULT_EPISODE_START}."
        ),
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        default=DEFAULT_EPISODE_END,
        help=(
            "Last raw numeric episode id to convert, inclusive. Ignored when --episode-ids is set. "
            f"Default: {DEFAULT_EPISODE_END}."
        ),
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
        help="Keep every Nth raw frame. Default: 1.",
    )
    parser.add_argument(
        "--target-offset",
        type=int,
        default=1,
        help=(
            "Absolute action target offset in selected frames. "
            "Default 1 means actions[t] = state[t + 1]."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "FPS recorded in LeRobot metadata. Default: infer from sample_timestamps "
            f"after frame-stride, fallback {DEFAULT_FPS}."
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


def sorted_episode_dirs(raw_dir: Path, episode_ids: list[str] | None) -> list[Path]:
    if episode_ids:
        episodes = [raw_dir / episode_id for episode_id in episode_ids]
    else:
        episodes = [path for path in raw_dir.iterdir() if path.is_dir() and path.name.isdigit()]
        episodes = sorted(episodes, key=lambda path: int(path.name))

    missing = [str(path) for path in episodes if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing episode directories: {missing}")
    if not episodes:
        raise FileNotFoundError(f"No episode directories found in {raw_dir}")
    return episodes


def filter_episode_range(episodes: list[Path], start: int, end: int) -> list[Path]:
    if start < 0 or end < 0:
        raise ValueError(f"Episode range must be non-negative, got {start}-{end}")
    if start > end:
        raise ValueError(f"--episode-start must be <= --episode-end, got {start}>{end}")

    filtered = [path for path in episodes if start <= int(path.name) <= end]
    if not filtered:
        raise FileNotFoundError(f"No episode directories found in requested numeric range {start}-{end}")
    return filtered


def sorted_jpgs(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")
    return paths


def read_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_rgb_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}")
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_shape(path: Path, image_size: tuple[int, int]) -> tuple[int, int, int]:
    image = read_rgb_image(path)
    image = resize_rgb_image(image, image_size)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image at {path}, got shape {image.shape}")
    return tuple(image.shape)


def validate_frame_names(paths: list[Path], expected_len: int, label: str) -> None:
    if len(paths) != expected_len:
        raise ValueError(f"{label}: expected {expected_len} images, found {len(paths)}")
    for idx, path in enumerate(paths):
        expected = f"{idx:06d}"
        if path.stem != expected:
            raise ValueError(f"{label}: expected {expected}.jpg at index {idx}, got {path.name}")


def load_states(ep_dir: Path) -> dict[str, np.ndarray]:
    states_path = ep_dir / "states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing states.npz: {states_path}")

    with np.load(states_path) as states:
        missing = [key for key in REQUIRED_STATE_KEYS if key not in states]
        if missing:
            raise KeyError(f"{states_path}: missing required keys {missing}")
        return {key: states[key] for key in states.files}


def rot6d_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    matrices = Rotation.from_quat(quat_xyzw).as_matrix()
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1).astype(np.float32)


def make_abs_eef_state(states: dict[str, np.ndarray]) -> np.ndarray:
    eef_pos = np.asarray(states["eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(states["eef_quat"], dtype=np.float32)
    gripper_q = np.asarray(states["gripper_q"], dtype=np.float32).reshape(-1, 1)

    if eef_pos.ndim != 2 or eef_pos.shape[1] != 3:
        raise ValueError(f"Expected eef_pos shape [T, 3], got {eef_pos.shape}")
    if eef_quat.ndim != 2 or eef_quat.shape[1] != 4:
        raise ValueError(f"Expected eef_quat shape [T, 4], got {eef_quat.shape}")
    if gripper_q.shape[0] != eef_pos.shape[0]:
        raise ValueError(f"gripper_q length {gripper_q.shape[0]} does not match eef_pos length {eef_pos.shape[0]}")

    rot6d = rot6d_from_quat_xyzw(eef_quat)
    return np.concatenate([eef_pos, rot6d, gripper_q], axis=1).astype(np.float32)


def inspect_episode(ep_dir: Path, image_size: tuple[int, int]) -> dict:
    states = load_states(ep_dir)
    num_frames = int(states["eef_pos"].shape[0])
    if num_frames < 2:
        raise ValueError(f"{ep_dir}: expected at least 2 frames, got {num_frames}")

    image_paths = {}
    image_shapes = {}
    for feature_name, rel_dir in IMAGE_STREAMS.items():
        paths = sorted_jpgs(ep_dir / rel_dir)
        validate_frame_names(paths, num_frames, f"{ep_dir.name}/{rel_dir}")
        image_paths[feature_name] = paths
        image_shapes[feature_name] = image_shape(paths[0], image_size)

    return {
        "num_frames": num_frames,
        "states": states,
        "image_paths": image_paths,
        "image_shapes": image_shapes,
    }


def infer_fps(states: dict[str, np.ndarray], selected_indices: np.ndarray, fallback: int) -> int:
    timestamps = states.get("sample_timestamps")
    if timestamps is None or len(selected_indices) < 2:
        return fallback

    selected_timestamps = np.asarray(timestamps, dtype=np.float64)[selected_indices]
    deltas = np.diff(selected_timestamps)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return fallback

    fps = round(1.0 / float(np.median(deltas)))
    return max(1, fps)


def selected_indices(num_frames: int, frame_stride: int) -> np.ndarray:
    if frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {frame_stride}")
    indices = np.arange(0, num_frames, frame_stride, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError(
            f"Need at least 2 selected frames, got {len(indices)} from {num_frames} frames with stride {frame_stride}"
        )
    return indices


def make_action_targets(abs_eef_state: np.ndarray, target_offset: int) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"--target-offset must be positive, got {target_offset}")
    indices = np.arange(len(abs_eef_state), dtype=np.int64) + target_offset
    indices = np.minimum(indices, len(abs_eef_state) - 1)
    return abs_eef_state[indices].astype(np.float32)


def build_features(image_shapes: dict[str, tuple[int, int, int]]) -> dict:
    features = {}
    for feature_name, shape in image_shapes.items():
        features[feature_name] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    features["state"] = {
        "dtype": "float32",
        "shape": (10,),
        "names": ["state"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (10,),
        "names": ["actions"],
    }
    return features


def convert_episode(
    dataset: LeRobotDataset,
    ep_dir: Path,
    task_prompt: str,
    frame_stride: int,
    target_offset: int,
    image_size: tuple[int, int],
) -> int:
    episode = inspect_episode(ep_dir, image_size)
    indices = selected_indices(episode["num_frames"], frame_stride)

    abs_eef_state_all = make_abs_eef_state(episode["states"])
    states = abs_eef_state_all[indices]
    actions = make_action_targets(states, target_offset)

    image_paths = episode["image_paths"]
    for out_idx, raw_idx in enumerate(indices):
        frame = {
            "state": states[out_idx],
            "actions": actions[out_idx],
            "task": task_prompt,
        }
        for feature_name, paths in image_paths.items():
            frame[feature_name] = resize_rgb_image(read_rgb_image(paths[int(raw_idx)]), image_size)
        dataset.add_frame(frame)

    dataset.save_episode()
    return len(indices)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_root = output_root / repo_id_to_path(args.repo_id)
    image_size = tuple(args.image_size)

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")

    episodes = sorted_episode_dirs(raw_dir, args.episode_ids)
    if args.episode_ids is None:
        episodes = filter_episode_range(episodes, args.episode_start, args.episode_end)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    first_episode = inspect_episode(episodes[0], image_size)
    indices = selected_indices(first_episode["num_frames"], args.frame_stride)
    fps = args.fps or infer_fps(first_episode["states"], indices, DEFAULT_FPS)
    features = build_features(first_episode["image_shapes"])

    print(f"Raw dir: {raw_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Episode ids: {episodes[0].name}..{episodes[-1].name}")
    print(f"Repo id: {args.repo_id}")
    print(f"Task prompt: {args.task_prompt!r}")
    print(f"Output dataset root: {dataset_root}")
    print(f"FPS: {fps}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Target offset: {args.target_offset} selected frame(s)")
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
    for ep_dir in progress(episodes, desc="Converting episodes"):
        total_frames += convert_episode(
            dataset,
            ep_dir,
            task_prompt=args.task_prompt,
            frame_stride=args.frame_stride,
            target_offset=args.target_offset,
            image_size=image_size,
        )

    print(f"Done. Wrote {len(episodes)} episodes and {total_frames} frames to {dataset_root}")


if __name__ == "__main__":
    main()
