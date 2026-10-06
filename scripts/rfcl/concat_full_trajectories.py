#!/usr/bin/env python3
"""Join Motion Plan prefixes to re-recorded RFCL success suffixes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


CORE_DATASETS = (
    "observation/head/rgb",
    "observation/wrist/rgb",
    "tactile/left_tactile/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/left_tactile/depth",
    "tactile/left_tactile/marker",
    "tactile/left_tactile/pose",
    "tactile/right_tactile/rgb",
    "tactile/right_tactile/rgb_marker",
    "tactile/right_tactile/depth",
    "tactile/right_tactile/marker",
    "tactile/right_tactile/pose",
    "embodiment/joint",
    "embodiment/ee",
    "actor/prism",
    "actor/slot",
    "actor/start_slot",
    "atom/id",
    "atom/tag",
    "phase/id",
    "phase/name",
    "phase/is_boundary",
    "phase/policy_step",
    "phase/sim_step",
    "step",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--suffix-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suffix-stride", type=int, default=2)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def frame_count(handle: h5py.File) -> int:
    missing = [name for name in CORE_DATASETS if name not in handle]
    if missing:
        raise ValueError(f"Missing core datasets in {handle.filename}: {missing}")
    lengths = {name: len(handle[name]) for name in CORE_DATASETS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset lengths disagree in {handle.filename}: {lengths}")
    return next(iter(lengths.values()))


def selected_suffix_indices(length: int, stride: int) -> np.ndarray:
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return np.asarray(indices, dtype=np.int64)


def downsample_transition_valid(
    transition_valid: np.ndarray,
    selected_indices: np.ndarray,
    *,
    first_valid: bool,
) -> np.ndarray:
    """Preserve invalid transitions skipped between selected frames."""
    selected_valid = np.ones(len(selected_indices), dtype=np.int8)
    selected_valid[0] = int(first_valid)
    for output_index in range(1, len(selected_indices)):
        previous = int(selected_indices[output_index - 1])
        current = int(selected_indices[output_index])
        selected_valid[output_index] = int(
            np.all(transition_valid[previous + 1 : current + 1])
        )
    return selected_valid


def output_dtype(prefix: h5py.Dataset, suffix: h5py.Dataset) -> np.dtype:
    if prefix.shape[1:] != suffix.shape[1:]:
        raise ValueError(
            f"Trailing shapes disagree for {prefix.name}: "
            f"{prefix.shape[1:]} != {suffix.shape[1:]}"
        )
    prefix_dtype = np.dtype(prefix.dtype)
    suffix_dtype = np.dtype(suffix.dtype)
    if prefix_dtype.kind == "S" and suffix_dtype.kind == "S":
        return np.dtype(f"S{max(prefix_dtype.itemsize, suffix_dtype.itemsize)}")
    if prefix_dtype != suffix_dtype:
        raise ValueError(
            f"Dtypes disagree for {prefix.name}: {prefix_dtype} != {suffix_dtype}"
        )
    return prefix_dtype


def decode_jpeg(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        encoded = np.asarray(value, dtype=np.uint8).reshape(-1)
    else:
        encoded = np.frombuffer(value, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG at splice boundary")
    return image


def pose_position_error_mm(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first[:3] - second[:3]) * 1000.0)


def boundary_metrics(
    prefix: h5py.File,
    suffix: h5py.File,
    prefix_index: int,
) -> dict[str, float | int]:
    prefix_step = int(prefix["step"][prefix_index])
    suffix_step = int(suffix["step"][0])
    prefix_wrist = decode_jpeg(prefix["observation/wrist/rgb"][prefix_index])
    suffix_wrist = decode_jpeg(suffix["observation/wrist/rgb"][0])
    if prefix_wrist.shape != suffix_wrist.shape:
        suffix_wrist = cv2.resize(
            suffix_wrist,
            (prefix_wrist.shape[1], prefix_wrist.shape[0]),
        )
    return {
        "prefix_last_sim_step": prefix_step,
        "suffix_first_sim_step": suffix_step,
        "sim_step_gap": suffix_step - prefix_step,
        "joint_max_gap_rad": float(
            np.max(
                np.abs(
                    np.asarray(prefix["embodiment/joint"][prefix_index])[:7]
                    - np.asarray(suffix["embodiment/joint"][0])[:7]
                )
            )
        ),
        "ee_position_gap_mm": pose_position_error_mm(
            np.asarray(prefix["embodiment/ee"][prefix_index]),
            np.asarray(suffix["embodiment/ee"][0]),
        ),
        "usb_position_gap_mm": pose_position_error_mm(
            np.asarray(prefix["actor/prism"][prefix_index]),
            np.asarray(suffix["actor/prism"][0]),
        ),
        "slot_position_gap_mm": pose_position_error_mm(
            np.asarray(prefix["actor/slot"][prefix_index]),
            np.asarray(suffix["actor/slot"][0]),
        ),
        "wrist_image_mae": float(
            np.mean(
                np.abs(
                    prefix_wrist.astype(np.float32) - suffix_wrist.astype(np.float32)
                )
            )
        ),
    }


def validate_output(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        frames = frame_count(handle)
        steps = np.asarray(handle["step"], dtype=np.int64)
        if not np.all(np.diff(steps) > 0):
            raise ValueError(f"Non-increasing step sequence in {path}")
        source = np.asarray(handle["provenance/source"], dtype=np.int8)
        if len(source) != frames or np.any(np.diff(source) < 0):
            raise ValueError(f"Invalid source provenance in {path}")
        split_index = int(handle.attrs["splice_prefix_frames"])
        if not (0 < split_index < frames):
            raise ValueError(f"Invalid splice index in {path}: {split_index}")
        if not np.all(source[:split_index] == 0) or not np.all(
            source[split_index:] == 1
        ):
            raise ValueError(f"Source provenance does not match splice in {path}")
        return {
            "frames": frames,
            "prefix_frames": split_index,
            "suffix_frames": frames - split_index,
            "first_step": int(steps[0]),
            "last_step": int(steps[-1]),
        }


def append_status(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def join_one(
    *,
    prefix_path: Path,
    suffix_path: Path,
    destination: Path,
    manifest: dict[str, Any],
    suffix_stride: int,
) -> dict[str, Any]:
    partial = destination.with_suffix(".partial.hdf5")
    partial.unlink(missing_ok=True)
    start_time = time.perf_counter()
    with h5py.File(prefix_path, "r") as prefix, h5py.File(
        suffix_path, "r"
    ) as suffix:
        prefix_total = frame_count(prefix)
        suffix_total = frame_count(suffix)
        demo_index = int(suffix.attrs["rfcl_demo_index"])
        raw_state_index = int(suffix.attrs["rfcl_raw_state_index"])
        demo = manifest["demos"][demo_index]
        snapshot_id = demo["snapshot_ids"][raw_state_index]
        snapshot = manifest["snapshots"][snapshot_id]
        splice_sim_step = int(snapshot["sim_step"])
        suffix_first_step = int(suffix["step"][0])
        if suffix_first_step != splice_sim_step:
            raise ValueError(
                f"Suffix {suffix_path.name} begins at {suffix_first_step}, "
                f"but {snapshot_id} is step {splice_sim_step}"
            )
        prefix_steps = np.asarray(prefix["step"], dtype=np.int64)
        exact_start = int(
            prefix.attrs.get("rfcl_prefix_snapshot_start_sim_step", splice_sim_step)
        )
        early_indices = np.flatnonzero(prefix_steps < exact_start)
        exact_indices = np.flatnonzero(
            (prefix_steps >= exact_start)
            & (prefix_steps < splice_sim_step)
            & ((splice_sim_step - prefix_steps) % suffix_stride == 0)
        )
        prefix_indices = np.concatenate((early_indices, exact_indices))
        if len(prefix_indices) == 0:
            raise ValueError(f"No prefix frames before step {splice_sim_step}")
        if not np.all(np.diff(prefix_steps[prefix_indices]) > 0):
            raise ValueError(f"Selected prefix steps are not ordered in {prefix_path}")
        suffix_indices = selected_suffix_indices(suffix_total, suffix_stride)
        prefix_frames = len(prefix_indices)
        suffix_frames = len(suffix_indices)
        output_frames = prefix_frames + suffix_frames
        metrics = boundary_metrics(prefix, suffix, int(prefix_indices[-1]))
        with h5py.File(partial, "w") as output:
            for key, value in prefix.attrs.items():
                output.attrs[key] = value
            for key, value in suffix.attrs.items():
                output.attrs[key] = value
            output.attrs["full_trajectory_schema"] = "motion_plan_rfcl_full_v1"
            output.attrs["splice_prefix_source"] = str(prefix_path.resolve())
            output.attrs["splice_suffix_source"] = str(suffix_path.resolve())
            output.attrs["splice_snapshot_id"] = snapshot_id
            output.attrs["splice_sim_step"] = splice_sim_step
            output.attrs["splice_prefix_frames"] = prefix_frames
            output.attrs["splice_suffix_frames"] = suffix_frames
            output.attrs["splice_suffix_stride"] = suffix_stride
            output.attrs["splice_created_at"] = datetime.now().astimezone().isoformat()
            for key, value in metrics.items():
                output.attrs[f"splice_{key}"] = value
            for name in CORE_DATASETS:
                prefix_dataset = prefix[name]
                suffix_dataset = suffix[name]
                dtype = output_dtype(prefix_dataset, suffix_dataset)
                if dtype.kind == "S":
                    values = [
                        bytes(value)
                        for value in prefix_dataset[prefix_indices]
                    ]
                    values.extend(
                        bytes(value)
                        for value in suffix_dataset[suffix_indices]
                    )
                    max_length = max(len(value) for value in values)
                    output.create_dataset(
                        name,
                        data=values,
                        dtype=f"S{max_length}",
                    )
                    continue
                dataset = output.create_dataset(
                    name,
                    shape=(output_frames, *prefix_dataset.shape[1:]),
                    dtype=dtype,
                )
                dataset[:prefix_frames] = prefix_dataset[prefix_indices]
                dataset[prefix_frames:] = suffix_dataset[suffix_indices]
            provenance = output.create_group("provenance")
            provenance.create_dataset(
                "source",
                data=np.concatenate(
                    (
                        np.zeros(prefix_frames, dtype=np.int8),
                        np.ones(suffix_frames, dtype=np.int8),
                    )
                ),
            )
            provenance.create_dataset(
                "source_frame_index",
                data=np.concatenate((prefix_indices, suffix_indices)),
            )
            provenance["source"].attrs["0"] = "motion_plan"
            provenance["source"].attrs["1"] = "rfcl"
            if "provenance/source_detail" in prefix:
                prefix_source_detail = np.asarray(
                    prefix["provenance/source_detail"][prefix_indices],
                    dtype=np.int8,
                )
                source_detail = np.concatenate(
                    (
                        prefix_source_detail,
                        np.full(suffix_frames, 2, dtype=np.int8),
                    )
                )
                provenance.create_dataset("source_detail", data=source_detail)
                provenance["source_detail"].attrs["0"] = "motion_plan_rerecord"
                provenance["source_detail"].attrs["1"] = "exact_rfcl_snapshot"
                provenance["source_detail"].attrs["2"] = "rfcl_rollout"
            transition_valid = np.ones(output_frames, dtype=np.int8)
            if "provenance/transition_valid" in prefix:
                transition_valid[:prefix_frames] = downsample_transition_valid(
                    np.asarray(
                        prefix["provenance/transition_valid"],
                        dtype=bool,
                    ),
                    prefix_indices,
                    first_valid=False,
                )
            else:
                transition_valid[0] = 0
            if "provenance/transition_valid" in suffix:
                suffix_transition_valid = np.asarray(
                    suffix["provenance/transition_valid"],
                    dtype=bool,
                )
                transition_valid[prefix_frames:] = downsample_transition_valid(
                    suffix_transition_valid,
                    suffix_indices,
                    first_valid=True,
                )
            provenance.create_dataset(
                "transition_valid",
                data=transition_valid,
            )
            phase = output["phase"]
            for key, value in prefix["phase"].attrs.items():
                phase.attrs[key] = value
            phase_ids = np.asarray(output["phase/id"], dtype=np.int64)
            phase.attrs["pre_move_saved_frames"] = int(np.sum(phase_ids == 0))
            phase.attrs["policy_saved_frames"] = int(np.sum(phase_ids == 1))
            phase.attrs["action_saved_frames"] = int(np.sum(phase_ids == 1))
            phase.attrs["terminal_saved_frames"] = int(np.sum(phase_ids == 2))
            phase.attrs["policy_start_saved_index"] = 0
            phase.attrs["policy_start_sim_step"] = int(output["step"][0])
            phase.attrs["save_frequency"] = suffix_stride
            phase.attrs["terminal_reason"] = "success"
    os.replace(partial, destination)
    validation = validate_output(destination)
    return {
        "output": str(destination.resolve()),
        "prefix": str(prefix_path.resolve()),
        "suffix": str(suffix_path.resolve()),
        "demo_index": demo_index,
        "seed": int(demo["seed"]),
        "raw_state_index": raw_state_index,
        "snapshot_id": snapshot_id,
        "suffix_original_frames": suffix_total,
        **validation,
        **metrics,
        "size_bytes": destination.stat().st_size,
        "elapsed_s": time.perf_counter() - start_time,
    }


def main() -> None:
    args = parse_args()
    if args.suffix_stride <= 0:
        raise ValueError("--suffix-stride must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    suffixes = sorted(args.suffix_dir.glob("*.hdf5"))
    if not suffixes:
        raise ValueError(f"No suffix HDF5 files found in {args.suffix_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "concat_status.jsonl"
    rows: list[dict[str, Any]] = []
    skipped = 0
    for index, suffix_path in enumerate(suffixes, start=1):
        with h5py.File(suffix_path, "r") as suffix:
            demo_index = int(suffix.attrs["rfcl_demo_index"])
        seed = int(manifest["demos"][demo_index]["seed"])
        prefix_path = args.prefix_dir / f"{seed}.hdf5"
        if not prefix_path.is_file():
            raise FileNotFoundError(f"Missing Motion Plan prefix: {prefix_path}")
        destination = args.output / suffix_path.name
        if args.resume and destination.is_file():
            try:
                row = validate_output(destination)
            except Exception as exc:
                print(f"[full-concat] invalid existing {destination.name}: {exc}", flush=True)
            else:
                skipped += 1
                print(
                    f"[full-concat] skip {index}/{len(suffixes)} "
                    f"frames={row['frames']} {destination.name}",
                    flush=True,
                )
                continue
        row = join_one(
            prefix_path=prefix_path,
            suffix_path=suffix_path,
            destination=destination,
            manifest=manifest,
            suffix_stride=args.suffix_stride,
        )
        rows.append(row)
        append_status(status_path, row)
        print(
            f"[full-concat] item={index}/{len(suffixes)} seed={row['seed']} "
            f"frames={row['frames']} prefix={row['prefix_frames']} "
            f"suffix={row['suffix_frames']} gap={row['sim_step_gap']} "
            f"elapsed_s={row['elapsed_s']:.2f} {destination.name}",
            flush=True,
        )
    all_outputs = sorted(args.output.glob("*.hdf5"))
    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "hdf5",
                "frames",
                "prefix_frames",
                "suffix_frames",
                "seed",
                "demo_index",
                "raw_state_index",
                "splice_step",
                "sim_step_gap",
                "joint_max_gap_rad",
                "usb_position_gap_mm",
                "size_bytes",
            )
        )
        for path in all_outputs:
            details = validate_output(path)
            with h5py.File(path, "r") as output:
                writer.writerow(
                    (
                        path.name,
                        details["frames"],
                        details["prefix_frames"],
                        details["suffix_frames"],
                        int(json.loads(output.attrs["episode_context_json"])["seed"]),
                        int(output.attrs["rfcl_demo_index"]),
                        int(output.attrs["rfcl_raw_state_index"]),
                        int(output.attrs["splice_sim_step"]),
                        int(output.attrs["splice_sim_step_gap"]),
                        float(output.attrs["splice_joint_max_gap_rad"]),
                        float(output.attrs["splice_usb_position_gap_mm"]),
                        path.stat().st_size,
                    )
                )
    summary = {
        "schema": "motion_plan_rfcl_full_dataset_v1",
        "prefix_dir": str(args.prefix_dir.resolve()),
        "suffix_dir": str(args.suffix_dir.resolve()),
        "output": str(args.output.resolve()),
        "selected": len(suffixes),
        "completed": len(all_outputs),
        "joined_this_run": len(rows),
        "skipped_this_run": skipped,
        "suffix_stride": args.suffix_stride,
        "core_datasets": list(CORE_DATASETS),
        "size_bytes": sum(path.stat().st_size for path in all_outputs),
        "finished_at": datetime.now().astimezone().isoformat(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
