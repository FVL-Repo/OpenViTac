"""Validate and summarize RGB/tactile RFCL re-recording outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "observation/head/rgb",
    "observation/wrist/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
    "embodiment/joint",
    "actor/prism",
    "actor/slot",
    "step",
    "rfcl/action",
    "rfcl/action_valid",
    "rfcl/reward",
    "rfcl/source_step",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def discover_hdf5(root: Path) -> list[Path]:
    worker_root = root / "parallel_workers"
    if worker_root.is_dir():
        return sorted(worker_root.glob("worker_*/hdf5/*.hdf5"))
    hdf5_root = root / "hdf5"
    if hdf5_root.is_dir():
        return sorted(hdf5_root.glob("*.hdf5"))
    return sorted(root.glob("*.hdf5"))


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def validate_file(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"missing datasets: {missing}")
        lengths = {name: int(len(handle[name])) for name in REQUIRED_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"dataset lengths disagree: {lengths}")
        frames = next(iter(lengths.values()))
        if frames < 2:
            raise ValueError(f"too few frames: {frames}")
        if not bool(handle.attrs.get("rfcl_replay_success", False)):
            raise ValueError("replay is not marked successful")
        for name in REQUIRED_DATASETS[:4]:
            if any(len(value) == 0 for value in handle[name]):
                raise ValueError(f"empty encoded sensor frame in {name}")
        return {
            "path": str(path.resolve()),
            "frames": frames,
            "demo_index": int(handle.attrs["rfcl_demo_index"]),
            "state_index": int(handle.attrs["rfcl_state_index"]),
            "source_actions": int(handle.attrs["rfcl_source_actions"]),
            "consumed_actions": int(handle.attrs["rfcl_consumed_actions"]),
            "elapsed_s": float(handle.attrs["rfcl_elapsed_s"]),
            "trajectory_uuid": str(handle.attrs["rfcl_trajectory_uuid"]),
            "size_bytes": int(path.stat().st_size),
        }


def selected_count(root: Path) -> int | None:
    manifest_path = root / "parallel_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    value = manifest.get("selected")
    return None if value is None else int(value)


def summarize(root: Path) -> dict[str, Any]:
    paths = discover_hdf5(root)
    valid = []
    invalid = []
    for path in paths:
        try:
            valid.append(validate_file(path))
        except Exception as exc:
            invalid.append({"path": str(path.resolve()), "error": str(exc)})

    selected = selected_count(root)
    completed = len(valid)
    demo_counts = Counter(row["demo_index"] for row in valid)
    unique_uuids = {row["trajectory_uuid"] for row in valid}
    report = {
        "schema": "rfcl_rgb_tactile_statistics_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "root": str(root.resolve()),
        "selected": selected,
        "completed": completed,
        "completion_rate": (
            None if selected in (None, 0) else completed / selected
        ),
        "invalid": invalid,
        "unique_trajectory_uuids": len(unique_uuids),
        "duplicate_trajectory_uuids": completed - len(unique_uuids),
        "unique_demo_indices": len(demo_counts),
        "demo_counts": {str(key): value for key, value in sorted(demo_counts.items())},
        "unique_start_states": len(
            {(row["demo_index"], row["state_index"]) for row in valid}
        ),
        "frames": numeric_summary([row["frames"] for row in valid]),
        "source_actions": numeric_summary([row["source_actions"] for row in valid]),
        "consumed_actions": numeric_summary(
            [row["consumed_actions"] for row in valid]
        ),
        "rerecord_elapsed_s": numeric_summary([row["elapsed_s"] for row in valid]),
        "hdf5_size_bytes": numeric_summary([row["size_bytes"] for row in valid]),
        "total_hdf5_size_bytes": int(sum(row["size_bytes"] for row in valid)),
        "required_datasets": list(REQUIRED_DATASETS),
    }
    return report


def main() -> None:
    args = parse_args()
    report = summarize(args.root)
    output = args.output or args.root / "statistics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
