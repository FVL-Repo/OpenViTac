"""Merge independently generated RFCL snapshot shards without duplicating data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.rfcl._bootstrap import add_repository_root

add_repository_root()

from policy.RL.rfcl_snapshot import (
    SNAPSHOT_DATASET_SCHEMA,
    SNAPSHOT_MANIFEST_NAME,
    RFCLSnapshotDataset,
)


MANIFEST_INVARIANTS = (
    "schema",
    "task",
    "task_config",
    "stride",
    "step_limit",
    "action_repeat",
    "action_mode",
    "action_dim",
    "action_scale",
    "demo_plan",
    "profile_plan_id",
    "adapter",
    "reward",
    "terminal_definition",
    "gripper_control",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-demos", type=int)
    return parser.parse_args()


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise FileExistsError(f"Conflicting shard file: {destination}")
        if _digest(source) != _digest(destination):
            raise FileExistsError(f"Conflicting shard file: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _frame_from_dump_name(path: Path) -> int | None:
    name = path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    suffix = name.rsplit(".", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _profile_sort_key(demo: dict[str, Any]) -> tuple[int, str]:
    demo_id = str(demo.get("demo_id", ""))
    try:
        return int(demo_id.rsplit("_", 1)[-1]), demo_id
    except ValueError:
        return sys.maxsize, demo_id


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / SNAPSHOT_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SNAPSHOT_DATASET_SCHEMA:
        raise ValueError(f"Unsupported snapshot schema in {path}")
    return payload


def _copy_static_scene_files(source_root: Path, output: Path) -> None:
    for relative in (Path("metadata.json"), Path("scene/systems.json")):
        source = source_root / relative
        if source.is_file():
            _link_or_copy(source, output / relative)
    sanity_root = source_root / "scene" / "sanity_check"
    if sanity_root.is_dir():
        for source in sanity_root.rglob("*"):
            if source.is_file():
                _link_or_copy(source, output / source.relative_to(source_root))


def merge_snapshot_shards(
    inputs: Sequence[Path],
    output: Path,
    *,
    expected_demos: int | None = None,
) -> dict[str, Any]:
    roots = [Path(root) for root in inputs]
    if not roots:
        raise ValueError("At least one snapshot shard is required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Merged output is not empty: {output}")

    manifests = [_load_manifest(root) for root in roots]
    reference = manifests[0]
    for root, manifest in zip(roots[1:], manifests[1:]):
        for key in MANIFEST_INVARIANTS:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"Manifest mismatch for {key!r} in {root}")

    demos_by_id: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    snapshot_sources: dict[str, Path] = {}
    frame_owners: dict[int, str] = {}
    for root, manifest in zip(roots, manifests):
        failures.extend(manifest.get("failed_demos", ()))
        shard_snapshots = dict(manifest.get("snapshots", {}))
        for demo in manifest.get("demos", ()):
            demo_id = str(demo["demo_id"])
            if demo_id in demos_by_id:
                raise ValueError(f"Duplicate demo across shards: {demo_id}")
            demos_by_id[demo_id] = dict(demo)
            for snapshot_id in demo.get("snapshot_ids", ()):
                snapshot_id = str(snapshot_id)
                metadata = shard_snapshots.get(snapshot_id)
                if metadata is None:
                    raise KeyError(f"Missing metadata for {snapshot_id!r} in {root}")
                frame = int(metadata["uipc_frame"])
                previous = frame_owners.get(frame)
                if previous is not None:
                    raise ValueError(
                        f"UIPC frame {frame} belongs to both {previous} and {snapshot_id}"
                    )
                frame_owners[frame] = snapshot_id
                snapshots[snapshot_id] = metadata
                snapshot_sources[snapshot_id] = root

    demos = sorted(demos_by_id.values(), key=_profile_sort_key)
    if expected_demos is not None and len(demos) != int(expected_demos):
        raise ValueError(
            f"Expected {expected_demos} merged demos, found {len(demos)}"
        )

    output.mkdir(parents=True, exist_ok=True)
    _copy_static_scene_files(roots[0], output)
    frames_by_root: dict[Path, set[int]] = {root: set() for root in roots}
    for snapshot_id, metadata in snapshots.items():
        source_root = snapshot_sources[snapshot_id]
        state_file = Path(str(metadata["state_file"]))
        _link_or_copy(source_root / state_file, output / state_file)
        frames_by_root[source_root].add(int(metadata["uipc_frame"]))

    for source_root, frames in frames_by_root.items():
        dump_root = source_root / "scene" / "dump"
        found_frames: set[int] = set()
        for source in dump_root.rglob("*"):
            if not source.is_file():
                continue
            frame = _frame_from_dump_name(source)
            if frame not in frames:
                continue
            found_frames.add(frame)
            _link_or_copy(source, output / source.relative_to(source_root))
        missing = frames - found_frames
        if missing:
            raise FileNotFoundError(
                f"Missing UIPC dump files under {dump_root} for frames "
                f"{sorted(missing)[:10]}"
            )

    merged = {
        key: reference.get(key)
        for key in MANIFEST_INVARIANTS
    }
    merged.update(
        {
            "demos": demos,
            "snapshots": snapshots,
            "failed_demos": failures,
            "merged_from": [str(root.resolve()) for root in roots],
        }
    )
    manifest_path = output / SNAPSHOT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    dataset = RFCLSnapshotDataset(output)
    if len(dataset.demos) != len(demos):
        raise RuntimeError("Merged snapshot dataset failed validation")
    return merged


def main() -> None:
    args = parse_args()
    merged = merge_snapshot_shards(
        args.inputs,
        args.output,
        expected_demos=args.expected_demos,
    )
    print(
        f"[rfcl-snapshot-merge] output={args.output} "
        f"demos={len(merged['demos'])} snapshots={len(merged['snapshots'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
