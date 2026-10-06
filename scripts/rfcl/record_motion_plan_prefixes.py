"""Record exact Motion Plan and snapshot prefixes used by RFCL trajectories."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

try:
    from ._bootstrap import add_repository_root
except ImportError:
    from _bootstrap import add_repository_root

add_repository_root()

from isaaclab.app import AppLauncher


REQUIRED_DATASETS = (
    "observation/head/rgb",
    "observation/wrist/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
    "embodiment/joint",
    "actor/prism",
    "actor/slot",
    "step",
)


class PrefixBoundaryReached(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--step-limit", type=int, default=600)
    parser.add_argument(
        "--demo-indices",
        type=int,
        nargs="+",
        help="Zero-based manifest demo indices to record (default: all).",
    )
    parser.add_argument(
        "--suffix-dir",
        type=Path,
        help="Limit demos and snapshot ranges to the selected suffix HDF5 files.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def validate(path: Path) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        lengths = {name: len(handle[name]) for name in REQUIRED_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Dataset lengths disagree in {path}: {lengths}")
        return {"frames": next(iter(lengths.values()))}


def selected_raw_state_limits(suffix_dir: Path | None) -> dict[int, int] | None:
    if suffix_dir is None:
        return None
    limits: dict[int, int] = {}
    for path in sorted(suffix_dir.glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            demo_index = int(handle.attrs["rfcl_demo_index"])
            raw_state_index = int(handle.attrs["rfcl_raw_state_index"])
        limits[demo_index] = max(limits.get(demo_index, -1), raw_state_index)
    if not limits:
        raise ValueError(f"No suffix HDF5 files found in {suffix_dir}")
    return limits


def play_to_snapshot_boundary(task, boundary_step: int) -> None:
    original_step = task._step

    def boundary_step_wrapper(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if int(task.step_count) >= int(boundary_step):
            raise PrefixBoundaryReached
        return result

    task._step = boundary_step_wrapper
    try:
        task.play_once()
    except PrefixBoundaryReached:
        return
    finally:
        task._step = original_step
    raise RuntimeError(
        f"Motion Plan ended before snapshot boundary step {boundary_step}"
    )


def replace_snapshot_segment(
    task,
    dataset,
    demo_index: int,
    cache: Path,
    max_state_index: int,
) -> dict[str, int]:
    snapshot_count = int(max_state_index) + 1
    first_snapshot = dataset.raw_snapshot(demo_index, 0)
    retained = 0
    for path in sorted(cache.glob("*.pkl"), key=lambda item: int(item.stem)):
        with path.open("rb") as handle:
            observation = pickle.load(handle)
        if int(observation["step"]) < first_snapshot.sim_step:
            if int(path.stem) != retained:
                raise ValueError(f"Non-contiguous Motion Plan cache at {path}")
            retained += 1
        else:
            path.unlink()

    task.save_count = retained
    for state_index in range(snapshot_count):
        snapshot = dataset.raw_snapshot(demo_index, state_index)
        from policy.RL.rfcl_snapshot import restore_snapshot

        restore_snapshot(task, dataset.adapter, snapshot)
        task._update_render()
        task.save_observations(task._get_observations())
    return {
        "motion_plan_frames": retained,
        "snapshot_frames": snapshot_count,
        "snapshot_start_sim_step": first_snapshot.sim_step,
        "snapshot_max_state_index": int(max_state_index),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.snapshot_root / "rfcl_manifest.json").read_text(encoding="utf-8")
    )
    state_limits = selected_raw_state_limits(args.suffix_dir)
    selected_indices = set(range(len(manifest["demos"])))
    if state_limits is not None:
        selected_indices.intersection_update(state_limits)
    if args.demo_indices is not None:
        requested = list(dict.fromkeys(map(int, args.demo_indices)))
        invalid = [
            index
            for index in requested
            if not 0 <= index < len(manifest["demos"])
        ]
        if invalid:
            raise ValueError(
                "--demo-indices must be in "
                f"[0, {len(manifest['demos']) - 1}], got {invalid}"
            )
        selected_indices.intersection_update(requested)
    demos = [
        demo
        for index, demo in enumerate(manifest["demos"])
        if index in selected_indices
    ]
    seeds = [int(demo["seed"]) for demo in demos]
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "recording_status.jsonl"
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    completed = 0
    failed = 0
    try:
        from policy.RL.rfcl_snapshot import RFCLSnapshotDataset
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.snapshot_root,
            video_frequency=0,
            step_limit=args.step_limit,
            mode="collect",
            save_pre_move=False,
            task_variant="rfcl",
            device=args.device,
        )
        dataset = RFCLSnapshotDataset(args.snapshot_root)
        for demo_index, demo in enumerate(manifest["demos"]):
            if demo_index not in selected_indices:
                continue
            seed = int(demo["seed"])
            demo_id = str(demo.get("demo_id", seed))
            profile = demo.get("profile")
            destination = args.output / "hdf5" / f"{seed}.hdf5"
            if args.resume and destination.is_file():
                try:
                    details = validate(destination)
                except Exception as exc:
                    print(f"[prefix-record] invalid existing seed={seed}: {exc}", flush=True)
                else:
                    completed += 1
                    print(
                        f"[prefix-record] skip seed={seed} frames={details['frames']}",
                        flush=True,
                    )
                    continue
            row = {
                "demo_id": demo_id,
                "seed": seed,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
            try:
                cache = args.output / ".cache" / str(seed)
                if cache.exists():
                    shutil.rmtree(cache)
                destination.unlink(missing_ok=True)
                task.set_rfcl_demo_profile(profile)
                task.reset(seed=seed)
                task.tmp_save_dir = cache
                task.save_path = destination
                task.save_count = 0
                task._update_render()
                task.save_observations(task._get_observations())
                first_snapshot = dataset.raw_snapshot(demo_index, 0)
                play_to_snapshot_boundary(task, first_snapshot.sim_step)
                max_state_index = (
                    len(dataset.demos[demo_index]["snapshot_ids"]) - 1
                    if state_limits is None
                    else state_limits[demo_index]
                )
                segment = replace_snapshot_segment(
                    task,
                    dataset,
                    demo_index,
                    cache,
                    max_state_index,
                )
                task.save_to_hdf5()
                with h5py.File(destination, "a") as handle:
                    handle.attrs["rfcl_prefix_schema"] = (
                        "motion_plan_exact_snapshot_prefix_v2"
                    )
                    handle.attrs["rfcl_demo_index"] = demo_index
                    handle.attrs["rfcl_demo_id"] = demo_id
                    handle.attrs["rfcl_snapshot_root"] = str(
                        args.snapshot_root.resolve()
                    )
                    for key, value in segment.items():
                        handle.attrs[f"rfcl_prefix_{key}"] = value
                    provenance = handle.require_group("provenance")
                    source = np.ones(task.save_count, dtype=np.int8)
                    source[: segment["motion_plan_frames"]] = 0
                    provenance.create_dataset("source_detail", data=source)
                    provenance["source_detail"].attrs["0"] = "motion_plan_rerecord"
                    provenance["source_detail"].attrs["1"] = "exact_rfcl_snapshot"
                    transition_valid = np.ones(task.save_count, dtype=np.int8)
                    transition_valid[0] = 0
                    transition_valid[segment["motion_plan_frames"]] = 0
                    provenance.create_dataset(
                        "transition_valid",
                        data=transition_valid,
                    )
                details = validate(destination)
                shutil.rmtree(cache)
                row.update(
                    {
                        "success": True,
                        **details,
                        **segment,
                        "output": str(destination),
                    }
                )
                completed += 1
            except BaseException as exc:
                failed += 1
                row.update(
                    {
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if cache.exists():
                    shutil.rmtree(cache)
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(
                f"[prefix-record] seed={seed} success={row['success']} "
                f"frames={row.get('frames')}",
                flush=True,
            )
        summary = {
            "schema": "rfcl_motion_plan_prefixes_v1",
            "snapshot_root": str(args.snapshot_root.resolve()),
            "demo_ids": [str(demo.get("demo_id", demo["seed"])) for demo in demos],
            "seeds": seeds,
            "completed": completed,
            "failed": failed,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    finally:
        if task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
