"""Re-record successful privileged RFCL trajectories with RGB and tactile data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.rfcl._bootstrap import add_repository_root

add_repository_root()

from isaaclab.app import AppLauncher

from policy.RL.rfcl_collection import resolve_snapshot_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many selected trajectories; useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip HDF5 files that already pass validation.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def load_selection(path: Path, limit: int | None) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection file does not exist: {path}")
    trajectories = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        trajectory = Path(value)
        if not trajectory.is_file():
            candidate = path.parent / trajectory
            if candidate.is_file():
                trajectory = candidate
            else:
                raise FileNotFoundError(f"Selected trajectory does not exist: {value}")
        trajectories.append(trajectory.resolve())
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        trajectories = trajectories[:limit]
    if not trajectories:
        raise ValueError(f"Selection is empty: {path}")
    return trajectories


def load_trajectory(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        common_required = {
            "schema",
            "actions",
            "rewards",
            "terminated",
            "demo_index",
            "state_index",
            "action_scale",
            "action_mode",
        }
        missing = common_required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        result = {name: archive[name].copy() for name in archive.files}
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or len(actions) == 0:
        raise ValueError(f"Invalid actions in {path}: shape={actions.shape}")
    if float(np.asarray(result["rewards"])[-1]) != 1.0:
        raise ValueError(f"Selected trajectory is not terminal-success: {path}")
    if not bool(np.asarray(result["terminated"])[-1]):
        raise ValueError(f"Selected trajectory is not terminated: {path}")
    schema = str(np.asarray(result["schema"]).item())
    if schema not in {
        "rfcl_privileged_trajectory_v2",
        "rfcl_distributed_training_trajectory_v1",
    }:
        raise ValueError(f"Unsupported trajectory schema {schema!r}: {path}")
    result["actions"] = actions
    result["schema"] = schema
    if "trajectory_uuid" in result:
        result["trajectory_uuid"] = str(
            np.asarray(result["trajectory_uuid"]).item()
        )
    else:
        identity = ":".join(
            str(np.asarray(result[name]).item())
            for name in (
                "episode",
                "worker_id",
                "policy_version",
                "demo_index",
                "state_index",
            )
        )
        result["trajectory_uuid"] = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"openvitacorge:rfcl-training:{identity}")
        )
    result["demo_index"] = int(np.asarray(result["demo_index"]).item())
    result["state_index"] = int(np.asarray(result["state_index"]).item())
    result["adapter_id"] = (
        str(np.asarray(result["adapter_id"]).item())
        if "adapter_id" in result
        else None
    )
    result["checkpoint_sha256"] = (
        str(np.asarray(result["checkpoint_sha256"]).item())
        if "checkpoint_sha256" in result
        else ""
    )
    result["policy_version"] = (
        int(np.asarray(result["policy_version"]).item())
        if "policy_version" in result
        else -1
    )
    result["action_scale"] = np.asarray(result["action_scale"], dtype=np.float32)
    result["action_mode"] = str(np.asarray(result["action_mode"]).item())
    return result


def output_name(source: Path) -> str:
    return f"{source.stem}.hdf5"


def validate_hdf5(
    path: Path,
    *,
    required_datasets: tuple[str, ...],
    source: Path | None = None,
    require_success: bool = True,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in required_datasets if name not in handle]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        lengths = {name: int(len(handle[name])) for name in required_datasets}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Dataset lengths disagree in {path}: {lengths}")
        frame_count = next(iter(lengths.values()))
        if frame_count < 2:
            raise ValueError(f"Too few frames in {path}: {frame_count}")
        success = bool(handle.attrs.get("rfcl_replay_success", False))
        if require_success and not success:
            raise ValueError(f"Replay is not marked successful: {path}")
        if source is not None:
            recorded_source = str(handle.attrs.get("rfcl_source_trajectory", ""))
            if recorded_source != str(source.resolve()):
                raise ValueError(
                    f"Source mismatch in {path}: {recorded_source!r} != {str(source.resolve())!r}"
                )
        return {
            "frames": frame_count,
            "success": success,
            "consumed_actions": int(handle.attrs.get("rfcl_consumed_actions", -1)),
            "datasets": lengths,
        }


def append_status(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def reset_recording_state(task: Any, cache_dir: Path, partial_path: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    partial_path.unlink(missing_ok=True)
    task.tmp_save_dir = cache_dir
    task.save_path = partial_path
    task.save_count = 0
    task.policy_start_saved_index = None
    task.last_saved_phase_id = None
    task.phase_saved_counts = {
        task.PHASE_PRE_MOVE: 0,
        task.PHASE_POLICY: 0,
        task.PHASE_TERMINAL: 0,
    }


def add_rfcl_fields(
    observation: dict[str, Any],
    *,
    action: np.ndarray,
    action_valid: bool,
    reward: float,
    source_step: int,
) -> dict[str, Any]:
    observation["rfcl"] = {
        "action": np.asarray(action, dtype=np.float32),
        "action_valid": np.asarray(int(action_valid), dtype=np.int8),
        "reward": np.asarray(float(reward), dtype=np.float32),
        "source_step": np.asarray(int(source_step), dtype=np.int64),
    }
    return observation


def save_frame(
    task: Any,
    *,
    action: np.ndarray,
    action_valid: bool,
    reward: float,
    source_step: int,
) -> None:
    task._update_render()
    observation = task._get_observations()
    task.save_observations(
        add_rfcl_fields(
            observation,
            action=action,
            action_valid=action_valid,
            reward=reward,
            source_step=source_step,
        )
    )


def write_replay_metadata(
    path: Path,
    *,
    source: Path,
    trajectory: dict[str, Any],
    reset_info: dict[str, Any],
    consumed_actions: int,
    frame_count: int,
    elapsed_s: float,
) -> None:
    with h5py.File(path, "a") as handle:
        handle.attrs["rfcl_rerecord_schema"] = "rfcl_rgb_tactile_rerecord_v2"
        handle.attrs["rfcl_replay_success"] = True
        handle.attrs["rfcl_source_trajectory"] = str(source.resolve())
        handle.attrs["rfcl_trajectory_uuid"] = trajectory["trajectory_uuid"]
        handle.attrs["rfcl_adapter_id"] = trajectory["adapter_id"] or ""
        handle.attrs["rfcl_checkpoint_sha256"] = trajectory["checkpoint_sha256"]
        handle.attrs["rfcl_source_schema"] = trajectory["schema"]
        handle.attrs["rfcl_policy_version"] = trajectory["policy_version"]
        handle.attrs["rfcl_demo_index"] = int(trajectory["demo_index"])
        handle.attrs["rfcl_state_index"] = int(trajectory["state_index"])
        handle.attrs["rfcl_raw_state_index"] = int(reset_info["raw_state_index"])
        handle.attrs["rfcl_action_mode"] = str(trajectory["action_mode"])
        handle.attrs["rfcl_source_actions"] = int(len(trajectory["actions"]))
        handle.attrs["rfcl_consumed_actions"] = int(consumed_actions)
        handle.attrs["rfcl_frame_count"] = int(frame_count)
        handle.attrs["rfcl_elapsed_s"] = float(elapsed_s)
        handle.attrs["rfcl_recorded_at"] = datetime.now().astimezone().isoformat()
        handle.create_dataset(
            "rfcl/action_scale",
            data=np.asarray(trajectory["action_scale"], dtype=np.float32),
        )


def record_attempt(
    task: Any,
    env: Any,
    *,
    source: Path,
    trajectory: dict[str, Any],
    output: Path,
    required_datasets: tuple[str, ...],
) -> tuple[bool, dict[str, Any]]:
    name = output_name(source)
    final_path = output / "hdf5" / name
    partial_path = output / ".partial" / name
    cache_dir = output / ".cache" / source.stem
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    reset_recording_state(task, cache_dir, partial_path)
    start_time = time.perf_counter()
    _, reset_info = env.reset(
        options={
            "demo_index": int(trajectory["demo_index"]),
            "state_index": int(trajectory["state_index"]),
            "skip_task_reset": True,
        }
    )
    task.metadata.update(
        {
            "rfcl_source_trajectory": str(source.resolve()),
            "rfcl_trajectory_uuid": trajectory["trajectory_uuid"],
            "rfcl_adapter_id": trajectory["adapter_id"],
            "rfcl_checkpoint_sha256": trajectory["checkpoint_sha256"],
            "rfcl_demo_index": int(trajectory["demo_index"]),
            "rfcl_state_index": int(trajectory["state_index"]),
        }
    )
    zero_action = np.zeros_like(trajectory["actions"][0], dtype=np.float32)
    save_frame(
        task,
        action=zero_action,
        action_valid=False,
        reward=0.0,
        source_step=-1,
    )
    info: dict[str, Any] = {}
    terminated = False
    truncated = False
    consumed_actions = 0
    for source_step, action in enumerate(trajectory["actions"]):
        _, reward, terminated, truncated, info = env.step(action)
        consumed_actions += 1
        save_frame(
            task,
            action=action,
            action_valid=True,
            reward=float(reward),
            source_step=source_step,
        )
        if terminated or truncated:
            break
    success = bool(info.get("success", False))
    elapsed_s = time.perf_counter() - start_time
    details = {
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "consumed_actions": int(consumed_actions),
        "source_actions": int(len(trajectory["actions"])),
        "frames": int(task.save_count),
        "elapsed_s": elapsed_s,
        "terminal_reason": getattr(task, "terminal_reason", None),
        "final_info": {
            key: value
            for key, value in info.items()
            if key in ("success", "exec_success", "policy_step", "episode_horizon")
        },
    }
    if not success:
        shutil.rmtree(cache_dir, ignore_errors=True)
        partial_path.unlink(missing_ok=True)
        return False, details
    task.save_to_hdf5()
    write_replay_metadata(
        partial_path,
        source=source,
        trajectory=trajectory,
        reset_info=reset_info,
        consumed_actions=consumed_actions,
        frame_count=int(task.save_count),
        elapsed_s=elapsed_s,
    )
    validation = validate_hdf5(
        partial_path,
        required_datasets=required_datasets,
        source=source,
    )
    os.replace(partial_path, final_path)
    shutil.rmtree(cache_dir, ignore_errors=True)
    details["output"] = str(final_path)
    details["validation"] = validation
    return True, details


def main() -> None:
    args = parse_args()
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")
    trajectories = load_selection(args.selection_file, args.limit)
    args.task_name, args.task_config = resolve_snapshot_identity(
        args.snapshot_root,
        task_name=args.task_name,
        task_config=args.task_config,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "recording_status.jsonl"
    first = load_trajectory(trajectories[0])
    for source in trajectories[1:]:
        candidate = load_trajectory(source)
        if candidate["action_mode"] != first["action_mode"]:
            raise ValueError("Selected trajectories use different action modes")
        if not np.array_equal(candidate["action_scale"], first["action_scale"]):
            raise ValueError("Selected trajectories use different action scales")
        if (
            candidate["adapter_id"] is not None
            and first["adapter_id"] is not None
            and candidate["adapter_id"] != first["adapter_id"]
        ):
            raise ValueError("Selected trajectories use different task adapters")

    print(
        f"[rfcl-rerecord] selected={len(trajectories)} output={args.output} "
        f"action_mode={first['action_mode']}",
        flush=True,
    )
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    env = None
    completed = 0
    failed = 0
    current_bootstrap_demo: int | None = None
    bootstrap_valid = False
    try:
        from policy.RL.rfcl_env import RFCLPrivilegedEnv
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            # UIPC frames are stored inside the snapshot workspace.  The
            # per-trajectory HDF5/cache paths are redirected separately.
            save_dir=args.snapshot_root,
            video_frequency=0,
            step_limit=args.step_limit,
            mode="eval_test",
            save_pre_move=False,
            task_variant="rfcl",
            device=args.device,
        )
        env = RFCLPrivilegedEnv(
            task,
            args.snapshot_root,
            action_scale=first["action_scale"],
            action_repeat=1,
            snapshot_sync_steps=0,
            demo_horizon_to_max_steps_ratio=1.0,
            minimum_episode_horizon=16,
            seed=0,
            action_mode=first["action_mode"],
        )
        if (
            first["adapter_id"] is not None
            and first["adapter_id"] != env.adapter.adapter_id
        ):
            raise ValueError("Selected trajectories do not match the snapshot adapter")
        required_datasets = tuple(env.adapter.required_hdf5_keys)
        for selection_index, source in enumerate(trajectories):
            final_path = args.output / "hdf5" / output_name(source)
            if args.resume and final_path.is_file():
                try:
                    validation = validate_hdf5(
                        final_path,
                        required_datasets=required_datasets,
                        source=source,
                    )
                except Exception as exc:
                    print(
                        f"[rfcl-rerecord] invalid existing file={final_path}: {exc}",
                        flush=True,
                    )
                else:
                    completed += 1
                    print(
                        f"[rfcl-rerecord] skip {selection_index + 1}/{len(trajectories)} "
                        f"frames={validation['frames']} source={source.name}",
                        flush=True,
                    )
                    continue
            trajectory = load_trajectory(source)
            if trajectory["adapter_id"] is None:
                trajectory["adapter_id"] = env.adapter.adapter_id
            success = False
            for attempt in range(1, args.max_retries + 1):
                demo_index = int(trajectory["demo_index"])
                needs_bootstrap = (
                    not bootstrap_valid or current_bootstrap_demo != demo_index
                )
                if needs_bootstrap:
                    demo_seed = env.dataset.demos[demo_index].get("seed")
                    print(
                        f"[rfcl-rerecord] bootstrap demo={demo_index} seed={demo_seed}",
                        flush=True,
                    )
                    task.reset(seed=None if demo_seed is None else int(demo_seed))
                    env.adapter.prepare_handoff(task)
                    current_bootstrap_demo = demo_index
                    bootstrap_valid = True
                row = {
                    "selection_index": selection_index,
                    "source": str(source),
                    "trajectory_uuid": trajectory["trajectory_uuid"],
                    "demo_index": demo_index,
                    "state_index": int(trajectory["state_index"]),
                    "attempt": attempt,
                    "bootstrap_ran": needs_bootstrap,
                    "timestamp": datetime.now().astimezone().isoformat(),
                }
                try:
                    success, details = record_attempt(
                        task,
                        env,
                        source=source,
                        trajectory=trajectory,
                        output=args.output,
                        required_datasets=required_datasets,
                    )
                    row.update(details)
                except BaseException as exc:
                    success = False
                    row.update(
                        {
                            "success": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                append_status(status_path, row)
                print(
                    f"[rfcl-rerecord] item={selection_index + 1}/{len(trajectories)} "
                    f"attempt={attempt} success={success} source={source.name}",
                    flush=True,
                )
                if success:
                    completed += 1
                    break
                bootstrap_valid = False
            if not success:
                failed += 1
        summary = {
            "schema": "rfcl_rgb_tactile_rerecord_summary_v1",
            "selection_file": str(args.selection_file.resolve()),
            "snapshot_root": str(args.snapshot_root.resolve()),
            "selected": len(trajectories),
            "completed": completed,
            "failed": failed,
            "output": str(args.output.resolve()),
            "required_datasets": list(required_datasets),
            "finished_at": datetime.now().astimezone().isoformat(),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    finally:
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
