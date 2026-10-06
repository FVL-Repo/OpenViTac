"""Collect successful trajectories from one frozen RFCL checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.rfcl._bootstrap import add_repository_root

add_repository_root()

from isaaclab.app import AppLauncher

from policy.RL.rfcl_collection import (
    checkpoint_sha256,
    resolve_snapshot_identity,
    sample_rollout_start,
    trajectory_uuid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--successes", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--minimum-steps", type=int, default=20)
    parser.add_argument("--frontier-window", type=int, default=8)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exploration-noise", type=float, default=0.0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-seed", type=int, default=0)
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


def append_status(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def completed_attempts(status_path: Path) -> set[int]:
    if not status_path.is_file():
        return set()
    attempts = set()
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        attempts.add(int(row["attempt"]))
    return attempts


def existing_successes(output: Path) -> int:
    return sum(1 for _ in (output / "success_trajectories").glob("*.npz"))


def save_trajectory(
    output: Path,
    *,
    trajectory_id: str,
    transitions: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]],
    metadata: dict[str, Any],
    action_scale: np.ndarray,
    action_mode: str,
    start_features: np.ndarray,
    final_features: np.ndarray,
) -> Path:
    if not transitions:
        raise ValueError("Cannot save an empty rollout")
    trajectory_root = output / "success_trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    destination = trajectory_root / f"trajectory_{trajectory_id}.npz"
    temporary = trajectory_root / f".trajectory_{trajectory_id}.tmp.npz"
    np.savez_compressed(
        temporary,
        schema=np.asarray("rfcl_privileged_trajectory_v2"),
        trajectory_uuid=np.asarray(trajectory_id),
        states=np.stack([item[0] for item in transitions]).astype(np.float32),
        actions=np.stack([item[1] for item in transitions]).astype(np.float32),
        rewards=np.asarray([item[2] for item in transitions], dtype=np.float32),
        next_states=np.stack([item[3] for item in transitions]).astype(np.float32),
        terminated=np.asarray([item[4] for item in transitions], dtype=np.bool_),
        action_scale=np.asarray(action_scale, dtype=np.float32),
        action_mode=np.asarray(action_mode),
        start_diversity_features=np.asarray(start_features, dtype=np.float32),
        final_diversity_features=np.asarray(final_features, dtype=np.float32),
        **{name: np.asarray(value) for name, value in metadata.items()},
    )
    os.replace(temporary, destination)
    return destination


def frozen_trainer(checkpoint: Path, *, device: str):
    from policy.RL.rfcl_sac import RFCLSACTrainer

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    log_alpha = float(torch.as_tensor(payload["log_alpha"]).exp().item())
    trainer = RFCLSACTrainer(
        state_dim=int(payload["state_dim"]),
        action_dim=int(payload["action_dim"]),
        device=device,
        gamma=float(payload["gamma"]),
        tau=float(payload["tau"]),
        alpha=log_alpha,
        auto_alpha=bool(payload["auto_alpha"]),
        backup_entropy=bool(payload["backup_entropy"]),
        target_entropy=float(payload["target_entropy"]),
        num_qs=int(payload["num_qs"]),
        num_min_qs=int(payload["num_min_qs"]),
        initial_log_std=float(payload["initial_log_std"]),
    )
    payload = trainer.load(checkpoint)
    trainer.actor.eval()
    trainer.critic.eval()
    trainer.target_critic.eval()
    return trainer, payload


def main() -> None:
    args = parse_args()
    if args.successes <= 0:
        raise ValueError("--successes must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.minimum_steps <= 0:
        raise ValueError("--minimum-steps must be positive")
    if args.frontier_window < 0:
        raise ValueError("--frontier-window cannot be negative")
    if args.exploration_noise < 0.0:
        raise ValueError("--exploration-noise cannot be negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.task_name, args.task_config = resolve_snapshot_identity(
        args.snapshot_root,
        task_name=args.task_name,
        task_config=args.task_config,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.jsonl"
    summary_path = args.output / "summary.json"
    if not args.resume and status_path.exists():
        raise FileExistsError(f"Output already contains status: {status_path}")
    checkpoint_digest = checkpoint_sha256(args.checkpoint)
    finished_attempts = completed_attempts(status_path) if args.resume else set()
    success_count = existing_successes(args.output) if args.resume else 0
    snapshot_manifest = json.loads(
        (args.snapshot_root / "rfcl_manifest.json").read_text(encoding="utf-8")
    )
    adapter_metadata = snapshot_manifest.get("adapter", {})
    run_identity = {
        "schema": "rfcl_rollout_worker_v2",
        "checkpoint_sha256": checkpoint_digest,
        "snapshot_root": str(args.snapshot_root.resolve()),
        "adapter_id": adapter_metadata.get("adapter_id"),
        "worker_id": int(args.worker_id),
        "worker_seed": int(args.worker_seed),
        "deterministic": bool(args.deterministic),
        "exploration_noise": float(args.exploration_noise),
        "minimum_steps": int(args.minimum_steps),
        "frontier_window": int(args.frontier_window),
    }
    run_manifest_path = args.output / "run_manifest.json"
    if run_manifest_path.is_file():
        existing_identity = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        if existing_identity != run_identity:
            raise ValueError("Resume output belongs to a different rollout worker")
    else:
        write_json_atomic(run_manifest_path, run_identity)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    env = None
    started_at = time.perf_counter()
    try:
        from policy.RL.rfcl_env import RFCLPrivilegedEnv
        from policy.RL.rfcl_snapshot import RFCLSnapshotDataset
        from policy.RL.task_factory import create_task

        dataset = RFCLSnapshotDataset(args.snapshot_root)
        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.snapshot_root,
            video_frequency=0,
            step_limit=int(dataset.manifest.get("step_limit", 200)),
            mode="eval_test",
            save_pre_move=False,
            task_variant="rfcl",
            device=args.device,
        )
        trainer, checkpoint_payload = frozen_trainer(
            args.checkpoint,
            device=args.device,
        )
        extra = checkpoint_payload.get("extra", {})
        if str(extra.get("adapter_id")) != dataset.adapter.adapter_id:
            raise ValueError("Checkpoint and snapshot adapter do not match")
        env = RFCLPrivilegedEnv(
            task,
            args.snapshot_root,
            action_scale=np.asarray(extra["action_scale"], dtype=np.float32),
            action_repeat=int(extra["action_repeat"]),
            snapshot_sync_steps=int(extra["snapshot_sync_steps"]),
            reverse_step_size=int(extra["reverse_step_size"]),
            per_demo_buffer_size=int(extra["per_demo_buffer_size"]),
            geometric_p=float(extra["geometric_p"]),
            demo_horizon_to_max_steps_ratio=float(
                extra["demo_horizon_to_max_steps_ratio"]
            ),
            minimum_episode_horizon=int(extra["minimum_episode_horizon"]),
            seed=args.worker_seed,
            action_mode=str(extra["action_mode"]),
        )
        env.curriculum.load_state_dict(extra["curriculum"])
        frontiers = env.curriculum.frontiers.copy()
        state_counts = [env.dataset.state_count(index) for index in range(len(frontiers))]

        for attempt in range(args.max_attempts):
            if success_count >= args.successes:
                break
            if attempt in finished_attempts:
                continue
            trajectory_id = trajectory_uuid(
                checkpoint_digest=checkpoint_digest,
                adapter_id=env.adapter.adapter_id,
                worker_seed=args.worker_seed,
                attempt=attempt,
            )
            attempt_rng = np.random.default_rng(
                np.random.SeedSequence([args.worker_seed, attempt])
            )
            demo_index, state_index = sample_rollout_start(
                frontiers=frontiers,
                state_counts=state_counts,
                attempt=attempt,
                worker_seed=args.worker_seed,
                window=args.frontier_window,
                minimum_remaining_steps=args.minimum_steps,
                rng=attempt_rng,
            )
            row: dict[str, Any] = {
                "schema": "rfcl_rollout_status_v2",
                "trajectory_uuid": trajectory_id,
                "attempt": attempt,
                "worker_id": args.worker_id,
                "worker_seed": args.worker_seed,
                "demo_index": demo_index,
                "state_index": state_index,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
            try:
                demo_seed = env.dataset.demos[demo_index].get("seed")
                task.reset(seed=None if demo_seed is None else int(demo_seed))
                env.adapter.prepare_handoff(task)
                state, reset_info = env.reset(
                    options={
                        "demo_index": demo_index,
                        "state_index": state_index,
                        "skip_task_reset": True,
                    }
                )
                start_features = env.adapter.diversity_features(state)
                transitions = []
                info: dict[str, Any] = {}
                while True:
                    action = trainer.act(state, deterministic=args.deterministic)
                    if args.exploration_noise:
                        action = np.clip(
                            action
                            + attempt_rng.normal(
                                0.0,
                                args.exploration_noise,
                                size=action.shape,
                            ),
                            -1.0,
                            1.0,
                        ).astype(np.float32)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    transition = env.pop_transition()
                    if transition is None:
                        raise RuntimeError("RFCL environment did not expose a transition")
                    transitions.append(transition)
                    state = next_state
                    if terminated or truncated:
                        break
                success = bool(info.get("success", False))
                long_enough = len(transitions) >= args.minimum_steps
                row.update(
                    {
                        "success": success,
                        "steps": len(transitions),
                        "result": (
                            "success"
                            if success and long_enough
                            else "short_success"
                            if success
                            else "failure"
                        ),
                        "irrecoverable_failure": info.get(
                            "irrecoverable_failure"
                        ),
                    }
                )
                if success and long_enough:
                    path = save_trajectory(
                        args.output,
                        trajectory_id=trajectory_id,
                        transitions=transitions,
                        metadata={
                            "worker_id": args.worker_id,
                            "worker_seed": args.worker_seed,
                            "attempt": attempt,
                            "demo_index": demo_index,
                            "state_index": state_index,
                            "raw_state_index": reset_info["raw_state_index"],
                            "task_name": env.adapter.task_name,
                            "adapter_id": env.adapter.adapter_id,
                            "checkpoint_sha256": checkpoint_digest,
                            "deterministic": int(args.deterministic),
                            "exploration_noise": args.exploration_noise,
                        },
                        action_scale=env.action_scale,
                        action_mode=env.action_mode,
                        start_features=start_features,
                        final_features=env.adapter.diversity_features(state),
                    )
                    row["trajectory"] = str(path.resolve())
                    success_count += 1
            except Exception as exc:
                row.update(
                    {
                        "success": False,
                        "result": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            append_status(status_path, row)
            print(json.dumps(row, ensure_ascii=True), flush=True)

        summary = {
            "schema": "rfcl_rollout_summary_v2",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_digest,
            "snapshot_root": str(args.snapshot_root.resolve()),
            "adapter_id": env.adapter.adapter_id,
            "worker_id": args.worker_id,
            "worker_seed": args.worker_seed,
            "requested_successes": args.successes,
            "completed_successes": success_count,
            "minimum_steps": args.minimum_steps,
            "frontier_window": args.frontier_window,
            "deterministic": args.deterministic,
            "exploration_noise": args.exploration_noise,
            "elapsed_s": time.perf_counter() - started_at,
            "complete": success_count >= args.successes,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
        write_json_atomic(summary_path, summary)
        if not summary["complete"]:
            raise RuntimeError(
                f"Collected {success_count}/{args.successes} successful trajectories"
            )
    finally:
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
