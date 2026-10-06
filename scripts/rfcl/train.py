"""Train one RFCL SAC policy with asynchronous multi-GPU rollout workers."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from ._bootstrap import REPOSITORY_ROOT, add_repository_root
except ImportError:
    from _bootstrap import REPOSITORY_ROOT, add_repository_root

add_repository_root()

from policy.RL.rfcl import MixedReplayBuffer, RFCLTransition, ReverseCurriculum
from policy.RL.rfcl_collection import resolve_snapshot_identity
from policy.RL.rfcl_distributed import (
    DistributedDemoScheduler,
    expand_worker_devices,
    export_actor_policy,
    load_worker_result,
    write_json_atomic,
)
from policy.RL.rfcl_sac import (
    RFCLSACTrainer,
    add_demo_transitions,
    demo_state_statistics,
)
from policy.RL.rfcl_snapshot import RFCLSnapshotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--snapshot-sync-steps", type=int, default=0)
    parser.add_argument("--demo-horizon-to-max-steps-ratio", type=float, default=1.0)
    parser.add_argument("--reverse-step-size", type=int, default=2)
    parser.add_argument("--per-demo-buffer-size", type=int, default=3)
    parser.add_argument("--geometric-p", type=float, default=0.5)
    parser.add_argument("--minimum-episode-horizon", type=int, default=16)
    parser.add_argument("--action-scale-margin", type=float, default=1.05)
    parser.add_argument("--bootstrap-handoff", action="store_true")
    parser.add_argument("--demo-block-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--demo-pretrain-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--auto-alpha", action="store_true")
    parser.add_argument("--backup-entropy", action="store_true")
    parser.add_argument("--num-qs", type=int, default=2)
    parser.add_argument("--num-min-qs", type=int, default=None)
    parser.add_argument("--initial-log-std", type=float, default=-3.0)
    parser.add_argument("--demo-fraction", type=float, default=0.5)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--save-success-trajectories", action="store_true")
    parser.add_argument("--checkpoint-frequency", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stop-when-all-solved",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--target-progress", type=float, default=None)
    parser.add_argument("--target-demo-fraction", type=float, default=1.0)
    parser.add_argument("--target-success-trajectories", type=int, default=0)
    parser.add_argument("--target-min-trajectory-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learner-device", default="cuda:0")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="Simulator devices; never translated through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--worker-base-seed", type=int, default=10_000)
    parser.add_argument("--policy-sync-updates", type=int, default=100)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--max-worker-restarts", type=int, default=3)
    parser.add_argument("--worker-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def build_worker_command(
    *,
    python_executable: str,
    worker_script: Path,
    run_config: Path,
    ipc_root: Path,
    worker_id: int,
    worker_seed: int,
    device: str,
    worker_args: Sequence[str] = (),
) -> list[str]:
    command = [
        python_executable,
        str(worker_script),
        "--run-config",
        str(run_config),
        "--ipc-root",
        str(ipc_root),
        "--worker-id",
        str(int(worker_id)),
        "--worker-seed",
        str(int(worker_seed)),
        "--device",
        str(device),
        "--headless",
    ]
    command.extend(str(value) for value in worker_args)
    return command


def resolve_worker_device(device: str) -> tuple[str, str | None]:
    value = str(device)
    if value.startswith("cuda:"):
        physical_index = value.split(":", 1)[1]
        if not physical_index.isdigit():
            raise ValueError(f"Invalid CUDA worker device {value!r}")
        return "cuda:0", physical_index
    if value in ("cuda", "cpu"):
        return value, None
    raise ValueError(f"Unsupported worker device {value!r}")


def _bc_pretrain(
    trainer: RFCLSACTrainer,
    replay: MixedReplayBuffer,
    *,
    steps: int,
    batch_size: int,
) -> None:
    for _ in range(int(steps)):
        batch = replay.sample(batch_size, demo_fraction=1.0)
        states = trainer._states(np.stack([item.state for item in batch]))
        actions = torch.as_tensor(
            np.stack([item.action for item in batch]),
            dtype=torch.float32,
            device=trainer.device,
        )
        mean, _ = trainer.actor(states)
        loss = torch.nn.functional.mse_loss(torch.tanh(mean), actions)
        trainer.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), 10.0)
        trainer.actor_optimizer.step()


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _update_latest_checkpoint(output: Path, checkpoint: Path) -> None:
    latest = output / "latest.pt"
    temporary = output / ".latest.pt.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint.name)
    os.replace(temporary, latest)


def _prune_checkpoints(output: Path, *, keep: int) -> None:
    checkpoints = sorted(output.glob("rfcl_distributed_completed_*.pt"))
    for path in checkpoints[: max(0, len(checkpoints) - int(keep))]:
        path.unlink()


def _truncate_resume_outputs(output: Path, *, completed_episodes: int) -> None:
    metrics_path = output / "metrics.jsonl"
    if metrics_path.is_file():
        retained = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["episode"]) < int(completed_episodes):
                retained.append(line)
        temporary = metrics_path.with_name(f".{metrics_path.name}.tmp")
        temporary.write_text(
            "".join(f"{line}\n" for line in retained), encoding="utf-8"
        )
        os.replace(temporary, metrics_path)
    trajectory_root = output / "success_trajectories"
    if trajectory_root.is_dir():
        for path in trajectory_root.glob("episode_*.npz"):
            try:
                episode = int(path.name.split("_", 2)[1])
            except (IndexError, ValueError):
                continue
            if episode >= int(completed_episodes):
                path.unlink()


def _save_success_trajectory(
    output: Path,
    *,
    episode: int,
    result: dict[str, Any],
    action_scale: np.ndarray,
    action_mode: str,
) -> Path:
    trajectory_root = output / "success_trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    worker_id = int(np.asarray(result["worker_id"]).item())
    demo_index = int(np.asarray(result["demo_index"]).item())
    state_index = int(np.asarray(result["state_index"]).item())
    destination = trajectory_root / (
        f"episode_{episode:09d}_worker_{worker_id:03d}_"
        f"demo_{demo_index:06d}_state_{state_index:06d}.npz"
    )
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray("rfcl_distributed_training_trajectory_v1"),
        states=np.asarray(result["states"], dtype=np.float32),
        actions=np.asarray(result["actions"], dtype=np.float32),
        rewards=np.asarray(result["rewards"], dtype=np.float32),
        next_states=np.asarray(result["next_states"], dtype=np.float32),
        terminated=np.asarray(result["terminated"], dtype=np.bool_),
        episode=np.asarray(episode, dtype=np.int64),
        worker_id=np.asarray(worker_id, dtype=np.int64),
        worker_seed=np.asarray(result["worker_seed"], dtype=np.int64),
        policy_version=np.asarray(result["policy_version"], dtype=np.int64),
        demo_index=np.asarray(demo_index, dtype=np.int64),
        state_index=np.asarray(state_index, dtype=np.int64),
        raw_state_index=np.asarray(result["raw_state_index"], dtype=np.int64),
        action_scale=np.asarray(action_scale, dtype=np.float32),
        action_mode=np.asarray(action_mode),
    )
    os.replace(temporary, destination)
    return destination


def _scalar(result: dict[str, Any], name: str) -> Any:
    return np.asarray(result[name]).item()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.batch_size <= 0 or args.gradient_steps < 0:
        raise ValueError("batch-size must be positive and gradient-steps non-negative")
    if not 0.0 <= args.demo_fraction <= 1.0:
        raise ValueError("demo-fraction must be in [0, 1]")
    if args.replay_capacity <= 0:
        raise ValueError("replay-capacity must be positive")
    if args.checkpoint_frequency <= 0 or args.keep_checkpoints <= 0:
        raise ValueError("checkpoint settings must be positive")
    if args.policy_sync_updates <= 0:
        raise ValueError("policy-sync-updates must be positive")
    if args.worker_startup_timeout <= 0 or args.poll_interval <= 0:
        raise ValueError("worker timing settings must be positive")
    if args.max_worker_restarts < 0:
        raise ValueError("max-worker-restarts cannot be negative")
    if args.target_progress is not None and not 0.0 <= args.target_progress <= 1.0:
        raise ValueError("target-progress must be in [0, 1]")
    if not 0.0 < args.target_demo_fraction <= 1.0:
        raise ValueError("target-demo-fraction must be in (0, 1]")
    if args.target_success_trajectories < 0:
        raise ValueError("target-success-trajectories cannot be negative")
    if args.target_min_trajectory_steps <= 0:
        raise ValueError("target-min-trajectory-steps must be positive")
    worker_devices = expand_worker_devices(args.devices, args.workers_per_device)
    worker_count = len(worker_devices)
    if args.resume == Path("latest"):
        args.resume = args.output / "latest.pt"
    if args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(args.resume)
    if args.resume is None and (args.output / "metrics.jsonl").exists():
        raise FileExistsError(
            f"{args.output} already contains a run; pass --resume or use a new output"
        )

    args.task_name, args.task_config = resolve_snapshot_identity(
        args.snapshot_root,
        task_name=args.task_name,
        task_config=args.task_config,
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = RFCLSnapshotDataset(args.snapshot_root)
    action_mode = str(dataset.manifest.get("action_mode", "qpos_delta"))
    if action_mode in ("target_pos_vel", "target_pos_vel_force"):
        action_scale = dataset.infer_target_pos_vel_action_scale(
            margin=args.action_scale_margin
        )
        if action_mode == "target_pos_vel_force":
            action_scale = np.concatenate(
                (action_scale, np.ones(1, dtype=np.float32))
            )
    else:
        action_scale = dataset.infer_action_scale(margin=args.action_scale_margin)
    if action_mode == "target_pos_vel_force":
        if not args.bootstrap_handoff:
            raise ValueError(
                "target_pos_vel_force distributed training requires --bootstrap-handoff"
            )
        if args.snapshot_sync_steps != 0:
            raise ValueError(
                "target_pos_vel_force distributed training requires "
                "--snapshot-sync-steps 0"
            )
    demos = dataset.to_demo_trajectories(action_scale, action_mode=action_mode)
    curriculum = ReverseCurriculum(
        demos,
        reverse_step_size=args.reverse_step_size,
        per_demo_buffer_size=args.per_demo_buffer_size,
        geometric_p=args.geometric_p,
        demo_horizon_to_max_steps_ratio=args.demo_horizon_to_max_steps_ratio,
        minimum_episode_horizon=args.minimum_episode_horizon,
        seed=args.seed,
    )
    replay = MixedReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
    add_demo_transitions(replay, demos)
    state_mean, state_std = demo_state_statistics(demos)
    trainer = RFCLSACTrainer(
        state_dim=dataset.adapter.state_dim,
        action_dim=int(action_scale.shape[0]),
        device=args.learner_device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        backup_entropy=args.backup_entropy,
        num_qs=args.num_qs,
        num_min_qs=args.num_min_qs,
        initial_log_std=args.initial_log_std,
        state_mean=state_mean,
        state_std=state_std,
    )
    scheduler = DistributedDemoScheduler(
        len(demos), worker_count, block_size=args.demo_block_size
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    completed_episodes = 0
    total_steps = 0
    qualifying_success_trajectories = 0
    elapsed_before_resume = 0.0
    next_job_id = 0
    if args.resume is None:
        _bc_pretrain(
            trainer,
            replay,
            steps=args.demo_pretrain_steps,
            batch_size=args.batch_size,
        )
    else:
        payload = trainer.load(args.resume)
        extra = payload.get("extra", {})
        if extra.get("runner_schema") != "rfcl_distributed_runner_checkpoint_v1":
            raise ValueError("Checkpoint is not a resumable distributed RFCL run")
        if Path(extra["snapshot_root"]).resolve() != args.snapshot_root.resolve():
            raise ValueError("Resume snapshot-root does not match")
        expected = {
            "action_mode": action_mode,
            "adapter_id": dataset.adapter.adapter_id,
            "worker_count": worker_count,
            "workers_per_device": int(args.workers_per_device),
            "demo_block_size": int(args.demo_block_size),
            "batch_size": int(args.batch_size),
            "gradient_steps": int(args.gradient_steps),
            "replay_capacity": int(args.replay_capacity),
        }
        for name, value in expected.items():
            if extra.get(name) != value:
                raise ValueError(
                    f"Resume {name} mismatch: checkpoint={extra.get(name)!r}, "
                    f"current={value!r}"
                )
        if list(extra.get("worker_devices", ())) != worker_devices:
            raise ValueError("Resume worker device assignment does not match")
        if not np.allclose(extra.get("action_scale"), action_scale):
            raise ValueError("Resume action scale does not match")
        curriculum.load_state_dict(extra["curriculum"])
        replay.load_state_dict(extra["replay"])
        scheduler.load_state_dict(extra["scheduler"])
        completed_episodes = int(extra["completed_episodes"])
        total_steps = int(extra["total_steps"])
        qualifying_success_trajectories = int(
            extra.get("qualifying_success_trajectories", 0)
        )
        elapsed_before_resume = float(extra.get("elapsed_s", 0.0))
        next_job_id = int(extra.get("next_job_id", completed_episodes))
        _truncate_resume_outputs(
            args.output, completed_episodes=completed_episodes
        )
    if completed_episodes >= args.episodes:
        print(
            f"[rfcl-distributed] already complete: {completed_episodes}/{args.episodes}",
            flush=True,
        )
        return

    session_id = uuid.uuid4().hex
    ipc_root = args.output / "distributed_ipc" / session_id
    for name in ("jobs", "results", "ready"):
        (ipc_root / name).mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema": "rfcl_distributed_run_v1",
        "session_id": session_id,
        "snapshot_root": str(args.snapshot_root.resolve()),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "step_limit": int(args.step_limit),
        "action_repeat": int(args.action_repeat),
        "snapshot_sync_steps": int(args.snapshot_sync_steps),
        "reverse_step_size": int(args.reverse_step_size),
        "per_demo_buffer_size": int(args.per_demo_buffer_size),
        "geometric_p": float(args.geometric_p),
        "demo_horizon_to_max_steps_ratio": float(
            args.demo_horizon_to_max_steps_ratio
        ),
        "minimum_episode_horizon": int(args.minimum_episode_horizon),
        "action_scale_margin": float(args.action_scale_margin),
        "bootstrap_handoff": bool(args.bootstrap_handoff),
        "action_scale": action_scale.tolist(),
        "action_mode": action_mode,
    }
    run_config_path = ipc_root / "run_config.json"
    write_json_atomic(run_config_path, run_config)
    write_json_atomic(
        args.output / "distributed_manifest.json",
        {
            **run_config,
            "learner_device": args.learner_device,
            "devices": list(args.devices),
            "workers_per_device": int(args.workers_per_device),
            "worker_devices": worker_devices,
            "worker_count": worker_count,
            "output": str(args.output.resolve()),
        },
    )
    actor_path = ipc_root / "actor_latest.pt"
    export_actor_policy(trainer, actor_path, version=trainer.update_count)
    last_published_version = trainer.update_count

    worker_script = Path(__file__).resolve().parent / "internal/training_worker.py"
    log_root = args.output / "distributed_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    process_environment = os.environ.copy()
    process_environment.pop("CUDA_VISIBLE_DEVICES", None)
    processes: dict[int, subprocess.Popen] = {}
    log_handles: dict[int, Any] = {}
    restart_counts = np.zeros(worker_count, dtype=np.int64)
    ready = np.zeros(worker_count, dtype=bool)
    outstanding: dict[int, dict[str, Any]] = {}

    def start_worker(worker_id: int) -> None:
        ready_path = ipc_root / "ready" / f"worker_{worker_id:03d}.json"
        ready_path.unlink(missing_ok=True)
        process_device, visible_device = resolve_worker_device(
            worker_devices[worker_id]
        )
        command = build_worker_command(
            python_executable=sys.executable,
            worker_script=worker_script,
            run_config=run_config_path,
            ipc_root=ipc_root,
            worker_id=worker_id,
            worker_seed=args.worker_base_seed + worker_id,
            device=process_device,
            worker_args=args.worker_args,
        )
        worker_environment = process_environment.copy()
        if visible_device is not None:
            worker_environment["CUDA_VISIBLE_DEVICES"] = visible_device
        log_path = log_root / f"worker_{worker_id:03d}.log"
        handle = log_path.open("a", encoding="utf-8")
        log_handles[worker_id] = handle
        processes[worker_id] = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=worker_environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(
            f"[rfcl-distributed] started worker={worker_id} "
            f"physical_device={worker_devices[worker_id]} "
            f"process_device={process_device} command={shlex.join(command)}",
            flush=True,
        )

    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[rfcl-distributed] received signal {signum}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def wait_until_ready(worker_id: int) -> None:
        ready_path = ipc_root / "ready" / f"worker_{worker_id:03d}.json"
        started_at = time.perf_counter()
        while not ready_path.is_file():
            process = processes[worker_id]
            if process.poll() is not None:
                raise RuntimeError(
                    f"Worker {worker_id} exited during startup with code "
                    f"{process.returncode}; inspect "
                    f"{log_root / f'worker_{worker_id:03d}.log'}"
                )
            if time.perf_counter() - started_at > args.worker_startup_timeout:
                raise TimeoutError(f"Timed out waiting for worker {worker_id}")
            time.sleep(args.poll_interval)
        payload = json.loads(ready_path.read_text(encoding="utf-8"))
        if str(payload.get("session_id")) != session_id:
            raise ValueError("Worker ready file belongs to another session")
        ready[worker_id] = True
        print(
            f"[rfcl-distributed] worker={worker_id} ready "
            f"physical_device={worker_devices[worker_id]}",
            flush=True,
        )

    try:
        for worker_id in range(worker_count):
            start_worker(worker_id)
            wait_until_ready(worker_id)
    except Exception:
        (ipc_root / "stop").touch()
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in log_handles.values():
            if not handle.closed:
                handle.close()
        raise
    print(
        f"[rfcl-distributed] all {worker_count} workers ready; "
        f"devices={worker_devices}",
        flush=True,
    )

    target_active = bool(
        args.target_progress is not None or args.target_success_trajectories > 0
    )
    start_time = time.perf_counter()
    last_checkpoint_completed = -1
    metrics_file = metrics_path.open(
        "a" if args.resume is not None else "w", encoding="utf-8"
    )

    def progress_status() -> dict[str, Any] | None:
        if args.target_progress is None:
            return None
        return curriculum.progress_status(
            target_progress=args.target_progress,
            target_demo_fraction=args.target_demo_fraction,
        )

    def target_reached() -> bool:
        status = progress_status()
        progress_ready = status is None or bool(status["complete"])
        trajectory_ready = (
            qualifying_success_trajectories
            >= args.target_success_trajectories
        )
        return bool(target_active and progress_ready and trajectory_ready)

    def scheduled_unavailable() -> np.ndarray:
        status = progress_status()
        if status is not None and not bool(status["complete"]):
            return np.asarray(status["reached"], dtype=bool)
        if not target_active and args.stop_when_all_solved:
            return curriculum.solved.copy()
        return np.zeros(len(demos), dtype=bool)

    def save_checkpoint() -> Path:
        nonlocal last_checkpoint_completed
        elapsed_s = elapsed_before_resume + time.perf_counter() - start_time
        destination = args.output / (
            f"rfcl_distributed_completed_{completed_episodes:09d}.pt"
        )
        trainer.save(
            destination,
            extra={
                "runner_schema": "rfcl_distributed_runner_checkpoint_v1",
                "snapshot_root": str(args.snapshot_root.resolve()),
                "action_scale": action_scale.tolist(),
                "action_scale_margin": float(args.action_scale_margin),
                "action_mode": action_mode,
                "adapter_id": dataset.adapter.adapter_id,
                "task_name": dataset.adapter.task_name,
                "task_config": str(args.task_config),
                "step_limit": int(args.step_limit),
                "action_repeat": int(args.action_repeat),
                "snapshot_sync_steps": int(args.snapshot_sync_steps),
                "bootstrap_handoff": bool(args.bootstrap_handoff),
                "reverse_step_size": int(args.reverse_step_size),
                "per_demo_buffer_size": int(args.per_demo_buffer_size),
                "geometric_p": float(args.geometric_p),
                "demo_horizon_to_max_steps_ratio": float(
                    args.demo_horizon_to_max_steps_ratio
                ),
                "minimum_episode_horizon": int(args.minimum_episode_horizon),
                "demo_fraction": float(args.demo_fraction),
                "demo_block_size": int(args.demo_block_size),
                "batch_size": int(args.batch_size),
                "gradient_steps": int(args.gradient_steps),
                "replay_capacity": int(args.replay_capacity),
                "worker_count": worker_count,
                "workers_per_device": int(args.workers_per_device),
                "worker_devices": worker_devices,
                "learner_device": str(args.learner_device),
                "policy_sync_updates": int(args.policy_sync_updates),
                "target_progress": args.target_progress,
                "target_demo_fraction": float(args.target_demo_fraction),
                "target_success_trajectories": int(
                    args.target_success_trajectories
                ),
                "target_min_trajectory_steps": int(
                    args.target_min_trajectory_steps
                ),
                "qualifying_success_trajectories": int(
                    qualifying_success_trajectories
                ),
                "curriculum": curriculum.state(),
                "replay": replay.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": _capture_rng_state(),
                "completed_episodes": int(completed_episodes),
                "episode": int(completed_episodes - 1),
                "total_steps": int(total_steps),
                "next_job_id": int(next_job_id),
                "elapsed_s": float(elapsed_s),
            },
        )
        _update_latest_checkpoint(args.output, destination)
        _prune_checkpoints(args.output, keep=args.keep_checkpoints)
        last_checkpoint_completed = completed_episodes
        return destination

    def dispatch(worker_id: int) -> None:
        nonlocal next_job_id
        if worker_id in outstanding or not ready[worker_id]:
            return
        if completed_episodes + len(outstanding) >= args.episodes:
            return
        unavailable = scheduled_unavailable()
        if bool(unavailable.all()):
            return
        demo_index, _new_block, needs_handoff = scheduler.select_demo(
            worker_id, unavailable
        )
        _, state_index = curriculum.sample_checkpoint(demo_id=demo_index)
        job = {
            "schema": "rfcl_distributed_job_v1",
            "session_id": session_id,
            "job_id": next_job_id,
            "worker_id": worker_id,
            "demo_index": int(demo_index),
            "state_index": int(state_index),
            "bootstrap_handoff": bool(args.bootstrap_handoff and needs_handoff),
            "policy_version": int(last_published_version),
        }
        job_path = ipc_root / "jobs" / f"worker_{worker_id:03d}.json"
        write_json_atomic(job_path, job)
        outstanding[worker_id] = job
        next_job_id += 1

    try:
        while completed_episodes < args.episodes and not stop_requested:
            if target_reached() or (
                not target_active
                and args.stop_when_all_solved
                and bool(curriculum.solved.all())
            ):
                break
            for worker_id in range(worker_count):
                dispatch(worker_id)

            made_progress = False
            for worker_id in range(worker_count):
                job = outstanding.get(worker_id)
                if job is None:
                    continue
                result_path = (
                    ipc_root / "results" / f"result_{int(job['job_id']):012d}.npz"
                )
                if not result_path.is_file():
                    continue
                result = load_worker_result(result_path)
                if str(_scalar(result, "session_id")) != session_id:
                    raise ValueError("Worker result belongs to another session")
                if int(_scalar(result, "job_id")) != int(job["job_id"]):
                    raise ValueError("Worker result job id mismatch")
                error = str(_scalar(result, "error"))
                result_path.unlink()
                outstanding.pop(worker_id)
                made_progress = True
                if error:
                    scheduler.invalidate_worker(worker_id)
                    with (args.output / "worker_errors.jsonl").open(
                        "a", encoding="utf-8"
                    ) as error_file:
                        error_file.write(
                            json.dumps(
                                {
                                    "worker_id": worker_id,
                                    "job_id": int(job["job_id"]),
                                    "error": error,
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                    continue

                states = np.asarray(result["states"], dtype=np.float32)
                actions = np.asarray(result["actions"], dtype=np.float32)
                rewards = np.asarray(result["rewards"], dtype=np.float32)
                next_states = np.asarray(result["next_states"], dtype=np.float32)
                terminated = np.asarray(result["terminated"], dtype=np.bool_)
                eligible = np.asarray(result["replay_eligible"], dtype=np.bool_)
                for timestep in range(len(states)):
                    if not bool(eligible[timestep]):
                        continue
                    replay.add(
                        RFCLTransition(
                            state=states[timestep],
                            action=actions[timestep],
                            reward=float(rewards[timestep]),
                            next_state=next_states[timestep],
                            terminated=bool(terminated[timestep]),
                            demo_id=None,
                            timestep=timestep,
                            source="online",
                        )
                    )
                update_count = int(eligible.sum()) * int(args.gradient_steps)
                metrics = None
                for _ in range(update_count):
                    if len(replay) >= args.batch_size:
                        metrics = trainer.update(
                            replay,
                            batch_size=args.batch_size,
                            demo_fraction=args.demo_fraction,
                        )
                success = bool(int(_scalar(result, "success")))
                demo_index = int(_scalar(result, "demo_index"))
                state_index = int(_scalar(result, "state_index"))
                curriculum.record_result(
                    demo_index, state_index, success=success
                )
                scheduler.record_episode(
                    worker_id, demo_index, success=success
                )
                episode = completed_episodes
                episode_steps = int(_scalar(result, "steps"))
                total_steps += episode_steps
                qualifying_success = (
                    success
                    and episode_steps >= args.target_min_trajectory_steps
                )
                if qualifying_success:
                    qualifying_success_trajectories += 1
                trajectory_path = None
                if qualifying_success and args.save_success_trajectories:
                    trajectory_path = _save_success_trajectory(
                        args.output,
                        episode=episode,
                        result=result,
                        action_scale=action_scale,
                        action_mode=action_mode,
                    )
                completed_episodes += 1
                if (
                    trainer.update_count - last_published_version
                    >= args.policy_sync_updates
                ):
                    export_actor_policy(
                        trainer, actor_path, version=trainer.update_count
                    )
                    last_published_version = trainer.update_count
                status = progress_status()
                row: dict[str, Any] = {
                    "episode": episode,
                    "completed_episodes": completed_episodes,
                    "total_steps": total_steps,
                    "episode_steps": episode_steps,
                    "episode_reward": float(rewards.sum()),
                    "success": success,
                    "worker_id": worker_id,
                    "worker_device": worker_devices[worker_id],
                    "worker_policy_version": int(
                        _scalar(result, "policy_version")
                    ),
                    "learner_policy_version": int(trainer.update_count),
                    "policy_lag_updates": int(
                        trainer.update_count
                        - int(_scalar(result, "policy_version"))
                    ),
                    "demo_index": demo_index,
                    "state_index": state_index,
                    "bootstrap_handoff_ran": bool(
                        int(_scalar(result, "bootstrap_handoff"))
                    ),
                    "frontiers": curriculum.frontiers.tolist(),
                    "solved": curriculum.solved.tolist(),
                    "demo_visit_counts": scheduler.visit_counts.tolist(),
                    "replay_source_counts": replay.source_counts(),
                    "curriculum_progress": curriculum.progress().tolist(),
                    "qualifying_success_trajectories": int(
                        qualifying_success_trajectories
                    ),
                    "target_reached": target_reached(),
                    "worker_episode_elapsed_s": float(
                        _scalar(result, "elapsed_s")
                    ),
                    "elapsed_s": elapsed_before_resume
                    + time.perf_counter()
                    - start_time,
                }
                if status is not None:
                    row["progress_reached_demos"] = int(status["reached_demos"])
                    row["progress_required_demos"] = int(status["required_demos"])
                if trajectory_path is not None:
                    row["success_trajectory"] = str(trajectory_path)
                if metrics is not None:
                    row.update(
                        {
                            "critic_loss": metrics.critic_loss,
                            "actor_loss": metrics.actor_loss,
                            "alpha": metrics.alpha,
                            "q_mean": metrics.q_mean,
                            "batch_source_counts": trainer.last_batch_source_counts,
                        }
                    )
                metrics_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                metrics_file.flush()
                print(json.dumps(row, ensure_ascii=True), flush=True)
                should_checkpoint = bool(
                    completed_episodes % args.checkpoint_frequency == 0
                    or completed_episodes >= args.episodes
                    or target_reached()
                )
                if should_checkpoint:
                    save_checkpoint()

            for worker_id, process in list(processes.items()):
                if process.poll() is None:
                    continue
                if worker_id in outstanding:
                    job_path = ipc_root / "jobs" / f"worker_{worker_id:03d}.json"
                    job_path.unlink(missing_ok=True)
                    outstanding.pop(worker_id)
                    scheduler.invalidate_worker(worker_id)
                if stop_requested or target_reached():
                    continue
                if restart_counts[worker_id] >= args.max_worker_restarts:
                    raise RuntimeError(
                        f"Worker {worker_id} exceeded restart limit; inspect "
                        f"{log_root / f'worker_{worker_id:03d}.log'}"
                    )
                log_handles[worker_id].close()
                restart_counts[worker_id] += 1
                ready[worker_id] = False
                start_worker(worker_id)
                wait_until_ready(worker_id)
            if not made_progress:
                time.sleep(args.poll_interval)
    finally:
        (ipc_root / "stop").touch()
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.perf_counter() + 10.0
        for process in processes.values():
            remaining = max(0.0, deadline - time.perf_counter())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in log_handles.values():
            if not handle.closed:
                handle.close()
        metrics_file.close()
        if completed_episodes != last_checkpoint_completed:
            checkpoint = save_checkpoint()
            print(
                f"[rfcl-distributed] final checkpoint={checkpoint} "
                f"completed={completed_episodes}",
                flush=True,
            )


if __name__ == "__main__":
    main()
