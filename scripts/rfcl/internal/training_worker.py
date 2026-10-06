"""Persistent Isaac/UIPC rollout worker for one distributed RFCL learner."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.rfcl._bootstrap import add_repository_root

add_repository_root()

from isaaclab.app import AppLauncher

from policy.RL.rfcl_distributed import save_worker_result, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--ipc-root", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-seed", type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


class ActorPolicy:
    def __init__(self) -> None:
        self.actor = None
        self.state_mean = None
        self.state_std = None
        self.version = -1

    def load(self, path: Path, *, minimum_version: int) -> None:
        from policy.RL.rfcl_sac import GaussianPolicy

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "rfcl_distributed_actor_v1":
            raise ValueError("Unsupported distributed actor checkpoint")
        version = int(payload["version"])
        if version < int(minimum_version):
            raise RuntimeError(
                f"Actor policy version {version} is older than job "
                f"version {minimum_version}"
            )
        if self.actor is None:
            self.actor = GaussianPolicy(
                int(payload["state_dim"]),
                int(payload["action_dim"]),
                float(payload["initial_log_std"]),
            )
            self.actor.eval()
        self.actor.load_state_dict(payload["actor"])
        self.state_mean = torch.as_tensor(payload["state_mean"], dtype=torch.float32)
        self.state_std = torch.as_tensor(payload["state_std"], dtype=torch.float32)
        self.version = version

    def act(self, state: np.ndarray) -> np.ndarray:
        if self.actor is None or self.state_mean is None or self.state_std is None:
            raise RuntimeError("Actor policy has not been loaded")
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32)[None, :], dtype=torch.float32
        )
        normalized = (state_tensor - self.state_mean) / self.state_std
        with torch.no_grad():
            action, _ = self.actor.sample(normalized, deterministic=False)
        return action[0].numpy().astype(np.float32)


def wait_for_job(job_path: Path, stop_path: Path) -> dict[str, Any] | None:
    while not stop_path.exists():
        if job_path.is_file():
            try:
                return json.loads(job_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                time.sleep(0.05)
                continue
        time.sleep(0.05)
    return None


def main() -> None:
    args = parse_args()
    config = json.loads(args.run_config.read_text(encoding="utf-8"))
    worker_id = int(args.worker_id)
    if worker_id < 0:
        raise ValueError("worker-id must be non-negative")
    np.random.seed(args.worker_seed)
    torch.manual_seed(args.worker_seed)
    torch.set_num_threads(1)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    env = None
    jobs_root = args.ipc_root / "jobs"
    results_root = args.ipc_root / "results"
    ready_root = args.ipc_root / "ready"
    job_path = jobs_root / f"worker_{worker_id:03d}.json"
    stop_path = args.ipc_root / "stop"
    actor_path = args.ipc_root / "actor_latest.pt"
    ready_path = ready_root / f"worker_{worker_id:03d}.json"
    policy = ActorPolicy()
    try:
        from policy.RL.rfcl_env import RFCLPrivilegedEnv
        from policy.RL.task_factory import create_task

        task = create_task(
            config["task_name"],
            config["task_config"],
            save_dir=Path(config["snapshot_root"]),
            video_frequency=0,
            step_limit=int(config["step_limit"]),
            mode="eval_test",
            save_pre_move=False,
            task_variant="rfcl",
            device=args.device,
        )
        env = RFCLPrivilegedEnv(
            task,
            config["snapshot_root"],
            action_scale=np.asarray(config["action_scale"], dtype=np.float32),
            action_repeat=int(config["action_repeat"]),
            snapshot_sync_steps=int(config["snapshot_sync_steps"]),
            reverse_step_size=int(config["reverse_step_size"]),
            per_demo_buffer_size=int(config["per_demo_buffer_size"]),
            geometric_p=float(config["geometric_p"]),
            demo_horizon_to_max_steps_ratio=float(
                config["demo_horizon_to_max_steps_ratio"]
            ),
            minimum_episode_horizon=int(config["minimum_episode_horizon"]),
            action_scale_margin=float(config["action_scale_margin"]),
            seed=args.worker_seed,
            action_mode=str(config["action_mode"]),
        )
        write_json_atomic(
            ready_path,
            {
                "schema": "rfcl_distributed_worker_ready_v1",
                "session_id": config["session_id"],
                "worker_id": worker_id,
                "worker_seed": int(args.worker_seed),
                "device": str(args.device),
                "pid": os.getpid(),
            },
        )
        while True:
            job = wait_for_job(job_path, stop_path)
            if job is None:
                break
            if str(job.get("session_id")) != str(config["session_id"]):
                raise ValueError("Worker received a job from another session")
            job_id = int(job["job_id"])
            result_path = results_root / f"result_{job_id:012d}.npz"
            if result_path.exists():
                job_path.unlink(missing_ok=True)
                continue
            minimum_policy_version = int(job["policy_version"])
            if policy.version < minimum_policy_version:
                policy.load(actor_path, minimum_version=minimum_policy_version)
            demo_index = int(job["demo_index"])
            state_index = int(job["state_index"])
            transitions = []
            replay_eligible = []
            info: dict[str, Any] = {}
            started_at = time.perf_counter()
            try:
                if bool(job["bootstrap_handoff"]):
                    demo_seed = env.dataset.demos[demo_index].get("seed")
                    task.reset(seed=None if demo_seed is None else int(demo_seed))
                    env.adapter.prepare_handoff(task)
                state, reset_info = env.reset(
                    options={
                        "demo_index": demo_index,
                        "state_index": state_index,
                        "skip_task_reset": bool(config["bootstrap_handoff"]),
                    }
                )
                while True:
                    action = policy.act(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    transition = env.pop_transition()
                    if transition is None:
                        raise RuntimeError("RFCL environment did not expose a transition")
                    transitions.append(transition)
                    replay_eligible.append(bool(info.get("replay_eligible", True)))
                    state = next_state
                    if terminated or truncated:
                        break
                metadata = {
                    "session_id": str(config["session_id"]),
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "worker_seed": int(args.worker_seed),
                    "requested_policy_version": minimum_policy_version,
                    "policy_version": int(policy.version),
                    "demo_index": demo_index,
                    "state_index": state_index,
                    "raw_state_index": int(reset_info["raw_state_index"]),
                    "bootstrap_handoff": int(bool(job["bootstrap_handoff"])),
                    "success": int(bool(info.get("success", False))),
                    "steps": len(transitions),
                    "elapsed_s": time.perf_counter() - started_at,
                    "irrecoverable_failure": json.dumps(
                        info.get("irrecoverable_failure"), ensure_ascii=True
                    ),
                    "error": "",
                    "state_dim": int(env.observation_space.shape[0]),
                    "action_dim": int(env.action_space.shape[0]),
                }
                save_worker_result(
                    result_path,
                    metadata=metadata,
                    transitions=transitions,
                    replay_eligible=replay_eligible,
                )
            except Exception as exc:
                save_worker_result(
                    result_path,
                    metadata={
                        "session_id": str(config["session_id"]),
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "worker_seed": int(args.worker_seed),
                        "requested_policy_version": minimum_policy_version,
                        "policy_version": int(policy.version),
                        "demo_index": demo_index,
                        "state_index": state_index,
                        "bootstrap_handoff": int(bool(job["bootstrap_handoff"])),
                        "success": 0,
                        "steps": len(transitions),
                        "elapsed_s": time.perf_counter() - started_at,
                        "irrecoverable_failure": "null",
                        "error": traceback.format_exc(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "state_dim": int(env.observation_space.shape[0]),
                        "action_dim": int(env.action_space.shape[0]),
                    },
                    transitions=transitions,
                    replay_eligible=replay_eligible,
                )
                raise
            finally:
                job_path.unlink(missing_ok=True)
    finally:
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
