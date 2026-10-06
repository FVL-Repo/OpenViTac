from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train direct SAC RL with the BC actor as the initial policy."
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("bc_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=0,
        help=(
            "Number of steps before gradient updates. Keep this at 0 for "
            "BC-initialized collection so SAC does not take random warmup actions."
        ),
    )
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument(
        "--sac-policy-warmup-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the BC-initialized SAC actor, not random action-space "
            "samples, before --learning-starts. This lets learning_starts "
            "delay gradient updates without destroying insertion rollouts."
        ),
    )
    parser.add_argument(
        "--sac-actor-log-std-override",
        type=float,
        help=(
            "Optionally override the restored BC Gaussian actor log_std at SAC "
            "initialization. For insertion, values like -4.0 keep stochastic "
            "collection close to the deterministic BC mean while still allowing "
            "SAC exploration."
        ),
    )
    parser.add_argument(
        "--sac-actor-log-std-success-schedule",
        nargs="*",
        default=(),
        metavar="SUCCESSES:LOG_STD",
        help=(
            "Optional success-gated SAC actor log_std schedule, for example "
            "0:-4.0 5:-3.8 20:-3.6. The scheduled log_std is applied from "
            "the BC Gaussian actor and is keyed by collected successful "
            "archives, not timesteps."
        ),
    )
    parser.add_argument(
        "--sac-actor-update-after-successes",
        type=int,
        default=0,
        help=(
            "If positive, skip SAC actor optimizer updates until at least "
            "this many successful episodes have been archived. Critic updates "
            "still run, so BC behavior is protected while the replay buffer "
            "gets successful insertion data."
        ),
    )
    parser.add_argument(
        "--sac-actor-update-after-timesteps",
        type=int,
        default=0,
        help=(
            "If positive, also unlock SAC actor optimizer updates once this "
            "many environment timesteps have elapsed. When both success and "
            "timestep gates are set, reaching either one unlocks the actor."
        ),
    )
    parser.add_argument(
        "--sac-alpha-update-before-actor-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With automatic SAC entropy tuning, allow alpha updates while the "
            "actor update gate is still locked. Disabled by default so the "
            "entropy coefficient does not drift before actor learning starts."
        ),
    )
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--control-mode", choices=("direct",), default="direct")
    parser.add_argument("--reward-mode", choices=("sparse_success", "task"), default="sparse_success")
    parser.add_argument("--handoff-mode", choices=("auto", "none", "insert_usb_collect"), default="auto")
    parser.add_argument(
        "--insert-usb-handoff-distribution",
        choices=(
            "legacy",
            "coarse_preinsert",
            "direct",
            "precontact",
            "diverse_v1",
            "diverse_mild",
            "diverse_tiny",
            "curriculum_v1",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--insert-usb-curriculum-success-thresholds",
        type=int,
        nargs=3,
        default=(50, 100, 150),
        metavar=("STAGE1", "STAGE2", "STAGE3"),
        help=(
            "Successful archive counts that advance curriculum_v1 through "
            "larger xy/z-offset-only stages."
        ),
    )
    parser.add_argument("--insert-usb-xy-quit-threshold", type=float)
    parser.add_argument(
        "--insert-usb-coarse-z-jitter",
        type=float,
        default=0.002,
        help="Coarse-preinsert Z jitter half-range in meters; use 0 to disable.",
    )
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--task-mode", choices=("eval", "collect"), default="eval")
    parser.add_argument("--video-frequency", type=int, default=0)
    parser.add_argument("--collect-success-target", type=int, default=0)
    parser.add_argument(
        "--deterministic-bootstrap-episodes",
        type=int,
        default=0,
        help=(
            "Run this many deterministic BC-initialized SAC episodes before "
            "normal stochastic SAC learning, and add their transitions to "
            "the replay buffer."
        ),
    )
    parser.add_argument(
        "--save-successful-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sac-deterministic-collect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Debug mode: collect SAC rollouts with deterministic actor actions. "
            "This isolates environment/SAC plumbing from stochastic exploration."
        ),
    )
    parser.add_argument(
        "--debug-env-logging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-episode handoff diagnostics from TactileControlEnv.",
    )
    parser.add_argument(
        "--debug-step-log-frequency",
        type=int,
        default=0,
        help=(
            "If positive with --debug-env-logging, print env step diagnostics "
            "every N policy steps and on termination."
        ),
    )
    parser.add_argument("--force-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-freq", type=int, default=2000)
    parser.add_argument(
        "--resume-model",
        type=Path,
        help="Optional SAC/PPO .zip checkpoint to continue from.",
    )
    parser.add_argument(
        "--resume-collection-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume collected success count, attempts, and next seed from "
            "the output directory's environment/metadata.json and hdf5 folder."
        ),
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "SB3 reset_num_timesteps flag. Defaults to true for fresh runs "
            "and false when --resume-model is used."
        ),
    )
    parser.add_argument(
        "--memory-watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop, save, and exit cleanly before the container reaches its "
            "memory limit."
        ),
    )
    parser.add_argument(
        "--memory-watchdog-max-gb",
        type=float,
        default=None,
        help=(
            "Absolute cgroup/process memory threshold in GiB. If omitted, "
            "the watchdog uses --memory-watchdog-cgroup-fraction of the "
            "detected cgroup limit."
        ),
    )
    parser.add_argument(
        "--memory-watchdog-cgroup-fraction",
        type=float,
        default=0.86,
        help="Fraction of the cgroup memory limit used as the watchdog threshold.",
    )
    parser.add_argument(
        "--memory-watchdog-check-freq",
        type=int,
        default=20,
        help="Check memory every N SB3 callback steps.",
    )
    parser.add_argument(
        "--memory-watchdog-save-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also save the SAC replay buffer when the memory watchdog stops. "
            "Disabled by default because replay buffers can be large."
        ),
    )
    if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
        parser.print_help()
        raise SystemExit(0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import polyak_update

from policy.RL.gym_env import TactileControlEnv
from policy.RL.sb3_features import BCFeatureExtractor
from policy.RL.sb3_policy import BCGaussianSACPolicy
from policy.RL.task_factory import create_task


def log_stage(message: str) -> None:
    print(f"[train_rl] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


BYTES_PER_GIB = 1024**3


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / BYTES_PER_GIB


def format_gib(value: int | None) -> str:
    gib = bytes_to_gib(value)
    return "unknown" if gib is None else f"{gib:.2f}GiB"


def read_int_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    # cgroup v1 often reports a huge sentinel instead of "max".
    if value >= 1 << 60:
        return None
    return value


def process_rss_bytes() -> int | None:
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def cgroup_memory_current_bytes() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        value = read_int_file(path)
        if value is not None:
            return value
    return None


def cgroup_memory_limit_bytes() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        value = read_int_file(path)
        if value is not None:
            return value
    return None


def memory_snapshot() -> dict[str, int | None]:
    return {
        "process_rss_bytes": process_rss_bytes(),
        "cgroup_current_bytes": cgroup_memory_current_bytes(),
        "cgroup_limit_bytes": cgroup_memory_limit_bytes(),
    }


def load_existing_collection_state(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "environment" / "metadata.json"
    hdf5_dir = run_dir / "environment" / "hdf5"
    hdf5_successes = 0
    if hdf5_dir.exists():
        hdf5_successes = sum(1 for _ in hdf5_dir.glob("*.hdf5"))

    attempts = 0
    metadata_successes = 0
    failures = 0
    max_seed: int | None = None
    success_steps: list[float] = []

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse existing collection metadata: {metadata_path}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Expected metadata dict in existing collection: {metadata_path}"
            )
        for key, entry in metadata.items():
            if not isinstance(entry, dict):
                continue
            attempts += 1
            result = entry.get("result")
            if result == "success":
                metadata_successes += 1
                if "cost_step" in entry:
                    try:
                        success_steps.append(float(entry["cost_step"]))
                    except (TypeError, ValueError):
                        pass
            elif result == "fail":
                failures += 1

            seed_value = entry.get("rl_episode_seed", key)
            try:
                seed_int = int(seed_value)
            except (TypeError, ValueError):
                pass
            else:
                max_seed = seed_int if max_seed is None else max(max_seed, seed_int)

    successes = max(hdf5_successes, metadata_successes)
    return {
        "metadata_path": str(metadata_path),
        "hdf5_dir": str(hdf5_dir),
        "attempts": attempts,
        "metadata_successes": metadata_successes,
        "hdf5_successes": hdf5_successes,
        "successes": successes,
        "failures": failures,
        "next_seed": None if max_seed is None else max_seed + 1,
        "mean_success_steps": (
            None if not success_steps else float(np.mean(success_steps))
        ),
    }


def parse_success_log_std_schedule(
    entries: tuple[str, ...] | list[str],
) -> list[tuple[int, float]]:
    schedule: list[tuple[int, float]] = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                "--sac-actor-log-std-success-schedule entries must use "
                f"SUCCESSES:LOG_STD format, got {entry!r}"
            )
        successes_text, log_std_text = entry.split(":", 1)
        try:
            successes = int(successes_text)
            log_std = float(log_std_text)
        except ValueError as exc:
            raise ValueError(
                "--sac-actor-log-std-success-schedule entries must use "
                f"integer successes and float log_std, got {entry!r}"
            ) from exc
        if successes < 0:
            raise ValueError(
                "--sac-actor-log-std-success-schedule successes must be "
                f"non-negative, got {successes}"
            )
        schedule.append((successes, log_std))

    schedule.sort(key=lambda item: item[0])
    for (previous_successes, _), (successes, _) in zip(schedule, schedule[1:]):
        if successes == previous_successes:
            raise ValueError(
                "--sac-actor-log-std-success-schedule cannot repeat "
                f"success threshold {successes}"
            )
    return schedule


class SuccessTargetCallback(BaseCallback):
    def __init__(self, target_successes: int, initial_collected_successes: int = 0):
        super().__init__()
        self.target_successes = int(target_successes)
        self.collected_successes = int(initial_collected_successes)
        self.reached_target = False

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "collected_successes" in info:
                self.collected_successes = max(
                    self.collected_successes,
                    int(info["collected_successes"]),
                )
        if (
            self.target_successes > 0
            and self.collected_successes >= self.target_successes
        ):
            self.reached_target = True
            print(
                "[RL collect] reached success target: "
                f"{self.collected_successes}/{self.target_successes}",
                flush=True,
            )
            return False
        return True


def configure_sac_actor_update_gate(
    model,
    *,
    after_successes: int,
    after_timesteps: int,
    initial_collected_successes: int,
    alpha_update_before_gate: bool,
) -> None:
    after_successes = int(after_successes)
    after_timesteps = int(after_timesteps)
    if after_successes < 0:
        raise ValueError("--sac-actor-update-after-successes must be non-negative")
    if after_timesteps < 0:
        raise ValueError("--sac-actor-update-after-timesteps must be non-negative")
    model.sac_actor_update_after_successes = after_successes
    model.sac_actor_update_after_timesteps = after_timesteps
    model.sac_actor_update_collected_successes = int(initial_collected_successes)
    model.sac_alpha_update_before_actor_gate = bool(alpha_update_before_gate)
    model.sac_actor_updates_skipped_by_gate = 0


def sac_actor_update_gate_state(model) -> dict[str, object]:
    after_successes = int(getattr(model, "sac_actor_update_after_successes", 0) or 0)
    after_timesteps = int(getattr(model, "sac_actor_update_after_timesteps", 0) or 0)
    collected_successes = int(
        getattr(model, "sac_actor_update_collected_successes", 0) or 0
    )
    num_timesteps = int(getattr(model, "num_timesteps", 0) or 0)

    if after_successes <= 0 and after_timesteps <= 0:
        return {
            "enabled": True,
            "reason": "gate_disabled",
            "collected_successes": collected_successes,
            "after_successes": after_successes,
            "num_timesteps": num_timesteps,
            "after_timesteps": after_timesteps,
        }

    success_ready = after_successes > 0 and collected_successes >= after_successes
    timestep_ready = after_timesteps > 0 and num_timesteps >= after_timesteps
    enabled = bool(success_ready or timestep_ready)
    if success_ready:
        reason = "success_threshold"
    elif timestep_ready:
        reason = "timestep_threshold"
    else:
        reason = "waiting"

    return {
        "enabled": enabled,
        "reason": reason,
        "collected_successes": collected_successes,
        "after_successes": after_successes,
        "num_timesteps": num_timesteps,
        "after_timesteps": after_timesteps,
    }


class SacActorUpdateGateCallback(BaseCallback):
    def __init__(
        self,
        *,
        after_successes: int,
        after_timesteps: int,
        initial_collected_successes: int = 0,
    ):
        super().__init__()
        self.after_successes = int(after_successes)
        self.after_timesteps = int(after_timesteps)
        self.collected_successes = int(initial_collected_successes)
        self.last_enabled: bool | None = None
        self.last_successes: int | None = None

    def _log_state_if_needed(self, *, force: bool = False) -> None:
        state = sac_actor_update_gate_state(self.model)
        enabled = bool(state["enabled"])
        successes = int(state["collected_successes"])
        if (
            force
            or self.last_enabled is None
            or enabled != self.last_enabled
            or successes != self.last_successes
        ):
            self.last_enabled = enabled
            self.last_successes = successes
            print(
                "[SAC actor gate] "
                f"enabled={enabled} reason={state['reason']} "
                f"collected_successes={successes}/"
                f"{state['after_successes']} "
                f"num_timesteps={state['num_timesteps']}/"
                f"{state['after_timesteps']} "
                "alpha_update_before_gate="
                f"{getattr(self.model, 'sac_alpha_update_before_actor_gate', False)}",
                flush=True,
            )

    def _on_training_start(self) -> None:
        self.model.sac_actor_update_collected_successes = self.collected_successes
        self._log_state_if_needed(force=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "collected_successes" in info:
                self.collected_successes = max(
                    self.collected_successes,
                    int(info["collected_successes"]),
                )
        self.model.sac_actor_update_collected_successes = self.collected_successes
        self._log_state_if_needed()
        return True


class SacActorLogStdSuccessScheduleCallback(BaseCallback):
    def __init__(
        self,
        schedule: list[tuple[int, float]],
        initial_collected_successes: int = 0,
    ):
        super().__init__()
        self.schedule = list(schedule)
        self.collected_successes = int(initial_collected_successes)
        self.active_index = -1

    def _scheduled_index(self) -> int:
        index = -1
        for candidate_index, (successes, _) in enumerate(self.schedule):
            if self.collected_successes >= successes:
                index = candidate_index
            else:
                break
        return index

    def _set_actor_log_std(self, log_std: float, *, reason: str) -> None:
        actor = getattr(self.model.policy, "actor", None)
        gaussian_actor = getattr(actor, "gaussian_actor", None)
        if gaussian_actor is None:
            raise AttributeError(
                "SAC actor log_std schedule requires BCGaussianSACPolicy "
                "with actor.gaussian_actor"
            )
        if not gaussian_actor.log_std_min <= log_std <= gaussian_actor.log_std_max:
            raise ValueError(
                "Scheduled SAC actor log_std must be within the restored "
                f"Gaussian actor range [{gaussian_actor.log_std_min}, "
                f"{gaussian_actor.log_std_max}], got {log_std}"
            )
        with torch.no_grad():
            gaussian_actor.log_std_head.weight.zero_()
            gaussian_actor.log_std_head.bias.fill_(float(log_std))
        print(
            "[SAC log_std schedule] "
            f"reason={reason} collected_successes={self.collected_successes} "
            f"log_std={float(log_std):.4f} std={float(np.exp(log_std)):.6f}",
            flush=True,
        )

    def _apply_current_schedule(self, *, reason: str, force: bool = False) -> None:
        index = self._scheduled_index()
        if index < 0:
            return
        if force or index != self.active_index:
            self.active_index = index
            self._set_actor_log_std(self.schedule[index][1], reason=reason)

    def _on_training_start(self) -> None:
        self._apply_current_schedule(reason="training_start", force=True)

    def _on_rollout_start(self) -> None:
        self._apply_current_schedule(reason="rollout_start", force=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "collected_successes" in info:
                self.collected_successes = max(
                    self.collected_successes,
                    int(info["collected_successes"]),
                )
        self._apply_current_schedule(reason="success_threshold")
        return True


class MemoryWatchdogCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path,
        max_gb: float | None,
        cgroup_fraction: float,
        check_freq: int,
        save_replay_buffer: bool,
    ):
        super().__init__()
        if max_gb is not None and max_gb <= 0:
            raise ValueError("--memory-watchdog-max-gb must be positive")
        if not 0.0 < float(cgroup_fraction) <= 1.0:
            raise ValueError("--memory-watchdog-cgroup-fraction must be in (0, 1]")
        if int(check_freq) < 1:
            raise ValueError("--memory-watchdog-check-freq must be at least 1")
        self.run_dir = Path(run_dir)
        self.max_bytes = None if max_gb is None else int(float(max_gb) * BYTES_PER_GIB)
        self.cgroup_fraction = float(cgroup_fraction)
        self.check_freq = int(check_freq)
        self.save_replay_buffer = bool(save_replay_buffer)
        self.threshold_bytes: int | None = None
        self.threshold_source: str | None = None
        self.triggered = False
        self.stop_payload: dict[str, object] | None = None

    def _on_training_start(self) -> None:
        snapshot = memory_snapshot()
        if self.max_bytes is not None:
            self.threshold_bytes = self.max_bytes
            self.threshold_source = "absolute"
        elif snapshot["cgroup_limit_bytes"] is not None:
            self.threshold_bytes = int(
                snapshot["cgroup_limit_bytes"] * self.cgroup_fraction
            )
            self.threshold_source = "cgroup_fraction"
        else:
            self.threshold_bytes = None
            self.threshold_source = None

        if self.threshold_bytes is None:
            print(
                "[memory watchdog] disabled: no finite cgroup limit and no "
                "--memory-watchdog-max-gb",
                flush=True,
            )
            return

        print(
            "[memory watchdog] enabled "
            f"threshold={format_gib(self.threshold_bytes)} "
            f"source={self.threshold_source} check_freq={self.check_freq} "
            f"current_cgroup={format_gib(snapshot['cgroup_current_bytes'])} "
            f"process_rss={format_gib(snapshot['process_rss_bytes'])}",
            flush=True,
        )

    def _used_bytes(self, snapshot: dict[str, int | None]) -> int | None:
        return snapshot["cgroup_current_bytes"] or snapshot["process_rss_bytes"]

    def _save_stop_artifacts(self, payload: dict[str, object]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stop_path = self.run_dir / "memory_watchdog_stop.json"
        with stop_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

        model_path = self.run_dir / "memory_watchdog_model"
        self.model.save(model_path)
        payload["saved_model"] = str(model_path.with_suffix(".zip"))
        with stop_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

        if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
            replay_path = self.run_dir / "memory_watchdog_replay_buffer"
            self.model.save_replay_buffer(replay_path)
            payload["saved_replay_buffer"] = str(replay_path)
            with stop_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)

    def _on_step(self) -> bool:
        if self.threshold_bytes is None:
            return True
        if self.n_calls % self.check_freq != 0:
            return True

        snapshot = memory_snapshot()
        used_bytes = self._used_bytes(snapshot)
        if used_bytes is None or used_bytes < self.threshold_bytes:
            return True

        self.triggered = True
        self.stop_payload = {
            "reason": "memory_watchdog",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_timesteps": int(self.num_timesteps),
            "callback_calls": int(self.n_calls),
            "threshold_source": self.threshold_source,
            "threshold_bytes": int(self.threshold_bytes),
            "threshold_gib": bytes_to_gib(self.threshold_bytes),
            "used_bytes": int(used_bytes),
            "used_gib": bytes_to_gib(used_bytes),
            **snapshot,
        }
        print(
            "[memory watchdog] stopping cleanly before OOM "
            f"used={format_gib(used_bytes)} "
            f"threshold={format_gib(self.threshold_bytes)} "
            f"num_timesteps={self.num_timesteps}",
            flush=True,
        )
        self._save_stop_artifacts(self.stop_payload)
        return False


class BCPolicyWarmupSAC(SAC):
    """SAC variant that uses the BC-initialized actor during warmup collection."""

    def _sac_actor_gate_state(self) -> dict[str, object]:
        return sac_actor_update_gate_state(self)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        actor_gate_state = self._sac_actor_gate_state()
        actor_updates_enabled = bool(actor_gate_state["enabled"])
        alpha_update_before_gate = bool(
            getattr(self, "sac_alpha_update_before_actor_gate", False)
        )

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        skipped_actor_updates = 0

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                if actor_updates_enabled or alpha_update_before_gate:
                    ent_coef_loss = -(
                        self.log_ent_coef
                        * (log_prob + self.target_entropy).detach()
                    ).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = torch.cat(
                    self.critic_target(
                        replay_data.next_observations,
                        next_actions,
                    ),
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = (
                    next_q_values
                    - ent_coef * next_log_prob.reshape(-1, 1)
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions,
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if actor_updates_enabled:
                q_values_pi = torch.cat(
                    self.critic(replay_data.observations, actions_pi),
                    dim=1,
                )
                min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
            else:
                skipped_actor_updates += 1

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.sac_actor_updates_skipped_by_gate = int(
            getattr(self, "sac_actor_updates_skipped_by_gate", 0)
            + skipped_actor_updates
        )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(
            "train/actor_update_enabled",
            float(actor_updates_enabled),
        )
        self.logger.record(
            "train/actor_updates_skipped_by_gate",
            int(getattr(self, "sac_actor_updates_skipped_by_gate", 0)),
        )
        self.logger.record(
            "train/actor_gate_collected_successes",
            int(actor_gate_state["collected_successes"]),
        )
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _sample_policy_action(
        self,
        *,
        deterministic: bool,
        action_noise: ActionNoise | None,
        n_envs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._last_obs is not None, "self._last_obs was not set"
        unscaled_action, _ = self.predict(
            self._last_obs,
            deterministic=deterministic,
        )

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.num_timesteps < learning_starts
            and not (self.use_sde and self.use_sde_at_warmup)
        ):
            return self._sample_policy_action(
                deterministic=False,
                action_noise=action_noise,
                n_envs=n_envs,
            )
        return super()._sample_action(
            learning_starts=learning_starts,
            action_noise=action_noise,
            n_envs=n_envs,
        )


class DeterministicCollectSAC(BCPolicyWarmupSAC):
    """SAC variant for debugging rollout collection with deterministic actions."""

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        del learning_starts
        return self._sample_policy_action(
            deterministic=True,
            action_noise=action_noise,
            n_envs=n_envs,
        )


def run_deterministic_bootstrap(model, episodes: int) -> dict[str, object]:
    episodes = int(episodes)
    if episodes <= 0:
        return {
            "episodes": 0,
            "successes": 0,
            "transitions": 0,
            "success_rate": 0.0,
        }
    if model.replay_buffer is None:
        raise RuntimeError("SAC replay buffer must exist before bootstrap")

    log_stage(f"deterministic bootstrap reset start episodes={episodes}")
    observation = model.env.reset()
    log_stage("deterministic bootstrap reset done")
    model._last_obs = observation
    successes = 0
    transitions = 0
    completed = 0
    episode_steps = 0

    while completed < episodes:
        action, _ = model.predict(observation, deterministic=True)
        buffer_action = model.policy.scale_action(action)
        if episode_steps == 0:
            log_stage(
                "deterministic bootstrap episode "
                f"{completed + 1}/{episodes} step start"
            )
        next_observation, reward, done, infos = model.env.step(action)
        model._store_transition(
            model.replay_buffer,
            buffer_action,
            next_observation,
            reward,
            done,
            infos,
        )
        transitions += int(model.env.num_envs)
        episode_steps += 1
        if episode_steps % 10 == 0:
            log_stage(
                "deterministic bootstrap episode "
                f"{completed + 1}/{episodes} step={episode_steps}"
            )
        observation = next_observation

        for env_index, info in enumerate(infos):
            if done[env_index]:
                success = bool(info.get("success", False))
                successes += int(success)
                completed += 1
                print(
                    "[deterministic bootstrap] "
                    f"episode {completed}/{episodes}: "
                    f"success={success} "
                    f"collected_successes={info.get('collected_successes')}",
                    flush=True,
                )
                episode_steps = 0
                if completed >= episodes:
                    break

    # Force model.learn() to reset cleanly after bootstrap while keeping replay data.
    model._last_obs = None
    return {
        "episodes": completed,
        "successes": successes,
        "transitions": transitions,
        "success_rate": successes / max(completed, 1),
        "replay_buffer_size": int(model.replay_buffer.size()),
    }


def summarize_sac_actor_log_std(model) -> dict[str, float]:
    policy_actor = getattr(getattr(model, "policy", None), "actor", None)
    gaussian_actor = getattr(policy_actor, "gaussian_actor", None)
    if gaussian_actor is None:
        return {}
    bias = gaussian_actor.log_std_head.bias.detach().cpu().numpy()
    weight = gaussian_actor.log_std_head.weight.detach().cpu().numpy()
    return {
        "log_std_min": float(gaussian_actor.log_std_min),
        "log_std_max": float(gaussian_actor.log_std_max),
        "bias_mean": float(np.mean(bias)),
        "bias_min": float(np.min(bias)),
        "bias_max": float(np.max(bias)),
        "std_from_bias_mean": float(np.exp(np.mean(bias))),
        "weight_abs_max": float(np.max(np.abs(weight))),
    }


def synchronize_model_seed(model, seed: int) -> None:
    """Keep SB3's stored seed aligned with the collection episode seed."""
    seed = int(seed)
    model.seed = seed
    model.set_random_seed(seed)
    log_stage(f"model seed synchronized seed={seed}")


def main() -> None:
    if args.algorithm == "ppo":
        raise NotImplementedError(
            "PPO is not wired to the BC GaussianActor yet. "
            "Use --algorithm sac for BC-initialized direct RL."
        )

    sac_actor_log_std_success_schedule = parse_success_log_std_schedule(
        args.sac_actor_log_std_success_schedule
    )
    run_dir = args.output_dir.expanduser().resolve() / args.algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device is not None else "cuda:0"
    collection_state: dict[str, object] = {}
    if args.resume_collection_state:
        collection_state = load_existing_collection_state(run_dir)
    initial_collected_successes = int(collection_state.get("successes", 0))
    initial_collection_attempts = int(collection_state.get("attempts", 0))
    initial_collection_failures = int(collection_state.get("failures", 0))
    resumed_next_seed = collection_state.get("next_seed")
    effective_seed = (
        int(resumed_next_seed) if resumed_next_seed is not None else int(args.seed)
    )
    reset_num_timesteps = args.reset_num_timesteps
    if reset_num_timesteps is None:
        reset_num_timesteps = args.resume_model is None
    log_stage(
        "main start "
        f"run_dir={run_dir} algorithm={args.algorithm} "
        f"step_limit={args.step_limit} action_repeat={args.action_repeat} "
        f"resume_model={args.resume_model} "
        f"resume_collection_state={args.resume_collection_state} "
        f"initial_collected_successes={initial_collected_successes} "
        f"effective_seed={effective_seed} "
        f"reset_num_timesteps={reset_num_timesteps} "
        f"memory_watchdog={args.memory_watchdog} "
        f"deterministic_bootstrap_episodes={args.deterministic_bootstrap_episodes} "
        f"sac_policy_warmup_actions={args.sac_policy_warmup_actions} "
        f"sac_deterministic_collect={args.sac_deterministic_collect} "
        f"sac_actor_log_std_override={args.sac_actor_log_std_override} "
        "sac_actor_log_std_success_schedule="
        f"{sac_actor_log_std_success_schedule} "
        "sac_actor_update_after_successes="
        f"{args.sac_actor_update_after_successes} "
        "sac_actor_update_after_timesteps="
        f"{args.sac_actor_update_after_timesteps} "
        "sac_alpha_update_before_actor_gate="
        f"{args.sac_alpha_update_before_actor_gate}"
    )
    if collection_state:
        log_stage(
            "resumed collection state "
            f"{json.dumps(collection_state, sort_keys=True)}"
        )
    collect_successful_episodes = (
        args.save_successful_episodes
        or args.collect_success_target > 0
    )
    clean_finished_episodes = (
        collect_successful_episodes
        or args.task_mode == "collect"
    )
    log_stage("create_task start")
    task = create_task(
        args.task_name,
        args.task_config,
        save_dir=run_dir / "environment",
        video_frequency=args.video_frequency,
        step_limit=args.step_limit,
        mode=args.task_mode,
        save_pre_move=True,
        device=device,
    )
    log_stage(
        "create_task done "
        f"task_mode={getattr(task, 'mode', None)} "
        f"save_frequency={getattr(task.cfg, 'save_frequency', None)} "
        f"save_pre_move={getattr(task.cfg, 'save_pre_move', None)} "
        f"tmp_save_dir={getattr(task, 'tmp_save_dir', None)}"
    )
    log_stage("TactileControlEnv init start")
    env = TactileControlEnv(
        task,
        args.bc_checkpoint,
        image_size=args.image_size,
        action_repeat=args.action_repeat,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        handoff_mode=args.handoff_mode,
        insert_usb_handoff_distribution=args.insert_usb_handoff_distribution,
        insert_usb_coarse_z_jitter=args.insert_usb_coarse_z_jitter,
        insert_usb_curriculum_success_thresholds=tuple(
            args.insert_usb_curriculum_success_thresholds
        ),
        insert_usb_xy_quit_threshold=args.insert_usb_xy_quit_threshold,
        force_control=args.force_control,
        collect_successful_episodes=collect_successful_episodes,
        clean_finished_episodes=clean_finished_episodes,
        debug_logging=args.debug_env_logging,
        debug_step_log_frequency=args.debug_step_log_frequency,
        collection_metadata={
            "rl_algorithm": args.algorithm,
            "rl_output_dir": str(run_dir),
            "rl_resume_collection_state": collection_state,
        },
        seed=effective_seed,
        device=device,
    )
    if args.resume_collection_state:
        env.collected_successes = initial_collected_successes
        env.collection_attempts = initial_collection_attempts
        env.collection_failures = initial_collection_failures
        if collection_state.get("mean_success_steps") is not None:
            env.collection_mean_steps = float(collection_state["mean_success_steps"])
        log_stage(
            "TactileControlEnv resumed "
            f"collected_successes={env.collected_successes} "
            f"attempts={env.collection_attempts} "
            f"failures={env.collection_failures} "
            f"next_seed={env.next_seed}"
        )
    log_stage("TactileControlEnv init done")
    environment = Monitor(
        env,
        filename=str(run_dir / "monitor.csv"),
        override_existing=not args.resume_collection_state,
        info_keywords=("success",),
    )
    log_stage("Monitor init done")

    policy_kwargs = {
        "features_extractor_class": BCFeatureExtractor,
        "features_extractor_kwargs": {
            "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
            "freeze": args.freeze_encoder,
        },
        "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
        "freeze_actor_encoder": args.freeze_encoder,
        "bc_actor_log_std_override": args.sac_actor_log_std_override,
        "normalize_images": True,
        "net_arch": [256, 256],
        "activation_fn": nn.GELU,
    }

    if args.algorithm == "sac":
        if args.sac_deterministic_collect:
            sac_class = DeterministicCollectSAC
        elif args.sac_policy_warmup_actions:
            sac_class = BCPolicyWarmupSAC
        else:
            sac_class = SAC
        if (
            (args.sac_actor_update_after_successes > 0
            or args.sac_actor_update_after_timesteps > 0)
            and not issubclass(sac_class, BCPolicyWarmupSAC)
        ):
            raise ValueError(
                "SAC actor update gates require --sac-policy-warmup-actions "
                "or --sac-deterministic-collect so the gated SAC subclass is used."
            )
        if args.resume_model is not None:
            resume_model_path = args.resume_model.expanduser().resolve()
            if not resume_model_path.exists():
                raise FileNotFoundError(f"--resume-model not found: {resume_model_path}")
            log_stage(f"SAC load start resume_model={resume_model_path}")
            model = sac_class.load(
                resume_model_path,
                env=environment,
                device=device,
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            log_stage("SAC load done")
        else:
            log_stage("SAC init start")
            model = sac_class(
                BCGaussianSACPolicy,
                environment,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                train_freq=args.train_frequency,
                gradient_steps=args.gradient_steps,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                seed=effective_seed,
                device=device,
                verbose=1,
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            log_stage("SAC init done")
    else:
        if args.resume_model is not None:
            resume_model_path = args.resume_model.expanduser().resolve()
            if not resume_model_path.exists():
                raise FileNotFoundError(f"--resume-model not found: {resume_model_path}")
            log_stage(f"PPO load start resume_model={resume_model_path}")
            model = PPO.load(
                resume_model_path,
                env=environment,
                device=device,
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            log_stage("PPO load done")
        else:
            log_stage("PPO init start")
            model = PPO(
                "MultiInputPolicy",
                environment,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                n_steps=256,
                batch_size=args.batch_size,
                n_epochs=10,
                gamma=args.gamma,
                gae_lambda=0.95,
                seed=effective_seed,
                device=device,
                verbose=1,
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            log_stage("PPO init done")

    synchronize_model_seed(model, effective_seed)
    if args.algorithm == "sac":
        configure_sac_actor_update_gate(
            model,
            after_successes=args.sac_actor_update_after_successes,
            after_timesteps=args.sac_actor_update_after_timesteps,
            initial_collected_successes=initial_collected_successes,
            alpha_update_before_gate=args.sac_alpha_update_before_actor_gate,
        )
        log_stage(
            "SAC actor update gate "
            f"{json.dumps(sac_actor_update_gate_state(model), sort_keys=True)}"
        )

    sac_actor_log_std_summary = (
        summarize_sac_actor_log_std(model)
        if args.algorithm == "sac"
        else {}
    )
    if sac_actor_log_std_summary:
        log_stage(
            "SAC actor log_std "
            f"{json.dumps(sac_actor_log_std_summary, sort_keys=True)}"
        )

    log_stage("callbacks init start")
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix=args.algorithm,
        save_replay_buffer=args.algorithm == "sac" and args.save_replay_buffer,
    )
    callbacks = [checkpoint_callback]
    memory_watchdog_callback = None
    if args.memory_watchdog:
        memory_watchdog_callback = MemoryWatchdogCallback(
            run_dir=run_dir,
            max_gb=args.memory_watchdog_max_gb,
            cgroup_fraction=args.memory_watchdog_cgroup_fraction,
            check_freq=args.memory_watchdog_check_freq,
            save_replay_buffer=args.memory_watchdog_save_replay_buffer,
        )
        callbacks.append(memory_watchdog_callback)
    sac_actor_update_gate_callback = None
    if (
        args.algorithm == "sac"
        and (
            args.sac_actor_update_after_successes > 0
            or args.sac_actor_update_after_timesteps > 0
        )
    ):
        sac_actor_update_gate_callback = SacActorUpdateGateCallback(
            after_successes=args.sac_actor_update_after_successes,
            after_timesteps=args.sac_actor_update_after_timesteps,
            initial_collected_successes=initial_collected_successes,
        )
        callbacks.append(sac_actor_update_gate_callback)
    if sac_actor_log_std_success_schedule:
        callbacks.append(
            SacActorLogStdSuccessScheduleCallback(
                sac_actor_log_std_success_schedule,
                initial_collected_successes=initial_collected_successes,
            )
        )
    success_target_callback = None
    if args.collect_success_target > 0:
        success_target_callback = SuccessTargetCallback(
            args.collect_success_target,
            initial_collected_successes=initial_collected_successes,
        )
        callbacks.append(success_target_callback)
    callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
    log_stage("callbacks init done")

    log_stage("deterministic bootstrap start")
    bootstrap_summary = run_deterministic_bootstrap(
        model,
        args.deterministic_bootstrap_episodes,
    )
    log_stage("deterministic bootstrap done")
    if bootstrap_summary["episodes"]:
        if args.algorithm == "sac":
            bootstrap_collected_successes = (
                initial_collected_successes
                + int(bootstrap_summary["successes"])
            )
            model.sac_actor_update_collected_successes = max(
                int(getattr(model, "sac_actor_update_collected_successes", 0)),
                bootstrap_collected_successes,
            )
            if sac_actor_update_gate_callback is not None:
                sac_actor_update_gate_callback.collected_successes = int(
                    model.sac_actor_update_collected_successes
                )
        print(
            "[deterministic bootstrap] summary: "
            f"{json.dumps(bootstrap_summary, sort_keys=True)}",
            flush=True,
        )

    log_stage("model.learn start")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    log_stage("model.learn done")
    if memory_watchdog_callback is not None and memory_watchdog_callback.triggered:
        stop_reason = "memory_watchdog"
    elif success_target_callback is not None and success_target_callback.reached_target:
        stop_reason = "success_target"
    else:
        stop_reason = "total_timesteps"
    model.save(run_dir / "final_model")
    log_stage("final model saved")
    summary = {
        "algorithm": args.algorithm,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
        "total_timesteps": args.total_timesteps,
        "learning_starts": args.learning_starts,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "train_frequency": args.train_frequency,
        "gradient_steps": args.gradient_steps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "sac_actor_log_std_override": args.sac_actor_log_std_override,
        "sac_actor_log_std_success_schedule": [
            {"successes": successes, "log_std": log_std}
            for successes, log_std in sac_actor_log_std_success_schedule
        ],
        "sac_actor_update_after_successes": (
            args.sac_actor_update_after_successes
        ),
        "sac_actor_update_after_timesteps": (
            args.sac_actor_update_after_timesteps
        ),
        "sac_alpha_update_before_actor_gate": (
            args.sac_alpha_update_before_actor_gate
        ),
        "sac_actor_update_gate_state": (
            sac_actor_update_gate_state(model)
            if args.algorithm == "sac"
            else None
        ),
        "sac_actor_updates_skipped_by_gate": int(
            getattr(model, "sac_actor_updates_skipped_by_gate", 0)
        ),
        "sac_actor_log_std_summary": sac_actor_log_std_summary,
        "seed": args.seed,
        "effective_seed": effective_seed,
        "resume_model": (
            None
            if args.resume_model is None
            else str(args.resume_model.expanduser().resolve())
        ),
        "resume_collection_state": args.resume_collection_state,
        "collection_state_at_start": collection_state,
        "initial_collected_successes": initial_collected_successes,
        "initial_collection_attempts": initial_collection_attempts,
        "initial_collection_failures": initial_collection_failures,
        "reset_num_timesteps": reset_num_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "stop_reason": stop_reason,
        "memory_watchdog": args.memory_watchdog,
        "memory_watchdog_max_gb": args.memory_watchdog_max_gb,
        "memory_watchdog_cgroup_fraction": args.memory_watchdog_cgroup_fraction,
        "memory_watchdog_check_freq": args.memory_watchdog_check_freq,
        "memory_watchdog_triggered": (
            False
            if memory_watchdog_callback is None
            else memory_watchdog_callback.triggered
        ),
        "memory_watchdog_stop_payload": (
            None
            if memory_watchdog_callback is None
            else memory_watchdog_callback.stop_payload
        ),
        "image_size": args.image_size,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "handoff_mode": args.handoff_mode,
        "insert_usb_handoff_distribution": args.insert_usb_handoff_distribution,
        "insert_usb_coarse_z_jitter": args.insert_usb_coarse_z_jitter,
        "insert_usb_curriculum_success_thresholds": list(
            args.insert_usb_curriculum_success_thresholds
        ),
        "insert_usb_xy_quit_threshold": args.insert_usb_xy_quit_threshold,
        "action_repeat": args.action_repeat,
        "step_limit": args.step_limit,
        "force_control": args.force_control,
        "freeze_encoder": args.freeze_encoder,
        "task_mode": args.task_mode,
        "video_frequency": args.video_frequency,
        "collect_success_target": args.collect_success_target,
        "sac_policy_warmup_actions": args.sac_policy_warmup_actions,
        "sac_deterministic_collect": args.sac_deterministic_collect,
        "debug_env_logging": args.debug_env_logging,
        "debug_step_log_frequency": args.debug_step_log_frequency,
        "deterministic_bootstrap_episodes": (
            args.deterministic_bootstrap_episodes
        ),
        "deterministic_bootstrap_summary": bootstrap_summary,
        "save_successful_episodes": args.save_successful_episodes,
        "collect_successful_episodes": collect_successful_episodes,
        "clean_finished_episodes": clean_finished_episodes,
    }
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    log_stage("training summary saved")
    environment.close()
    log_stage("environment closed")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
