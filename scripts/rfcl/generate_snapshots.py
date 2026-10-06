"""Generate persistent full-state RFCL demos from Motion Plan episodes.

The recorder keeps the final pre-insert checkpoint, then dumps every
``--stride`` simulator step until the first frame satisfying the task's
original success predicate.  It deliberately stops before the scripted
gripper release, so the resulting demos match the RFCL environment's fixed
closed-gripper action space.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ._bootstrap import add_repository_root
except ImportError:
    from _bootstrap import add_repository_root

add_repository_root()

from isaaclab.app import AppLauncher

from policy.RL.checkpoint import extract_action_scale_from_bc_checkpoint
from policy.RL.tasks.insert_usb import (
    BALANCED_40_PLAN_ID,
    balanced_40_profiles,
)
from policy.RL.rfcl_snapshot import (
    SNAPSHOT_DATASET_SCHEMA,
    SNAPSHOT_MANIFEST_NAME,
    RFCLSnapshot,
    archive_uipc_frame,
    capture_snapshot,
    write_snapshot_sidecar,
)
from policy.RL.rfcl_task_adapter import RFCLTaskAdapter, create_rfcl_task_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--demo-plan",
        choices=("seeded", "insert_usb_balanced40"),
        default="seeded",
    )
    parser.add_argument("--profile-indices", type=int, nargs="+")
    parser.add_argument("--seed-base", type=int, default=40_000)
    parser.add_argument("--max-attempts-per-profile", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--action-mode",
        choices=("qpos_delta", "target_pos_vel", "target_pos_vel_force"),
        default="qpos_delta",
        help="RFCL action representation stored in the manifest.",
    )
    parser.add_argument("--step-limit", type=int, default=600)
    parser.add_argument(
        "--skip-failed-demos",
        action="store_true",
        help=(
            "Record failed Motion Plan seeds as diagnostics and continue; only "
            "successful demos are added to the RFCL manifest."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rfcl/snapshot_pilot_v2"),
    )
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument(
        "--action-scale",
        type=float,
        nargs="+",
        metavar="SCALE",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def _action_scale(args: argparse.Namespace) -> np.ndarray:
    if args.action_scale is not None and args.bc_checkpoint is not None:
        raise ValueError("Pass either --action-scale or --bc-checkpoint, not both")
    if args.action_scale is not None:
        scale = np.asarray(args.action_scale, dtype=np.float32)
    elif args.bc_checkpoint is not None:
        checkpoint = torch.load(
            args.bc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        scale = extract_action_scale_from_bc_checkpoint(
            checkpoint,
            expected_action_dim=7,
            device="cpu",
        ).numpy().astype(np.float32)
    else:
        # This is the validated scale used by the existing qpos BC datasets.
        scale = np.asarray(
            [
                0.000938745797611773,
                0.0023768171668052673,
                0.0007808265509083867,
                0.0016817807918414474,
                0.0012212832225486636,
                0.00456193694844842,
                0.001278358744457364,
            ],
            dtype=np.float32,
        )
    if args.action_mode in ("target_pos_vel", "target_pos_vel_force"):
        if scale.shape == (7,):
            # Planner velocity targets are in rad/s.  This range covers the
            # saved USB insertion commands while leaving exploration headroom.
            scale = np.concatenate((scale, np.full(7, 0.25, dtype=np.float32)))
        if scale.shape == (14,) and args.action_mode == "target_pos_vel_force":
            scale = np.concatenate((scale, np.ones(1, dtype=np.float32)))
        expected = 15 if args.action_mode == "target_pos_vel_force" else 14
        if scale.shape != (expected,):
            raise ValueError(f"{args.action_mode} action scale must have shape ({expected},), got {scale.shape}")
    elif scale.shape != (7,):
        raise ValueError(f"qpos_delta action scale must have shape (7,), got {scale.shape}")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"action scale must be finite positive, got {scale}")
    return scale


def _tag(task: Any) -> str:
    return str(task.atom_tag).strip().lower()


def _snapshot_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.demo_plan == "seeded":
        if args.profile_indices is not None:
            raise ValueError("--profile-indices requires --demo-plan insert_usb_balanced40")
        seeds = [0, 1, 2, 3, 4] if args.seeds is None else list(args.seeds)
        if len(set(map(int, seeds))) != len(seeds):
            raise ValueError("--seeds must not contain duplicates")
        return [
            {
                "demo_index": index,
                "demo_id": str(int(seed)),
                "profile": None,
                "attempt_seeds": [int(seed)],
            }
            for index, seed in enumerate(seeds)
        ]

    if args.task_name != "insert_USB":
        raise ValueError("insert_usb_balanced40 requires --task-name insert_USB")
    if args.seeds is not None:
        raise ValueError(
            "--seeds cannot be combined with --demo-plan insert_usb_balanced40; "
            "use --seed-base and --max-attempts-per-profile"
        )
    if args.max_attempts_per_profile <= 0:
        raise ValueError("--max-attempts-per-profile must be positive")
    indexed_profiles = list(enumerate(balanced_40_profiles(), start=1))
    if args.profile_indices is not None:
        indices = list(map(int, args.profile_indices))
        if len(set(indices)) != len(indices):
            raise ValueError("--profile-indices must not contain duplicates")
        invalid = [
            index for index in indices if not 1 <= index <= len(indexed_profiles)
        ]
        if invalid:
            raise ValueError(f"Profile indices must be in [1, 40], got {invalid}")
        requested = set(indices)
        indexed_profiles = [
            item for item in indexed_profiles if item[0] in requested
        ]
    return [
        {
            "demo_index": profile_number - 1,
            "demo_id": profile.profile_id,
            "profile": profile,
            "attempt_seeds": [
                int(args.seed_base + profile_number * 100 + attempt)
                for attempt in range(args.max_attempts_per_profile)
            ],
        }
        for profile_number, profile in indexed_profiles
    ]


class SnapshotRecorder:
    def __init__(
        self,
        task: Any,
        adapter: RFCLTaskAdapter,
        root: Path,
        *,
        demo_index: int,
        seed: int,
        stride: int,
        demo_id: str | None = None,
    ) -> None:
        self.task = task
        self.adapter = adapter
        self.root = root
        self.demo_index = int(demo_index)
        self.seed = int(seed)
        self.demo_id = str(seed) if demo_id is None else str(demo_id)
        if demo_id is None:
            self.snapshot_prefix = f"demo_{self.seed:06d}"
        else:
            safe_demo_id = "".join(
                character
                if character.isalnum() or character in ("-", "_")
                else "_"
                for character in self.demo_id
            )
            self.snapshot_prefix = f"demo_{safe_demo_id}"
        self.stride = int(stride)
        self.snapshots: list[dict[str, Any]] = []
        self.snapshot_objects: list[RFCLSnapshot] = []
        self.pending_start: RFCLSnapshot | None = None
        self.suffix_started = False
        self.terminal_success = False
        self._original_step = None
        self._original_set_arm = None
        self._original_set_gripper = None
        self._force_position_write_pending = False
        self._current_force_position_write = True
        self._command_arm_pos = None
        self._command_arm_vel = None

    def install(self) -> None:
        self._original_step = self.task._step
        manager = self.task._robot_manager
        self._original_set_arm = manager.set_arm
        self._original_set_gripper = manager.set_gripper

        def recorded_set_arm(*args: Any, **kwargs: Any) -> Any:
            if bool(kwargs.get("force", True)):
                self._force_position_write_pending = True
            if args:
                self._command_arm_pos = np.asarray(
                    args[0].detach().cpu().numpy()
                    if torch.is_tensor(args[0])
                    else args[0],
                    dtype=np.float32,
                ).reshape(-1)[:7].copy()
                velocity = kwargs.get("vel", args[1] if len(args) > 1 else None)
                if velocity is not None:
                    self._command_arm_vel = np.asarray(
                        velocity.detach().cpu().numpy()
                        if torch.is_tensor(velocity)
                        else velocity,
                        dtype=np.float32,
                    ).reshape(-1)[:7].copy()
            return self._original_set_arm(*args, **kwargs)

        def recorded_set_gripper(*args: Any, **kwargs: Any) -> Any:
            if bool(kwargs.get("force", True)):
                self._force_position_write_pending = True
            return self._original_set_gripper(*args, **kwargs)

        def recorded_step(*args: Any, **kwargs: Any) -> Any:
            result = self._original_step(*args, **kwargs)
            self._current_force_position_write = self._force_position_write_pending
            self._force_position_write_pending = False
            self._observe_step()
            return result

        self.task._step = recorded_step
        manager.set_arm = recorded_set_arm
        manager.set_gripper = recorded_set_gripper

    def uninstall(self) -> None:
        if self._original_step is not None:
            self.task._step = self._original_step
        manager = self.task._robot_manager
        if self._original_set_arm is not None:
            manager.set_arm = self._original_set_arm
        if self._original_set_gripper is not None:
            manager.set_gripper = self._original_set_gripper

    def _capture(self, success: bool) -> RFCLSnapshot:
        snapshot_id = (
            f"{self.snapshot_prefix}_state_{len(self.snapshot_objects):04d}"
        )
        snapshot = capture_snapshot(
            self.task,
            self.adapter,
            snapshot_id=snapshot_id,
            demo_id=self.demo_id,
            state_index=-1,
            success=bool(success),
            force_position_write=self._current_force_position_write,
        )
        if self._command_arm_pos is not None:
            snapshot.robot_state["joint_pos_target"] = np.asarray(
                snapshot.robot_state["joint_pos_target"]
            ).copy()
            snapshot.robot_state["joint_pos_target"].reshape(-1)[:7] = (
                self._command_arm_pos
            )
        if self._command_arm_vel is not None:
            snapshot.robot_state["joint_vel_target"] = np.asarray(
                snapshot.robot_state["joint_vel_target"]
            ).copy()
            snapshot.robot_state["joint_vel_target"].reshape(-1)[:7] = (
                self._command_arm_vel
            )
        # Each reset can reuse low world.frame() values.  Reserve one million
        # logical frames per demo so all checkpoints remain addressable in the
        # single UIPC workspace used by the training environment.  Repeated
        # pending handoff captures intentionally replace local state zero.
        if len(self.snapshot_objects) >= 1_000_000:
            raise RuntimeError("RFCL demo exceeds its logical UIPC frame range")
        logical_frame = (
            (self.demo_index + 1) * 1_000_000 + len(self.snapshot_objects)
        )
        archive_uipc_frame(
            self.root / "scene" / "dump",
            snapshot.uipc_frame,
            logical_frame,
        )
        snapshot.uipc_frame = logical_frame
        return snapshot

    def _emit(self, snapshot: RFCLSnapshot) -> None:
        snapshot.state_index = len(self.snapshot_objects)
        metadata = write_snapshot_sidecar(self.root, snapshot)
        self.snapshot_objects.append(snapshot)
        self.snapshots.append(metadata)

    def _observe_step(self) -> None:
        if self.terminal_success:
            return
        current_tag = _tag(self.task)

        # Keep replacing the pending checkpoint until the first policy-owned
        # transition starts.  Insert USB can temporarily leave and re-enter
        # move_usb_to_pre_insert while it settles at the coarse handoff, so
        # merely detecting a tag change would start RFCL too early.
        if not self.suffix_started:
            if self.pending_start is None:
                if not self.adapter.is_snapshot_start(current_tag):
                    return
                if int(self.task.step_count) % self.stride == 0:
                    self.pending_start = self._capture(success=False)
                return

            if not self.adapter.is_policy_entry(current_tag):
                if int(self.task.step_count) % self.stride == 0:
                    self.pending_start = self._capture(success=False)
                return
            self.suffix_started = True
            self._emit(self.pending_start)

        if not self.adapter.is_snapshot_eligible(current_tag):
            raise RuntimeError(
                f"Unexpected RFCL suffix tag {current_tag!r} at step "
                f"{self.task.step_count}"
            )

        success = self.adapter.check_success(self.task)
        due = int(self.task.step_count) % self.stride == 0
        if due or success:
            self._emit(self._capture(success=success))
        if success:
            self.terminal_success = True

    def finish(self) -> dict[str, Any]:
        if self.pending_start is not None and not self.suffix_started:
            raise RuntimeError("Motion Plan never left the pre-insert atom")
        if len(self.snapshot_objects) < 2:
            raise RuntimeError(
                f"RFCL demo {self.demo_id} has too few snapshots: "
                f"{len(self.snapshot_objects)}"
            )
        if not self.terminal_success:
            raise RuntimeError(f"RFCL demo {self.demo_id} never reached success")
        handoff_errors = [
            self.adapter.handoff_error(snapshot.poses)
            for snapshot in self.snapshot_objects
        ]
        valid_handoffs = [
            index
            for index, error in enumerate(handoff_errors)
            if error <= self.adapter.handoff_tolerance_m
        ]
        if not valid_handoffs or valid_handoffs[0] >= len(self.snapshot_objects) - 1:
            raise RuntimeError(
                f"RFCL demo {self.demo_id} has no valid handoff before terminal: "
                f"min_xy_error={min(handoff_errors):.6f} m"
            )
        policy_start = int(valid_handoffs[0])
        return {
            "demo_id": self.demo_id,
            "seed": self.seed,
            "snapshot_ids": [snapshot["snapshot_id"] for snapshot in self.snapshots],
            "state_count": len(self.snapshots),
            "transition_count": len(self.snapshots) - 1,
            "terminal_snapshot_id": self.snapshots[-1]["snapshot_id"],
            "terminal_sim_step": self.snapshots[-1]["sim_step"],
            "policy_start_state_index": policy_start,
            "handoff_error_m": float(handoff_errors[policy_start]),
            "handoff_tolerance_m": float(self.adapter.handoff_tolerance_m),
        }


def _write_manifest(
    root: Path,
    *,
    args: argparse.Namespace,
    adapter: RFCLTaskAdapter,
    action_scale: np.ndarray,
    demos: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "schema": SNAPSHOT_DATASET_SCHEMA,
        "task": args.task_name,
        "task_config": args.task_config,
        "stride": int(args.stride),
        "step_limit": int(args.step_limit),
        "action_repeat": int(args.stride),
        "action_mode": str(args.action_mode),
        "action_dim": int(action_scale.shape[0]),
        "action_scale": action_scale.tolist(),
        "demo_plan": str(args.demo_plan),
        "profile_plan_id": (
            BALANCED_40_PLAN_ID
            if args.demo_plan == "insert_usb_balanced40"
            else None
        ),
        "adapter": adapter.manifest_metadata(),
        "reward": "success ? 1 : 0",
        "terminal_definition": "first_frame_satisfying_adapter.check_success",
        "gripper_control": adapter.gripper_mode,
        "demos": demos,
        "snapshots": snapshots,
        "failed_demos": failures or [],
    }
    (root / SNAPSHOT_MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    jobs = _snapshot_jobs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / SNAPSHOT_MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing RFCL dataset: {manifest_path}"
        )
    action_scale = _action_scale(args)
    print(f"[rfcl-snapshot] output={args.output}", flush=True)
    print(
        f"[rfcl-snapshot] demo_plan={args.demo_plan} jobs={len(jobs)} "
        f"stride={args.stride}",
        flush=True,
    )
    print(f"[rfcl-snapshot] action_scale={action_scale.tolist()}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    demos: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    try:
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.output,
            video_frequency=0,
            step_limit=int(args.step_limit),
            mode="eval_test",
            save_pre_move=False,
            task_variant=(
                "rfcl" if args.demo_plan == "insert_usb_balanced40" else None
            ),
            device=args.device,
        )
        task.cfg.save_frequency = 0
        task.cfg.video_frequency = 0
        adapter = create_rfcl_task_adapter(task_name=args.task_name)

        for job in jobs:
            demo_index = int(job["demo_index"])
            demo_id = str(job["demo_id"])
            profile = job["profile"]
            profile_payload = None if profile is None else profile.to_dict()
            completed = False
            for attempt, seed in enumerate(job["attempt_seeds"]):
                seed = int(seed)
                print(
                    f"\n[rfcl-snapshot] demo={demo_id} attempt={attempt + 1}/"
                    f"{len(job['attempt_seeds'])} seed={seed} reset",
                    flush=True,
                )
                if args.demo_plan == "insert_usb_balanced40":
                    task.set_rfcl_demo_profile(profile)
                recorder = None
                try:
                    task.reset(seed=seed)
                    recorder = SnapshotRecorder(
                        task,
                        adapter,
                        args.output,
                        demo_index=demo_index,
                        seed=seed,
                        stride=int(args.stride),
                        demo_id=None if profile is None else demo_id,
                    )
                    recorder.install()
                    try:
                        task.play_once()
                    finally:
                        recorder.uninstall()
                    if not recorder.terminal_success:
                        diagnostics = adapter.success_diagnostics(task)
                        raise RuntimeError(
                            f"Motion Plan demo {demo_id} seed {seed} failed "
                            f"before first success: {diagnostics}"
                        )
                    demo = recorder.finish()
                    demo["attempt"] = int(attempt)
                    if profile_payload is not None:
                        demo["profile"] = profile_payload
                    demos.append(demo)
                    snapshots.update(
                        {
                            item["snapshot_id"]: item
                            for item in recorder.snapshots
                        }
                    )
                    completed = True
                    _write_manifest(
                        args.output,
                        args=args,
                        adapter=adapter,
                        action_scale=action_scale,
                        demos=demos,
                        snapshots=snapshots,
                        failures=failures,
                    )
                    print(
                        f"[rfcl-snapshot] demo={demo_id} seed={seed} success; "
                        f"states={demo['state_count']} "
                        f"terminal_step={demo['terminal_sim_step']}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    diagnostics = None
                    try:
                        diagnostics = adapter.success_diagnostics(task)
                    except Exception:
                        pass
                    failure = {
                        "demo_id": demo_id,
                        "profile": profile_payload,
                        "attempt": int(attempt),
                        "seed": seed,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "diagnostics": diagnostics,
                    }
                    failures.append(failure)
                    print(
                        f"[rfcl-snapshot] demo={demo_id} seed={seed} failed: "
                        f"{failure['message']}",
                        flush=True,
                    )
                    _write_manifest(
                        args.output,
                        args=args,
                        adapter=adapter,
                        action_scale=action_scale,
                        demos=demos,
                        snapshots=snapshots,
                        failures=failures,
                    )
                    if profile is None and not args.skip_failed_demos:
                        raise
            if not completed and not args.skip_failed_demos:
                raise RuntimeError(
                    f"RFCL demo {demo_id} failed after "
                    f"{len(job['attempt_seeds'])} attempts"
                )
        if not demos:
            raise RuntimeError("No successful RFCL demos were generated")
    except BaseException as exc:
        error_path = args.output / "rfcl_generation_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed_demos": demos,
                    "failed_demos": failures,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if task is not None:
            task.close()
        simulation_app.close()

    print(f"[rfcl-snapshot] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
