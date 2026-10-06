"""Adapter-driven privileged-state RFCL environment.

The environment is intentionally independent of the RGB/tactile BC actor.
It restores a complete simulator snapshot, exposes a 46-dimensional state,
and accepts seven normalized arm delta-qpos actions.  The gripper remains at
the closed qpos captured in the demo.  This is the narrow interface used by
the first RFCL pilot; observation re-recording happens only after a policy has
generated successful trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from envs.utils.transforms import Pose
from uipc import view

from policy.RL.action import (
    TARGET_QPOS_VELOCITY_ACTION_DIM,
    TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM,
    action_to_target_qpos_velocity_force,
    action_to_target_qpos,
    action_to_target_qpos_velocity,
    clip_action,
)
from policy.RL.rfcl import ReverseCurriculum
from policy.RL.rfcl_snapshot import (
    RFCLSnapshotDataset,
    prepare_snapshot_for_policy,
    read_robot_physics_state,
)


class RFCLPrivilegedEnv(gym.Env):
    """Single-environment Gym adapter around the existing Isaac task."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: Any,
        snapshot_root: str | Path,
        *,
        action_scale: np.ndarray | None = None,
        action_repeat: int = 2,
        reverse_step_size: int = 2,
        per_demo_buffer_size: int = 3,
        geometric_p: float = 0.5,
        demo_horizon_to_max_steps_ratio: float = 1.25,
        minimum_episode_horizon: int = 16,
        action_scale_margin: float = 1.05,
        snapshot_sync_steps: int = 1,
        snapshot_sync_constraint_strength: float = 1.0e5,
        snapshot_rerestore_after_sync: bool = True,
        seed: int = 0,
        action_mode: str | None = None,
    ) -> None:
        super().__init__()
        if int(action_repeat) < 1:
            raise ValueError("action_repeat must be positive")
        if int(snapshot_sync_steps) < 0:
            raise ValueError("snapshot_sync_steps must be non-negative")
        if not np.isfinite(float(action_scale_margin)) or float(action_scale_margin) < 1.0:
            raise ValueError("action_scale_margin must be finite and at least 1")
        if (
            not np.isfinite(snapshot_sync_constraint_strength)
            or float(snapshot_sync_constraint_strength) <= 0.0
        ):
            raise ValueError(
                "snapshot_sync_constraint_strength must be finite and positive"
            )
        self.task = task
        self.dataset = RFCLSnapshotDataset(snapshot_root)
        self.adapter = self.dataset.adapter
        self.action_mode = str(
            self.dataset.manifest.get("action_mode", "qpos_delta")
            if action_mode is None
            else action_mode
        )
        self.action_scale_margin = float(action_scale_margin)
        if self.action_mode not in (
            "qpos_delta", "target_pos_vel", "target_pos_vel_force"
        ):
            raise ValueError(f"Unsupported RFCL action_mode: {self.action_mode!r}")
        if self.action_mode in ("target_pos_vel", "target_pos_vel_force") and int(action_repeat) != 1:
            raise ValueError("target_pos_vel RFCL actions require action_repeat=1")
        inferred_scale = (
            self.dataset.infer_target_pos_vel_action_scale(
                margin=self.action_scale_margin
            )
            if self.action_mode in ("target_pos_vel", "target_pos_vel_force")
            else self.dataset.infer_action_scale(margin=self.action_scale_margin)
        )
        if self.action_mode == "target_pos_vel_force":
            inferred_scale = np.concatenate((inferred_scale, np.ones(1, dtype=np.float32)))
        self.action_scale = np.asarray(
            inferred_scale if action_scale is None else action_scale,
            dtype=np.float32,
        )
        expected_action_dim = {
            "qpos_delta": 7,
            "target_pos_vel": TARGET_QPOS_VELOCITY_ACTION_DIM,
            "target_pos_vel_force": TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM,
        }[self.action_mode]
        if self.action_scale.shape != (expected_action_dim,):
            raise ValueError(
                f"action_scale must have shape ({expected_action_dim},), got {self.action_scale.shape}"
            )
        if not np.isfinite(self.action_scale).all() or np.any(self.action_scale <= 0):
            raise ValueError("action_scale must be finite and positive")
        self.action_repeat = int(action_repeat)
        self.snapshot_sync_steps = int(snapshot_sync_steps)
        self.snapshot_sync_constraint_strength = float(
            snapshot_sync_constraint_strength
        )
        self.snapshot_rerestore_after_sync = bool(snapshot_rerestore_after_sync)
        self.curriculum = ReverseCurriculum(
            self.dataset.to_demo_trajectories(
                self.action_scale, action_mode=self.action_mode
            ),
            reverse_step_size=reverse_step_size,
            per_demo_buffer_size=per_demo_buffer_size,
            geometric_p=geometric_p,
            demo_horizon_to_max_steps_ratio=demo_horizon_to_max_steps_ratio,
            minimum_episode_horizon=minimum_episode_horizon,
            seed=seed,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.adapter.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(expected_action_dim,),
            dtype=np.float32,
        )
        self._demo_index: int | None = None
        self._state_index: int | None = None
        self._state: np.ndarray | None = None
        self._episode_horizon = 0
        self._policy_steps = 0
        self._done = False
        self._last_transition: tuple[np.ndarray, np.ndarray, float, np.ndarray, bool] | None = None

        # RFCL is a policy-only phase.  No observation/video files should be
        # written while the simulator is being used as a data generator.
        if hasattr(self.task, "cfg"):
            self.task.cfg.save_frequency = 0
            self.task.cfg.video_frequency = 0

    def _live_state(self) -> np.ndarray:
        physics_state = read_robot_physics_state(self.task)
        return self.adapter.build_privileged_state(
            physics_state=physics_state,
            entities=self.adapter.capture_entities(self.task),
        )

    def _stage_snapshot_actor_poses(self, snapshot) -> dict[str, np.ndarray]:
        """Apply saved affine poses through the actor animation path.

        ``world.recover`` restores the UIPC frame, but affine actors expose the
        recovered transform only after their animator callback runs.  Staging
        the saved pose for the synchronization step makes that callback use the
        checkpoint pose; held-object constraints configured by the adapter are
        released immediately afterwards so RFCL retains physical control.
        """
        saved_strengths: dict[str, np.ndarray] = {}
        for binding in self.adapter.snapshot_actor_bindings():
            actor_name = binding.actor_name
            pose_name = binding.pose_key
            actor = self.task._actor_manager.actors.get(actor_name)
            pose = snapshot.poses.get(pose_name)
            if actor is None or pose is None:
                continue
            actor.set_pose(Pose.from_list(np.asarray(pose, dtype=np.float64).tolist()))
            if binding.release_after_sync:
                strength = actor.uipc_meshes[0].instances().find(
                    "strength_ratio"
                )
                if strength is None:
                    raise RuntimeError(
                        f"RFCL actor {actor_name!r} is missing constraint strength_ratio"
                    )
                strength_values = view(strength)
                saved_strengths[actor_name] = np.asarray(
                    strength_values
                ).copy()
                strength_values[...] = self.snapshot_sync_constraint_strength
        self.task._actor_manager.update(dt=0.0)
        return saved_strengths

    def _restore_actor_constraint_strengths(
        self, saved_strengths: dict[str, np.ndarray]
    ) -> None:
        for actor_name, saved in saved_strengths.items():
            actor = self.task._actor_manager.actors[actor_name]
            strength = actor.uipc_meshes[0].instances().find("strength_ratio")
            if strength is None:
                raise RuntimeError(
                    f"RFCL actor {actor_name!r} lost constraint strength_ratio"
                )
            view(strength)[...] = saved

    def _reset_task_for_snapshot(self, demo_index: int) -> None:
        demo = self.dataset.demos[demo_index]
        seed = demo.get("seed")
        self.task.reset(seed=None if seed is None else int(seed))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        if "demo_index" in options or "state_index" in options:
            if "demo_index" not in options or "state_index" not in options:
                raise ValueError(
                    "reset options must provide both demo_index and state_index"
                )
            self._demo_index = int(options["demo_index"])
            self._state_index = int(options["state_index"])
            self.dataset.snapshot(self._demo_index, self._state_index)
        else:
            self._demo_index, self._state_index = self.curriculum.sample_checkpoint()
        # A persistent UIPC snapshot cannot serialize PhysX contact history
        # between the gripper and a held object. Diagnostic/training callers
        # may bootstrap the grasp in the current process, then recover only
        # the requested suffix while preserving that contact manifold.
        if not bool(options.pop("skip_task_reset", False)):
            self._reset_task_for_snapshot(self._demo_index)
        snapshot = self.dataset.snapshot(self._demo_index, self._state_index)
        prepare_snapshot_for_policy(self.task, self.adapter, snapshot)
        saved_strengths = {}
        if self.snapshot_sync_steps:
            saved_strengths = self._stage_snapshot_actor_poses(snapshot)
        for _ in range(self.snapshot_sync_steps):
            # Actor constraints and UIPC attachment callbacks are applied by
            # the simulator step, not by ``world.retrieve`` alone.  This
            # synchronization step is an action-free settling step; it keeps
            # the policy state at the requested checkpoint while making the
            # first real action physically consistent with it.
            self.task._step(is_save=False)
        if self.snapshot_sync_steps:
            self._restore_actor_constraint_strengths(saved_strengths)
            if self.snapshot_rerestore_after_sync:
                # The settling step initializes PhysX articulation and
                # attachment callbacks but perturbs contact-rich UIPC states.
                # Recover once more so the policy still observes the exact
                # checkpoint with those external callbacks now warm.
                prepare_snapshot_for_policy(self.task, self.adapter, snapshot)
            else:
                for binding in self.adapter.snapshot_actor_bindings():
                    if not binding.release_after_sync:
                        continue
                    actor = self.task._actor_manager.actors.get(
                        binding.actor_name
                    )
                    if actor is not None:
                        actor.remove_animate(force=True)
        self._state = self._live_state()
        self._episode_horizon = self.curriculum.episode_horizon(
            self._demo_index, self._state_index
        )
        self._policy_steps = 0
        self._done = False
        self._last_transition = None
        return self._state.copy(), {
            "demo_index": int(self._demo_index),
            "demo_id": str(snapshot.demo_id),
            "state_index": int(self._state_index),
            "raw_state_index": int(snapshot.state_index),
            "episode_horizon": int(self._episode_horizon),
            "handoff": self.dataset.handoff_diagnostics(self._demo_index),
            "curriculum": self.curriculum.state(),
        }

    def step(self, action: np.ndarray):
        if self._state is None or self._demo_index is None or self._state_index is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            raise RuntimeError("step() called after episode termination")
        action_array = np.asarray(action, dtype=np.float32)
        if self.action_mode in ("target_pos_vel", "target_pos_vel_force"):
            if action_array.shape != (TARGET_QPOS_VELOCITY_ACTION_DIM,):
                expected_dim = self.action_space.shape[0]
                if action_array.shape != (expected_dim,):
                    raise ValueError(
                        f"{self.action_mode} action must have shape ({expected_dim},), got {action_array.shape}"
                    )
            if not np.isfinite(action_array).all():
                raise ValueError("action contains NaN or infinite values")
            action = np.clip(action_array, -1.0, 1.0).astype(np.float32)
        else:
            action = clip_action(action_array)
        previous_state = self._state.copy()
        manager = self.task._robot_manager
        current_qpos = read_robot_physics_state(self.task)["joint_pos"]
        current_arm = current_qpos[:7].astype(np.float32, copy=False)
        velocity = None
        force = True
        target = None
        if self.action_mode == "target_pos_vel":
            target_arm, velocity = action_to_target_qpos_velocity(
                current_arm,
                action,
                self.action_scale[:7],
                self.action_scale[7:],
            )
        elif self.action_mode == "target_pos_vel_force":
            target_arm, velocity, force = action_to_target_qpos_velocity_force(
                current_arm,
                action,
                self.action_scale[:7],
                self.action_scale[7:14],
            )
        else:
            target_arm = action_to_target_qpos(
                current_arm,
                action,
                self.action_scale,
            )
        if force or self.action_mode != "target_pos_vel_force":
            gripper_qpos = float(current_qpos[int(manager._gripper_ids[0])])
            target = np.concatenate(
                (target_arm, np.asarray([gripper_qpos], dtype=np.float32))
            )
            _, _, task_terminated, task_truncated, task_info = self.task.env_step(
                target,
                action_type="qpos",
                force=True,
                action_repeat=self.action_repeat,
                joint_velocity=velocity,
            )
        else:
            # A recorded ``force_position_write=False`` transition is a real
            # Motion Plan delay step: the controller advances one simulator
            # frame without replacing the previously written arm/gripper
            # targets.  Calling ``env_step`` here would invoke set_arm and
            # set_gripper and therefore change the command stream.
            task_terminated, task_truncated, task_info = self._step_without_command()
        self._policy_steps += 1
        next_state = self._live_state()
        success = bool(task_info.get("success", False))
        irrecoverable_failure = (
            None if success else self.adapter.irrecoverable_failure(self.task)
        )
        reward = 1.0 if success else 0.0
        truncated = bool(task_truncated)
        if not success and self._policy_steps >= self._episode_horizon:
            truncated = True
        terminated = bool(task_terminated or success or irrecoverable_failure)
        self._done = bool(terminated or truncated)
        if self._done:
            self.curriculum.record_result(
                self._demo_index,
                self._state_index,
                success=success,
            )
        self._last_transition = (
            previous_state,
            action.copy(),
            float(reward),
            next_state.copy(),
            bool(terminated),
        )
        self._state = next_state
        info = dict(task_info)
        info.update(
            {
                "success": success,
                "reward_definition": "success ? 1 : 0",
                "irrecoverable_failure": irrecoverable_failure,
                "replay_eligible": irrecoverable_failure is None,
                "demo_index": int(self._demo_index),
                "state_index": int(self._state_index),
                "raw_state_index": int(
                    self.dataset.local_to_raw_state_index(
                        self._demo_index, self._state_index
                    )
                ),
                "policy_step": int(self._policy_steps),
                "episode_horizon": int(self._episode_horizon),
                "target_qpos": target,
                "action_scale": self.action_scale.copy(),
                "curriculum": self.curriculum.state(),
            }
        )
        return next_state.copy(), reward, terminated, truncated, info

    def _step_without_command(self) -> tuple[bool, bool, dict[str, Any]]:
        """Advance one RFCL frame while preserving the active low-level target."""
        task = self.task
        if task.phase_id != task.PHASE_POLICY:
            raise RuntimeError("RFCL no-command step requires the POLICY phase")

        if task.take_action_cnt >= int(task.cfg.step_lim) or task.eval_success:
            success = bool(task.eval_success)
            task_terminated = success
            task_truncated = not success
            info = {
                "exec_success": True,
                "success": success,
                "rl_early_stop": False,
                "task_early_stop": False,
                "take_action_count": int(task.take_action_cnt),
            }
            return task_terminated, task_truncated, info

        task.take_action_cnt += 1
        task._step(is_save=False)
        success = self.adapter.check_success(task)
        if success:
            task.eval_success = True

        metrics = task.get_rl_metrics()
        rl_early_stop = bool(task.check_rl_early_stop(metrics))
        task_early_stop = bool(task.check_early_stop())
        task_terminated = success
        task_truncated = bool(
            task.take_action_cnt >= int(task.cfg.step_lim)
            or rl_early_stop
            or task_early_stop
        )
        if task_terminated:
            task._set_phase(task.PHASE_TERMINAL, terminal_reason="success")
        elif task_truncated:
            if task.take_action_cnt >= int(task.cfg.step_lim):
                reason = "step_limit"
            elif rl_early_stop:
                reason = "rl_early_stop"
            else:
                reason = "task_early_stop"
            task._set_phase(task.PHASE_TERMINAL, terminal_reason=reason)
        task.policy_step_count += 1
        return task_terminated, task_truncated, {
            "exec_success": True,
            "success": success,
            "rl_early_stop": rl_early_stop,
            "task_early_stop": task_early_stop,
            "take_action_count": int(task.take_action_cnt),
            "metrics": metrics,
            "force_position_write": False,
        }

    def pop_transition(self):
        transition = self._last_transition
        self._last_transition = None
        return transition

    def curriculum_state(self) -> dict[str, object]:
        return self.curriculum.state()

    def close(self) -> None:
        if hasattr(self.task, "close"):
            self.task.close()
