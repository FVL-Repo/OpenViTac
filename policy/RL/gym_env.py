from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, Mapping

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from policy.RL.action import action_to_target_qpos, clip_action
from policy.RL.checkpoint import (
    extract_action_scale_from_bc_checkpoint,
    load_bc_checkpoint,
    restore_actor_from_bc_checkpoint,
)


RewardMode = Literal["sparse_success", "task"]
ControlMode = Literal["bc", "direct"]
HandoffMode = Literal["auto", "none", "insert_usb_collect"]
InsertUsbHandoffDistribution = Literal[
    "legacy",
    "coarse_preinsert",
    "direct",
    "precontact",
    "diverse_v1",
    "diverse_mild",
    "diverse_tiny",
    "curriculum_v1",
]

INSERT_USB_HANDOFF_DISTRIBUTIONS = (
    "legacy",
    "coarse_preinsert",
    "direct",
    "precontact",
    "diverse_v1",
    "diverse_mild",
    "diverse_tiny",
    "curriculum_v1",
)

INSERT_USB_CURRICULUM_STAGE_NAMES = (
    "bootstrap_xy_z_offset",
    "expand_xy_z_offset",
    "wide_xy_z_offset",
    "large_xy_z_offset",
)
DEFAULT_INSERT_USB_CURRICULUM_SUCCESS_THRESHOLDS = (50, 100, 150)
DEFAULT_INSERT_USB_COARSE_Z_JITTER = 0.002


def _to_hwc_uint8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        image = value.detach().cpu().numpy()
    else:
        image = np.asarray(value)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (
        1,
        3,
        4,
    ):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.rint(np.clip(image, 0, 255)).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class TactileControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        task,
        bc_checkpoint: str | Path,
        *,
        image_size: int | None = None,
        action_repeat: int = 2,
        bc_action_gain: float = 1.0,
        control_mode: ControlMode = "direct",
        reward_mode: RewardMode = "sparse_success",
        handoff_mode: HandoffMode = "auto",
        force_control: bool = True,
        collect_successful_episodes: bool = False,
        clean_finished_episodes: bool = False,
        collection_metadata: Mapping[str, object] | None = None,
        debug_logging: bool = False,
        debug_step_log_frequency: int = 0,
        insert_usb_handoff_distribution: InsertUsbHandoffDistribution = "legacy",
        insert_usb_coarse_z_jitter: float = DEFAULT_INSERT_USB_COARSE_Z_JITTER,
        insert_usb_curriculum_success_thresholds: tuple[int, int, int]
        | None = None,
        insert_usb_xy_quit_threshold: float | None = None,
        zero_qpos: bool = False,
        seed: int = 0,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__()
        if control_mode not in ("bc", "direct"):
            raise ValueError(f"Unsupported control_mode: {control_mode!r}")
        if reward_mode not in ("sparse_success", "task"):
            raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")
        if handoff_mode not in ("auto", "none", "insert_usb_collect"):
            raise ValueError(f"Unsupported handoff_mode: {handoff_mode!r}")
        if insert_usb_handoff_distribution not in INSERT_USB_HANDOFF_DISTRIBUTIONS:
            raise ValueError(
                "Unsupported insert_usb_handoff_distribution: "
                f"{insert_usb_handoff_distribution!r}"
            )
        if (
            insert_usb_xy_quit_threshold is not None
            and float(insert_usb_xy_quit_threshold) <= 0.0
        ):
            raise ValueError("insert_usb_xy_quit_threshold must be positive")
        if (
            not np.isfinite(insert_usb_coarse_z_jitter)
            or float(insert_usb_coarse_z_jitter) < 0.0
        ):
            raise ValueError(
                "insert_usb_coarse_z_jitter must be finite and non-negative"
            )
        if insert_usb_curriculum_success_thresholds is None:
            curriculum_thresholds = DEFAULT_INSERT_USB_CURRICULUM_SUCCESS_THRESHOLDS
        else:
            curriculum_thresholds = tuple(
                int(value) for value in insert_usb_curriculum_success_thresholds
            )
        if len(curriculum_thresholds) != 3:
            raise ValueError(
                "insert_usb_curriculum_success_thresholds must contain "
                "exactly three success counts"
            )
        if any(value < 0 for value in curriculum_thresholds):
            raise ValueError(
                "insert_usb_curriculum_success_thresholds must be non-negative"
            )
        if tuple(sorted(curriculum_thresholds)) != curriculum_thresholds:
            raise ValueError(
                "insert_usb_curriculum_success_thresholds must be non-decreasing"
            )
        if int(action_repeat) < 1:
            raise ValueError("action_repeat must be at least 1")
        if not np.isfinite(bc_action_gain) or float(bc_action_gain) <= 0.0:
            raise ValueError("bc_action_gain must be finite and positive")
        if int(debug_step_log_frequency) < 0:
            raise ValueError("debug_step_log_frequency must be non-negative")

        self.task = task
        self.checkpoint_path = Path(bc_checkpoint)
        self.device = torch.device(device)
        self.checkpoint = load_bc_checkpoint(self.checkpoint_path, map_location="cpu")
        self.actor = restore_actor_from_bc_checkpoint(
            self.checkpoint,
            device=self.device,
        )
        self.actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

        observation_contract = self.checkpoint["observation_contract"]
        self.image_size = int(image_size or observation_contract["image_size"])
        self.camera_keys = tuple(observation_contract["camera_keys"])
        self.tactile_keys = tuple(observation_contract["tactile_keys"])
        self.action_scale = (
            extract_action_scale_from_bc_checkpoint(
                self.checkpoint,
                expected_action_dim=self.actor.action_dim,
                device="cpu",
            )
            .numpy()
            .astype(np.float32)
        )
        self.action_repeat = int(action_repeat)
        self.bc_action_gain = float(bc_action_gain)
        self.control_mode = control_mode
        self.reward_mode = reward_mode
        self.handoff_mode = handoff_mode
        self.force_control = bool(force_control)
        self.collect_successful_episodes = bool(collect_successful_episodes)
        self.clean_finished_episodes = bool(
            clean_finished_episodes or collect_successful_episodes
        )
        self.collection_metadata = dict(collection_metadata or {})
        self.debug_logging = bool(debug_logging)
        self.debug_step_log_frequency = int(debug_step_log_frequency)
        self.insert_usb_handoff_distribution = insert_usb_handoff_distribution
        self.insert_usb_coarse_z_jitter = float(insert_usb_coarse_z_jitter)
        self.insert_usb_curriculum_success_thresholds = curriculum_thresholds
        self.insert_usb_xy_quit_threshold = (
            None
            if insert_usb_xy_quit_threshold is None
            else float(insert_usb_xy_quit_threshold)
        )
        self.zero_qpos = bool(zero_qpos)
        self.collection_attempts = 0
        self.collection_failures = 0
        self.collected_successes = 0
        self.collection_mean_steps = 0.0
        self.next_seed = int(seed)

        observation_spaces: dict[str, spaces.Space] = {
            "qpos": spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.actor.encoder.qpos_dim,),
                dtype=np.float32,
            )
        }
        for key in self.camera_keys + self.tactile_keys:
            observation_spaces[key] = spaces.Box(
                0,
                255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.actor.action_dim,),
            dtype=np.float32,
        )

        self.last_observation: dict[str, np.ndarray] | None = None
        self.execution_qpos: np.ndarray | None = None
        self.gripper_hold_qpos = 0.0
        self.policy_step = 0

    @staticmethod
    def _sensor(raw_observation: dict, side: str) -> dict:
        tactile = raw_observation["tactile"]
        for key in (f"{side}_tactile", f"{side}_gsmini"):
            if key in tactile:
                return tactile[key]
        raise KeyError(f"Missing {side} tactile observation: {list(tactile)}")

    def _image(self, value: torch.Tensor | np.ndarray) -> np.ndarray:
        image = _to_hwc_uint8(value)
        # Training JPEGs are decoded by OpenCV as BGR. Online observations are
        # RGB, so keep the same channel contract the BC rollout used.
        image = image[..., [2, 1, 0]]
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.uint8)

    def encode_observation(self, raw_observation: dict) -> dict[str, np.ndarray]:
        joint = raw_observation["embodiment"]["joint"]
        if torch.is_tensor(joint):
            joint = joint.detach().cpu().numpy()
        encoded: dict[str, np.ndarray] = {
            "qpos": np.asarray(
                np.zeros(self.actor.encoder.qpos_dim, dtype=np.float32)
                if self.zero_qpos
                else joint[: self.actor.encoder.qpos_dim],
                dtype=np.float32,
            ).copy()
        }
        if "cam_high" in self.camera_keys:
            encoded["cam_high"] = self._image(
                raw_observation["observation"]["head"]["rgb"]
            )
        if "cam_wrist" in self.camera_keys:
            encoded["cam_wrist"] = self._image(
                raw_observation["observation"]["wrist"]["rgb"]
            )
        if "tac_left" in self.tactile_keys:
            encoded["tac_left"] = self._image(
                self._sensor(raw_observation, "left")["rgb_marker"]
            )
        if "tac_right" in self.tactile_keys:
            encoded["tac_right"] = self._image(
                self._sensor(raw_observation, "right")["rgb_marker"]
            )
        return encoded

    def _actor_batch(self, observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in observation.items()
        }

    def bc_normalized_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        with torch.inference_mode():
            action = self.actor.deterministic_action(self._actor_batch(observation))[0]
        return action.detach().cpu().numpy().astype(np.float32)

    def _should_use_insert_usb_handoff(self) -> bool:
        if self.handoff_mode == "none":
            return False
        if self.handoff_mode == "insert_usb_collect":
            return True
        module_name = self.task.__class__.__module__
        return module_name.endswith("insert_USB") and all(
            hasattr(self.task, name)
            for name in (
                "_prepare_usb_standard",
                "_update_insert_reference_poses",
                "opening_pose",
                "atom",
                "prism",
            )
        )

    def _rng_uniform(self, low: float, high: float) -> float:
        rng = getattr(self.task, "rng", None)
        if rng is None:
            rng = self.np_random
        return float(rng.uniform(float(low), float(high)))

    def _sample_xy_offset(self, min_radius: float, max_radius: float) -> np.ndarray:
        angle = self._rng_uniform(-np.pi, np.pi)
        radius = self._rng_uniform(min_radius, max_radius)
        return np.array(
            [radius * np.cos(angle), radius * np.sin(angle)],
            dtype=np.float64,
        )

    def _sample_signed_magnitude(self, min_abs: float, max_abs: float) -> float:
        sign = -1.0 if self._rng_uniform(0.0, 1.0) < 0.5 else 1.0
        return sign * self._rng_uniform(min_abs, max_abs)

    def _insert_usb_curriculum_stage(self) -> tuple[int, str]:
        successes = int(self.collected_successes)
        stage_index = sum(
            successes >= threshold
            for threshold in self.insert_usb_curriculum_success_thresholds
        )
        stage_index = min(stage_index, len(INSERT_USB_CURRICULUM_STAGE_NAMES) - 1)
        return stage_index, INSERT_USB_CURRICULUM_STAGE_NAMES[stage_index]

    def _sample_insert_usb_handoff(self, default_clearance: float) -> dict[str, object]:
        distribution = self.insert_usb_handoff_distribution
        profile = distribution
        sample: float | None = None
        curriculum_stage_index: int | None = None
        curriculum_stage_name: str | None = None
        if distribution == "diverse_v1":
            sample = self._rng_uniform(0.0, 1.0)
            if sample < 0.20:
                profile = "direct"
            else:
                profile = "precontact"
        elif distribution == "diverse_mild":
            sample = self._rng_uniform(0.0, 1.0)
            if sample < 0.20:
                profile = "direct"
            else:
                profile = "precontact"
        elif distribution == "diverse_tiny":
            sample = self._rng_uniform(0.0, 1.0)
            if sample < 0.80:
                profile = "direct"
            else:
                profile = "precontact"
        elif distribution == "curriculum_v1":
            curriculum_stage_index, curriculum_stage_name = (
                self._insert_usb_curriculum_stage()
            )
            sample = self._rng_uniform(0.0, 1.0)
            if curriculum_stage_index == 0:
                if sample < 0.80:
                    profile = "direct"
                else:
                    profile = "precontact"
            elif curriculum_stage_index == 1:
                if sample < 0.10:
                    profile = "direct"
                else:
                    profile = "precontact"
            elif curriculum_stage_index == 2:
                if sample < 0.10:
                    profile = "direct"
                else:
                    profile = "precontact"
            else:
                if sample < 0.05:
                    profile = "direct"
                else:
                    profile = "precontact"

        xy_offset = np.zeros(2, dtype=np.float64)
        rpy_offset = np.zeros(3, dtype=np.float64)
        z_clearance = float(default_clearance)

        if profile == "legacy":
            contact_goal = "legacy"
        elif profile == "direct":
            xy_offset = self._sample_xy_offset(0.0002, 0.0008)
            z_clearance = self._rng_uniform(0.0010, 0.0030)
            contact_goal = "direct_xy_z_offset"
        elif profile == "precontact":
            if distribution == "curriculum_v1":
                if curriculum_stage_index == 0:
                    xy_offset = self._sample_xy_offset(0.0008, 0.0015)
                    z_clearance = self._rng_uniform(0.0010, 0.0030)
                    contact_goal = "curriculum_tiny_xy_z_offset"
                elif curriculum_stage_index == 1:
                    xy_offset = self._sample_xy_offset(0.0010, 0.0025)
                    z_clearance = self._rng_uniform(0.0015, 0.0040)
                    contact_goal = "curriculum_expanded_xy_z_offset"
                elif curriculum_stage_index == 2:
                    xy_offset = self._sample_xy_offset(0.0015, 0.0035)
                    z_clearance = self._rng_uniform(0.0015, 0.0050)
                    contact_goal = "curriculum_wide_xy_z_offset"
                else:
                    xy_offset = self._sample_xy_offset(0.0020, 0.0045)
                    z_clearance = self._rng_uniform(0.0020, 0.0060)
                    contact_goal = "curriculum_large_xy_z_offset"
            elif distribution == "diverse_tiny":
                xy_offset = self._sample_xy_offset(0.0005, 0.0015)
                z_clearance = self._rng_uniform(0.0010, 0.0030)
                contact_goal = "tiny_xy_z_offset"
            elif distribution == "diverse_mild":
                xy_offset = self._sample_xy_offset(0.0008, 0.0030)
                z_clearance = self._rng_uniform(0.0015, 0.0040)
                contact_goal = "mild_xy_z_offset"
            elif distribution == "diverse_v1":
                xy_offset = self._sample_xy_offset(0.0010, 0.0040)
                z_clearance = self._rng_uniform(0.0020, 0.0050)
                contact_goal = "diverse_xy_z_offset"
            else:
                xy_offset = self._sample_xy_offset(0.0010, 0.0040)
                z_clearance = self._rng_uniform(0.0020, 0.0050)
                contact_goal = "precontact_xy_z_offset"
        else:
            raise ValueError(f"Unsupported sampled Insert USB profile: {profile!r}")

        return {
            "distribution": distribution,
            "profile": profile,
            "selection_sample": sample,
            "curriculum_stage_index": curriculum_stage_index,
            "curriculum_stage_name": curriculum_stage_name,
            "curriculum_success_thresholds": (
                self.insert_usb_curriculum_success_thresholds
                if distribution == "curriculum_v1"
                else None
            ),
            "curriculum_collected_successes": (
                int(self.collected_successes)
                if distribution == "curriculum_v1"
                else None
            ),
            "xy_offset": xy_offset,
            "z_clearance": float(z_clearance),
            "rpy_offset": rpy_offset,
            "contact_goal": contact_goal,
        }

    def _prepare_insert_usb_collect_handoff(self) -> None:
        module = sys.modules[self.task.__class__.__module__]
        clearance = float(getattr(module, "PLAY_PRE_INSERT_CLEARANCE", 0.0))
        self.task._set_phase(self.task.PHASE_PRE_MOVE)
        self.task._prepare_usb_standard()
        if not self.task.plan_success:
            raise RuntimeError("Scripted USB preparation failed before RL handoff")

        self.task._update_insert_reference_poses()
        if self.insert_usb_handoff_distribution == "coarse_preinsert":
            z_jitter = self.insert_usb_coarse_z_jitter
            z_offset = (
                self._rng_uniform(-z_jitter, z_jitter)
                if z_jitter > 0.0
                else 0.0
            )
            if z_jitter > 0.0:
                moved = self.task.move(
                    self.task.atom.move_by_displacement(
                        z=z_offset,
                        xyz_coord="world",
                    ),
                    tag="rl_handoff_z_jitter",
                    time_dilation_factor=0.5,
                )
                if not moved or not self.task.plan_success:
                    raise RuntimeError("Scripted USB Z-jitter handoff failed")
            handoff_pose = self.task.prism.get_pose()
            relative_position = handoff_pose.p - self.task.opening_pose.p
            xy_offset = np.asarray(relative_position[:2], dtype=np.float64)
            rpy_offset = np.zeros(3, dtype=np.float64)
            z_clearance = float(relative_position[2])
            handoff_sample = {
                "distribution": "coarse_preinsert",
                "profile": (
                    "coarse_preinsert_xyz_offset"
                    if z_jitter > 0.0
                    else "coarse_preinsert_xy_offset"
                ),
                "selection_sample": None,
                "curriculum_stage_index": None,
                "curriculum_stage_name": None,
                "curriculum_success_thresholds": None,
                "curriculum_collected_successes": None,
                "xy_offset": xy_offset,
                "z_clearance": z_clearance,
                "rpy_offset": rpy_offset,
                "contact_goal": "collect_preinsert_xyz_offset",
                "coarse_z_jitter_amplitude": z_jitter,
                "coarse_z_offset": z_offset,
            }
        else:
            handoff_sample = self._sample_insert_usb_handoff(clearance)
            xy_offset = handoff_sample["xy_offset"]
            rpy_offset = handoff_sample["rpy_offset"]
            z_clearance = float(handoff_sample["z_clearance"])
            handoff_pose = self.task.opening_pose.add_bias(
                [float(xy_offset[0]), float(xy_offset[1]), z_clearance]
            ).add_rotation(
                [
                    float(rpy_offset[0]),
                    float(rpy_offset[1]),
                    float(rpy_offset[2]),
                ]
            )
            moved = self.task.move(
                self.task.atom.place_actor(
                    self.task.prism,
                    target_pose=handoff_pose,
                    pre_dis=0.01,
                    dis=0.0,
                    is_open=False,
                ),
                tag="move_usb_to_play_pre_insert",
                time_dilation_factor=0.5,
            )
            if not moved or not self.task.plan_success:
                raise RuntimeError("Scripted USB pre-insert handoff failed")

        self.task.metadata["rl_handoff_source"] = "insert_USB collect_data preparation"
        self.task.metadata["rl_handoff_distribution"] = handoff_sample[
            "distribution"
        ]
        self.task.metadata["rl_handoff_profile"] = handoff_sample["profile"]
        self.task.metadata["rl_handoff_contact_goal"] = handoff_sample[
            "contact_goal"
        ]
        selection_sample = handoff_sample["selection_sample"]
        self.task.metadata["rl_handoff_selection_sample"] = (
            None if selection_sample is None else float(selection_sample)
        )
        self.task.metadata["rl_handoff_curriculum_stage_index"] = (
            handoff_sample["curriculum_stage_index"]
        )
        self.task.metadata["rl_handoff_curriculum_stage_name"] = (
            handoff_sample["curriculum_stage_name"]
        )
        self.task.metadata["rl_handoff_curriculum_success_thresholds"] = (
            handoff_sample["curriculum_success_thresholds"]
        )
        self.task.metadata["rl_handoff_curriculum_collected_successes"] = (
            handoff_sample["curriculum_collected_successes"]
        )
        self.task.metadata["rl_handoff_xy_offset"] = xy_offset.tolist()
        self.task.metadata["rl_handoff_xy_offset_norm"] = float(
            np.linalg.norm(xy_offset)
        )
        self.task.metadata["rl_handoff_z_clearance"] = z_clearance
        self.task.metadata["rl_handoff_coarse_z_jitter_amplitude"] = (
            handoff_sample.get("coarse_z_jitter_amplitude")
        )
        self.task.metadata["rl_handoff_coarse_z_offset"] = handoff_sample.get(
            "coarse_z_offset"
        )
        self.task.metadata["rl_handoff_rpy_offset"] = rpy_offset.tolist()
        self.task.metadata["rl_handoff_pose"] = handoff_pose.tolist()
        self.task.metadata["play_pre_insert_clearance"] = z_clearance
        self.task._update_render()
        self.task._set_phase(self.task.PHASE_POLICY)
        self.task.policy_start_step = int(self.task.step_count)
        self.task.policy_step_count = 0

    def _prepare_handoff(self) -> None:
        if self._should_use_insert_usb_handoff():
            self._prepare_insert_usb_collect_handoff()

    def _annotate_episode_metadata(self, episode_seed: int) -> None:
        metadata = getattr(self.task, "metadata", None)
        if not isinstance(metadata, dict):
            return
        metadata.update(
            {
                "rl_bc_checkpoint": str(self.checkpoint_path),
                "rl_episode_seed": int(episode_seed),
                "rl_control_mode": self.control_mode,
                "rl_reward_mode": self.reward_mode,
                "rl_action_repeat": self.action_repeat,
                "rl_handoff_mode": self.handoff_mode,
                "rl_force_control": self.force_control,
                "rl_collect_successful_episodes": self.collect_successful_episodes,
                "rl_insert_usb_handoff_distribution": (
                    self.insert_usb_handoff_distribution
                ),
                "rl_insert_usb_coarse_z_jitter": (
                    self.insert_usb_coarse_z_jitter
                ),
                "rl_insert_usb_xy_quit_threshold": (
                    self.insert_usb_xy_quit_threshold
                ),
                "rl_insert_usb_curriculum_success_thresholds": (
                    self.insert_usb_curriculum_success_thresholds
                ),
            }
        )
        metadata.update(self.collection_metadata)

    def _episode_handoff_info(self) -> dict[str, object]:
        metadata = getattr(self.task, "metadata", None)
        if not isinstance(metadata, dict):
            return {}
        keys = (
            "rl_handoff_source",
            "rl_handoff_distribution",
            "rl_handoff_profile",
            "rl_handoff_contact_goal",
            "rl_handoff_selection_sample",
            "rl_handoff_curriculum_stage_index",
            "rl_handoff_curriculum_stage_name",
            "rl_handoff_curriculum_success_thresholds",
            "rl_handoff_curriculum_collected_successes",
            "rl_handoff_xy_offset",
            "rl_handoff_xy_offset_norm",
            "rl_handoff_z_clearance",
            "rl_handoff_coarse_z_jitter_amplitude",
            "rl_handoff_coarse_z_offset",
            "rl_handoff_rpy_offset",
            "play_pre_insert_clearance",
        )
        return {key: metadata[key] for key in keys if key in metadata}

    @staticmethod
    def _debug_format_value(value: object, *, scale: float = 1.0) -> str:
        if value is None:
            return "None"
        try:
            return f"{float(value) * scale:.4f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _debug_format_vector(value: object, *, scale: float = 1.0) -> str:
        if value is None:
            return "None"
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        return "[" + ",".join(f"{item * scale:.4f}" for item in array) + "]"

    @staticmethod
    def _debug_diagnostic_value(
        diagnostics: Mapping[str, object] | None,
        *keys: str,
    ) -> object:
        if diagnostics is None:
            return None
        for key in keys:
            value = diagnostics.get(key)
            if value is not None:
                return value
        return None

    def _debug_should_log_step(self, next_policy_step: int, *, terminal: bool) -> bool:
        if not self.debug_logging:
            return False
        if terminal or next_policy_step == 1:
            return True
        frequency = self.debug_step_log_frequency
        return frequency > 0 and next_policy_step % frequency == 0

    def _debug_log_reset(
        self,
        *,
        episode_seed: int,
        handoff_info: Mapping[str, object],
        diagnostics: Mapping[str, object] | None,
    ) -> None:
        if not self.debug_logging:
            return
        xy_error = self._debug_diagnostic_value(diagnostics, "xy_error")
        z_error = self._debug_diagnostic_value(
            diagnostics,
            "abs_z_error",
            "z_error",
        )
        print(
            "[TactileControlEnv debug] reset "
            f"seed={episode_seed} "
            f"profile={handoff_info.get('rl_handoff_profile')} "
            f"stage={handoff_info.get('rl_handoff_curriculum_stage_name')} "
            f"goal={handoff_info.get('rl_handoff_contact_goal')} "
            "xy_offset_mm="
            f"{self._debug_format_vector(handoff_info.get('rl_handoff_xy_offset'), scale=1000.0)} "
            "xy_offset_norm_mm="
            f"{self._debug_format_value(handoff_info.get('rl_handoff_xy_offset_norm'), scale=1000.0)} "
            "z_clearance_mm="
            f"{self._debug_format_value(handoff_info.get('rl_handoff_z_clearance'), scale=1000.0)} "
            "rpy_deg="
            f"{self._debug_format_vector(handoff_info.get('rl_handoff_rpy_offset'), scale=180.0 / np.pi)} "
            f"xy_error_mm={self._debug_format_value(xy_error, scale=1000.0)} "
            f"z_error_mm={self._debug_format_value(z_error, scale=1000.0)}",
            flush=True,
        )

    def _debug_log_step_begin(
        self,
        *,
        next_policy_step: int,
        policy_action: np.ndarray,
        bc_action: np.ndarray,
        normalized_action: np.ndarray,
        target_qpos: np.ndarray,
    ) -> None:
        if not self._debug_should_log_step(next_policy_step, terminal=False):
            return
        action_delta = policy_action - bc_action
        print(
            "[TactileControlEnv debug] step_begin "
            f"step={next_policy_step} "
            f"profile={getattr(self.task, 'metadata', {}).get('rl_handoff_profile')} "
            f"policy_abs_max={float(np.max(np.abs(policy_action))):.4f} "
            f"bc_abs_max={float(np.max(np.abs(bc_action))):.4f} "
            f"policy_bc_l2={float(np.linalg.norm(action_delta)):.4f} "
            f"policy_bc_abs_max={float(np.max(np.abs(action_delta))):.4f} "
            f"normalized={self._debug_format_vector(normalized_action)} "
            f"target_qpos={self._debug_format_vector(target_qpos)}",
            flush=True,
        )

    def _debug_log_step_end(
        self,
        *,
        policy_step: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool,
        diagnostics: Mapping[str, object] | None,
        info: Mapping[str, object],
    ) -> None:
        terminal = bool(terminated or truncated)
        if not self._debug_should_log_step(policy_step, terminal=terminal):
            return
        xy_error = self._debug_diagnostic_value(diagnostics, "xy_error")
        z_error = self._debug_diagnostic_value(
            diagnostics,
            "abs_z_error",
            "z_error",
        )
        tilt_deg = self._debug_diagnostic_value(diagnostics, "tilt_angle_deg")
        print(
            "[TactileControlEnv debug] step_end "
            f"step={policy_step} "
            f"reward={float(reward):.3f} "
            f"success={success} "
            f"terminated={terminated} "
            f"truncated={truncated} "
            f"terminal_reason={info.get('terminal_reason')} "
            f"xy_error_mm={self._debug_format_value(xy_error, scale=1000.0)} "
            f"z_error_mm={self._debug_format_value(z_error, scale=1000.0)} "
            f"tilt_deg={self._debug_format_value(tilt_deg)} "
            f"xy_out={info.get('rl_xy_out_of_slot')}",
            flush=True,
        )

    def _finalize_finished_episode(
        self,
        *,
        success: bool,
    ) -> dict[str, object]:
        if not (self.collect_successful_episodes or self.clean_finished_episodes):
            return {}

        self.collection_attempts += 1
        saved_hdf5_path = None
        result = "success" if success else "fail"
        if success:
            if self.collect_successful_episodes:
                metadata = getattr(self.task, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["rl_policy_steps"] = int(self.policy_step)
                    metadata["rl_collection_attempt_index"] = (
                        self.collection_attempts - 1
                    )
                    metadata["rl_collection_success_index"] = (
                        self.collected_successes
                    )
                tmp_save_dir = getattr(self.task, "tmp_save_dir", None)
                pkl_count = None
                if tmp_save_dir is not None:
                    tmp_save_path = Path(tmp_save_dir)
                    pkl_count = (
                        sum(1 for _ in tmp_save_path.glob("*.pkl"))
                        if tmp_save_path.exists()
                        else 0
                    )
                if self.debug_logging:
                    print(
                        "[TactileControlEnv debug] save_success begin "
                        f"success_index={self.collected_successes} "
                        f"tmp_save_dir={tmp_save_dir} "
                        f"pkl_count={pkl_count} "
                        f"save_path={getattr(self.task, 'save_path', None)}",
                        flush=True,
                    )
                if pkl_count == 0:
                    raise RuntimeError(
                        "Cannot save a successful RL episode because no cached "
                        "observations were written. Use task mode 'collect' "
                        "when --save-successful-episodes or "
                        "--collect-success-target is enabled."
                    )
                save_start = time.perf_counter()
                self.task.save_to_hdf5()
                saved_hdf5_path = str(self.task.save_path)
                if self.debug_logging:
                    print(
                        "[TactileControlEnv debug] save_success done "
                        f"success_index={self.collected_successes} "
                        f"save_path={saved_hdf5_path} "
                        f"elapsed_s={time.perf_counter() - save_start:.3f}",
                        flush=True,
                    )
                self.collected_successes += 1
                if self.collection_mean_steps > 0.0:
                    self.collection_mean_steps = (
                        (self.collected_successes - 1) * self.collection_mean_steps
                        + self.task.step_count
                    ) / self.collected_successes
                else:
                    self.collection_mean_steps = float(self.task.step_count)
        else:
            self.collection_failures += 1

        if self.clean_finished_episodes:
            self.task.clean_cache(
                mean_steps=self.collection_mean_steps,
                result=result,
            )

        return {
            "collection_attempts": self.collection_attempts,
            "collection_failures": self.collection_failures,
            "collected_successes": self.collected_successes,
            "saved_hdf5_path": saved_hdf5_path,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = self.next_seed if seed is None else int(seed)
        self.next_seed = episode_seed + 1
        self.task.reset(seed=episode_seed)
        self._prepare_handoff()
        self._annotate_episode_metadata(episode_seed)
        self.policy_step = 0
        self.gripper_hold_qpos = float(self.task._robot_manager.get_gripper_qpos())
        raw_observation = self.task._get_observations()
        joint = raw_observation["embodiment"]["joint"]
        if torch.is_tensor(joint):
            joint = joint.detach().cpu().numpy()
        self.execution_qpos = np.asarray(
            joint[: self.actor.encoder.qpos_dim], dtype=np.float32
        ).copy()
        self.last_observation = self.encode_observation(raw_observation)
        info = {
            "seed": episode_seed,
            "metrics": self.task.get_rl_metrics(),
        }
        handoff_info = self._episode_handoff_info()
        if handoff_info:
            info["handoff"] = handoff_info
        diagnostics = None
        if hasattr(self.task, "_get_success_diagnostics"):
            diagnostics = self.task._get_success_diagnostics()
            info["success_diagnostics"] = diagnostics
        self._debug_log_reset(
            episode_seed=episode_seed,
            handoff_info=handoff_info,
            diagnostics=diagnostics,
        )
        return self.last_observation, info

    def _final_normalized_action(
        self,
        policy_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.last_observation is None:
            raise RuntimeError("Environment must be reset before step()")
        bc_action = self.bc_normalized_action(self.last_observation)
        scaled_bc_action = clip_action(
            bc_action * self.bc_action_gain
        )
        if self.control_mode == "bc":
            final_action = scaled_bc_action
        else:
            final_action = policy_action
        return bc_action, scaled_bc_action, clip_action(final_action)

    def _check_insert_usb_xy_quit(
        self,
        diagnostics: Mapping[str, object] | None,
    ) -> dict[str, object]:
        threshold = self.insert_usb_xy_quit_threshold
        if threshold is None or diagnostics is None:
            return {"rl_xy_out_of_slot": False}
        rel_xyz = diagnostics.get("rel_xyz")
        if rel_xyz is None:
            return {"rl_xy_out_of_slot": False}
        rel_xy = np.asarray(rel_xyz, dtype=np.float64).reshape(-1)[:2]
        if rel_xy.shape != (2,) or not np.all(np.isfinite(rel_xy)):
            return {"rl_xy_out_of_slot": False}

        axis_error = float(np.max(np.abs(rel_xy)))
        xy_error = float(np.linalg.norm(rel_xy))
        out_of_slot = bool(axis_error > threshold)
        return {
            "rl_xy_out_of_slot": out_of_slot,
            "rl_xy_quit_threshold": float(threshold),
            "rl_xy_axis_error": axis_error,
            "rl_xy_error": xy_error,
        }

    def step(self, policy_action):
        if self.last_observation is None:
            raise RuntimeError("Environment must be reset before step()")
        policy_action = clip_action(np.asarray(policy_action, dtype=np.float32))
        (
            bc_action,
            scaled_bc_action,
            normalized_action,
        ) = self._final_normalized_action(policy_action)
        if self.execution_qpos is None:
            raise RuntimeError("Execution qpos is unavailable")
        current_qpos = self.execution_qpos
        target_qpos = action_to_target_qpos(
            current_qpos=current_qpos,
            action=normalized_action,
            action_scale=self.action_scale,
        )
        full_target = np.concatenate(
            [target_qpos, np.asarray([self.gripper_hold_qpos], dtype=np.float32)]
        )
        next_policy_step = self.policy_step + 1
        self._debug_log_step_begin(
            next_policy_step=next_policy_step,
            policy_action=policy_action,
            bc_action=bc_action,
            normalized_action=normalized_action,
            target_qpos=target_qpos,
        )

        raw_observation, task_reward, terminated, truncated, info = self.task.env_step(
            full_target,
            action_type="qpos",
            force=self.force_control,
            action_repeat=self.action_repeat,
        )
        self.policy_step += 1
        joint = raw_observation["embodiment"]["joint"]
        if torch.is_tensor(joint):
            joint = joint.detach().cpu().numpy()
        self.execution_qpos = np.asarray(
            joint[: self.actor.encoder.qpos_dim], dtype=np.float32
        ).copy()
        self.last_observation = self.encode_observation(raw_observation)
        success = bool(info.get("success", False))
        reward = 1.0 if success else 0.0
        if self.reward_mode == "task":
            reward = float(task_reward)

        diagnostics = None
        if hasattr(self.task, "_get_success_diagnostics"):
            diagnostics = self.task._get_success_diagnostics()
        xy_quit_info = self._check_insert_usb_xy_quit(diagnostics)
        if (
            not success
            and not terminated
            and bool(xy_quit_info.get("rl_xy_out_of_slot", False))
        ):
            truncated = True
            xy_quit_info["terminal_reason"] = "xy_out_of_slot"
            if (
                hasattr(self.task, "_set_phase")
                and hasattr(self.task, "PHASE_TERMINAL")
                and getattr(self.task, "phase_id", None)
                != getattr(self.task, "PHASE_TERMINAL")
            ):
                self.task._set_phase(
                    self.task.PHASE_TERMINAL,
                    terminal_reason="xy_out_of_slot",
                )
        info.update(
            {
                "bc_action": bc_action,
                "scaled_bc_action": scaled_bc_action,
                "bc_action_gain": self.bc_action_gain,
                "policy_action": policy_action,
                "normalized_action": normalized_action,
                "target_qpos": target_qpos,
                "gripper_hold_qpos": self.gripper_hold_qpos,
                "control_mode": self.control_mode,
                "reward_mode": self.reward_mode,
                "task_reward": float(task_reward),
                "success_diagnostics": diagnostics,
                **xy_quit_info,
            }
        )
        terminal_reason = getattr(self.task, "terminal_reason", None)
        if terminal_reason is not None:
            info["terminal_reason"] = terminal_reason
        self._debug_log_step_end(
            policy_step=self.policy_step,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            diagnostics=diagnostics,
            info=info,
        )
        if terminated or truncated:
            info.update(self._finalize_finished_episode(success=success))
        return self.last_observation, reward, terminated, truncated, info

    def capture_insert_usb_pose_snapshot(self) -> dict[str, object]:
        """Capture the held USB pose in world and gripper-center frames."""
        required_attributes = (
            "prism",
            "_robot_manager",
        )
        missing = [
            name for name in required_attributes if not hasattr(self.task, name)
        ]
        if missing or not hasattr(
            self.task._robot_manager,
            "get_gripper_center_pose",
        ):
            raise RuntimeError(
                "USB pose snapshots are only supported for tasks with a held "
                f"actor and gripper-center pose; missing {missing}"
            )

        usb_pose = self.task.prism.get_pose()
        gripper_center_pose = self.task._robot_manager.get_gripper_center_pose()
        usb_in_gripper_center = usb_pose.rebase(
            to_coord=gripper_center_pose
        )
        snapshot = {
            "usb_world_pose": usb_pose.tolist(),
            "gripper_center_world_pose": gripper_center_pose.tolist(),
            "usb_in_gripper_center_pose": usb_in_gripper_center.tolist(),
        }
        if hasattr(self.task._robot_manager, "get_gripper_qpos"):
            snapshot["gripper_qpos"] = float(
                self.task._robot_manager.get_gripper_qpos()
            )
        return snapshot

    def complete_insert_usb_with_motion_plan(
        self,
        *,
        retreat_clearance: float | None = None,
        initial_pose_snapshot: Mapping[str, object] | None = None,
    ):
        """Finish an Insert USB rollout with the task's scripted planner."""
        if retreat_clearance is not None and (
            not np.isfinite(retreat_clearance)
            or float(retreat_clearance) < 0.0
        ):
            raise ValueError(
                "retreat_clearance must be finite and non-negative"
            )

        required_attributes = (
            "_update_insert_reference_poses",
            "_get_success_diagnostics",
            "_open_gripper_after_insert",
            "check_success",
            "opening_pose",
            "target_pose",
            "prism",
            "atom",
            "move",
        )
        missing = [
            name for name in required_attributes if not hasattr(self.task, name)
        ]
        if missing:
            raise RuntimeError(
                "Motion-plan completion is only supported for Insert USB tasks; "
                f"missing {missing}"
            )

        switch_diagnostics = self.task._get_success_diagnostics()
        switch_pose_snapshot = self.capture_insert_usb_pose_snapshot()
        module = sys.modules[self.task.__class__.__module__]
        clearance = float(getattr(module, "PLAY_PRE_INSERT_CLEARANCE", 0.0))
        self.task._update_insert_reference_poses()

        retreat_distance = 0.0
        retreat_moved = None
        retreated_diagnostics = None
        retreated_pose_snapshot = None
        if retreat_clearance is not None:
            retreat_target_pose = self.task.opening_pose.add_bias(
                [0.0, 0.0, float(retreat_clearance)]
            )
            retreat_distance = max(
                0.0,
                float(
                    retreat_target_pose.p[2]
                    - self.task.prism.get_pose().p[2]
                ),
            )
            retreat_moved = True
            if retreat_distance > 0.0:
                retreat_moved = bool(
                    self.task.move(
                        self.task.atom.move_by_displacement(
                            z=retreat_distance,
                            xyz_coord="world",
                        ),
                        tag="hybrid_motion_plan_retreat",
                        time_dilation_factor=0.5,
                    )
                )
            retreated_diagnostics = self.task._get_success_diagnostics()
            retreated_pose_snapshot = self.capture_insert_usb_pose_snapshot()

        pre_insert_pose = self.task.opening_pose.add_bias(
            [0.0, 0.0, clearance]
        )
        alignment_moved = False
        retreat_succeeded = retreat_moved is not False
        if retreat_succeeded and bool(getattr(self.task, "plan_success", True)):
            alignment_moved = bool(
                self.task.move(
                    self.task.atom.place_actor(
                        self.task.prism,
                        target_pose=pre_insert_pose,
                        pre_dis=0.01,
                        dis=0.0,
                        is_open=False,
                    ),
                    tag="hybrid_motion_plan_align",
                    time_dilation_factor=0.5,
                )
            )
        aligned_diagnostics = self.task._get_success_diagnostics()
        aligned_pose_snapshot = self.capture_insert_usb_pose_snapshot()

        insert_distance = max(
            0.0,
            float(self.task.prism.get_pose().p[2] - self.task.target_pose.p[2]),
        )
        insertion_moved = False
        if alignment_moved and bool(getattr(self.task, "plan_success", True)):
            insertion_moved = bool(
                self.task.move(
                    self.task.atom.move_by_displacement(
                        z=-insert_distance,
                        xyz_coord="world",
                    ),
                    tag="hybrid_motion_plan_insert",
                    time_dilation_factor=0.5,
                    constraint_pose=[1, 1, 1, 1, 1, 0],
                )
            )
        if insertion_moved and bool(getattr(self.task, "plan_success", True)):
            self.task._open_gripper_after_insert()
            self.task.delay(40, is_save=True)

        success = bool(self.task.check_success())
        self.task.eval_success = success
        terminal_reason = (
            "hybrid_motion_plan_success"
            if success
            else "hybrid_motion_plan_failure"
        )
        if hasattr(self.task, "_set_phase") and hasattr(
            self.task, "PHASE_TERMINAL"
        ):
            self.task._set_phase(
                self.task.PHASE_TERMINAL,
                terminal_reason=terminal_reason,
            )

        final_diagnostics = self.task._get_success_diagnostics()
        raw_observation = self.task._get_observations()
        self.last_observation = self.encode_observation(raw_observation)
        reward = 1.0 if success else 0.0
        info = {
            "success": success,
            "success_diagnostics": final_diagnostics,
            "terminal_reason": terminal_reason,
            "hybrid_motion_plan": {
                "triggered": True,
                "policy_step": int(self.policy_step),
                "switch_diagnostics": switch_diagnostics,
                "retreat_clearance": retreat_clearance,
                "retreat_distance": retreat_distance,
                "retreat_moved": retreat_moved,
                "retreated_diagnostics": retreated_diagnostics,
                "alignment_moved": alignment_moved,
                "aligned_diagnostics": aligned_diagnostics,
                "insert_distance": insert_distance,
                "insertion_moved": insertion_moved,
                "plan_success": bool(getattr(self.task, "plan_success", True)),
                "pose_snapshots": {
                    "initial": (
                        dict(initial_pose_snapshot)
                        if initial_pose_snapshot is not None
                        else None
                    ),
                    "switch": switch_pose_snapshot,
                    "retreated": retreated_pose_snapshot,
                    "aligned": aligned_pose_snapshot,
                },
            },
        }
        return self.last_observation, reward, success, not success, info

    def close(self):
        self.task.close()
