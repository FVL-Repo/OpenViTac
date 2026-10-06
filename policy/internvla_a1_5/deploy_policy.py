from __future__ import annotations

import numpy as np
import torch

from .._base_policy import BasePolicy
from .client import InternVLAServerConfig, InternVLAWebsocketClient
from .debug import save_internvla_observation
from .transforms import internvla_obs_from_openvitac, sanitize_absolute_eef8_action


class Policy(BasePolicy):
    def __init__(self, deploy_config: dict):
        super().__init__(deploy_config)
        cfg = dict(deploy_config.get("internvla_a1_5", deploy_config.get("internvla", {})))
        if not cfg:
            raise KeyError("deploy config must contain internvla_a1_5 config.")

        self.prompt = str(
            cfg.get(
                "prompt",
                "Pick up the USB plug from the blue slot and insert it into the red USB slot.",
            )
        )
        self.prompt_from_task_instruction = bool(cfg.get("prompt_from_task_instruction", True))
        self.open_loop_horizon = max(1, int(cfg.get("open_loop_horizon", 5)))
        self.execute_from_index = max(0, int(cfg.get("execute_from_index", 0)))
        self.action_dim = int(cfg.get("action_dim", 8))
        self.expected_action_schema = str(cfg.get("action_schema", "absolute_eef8_wxyz"))
        self.image_color_order = str(cfg.get("image_color_order", "rgb")).lower()
        self.image_width = _optional_int(cfg.get("image_width", 320))
        self.image_height = _optional_int(cfg.get("image_height", 240))
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError("InternVLA image_width and image_height must both be set, or both be null/0.")
        self.image_size = None
        if self.image_width is not None and self.image_height is not None:
            if self.image_width <= 0 or self.image_height <= 0:
                raise ValueError(
                    f"InternVLA image_width/image_height must be positive, got "
                    f"{self.image_width}x{self.image_height}."
                )
            self.image_size = (self.image_width, self.image_height)
        self.debug_dump_first_n_obs = max(0, int(cfg.get("debug_dump_first_n_obs", 0)))
        self.debug_dump_dir = str(cfg.get("debug_dump_dir", "debug/internvla_a1_5_obs"))
        self.ik_servo_force = bool(cfg.get("ik_servo_force", True))

        api_key = cfg.get("api_key")
        if api_key == "":
            api_key = None
        self.client = InternVLAWebsocketClient(
            InternVLAServerConfig(
                host=str(cfg.get("host", "127.0.0.1")),
                port=int(cfg.get("port", 8020)),
                api_key=None if api_key is None else str(api_key),
                reconnect_sleep_s=float(cfg.get("reconnect_sleep_s", 1.0)),
                request_retries=int(cfg.get("request_retries", 1)),
                websocket_open_timeout=_optional_float(cfg.get("websocket_open_timeout", 10.0)),
                websocket_close_timeout=_optional_float(cfg.get("websocket_close_timeout", 10.0)),
            )
        )

        self._action_chunk: np.ndarray | None = None
        self._chunk_step = 0
        self._policy_step_index = 0
        print(
            "InternVLA A1.5 policy connected: "
            f"{self.client.config.host}:{self.client.config.port}, "
            f"action_schema={self.expected_action_schema}, "
            f"open_loop_horizon={self.open_loop_horizon}, "
            f"execute_from_index={self.execute_from_index}, "
            f"prompt_from_task_instruction={self.prompt_from_task_instruction}, "
            f"image_color_order={self.image_color_order}, "
            f"image_size={self.image_size}, "
            f"ik_servo_force={self.ik_servo_force}",
            flush=True,
        )

    def eval(self, task, observation):
        obs = internvla_obs_from_openvitac(
            observation=observation,
            prompt=self._get_prompt(task),
            image_color_order=self.image_color_order,
            image_size=self.image_size,
        )
        if self._policy_step_index < self.debug_dump_first_n_obs:
            saved_paths = save_internvla_observation(obs, self.debug_dump_dir, self._policy_step_index)
            print(f"InternVLA debug obs dump step={self._policy_step_index}: {saved_paths}", flush=True)

        action = self._open_loop_action(obs)
        qpos_action = self._absolute_eef8_to_qpos8(action, task)
        result = task.take_action(qpos_action, action_type="qpos", force=self.ik_servo_force)
        self._policy_step_index += 1
        return result

    def reset(self):
        self._action_chunk = None
        self._chunk_step = 0
        self._policy_step_index = 0
        self.client.reset()

    def close(self):
        self.client.close()

    def _get_prompt(self, task) -> str:
        if self.prompt_from_task_instruction:
            task_instruction = str(getattr(task, "instruction", "") or "").strip()
            if task_instruction:
                return task_instruction
        return self.prompt

    def _open_loop_action(self, obs: dict) -> np.ndarray:
        if self._action_chunk is None or self._chunk_step >= self.open_loop_horizon:
            result = self.client.infer(obs)
            self._action_chunk = self._extract_actions(result)
            self._chunk_step = 0
            print(f"InternVLA action chunk: {self._action_chunk.shape}", flush=True)
        action = np.asarray(self._action_chunk[self._chunk_step], dtype=np.float64)
        self._chunk_step += 1
        return action

    def _extract_actions(self, result: dict) -> np.ndarray:
        if "actions" not in result:
            raise RuntimeError(f"InternVLA server response missing actions: {list(result.keys())}")
        action_schema = result.get("action_schema", self.expected_action_schema)
        if action_schema != self.expected_action_schema:
            raise ValueError(
                f"InternVLA action_schema mismatch: expected {self.expected_action_schema!r}, "
                f"got {action_schema!r}."
            )
        execute_from_index = int(result.get("execute_from_index", self.execute_from_index))
        if execute_from_index < 0:
            raise ValueError(f"InternVLA execute_from_index must be non-negative, got {execute_from_index}.")
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"InternVLA actions must be [T,{self.action_dim}], got shape={actions.shape}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("InternVLA actions contain NaN or Inf.")
        executable = actions[execute_from_index:]
        if executable.shape[0] < self.open_loop_horizon:
            raise ValueError(
                f"InternVLA executable action count {executable.shape[0]} is smaller than "
                f"open_loop_horizon={self.open_loop_horizon}."
            )
        return np.ascontiguousarray(executable)

    def _absolute_eef8_to_qpos8(self, action: np.ndarray, task) -> torch.Tensor:
        action_np = sanitize_absolute_eef8_action(action, task)
        robot = task._robot_manager
        if robot._ik_controller is None:
            robot._setup_ik_controller()

        ee_pos_b, ee_quat_b = robot.get_ee_pose_tensor()
        target_pos_b = torch.as_tensor(action_np[:3], dtype=torch.float32, device=task.device).reshape(1, 3)
        target_quat_b = torch.as_tensor(action_np[3:7], dtype=torch.float32, device=task.device).reshape(1, 4)
        target_pos_b = target_pos_b.repeat(task.num_envs, 1)
        target_quat_b = target_quat_b.repeat(task.num_envs, 1)
        target_quat_b = target_quat_b / torch.clamp(
            torch.linalg.vector_norm(target_quat_b, dim=1, keepdim=True),
            min=1.0e-8,
        )

        same_rotation_opposite_sign = torch.sum(target_quat_b * ee_quat_b, dim=1, keepdim=True) < 0.0
        target_quat_b = torch.where(same_rotation_opposite_sign, -target_quat_b, target_quat_b)

        robot._ik_controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
        jacobian = robot.jacobian_b[:, :, robot._arm_ids]
        joint_pos = robot.robot.data.joint_pos[:, robot._arm_ids]
        joint_pos_des = robot._ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
        )
        target_gripper = torch.as_tensor([action_np[7]], dtype=torch.float32, device=task.device)
        return torch.cat([joint_pos_des[0], target_gripper], dim=0)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null", ""):
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null", ""):
        return None
    parsed = int(value)
    return None if parsed == 0 else parsed
