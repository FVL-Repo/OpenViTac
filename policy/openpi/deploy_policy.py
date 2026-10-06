from __future__ import annotations

import numpy as np
import torch

from .._base_policy import BasePolicy
from .client import OpenPiClientRuntime, OpenPiServerConfig
from .debug import save_openpi_observation_images
from .transforms import (
    openpi_obs_from_openvitac,
    relative_joint_chunk_to_abs_qpos8,
    sanitize_abs_joint_action,
    sanitize_delta_eef_action,
)


class Policy(BasePolicy):
    """OpenViTac OpenPI eval policy。

    输入:
        deploy_config: eval_policy.py 读取 deploy.yml 后传入的配置 dict。

    输出:
        一个兼容 OpenViTac BasePolicy 接口的 policy 实例。

    支持模式:
        abs_joint: state/action 都是 [7 arm qpos, gripper qpos]。
        relative_joint: state 是 [7 arm qpos, gripper qpos]；
            server action 是 [target_joint7 - state_joint7, gripper qpos]，
            client 会用本次 infer 的 state 转回 absolute qpos 后执行。
        delta_eef: state 支持 [ee_pos(3), ee_quat_xyzw(4), gripper qpos]
            或 [ee_pos(3), ee_rot6d(6), gripper qpos]；
            action 是 [delta_xyz(3), delta_rotvec(3), gripper_abs_qpos(1)]。
    """

    def __init__(self, deploy_config: dict):
        super().__init__(deploy_config)
        openpi_cfg = dict(deploy_config.get("openpi", {}))
        if not openpi_cfg:
            raise KeyError("deploy config 缺少 openpi 配置段。")

        requested_control_mode = str(openpi_cfg.get("control_mode", "abs_joint")).lower()
        self.control_mode = "relative_joint" if requested_control_mode == "delta_joint" else requested_control_mode
        if self.control_mode not in ("abs_joint", "relative_joint", "delta_eef"):
            raise ValueError(
                "OpenPI control_mode 只支持 'abs_joint'、'relative_joint' 或 'delta_eef'，"
                f"实际为 {requested_control_mode!r}"
            )

        default_state_dim = 8
        default_action_dim = 7 if self.control_mode == "delta_eef" else 8
        self.state_dim = int(openpi_cfg.get("state_dim", default_state_dim))
        self.action_dim = int(openpi_cfg.get("action_dim", default_action_dim))
        if self.control_mode in ("abs_joint", "relative_joint") and (self.state_dim != 8 or self.action_dim != 8):
            raise ValueError(
                f"{self.control_mode} 模式要求 state_dim=8 且 action_dim=8，"
                f"实际 state_dim={self.state_dim}, action_dim={self.action_dim}"
            )
        self.eef_state_mode = str(openpi_cfg.get("eef_state_mode", "")).lower()
        if self.control_mode == "delta_eef":
            if not self.eef_state_mode:
                self.eef_state_mode = "rot6d_10" if self.state_dim == 10 else "quat_xyzw_8"
            if self.eef_state_mode not in ("quat_xyzw_8", "rot6d_10"):
                raise ValueError(
                    "openpi.eef_state_mode 只支持 'quat_xyzw_8' 或 'rot6d_10'，"
                    f"实际为 {self.eef_state_mode!r}"
                )
        else:
            self.eef_state_mode = ""
        expected_delta_state_dim = 10 if self.eef_state_mode == "rot6d_10" else 8
        if self.control_mode == "delta_eef" and (self.state_dim != expected_delta_state_dim or self.action_dim != 7):
            raise ValueError(
                f"delta_eef/{self.eef_state_mode} 模式要求 "
                f"state_dim={expected_delta_state_dim} 且 action_dim=7，"
                f"实际 state_dim={self.state_dim}, action_dim={self.action_dim}"
            )

        self.force_task_action_repeat = None
        if deploy_config.get("task_name") in ("grasp_fragile_chip", "bulb_tightening", "tension_strap", "wipe_vase"):
            from envs._force_task_utils import force_task_openpi_repeat
            self.force_task_action_repeat = force_task_openpi_repeat(deploy_config)

        api_key = openpi_cfg.get("api_key")
        if api_key == "":
            api_key = None

        self.prompt = str(openpi_cfg.get("prompt", "pick and insert the HDMI"))
        self.prompt_from_task_instruction = bool(openpi_cfg.get("prompt_from_task_instruction", True))
        self.image_size = int(openpi_cfg.get("image_size", 224))
        self.send_tactile = bool(openpi_cfg.get("send_tactile", True))
        tactile_image_key = openpi_cfg.get("tactile_image_key", None)
        if tactile_image_key is None:
            self.tactile_image_keys = None
        elif isinstance(tactile_image_key, str):
            self.tactile_image_keys = (tactile_image_key,)
        else:
            self.tactile_image_keys = tuple(str(item) for item in tactile_image_key)
        self.image_color_order = str(openpi_cfg.get("image_color_order", "rgb")).lower()
        if self.image_color_order not in ("rgb", "bgr"):
            raise ValueError(f"image_color_order 只支持 'rgb' 或 'bgr'，实际为 {self.image_color_order!r}")
        self.open_loop_horizon = max(1, int(openpi_cfg.get("open_loop_horizon", 20)))
        self.debug_dump_first_n_obs = max(0, int(openpi_cfg.get("debug_dump_first_n_obs", 0)))
        self.debug_dump_dir = str(openpi_cfg.get("debug_dump_dir", "debug/openpi_obs"))
        self.max_position_delta = _optional_float(openpi_cfg.get("max_position_delta", None))
        self.max_rotation_delta = _optional_float(openpi_cfg.get("max_rotation_delta", None))
        self.max_joint_delta = _optional_float(openpi_cfg.get("max_joint_delta", None))
        self.eef_action_type = str(openpi_cfg.get("eef_action_type", "delta_ee_rotvec_ik"))
        if self.eef_action_type not in ("delta_ee_rotvec_ik", "delta_ee_rotvec"):
            raise ValueError(
                "openpi.eef_action_type 只支持 'delta_ee_rotvec_ik' 或 'delta_ee_rotvec'，"
                f"实际为 {self.eef_action_type!r}"
            )
        # EEF OpenPI 默认和 abs_joint 一样直接写入 IK 求得的 joint position。
        self.eef_servo_force = bool(openpi_cfg.get("eef_servo_force", True))

        self.temporal_ensemble = bool(openpi_cfg.get("temporal_ensemble", False))
        self.temporal_ensemble_k = float(
            openpi_cfg.get(
                "temporal_ensemble_k",
                openpi_cfg.get("ensemble_K", 0.01),
            )
        )
        self.temporal_ensemble_horizon = max(
            1,
            int(
                openpi_cfg.get(
                    "temporal_ensemble_horizon",
                    openpi_cfg.get("chunk_first_n", self.open_loop_horizon),
                )
            ),
        )
        self.temporal_ensemble_space = str(openpi_cfg.get("temporal_ensemble_space", "action")).lower()
        if self.temporal_ensemble_space not in ("action", "delta_eef_qpos_rollout"):
            raise ValueError(
                "openpi.temporal_ensemble_space 只支持 'action' 或 'delta_eef_qpos_rollout'，"
                f"实际为 {self.temporal_ensemble_space!r}"
            )
        if self.temporal_ensemble_space == "delta_eef_qpos_rollout" and self.control_mode != "delta_eef":
            raise ValueError("temporal_ensemble_space='delta_eef_qpos_rollout' 只能用于 delta_eef 模式。")
        self.temporal_ensemble_query_interval = self.open_loop_horizon

        self.client = OpenPiClientRuntime(
            OpenPiServerConfig(
                host=str(openpi_cfg.get("host", "")),
                port=int(openpi_cfg.get("port", 8000)),
                api_key=None if api_key is None else str(api_key),
                action_dim=self.action_dim,
                open_loop_horizon=self.open_loop_horizon,
                websocket_ping_interval=_optional_float(openpi_cfg.get("websocket_ping_interval", None)),
                websocket_ping_timeout=_optional_float(openpi_cfg.get("websocket_ping_timeout", None)),
                websocket_open_timeout=_optional_float(openpi_cfg.get("websocket_open_timeout", 10.0)),
                websocket_close_timeout=_optional_float(openpi_cfg.get("websocket_close_timeout", 10.0)),
            )
        )
        self.client.connect()

        self._action_chunk: np.ndarray | None = None
        self._chunk_step = 0
        self._policy_step_index = 0
        self._action_predictions: dict[int, list[tuple[int, np.ndarray]]] = {}
        print(
            "OpenPI policy connected: "
            f"{self.client.config.host}:{self.client.config.port}, "
            f"control_mode={self.control_mode}, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, image_size={self.image_size}, "
            f"send_tactile={self.send_tactile}, image_color_order={self.image_color_order}, "
            f"open_loop_horizon={self.open_loop_horizon}, "
            f"temporal_ensemble={self.temporal_ensemble}, "
            f"temporal_ensemble_horizon={self.temporal_ensemble_horizon}, "
            f"temporal_ensemble_query_interval={self.temporal_ensemble_query_interval}, "
            f"ensemble_K={self.temporal_ensemble_k}, "
            f"temporal_ensemble_space={self.temporal_ensemble_space}"
        )

    def eval(self, task, observation):
        """执行一次 OpenPI policy step。

        输入:
            task: OpenViTac BaseTask 实例。
            observation: task._get_observations() 返回的当前 observation。

        输出:
            task.take_action() 返回的 (exec_succ, eval_succ)，用于兼容 OpenViTac 现有 policy 接口。
        """

        obs = openpi_obs_from_openvitac(
            observation=observation,
            prompt=self._get_prompt(task),
            image_size=self.image_size,
            send_tactile=self.send_tactile,
            tactile_image_keys=self.tactile_image_keys,
            image_color_order=self.image_color_order,
            control_mode=self.control_mode,
            eef_state_mode=self.eef_state_mode,
        )
        if self._policy_step_index < self.debug_dump_first_n_obs:
            saved_paths = save_openpi_observation_images(
                obs,
                self.debug_dump_dir,
                self._policy_step_index,
            )
            print(f"OpenPI debug obs dump step={self._policy_step_index}: {saved_paths}")

        if self.temporal_ensemble:
            action = self._temporal_ensemble_action(obs, task)
        else:
            action = self._open_loop_action(obs, task)

        if self.control_mode in ("abs_joint", "relative_joint"):
            torch_action = sanitize_abs_joint_action(action, task)
            if self.force_task_action_repeat is None:
                result = task.take_action(
                    torch_action,
                    action_type="qpos",
                )
            else:
                result = task.take_action(
                    torch_action, action_type="qpos",
                    action_repeat=self.force_task_action_repeat,
                )
        elif self.control_mode == "delta_eef":
            if self.temporal_ensemble and self.temporal_ensemble_space == "delta_eef_qpos_rollout":
                torch_action = sanitize_abs_joint_action(action, task)
                result = task.take_action(
                    torch_action,
                    action_type="qpos",
                    force=self.eef_servo_force,
                )
            else:
                torch_action = sanitize_delta_eef_action(
                    action,
                    task,
                    max_position_delta=self.max_position_delta,
                    max_rotation_delta=self.max_rotation_delta,
                )
                result = task.take_action(
                    torch_action,
                    action_type=self.eef_action_type,
                    force=self.eef_servo_force,
                )
        else:
            raise RuntimeError(f"不支持的 control_mode: {self.control_mode!r}")
        self._policy_step_index += 1
        return result

    def reset(self):
        """重置本地 action chunk 和 server policy 状态。

        输入:
            无。

        输出:
            无。
        """

        self._clear_action_state()
        self.client.reset()

    def close(self):
        """关闭 OpenPI client。

        输入:
            无。

        输出:
            无。
        """

        self.client.close()

    def _get_prompt(self, task) -> str:
        """Return the prompt sent to OpenPI for the current eval step."""

        if self.prompt_from_task_instruction:
            task_instruction = str(getattr(task, "instruction", "") or "").strip()
            if task_instruction:
                return task_instruction
        return self.prompt

    def _open_loop_action(self, obs: dict, task) -> np.ndarray:
        """按 open_loop_horizon 消费 server 返回的动作块。

        输入:
            obs: OpenPI observation dict。

        输出:
            np.ndarray，shape [action_dim]，当前要执行的动作。
        """

        if self._action_chunk is None or self._chunk_step >= self.open_loop_horizon:
            self._action_chunk = self._normalize_server_chunk(self.client.infer(obs), obs, task)
            self._chunk_step = 0
            print(f"OpenPI action chunk: {self._action_chunk.shape}")

        action = np.asarray(self._action_chunk[self._chunk_step], dtype=np.float64)
        self._chunk_step += 1
        return action

    def _normalize_server_chunk(self, chunk: np.ndarray, obs: dict, task) -> np.ndarray:
        """Convert server action chunk into the action representation used for execution."""

        if self.control_mode != "relative_joint":
            return chunk

        gripper_max_qpos = float(getattr(task._robot_manager, "gripper_max_qpos", 0.039))
        return relative_joint_chunk_to_abs_qpos8(
            chunk,
            obs["observation/state"],
            gripper_max_qpos=gripper_max_qpos,
            max_joint_delta=self.max_joint_delta,
        )

    def _temporal_ensemble_action(self, obs: dict, task) -> np.ndarray:
        """模仿 ACT temporal aggregation，对重叠 action chunks 做指数加权平滑。

        输入:
            obs: OpenPI observation dict。

        输出:
            np.ndarray，shape [action_dim]，ensemble 后当前 step 的动作。

        行为:
            每 open_loop_horizon 个 env step 向 server query 一个 action chunk。
            chunk[t] 会被登记为对未来 step=current_step+t 的预测。
            当前 step 如果有多个历史 chunk 都预测过它，就按 ACT 风格权重:
            exp(-k * arange(num_predictions)) 加权平均。
        """

        current_step = self._policy_step_index
        should_query = (
            current_step == 0
            or current_step % self.temporal_ensemble_query_interval == 0
            or current_step not in self._action_predictions
        )
        if should_query:
            chunk = self._normalize_server_chunk(self.client.infer(obs), obs, task)
            if self.temporal_ensemble_space == "delta_eef_qpos_rollout":
                chunk = self._delta_eef_chunk_to_qpos_chunk(chunk, task)
            for offset, action in enumerate(chunk[: self.temporal_ensemble_horizon]):
                target_step = current_step + offset
                self._action_predictions.setdefault(target_step, []).append(
                    (current_step, np.asarray(action, dtype=np.float64))
                )
            if current_step == 0:
                print(
                    "OpenPI temporal ensemble: "
                    f"chunk_shape={chunk.shape}, "
                    f"temporal_ensemble_horizon={self.temporal_ensemble_horizon}, "
                    f"query_interval={self.temporal_ensemble_query_interval}",
                    flush=True,
                )

        predictions = self._action_predictions.get(current_step, [])
        predictions = [
            (query_step, action)
            for query_step, action in predictions
            if 0 <= current_step - query_step < self.temporal_ensemble_horizon
        ]
        if not predictions:
            raise RuntimeError("temporal ensemble 没有当前 step 可用预测。")

        predictions = sorted(predictions, key=lambda item: item[0])
        weights = np.exp(-self.temporal_ensemble_k * np.arange(len(predictions), dtype=np.float64))
        weights = weights / np.sum(weights)
        actions = np.asarray([action for _, action in predictions], dtype=np.float64)
        action = np.sum(actions * weights[:, None], axis=0)
        if not np.all(np.isfinite(action)):
            raise ValueError("OpenPI temporal ensemble 产生了 NaN 或 Inf。")

        stale_before = current_step - self.temporal_ensemble_horizon
        for step in list(self._action_predictions):
            if step <= stale_before or step < current_step:
                self._action_predictions.pop(step, None)
        return action

    def _delta_eef_chunk_to_qpos_chunk(self, chunk: np.ndarray, task) -> np.ndarray:
        """Roll out a delta EEF chunk into absolute qpos8 targets before ensembling."""

        if self.control_mode != "delta_eef":
            raise RuntimeError("delta_eef_qpos_rollout 只能用于 delta_eef control_mode。")

        robot = task._robot_manager
        ee_pos_b, ee_quat_b = robot.get_ee_pose_tensor()
        joint_pos = robot.robot.data.joint_pos[:, robot._arm_ids]
        gripper_qpos = torch.as_tensor(robot.get_gripper_qpos(), dtype=torch.float32, device=task.device)

        qpos_actions = []
        rollout_len = min(int(chunk.shape[0]), self.temporal_ensemble_horizon)
        gripper_max_qpos = float(getattr(robot, "gripper_max_qpos", 0.039))
        for raw_action in np.asarray(chunk[:rollout_len], dtype=np.float64):
            action_np = raw_action.reshape(-1).astype(np.float32)
            if action_np.shape[0] != 7:
                raise ValueError(f"delta_eef action 必须是 7D，实际 shape={action_np.shape}")
            if not np.all(np.isfinite(action_np)):
                raise ValueError(f"delta_eef action 中包含 NaN 或 Inf: {action_np}")
            if self.max_position_delta is not None:
                action_np[:3] = np.clip(action_np[:3], -self.max_position_delta, self.max_position_delta)
            if self.max_rotation_delta is not None:
                action_np[3:6] = np.clip(action_np[3:6], -self.max_rotation_delta, self.max_rotation_delta)

            target_gripper_abs = float(np.clip(action_np[6], 0.0, gripper_max_qpos))
            action_np[6] = target_gripper_abs - float(gripper_qpos.detach().cpu().item())
            action_tensor = torch.as_tensor(action_np, dtype=torch.float32, device=task.device)
            joint_pos, gripper_qpos, ee_pos_b, ee_quat_b = robot.compute_delta_ee_rotvec_qpos_target(
                action_tensor,
                ee_pos_b=ee_pos_b,
                ee_quat_b=ee_quat_b,
                joint_pos=joint_pos,
                gripper_qpos=gripper_qpos,
            )
            qpos_action = torch.cat([joint_pos[0], gripper_qpos.reshape(1)], dim=0)
            qpos_actions.append(qpos_action.detach().cpu().numpy())

        if not qpos_actions:
            raise ValueError("OpenPI delta EEF chunk 为空，无法 rollout 成 qpos。")
        return np.ascontiguousarray(np.stack(qpos_actions, axis=0), dtype=np.float64)

    def _clear_action_state(self) -> None:
        self._action_chunk = None
        self._chunk_step = 0
        self._policy_step_index = 0
        self._action_predictions.clear()


def _optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null", ""):
        return None
    return float(value)
