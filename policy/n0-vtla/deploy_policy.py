from __future__ import annotations

import numpy as np

from .._base_policy import BasePolicy
from .client import N0VTLAWebsocketClient, N0VTLAWebsocketConfig
from .transforms import n0vtla_obs_from_openvitac, sanitize_abs_qpos8_action


class Policy(BasePolicy):
    """Run a OpenViTac single-arm N0-VTLA checkpoint through a remote WebSocket server."""

    def __init__(self, deploy_config: dict):
        super().__init__(deploy_config)
        cfg = dict(deploy_config.get("n0_vtla", deploy_config.get("n0-vtla", {})))
        if not cfg:
            raise KeyError("deploy config must contain an n0_vtla section.")
        self.prompt = str(cfg.get("prompt", "do the task"))
        self.prompt_from_task_instruction = bool(cfg.get("prompt_from_task_instruction", True))
        self.open_loop_horizon = max(1, int(cfg.get("open_loop_horizon", 5)))
        self.action_dim = int(cfg.get("action_dim", 8))
        self.expected_action_schema = str(cfg.get("action_schema", "absolute_qpos8"))
        self.image_color_order = str(cfg.get("image_color_order", "rgb")).lower()
        self.gripper_mode = str(cfg.get("gripper_mode", "mean")).lower()
        tactile_key = cfg.get("tactile_image_key", None)
        self.tactile_image_keys = (str(tactile_key),) if isinstance(tactile_key, str) else None
        api_key = cfg.get("api_key") or None
        self.client = N0VTLAWebsocketClient(
            N0VTLAWebsocketConfig(
                host=str(cfg.get("host", "127.0.0.1")), port=int(cfg.get("port", 8020)),
                api_key=str(api_key) if api_key is not None else None,
                reconnect_sleep_s=float(cfg.get("reconnect_sleep_s", 1.0)),
                request_retries=int(cfg.get("request_retries", 1)),
                websocket_open_timeout=_optional_float(cfg.get("websocket_open_timeout", 10.0)),
                websocket_close_timeout=_optional_float(cfg.get("websocket_close_timeout", 10.0)),
            )
        )
        self._action_chunk: np.ndarray | None = None
        self._chunk_step = 0
        print(
            "N0-VTLA policy connected: "
            f"{self.client.config.host}:{self.client.config.port}, open_loop_horizon={self.open_loop_horizon}, "
            f"image_color_order={self.image_color_order}, gripper_mode={self.gripper_mode}",
            flush=True,
        )

    def eval(self, task, observation):
        obs = n0vtla_obs_from_openvitac(
            observation, self._get_prompt(task), tactile_image_keys=self.tactile_image_keys,
            gripper_mode=self.gripper_mode, image_color_order=self.image_color_order,
        )
        if self._action_chunk is None or self._chunk_step >= self.open_loop_horizon:
            self._action_chunk = self._query_action_chunk(obs)
            self._chunk_step = 0
            print(f"N0-VTLA action chunk: {self._action_chunk.shape}; executing first {self.open_loop_horizon} steps", flush=True)
        action = self._action_chunk[self._chunk_step]
        self._chunk_step += 1
        return task.take_action(sanitize_abs_qpos8_action(action, task), action_type="qpos")

    def reset(self):
        self._action_chunk = None
        self._chunk_step = 0
        self.client.reset()

    def close(self):
        self.client.close()

    def _get_prompt(self, task) -> str:
        if self.prompt_from_task_instruction:
            instruction = str(getattr(task, "instruction", "") or "").strip()
            if instruction:
                return instruction
        return self.prompt

    def _query_action_chunk(self, obs: dict) -> np.ndarray:
        result = self.client.infer(obs)
        schema = result.get("action_schema", self.expected_action_schema)
        if schema != self.expected_action_schema:
            raise ValueError(f"N0-VTLA action_schema mismatch: expected {self.expected_action_schema!r}, got {schema!r}.")
        execute_from_index = int(result.get("execute_from_index", 0))
        actions = np.asarray(result.get("actions"), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"N0-VTLA actions must be [T,{self.action_dim}], got {actions.shape}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("N0-VTLA actions contain NaN or Inf.")
        executable = actions[execute_from_index:]
        if executable.shape[0] < self.open_loop_horizon:
            raise ValueError(f"N0-VTLA returned only {executable.shape[0]} executable actions; need {self.open_loop_horizon}.")
        return np.ascontiguousarray(executable)


def _optional_float(value) -> float | None:
    if value is None or (isinstance(value, str) and value.lower() in ("none", "null", "")):
        return None
    return float(value)
