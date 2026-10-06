from __future__ import annotations

import numpy as np

from .._base_policy import BasePolicy
from .client import FtpServerConfig, FtpWebsocketClient
from .debug import save_ftp_observation
from .transforms import ftp_obs_from_openvitacorge, sanitize_qpos8_action


class Policy(BasePolicy):
    def __init__(self, deploy_config: dict):
        super().__init__(deploy_config)
        ftp_cfg = dict(deploy_config.get("ftp_1", deploy_config.get("ftp", {})))
        if not ftp_cfg:
            raise KeyError("deploy config must contain ftp_1 config.")

        self.prompt = str(
            ftp_cfg.get(
                "prompt",
                "Pick up the USB plug from the blue slot and insert it into the red USB slot.",
            )
        )
        self.prompt_from_task_instruction = bool(ftp_cfg.get("prompt_from_task_instruction", False))
        self.open_loop_horizon = max(1, int(ftp_cfg.get("open_loop_horizon", 20)))
        self.execute_from_index = int(ftp_cfg.get("execute_from_index", 1))
        self.action_dim = int(ftp_cfg.get("action_dim", 8))
        self.expected_action_schema = str(ftp_cfg.get("action_schema", "absolute_qpos8"))
        self.temporal_ensemble = bool(ftp_cfg.get("temporal_ensemble", False))
        self.temporal_ensemble_horizon = max(
            0,
            int(ftp_cfg.get("temporal_ensemble_horizon", ftp_cfg.get("chunk_first_n", 0))),
        )
        self.ensemble_k = float(
            ftp_cfg.get("temporal_ensemble_k", ftp_cfg.get("ensemble_K", ftp_cfg.get("ensemble_k", 0.01)))
        )
        self.debug_dump_first_n_obs = max(0, int(ftp_cfg.get("debug_dump_first_n_obs", 0)))
        self.debug_dump_dir = str(ftp_cfg.get("debug_dump_dir", "debug/ftp_1_obs"))
        tactile_image_key = ftp_cfg.get("tactile_image_key", None)
        if tactile_image_key is None:
            self.tactile_image_keys = None
        elif isinstance(tactile_image_key, str):
            self.tactile_image_keys = (tactile_image_key,)
        else:
            self.tactile_image_keys = tuple(str(item) for item in tactile_image_key)

        api_key = ftp_cfg.get("api_key")
        if api_key == "":
            api_key = None
        self.client = FtpWebsocketClient(
            FtpServerConfig(
                host=str(ftp_cfg.get("host", "127.0.0.1")),
                port=int(ftp_cfg.get("port", 8000)),
                api_key=None if api_key is None else str(api_key),
                reconnect_sleep_s=float(ftp_cfg.get("reconnect_sleep_s", 1.0)),
                request_retries=int(ftp_cfg.get("request_retries", 1)),
                websocket_open_timeout=_optional_float(ftp_cfg.get("websocket_open_timeout", 10.0)),
                websocket_close_timeout=_optional_float(ftp_cfg.get("websocket_close_timeout", 10.0)),
            )
        )
        self._action_chunk: np.ndarray | None = None
        self._chunk_history: list[tuple[np.ndarray, int, int]] = []
        self._chunk_step = 0
        self._policy_step_index = 0
        print(
            "FTP-1 policy connected: "
            f"{self.client.config.host}:{self.client.config.port}, "
            f"open_loop_horizon={self.open_loop_horizon}, "
            f"execute_from_index={self.execute_from_index}, action_dim={self.action_dim}, "
            f"prompt_from_task_instruction={self.prompt_from_task_instruction}, "
            f"temporal_ensemble={self.temporal_ensemble}, "
            f"temporal_ensemble_horizon={self.temporal_ensemble_horizon}, "
            f"ensemble_K={self.ensemble_k}",
            flush=True,
        )

    def eval(self, task, observation):
        obs = ftp_obs_from_openvitacorge(
            observation=observation,
            prompt=self._get_prompt(task),
            tactile_image_keys=self.tactile_image_keys,
        )
        if self._policy_step_index < self.debug_dump_first_n_obs:
            saved_paths = save_ftp_observation(obs, self.debug_dump_dir, self._policy_step_index)
            print(f"FTP-1 debug obs dump step={self._policy_step_index}: {saved_paths}", flush=True)
        if self.temporal_ensemble:
            action = self._temporal_ensemble_action(obs)
        else:
            action = self._open_loop_action(obs)
        torch_action = sanitize_qpos8_action(action, task)
        result = task.take_action(torch_action, action_type="qpos")
        self._policy_step_index += 1
        return result

    def reset(self):
        self._clear_action_state()
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
            print(f"FTP-1 action chunk: {self._action_chunk.shape}", flush=True)
        action = np.asarray(self._action_chunk[self._chunk_step], dtype=np.float64)
        self._chunk_step += 1
        return action

    def _temporal_ensemble_action(self, obs: dict) -> np.ndarray:
        query_now = not self._chunk_history or self._policy_step_index % self.open_loop_horizon == 0
        if query_now:
            result = self.client.infer(obs)
            actions, execute_from_index = self._extract_action_response(result)
            self._chunk_history.append((actions, self._policy_step_index, execute_from_index))
            print(
                "FTP-1 temporal ensemble queried chunk: "
                f"shape={actions.shape}, step={self._policy_step_index}, "
                f"query_every={self.open_loop_horizon}",
                flush=True,
            )

        candidates = []
        pruned_history: list[tuple[np.ndarray, int, int]] = []
        for chunk, infer_step, chunk_execute_from_index in self._chunk_history:
            relative_step = self._policy_step_index - infer_step
            pred_idx = relative_step + chunk_execute_from_index
            if pred_idx < chunk_execute_from_index or pred_idx >= chunk.shape[0]:
                continue
            if self.temporal_ensemble_horizon > 0 and relative_step >= self.temporal_ensemble_horizon:
                continue
            candidates.append(chunk[pred_idx])
            pruned_history.append((chunk, infer_step, chunk_execute_from_index))
        self._chunk_history = pruned_history

        if not candidates:
            raise RuntimeError(
                f"FTP-1 temporal ensemble has no candidate actions at policy_step={self._policy_step_index}."
            )
        actions_stack = np.stack(candidates, axis=0).astype(np.float64)
        weights = np.exp(-self.ensemble_k * np.arange(actions_stack.shape[0], dtype=np.float64))
        weights = weights / weights.sum()
        action = (actions_stack * weights[:, None]).sum(axis=0)
        if not np.all(np.isfinite(action)):
            raise ValueError("FTP-1 temporal ensemble produced NaN or Inf.")
        if self._policy_step_index == 0:
            print(
                "FTP-1 temporal ensemble: "
                f"chunk_shape={actions.shape}, execute_from_index={execute_from_index}, "
                f"temporal_ensemble_horizon={self.temporal_ensemble_horizon}, "
                f"candidates={actions_stack.shape[0]}",
                flush=True,
            )
        return action

    def _extract_actions(self, result: dict) -> np.ndarray:
        actions, execute_from_index = self._extract_action_response(result)
        executable = actions[execute_from_index:]
        if executable.shape[0] < self.open_loop_horizon:
            raise ValueError(
                f"FTP-1 executable action count {executable.shape[0]} is smaller than "
                f"open_loop_horizon={self.open_loop_horizon}."
            )
        return np.ascontiguousarray(executable)

    def _extract_action_response(self, result: dict) -> tuple[np.ndarray, int]:
        if "actions" not in result:
            raise RuntimeError(f"FTP-1 server response missing actions: {list(result.keys())}")
        action_schema = result.get("action_schema", self.expected_action_schema)
        if action_schema != self.expected_action_schema:
            raise ValueError(
                f"FTP-1 action_schema mismatch: expected {self.expected_action_schema!r}, "
                f"got {action_schema!r}."
            )
        execute_from_index = int(result.get("execute_from_index", self.execute_from_index))
        if execute_from_index < 0:
            raise ValueError(f"FTP-1 execute_from_index must be non-negative, got {execute_from_index}.")
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"FTP-1 actions must be [T,{self.action_dim}], got shape={actions.shape}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("FTP-1 actions contain NaN or Inf.")
        if execute_from_index >= actions.shape[0]:
            raise ValueError(
                f"FTP-1 execute_from_index={execute_from_index} is outside actions shape={actions.shape}."
            )
        return np.ascontiguousarray(actions), execute_from_index

    def _clear_action_state(self) -> None:
        self._action_chunk = None
        self._chunk_history = []
        self._chunk_step = 0
        self._policy_step_index = 0


def _optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null", ""):
        return None
    return float(value)
