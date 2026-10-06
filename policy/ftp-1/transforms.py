from __future__ import annotations

from typing import Any

import numpy as np
import torch


DEFAULT_TACTILE_IMAGE_KEYS = (
    "rgb_marker",
    "gel_particle",
    "force_field_img",
    "marker_force_img",
    "rgb",
)


def ftp_obs_from_openvitacorge(
    observation: dict[str, Any],
    prompt: str,
    tactile_image_keys: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    left_tactile = rgb_uint8_hwc(
        _get_tactile_image(
            observation,
            sensor_candidates=("left_tactile", "left_gsmini"),
            image_key_candidates=tuple(tactile_image_keys or DEFAULT_TACTILE_IMAGE_KEYS),
        )
    )
    right_tactile = rgb_uint8_hwc(
        _get_tactile_image(
            observation,
            sensor_candidates=("right_tactile", "right_gsmini"),
            image_key_candidates=tuple(tactile_image_keys or DEFAULT_TACTILE_IMAGE_KEYS),
        )
    )
    return {
        "prompt": str(prompt),
        "qpos": qpos8_from_observation(observation),
        "camera_ego_rgb": rgb_uint8_hwc(_get_camera_image(observation, "head")),
        "right_wrist_camera_rgb": rgb_uint8_hwc(_get_camera_image(observation, "wrist")),
        "right_tactile_data_gripper": np.stack(
            [left_tactile, right_tactile],
            axis=0,
        ).astype(np.uint8),
    }


def qpos8_from_observation(observation: dict[str, Any]) -> np.ndarray:
    try:
        joint = observation["embodiment"]["joint"]
    except KeyError as exc:
        raise KeyError("observation missing embodiment/joint for FTP-1 qpos.") from exc
    qpos = _to_numpy(joint).reshape(-1)
    if qpos.shape[0] < 8:
        raise ValueError(f"FTP-1 qpos requires at least 8 dims, got shape={qpos.shape}.")
    qpos8 = qpos[:8].astype(np.float32)
    if not np.all(np.isfinite(qpos8)):
        raise ValueError(f"FTP-1 qpos contains NaN or Inf: {qpos8}")
    return qpos8


def sanitize_qpos8_action(action: np.ndarray | torch.Tensor, task: Any) -> torch.Tensor:
    action_np = _to_numpy(action).reshape(-1).astype(np.float32)
    if action_np.shape[0] != 8:
        raise ValueError(f"FTP-1 action must be 8D absolute qpos, got shape={action_np.shape}.")
    if not np.all(np.isfinite(action_np)):
        raise ValueError(f"FTP-1 action contains NaN or Inf: {action_np}")
    gripper_max_qpos = float(getattr(task._robot_manager, "gripper_max_qpos", 0.039))
    action_np[-1] = float(np.clip(action_np[-1], 0.0, gripper_max_qpos))
    return torch.as_tensor(action_np, dtype=torch.float32, device=task.device)


def rgb_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"FTP-1 image must be HWC/CHW 3D, got shape={array.shape}.")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"FTP-1 image must have 3 channels, got shape={array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.5:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _get_camera_image(observation: dict[str, Any], name: str) -> Any:
    try:
        return observation["observation"][name]["rgb"]
    except KeyError as exc:
        raise KeyError(f"observation missing observation/{name}/rgb for FTP-1.") from exc


def _get_tactile_image(
    observation: dict[str, Any],
    sensor_candidates: tuple[str, ...],
    image_key_candidates: tuple[str, ...],
) -> Any:
    tactile = observation.get("tactile", {})
    for name in sensor_candidates:
        sensor_obs = tactile.get(name, {})
        for image_key in image_key_candidates:
            if image_key in sensor_obs:
                return sensor_obs[image_key]
    raise KeyError(
        "observation missing tactile image for sensors "
        f"{sensor_candidates} and keys {image_key_candidates}."
    )


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
