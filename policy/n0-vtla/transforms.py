from __future__ import annotations

from typing import Any

import numpy as np
import torch


DEFAULT_TACTILE_IMAGE_KEYS = ("rgb_marker", "gel_particle", "force_field_img", "marker_force_img", "rgb")


def n0vtla_obs_from_openvitac(
    observation: dict[str, Any],
    prompt: str,
    *,
    tactile_image_keys: tuple[str, ...] | list[str] | None = None,
    gripper_mode: str = "mean",
    image_color_order: str = "rgb",
) -> dict[str, Any]:
    """Build the exact raw OpenViTac input expected by the remote N0-VTLA server.

    Images intentionally remain at native resolution. The model applies its training-time
    aspect-preserving 224x224 letterbox transform on the server.
    """
    keys = tuple(tactile_image_keys or DEFAULT_TACTILE_IMAGE_KEYS)
    return {
        "qpos": qpos8_from_observation(observation, gripper_mode=gripper_mode),
        "camera_ego_rgb": image_uint8_hwc(_get_camera_image(observation, "head"), image_color_order),
        "right_wrist_camera_rgb": image_uint8_hwc(_get_camera_image(observation, "wrist"), image_color_order),
        "right_tactile_data_gripper": np.stack(
            [
                image_uint8_hwc(_get_tactile_image(observation, ("left_tactile", "left_gsmini"), keys), image_color_order),
                image_uint8_hwc(_get_tactile_image(observation, ("right_tactile", "right_gsmini"), keys), image_color_order),
            ],
            axis=0,
        ),
        "prompt": str(prompt),
    }


def qpos8_from_observation(observation: dict[str, Any], *, gripper_mode: str) -> np.ndarray:
    try:
        joint = _to_numpy(observation["embodiment"]["joint"]).reshape(-1)
    except KeyError as exc:
        raise KeyError("observation missing embodiment/joint for N0-VTLA qpos.") from exc
    if joint.size < 9:
        raise ValueError(f"N0-VTLA requires embodiment/joint [7 arm + 2 fingers], got {joint.shape}.")
    if gripper_mode == "mean":
        gripper = 0.5 * (joint[7] + joint[8])
    elif gripper_mode == "left":
        gripper = joint[7]
    elif gripper_mode == "right":
        gripper = joint[8]
    else:
        raise ValueError(f"gripper_mode must be mean, left, or right; got {gripper_mode!r}.")
    qpos = np.concatenate([joint[:7], np.asarray([gripper])]).astype(np.float32)
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"N0-VTLA qpos contains NaN or Inf: {qpos}")
    return np.ascontiguousarray(qpos)


def image_uint8_hwc(image: torch.Tensor | np.ndarray, color_order: str) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"N0-VTLA image must be 3D HWC/CHW, got {array.shape}.")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"N0-VTLA image must have 3 channels, got {array.shape}.")
    if np.issubdtype(array.dtype, np.floating) and (float(np.nanmax(array)) if array.size else 0.0) <= 1.5:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if color_order == "bgr":
        array = array[..., ::-1]
    elif color_order != "rgb":
        raise ValueError(f"image_color_order must be rgb or bgr, got {color_order!r}.")
    return np.ascontiguousarray(array)


def sanitize_abs_qpos8_action(action: np.ndarray | torch.Tensor, task: Any) -> torch.Tensor:
    action_np = _to_numpy(action).reshape(-1).astype(np.float32)
    if action_np.shape[0] != 8 or not np.all(np.isfinite(action_np)):
        raise ValueError(f"N0-VTLA action must be finite 8D absolute qpos, got {action_np}.")
    gripper_max_qpos = float(getattr(task._robot_manager, "gripper_max_qpos", 0.039))
    action_np[-1] = np.clip(action_np[-1], 0.0, gripper_max_qpos)
    return torch.as_tensor(action_np, dtype=torch.float32, device=task.device)


def _get_camera_image(observation: dict[str, Any], name: str) -> Any:
    try:
        return observation["observation"][name]["rgb"]
    except KeyError as exc:
        raise KeyError(f"observation missing observation/{name}/rgb for N0-VTLA.") from exc


def _get_tactile_image(observation: dict[str, Any], sensor_candidates: tuple[str, ...], image_keys: tuple[str, ...]) -> Any:
    for sensor in sensor_candidates:
        values = observation.get("tactile", {}).get(sensor, {})
        for key in image_keys:
            if key in values:
                return values[key]
    raise KeyError(f"observation missing tactile image for sensors {sensor_candidates} and keys {image_keys}.")


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
