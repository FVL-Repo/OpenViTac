from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def internvla_obs_from_openvitac(
    observation: dict[str, Any],
    prompt: str,
    image_color_order: str = "rgb",
    image_size: tuple[int, int] | None = (320, 240),
) -> dict[str, Any]:
    if image_color_order not in ("rgb", "bgr"):
        raise ValueError(f"image_color_order must be 'rgb' or 'bgr', got {image_color_order!r}")

    head = rgb_uint8_hwc(_get_camera_image(observation, "head"))
    wrist = rgb_uint8_hwc(_get_camera_image(observation, "wrist"))
    if image_size is not None:
        head = resize_rgb_uint8_hwc(head, image_size)
        wrist = resize_rgb_uint8_hwc(wrist, image_size)
    if image_color_order == "bgr":
        head = np.ascontiguousarray(head[..., ::-1])
        wrist = np.ascontiguousarray(wrist[..., ::-1])

    return {
        "prompt": str(prompt),
        "state": abs_eef8_wxyz_state_from_observation(observation),
        "head_image": head,
        "wrist_image": wrist,
        "state_schema": "absolute_eef8_wxyz",
        "image_schema": "head_wrist_rgb_uint8_hwc",
        "image_size": np.asarray([head.shape[0], head.shape[1]], dtype=np.int32),
    }


def abs_eef8_wxyz_state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    try:
        ee = observation["embodiment"]["ee"]
    except KeyError as exc:
        raise KeyError("observation missing embodiment/ee for InternVLA absolute EEF state.") from exc

    ee_np = _to_numpy(ee).reshape(-1).astype(np.float32)
    if ee_np.shape[0] != 7:
        raise ValueError(f"embodiment/ee must be [x,y,z,qw,qx,qy,qz], got shape={ee_np.shape}")
    quat = ee_np[3:7]
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm <= 1.0e-8:
        raise ValueError(f"Invalid EEF quaternion norm={quat_norm}: {quat}")
    ee_np[3:7] = quat / quat_norm

    try:
        joint = observation["embodiment"]["joint"]
    except KeyError as exc:
        raise KeyError("observation missing embodiment/joint for InternVLA gripper state.") from exc

    joint_np = _to_numpy(joint).reshape(-1).astype(np.float32)
    if joint_np.shape[0] < 9:
        raise ValueError(
            "InternVLA OpenViTac training used the mean of the two gripper joints; "
            f"expected at least 9 joint values, got shape={joint_np.shape}."
        )
    gripper = np.asarray([float(np.mean(joint_np[-2:]))], dtype=np.float32)
    state = np.concatenate([ee_np, gripper], axis=0).astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"InternVLA state contains NaN or Inf: {state}")
    return np.ascontiguousarray(state)


def sanitize_absolute_eef8_action(action: np.ndarray | torch.Tensor, task: Any) -> np.ndarray:
    action_np = _to_numpy(action).reshape(-1).astype(np.float32)
    if action_np.shape[0] != 8:
        raise ValueError(f"InternVLA absolute EEF action must be 8D, got shape={action_np.shape}")
    if not np.all(np.isfinite(action_np)):
        raise ValueError(f"InternVLA absolute EEF action contains NaN or Inf: {action_np}")

    quat = action_np[3:7]
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm <= 1.0e-8:
        raise ValueError(f"Invalid InternVLA action quaternion norm={quat_norm}: {quat}")
    action_np[3:7] = quat / quat_norm

    gripper_max_qpos = float(getattr(task._robot_manager, "gripper_max_qpos", 0.039))
    action_np[7] = float(np.clip(action_np[7], 0.0, gripper_max_qpos))
    return np.ascontiguousarray(action_np)


def rgb_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"InternVLA image must be HWC/CHW 3D, got shape={array.shape}.")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"InternVLA image must have 3 channels, got shape={array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.5:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def resize_rgb_uint8_hwc(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"InternVLA image_size must be positive, got {(width, height)}")
    if image.shape[0] == height and image.shape[1] == width:
        return np.ascontiguousarray(image)
    interpolation = cv2.INTER_AREA if image.shape[0] >= height and image.shape[1] >= width else cv2.INTER_LINEAR
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(resized)


def _get_camera_image(observation: dict[str, Any], name: str) -> Any:
    try:
        return observation["observation"][name]["rgb"]
    except KeyError as exc:
        raise KeyError(f"observation missing observation/{name}/rgb for InternVLA.") from exc


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
