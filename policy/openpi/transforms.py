from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import transforms3d as t3d


OPENPI_IMAGE_KEYS = ("image", "wrist_image", "left_tactile_image", "right_tactile_image")
DEFAULT_TACTILE_IMAGE_KEYS = (
    "rgb_marker",
    "gel_particle",
    "force_field_img",
    "marker_force_img",
    "rgb",
)


def resize_rgb_image(image: torch.Tensor | np.ndarray, size: int) -> np.ndarray:
    """把 OpenViTac 仿真 RGB 图直接缩放成 OpenPI client 输入图。

    输入:
        image: RGB 图像，支持 torch.Tensor 或 np.ndarray。常见 shape 为 [H,W,3]。
        size: 输出图像边长。

    输出:
        np.ndarray，dtype uint8，shape [size,size,3]，RGB 顺序。

    说明:
        OpenViTac 仿真图像已经按 RGB 使用，这里不做 BGR->RGB。
        训练数据使用直接拉伸到正方形的 resize，因此这里不保留原始宽高比、
        不做 padding。
    """

    if size <= 0:
        raise ValueError(f"image_size 必须为正数，实际为 {size}")

    array = _to_numpy_image(image)
    return _resize_direct_cv2(array, size, size)


def convert_image_color_order(image: np.ndarray, color_order: str) -> np.ndarray:
    """按配置调整发送给 OpenPI server 的图像通道顺序。

    输入:
        image: np.ndarray，dtype uint8，shape [H,W,3]。函数入口约定当前数组语义是 RGB。
        color_order: "rgb" 或 "bgr"。

    输出:
        np.ndarray，dtype uint8，shape [H,W,3]。
        color_order="rgb" 时原样返回；color_order="bgr" 时交换 R/B 通道。
    """

    if color_order == "rgb":
        return np.ascontiguousarray(image)
    if color_order == "bgr":
        return np.ascontiguousarray(image[..., ::-1])
    raise ValueError(f"image_color_order 只支持 'rgb' 或 'bgr'，实际为 {color_order!r}")


def state8_from_openvitac_observation(observation: dict[str, Any]) -> np.ndarray:
    """从 OpenViTac observation 提取 abs_joint 8D 状态。

    输入:
        observation: task._get_observations() 返回的 observation dict。

    输出:
        np.ndarray，dtype float32，shape [8]。
        内容为 [7 个 Franka arm joint qpos, left finger gripper qpos]。
    """

    try:
        joint = observation["embodiment"]["joint"]
    except KeyError as exc:
        raise KeyError("observation 缺少 embodiment/joint，无法构造 OpenPI abs_joint state。") from exc

    joint_np = _to_numpy(joint).reshape(-1)
    if joint_np.shape[0] < 8:
        raise ValueError(f"embodiment/joint 至少需要 8 维，实际 shape={joint_np.shape}")
    state = joint_np[:8].astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"OpenPI state 中包含 NaN 或 Inf: {state}")
    return state


def state8_eef_quat_xyzw_from_openvitac_observation(observation: dict[str, Any]) -> np.ndarray:
    """从 OpenViTac observation 提取 EEF quaternion 8D 状态。

    输入:
        observation: task._get_observations() 返回的 observation dict。

    输出:
        np.ndarray，dtype float32，shape [8]。
        内容为 [ee_pos(3), ee_quat_xyzw(4), gripper_qpos(1)]。

    说明:
        OpenViTac 的 embodiment/ee 是 [x,y,z,qw,qx,qy,qz]。
        OpenPI server 期望 [x,y,z,qx,qy,qz,qw,gripper]。
    """

    try:
        ee = observation["embodiment"]["ee"]
    except KeyError as exc:
        raise KeyError("observation 缺少 embodiment/ee，无法构造 OpenPI delta_eef state。") from exc

    ee_np = _to_numpy(ee).reshape(-1).astype(np.float32)
    if ee_np.shape[0] != 7:
        raise ValueError(f"embodiment/ee 必须是 7 维 [pos,quat_wxyz]，实际 shape={ee_np.shape}")
    if not np.all(np.isfinite(ee_np)):
        raise ValueError(f"EEF state 中包含 NaN 或 Inf: {ee_np}")

    quat_wxyz = ee_np[3:7].astype(np.float32)
    quat_norm = float(np.linalg.norm(quat_wxyz))
    if quat_norm <= 1e-8:
        raise ValueError(f"EEF quaternion 非法，norm={quat_norm}: {quat_wxyz}")
    quat_wxyz = quat_wxyz / quat_norm
    quat_xyzw = np.asarray(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float32,
    )

    gripper = state8_from_openvitac_observation(observation)[-1:]
    state = np.concatenate([ee_np[:3], quat_xyzw, gripper], axis=0).astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"OpenPI delta_eef state 中包含 NaN 或 Inf: {state}")
    return state


def state10_eef_rot6d_from_openvitac_observation(observation: dict[str, Any]) -> np.ndarray:
    """从 OpenViTac observation 提取 EEF rot6d 10D 状态。

    输出内容为 [ee_pos(3), ee_rot6d(6), gripper_qpos(1)]。
    输入 observation["embodiment"]["ee"] 是 [x,y,z,qw,qx,qy,qz]。
    """

    try:
        ee = observation["embodiment"]["ee"]
    except KeyError as exc:
        raise KeyError("observation 缺少 embodiment/ee，无法构造 OpenPI delta_eef state。") from exc

    ee_np = _to_numpy(ee).reshape(-1).astype(np.float32)
    if ee_np.shape[0] != 7:
        raise ValueError(f"embodiment/ee 必须是 7 维 [pos,quat_wxyz]，实际 shape={ee_np.shape}")
    if not np.all(np.isfinite(ee_np)):
        raise ValueError(f"EEF state 中包含 NaN 或 Inf: {ee_np}")

    quat_wxyz = ee_np[3:7].astype(np.float32)
    quat_norm = float(np.linalg.norm(quat_wxyz))
    if quat_norm <= 1e-8:
        raise ValueError(f"EEF quaternion 非法，norm={quat_norm}: {quat_wxyz}")
    quat_wxyz = quat_wxyz / quat_norm

    rot = t3d.quaternions.quat2mat(quat_wxyz)
    rot6d = np.concatenate([rot[:, 0], rot[:, 1]], axis=0).astype(np.float32)
    gripper = state8_from_openvitac_observation(observation)[-1:]
    state = np.concatenate([ee_np[:3], rot6d, gripper], axis=0).astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"OpenPI delta_eef 10D state 中包含 NaN 或 Inf: {state}")
    return state


def openpi_obs_from_openvitac(
    observation: dict[str, Any],
    prompt: str,
    image_size: int,
    send_tactile: bool = True,
    tactile_image_keys: tuple[str, ...] | list[str] | None = None,
    image_color_order: str = "rgb",
    control_mode: str = "abs_joint",
    eef_state_mode: str = "quat_xyzw_8",
) -> dict[str, Any]:
    """把 OpenViTac observation 打包成 OpenPI server observation。

    输入:
        observation: task._get_observations() 返回的 observation dict。
        prompt: 发送给 OpenPI 的语言指令。
        image_size: 输出图像边长，通常为 224。
        send_tactile: 是否加入 left/right tactile 图像。
        tactile_image_keys: 触觉图像字段优先级。None 时按
            rgb_marker, gel_particle, force_field_img, marker_force_img, rgb
            自动选择，兼容 gelsight/xense/neote。
        image_color_order: 发送给 server 的图像通道顺序。"rgb" 表示保持 RGB；
            "bgr" 表示在发送前交换 R/B 通道，用于和 BGR 训练数据临时对齐。
        control_mode: "abs_joint"、"relative_joint" 或 "delta_eef"。
            "relative_joint" 和 "abs_joint" 发送相同的 8D absolute joint state。
            区别只在 server 返回 action 的解释方式。
        eef_state_mode: delta_eef state 表达，支持 "quat_xyzw_8" 或 "rot6d_10"。

    输出:
        dict，包含:
            observation/state: float32 [8]
            observation/image: uint8 [image_size,image_size,3]
            observation/wrist_image: uint8 [image_size,image_size,3]
            observation/left_tactile_image: uint8 [image_size,image_size,3]，可选
            observation/right_tactile_image: uint8 [image_size,image_size,3]，可选
            prompt: str
    """

    if control_mode in ("abs_joint", "relative_joint", "delta_joint"):
        state = state8_from_openvitac_observation(observation)
    elif control_mode == "delta_eef":
        if eef_state_mode == "quat_xyzw_8":
            state = state8_eef_quat_xyzw_from_openvitac_observation(observation)
        elif eef_state_mode == "rot6d_10":
            state = state10_eef_rot6d_from_openvitac_observation(observation)
        else:
            raise ValueError(
                "openpi.eef_state_mode 只支持 'quat_xyzw_8' 或 'rot6d_10'，"
                f"实际为 {eef_state_mode!r}"
            )
    else:
        raise ValueError(f"不支持的 OpenPI control_mode: {control_mode!r}")

    obs = {
        "observation/state": state,
        "observation/image": convert_image_color_order(
            resize_rgb_image(_get_camera_image(observation, "head"), image_size),
            image_color_order,
        ),
        "observation/wrist_image": convert_image_color_order(
            resize_rgb_image(_get_camera_image(observation, "wrist"), image_size),
            image_color_order,
        ),
        "prompt": prompt,
    }
    if send_tactile:
        tactile_image_keys = tuple(tactile_image_keys or DEFAULT_TACTILE_IMAGE_KEYS)
        obs["observation/left_tactile_image"] = convert_image_color_order(
            resize_rgb_image(
                _get_tactile_image(
                    observation,
                    sensor_candidates=("left_tactile", "left_gsmini"),
                    image_key_candidates=tactile_image_keys,
                ),
                image_size,
            ),
            image_color_order,
        )
        obs["observation/right_tactile_image"] = convert_image_color_order(
            resize_rgb_image(
                _get_tactile_image(
                    observation,
                    sensor_candidates=("right_tactile", "right_gsmini"),
                    image_key_candidates=tactile_image_keys,
                ),
                image_size,
            ),
            image_color_order,
        )
    return obs


def sanitize_abs_joint_action(action: np.ndarray | torch.Tensor, task: Any) -> torch.Tensor:
    """检查并裁剪 OpenPI 返回的 abs_joint 动作。

    输入:
        action: server 返回的单步动作，shape [8]。
        task: OpenViTac BaseTask 实例，用于获取 device 和 gripper_max_qpos。

    输出:
        torch.Tensor，dtype float32，shape [8]，device=task.device。

    行为:
        检查 action 为有限数；将 gripper qpos 裁剪到 [0, gripper_max_qpos]。
    """

    action_np = _to_numpy(action).reshape(-1).astype(np.float32)
    if action_np.shape[0] != 8:
        raise ValueError(f"abs_joint action 必须是 8D，实际 shape={action_np.shape}")
    if not np.all(np.isfinite(action_np)):
        raise ValueError(f"abs_joint action 中包含 NaN 或 Inf: {action_np}")

    gripper_max_qpos = float(getattr(task._robot_manager, "gripper_max_qpos", 0.039))
    action_np[-1] = float(np.clip(action_np[-1], 0.0, gripper_max_qpos))
    return torch.as_tensor(action_np, dtype=torch.float32, device=task.device)


def relative_joint_chunk_to_abs_qpos8(
    chunk: np.ndarray | torch.Tensor,
    base_qpos8: np.ndarray | torch.Tensor,
    gripper_max_qpos: float = 0.039,
    max_joint_delta: float | None = None,
) -> np.ndarray:
    """把 server 返回的 relative-joint chunk 转成可执行的 absolute qpos8 chunk。

    输入:
        chunk: server 返回的 action chunk，shape [T,8]。
            语义为 [target_joint7 - observation/state[:7], target_gripper_q]。
        base_qpos8: 这次 infer 时发送的 observation/state，shape [8]。
            所有 chunk 内的 delta 都相对这个 base，而不是相对执行时的最新状态。
        gripper_max_qpos: gripper qpos 裁剪上限。
        max_joint_delta: 可选，relative joint 每一维的绝对值安全裁剪上限，单位 rad。

    输出:
        np.ndarray，dtype float64，shape [T,8]。
        语义为 [target_joint7, target_gripper_q]，可按 qpos 下发。
    """

    chunk_np = _to_numpy(chunk).astype(np.float64)
    if chunk_np.ndim != 2 or chunk_np.shape[1] != 8:
        raise ValueError(f"relative_joint chunk 必须是 [T,8]，实际 shape={chunk_np.shape}")
    if not np.all(np.isfinite(chunk_np)):
        raise ValueError("relative_joint chunk 中包含 NaN 或 Inf。")

    base_np = _to_numpy(base_qpos8).reshape(-1).astype(np.float64)
    if base_np.shape[0] != 8:
        raise ValueError(f"relative_joint base_qpos8 必须是 8D，实际 shape={base_np.shape}")
    if not np.all(np.isfinite(base_np)):
        raise ValueError(f"relative_joint base_qpos8 中包含 NaN 或 Inf: {base_np}")

    if max_joint_delta is not None:
        max_joint_delta = float(max_joint_delta)
        if max_joint_delta <= 0:
            raise ValueError(f"max_joint_delta 必须为正数，实际为 {max_joint_delta}")
        chunk_np[:, :7] = np.clip(chunk_np[:, :7], -max_joint_delta, max_joint_delta)

    qpos_chunk = np.empty_like(chunk_np, dtype=np.float64)
    qpos_chunk[:, :7] = base_np[:7][None, :] + chunk_np[:, :7]
    qpos_chunk[:, 7] = np.clip(chunk_np[:, 7], 0.0, float(gripper_max_qpos))
    return np.ascontiguousarray(qpos_chunk, dtype=np.float64)


def delta_joint_chunk_to_abs_qpos8(
    chunk: np.ndarray | torch.Tensor,
    base_qpos8: np.ndarray | torch.Tensor,
    gripper_max_qpos: float = 0.039,
    max_joint_delta: float | None = None,
) -> np.ndarray:
    """Backward-compatible alias for relative_joint_chunk_to_abs_qpos8."""

    return relative_joint_chunk_to_abs_qpos8(
        chunk,
        base_qpos8,
        gripper_max_qpos=gripper_max_qpos,
        max_joint_delta=max_joint_delta,
    )


def sanitize_delta_eef_action(
    action: np.ndarray | torch.Tensor,
    task: Any,
    max_position_delta: float | None = None,
    max_rotation_delta: float | None = None,
) -> torch.Tensor:
    """检查 OpenPI 返回的 delta_eef 动作，并把绝对 gripper 转成底层 delta。

    输入:
        action: server 返回的单步动作，shape [7]。
            语义为 [delta_xyz(3), delta_rotvec(3), gripper_abs_qpos(1)]。
            delta_rotvec 满足 log(R_target * R_current^-1)，单位 rad。
        task: OpenViTac BaseTask 实例，用于获取当前 gripper qpos、device 和 gripper_max_qpos。
        max_position_delta: 可选，单步 xyz delta 的绝对值裁剪上限，单位 m。
        max_rotation_delta: 可选，单步 rotvec 各分量的绝对值裁剪上限，单位 rad。

    输出:
        torch.Tensor，dtype float32，shape [7]，device=task.device。
        语义为 OpenViTac take_action(action_type="delta_ee_rotvec_ik") 需要的
        [delta_xyz(3), delta_rotvec(3), delta_gripper_qpos(1)]。
    """

    action_np = _to_numpy(action).reshape(-1).astype(np.float32)
    if action_np.shape[0] != 7:
        raise ValueError(f"delta_eef action 必须是 7D，实际 shape={action_np.shape}")
    if not np.all(np.isfinite(action_np)):
        raise ValueError(f"delta_eef action 中包含 NaN 或 Inf: {action_np}")

    if max_position_delta is not None:
        max_position_delta = float(max_position_delta)
        if max_position_delta <= 0:
            raise ValueError(f"max_position_delta 必须为正数，实际为 {max_position_delta}")
        action_np[:3] = np.clip(action_np[:3], -max_position_delta, max_position_delta)

    if max_rotation_delta is not None:
        max_rotation_delta = float(max_rotation_delta)
        if max_rotation_delta <= 0:
            raise ValueError(f"max_rotation_delta 必须为正数，实际为 {max_rotation_delta}")
        action_np[3:6] = np.clip(action_np[3:6], -max_rotation_delta, max_rotation_delta)

    gripper_max_qpos = float(getattr(task._robot_manager, "gripper_max_qpos", 0.039))
    current_gripper_qpos = float(task._robot_manager.get_gripper_qpos())
    target_gripper_qpos = float(np.clip(action_np[-1], 0.0, gripper_max_qpos))
    action_np[-1] = target_gripper_qpos - current_gripper_qpos
    return torch.as_tensor(action_np, dtype=torch.float32, device=task.device)


def _get_camera_image(observation: dict[str, Any], name: str) -> Any:
    try:
        return observation["observation"][name]["rgb"]
    except KeyError as exc:
        raise KeyError(f"observation 缺少 observation/{name}/rgb。") from exc


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
        "observation 缺少 tactile "
        f"{sensor_candidates} 中任一图像字段 {image_key_candidates}。"
    )


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"图像必须是 3 维数组，实际 shape={array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"图像最后一维必须为 3 通道，实际 shape={array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resize_direct_cv2(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)
