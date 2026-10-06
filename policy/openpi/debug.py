from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


OPENPI_IMAGE_OBS_KEYS = (
    "observation/image",
    "observation/wrist_image",
    "observation/left_tactile_image",
    "observation/right_tactile_image",
)


def save_openpi_observation_images(
    obs: dict[str, Any],
    output_dir: str | Path,
    step_index: int,
) -> list[Path]:
    """保存本次真正发给 OpenPI server 的 observation 图像。

    输入:
        obs: openpi_obs_from_openvitac() 生成的 observation dict。
        output_dir: 图片保存目录。
        step_index: 当前 policy step，用于生成文件名前缀。

    输出:
        list[Path]，实际写出的图片路径。

    说明:
        只保存一份发送给 server 的原始数组，不再额外写 RGB/BGR 对照图。
        文件内容保持和发送时一致，具体通道语义由当前 deploy 配置决定。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    prefix = f"{step_index:06d}"
    for obs_key in OPENPI_IMAGE_OBS_KEYS:
        if obs_key not in obs:
            continue

        image = _as_uint8_hwc3(obs[obs_key], obs_key)
        stem = obs_key.removeprefix("observation/").replace("/", "_")
        image_path = output_dir / f"{prefix}_{stem}_sent.png"
        Image.fromarray(image, mode="RGB").save(image_path)
        saved_paths.append(image_path)

    state = obs.get("observation/state")
    if state is not None:
        state_path = output_dir / f"{prefix}_state.txt"
        np.savetxt(state_path, np.asarray(state, dtype=np.float32).reshape(1, -1), fmt="%.8f")
        saved_paths.append(state_path)

    return saved_paths


def _as_uint8_hwc3(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} 必须是 HWC 3 通道图像，实际 shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)
