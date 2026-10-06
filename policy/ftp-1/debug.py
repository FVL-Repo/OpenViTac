from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FTP_IMAGE_KEYS = (
    "camera_ego_rgb",
    "right_wrist_camera_rgb",
)


def save_ftp_observation(
    obs: dict[str, Any],
    output_dir: str | Path,
    step_index: int,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    prefix = f"{step_index:06d}"
    for key in FTP_IMAGE_KEYS:
        if key not in obs:
            continue
        image = _as_uint8_hwc3(obs[key], key)
        path = output_dir / f"{prefix}_{key}_sent.png"
        Image.fromarray(image, mode="RGB").save(path)
        saved_paths.append(path)

    tactile = obs.get("right_tactile_data_gripper")
    if tactile is not None:
        tactile_array = np.asarray(tactile)
        if tactile_array.ndim != 4 or tactile_array.shape[0] < 2:
            raise ValueError(
                "right_tactile_data_gripper must be [2,H,W,3], "
                f"got shape={tactile_array.shape}"
            )
        for index, name in enumerate(("left_tactile", "right_tactile")):
            image = _as_uint8_hwc3(tactile_array[index], name)
            path = output_dir / f"{prefix}_{name}_sent.png"
            Image.fromarray(image, mode="RGB").save(path)
            saved_paths.append(path)

    qpos = obs.get("qpos")
    if qpos is not None:
        path = output_dir / f"{prefix}_qpos.txt"
        np.savetxt(path, np.asarray(qpos, dtype=np.float32).reshape(1, -1), fmt="%.8f")
        saved_paths.append(path)

    prompt = obs.get("prompt")
    if prompt is not None:
        path = output_dir / f"{prefix}_prompt.txt"
        path.write_text(str(prompt), encoding="utf-8")
        saved_paths.append(path)

    return saved_paths


def _as_uint8_hwc3(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be HWC 3-channel image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)
