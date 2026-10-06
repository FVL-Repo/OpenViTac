from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def save_internvla_observation(obs: dict[str, Any], output_dir: str | Path, step: int) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for key in ("head_image", "wrist_image"):
        if key not in obs:
            continue
        image = np.asarray(obs[key])
        path = out / f"{step:06d}_{key}.png"
        cv2.imwrite(str(path), image[..., ::-1])
        saved[key] = str(path)

    state_path = out / f"{step:06d}_state.npy"
    np.save(state_path, np.asarray(obs["state"], dtype=np.float32))
    saved["state"] = str(state_path)
    return saved
