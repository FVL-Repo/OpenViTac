"""Shared task names and paths for OpenViTac simulation training."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "training" / "openpi_runtime"
DATA = ROOT / "data" / "training"
WEIGHTS = ROOT / "weights"
OUTPUTS = ROOT / "outputs" / "training"

TASKS = {
    "wc": "Weight Classification",
    "hc": "Hardness Classification",
    "rc": "Roughness Classification",
    "rr": "Roughness-Guided Regrasp",
    "ecs": "Empty-Can Selection",
    "gc": "Grasp Chip",
    "ga": "Gear Assembly",
    "pd": "Pull Drawer",
    "in_usb": "Insert USB",
    "in_b_v1": "Insert Block v1",
    "in_b_v2": "Insert Block v2",
}
MODELS = ("pi05", "openvtla")
BATCH_SIZE = 64
ACTION_HORIZON = 50


def steps_for(task: str) -> int:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return 20_000 if task == "in_b_v2" else 10_000
