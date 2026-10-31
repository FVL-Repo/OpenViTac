from typing import Literal

from .base import Task
from .tasks import (
    HardnessClassify,
    InsertBlock,
    InsertBlockV2,
    InsertUsb,
    PullDrawer,
    RoughnessClassify,
    RoughnessRegrasp,
    TurnGearPair,
    WeightClassify,
)

TaskName = Literal[
    "hardness_classify",
    "insert_block",
    "insert_block_v2",
    "insert_usb",
    "pull_drawer",
    "roughness_classify",
    "roughness_regrasp",
    "turn_gear_pair",
    "weight_classify",
]

TASKS: dict[TaskName, Task] = {
    "hardness_classify": HardnessClassify,
    "insert_block": InsertBlock,
    "insert_block_v2": InsertBlockV2,
    "insert_usb": InsertUsb,
    "pull_drawer": PullDrawer,
    "roughness_classify": RoughnessClassify,
    "roughness_regrasp": RoughnessRegrasp,
    "turn_gear_pair": TurnGearPair,
    "weight_classify": WeightClassify,
}
