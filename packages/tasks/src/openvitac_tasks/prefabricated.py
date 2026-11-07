from typing import Literal

from .base import Task
from .tasks import (
    GearAssembly,
    HardnessClassification,
    InsertBlock,
    InsertUsb,
    PullDrawer,
    RoughnessClassification,
    RoughnessGuidedRegrasp,
    WeightClassification,
)

TaskName = Literal[
    "gear_assembly",
    "hardness_classification",
    "insert_block",
    "insert_USB",
    "pull_drawer",
    "roughness_classification",
    "roughness_guided_regrasp",
    "weight_classification",
]

TASKS: dict[TaskName, Task] = {
    "gear_assembly": GearAssembly,
    "hardness_classification": HardnessClassification,
    "insert_block": InsertBlock,
    "insert_USB": InsertUsb,
    "pull_drawer": PullDrawer,
    "roughness_classification": RoughnessClassification,
    "roughness_guided_regrasp": RoughnessGuidedRegrasp,
    "weight_classification": WeightClassification,
}
