"""openvitac_tasks: prefabricated task specs and the task registry.

Pure data only: no isaaclab, pxr, pyuipc, or sim imports.  Registry names align
with the paper abbreviations; class-B tasks (ECS/GC) remain out of scope.
"""

from .base import (
    EpisodeRandomization,
    Task,
)
from .expert import (
    Action,
    ExpertStateMachine,
    ExpertWorld,
    GripperHandle,
    Quat,
    SuccessFn,
    TaskLogger,
    Vec3,
)
from .prefabricated import TASKS, TaskName
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

__all__ = [
    "TASKS",
    "Action",
    "EpisodeRandomization",
    "ExpertStateMachine",
    "ExpertWorld",
    "GripperHandle",
    "HardnessClassify",
    "InsertBlock",
    "InsertBlockV2",
    "InsertUsb",
    "PullDrawer",
    "Quat",
    "RoughnessClassify",
    "RoughnessRegrasp",
    "SuccessFn",
    "Task",
    "TaskLogger",
    "TaskName",
    "TurnGearPair",
    "Vec3",
    "WeightClassify",
]
