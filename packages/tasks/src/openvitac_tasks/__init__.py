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
    GearAssembly,
    HardnessClassification,
    InsertBlock,
    InsertUsb,
    PullDrawer,
    RoughnessClassification,
    RoughnessGuidedRegrasp,
    WeightClassification,
)

__all__ = [
    "TASKS",
    "Action",
    "EpisodeRandomization",
    "ExpertStateMachine",
    "ExpertWorld",
    "GearAssembly",
    "GripperHandle",
    "HardnessClassification",
    "InsertBlock",
    "InsertUsb",
    "PullDrawer",
    "Quat",
    "RoughnessClassification",
    "RoughnessGuidedRegrasp",
    "SuccessFn",
    "Task",
    "TaskLogger",
    "TaskName",
    "Vec3",
    "WeightClassification",
]
