"""Prefabricated task spec instances, one module per task (paper-abbr-aligned)."""

from .gear_assembly import GearAssembly
from .hardness_classification import HardnessClassification
from .insert_block import InsertBlock
from .insert_USB import InsertUsb
from .pull_drawer import PullDrawer
from .roughness_classification import RoughnessClassification
from .roughness_guided_regrasp import RoughnessGuidedRegrasp
from .weight_classification import WeightClassification

__all__ = [
    "GearAssembly",
    "HardnessClassification",
    "InsertBlock",
    "InsertUsb",
    "PullDrawer",
    "RoughnessClassification",
    "RoughnessGuidedRegrasp",
    "WeightClassification",
]
