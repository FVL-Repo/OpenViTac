"""Prefabricated task spec instances, one module per task (paper-abbr-aligned)."""

from .hardness_classify import HardnessClassify
from .insert_block import InsertBlock, InsertBlockV2
from .insert_usb import InsertUsb
from .pull_drawer import PullDrawer
from .roughness_classify import RoughnessClassify
from .roughness_regrasp import RoughnessRegrasp
from .turn_gear_pair import TurnGearPair
from .weight_classify import WeightClassify

__all__ = [
    "HardnessClassify",
    "InsertBlock",
    "InsertBlockV2",
    "InsertUsb",
    "PullDrawer",
    "RoughnessClassify",
    "RoughnessRegrasp",
    "TurnGearPair",
    "WeightClassify",
]
