"""Insert USB task extensions for RFCL demonstration generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


BALANCED_40_PLAN_ID = "insert_usb_balanced40_v1"


@dataclass(frozen=True)
class InsertUSBDemoProfile:
    profile_id: str
    family: str
    trajectory_template: str
    long_offset_m: float
    short_offset_m: float
    clearance_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _profile(
    index: int,
    family: str,
    template: str,
    long_mm: float,
    short_mm: float,
    clearance_mm: float,
) -> InsertUSBDemoProfile:
    return InsertUSBDemoProfile(
        profile_id=f"usb40_{index:02d}",
        family=family,
        trajectory_template=template,
        long_offset_m=float(long_mm) / 1000.0,
        short_offset_m=float(short_mm) / 1000.0,
        clearance_m=float(clearance_mm) / 1000.0,
    )


BALANCED_40_PROFILES = (
    _profile(1, "center", "direct", 0.0, 0.0, 6.0),
    _profile(2, "center", "direct", 0.0, 0.0, 10.0),
    _profile(3, "center", "direct", 0.0, 0.0, 14.0),
    _profile(4, "center", "inner_wall", 0.6, 0.0, 6.0),
    _profile(5, "center", "inner_wall", -0.6, 0.0, 10.0),
    _profile(6, "center", "inner_wall", 0.0, 0.6, 10.0),
    _profile(7, "center", "inner_wall", 0.0, -0.6, 14.0),
    _profile(8, "center", "contact_rich", 0.0, 0.6, 10.0),
    _profile(9, "positive_long", "direct", 2.0, -1.0, 6.0),
    _profile(10, "positive_long", "direct", 2.0, 1.0, 6.0),
    _profile(11, "positive_long", "direct", 2.0, -1.0, 10.0),
    _profile(12, "positive_long", "rim_recovery", 2.0, 1.0, 10.0),
    _profile(13, "positive_long", "direct", 4.0, -1.0, 10.0),
    _profile(14, "positive_long", "rim_recovery", 4.0, 1.0, 10.0),
    _profile(15, "positive_long", "direct", 4.0, -1.0, 14.0),
    _profile(16, "positive_long", "contact_rich", 4.0, 1.0, 14.0),
    _profile(17, "negative_long", "direct", -2.0, -1.0, 6.0),
    _profile(18, "negative_long", "direct", -2.0, 1.0, 6.0),
    _profile(19, "negative_long", "direct", -2.0, -1.0, 10.0),
    _profile(20, "negative_long", "rim_recovery", -2.0, 1.0, 10.0),
    _profile(21, "negative_long", "direct", -4.0, -1.0, 10.0),
    _profile(22, "negative_long", "rim_recovery", -4.0, 1.0, 10.0),
    _profile(23, "negative_long", "direct", -4.0, -1.0, 14.0),
    _profile(24, "negative_long", "contact_rich", -4.0, 1.0, 14.0),
    _profile(25, "positive_short", "direct", -1.0, 2.0, 6.0),
    _profile(26, "positive_short", "direct", 1.0, 2.0, 6.0),
    _profile(27, "positive_short", "direct", -1.0, 2.0, 10.0),
    _profile(28, "positive_short", "rim_recovery", 1.0, 2.0, 10.0),
    _profile(29, "positive_short", "direct", -1.0, 4.0, 10.0),
    _profile(30, "positive_short", "rim_recovery", 1.0, 4.0, 10.0),
    _profile(31, "positive_short", "direct", -1.0, 4.0, 14.0),
    _profile(32, "positive_short", "contact_rich", 1.0, 4.0, 14.0),
    _profile(33, "negative_short", "direct", -1.0, -2.0, 6.0),
    _profile(34, "negative_short", "direct", 1.0, -2.0, 6.0),
    _profile(35, "negative_short", "direct", -1.0, -2.0, 10.0),
    _profile(36, "negative_short", "rim_recovery", 1.0, -2.0, 10.0),
    _profile(37, "negative_short", "direct", -1.0, -4.0, 10.0),
    _profile(38, "negative_short", "rim_recovery", 1.0, -4.0, 10.0),
    _profile(39, "negative_short", "direct", -1.0, -4.0, 14.0),
    _profile(40, "negative_short", "contact_rich", 1.0, -4.0, 14.0),
)


def balanced_40_profiles() -> tuple[InsertUSBDemoProfile, ...]:
    return BALANCED_40_PROFILES


def get_balanced_40_profile(profile_id: str) -> InsertUSBDemoProfile:
    for profile in BALANCED_40_PROFILES:
        if profile.profile_id == str(profile_id):
            return profile
    raise KeyError(f"Unknown Insert USB demo profile: {profile_id!r}")


class InsertUSBRLTaskMixin:
    """RL-only initialization controls layered over the original task."""

    _rfcl_source_module: Any

    def __init__(self, cfg, *args, fixed_target_slot=False, **kwargs):
        self._rl_fixed_target_slot = bool(fixed_target_slot)
        super().__init__(cfg, *args, **kwargs)

    def _reset_actors(self):
        super()._reset_actors()
        if not self._rl_fixed_target_slot:
            return

        sampled_target_slot_pose = self.slot.get_pose()
        fixed_target_slot_pose = self._rfcl_source_module.Pose(
            [0.52, 0.0, self.slot.get_pose()[2]],
            [1, 0, 0, 0],
        )
        self.slot.set_pose(fixed_target_slot_pose)
        self._update_insert_reference_poses()
        self.metadata["target_slot_pose"] = fixed_target_slot_pose.tolist()
        self.metadata["fixed_target_slot"] = True
        self.metadata["sampled_target_slot_pose"] = (
            sampled_target_slot_pose.tolist()
        )


class InsertUSBRFCLTaskMixin(InsertUSBRLTaskMixin):
    """Balanced demonstration behavior layered over the original task."""

    _rfcl_source_module: Any

    def __init__(self, *args, **kwargs):
        self.rfcl_demo_profile: dict[str, object] | None = None
        super().__init__(*args, **kwargs)

    def set_rfcl_demo_profile(self, profile: InsertUSBDemoProfile | dict | None):
        if profile is None:
            self.rfcl_demo_profile = None
            return
        if hasattr(profile, "to_dict"):
            profile = profile.to_dict()
        required = {
            "profile_id",
            "family",
            "trajectory_template",
            "long_offset_m",
            "short_offset_m",
            "clearance_m",
        }
        missing = required.difference(profile)
        if missing:
            raise ValueError(f"RFCL demo profile is missing fields: {sorted(missing)}")
        profile = dict(profile)
        if profile["trajectory_template"] not in {
            "direct",
            "rim_recovery",
            "inner_wall",
            "contact_rich",
        }:
            raise ValueError(
                "Unsupported Insert USB trajectory template: "
                f"{profile['trajectory_template']!r}"
            )
        for field in ("long_offset_m", "short_offset_m", "clearance_m"):
            profile[field] = float(profile[field])
            if not np.isfinite(profile[field]):
                raise ValueError(f"{field} must be finite")
        if profile["clearance_m"] < 0.0:
            raise ValueError("clearance_m must be non-negative")
        self.rfcl_demo_profile = profile

    def _prepare_usb_standard(self):
        if self.rfcl_demo_profile is None:
            return super()._prepare_usb_standard()

        source = self._rfcl_source_module
        profile = self.rfcl_demo_profile
        grasp_rotate = self.rng.uniform(-np.pi / 18, np.pi / 18)
        grasp_height = source.USB_GRASP_HEIGHT + self.rng.uniform(
            -source.USB_GRASP_HEIGHT_NOISE,
            source.USB_GRASP_HEIGHT_NOISE,
        )
        target_pose = (
            self.prism.get_pose()
            .add_bias([0, 0, grasp_height])
            .add_rotation([0, grasp_rotate, 0])
        )
        target_mat = target_pose.to_transformation_matrix()
        contact_pose = source.construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        contact_id = self.prism.register_point(contact_pose, type="contact")
        self.move(
            self.atom.grasp_actor(
                self.prism,
                contact_point_id=contact_id,
                is_close=False,
            ),
            tag="approach_usb",
        )
        self.move(self.atom.close_gripper(), tag="close_usb")

        lift_height = source.LIFT_HEIGHT + self.rng.uniform(
            -source.LIFT_HEIGHT_NOISE,
            source.LIFT_HEIGHT_NOISE,
        )
        self.move(self.atom.move_by_displacement(z=lift_height), tag="lift_usb")

        self._update_insert_reference_poses()
        approach_clearance = float(profile["clearance_m"])
        self._update_pre_insert_pose(approach_clearance)
        approach_offset = source.Pose(
            [
                float(profile["long_offset_m"]),
                float(profile["short_offset_m"]),
                0.0,
            ]
        )
        approach_pose = self.pre_insert_pose.add_offset(approach_offset)
        self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=approach_pose,
                pre_dis=0.02,
                dis=0.004,
                is_open=False,
            ),
            tag="move_usb_to_pre_insert",
        )
        self._move_held_usb_by_translation(
            approach_pose,
            tag="move_usb_to_pre_insert",
            time_dilation_factor=0.5,
            constraint_pose=[1, 1, 1, 1, 1, 0],
        )

        self.metadata["grasp_rotate"] = float(grasp_rotate)
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["lift_height"] = float(lift_height)
        self.metadata["approach_clearance"] = approach_clearance
        self.metadata["approach_xy_noise"] = approach_offset.p.tolist()
        self.metadata["rfcl_demo_profile"] = dict(profile)

    def _rfcl_demo_waypoint(self, long_offset, short_offset, clearance, tag):
        target_pose = self.opening_pose.add_bias(
            [float(long_offset), float(short_offset), float(clearance)]
        )
        return self._move_held_usb_by_translation(
            target_pose,
            tag=tag,
            time_dilation_factor=0.5,
            constraint_pose=[1, 1, 1, 1, 1, 0],
        )

    @staticmethod
    def _rfcl_profile_opposite_inner_offset(profile):
        long_offset = float(profile["long_offset_m"])
        short_offset = float(profile["short_offset_m"])
        if abs(long_offset) >= abs(short_offset):
            return (-np.copysign(0.0006, long_offset), 0.0)
        return (0.0, -np.copysign(0.0006, short_offset))

    def _execute_rfcl_demo_profile(self, profile):
        template = str(profile["trajectory_template"])
        long_offset = float(profile["long_offset_m"])
        short_offset = float(profile["short_offset_m"])
        clearance = float(profile["clearance_m"])

        if template == "direct":
            self._rfcl_demo_waypoint(0.0, 0.0, clearance, "rfcl_free_align")
        elif template == "rim_recovery":
            self._rfcl_demo_waypoint(
                long_offset, short_offset, -0.0005, "rfcl_outer_rim_touch"
            )
            self.delay(4, is_save=True)
            self._rfcl_demo_waypoint(
                long_offset, short_offset, 0.003, "rfcl_outer_rim_retract"
            )
            self._rfcl_demo_waypoint(0.0, 0.0, 0.003, "rfcl_recovery_align")
        elif template == "inner_wall":
            self._rfcl_demo_waypoint(
                long_offset,
                short_offset,
                -0.00025,
                "rfcl_inner_wall_shallow_insert",
            )
            self.delay(4, is_save=True)
            self._rfcl_demo_waypoint(
                long_offset, short_offset, 0.002, "rfcl_inner_wall_retract"
            )
            self._rfcl_demo_waypoint(
                0.0, 0.0, 0.002, "rfcl_inner_wall_correction"
            )
        elif template == "contact_rich":
            if max(abs(long_offset), abs(short_offset)) > 0.001:
                self._rfcl_demo_waypoint(
                    long_offset,
                    short_offset,
                    -0.0005,
                    "rfcl_contact_rich_outer_touch",
                )
                self.delay(4, is_save=True)
                self._rfcl_demo_waypoint(
                    long_offset,
                    short_offset,
                    0.003,
                    "rfcl_contact_rich_retract",
                )
            opposite_long, opposite_short = self._rfcl_profile_opposite_inner_offset(
                profile
            )
            self._rfcl_demo_waypoint(
                opposite_long,
                opposite_short,
                0.003,
                "rfcl_contact_rich_cross_align",
            )
            self._rfcl_demo_waypoint(
                opposite_long,
                opposite_short,
                -0.00025,
                "rfcl_contact_rich_inner_touch",
            )
            self.delay(4, is_save=True)
            self._rfcl_demo_waypoint(
                opposite_long,
                opposite_short,
                0.002,
                "rfcl_contact_rich_inner_retract",
            )
            self._rfcl_demo_waypoint(
                0.0, 0.0, 0.002, "rfcl_contact_rich_final_align"
            )

        self._rfcl_demo_waypoint(
            0.0,
            0.0,
            self._rfcl_source_module.PLAY_PRE_INSERT_CLEARANCE,
            "move_usb_to_play_pre_insert",
        )

    def _play_once(self):
        if self.rfcl_demo_profile is None:
            return super()._play_once()

        source = self._rfcl_source_module
        self._prepare_usb_standard()
        self._update_insert_reference_poses()
        self._execute_rfcl_demo_profile(self.rfcl_demo_profile)
        self.metadata["play_pre_insert_clearance"] = (
            source.PLAY_PRE_INSERT_CLEARANCE
        )
        insert_distance = max(
            0.0,
            float(self.prism.get_pose().p[2] - self.target_pose.p[2]),
        )
        self.metadata["insert_distance"] = insert_distance
        self.move(
            self.atom.move_by_displacement(
                z=-insert_distance,
                xyz_coord="world",
            ),
            tag="insert_USB_into_slot",
            time_dilation_factor=0.5,
            constraint_pose=[1, 1, 1, 1, 1, 0],
        )
        self.delay(40, is_save=True)


def build_insert_usb_rfcl_task_type(base_task_type, source_module):
    """Return an RFCL-only subclass without modifying the source task module."""

    class InsertUSBRFCLTask(InsertUSBRFCLTaskMixin, base_task_type):
        _rfcl_source_module = source_module

    InsertUSBRFCLTask.__module__ = __name__
    return InsertUSBRFCLTask


def build_insert_usb_rl_task_type(base_task_type, source_module):
    """Return an RL-only subclass without modifying the source task module."""

    class InsertUSBRLTask(InsertUSBRLTaskMixin, base_task_type):
        _rfcl_source_module = source_module

    InsertUSBRLTask.__module__ = __name__
    return InsertUSBRLTask
