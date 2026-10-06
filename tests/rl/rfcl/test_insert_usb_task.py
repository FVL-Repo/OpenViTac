from __future__ import annotations

from types import SimpleNamespace

import pytest

from policy.RL.task_factory import _resolve_task_type
from policy.RL.tasks.insert_usb import (
    BALANCED_40_PLAN_ID,
    InsertUSBRLTaskMixin,
    balanced_40_profiles,
    get_balanced_40_profile,
)


def test_balanced_40_profiles_are_complete_and_unique() -> None:
    profiles = balanced_40_profiles()

    assert BALANCED_40_PLAN_ID == "insert_usb_balanced40_v1"
    assert len(profiles) == 40
    assert len({profile.profile_id for profile in profiles}) == 40
    assert {profile.profile_id for profile in profiles} == {
        f"usb40_{index:02d}" for index in range(1, 41)
    }


def test_balanced_40_profiles_cover_expected_families_and_ranges() -> None:
    profiles = balanced_40_profiles()

    assert {profile.family for profile in profiles} == {
        "center",
        "positive_long",
        "negative_long",
        "positive_short",
        "negative_short",
    }
    assert {profile.trajectory_template for profile in profiles} == {
        "direct",
        "inner_wall",
        "rim_recovery",
        "contact_rich",
    }
    assert max(abs(profile.long_offset_m) for profile in profiles) == pytest.approx(
        0.004
    )
    assert max(abs(profile.short_offset_m) for profile in profiles) == pytest.approx(
        0.004
    )
    assert sorted({profile.clearance_m for profile in profiles}) == pytest.approx(
        [0.006, 0.010, 0.014]
    )


def test_get_balanced_40_profile() -> None:
    profile = get_balanced_40_profile("usb40_40")

    assert profile.family == "negative_short"
    assert profile.trajectory_template == "contact_rich"
    with pytest.raises(KeyError, match="Unknown Insert USB demo profile"):
        get_balanced_40_profile("missing")


def test_rfcl_task_variant_is_explicit_and_isolated() -> None:
    class OriginalTask:
        def __init__(self, config=None):
            self.config = config
            self.original_initialized = True

    task_module = SimpleNamespace(Task=OriginalTask)

    assert _resolve_task_type("insert_USB", task_module, None) is OriginalTask
    rl_task_type = _resolve_task_type("insert_USB", task_module, "rl")
    rfcl_task_type = _resolve_task_type("insert_USB", task_module, "rfcl")
    config = SimpleNamespace()
    rl_task = rl_task_type(config)
    rfcl_task = rfcl_task_type(config)

    assert isinstance(rl_task, OriginalTask)
    assert isinstance(rfcl_task, OriginalTask)
    assert rl_task.original_initialized
    assert rfcl_task.original_initialized
    assert rfcl_task.rfcl_demo_profile is None
    assert hasattr(rfcl_task, "set_rfcl_demo_profile")
    assert not hasattr(OriginalTask, "set_rfcl_demo_profile")


def test_rfcl_task_variant_rejects_unsupported_tasks() -> None:
    task_module = SimpleNamespace(Task=object)

    with pytest.raises(ValueError, match="Unknown task variant"):
        _resolve_task_type("insert_USB", task_module, "unknown")
    with pytest.raises(ValueError, match="not implemented"):
        _resolve_task_type("insert_block", task_module, "rfcl")


def test_fixed_target_slot_wraps_original_reset() -> None:
    class FakePose:
        def __init__(self, position, rotation=None):
            self.position = list(position)
            self.rotation = rotation

        def __getitem__(self, index):
            return self.position[index]

        def tolist(self):
            return list(self.position)

    class FakeActor:
        def __init__(self, pose):
            self.pose = pose

        def get_pose(self):
            return self.pose

        def set_pose(self, pose):
            self.pose = pose

    class OriginalTask:
        def __init__(self, config, *args, **kwargs):
            self.slot = FakeActor(FakePose([0.529, -0.008, 0.002]))
            self.metadata = {}
            self.original_reset_calls = 0
            self.reference_pose_updates = 0

        def _reset_actors(self):
            self.original_reset_calls += 1
            self.slot.set_pose(FakePose([0.527, 0.006, 0.002]))
            self.metadata["target_slot_pose"] = self.slot.get_pose().tolist()

        def _update_insert_reference_poses(self):
            self.reference_pose_updates += 1

    class RLTask(InsertUSBRLTaskMixin, OriginalTask):
        _rfcl_source_module = SimpleNamespace(Pose=FakePose)

    random_task = RLTask(SimpleNamespace(), fixed_target_slot=False)
    random_task._reset_actors()
    assert random_task.original_reset_calls == 1
    assert random_task.slot.get_pose().tolist() == [0.527, 0.006, 0.002]
    assert random_task.reference_pose_updates == 0

    fixed_task = RLTask(SimpleNamespace(), fixed_target_slot=True)
    fixed_task._reset_actors()
    assert fixed_task.original_reset_calls == 1
    assert fixed_task.slot.get_pose().tolist() == [0.52, 0.0, 0.002]
    assert fixed_task.reference_pose_updates == 1
    assert fixed_task.metadata["target_slot_pose"] == [0.52, 0.0, 0.002]
    assert fixed_task.metadata["sampled_target_slot_pose"] == [
        0.527,
        0.006,
        0.002,
    ]
