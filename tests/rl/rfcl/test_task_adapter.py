import numpy as np
import pytest

from policy.RL.rfcl import PRIVILEGED_STATE_LAYOUT
from policy.RL.rfcl_task_adapter import (
    InsertUSBTaskAdapter,
    create_rfcl_task_adapter,
)


class FakePose:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeActor:
    def __init__(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose


class FakeInsertUSBTask:
    def __init__(self):
        self.prism = FakeActor(FakePose([0.5, 0.0, 0.02, 1, 0, 0, 0]))
        self.slot = FakeActor(FakePose([0.52, 0.0, 0.002, 1, 0, 0, 0]))
        self.handoff_calls = 0

    def _usb_pose_in_slot(self, slot_pose):
        values = slot_pose.tolist()
        values[2] += 0.012
        return FakePose(values)

    def _prepare_usb_standard(self):
        self.handoff_calls += 1

    def check_success(self):
        return False


def test_insert_usb_adapter_builds_generic_entities_and_state():
    adapter = InsertUSBTaskAdapter()
    task = FakeInsertUSBTask()
    entities = adapter.capture_entities(task)
    assert set(entities) == {"controlled", "target", "slot"}
    assert entities["target"][2] == pytest.approx(0.014)

    state = adapter.build_privileged_state(
        physics_state={
            "joint_pos": np.zeros(9),
            "joint_vel": np.ones(9),
            "ee": np.asarray([0, 0, 0.2, 1, 0, 0, 0]),
        },
        entities=entities,
    )
    assert state.shape == (adapter.state_dim,)
    np.testing.assert_allclose(
        state[PRIVILEGED_STATE_LAYOUT.joint_delta],
        np.ones(9),
    )


def test_adapter_factory_is_explicit_and_handoff_is_task_owned():
    adapter = create_rfcl_task_adapter(task_name="insert_USB")
    task = FakeInsertUSBTask()
    adapter.prepare_handoff(task)
    assert task.handoff_calls == 1
    assert adapter.manifest_metadata()["adapter_id"] == "insert_usb_v2"
    with pytest.raises(KeyError, match="No RFCL adapter"):
        create_rfcl_task_adapter(task_name="insert_block")


def test_insert_usb_policy_entry_ignores_coarse_handoff_settling():
    adapter = InsertUSBTaskAdapter()
    assert not adapter.is_policy_entry("move_usb_to_pre_insert")
    assert not adapter.is_policy_entry("delay")
    assert adapter.is_policy_entry("rfcl_free_align")
    assert adapter.is_policy_entry("rfcl_inner_wall_shallow_insert")
    assert adapter.is_policy_entry("move_usb_to_play_pre_insert")
