import numpy as np
import pytest
import torch
from torch import nn

import policy.RL.gym_env as gym_env
from policy.RL.gym_env import TactileControlEnv


class StubEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.qpos_dim = 7
        self.feature_dim = 7
        self.camera_keys = ("cam_high", "cam_wrist")
        self.tactile_keys = ("tac_left", "tac_right")


class StubActor(nn.Module):
    action_dim = 7

    def __init__(self, action):
        super().__init__()
        self.encoder = StubEncoder()
        self.register_buffer(
            "action",
            torch.as_tensor(action, dtype=torch.float32),
        )

    def deterministic_action(self, observation):
        batch_size = observation["qpos"].shape[0]
        return self.action.unsqueeze(0).repeat(batch_size, 1)


class StubRobotManager:
    def __init__(self):
        self.gripper_qpos = 0.123

    def get_gripper_qpos(self):
        return self.gripper_qpos


class StubTask:
    def __init__(self, success=True):
        self._robot_manager = StubRobotManager()
        self.success = success
        self.reset_seed = None
        self.last_action = None
        self.closed = False

    def _raw_observation(self):
        image = np.zeros((6, 5, 3), dtype=np.uint8)
        return {
            "embodiment": {
                "joint": np.array(
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.123],
                    dtype=np.float32,
                )
            },
            "observation": {
                "head": {"rgb": image},
                "wrist": {"rgb": image},
            },
            "tactile": {
                "left_tactile": {"rgb_marker": image},
                "right_tactile": {"rgb_marker": image},
            },
        }

    def reset(self, seed):
        self.reset_seed = seed

    def _get_observations(self):
        return self._raw_observation()

    def get_rl_metrics(self):
        return {"metric": 1.0}

    def env_step(self, action, **kwargs):
        self.last_action = np.asarray(action, dtype=np.float32)
        self.last_kwargs = kwargs
        return (
            self._raw_observation(),
            -7.0,
            bool(self.success),
            False,
            {"success": bool(self.success)},
        )

    def close(self):
        self.closed = True


class DiagnosticStubTask(StubTask):
    PHASE_POLICY = 1
    PHASE_TERMINAL = 2

    def __init__(self, success=True, diagnostics=None):
        super().__init__(success=success)
        self.diagnostics = diagnostics or {}
        self.phase_id = self.PHASE_POLICY
        self.terminal_reason = None

    def _get_success_diagnostics(self):
        return self.diagnostics

    def _set_phase(self, phase_id, terminal_reason=None):
        self.phase_id = int(phase_id)
        if self.phase_id == self.PHASE_TERMINAL:
            self.terminal_reason = terminal_reason or "terminal"


class SequenceRng:
    def __init__(self, values):
        self.values = list(values)

    def uniform(self, low, high):
        del low, high
        if not self.values:
            raise AssertionError("SequenceRng is exhausted")
        return self.values.pop(0)


class HandoffPose:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=np.float64)

    def tolist(self):
        return self.p.tolist()

    def add_bias(self, offset):
        return HandoffPose(self.p + np.asarray(offset, dtype=np.float64))

    def rebase(self, to_coord="world"):
        if isinstance(to_coord, HandoffPose):
            return HandoffPose(self.p - to_coord.p)
        return HandoffPose(self.p)


class HandoffAtom:
    def move_by_displacement(self, **kwargs):
        return dict(kwargs)

    def place_actor(self, actor, target_pose, **kwargs):
        del actor, kwargs
        return {"target_pose": target_pose}


class CoarseHandoffTask(StubTask):
    PHASE_PRE_MOVE = 0
    PHASE_POLICY = 1

    def __init__(self):
        super().__init__(success=False)
        self.rng = SequenceRng([0.0015])
        self.plan_success = True
        self.metadata = {}
        self.atom = HandoffAtom()
        self.opening_pose = HandoffPose([0.5, 0.0, 0.1])
        self.prism = type(
            "Prism",
            (),
            {"get_pose": lambda _: HandoffPose([0.501, -0.001, 0.1115])},
        )()
        self.move_tags = []
        self.step_count = 7

    def _set_phase(self, phase_id):
        self.phase_id = phase_id

    def _prepare_usb_standard(self):
        self.move_tags.append("move_usb_to_pre_insert")

    def _update_insert_reference_poses(self):
        pass

    def move(self, action, tag, **kwargs):
        del action, kwargs
        self.move_tags.append(tag)
        return True

    def _update_render(self):
        pass


class HybridMotionPlanTask(StubTask):
    PHASE_POLICY = 1
    PHASE_TERMINAL = 2

    def __init__(self):
        super().__init__(success=False)
        self.phase_id = self.PHASE_POLICY
        self.terminal_reason = None
        self.plan_success = True
        self.eval_success = False
        self.atom = HandoffAtom()
        self.opening_pose = HandoffPose([0.0, 0.0, 0.012])
        self.target_pose = HandoffPose([0.0, 0.0, 0.0])
        self.prism_pose = HandoffPose([0.003, -0.002, 0.014])
        self.prism = type(
            "Prism",
            (),
            {"get_pose": lambda inner_self: self.prism_pose},
        )()
        self._robot_manager.get_gripper_center_pose = lambda: HandoffPose(
            self.prism_pose.p + np.array([0.0, 0.0, 0.03])
        )
        self.move_tags = []

    def _update_insert_reference_poses(self):
        self.opening_pose = HandoffPose([0.0, 0.0, 0.012])
        self.target_pose = HandoffPose([0.0, 0.0, 0.0])

    def _get_success_diagnostics(self):
        position = self.prism_pose.p - self.target_pose.p
        return {
            "rel_xyz": position.tolist(),
            "xy_error": float(np.linalg.norm(position[:2])),
            "abs_z_error": float(abs(position[2])),
        }

    def move(self, action, tag, **kwargs):
        del kwargs
        self.move_tags.append(tag)
        if tag == "hybrid_motion_plan_align":
            self.prism_pose = action["target_pose"]
        elif tag == "hybrid_motion_plan_retreat":
            self.prism_pose = HandoffPose(
                self.prism_pose.p
                + np.array([0.0, 0.0, action["z"]], dtype=np.float64)
            )
        elif tag == "hybrid_motion_plan_insert":
            self.prism_pose = HandoffPose(
                self.prism_pose.p
                + np.array([0.0, 0.0, action["z"]], dtype=np.float64)
            )
        return True

    def _open_gripper_after_insert(self):
        self.move_tags.append("open_gripper_after_insert")

    def delay(self, steps, is_save=False):
        del steps, is_save
        return True

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        return (
            diagnostics["xy_error"] < 0.002
            and diagnostics["abs_z_error"] < 0.003
        )

    def _set_phase(self, phase_id, terminal_reason=None):
        self.phase_id = int(phase_id)
        self.terminal_reason = terminal_reason


def patch_fake_bc(monkeypatch, action=None):
    if action is None:
        action = np.zeros(7, dtype=np.float32)
    monkeypatch.setattr(
        gym_env,
        "load_bc_checkpoint",
        lambda *args, **kwargs: {
            "observation_contract": {
                "image_size": 4,
                "camera_keys": ("cam_high",),
                "tactile_keys": (),
            }
        },
    )
    monkeypatch.setattr(
        gym_env,
        "restore_actor_from_bc_checkpoint",
        lambda *args, **kwargs: StubActor(action),
    )
    monkeypatch.setattr(
        gym_env,
        "extract_action_scale_from_bc_checkpoint",
        lambda *args, **kwargs: torch.ones(7),
    )


def test_direct_env_uses_policy_action_without_bc_addition(monkeypatch):
    bc_action = np.array(
        [0.8, -0.8, 0.0, 0.5, -0.5, 0.2, -0.2],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        gym_env,
        "load_bc_checkpoint",
        lambda *args, **kwargs: {
            "observation_contract": {
                "image_size": 4,
                "camera_keys": ("cam_high", "cam_wrist"),
                "tactile_keys": ("tac_left", "tac_right"),
            }
        },
    )
    monkeypatch.setattr(
        gym_env,
        "restore_actor_from_bc_checkpoint",
        lambda *args, **kwargs: StubActor(bc_action),
    )
    monkeypatch.setattr(
        gym_env,
        "extract_action_scale_from_bc_checkpoint",
        lambda *args, **kwargs: torch.full((7,), 0.1),
    )

    task = StubTask(success=True)
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        action_repeat=2,
        control_mode="direct",
        reward_mode="sparse_success",
        handoff_mode="none",
        force_control=True,
        device="cpu",
    )
    observation, reset_info = environment.reset(seed=42)
    policy_action = np.array(
        [1.0, -1.0, 0.5, 1.0, -1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    next_observation, reward, terminated, truncated, info = environment.step(
        policy_action
    )

    expected_normalized_action = np.array(
        [1.0, -1.0, 0.5, 1.0, -1.0, 0.0, 0.0],
        dtype=np.float32,
    )
    expected_target_qpos = (
        observation["qpos"] + 0.1 * expected_normalized_action
    )
    expected_full_target = np.concatenate(
        [expected_target_qpos, np.array([0.123], dtype=np.float32)]
    )

    assert reset_info["seed"] == 42
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert next_observation["cam_high"].shape == (3, 4, 4)
    np.testing.assert_allclose(task.last_action, expected_full_target)
    assert task.last_kwargs == {
        "action_type": "qpos",
        "force": True,
        "action_repeat": 2,
    }
    np.testing.assert_allclose(info["bc_action"], bc_action)
    np.testing.assert_allclose(
        info["normalized_action"],
        expected_normalized_action,
    )
    assert info["task_reward"] == -7.0


def test_sparse_reward_is_zero_without_success(monkeypatch):
    monkeypatch.setattr(
        gym_env,
        "load_bc_checkpoint",
        lambda *args, **kwargs: {
            "observation_contract": {
                "image_size": 4,
                "camera_keys": ("cam_high",),
                "tactile_keys": (),
            }
        },
    )
    monkeypatch.setattr(
        gym_env,
        "restore_actor_from_bc_checkpoint",
        lambda *args, **kwargs: StubActor(np.zeros(7, dtype=np.float32)),
    )
    monkeypatch.setattr(
        gym_env,
        "extract_action_scale_from_bc_checkpoint",
        lambda *args, **kwargs: torch.ones(7),
    )
    task = StubTask(success=False)
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        device="cpu",
    )
    environment.reset(seed=0)

    _, reward, terminated, truncated, _ = environment.step(
        np.ones(7, dtype=np.float32)
    )

    assert reward == 0.0
    assert terminated is False
    assert truncated is False


def test_bc_action_gain_scales_and_clips_only_bc_control(monkeypatch):
    bc_action = np.array(
        [0.8, -0.8, 0.4, -0.4, 0.2, -0.2, 0.0],
        dtype=np.float32,
    )
    patch_fake_bc(monkeypatch, action=bc_action)
    task = StubTask(success=False)
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        bc_action_gain=1.5,
        device="cpu",
    )
    observation, _ = environment.reset(seed=0)

    _, _, _, _, info = environment.step(
        np.full(7, 0.25, dtype=np.float32)
    )

    expected_scaled_action = np.clip(bc_action * 1.5, -1.0, 1.0)
    expected_target_qpos = observation["qpos"] + expected_scaled_action
    np.testing.assert_allclose(info["bc_action"], bc_action)
    np.testing.assert_allclose(
        info["scaled_bc_action"],
        expected_scaled_action,
    )
    np.testing.assert_allclose(
        info["normalized_action"],
        expected_scaled_action,
    )
    np.testing.assert_allclose(task.last_action[:7], expected_target_qpos)
    assert info["bc_action_gain"] == 1.5


@pytest.mark.parametrize("gain", [0.0, -1.0, float("nan"), float("inf")])
def test_bc_action_gain_must_be_finite_and_positive(monkeypatch, gain):
    patch_fake_bc(monkeypatch)
    with pytest.raises(ValueError, match="bc_action_gain"):
        TactileControlEnv(
            StubTask(success=False),
            "fake_bc.pt",
            handoff_mode="none",
            bc_action_gain=gain,
            device="cpu",
        )


def test_insert_usb_xy_quit_truncates_without_success(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = DiagnosticStubTask(
        success=False,
        diagnostics={
            "rel_xyz": [0.02, 0.0, 1.0],
            "xy_error": 0.02,
        },
    )
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        insert_usb_xy_quit_threshold=0.006,
        device="cpu",
    )
    environment.reset(seed=0)

    _, reward, terminated, truncated, info = environment.step(
        np.zeros(7, dtype=np.float32)
    )

    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["rl_xy_out_of_slot"] is True
    assert info["terminal_reason"] == "xy_out_of_slot"
    assert task.phase_id == task.PHASE_TERMINAL


def test_insert_usb_xy_quit_ignores_z_error(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = DiagnosticStubTask(
        success=False,
        diagnostics={
            "rel_xyz": [0.0, 0.0, 1.0],
            "xy_error": 0.0,
        },
    )
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        insert_usb_xy_quit_threshold=0.006,
        device="cpu",
    )
    environment.reset(seed=0)

    _, reward, terminated, truncated, info = environment.step(
        np.zeros(7, dtype=np.float32)
    )

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["rl_xy_out_of_slot"] is False
    assert task.phase_id == task.PHASE_POLICY


def test_diverse_mild_samples_xy_z_offset(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = StubTask(success=True)
    task.rng = SequenceRng(
        [
            0.35,
            0.0,
            0.0010,
            0.0020,
        ]
    )
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        insert_usb_handoff_distribution="diverse_mild",
        device="cpu",
    )

    sample = environment._sample_insert_usb_handoff(default_clearance=0.0)

    assert sample["distribution"] == "diverse_mild"
    assert sample["profile"] == "precontact"
    assert sample["contact_goal"] == "mild_xy_z_offset"
    assert np.isclose(np.linalg.norm(sample["xy_offset"]), 0.0010)
    assert sample["z_clearance"] == 0.0020
    np.testing.assert_allclose(sample["rpy_offset"], np.zeros(3))


def test_coarse_preinsert_keeps_xy_and_adds_z_jitter(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = CoarseHandoffTask()
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="insert_usb_collect",
        insert_usb_handoff_distribution="coarse_preinsert",
        device="cpu",
    )

    environment._prepare_insert_usb_collect_handoff()

    assert task.move_tags == [
        "move_usb_to_pre_insert",
        "rl_handoff_z_jitter",
    ]
    assert task.metadata["rl_handoff_distribution"] == "coarse_preinsert"
    np.testing.assert_allclose(
        task.metadata["rl_handoff_xy_offset"],
        [0.001, -0.001],
    )
    assert task.metadata["rl_handoff_z_clearance"] == pytest.approx(0.0115)
    assert task.metadata["rl_handoff_coarse_z_jitter_amplitude"] == 0.002
    assert task.metadata["rl_handoff_coarse_z_offset"] == pytest.approx(0.0015)
    assert task.phase_id == task.PHASE_POLICY


def test_coarse_preinsert_can_disable_z_jitter(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = CoarseHandoffTask()
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="insert_usb_collect",
        insert_usb_handoff_distribution="coarse_preinsert",
        insert_usb_coarse_z_jitter=0.0,
        device="cpu",
    )

    environment._prepare_insert_usb_collect_handoff()

    assert task.move_tags == ["move_usb_to_pre_insert"]
    assert task.metadata["rl_handoff_profile"] == "coarse_preinsert_xy_offset"
    assert task.metadata["rl_handoff_coarse_z_jitter_amplitude"] == 0.0
    assert task.metadata["rl_handoff_coarse_z_offset"] == 0.0
    assert task.metadata["rl_handoff_z_clearance"] == pytest.approx(0.0115)
    assert task.phase_id == task.PHASE_POLICY


def test_diverse_tiny_samples_mostly_direct_and_tiny_precontact(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = StubTask(success=True)
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        insert_usb_handoff_distribution="diverse_tiny",
        device="cpu",
    )

    task.rng = SequenceRng(
        [
            0.79,
            0.0,
            0.0005,
            0.0020,
        ]
    )
    direct_sample = environment._sample_insert_usb_handoff(default_clearance=0.0)

    assert direct_sample["distribution"] == "diverse_tiny"
    assert direct_sample["profile"] == "direct"
    assert direct_sample["contact_goal"] == "direct_xy_z_offset"
    assert np.isclose(np.linalg.norm(direct_sample["xy_offset"]), 0.0005)
    assert direct_sample["z_clearance"] == 0.0020
    np.testing.assert_allclose(direct_sample["rpy_offset"], np.zeros(3))

    task.rng = SequenceRng(
        [
            0.80,
            0.0,
            0.0012,
            0.0020,
        ]
    )
    precontact_sample = environment._sample_insert_usb_handoff(
        default_clearance=0.0
    )

    assert precontact_sample["distribution"] == "diverse_tiny"
    assert precontact_sample["profile"] == "precontact"
    assert (
        precontact_sample["contact_goal"]
        == "tiny_xy_z_offset"
    )
    assert np.isclose(np.linalg.norm(precontact_sample["xy_offset"]), 0.0012)
    assert precontact_sample["z_clearance"] == 0.0020
    np.testing.assert_allclose(precontact_sample["rpy_offset"], np.zeros(3))


def test_curriculum_v1_bootstrap_is_80_direct_20_tiny_precontact(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = StubTask(success=True)
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        insert_usb_handoff_distribution="curriculum_v1",
        insert_usb_curriculum_success_thresholds=(2, 4, 6),
        device="cpu",
    )

    task.rng = SequenceRng(
        [
            0.79,
            0.0,
            0.0005,
            0.0020,
        ]
    )
    direct_sample = environment._sample_insert_usb_handoff(default_clearance=0.0)

    assert direct_sample["distribution"] == "curriculum_v1"
    assert direct_sample["profile"] == "direct"
    assert direct_sample["curriculum_stage_index"] == 0
    assert (
        direct_sample["curriculum_stage_name"]
        == "bootstrap_xy_z_offset"
    )
    assert direct_sample["curriculum_success_thresholds"] == (2, 4, 6)
    assert direct_sample["curriculum_collected_successes"] == 0
    assert np.isclose(np.linalg.norm(direct_sample["xy_offset"]), 0.0005)
    assert direct_sample["z_clearance"] == 0.0020
    np.testing.assert_allclose(direct_sample["rpy_offset"], np.zeros(3))

    task.rng = SequenceRng(
        [
            0.80,
            0.0,
            0.0012,
            0.0020,
        ]
    )
    precontact_sample = environment._sample_insert_usb_handoff(
        default_clearance=0.0
    )

    assert precontact_sample["profile"] == "precontact"
    assert (
        precontact_sample["contact_goal"]
        == "curriculum_tiny_xy_z_offset"
    )
    assert np.isclose(np.linalg.norm(precontact_sample["xy_offset"]), 0.0012)
    assert precontact_sample["z_clearance"] == 0.0020
    np.testing.assert_allclose(precontact_sample["rpy_offset"], np.zeros(3))


def test_curriculum_v1_progresses_to_all_diversity_profiles(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = StubTask(success=True)
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        insert_usb_handoff_distribution="curriculum_v1",
        insert_usb_curriculum_success_thresholds=(2, 4, 6),
        device="cpu",
    )

    environment.collected_successes = 2
    task.rng = SequenceRng(
        [
            0.50,
            0.0,
            0.0020,
            0.0030,
        ]
    )
    expanded_sample = environment._sample_insert_usb_handoff(
        default_clearance=0.0
    )

    assert expanded_sample["curriculum_stage_index"] == 1
    assert (
        expanded_sample["curriculum_stage_name"]
        == "expand_xy_z_offset"
    )
    assert expanded_sample["profile"] == "precontact"
    assert (
        expanded_sample["contact_goal"]
        == "curriculum_expanded_xy_z_offset"
    )
    assert np.isclose(np.linalg.norm(expanded_sample["xy_offset"]), 0.0020)
    assert expanded_sample["z_clearance"] == 0.0030
    np.testing.assert_allclose(expanded_sample["rpy_offset"], np.zeros(3))

    environment.collected_successes = 4
    task.rng = SequenceRng(
        [
            0.50,
            0.0,
            0.0025,
            0.0035,
        ]
    )
    wide_sample = environment._sample_insert_usb_handoff(
        default_clearance=0.0
    )

    assert wide_sample["curriculum_stage_index"] == 2
    assert (
        wide_sample["curriculum_stage_name"]
        == "wide_xy_z_offset"
    )
    assert wide_sample["profile"] == "precontact"
    assert (
        wide_sample["contact_goal"]
        == "curriculum_wide_xy_z_offset"
    )
    assert np.isclose(np.linalg.norm(wide_sample["xy_offset"]), 0.0025)
    assert wide_sample["z_clearance"] == 0.0035
    np.testing.assert_allclose(wide_sample["rpy_offset"], np.zeros(3))

    environment.collected_successes = 6
    task.rng = SequenceRng(
        [
            0.90,
            0.0,
            0.0040,
            0.0050,
        ]
    )
    large_sample = environment._sample_insert_usb_handoff(
        default_clearance=0.0
    )

    assert large_sample["curriculum_stage_index"] == 3
    assert (
        large_sample["curriculum_stage_name"]
        == "large_xy_z_offset"
    )
    assert large_sample["profile"] == "precontact"
    assert (
        large_sample["contact_goal"]
        == "curriculum_large_xy_z_offset"
    )
    assert np.isclose(np.linalg.norm(large_sample["xy_offset"]), 0.0040)
    assert large_sample["z_clearance"] == 0.0050
    np.testing.assert_allclose(large_sample["rpy_offset"], np.zeros(3))


def test_curriculum_thresholds_must_be_ordered(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = StubTask(success=True)

    with np.testing.assert_raises(ValueError):
        TactileControlEnv(
            task,
            "fake_bc.pt",
            control_mode="bc",
            reward_mode="sparse_success",
            handoff_mode="none",
            insert_usb_handoff_distribution="curriculum_v1",
            insert_usb_curriculum_success_thresholds=(10, 5, 20),
            device="cpu",
        )


def test_hybrid_motion_plan_records_switch_and_completes_insert(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = HybridMotionPlanTask()
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        device="cpu",
    )
    environment.reset(seed=0)
    environment.policy_step = 37

    _, reward, terminated, truncated, info = (
        environment.complete_insert_usb_with_motion_plan()
    )

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["terminal_reason"] == "hybrid_motion_plan_success"
    hybrid = info["hybrid_motion_plan"]
    assert hybrid["policy_step"] == 37
    assert hybrid["switch_diagnostics"]["xy_error"] == pytest.approx(
        np.sqrt(0.003**2 + 0.002**2)
    )
    assert hybrid["switch_diagnostics"]["abs_z_error"] == 0.014
    assert hybrid["aligned_diagnostics"]["xy_error"] == 0.0
    assert hybrid["insert_distance"] == 0.012
    assert hybrid["alignment_moved"] is True
    assert hybrid["insertion_moved"] is True
    assert task.move_tags == [
        "hybrid_motion_plan_align",
        "hybrid_motion_plan_insert",
        "open_gripper_after_insert",
    ]
    assert task.phase_id == task.PHASE_TERMINAL


def test_hybrid_motion_plan_retreats_before_alignment(monkeypatch):
    patch_fake_bc(monkeypatch)
    task = HybridMotionPlanTask()
    environment = TactileControlEnv(
        task,
        "fake_bc.pt",
        control_mode="bc",
        reward_mode="sparse_success",
        handoff_mode="none",
        device="cpu",
    )
    environment.reset(seed=0)
    initial_snapshot = environment.capture_insert_usb_pose_snapshot()

    _, reward, terminated, truncated, info = (
        environment.complete_insert_usb_with_motion_plan(
            retreat_clearance=0.01,
            initial_pose_snapshot=initial_snapshot,
        )
    )

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    hybrid = info["hybrid_motion_plan"]
    assert hybrid["retreat_clearance"] == 0.01
    assert hybrid["retreat_distance"] == pytest.approx(0.008)
    assert hybrid["retreat_moved"] is True
    assert hybrid["retreated_diagnostics"]["abs_z_error"] == pytest.approx(
        0.022
    )
    assert hybrid["pose_snapshots"]["initial"] == initial_snapshot
    assert set(hybrid["pose_snapshots"]) == {
        "initial",
        "switch",
        "retreated",
        "aligned",
    }
    for snapshot in hybrid["pose_snapshots"].values():
        assert snapshot["usb_in_gripper_center_pose"] == pytest.approx(
            [0.0, 0.0, -0.03]
        )
    assert task.move_tags == [
        "hybrid_motion_plan_retreat",
        "hybrid_motion_plan_align",
        "hybrid_motion_plan_insert",
        "open_gripper_after_insert",
    ]
