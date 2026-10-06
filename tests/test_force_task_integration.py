"""CPU regressions for four-task integration and unchanged legacy policy behavior."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import types

import h5py
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import _force_task_utils as helpers
from envs.utils.data import HDF5Handler
from policy.openpi import deploy_policy


def method(file, cls, name):
    tree = ast.parse((ROOT / file).read_text())
    group = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in group.body if isinstance(n, ast.FunctionDef) and n.name == name)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    scope = dict(torch=torch, np=np, __package__="envs")
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / file), "exec"), scope)
    return scope[name]


BASE_ACTION = method("envs/_base_task.py", "BaseTask", "take_action")
BASE_ENV_STEP = method("envs/_base_task.py", "BaseTask", "env_step")


class Robot:
    gripper_max_qpos = .039

    def __init__(self):
        self.commands = []
        self.live = torch.zeros(9)
        self.robot = NS(root_physx_view=NS(get_dof_positions=lambda: self.live[None]))
        self.cached = torch.ones(9)

    def get_observations(self, keys):
        return {"joint": self.cached}

    def set_arm(self, value, **kwargs):
        self.commands.append((value.clone(), kwargs))

    def set_gripper(self, value, **kwargs):
        self.commands[-1] += (value.clone(),)


class Task:
    PHASE_PRE_MOVE, PHASE_POLICY, PHASE_TERMINAL = 0, 1, 2
    take_action = BASE_ACTION
    env_step = BASE_ENV_STEP

    def __init__(self, force=False, stop_at=None, succeed_at=None):
        self.cfg = NS(final_acceptance_contract=force, policy_action_repeat=2,
                      step_lim=20, live_action_joint_state=True)
        self._robot_manager = Robot()
        self.logger = NS(info=lambda *a: None)
        self.device = "cpu"
        self.take_action_cnt = self.step_count = self.policy_step_count = 0
        self.phase_id = 1
        self.eval_success = False
        self.plan_success = True
        self.stop_at, self.succeed_at = stop_at, succeed_at
        self.terminal_reason = None
        self.metadata = {}
        self.instruction = "stretch the strap"

    def _step(self):
        self.step_count += 1

    def check_success(self):
        return self.succeed_at is not None and self.step_count >= self.succeed_at

    def check_early_stop(self):
        return self.stop_at is not None and self.step_count >= self.stop_at

    def get_rl_metrics(self):
        return {}

    def compute_rl_reward(self, *args):
        return 0.

    def check_rl_early_stop(self, metrics):
        return False

    def _get_observations(self):
        return {"step": self.step_count}

    def _set_phase(self, phase, terminal_reason=None):
        self.phase_id, self.terminal_reason = phase, terminal_reason


def test_legacy_action_still_one_tick_and_no_interpolation():
    task = Task()
    task.take_action(torch.ones(8))
    assert task.take_action_cnt == task.step_count == 1
    torch.testing.assert_close(task._robot_manager.commands[0][0], torch.ones(7))


def test_legacy_env_step_keeps_cached_state_repeat_semantics():
    task = Task()
    result = task.env_step(torch.zeros(8), action_repeat=2)
    assert task.take_action_cnt == 1 and task.step_count == 2
    torch.testing.assert_close(task._robot_manager.commands[0][0], torch.full((7,), .5))
    assert result[-1]["action_repeat"] == 2


@pytest.mark.parametrize("via_env_step", [False, True])
def test_force_action_uses_live_state_two_ticks_one_decision(via_env_step):
    task = Task(force=True)
    getattr(task, "env_step" if via_env_step else "take_action")(torch.ones(8))
    assert task.take_action_cnt == 1 and task.step_count == 2
    torch.testing.assert_close(task._robot_manager.commands[0][0], torch.full((7,), .5))
    torch.testing.assert_close(task._robot_manager.commands[1][0], torch.ones(7))


@pytest.mark.parametrize("stop,success", [(1, None), (None, 1)])
def test_force_action_stops_before_second_tick(stop, success):
    task = Task(force=True, stop_at=stop, succeed_at=success)
    task.take_action(torch.ones(8))
    assert task.step_count == 1
    assert task.eval_success == bool(success)


@pytest.mark.parametrize("action", [torch.zeros(7), torch.full((8,), float("nan")), torch.full((8,), float("inf"))])
def test_force_action_rejects_invalid_target_before_robot_write(action):
    task = Task(force=True)
    with pytest.raises(ValueError):
        task.take_action(action)
    assert not task._robot_manager.commands and task.take_action_cnt == 0


def test_hdf5_empty_fix_is_gated_and_nonempty_schema_unchanged():
    for enabled in (False, True):
        with h5py.File("unused", "w", driver="core", backing_store=False) as f:
            HDF5Handler(allow_empty_strings=enabled).dict_to_hdf5(f, {"tag": ["abc", "def"], "joint": np.zeros((2, 9), np.float32)})
            assert f["tag"].dtype == np.dtype("S3")
            assert f["joint"].dtype == np.float32 and f["joint"].shape == (2, 9)
    with h5py.File("unused", "w", driver="core", backing_store=False) as f:
        HDF5Handler(allow_empty_strings=True).dict_to_hdf5(f, {"tag": ["", ""]})
        assert list(f["tag"][:]) == [b"", b""]


@pytest.mark.parametrize("name", sorted(helpers.FINAL_TASKS))
def test_profile_paths_and_seed_do_not_depend_on_cwd_or_environment(name, tmp_path, monkeypatch):
    path = ROOT / "task_config" / (name + ".yml")
    config = yaml.safe_load(path.read_text())
    original = copy.deepcopy(config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENVITAC_TASK_CONFIG", "/nonexistent/old-machine.json")
    cfg = NS(save_dir=tmp_path / "out")
    params = helpers.prepare_force_task_config(cfg, config, path, seed=42)
    assert config == original and params["physics_seed"] == 42
    assert cfg.seed is None  # Scene-dependent Task.seed must run only at reset.
    assert helpers.task_parameters(cfg) == params
    assert "cache_actor_surfaces" not in params and "reuse_same_step_tactile_depth" not in params
    for key in ("calibration", "mesh_path"):
        if key in params:
            assert Path(params[key]).is_file()
            assert not Path(config["task_parameters"][key]).is_absolute()
    if name == "wipe_vase":
        assert params["required_clean_fraction"] == .925
        assert params["pressure_band_N"] == [7.2, 8.8]


class Client:
    connections = 0

    def __init__(self, config):
        self.config = config
        self.observations = []

    def connect(self):
        Client.connections += 1

    def infer(self, obs):
        self.observations.append(obs)
        return np.tile(np.r_[np.arange(7) * .01, .02], (self.config.open_loop_horizon, 1))

    def reset(self):
        pass

    def close(self):
        pass


def observation():
    im = np.zeros((32, 32, 3), np.uint8)
    im[..., 0] = 231
    return dict(embodiment={"joint": np.arange(9, dtype=np.float32) * .01},
                observation={"head": {"rgb": im}, "wrist": {"rgb": im}},
                tactile={name: {"rgb_marker": im} for name in ("left_tactile", "right_tactile")})


@pytest.mark.parametrize("name,repeat", [("insert_USB", None), ("tension_strap", 2), ("wipe_vase", 2)])
@pytest.mark.parametrize("send_tactile", [False, True])
@pytest.mark.parametrize("control_mode", ["abs_joint", "relative_joint"])
def test_pi05_payload_chunk_and_action_cadence(monkeypatch, name, repeat, send_tactile, control_mode):
    monkeypatch.setattr(deploy_policy, "OpenPiClientRuntime", Client)
    cfg = dict(task_name=name, openpi=dict(control_mode=control_mode, open_loop_horizon=2, send_tactile=send_tactile))
    policy = deploy_policy.Policy(cfg)
    task = Task(force=repeat is not None)
    for _ in range(3):
        policy.eval(task, observation())
    assert task.take_action_cnt == 3 and task.step_count == 3 * (repeat or 1)
    assert policy._policy_step_index == 3 and len(policy.client.observations) == 2
    payload = policy.client.observations[0]
    np.testing.assert_array_equal(payload["observation/state"], observation()["embodiment"]["joint"][:8])
    assert payload["prompt"] == task.instruction
    assert payload["observation/image"].shape == (224, 224, 3)
    assert tuple(payload["observation/image"][0, 0]) == (231, 0, 0)
    assert ("observation/left_tactile_image" in payload) == send_tactile
    policy.reset()
    assert policy._policy_step_index == 0 and policy._action_chunk is None


@pytest.mark.parametrize("change", [
    {"eval_action_repeat": 1}, {"openpi": {"action_repeat": 1}},
    {"openpi": {"action_repeat": True}}, {"openpi": {"control_mode": "delta_eef"}},
])
def test_pi05_rejects_force_profile_mismatch_before_connect(monkeypatch, change):
    monkeypatch.setattr(deploy_policy, "OpenPiClientRuntime", Client)
    before = Client.connections
    cfg = dict(task_name="bulb_tightening", openpi={"control_mode": "abs_joint"})
    cfg.update(change)
    with pytest.raises(ValueError):
        deploy_policy.Policy(cfg)
    assert Client.connections == before


def converter_functions():
    path = ROOT / "policy/openpi/abs_joint/convert_insert_usb_sim_to_lerobot.py"
    tree = ast.parse(path.read_text())
    names = {"force_task_context", "episode_indices", "selected_indices", "make_abs_joint_state", "make_state_targets"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(np=np)
    nodes.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), scope)
    return scope


def test_converter_filters_force_reveal_keeps_legacy_frames():
    functions = converter_functions()
    with h5py.File("unused", "w", driver="core", backing_store=False) as f:
        f["step"] = [10, 12, 14, 16, 18, 20, 22]
        f["phase/id"] = [0, 1, 1, 1, 2, 2, 2]
        f["phase"].attrs["terminal_reason"] = "success"
        context = dict(task="wipe_vase", timing_contract={"policy_observation_dt_s": 1 / 60})
        f.attrs["episode_context_json"] = json.dumps(context)
        indices = functions["episode_indices"](f, 1, 1, False, "abs_joint", 7)
        np.testing.assert_array_equal(indices, [1, 2, 3])
        state = functions["make_abs_joint_state"](np.arange(63).reshape(7, 9), 7)
        np.testing.assert_array_equal(functions["make_state_targets"](state, indices, 1), state[[2, 3, 4]])
        f.attrs["episode_context_json"] = json.dumps({"task": "insert_USB"})
        np.testing.assert_array_equal(functions["episode_indices"](f, 1, 1, False, "abs_eef", 8), np.arange(7))


@pytest.mark.parametrize("stride,offset,mode,gripper", [(2, 1, "abs_joint", 7), (1, 2, "abs_joint", 7), (1, 1, "abs_eef", 8)])
def test_converter_rejects_force_training_cadence_mismatch(stride, offset, mode, gripper):
    functions = converter_functions()
    with h5py.File("unused", "w", driver="core", backing_store=False) as f:
        f.attrs["episode_context_json"] = json.dumps({"task": "wipe_vase"})
        with pytest.raises(ValueError):
            functions["episode_indices"](f, stride, offset, False, mode, gripper)


def test_dispatcher_is_noop_for_legacy_without_loading_config():
    assert helpers.dispatch_force_task_seeds("collect", NS(task="insert_USB")) is None


@pytest.mark.parametrize("worker_result,returncode,expected", [("success", 0, 0), ("fail", 0, 1), ("success", 9, 1), (None, 0, 1)])
def test_dispatcher_requires_valid_worker_result(tmp_path, monkeypatch, worker_result, returncode, expected):
    import subprocess
    monkeypatch.delenv("OPENVITAC_FORCE_WORKER", raising=False)
    monkeypatch.setattr(sys, "argv", ["collect_data.py", "wipe_vase", "wipe_vase", "--gpu", "2"])
    cfg = yaml.safe_load((ROOT / "task_config/wipe_vase.yml").read_text())
    cfg["save_dir_exact"] = str(tmp_path / "output")
    args = NS(task="wipe_vase", start_seed=4, max_seed=4, episode_num=1, gpu="2")
    calls = []
    class Process:
        pid = 987654321
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            if worker_result is not None:
                Path(kwargs["env"]["OPENVITAC_FORCE_RESULT"]).write_text(json.dumps(dict(seed=4, result=worker_result)))
        def wait(self, timeout):
            return returncode
        def poll(self):
            return returncode
    monkeypatch.setattr(subprocess, "Popen", Process)
    code = helpers.dispatch_force_task_seeds("collect", args, config=cfg, config_path=ROOT / "task_config/wipe_vase.yml")
    assert code == expected
    assert len(calls) == 1 and calls[0][0][-6:] == ["--start_seed", "4", "--max_seed", "4", "--episode_num", "1"]
    assert calls[0][0][4:8] == ["wipe_vase", "wipe_vase", "--gpu", "2"]
    state = json.loads((tmp_path / "output/force_task_progress.json").read_text())
    assert state["successes"] == int(worker_result == "success" and returncode == 0)


def test_eval_failure_counts_as_attempt_but_errors_do_not(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("OPENVITAC_FORCE_RESULT", str(result_path))
    task = Task(force=True, stop_at=1)
    task.reset = lambda **kwargs: None
    task.clean_cache = lambda **kwargs: None
    policy = NS(reset=lambda: None, eval=lambda task, obs: task.take_action(torch.zeros(8)))
    result = helpers.run_force_task_episode(task, seed=8, policy=policy)
    assert result == {"test_num": 1, "succ_num": 0}
    assert json.loads(result_path.read_text())["result"] == "fail"
    def broken(task, obs):
        raise RuntimeError("server disconnected")
    task = Task(force=True)
    task.reset = lambda **kwargs: None
    task.clean_cache = lambda **kwargs: None
    policy.eval = broken
    with pytest.raises(RuntimeError, match="server disconnected"):
        helpers.run_force_task_episode(task, seed=9, policy=policy)
    assert json.loads(result_path.read_text())["result"] == "error"


@pytest.mark.parametrize("enabled,force,expected_calls", [(False, True, 0), (True, True, 2), (True, False, 0)])
def test_robot_zero_velocity_targets_are_opt_in(enabled, force, expected_calls):
    arm = method("envs/robot/robot.py", "RobotManager", "set_arm")
    gripper = method("envs/robot/robot.py", "RobotManager", "set_gripper")
    velocities = []
    robot = NS(set_joint_position_target=lambda *a, **k: None,
               set_joint_velocity_target=lambda value, **k: velocities.append(value),
               root_physx_view=NS(set_dof_positions=lambda *a: None),
               _data=NS(joint_pos_target=torch.zeros(9)), _ALL_INDICES=torch.tensor([0]))
    manager = NS(task=NS(cfg=NS(absolute_joint_zero_velocity_targets=enabled)),
                 robot=robot, robot_type="franka", _arm_ids=list(range(7)), _gripper_ids=[7, 8],
                 _map_gripper_command=lambda v: torch.as_tensor(v).reshape(-1).repeat(2))
    arm(manager, torch.ones(7), force=force)
    gripper(manager, .02, force=force)
    assert len(velocities) == expected_calls
    assert all(torch.count_nonzero(v) == 0 for v in velocities)


@pytest.mark.parametrize("fps", [10., 60.])
def test_video_encoder_receives_correct_fps(tmp_path, monkeypatch, fps):
    from envs.utils import data
    calls = []
    monkeypatch.setattr(data.subprocess, "Popen", lambda command, **kwargs: calls.append(command) or NS())
    monkeypatch.setattr(data.VideoHandler, "_encoder_args", lambda *a: ["-vcodec", "mpeg4"])
    handler = data.VideoHandler()
    assert handler.fps == 10.
    handler.fps = fps
    handler.reset(tmp_path / "test.mp4", (32, 32))
    command = calls[0]
    assert float(command[command.index("-framerate") + 1]) == fps
    handler.ffmpeg = None


@pytest.mark.parametrize("clean,pressure,valid", [(.925, .95, True), (.924999, .95, False), (.925, .94999, False)])
def test_vase_validator_keeps_925_cleaning_and_95_pressure_separate(tmp_path, clean, pressure, valid):
    from scripts.validate_final_task_episode import validate_final
    episode = tmp_path / "hdf5/0.hdf5"
    episode.parent.mkdir()
    timing = dict(physics_dt_s=1 / 120, decimation=1, save_frequency=2,
                  policy_observation_dt_s=1 / 60, policy_action_repeat=2,
                  policy_action_dt_s=1 / 60, video_frequency=2, video_fps=60,
                  quaternion_order="wxyz", position_units="meters", pose_frame="world",
                  zero_velocity_targets=True)
    meta = dict(result="success", required_clean_fraction=.925,
                vase_wiping_parameters={"required_clean_fraction": .925}, stain_area={"ratio": 1.1},
                vase_wiping_final=dict(failure=None, cleaned_fraction=clean, required_clean_fraction=.925,
                    pressure_band_fraction=pressure, peak_normal_force_N=10.))
    (tmp_path / "metadata.json").write_text(json.dumps({"0": meta}))
    with h5py.File(episode, "w") as f:
        f.attrs["episode_context_json"] = json.dumps(dict(task="wipe_vase", seed=0, timing_contract=timing))
        f["step"] = [8, 12, 14, 16]
        f["phase/id"] = [0, 1, 1, 2]
        f["phase/sim_step"] = [8, 12, 14, 16]
        f["phase/policy_step"] = [-1, 2, 4, 6]
        f["phase/name"] = np.array([b"pre_move", b"policy", b"policy", b"terminal"])
        f["phase/is_boundary"] = [1, 1, 0, 1]
        f["phase"].attrs.update(pre_move_saved_frames=1, policy_saved_frames=2, action_saved_frames=2,
                               terminal_saved_frames=1, policy_start_saved_index=1,
                               policy_start_sim_step=10, save_frequency=2, terminal_reason="success")
    report = validate_final(episode, expect="success")
    assert report["valid"] == valid, report


@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_dispatcher_records_infrastructure_failure(tmp_path, monkeypatch, failure):
    import subprocess
    monkeypatch.delenv("OPENVITAC_FORCE_WORKER", raising=False)
    monkeypatch.setattr(sys, "argv", ["collect_data.py", "wipe_vase", "wipe_vase"])
    cfg = yaml.safe_load((ROOT / "task_config/wipe_vase.yml").read_text())
    cfg["save_dir_exact"] = str(tmp_path / "output")
    args = NS(task="wipe_vase", start_seed=0, max_seed=3, episode_num=1)
    class Process:
        pid = 987654321
        def __init__(self, command, **kwargs):
            if failure == "invalid_json":
                Path(kwargs["env"]["OPENVITAC_FORCE_RESULT"]).write_text("{broken")
        def wait(self, timeout):
            if failure == "timeout" and timeout > 10:
                raise subprocess.TimeoutExpired("mock worker", timeout)
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", Process)
    assert helpers.dispatch_force_task_seeds("collect", args, config=cfg, config_path=ROOT / "task_config/wipe_vase.yml") == 1
    state = json.loads((tmp_path / "output/force_task_progress.json").read_text())
    assert state["status"] == "error" and len(state["attempts"]) == 1 and state["successes"] == 0


def test_dispatcher_resume_does_not_rerun_or_overwrite_completed_seed(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.delenv("OPENVITAC_FORCE_WORKER", raising=False)
    monkeypatch.setattr(sys, "argv", ["collect_data.py", "wipe_vase", "wipe_vase"])
    cfg = yaml.safe_load((ROOT / "task_config/wipe_vase.yml").read_text())
    cfg["save_dir_exact"] = str(tmp_path / "output")
    args = NS(task="wipe_vase", start_seed=0, max_seed=3, episode_num=1)
    seeds = []
    class Process:
        pid = 987654321
        def __init__(self, command, **kwargs):
            seed = int(command[-5])
            seeds.append(seed)
            Path(kwargs["env"]["OPENVITAC_FORCE_RESULT"]).write_text(json.dumps(dict(seed=seed, result="success")))
        def wait(self, timeout):
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", Process)
    for goal in (1, 1, 2):
        args.episode_num = goal
        assert helpers.dispatch_force_task_seeds("collect", args, config=cfg, config_path=ROOT / "task_config/wipe_vase.yml") == 0
    assert seeds == [0, 1]


@pytest.mark.parametrize("metadata", ["{old-invalid-json", "[]", "null"])
def test_legacy_converter_still_ignores_optional_context(metadata):
    functions = converter_functions()
    with h5py.File("unused", "w", driver="core", backing_store=False) as f:
        f.attrs["episode_context_json"] = metadata
        f["step"] = [0, 2, 4, 6]
        np.testing.assert_array_equal(
            functions["episode_indices"](f, 1, 1, False, "abs_joint", 7),
            [0, 1, 2, 3])

def test_policy_smoke_action_budget_is_explicit_and_does_not_claim_success(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    monkeypatch.setenv("OPENVITAC_FORCE_RESULT", str(path))
    task = Task(force=True)
    task.reset = lambda **kwargs: None
    task.clean_cache = lambda **kwargs: None
    policy = NS(reset=lambda: None, eval=lambda task, obs: task.take_action(torch.zeros(8)))
    result = helpers.run_force_task_episode(task, seed=8, policy=policy, max_actions=3)
    row = json.loads(path.read_text())
    assert task.take_action_cnt == 3 and task.step_count == 6
    assert row["smoke_test"] and row["stop_reason"] == "action_budget"
    assert result["succ_num"] == 0
