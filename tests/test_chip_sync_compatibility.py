"""CPU regressions for the chip's collection/evaluation timing and video API."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from test_grasp_chip import symbol


configure_final_task = symbol("_force_task_utils", "configure_final_task")
chip_runtime_timing = symbol("grasp_chip", "chip_runtime_timing")
force_task_openpi_repeat = symbol("_force_task_utils", "force_task_openpi_repeat")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["collect", "eval", "eval_test"])
def test_chip_runtime_and_shared_config_use_60hz_policy(mode):
    hz, repeat = chip_runtime_timing(mode)
    cfg = NS(sim=NS(dt=1 / hz), max_save_frames=650)
    configure_final_task(cfg, {}, max_policy_seconds=20,
                         physics_hz=hz, action_repeat=repeat)
    assert cfg.sim.dt * cfg.policy_action_repeat == pytest.approx(1 / 60)
    assert cfg.sim.dt * cfg.save_frequency == pytest.approx(1 / 60)
    assert cfg.sim.dt * cfg.render_frequency == pytest.approx(1 / 60)
    assert cfg.sim.dt * cfg.video_frequency == pytest.approx(1 / 60)
    assert cfg.final_policy_timeout_seconds == 20
    assert cfg.step_lim == 1200


@pytest.mark.parametrize("hz,repeat", [(True, 1), (0, 1), (60, True), (60, 0), (120, 1), (60, 2)])
def test_shared_timing_rejects_invalid_or_non_60hz_policy(hz, repeat):
    cfg = NS(sim=NS(dt=1 / 120), max_save_frames=650)
    with pytest.raises(ValueError):
        configure_final_task(cfg, {}, max_policy_seconds=20,
                             physics_hz=hz, action_repeat=repeat)


def test_chip_eval_accepts_one_repeat_and_rejects_old_two_repeat_profile():
    assert force_task_openpi_repeat({"task_name": "grasp_chip"}) == 1
    assert force_task_openpi_repeat({"task_name": "grasp_chip", "eval_action_repeat": 1}) == 1
    with pytest.raises(ValueError):
        force_task_openpi_repeat({"task_name": "grasp_chip", "eval_action_repeat": 2})


def test_new_video_entrypoint_uses_existing_openvitac_recorder():
    tree = ast.parse((ROOT / "envs/_base_task.py").read_text())
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseTask")
    method = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "_write_video_observation")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "_base_task.py", "exec"), namespace)
    observation, frame, writes = object(), object(), []
    def get_frame_shot(actual):
        assert actual is observation
        return frame
    task = NS(video_handler=NS(write=writes.append), get_frame_shot=get_frame_shot)
    namespace["_write_video_observation"](task, observation)
    assert writes == [frame]
