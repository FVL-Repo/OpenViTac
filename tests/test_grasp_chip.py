
"""Load real pure task definitions without starting an Isaac Sim process."""
from pathlib import Path
import ast, sys, types
ROOT = Path(__file__).resolve().parents[1]
_modules = {}

def symbol(module, name):
    if module not in _modules:
        path = ROOT / "envs" / (module + ".py")
        tree = ast.parse(path.read_text())
        scope = types.ModuleType("_force_test_" + module)
        scope.__file__ = str(path)
        sys.modules[scope.__name__] = scope
        definitions = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        definitions[child.id] = node
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        definitions[alias.asname or alias.name.split(".")[0]] = node
        _modules[module] = (scope, definitions, set())
        exec("from __future__ import annotations", scope.__dict__)
    scope, definitions, loading = _modules[module]
    if name in scope.__dict__:
        return scope.__dict__[name]
    if name not in definitions:
        raise ImportError(f"{module}.{name} is not a task helper")
    if name in loading:
        return None
    loading.add(name)
    node = definitions[name]
    if isinstance(node, ast.ImportFrom) and node.level and node.module == "_force_task_utils":
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            scope.__dict__[alias.asname or alias.name] = symbol(node.module, alias.name)
    else:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in definitions:
                symbol(module, child.id)
        unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(unit), scope.__file__, "exec"), scope.__dict__)
    loading.remove(name)
    return scope.__dict__[name]


from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

next_grip_qpos = symbol("_force_task_utils", "next_grip_qpos")
ChipLifecycle = symbol("grasp_chip", "ChipLifecycle")
ChipPhysicalSample = symbol("grasp_chip", "ChipPhysicalSample")
impact_breaks_chip = symbol("grasp_chip", "impact_breaks_chip")
SupportImpactTracker = symbol("grasp_chip", "SupportImpactTracker")
update_chip_impact_statistics = symbol("grasp_chip", "update_chip_impact_statistics")
update_chip_strain_statistics = symbol("grasp_chip", "update_chip_strain_statistics")
chip_runtime_timing = symbol("grasp_chip", "chip_runtime_timing")
configure_final_task = symbol("_force_task_utils", "configure_final_task")
resolve_policy_action_repeat = symbol("_force_task_utils", "resolve_policy_action_repeat")


def test_chip_scene_only_registers_active_task_physics():
    path = ROOT / "envs/grasp_chip.py"
    tree = ast.parse(path.read_text())
    task = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Task"
    )
    create_actors = next(
        node for node in task.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_actors"
    )
    physical_scene = ast.unparse(create_actors)
    assert "chip_landing_tray" in physical_scene
    assert "variant = 'medium'" in physical_scene
    assert "chip_presentation_pedestal" not in physical_scene
    assert "chip_fragment_" not in physical_scene
    assert "for variant in CHIP_VARIANTS" not in physical_scene

    fracture_visuals = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChipFractureVisuals"
    )
    assert "chip_curved_shard_" in ast.unparse(fracture_visuals)


@pytest.mark.parametrize("depths,q,expected", [
    ([27.200005, 26.7], .0153, .015295),
    ([27.200001, 28.0], .016, .0159),
    ([27.19, 26.7], .0153, None),
    ([28., 28.], 1e-6, 0.),
])
def test_public_grasp_stops_bilaterally_and_closes_by_finite_steps(depths, q, expected):
    result = next_grip_qpos(depths, q)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)

@pytest.mark.parametrize("depths", [[float("nan"), 27.], [27.], [27., float("inf")]])
def test_public_grasp_rejects_invalid_depth(depths):
    with pytest.raises(ValueError):
        next_grip_qpos(depths, .02)

class ChipEpisode:
    def __init__(self):
        self.scorer = ChipLifecycle()
        self.sample = ChipPhysicalSample(step=0)

    def ticks(self, count, **changes):
        for _ in range(count):
            self.sample = replace(self.sample, step=self.sample.step + 1, **changes)
            self.scorer.advance(self.sample)

    def land(self, count=5):
        self.ticks(count, released=True, grip_contact=False, in_target=True,
                   support_contact=True, support_gap_m=.0002)

def test_chip_success_needs_sustained_released_support_in_tray():
    e = ChipEpisode()
    e.land(4)
    assert not e.scorer.success
    e.land(1)
    assert e.scorer.success
    snapshot = e.scorer.snapshot()
    for _ in range(100):
        e.scorer.advance(e.sample)
    assert e.scorer.snapshot() == snapshot

def test_chip_landing_hold_must_be_five_consecutive_physical_steps():
    e = ChipEpisode()
    e.land(4)
    e.ticks(1, support_contact=False)
    assert e.scorer.landed_ticks == 0
    e.land(4)
    assert not e.scorer.success
    e.land(1)
    assert e.scorer.success

def test_chip_opening_and_lost_grip_are_not_immediate_failures():
    e = ChipEpisode()
    e.ticks(20, opening=True, grip_contact=False, released=True)
    assert not e.scorer.failure
    e.land()
    assert e.scorer.success

def test_chip_motion_proxies_do_not_override_physical_outcome():
    e = ChipEpisode()
    e.ticks(20, pose_error_rad=np.pi, drift_m=.1, rotation_rad=np.pi,
            overtravel_m=.1, opening=True, grip_contact=False)
    assert not e.scorer.failure
    e.land()
    assert e.scorer.success

def test_chip_impact_speed_threshold_is_retained():
    assert not impact_breaks_chip(.089, True, critical_speed_m_s=.09)
    assert impact_breaks_chip(.09, True, critical_speed_m_s=.09)
    assert not impact_breaks_chip(.2, False, critical_speed_m_s=.09)

def test_chip_impact_tracker_is_separation_to_contact_not_free_fall():
    tracker = SupportImpactTracker(.09, contact_zone_m=.002)
    assert not tracker.advance(.2, False, .003)
    assert tracker.flight_armed
    assert tracker.advance(.2, True, .001)
    assert not tracker.flight_armed
    assert not tracker.advance(.2, True, .001)

def test_chip_impact_statistics_keep_all_candidate_contact_speeds():
    stats = {
        "event_count": 0,
        "max_incoming_speed_m_s": None,
        "max_incoming_speed_step": None,
        "reference_critical_speed_m_s": .09,
        "reference_threshold_exceedance_count": 0,
        "impact_failure_enabled": False,
    }
    stats = update_chip_impact_statistics(stats, .04, 20, False)
    stats = update_chip_impact_statistics(stats, .12, 21, True)
    stats = update_chip_impact_statistics(stats, .08, 22, False)
    assert stats == {
        "event_count": 3,
        "max_incoming_speed_m_s": .12,
        "max_incoming_speed_step": 21,
        "reference_critical_speed_m_s": .09,
        "reference_threshold_exceedance_count": 1,
        "impact_failure_enabled": False,
    }

def test_chip_strain_statistics_track_peak_without_applying_a_threshold():
    stats = {
        "sample_count": 0,
        "max_strain": None,
        "max_strain_percent": None,
        "max_strain_step": None,
        "squeeze_failure_enabled": False,
    }
    stats = update_chip_strain_statistics(stats, .0006, 10)
    stats = update_chip_strain_statistics(stats, .0012, 11)
    stats = update_chip_strain_statistics(stats, .0009, 12)
    assert stats == {
        "sample_count": 3,
        "max_strain": .0012,
        "max_strain_percent": .12,
        "max_strain_step": 11,
        "squeeze_failure_enabled": False,
    }

def test_chip_released_outside_tray_never_succeeds():
    e = ChipEpisode()
    e.ticks(100, released=True, in_target=False, support_contact=False)
    assert not e.scorer.success
    assert not e.scorer.failure
    e.scorer.finish("policy_timeout")
    assert e.scorer.failure == "policy_timeout"

@pytest.mark.parametrize("reason", ["squeeze", "impact"])
def test_chip_damage_is_latched(reason):
    e = ChipEpisode()
    e.ticks(1, fractured=True, damage_reason=reason)
    e.land(100)
    assert e.scorer.failure == reason
    assert not e.scorer.success


def test_chip_collection_uses_60hz_actions_at_120hz_physics():
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1/120), max_save_frames=800)
    configure_final_task(cfg, {}, max_policy_seconds=60)
    assert cfg.sim.dt * cfg.save_frequency == pytest.approx(1/60)
    assert cfg.policy_action_repeat * cfg.sim.dt == pytest.approx(1/60)
    assert cfg.max_save_frames >= 3601

@pytest.mark.parametrize("name,repeat", [
    ("grasp_chip", 1),
])
def test_force_task_policy_action_repeat(name, repeat):
    assert resolve_policy_action_repeat(name, {}) == repeat
    with pytest.raises(ValueError):
        resolve_policy_action_repeat(name, {"action_repeat": repeat + 1})

def test_chip_runtime_timing_only_changes_eval():
    assert chip_runtime_timing("collect") == (120, 2)
    assert chip_runtime_timing("eval") == (60, 1)
    assert chip_runtime_timing("eval_test") == (60, 1)
    with pytest.raises(ValueError):
        chip_runtime_timing("unknown")

def test_legacy_task_keeps_its_action_repeat():
    assert resolve_policy_action_repeat("insert_USB", {}) == 1
    assert resolve_policy_action_repeat("legacy", {"action_repeat": 4}) == 4


def test_chip_task_files_compile():
    for name in ("grasp_chip", "_force_task_utils"):
        path = ROOT / "envs" / (name + ".py")
        compile(path.read_text(), str(path), "exec")
