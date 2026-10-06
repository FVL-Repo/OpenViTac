
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


import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

ThreadSpec = symbol("bulb_tightening", "ThreadSpec")
sample_thread_spec = symbol("bulb_tightening", "sample_thread_spec")
next_grip_qpos = symbol("_force_task_utils", "next_grip_qpos")
ChipLifecycle = symbol("grasp_fragile_chip", "ChipLifecycle")
ChipPhysicalSample = symbol("grasp_fragile_chip", "ChipPhysicalSample")
StrapLifecycle = symbol("tension_strap", "StrapLifecycle")
SoilState = symbol("wipe_vase", "SoilState")
sample_vase_pose = symbol("wipe_vase", "sample_vase_pose")
configure_final_task = symbol("_force_task_utils", "configure_final_task")
resolve_policy_action_repeat = symbol("_force_task_utils", "resolve_policy_action_repeat")
ElasticResultant = symbol("_force_task_utils", "ElasticResultant")

def test_task_files_compile_and_have_no_retired_module_imports():
    names = {"grasp_fragile_chip", "bulb_tightening", "tension_strap",
             "wipe_vase", "_force_task_utils"}
    for name in names:
        path = ROOT / "envs" / (name + ".py")
        source = path.read_text()
        compile(source, str(path), "exec")
        assert "/root/" not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                assert node.module in {"_base_task", "_force_task_utils", "utils.transforms"}
    bulb = (ROOT / "envs/bulb_tightening.py").read_text()
    assert 'scripts/generate_screw_light_bulb_assets.py' in bulb
    assert (ROOT / "scripts/generate_screw_light_bulb_assets.py").is_file()

@pytest.mark.parametrize("hand", [-1, 1])
@pytest.mark.parametrize("turns", [1.12, 2., 2.88])
def test_bulb_handedness_and_lead_preserve_physical_progress(hand, turns):
    spec = ThreadSpec(hand, .0123 / turns)
    yaw = -hand * 360 * turns
    assert spec.signed_progress_turns(yaw) == pytest.approx(turns)
    assert spec.expected_advance(yaw) == pytest.approx(.0123)
    assert spec.expected_advance(-yaw) == pytest.approx(-.0123)
    assert spec.signed_progress_turns(sum([-4., 4., 4., -4.])) == 0.

def test_bulb_specs_are_seeded_and_vary():
    specs = [sample_thread_spec({"physics_seed": seed}) for seed in range(40)]
    assert {s.handedness for s in specs} == {-1, 1}
    assert np.ptp([s.lead_m for s in specs]) > .004
    assert sample_thread_spec({"physics_seed": 6}) == specs[6]

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

    def carry(self):
        self.ticks(6, grip_contact=True)
        self.ticks(10, lift_m=.012)

    def support(self, count=14):
        self.ticks(count, in_target=True, support_contact=True, support_gap_m=.0002)

def test_chip_success_needs_support_release_withdrawal_and_final_hold():
    e = ChipEpisode()
    e.carry()
    e.support()
    e.ticks(1, opening=True)
    e.ticks(24, released=True, grip_contact=False)
    assert not e.scorer.success
    e.ticks(23, withdrawn=True)
    assert not e.scorer.success
    e.ticks(1)
    assert e.scorer.success
    snapshot = e.scorer.snapshot()
    for _ in range(100):
        e.scorer.advance(e.sample)
    assert e.scorer.snapshot() == snapshot

@pytest.mark.parametrize("hold", [0, 1, 13])
def test_chip_premature_release_cannot_recover_into_success(hold):
    e = ChipEpisode()
    e.carry()
    e.support(hold)
    e.ticks(1, opening=True)
    e.support(100)
    e.ticks(100, released=True, withdrawn=True)
    assert not e.scorer.success
    assert e.scorer.failure == "early_release"

@pytest.mark.parametrize("reason", ["squeeze", "impact"])
def test_chip_damage_is_latched(reason):
    e = ChipEpisode()
    e.carry()
    e.support()
    e.ticks(1, fractured=True, damage_reason=reason)
    e.ticks(100, fractured=False, released=True, withdrawn=True)
    assert e.scorer.failure == reason
    assert not e.scorer.success

def strap_ticks(scorer, count, force, **kwargs):
    for _ in range(count):
        scorer.advance(scorer.last_step + 1, force, grasped=True, **kwargs)

def test_strap_requires_two_ordered_continuous_holds():
    s = StrapLifecycle(20)
    strap_ticks(s, 360, 12)
    assert s.stage_index == 0
    strap_ticks(s, 1, 12)
    assert s.stage_index == 1 and not s.success
    strap_ticks(s, 361, 18)
    assert s.success and s.completed_steps == [381, 742]

def test_strap_excursion_restarts_hold():
    s = StrapLifecycle(0)
    strap_ticks(s, 361, 12)
    strap_ticks(s, 300, 18)
    strap_ticks(s, 1, 18.50001)
    assert s.stage_index == 1 and s.stable_since is None
    strap_ticks(s, 360, 18)
    assert not s.success
    strap_ticks(s, 1, 18)
    assert s.success

def test_strap_damage_wins_on_the_success_tick():
    s = StrapLifecycle(0)
    strap_ticks(s, 361, 12)
    strap_ticks(s, 360, 18)
    strap_ticks(s, 1, 18, damage="strap_ruptured")
    assert not s.success and s.failure == "strap_ruptured"

def test_strap_elastic_force_matches_finite_difference_energy():
    box_mesh = symbol("bulb_tightening", "box_mesh")
    points, tets, _ = box_mesh(.016, .016, .010, (5, 5, 7))
    mu, lame = 4e4, 1.2e5
    model = ElasticResultant(points, tets, mu, lame)
    mask = points[:, 2] >= .005
    x = points * [1., 1., 1.05]
    def energy(position):
        ds = np.stack([position[tets[:, i]] - position[tets[:, 0]]
                       for i in (1, 2, 3)], axis=-1)
        f = ds @ model.dm_inv
        j = np.linalg.det(f)
        density = .5 * lame * (j - 1)**2 - mu * (j - 1)
        density += .5 * mu * ((f * f).sum(axis=(1, 2)) - 3)
        return (model.energy_volume * density).sum()
    xp, xm = x.copy(), x.copy()
    xp[mask, 2] += 1e-7
    xm[mask, 2] -= 1e-7
    numerical = -(energy(xp) - energy(xm)) / 2e-7
    assert model.measure(x, mask)["elastic_force_N"][2] == pytest.approx(numerical, rel=1e-6)

def vase_method(name):
    tree = ast.parse((ROOT / "envs/wipe_vase.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Task")
    return copy.deepcopy(next(n for n in cls.body
                              if isinstance(n, ast.FunctionDef) and n.name == name))

def test_vase_default_is_clean925():
    init = vase_method("__init__")
    defaults = [n.args[1].value for n in ast.walk(init)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and len(n.args) > 1
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "required_clean_fraction"]
    assert defaults == [.925]

@pytest.mark.parametrize("required,clean,force_fraction,peak,failure,expected", [
    (.925, .924999, .99, 8., None, False),
    (.925, .925, .95, 8., None, True),
    (.925, .933, .949999, 8., None, False),
    (.925, .95, .99, 12.500001, None, False),
    (.925, .95, .99, 8., "glaze_abraded", False),
    (.95, .933, .99, 8., None, False),
    (.95, .95, .95, 8., None, True),
])
def test_vase_acceptance_retains_pressure_and_damage_gates(required, clean, force_fraction, peak, failure, expected):
    method = vase_method("check_success")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "vase_acceptance", "exec"), scope)
    task = SimpleNamespace(_accepted_result=None, plan_success=True, failure=failure,
                           soil=SimpleNamespace(fraction=clean), required_clean_fraction=required,
                           max_force=peak, params={"overload_force_N": 12.5},
                           pressure_band_fraction=force_fraction)
    assert bool(scope["check_success"](task)) is expected

def test_vase_cleaning_requires_contact_and_sliding():
    data = np.load(ROOT / "assets/objects/task_assets/vase_wiping/vase.npz")
    s = SoilState(data["points"], data["faces"], 31, center_angle=np.pi / 2)
    s.advance(np.zeros_like(s.points), .1)
    assert s.fraction == 0.
    force = np.zeros_like(s.points)
    force[:, 1] = -.3
    s.advance(force, 0.)
    assert s.fraction == 0.
    s.advance(force, .008)
    assert s.fraction > .90

def test_vase_placement_is_seeded_and_within_bounds():
    bounds = {"vase_position_half_range_m": [.02, .02, .015]}
    positions = np.array([sample_vase_pose(seed, bounds)[0] for seed in range(101)])
    assert np.all(np.abs(positions - [.60, .09, .080]) <= [.02, .02, .015])
    assert np.all(np.ptp(positions, axis=0) > [.03, .03, .025])
    np.testing.assert_array_equal(sample_vase_pose(31, bounds)[0], positions[31])

@pytest.mark.parametrize("name", ["grasp_fragile_chip", "bulb_tightening", "tension_strap", "wipe_vase"])
def test_four_tasks_use_60hz_actions_at_120hz_physics(name):
    assert resolve_policy_action_repeat(name, {}) == 2
    with pytest.raises(ValueError):
        resolve_policy_action_repeat(name, {"action_repeat": 4})
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1/120), max_save_frames=800)
    configure_final_task(cfg, {}, max_policy_seconds=60)
    assert cfg.sim.dt * cfg.save_frequency == pytest.approx(1/60)
    assert cfg.policy_action_repeat * cfg.sim.dt == pytest.approx(1/60)
    assert cfg.max_save_frames >= 3601

def test_legacy_task_keeps_its_action_repeat():
    assert resolve_policy_action_repeat("insert_USB", {}) == 1
    assert resolve_policy_action_repeat("legacy", {"action_repeat": 4}) == 4
