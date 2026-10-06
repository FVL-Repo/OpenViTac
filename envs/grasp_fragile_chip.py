"""Pick and place a fragile chip. Geometry, tactile control and physical scoring live together."""
from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import asdict, dataclass
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, Vt
from typing import Callable, Literal
from uipc.unit import GPa
from ._base_task import *
from ._force_task_utils import (
    FRAGMENT_SEAM_SCALE,
    capture_geometry_state,
    estimate_affine_deformation,
    replace_squeezed_chip,
    set_actor_visible,
    task_parameters,
)


# Chip contact

def impact_breaks_chip(downward_speed_m_s, upward_contact, *, critical_speed_m_s):
    values = np.asarray([downward_speed_m_s, critical_speed_m_s])
    if not np.all(np.isfinite(values)) or critical_speed_m_s <= 0:
        raise ValueError("Impact speeds must be finite and threshold positive")
    return bool(upward_contact and downward_speed_m_s >= critical_speed_m_s)

class SupportImpactTracker:
    """Flight-to-surface impact, separate from continuous supported settling."""
    def __init__(self,critical_speed_m_s,contact_zone_m=.002):
        if not np.isfinite([critical_speed_m_s,contact_zone_m]).all() or critical_speed_m_s<=0 or contact_zone_m<=0:
            raise ValueError("Physical impact limits must be positive")
        self.critical_speed_m_s=float(critical_speed_m_s)
        self.contact_zone_m=float(contact_zone_m)
        self.flight_armed=False

    def advance(self,incoming_downward_speed,upward_contact,gap_m):
        if not np.isfinite([incoming_downward_speed,gap_m]).all():
            raise ValueError("Impact motion and separation must be finite")
        if not upward_contact and gap_m>self.contact_zone_m:
            self.flight_armed=True
        if upward_contact:
            impact=self.flight_armed and impact_breaks_chip(
                incoming_downward_speed,True,critical_speed_m_s=self.critical_speed_m_s)
            self.flight_armed=False
            return bool(impact)
        return False

def contact_patch(depth_mm, *, contact_depth_mm=33.0, surface_depth_mm=28.3):
    if hasattr(depth_mm, "detach"):
        depth_mm = depth_mm.detach().cpu().numpy()
    depth = np.asarray(depth_mm, dtype=np.float64)
    if depth.ndim != 2 or not np.all(np.isfinite(depth)):
        raise ValueError("Depth must be a finite two-dimensional array")
    mask = depth < contact_depth_mm
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return {"area_px": 0, "centroid_xy": None, "edge_fraction": None, "max_indentation_mm": 0.0}
    height, width = depth.shape
    edge = (rows < 0.1 * height) | (rows >= 0.9 * height) | (columns < 0.1 * width) | (columns >= 0.9 * width)
    return {
        "area_px": int(rows.size),
        "centroid_xy": [float(columns.mean() / width), float(rows.mean() / height)],
        "edge_fraction": float(edge.mean()),
        "max_indentation_mm": float(max(0.0, surface_depth_mm - depth[mask].min())),
    }

def marker_contact_shift(reference, current):
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    for observation in (reference, current):
        if (observation.ndim != 3 or observation.shape[0] != 2 or observation.shape[2] != 2
                or not np.all(np.isfinite(observation))):
            raise ValueError("Marker observations must be finite [2, markers, 2] pixel coordinates")
    reference = reference[:, np.any(reference[0] != 0.0, axis=1)]
    current = current[:, np.any(current[0] != 0.0, axis=1)]
    if min(reference.shape[1], current.shape[1]) < 16:
        raise ValueError("At least 16 tracked markers are required")
    _, unique_indices = np.unique(current[0], axis=0, return_index=True)
    current = current[:, unique_indices]
    distances = np.linalg.norm(current[0, :, None] - reference[0, None, :], axis=-1)
    matches = distances.argmin(axis=1)
    valid = distances[np.arange(len(matches)), matches] < 1.0
    if np.count_nonzero(valid) < 16:
        raise ValueError("At least 16 matching reference markers are required")
    displacement = ((current[1] - current[0])[valid]
                    - (reference[1] - reference[0])[matches[valid]])
    magnitude = np.linalg.norm(displacement, axis=-1)
    return {
        "p90_px": float(np.percentile(magnitude, 90)),
        "max_px": float(magnitude.max()),
        "median_vector_px": np.median(displacement, axis=0).tolist(),
        "matched_markers": int(len(magnitude)),
    }

def marker_detection_thresholds(background):
    if len(background) != 2:
        raise ValueError("Two tactile pads are required")
    thresholds = {}
    for name, samples in background.items():
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 1 or len(samples) < 8 or not np.all(np.isfinite(samples)) or np.any(samples < 0):
            raise ValueError("At least eight finite nonnegative pre-contact samples are required per pad")
        thresholds[name] = float(max(0.15, 3.0 * np.percentile(samples, 95)))
    return thresholds

def marker_signal_meets(shifts, thresholds, *, retention=1.0, min_active_pads=2):
    if (len(thresholds) != 2 or set(shifts) != set(thresholds)
            or not 0 < retention <= 1 or min_active_pads not in (1,2)):
        return False
    # Both sensors must remain valid. Plate load may redistribute onto just
    # one jaw on an asymmetric chip; that is distinct from losing grip contact.
    if not all(np.isfinite(shift.get('p90_px', np.nan))
               and np.isfinite(thresholds[name]) and thresholds[name]>0
               for name,shift in shifts.items()):
        return False
    return sum(shift['p90_px'] >= thresholds[name]*retention
               for name,shift in shifts.items()) >= min_active_pads


# Chip geometry

CHIP_CURVE_RISE_M = 0.016

def boundary_faces(tetrahedra):
    faces = np.concatenate([
        tetrahedra[:, [0, 2, 1]], tetrahedra[:, [0, 1, 3]],
        tetrahedra[:, [0, 3, 2]], tetrahedra[:, [1, 2, 3]],
    ])
    _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(count > 2):
        raise ValueError("Non-manifold tetrahedral mesh")
    return faces[first[count == 1]]


# Chip lifecycle

@dataclass(frozen=True)
class ChipAcceptanceLimits:
    grip_ticks: int = 6
    lift_ticks: int = 10
    support_ticks: int = 14
    released_ticks: int = 24
    withdrawn_ticks: int = 24
    lift_m: float = 0.008
    drift_m: float = 0.006
    rotation_rad: float = math.radians(13)
    stable_speed_m_s: float = 0.004
    stable_angular_speed_rad_s: float = 0.15
    support_gap_m: float = 0.00035
    support_penetration_m: float = 0.0003
    pose_error_rad: float = math.radians(3)
    overtravel_m: float = 0.0006

class ChipLifecycle:
    """A new instance is required for each reset. Queries never advance time."""
    def __init__(self, limits=None):
        self.limits = limits or ChipAcceptanceLimits()
        self.stage = "approach"
        self.failure = ""
        self.outcome = "pending"
        self.last_step = -1
        self.grip_ticks = self.lift_ticks = self.support_ticks = 0
        self.released_ticks = self.withdrawn_ticks = 0
        self.grasp_verified = False
        self.lift_verified = False
        self.release_supported = False
        self.transitions = []
        self.last_sample = None

    @property
    def success(self):
        return self.outcome == "success" and not self.failure

    def _transition(self, stage):
        self.stage = stage
        self.transitions.append({"step": self.last_step, "stage": stage})

    def fail(self, reason, outcome="failure"):
        # Damage and task failures survive later errors/timeouts.
        if not self.failure:
            self.failure = reason
            self.outcome = outcome
            self._transition("terminal")

    def finish(self, reason="incomplete"):
        if self.outcome == "pending":
            self.fail(reason, "invalid" if reason in ("error", "timeout", "invalid_physics") else "failure")

    def advance(self, sample):
        if sample.step < self.last_step:
            raise ValueError("Physical time moved backwards; reset the lifecycle")
        if sample.step == self.last_step:
            return
        previous_sample = self.last_sample
        self.last_step = sample.step
        self.last_sample = sample
        values = asdict(sample)
        finite = all(math.isfinite(value) for value in values.values() if isinstance(value, float))
        if not sample.valid or not finite:
            self.fail("invalid_physics", "invalid")
            return
        if sample.fractured:
            self.fail(sample.damage_reason or "fracture")
            return
        if sample.pose_error_rad > self.limits.pose_error_rad:
            self.fail("gripper_orientation_violation")
            return
        if self.failure:
            return
        if self.success:
            # A caller must freeze physical time after confirming success.
            # If it continues anyway, damage/support loss revoke the result.
            if not self._stable_supported(sample) or not sample.released or not sample.withdrawn:
                self.fail("post_success_state_changed")
            return
        supported = self._stable_supported(sample)
        if self.grasp_verified and sample.overtravel_m >= self.limits.overtravel_m:
            self.fail("plate_overpress")
            return
        if self.stage == "approach":
            self.grip_ticks = self.grip_ticks + 1 if sample.grip_contact else 0
            if self.grip_ticks >= self.limits.grip_ticks:
                self.grasp_verified = True
                self._transition("grasped")
            return
        if self.stage in ("grasped", "transport", "supported"):
            # Opening before actual support AND sufficient hold is a failure,
            # even if a subsequently intact chip happens to land in the plate.
            if sample.opening or sample.released:
                # Qualification belongs to the state BEFORE opening. A landing
                # caused by an airborne opening cannot qualify that same action.
                if (not self.lift_verified or previous_sample is None
                        or not self._stable_supported(previous_sample)
                        or self.support_ticks < self.limits.support_ticks):
                    self.fail("early_release")
                elif not sample.in_target:
                    self.fail("off_plate")
                else:
                    self.release_supported = True
                    self._transition("release")
                return
            if not sample.grip_contact and not supported:
                self.fail("slip")
                return
            if not supported and (sample.drift_m > self.limits.drift_m or sample.rotation_rad > self.limits.rotation_rad):
                self.fail("unstable_grasp")
                return
            if self.stage == "grasped":
                self.lift_ticks = self.lift_ticks + 1 if sample.lift_m >= self.limits.lift_m else 0
                if self.lift_ticks >= self.limits.lift_ticks:
                    self.lift_verified = True
                    self._transition("transport")
            if self.lift_verified:
                self.support_ticks = self.support_ticks + 1 if supported else 0
                if self.support_ticks >= self.limits.support_ticks and self.stage == "transport":
                    self._transition("supported")
            return
        if self.stage in ("release", "withdraw"):
            if not sample.in_target:
                self.fail("off_plate")
                return
            # Elastic unloading may briefly interrupt contact. It cannot accrue
            # final stability time; real support must return before success.
            if not supported or not sample.released:
                self.released_ticks = 0
                self.withdrawn_ticks = 0
                return
            self.released_ticks += 1
            if self.released_ticks >= self.limits.released_ticks and self.stage == "release":
                self._transition("withdraw")
            self.withdrawn_ticks = self.withdrawn_ticks + 1 if sample.withdrawn else 0
            if (self.stage == "withdraw" and self.withdrawn_ticks >= self.limits.withdrawn_ticks):
                self.outcome = "success"
                self._transition("terminal")

    def _contact_supported(self, sample):
        return bool(sample.in_target and sample.support_contact
                    and -self.limits.support_penetration_m <= sample.support_gap_m <= self.limits.support_gap_m)

    def _motion_stable(self, sample):
        return bool(sample.speed_m_s <= self.limits.stable_speed_m_s
                    and sample.angular_speed_rad_s <= self.limits.stable_angular_speed_rad_s)

    def _stable_supported(self, sample):
        return bool(self._contact_supported(sample) and self._motion_stable(sample))

    def snapshot(self):
        return {
            "schema_version": 3, "outcome": self.outcome, "success": self.success,
            "terminal_reason": "success" if self.success else self.failure,
            "stage": self.stage, "physical_step": self.last_step,
            "grasp_verified": self.grasp_verified, "lift_verified": self.lift_verified,
            "release_supported": self.release_supported,
            "support_ticks": self.support_ticks, "released_ticks": self.released_ticks,
            "withdrawn_ticks": self.withdrawn_ticks,
            "transitions": [dict(item) for item in self.transitions],
            "last_physical_sample": None if self.last_sample is None else asdict(self.last_sample),
            "limits": asdict(self.limits),
        }

@dataclass(frozen=True)
class ChipPhysicalSample:
    step: int
    pose_error_rad: float = 0.0
    fractured: bool = False
    damage_reason: str = ""
    grip_contact: bool = False
    released: bool = False
    opening: bool = False
    lift_m: float = 0.0
    drift_m: float = 0.0
    rotation_rad: float = 0.0
    in_target: bool = False
    support_contact: bool = False
    support_gap_m: float = 1.0
    overtravel_m: float = 0.0
    speed_m_s: float = 0.0
    angular_speed_rad_s: float = 0.0
    withdrawn: bool = False
    valid: bool = True


# Chip loads

def nodal_contact_pressure(points, faces, force_N, inward_normal, *, force_epsilon_N=1e-5):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    force = np.asarray(force_N, dtype=float).reshape(-1, 3)
    normal = np.asarray(inward_normal, dtype=float).reshape(3)
    if force.shape != points.shape or not np.isfinite(force).all() or not np.isfinite(points).all():
        raise ValueError("Contact positions and loads must be matching finite [N,3]")
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Contact normal must be finite and nonzero")
    normal /= length
    triangles = points[faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
    area = np.linalg.norm(area_vectors, axis=1)
    # Winding-independent front/back filtering by normal alignment; only
    # force-bearing vertices contribute to the final patch.
    selected = np.abs(area_vectors @ normal) > area * 0.5
    dual_area = np.zeros(len(points))
    np.add.at(dual_area, faces[selected].reshape(-1), np.repeat(area[selected] / 3, 3))
    nodal_normal = np.maximum(0.0, force @ normal)
    active = (nodal_normal > force_epsilon_N) & (dual_area > 0)
    patch_area = float(dual_area[active].sum())
    normal_load = float(nodal_normal[active].sum())
    pressure = nodal_normal[active] / dual_area[active]
    return {
        "normal_force_N": normal_load,
        "contact_area_m2": patch_area,
        "mean_pressure_Pa": normal_load / patch_area if patch_area else 0.0,
        "peak_nodal_pressure_Pa": float(pressure.max()) if len(pressure) else 0.0,
        "contact_nodes": int(active.sum()),
        "area_model": "force_bearing_surface_vertex_dual_area",
        "pressure_calibration_status": "uncalibrated_diagnostic_only; calibration_excluded_from_acceptance",
    }

def measure_pad_loads(task):
    result = {}
    chip_center = np.mean([sensor.get_attach_pose().p
                           for sensor in task._tactile_manager.tactiles.values()], axis=0)
    for name, sensor in task._tactile_manager.tactiles.items():
        points = sensor.gelpad.data.nodal_pos_w.detach().cpu().numpy().reshape(-1, 3)
        force = -sensor._get_contact_force().detach().cpu().numpy().reshape(-1, 3) / task.cfg.sim.dt**2
        if not hasattr(sensor, "_chip_pressure_faces"):
            geometry = sensor.gelpad.geo_slot_list[0].geometry()
            tets = np.asarray(geometry.tetrahedra().topo().view()).reshape(-1, 4)
            sensor._chip_pressure_faces = boundary_faces(tets)
        # Force exerted by the chip on the pad points away from the chip.
        outward = points.mean(0) - chip_center
        result[name] = nodal_contact_pressure(points, sensor._chip_pressure_faces, force, outward)
    return result


# Chip motion

def smooth_translation(start_position, target_position, *, dt, max_speed, max_acceleration):
    start_position = np.asarray(start_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    if start_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("Translation endpoints must have three coordinates")
    if not np.all(np.isfinite([start_position, target_position])):
        raise ValueError("Translation endpoints must be finite")
    limits = np.asarray([dt, max_speed, max_acceleration], dtype=np.float64)
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("Time step, speed, and acceleration must be positive and finite")
    displacement = target_position - start_position
    distance = float(np.linalg.norm(displacement))
    duration = max(
        dt,
        1.875 * distance / max_speed,
        np.sqrt((10.0 / np.sqrt(3.0)) * distance / max_acceleration),
    )
    step_count = max(1, int(np.ceil(duration / dt)))
    progress = np.linspace(0.0, 1.0, step_count + 1)
    fraction = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    return start_position + fraction[:, None] * displacement


# Tactile goals

_GVM_OFFSETS_KEY = "uipc::backend::cuda::GlobalVertexManager"

def contact_force_on_actor(
    task: BaseTask,
    actor: Actor,
    *,
    build_mask: Callable[[Actor], np.ndarray] | None = None,
    norm: Literal["sum", "max", "mean"] = "sum",
    threshold: float = 0.0,
) -> float | None:
    """Scalar UIPC contact-force proxy on an actor, optionally a sub-region.

    Generic form of peel_cucumber._read_blade_contact_force: reads the global
    per-step contact gradient, isolates this actor's vertex range via the
    GlobalVertexManager offset table, applies a lazy zone mask from
    build_mask(actor) when given, cuts the per-vertex gradient magnitudes at
    threshold (the 'pressure > epsilon' noise floor), and reduces them.
    'sum' -> total (default); 'max' -> peak; 'mean' -> per-contact-vertex mean
    (sum / n_contact, the area-normalized candidate for scaled meshes). force
    ~= -gradient, so the value is an uncalibrated RELATIVE proxy. Returns None
    on any transient sim error (fail closed) and 0.0 when the actor or its zone
    has no contact vertices above the threshold. A zone mask of the wrong
    length is a programming error -> ValueError.
    """
    if norm not in ("sum", "max", "mean"):
        raise ValueError(f"unknown norm: {norm!r}")
    try:
        indices, gradients = task.uipc_sim.get_contact_gradient()
        offsets = task.uipc_sim._system_vertex_offsets[_GVM_OFFSETS_KEY]
        start = int(offsets[actor.global_system_id])
        vertex_count = int(actor._vertex_count)
        zone_mask = None
        if build_mask is not None:
            zone_mask = build_mask(actor)
    except (
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    gradients = np.asarray(gradients, dtype=np.float64).reshape(-1, 3)
    if indices.shape[0] == 0:
        return 0.0
    actor_mask = (indices >= start) & (indices < start + vertex_count)
    if not np.any(actor_mask):
        return 0.0
    local_indices = indices[actor_mask] - start
    selected = np.ones(int(actor_mask.sum()), dtype=bool)
    if zone_mask is not None:
        zone = np.asarray(zone_mask, dtype=bool).reshape(-1)
        if zone.shape[0] != vertex_count:
            raise ValueError(
                f"zone mask length {zone.shape[0]} != actor vertex count {vertex_count}"
            )
        selected = zone[local_indices]
    if not np.any(selected):
        return 0.0
    norms = np.linalg.norm(gradients[actor_mask][selected], axis=1)
    if threshold > 0.0:
        norms = norms[norms > threshold]
    if norms.shape[0] == 0:
        return 0.0
    if norm == "max":
        return float(norms.max())
    if norm == "mean":
        return float(norms.mean())
    return float(norms.sum())


# Chip policy monitor

class ChipPolicyMonitor:
    """Used by the expert and public qpos path; never moves or pins an actor."""
    def __init__(self, task):
        self.task = task
        self.scorer = ChipLifecycle()
        self.command = None
        self.previous_pose = task.active_chip.get_pose()
        self.previous_qpos = float(task._robot_manager.get_gripper_qpos())
        self.initial_bottom = float(task.active_chip.vertices[:, 2].min())
        self.reference = None
        self.support_ee_z = None
        self.grip_qpos = None
        self.trace = []

    @property
    def failure(self):
        return self.scorer.failure

    @property
    def stage(self):
        return self.scorer.stage

    def register_command(self, command, action_type="qpos"):
        self.command = command
        self.command_type = action_type

    def _enforce_time_limit(self):
        t = self.task
        start = getattr(t, "policy_start_step", None)
        if start is None or t.phase_id != t.PHASE_POLICY:
            return
        elapsed = (t.step_count - start) * t.cfg.sim.dt * t.cfg.decimation
        if (self.scorer.outcome == "pending"
                and elapsed >= t.cfg.final_policy_timeout_seconds):
            self.scorer.fail("policy_timeout")

    def advance(self):
        t = self.task
        # Failure is already a frozen physical verdict. Passive fracture
        # aftermath can keep simulating, but cannot become POLICY evidence.
        if self.scorer.failure:
            return
        if self.scorer.last_step == t.step_count:
            return
        chip = t.active_chip.get_pose()
        ee = t._robot_manager.get_gripper_center_pose()
        vertices = t.active_chip.vertices
        dt = float(t.cfg.sim.dt * t.cfg.decimation)
        speed, angular_speed = t._pose_error(self.previous_pose, chip)
        speed /= dt
        angular_speed /= dt
        self.previous_pose = chip
        relative = chip.rebase(to_coord=ee)
        depths = t._read_tactile_depth()
        force = contact_force_on_actor(t, t.active_chip, norm="sum")
        valid = depths is not None and force is not None and np.isfinite(vertices).all()
        # Shared bilateral contact evidence uses the31mm surface gate from grasp verification.
        # A weaker, asymmetric pad must not be classified as a dropped chip at28.5mm.
        contact = bool(valid and np.all(depths < 31.0) and force > 0)
        if contact and not self.scorer.grasp_verified:
            self.reference = relative
        drift, rotation = (0.0, 0.0) if self.reference is None else t._pose_error(self.reference, relative)
        tray = t.tray.get_pose()
        local = (vertices - tray.p) @ tray.R
        # Entire object footprint must be inside the flat plate floor.
        in_target = bool(np.all((np.abs(local[:, 0]) / 0.060)**4
                                + (np.abs(local[:, 1]) / 0.051)**4 <= 1.0))
        gap = float(local[:, 2].min() - 0.006)
        support_force = t._chip_support_force()
        support = bool(in_target and support_force is not None and support_force > 0
                       and -0.0003 <= gap <= self.scorer.limits.support_gap_m)
        qpos = float(t._robot_manager.get_gripper_qpos())
        if self.scorer.grasp_verified and self.grip_qpos is None:
            self.grip_qpos = qpos
        if self.grip_qpos is not None and self.stage in ("grasped", "transport", "supported"):
            self.grip_qpos = min(self.grip_qpos, qpos)
        opening = bool(self.grip_qpos is not None and qpos > self.grip_qpos + 0.00015)
        self.previous_qpos = qpos
        pad_centers = [pad.get_attach_pose().p for pad in t._tactile_manager.tactiles.values()]
        pad_distance = min(float(np.linalg.norm(center - chip.p)) for center in pad_centers)
        released = bool(depths is not None and np.all(depths >= 32.5)
                        and t._robot_manager.get_gripper_percentage() > 0.55)
        withdrawn = bool(released and ee.p[2] - vertices[:, 2].max() > 0.030 and pad_distance > 0.045)
        if support and self.support_ee_z is None and self.scorer.lift_verified:
            self.support_ee_z = float(ee.p[2])
        overtravel = (0.0 if self.support_ee_z is None
                      else max(0.0, self.support_ee_z - float(ee.p[2])))
        # q and -q represent the same world-fixed top-down orientation.
        orientation_error = float(2 * np.arccos(np.clip(abs(float(ee.q[1])), 0, 1)))
        sample = ChipPhysicalSample(
            step=int(t.step_count), pose_error_rad=orientation_error,
            fractured=bool(t.fractured), damage_reason=str(t.metadata.get("fracture_reason", "fracture")),
            grip_contact=contact, released=released, opening=opening,
            lift_m=float(vertices[:, 2].min() - self.initial_bottom),
            drift_m=drift, rotation_rad=rotation,
            in_target=in_target, support_contact=support, support_gap_m=gap,
            overtravel_m=overtravel, speed_m_s=speed, angular_speed_rad_s=angular_speed,
            withdrawn=withdrawn, valid=bool(valid))
        self.scorer.advance(sample)
        self._enforce_time_limit()
        t.metadata["success_diagnostics"] = self.scorer.snapshot()
        t.metadata["success_contract"] = "chip_physical_lifecycle_v3"
        t.metadata["physical_result"] = self.scorer.outcome
        t.metadata["terminal_reason"] = self.scorer.snapshot()["terminal_reason"]
        if t.step_count % max(1, t.cfg.save_frequency) == 0 or self.scorer.failure:
            row = self.scorer.snapshot()
            row["depths_mm"] = None if depths is None else depths.tolist()
            row["gripper_qpos_m"] = qpos
            row["support_force_proxy"] = support_force
            row["plate_load_diagnostics"] = t._plate_support_resultant()
            row["pad_load_diagnostics"] = measure_pad_loads(t)
            self.trace.append(row)
            t.metadata["physical_acceptance_trace"] = self.trace


# Chip randomization

CHIP_FIXED_CAMERA_UP = np.array([1.0, 0.0, 0.0])

CHIP_FIXED_JAW_AXIS = np.array([0.0, -1.0, 0.0])


# Chip shapes

def tetrahedron_volumes(points, tetrahedra):
    corners = np.asarray(points)[np.asarray(tetrahedra)]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("Shape map inverted or collapsed a tetrahedron")
    return volumes


# Chip randomization

class ChipRestGeometry:
    def __init__(self, simulation, actors):
        self.simulation = simulation
        self.actors = actors
        self.geometry = []
        for body in simulation.uipc_objects:
            for current_slot in body.geo_slot_list:
                slots = simulation.scene.geometries().find(current_slot.id())
                records = []
                for slot in slots:
                    geometry = slot.geometry()
                    attributes = []
                    for collection in (geometry.vertices(), geometry.instances()):
                        for name in ('velocity', 'aim_position', 'aim_transform', 'is_constrained', 'volume'):
                            attribute = collection.find(name)
                            if attribute is not None:
                                attributes.append((name, attribute, np.asarray(attribute.view()).copy()))
                    records.append((geometry, np.asarray(geometry.positions().view()).copy(),
                                    np.asarray(geometry.transforms().view()).copy(), attributes))
                self.geometry.append((body, records))
        self.actor_vertices = {
            id(actor): (actor.init_vertex_pos.clone(), actor.origin_surf_pts.copy())
            for actor in actors
        }

    def rebuild(self, scaled_actors, scale, *, warp=None):
        from uipc import builtin, view
        from uipc.core import Engine, World
        import torch

        scaled_ids = {id(actor) for actor in scaled_actors}
        for body, records in self.geometry:
            for geometry, original, transform, attributes in records:
                positions = original.copy()
                if id(body) in scaled_ids:
                    points = positions.reshape(-1, 3)
                    local = points - body.init_pose.p
                    points[:] = (local * scale if warp is None else warp(local)) + body.init_pose.p
                volume_factor = float(np.prod(scale))
                if warp is not None and id(body) in scaled_ids:
                    tets = np.asarray(geometry.tetrahedra().topo().view()).reshape(-1, 4)
                    volume_factor = float(
                        tetrahedron_volumes(positions.reshape(-1, 3), tets).sum()
                        / tetrahedron_volumes(original.reshape(-1, 3), tets).sum())
                view(geometry.positions())[:] = positions
                view(geometry.transforms())[:] = transform
                for name, attribute, values in attributes:
                    factor = volume_factor if name == 'volume' and id(body) in scaled_ids else 1.0
                    view(attribute)[:] = values * factor
            if getattr(body, '_data', None) is not None:
                body._data.update(1.0)
        for actor in self.actors:
            original, surface = self.actor_vertices[id(actor)]
            actor.next_status = None
            actor.next_pts = None
            actor.next_mat = None
            actor.next_mask = None
            if id(actor) in scaled_ids:
                anchor = torch.as_tensor(actor.init_pose.p, device=original.device)
                factor = torch.as_tensor(scale, device=original.device)
                if warp is None:
                    actor.init_vertex_pos = (original - anchor) * factor + anchor
                    actor.origin_surf_pts = surface * scale
                else:
                    local = (original - anchor).detach().cpu().numpy().reshape(-1, 3)
                    warped = torch.as_tensor(warp(local), device=original.device, dtype=original.dtype)
                    actor.init_vertex_pos = warped.reshape(original.shape) + anchor
                    actor.origin_surf_pts = warp(surface)
            else:
                actor.init_vertex_pos = original.clone()
                actor.origin_surf_pts = surface.copy()
        simulation = self.simulation
        engine = Engine(simulation.cfg.device, simulation.cfg.workspace)
        world = World(engine)
        world.init(simulation.scene)
        if not world.is_valid():
            raise RuntimeError('invalid_physics: rest geometry initialization failed')
        world.retrieve()
        simulation.world = world
        simulation.engine = engine
        simulation._contact_grad_cache = None
        offsets = [0]
        for body in simulation.uipc_objects:
            offset = body.geo_slot_list[0].geometry().meta().find(builtin.global_vertex_offset)
            offsets.append(int(offset.view()[0]))
            body.global_system_id = len(offsets) - 1
        simulation._system_vertex_offsets['uipc::backend::cuda::GlobalVertexManager'] = offsets

def fixed_grasp_anchor(vertices, center, jaw_axis=CHIP_FIXED_JAW_AXIS):
    """Center fixed vertical jaws on the visible opposing rim patches.

    The two patches may have different heights, shapes and contact areas.
    Their midpoint need not coincide with the asset origin or its mass center.
    """
    vertices=np.asarray(vertices,dtype=float)
    axis=np.asarray(jaw_axis,dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-12):
        raise ValueError("Finite geometry and a nonzero fixed jaw axis are required")
    axis=axis/np.linalg.norm(axis)
    projection=vertices@axis
    masks=[projection<=projection.min()+.0002,projection>=projection.max()-.0002]
    patches=np.array([vertices[mask].mean(axis=0) for mask in masks])
    anchor=patches.mean(axis=0)
    return anchor,float(np.ptp(projection)),{
        "opposing_rim_centers_m":patches.tolist(),
        "opposing_rim_height_difference_m":float(abs(patches[1,2]-patches[0,2])),
        "anchor_offset_from_asset_origin_m":(anchor-np.asarray(center)).tolist(),
        "orientation_changed":False,"equal_contact_area_required":False}

def initialize_staged_actor_poses(simulation, actors):
    """Apply reset poses with zero velocity before any policy contact occurs."""
    from uipc import builtin, view
    from uipc.core import Engine, World

    staged = {id(actor): actor for actor in actors if actor.next_status == 'set' and actor.next_mat is not None}
    temporary = []
    for body in simulation.uipc_objects:
        geometry = body.geo_slot_list[0].geometry()
        affine = geometry.meta().find(builtin.backend_abd_body_offset) is not None
        collection = geometry.instances() if affine else geometry.vertices()
        if id(body) in staged:
            transform = body.next_mat.reshape(4, 4)
            view(geometry.transforms())[:] = transform
            view(geometry.instances().find(builtin.aim_transform))[:] = transform
            view(geometry.instances().find(builtin.is_constrained))[:] = 1
        slot = collection.find('velocity')
        original = None if slot is None else np.asarray(slot.view()).copy()
        if slot is None:
            slot = collection.create('velocity', np.zeros((4, 4)) if affine else np.zeros((3, 1)))
        view(slot)[:] = 0
        temporary.append((collection, slot, original))
    engine = Engine(simulation.cfg.device, simulation.cfg.workspace)
    world = World(engine)
    world.init(simulation.scene)
    if not world.is_valid():
        raise RuntimeError('invalid_physics: reset geometry initialization failed')
    world.retrieve()
    simulation.engine, simulation.world = engine, world
    simulation._contact_grad_cache = None
    for collection, slot, original in temporary:
        if original is None:
            collection.destroy('velocity')
        else:
            view(slot)[:] = original
    offsets = [0]
    for body in simulation.uipc_objects:
        offset = body.geo_slot_list[0].geometry().meta().find(builtin.global_vertex_offset)
        offsets.append(int(offset.view()[0]))
        body.global_system_id = len(offsets) - 1
        if getattr(body, '_data', None) is not None:
            body._data.update(1.0)
    simulation._system_vertex_offsets['uipc::backend::cuda::GlobalVertexManager'] = offsets

def staged_ground_clearances(actors, ground_height):
    """Fail before rebuilding if a staged actor intersects the shared plane."""
    report = {}
    for name, actor in actors.items():
        points = actor.next_pts if actor.next_status == 'set' else actor.vertices
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(points) or not np.all(np.isfinite(points)):
            raise ValueError(f"invalid_physics: nonfinite/empty reset geometry for {name}")
        gap = float(points[:, 2].min() - ground_height)
        if gap <= 0:
            raise ValueError(f"invalid_physics: {name} intersects tabletop at reset ({gap:.9f} m)")
        report[name] = gap
    return report


# Chip scene

class ChipKitchenScene:
    def __init__(self, stage, environment_paths, asset_root):
        self.stage = stage
        self.roots = []
        self.plate_supports = []
        asset_root = Path(asset_root)
        random = np.random.default_rng(20260909)
        for environment_path in environment_paths:
            root_path = f"{environment_path}/chip_kitchen_decoration"
            root = UsdGeom.Xform.Define(stage, root_path)
            translation = root.AddTranslateOp()
            self.roots.append(translation)
            jar_path = f"{root_path}/jar"
            jar = self._reference(jar_path, asset_root / "chip_container_visual.usda")
            jar.AddTranslateOp().Set(Gf.Vec3d(0.265, -0.100, 0.036))
            jar.AddRotateYOp().Set(-90.0)
            support = UsdGeom.Cylinder.Define(stage, f"{environment_path}/chip_plate_support")
            support.CreateRadiusAttr(0.046)
            support.CreateHeightAttr(1.0)
            support.CreateDisplayColorAttr([Gf.Vec3f(0.70, 0.65, 0.53)])
            self.plate_supports.append((support.AddTranslateOp(), support.AddScaleOp()))
            # A single axial stack fits the inner oval; decorative only.
            for index in range(20):
                chip = self._reference(f"{jar_path}/contents_{index}", asset_root / "fragile_chip_medium_scan_visual.usda")
                chip.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.007 + index * 0.007))
                chip.AddScaleOp().Set(Gf.Vec3f(0.94, 0.94, 0.94))
            # Decorative stack is separated from the isolated target and jaws.
            stack = self._reference(f"{root_path}/decorative_stack", asset_root / "chip_spill_stack_visual.usda")
            stack.AddTranslateOp().Set(Gf.Vec3d(0.105, -0.10, 0.001))
            for index, position in enumerate([(0.15, -0.155, 0.001), (0.21, -0.17, 0.001)]):
                chip = self._reference(f"{root_path}/spilled_{index}", asset_root / "fragile_chip_medium_scan_visual.usda")
                chip.AddTranslateOp().Set(Gf.Vec3d(*position))
                chip.AddRotateZOp().Set(float(random.uniform(-40, 40)))

    def _reference(self, path, asset):
        transform = UsdGeom.Xform.Define(self.stage, path)
        transform.GetPrim().GetReferences().AddReference(str(asset))
        return transform

    def reset(self, source_xy, table_z):
        for translation in self.roots:
            translation.Set(Gf.Vec3d(float(source_xy[0]), float(source_xy[1]), float(table_z)))

    def set_plate_support(self, plate_position, table_z):
        height = max(0.0001, float(plate_position[2] - table_z))
        for translation, scale in self.plate_supports:
            translation.Set(Gf.Vec3d(float(plate_position[0]), float(plate_position[1]), float(table_z + height / 2)))
            scale.Set(Gf.Vec3f(1.0, 0.85, height))


# Chip shapes

BASE_DIMENSIONS = np.array([0.064, 0.049, 0.0046])

FAMILIES = ("ellipse", "rounded_rectangle", "slender", "eccentric", "notched")

def _radial_factor(angle, shape):
    if shape.family == "rounded_rectangle":
        return (np.cos(angle)**4 + np.sin(angle)**4)**(-0.25)
    if shape.family == "eccentric":
        return 1 + shape.asymmetry * np.cos(angle) + 0.08 * np.sin(2 * angle)
    if shape.family == "notched":
        delta = np.angle(np.exp(1j * (angle - shape.notch_angle_rad)))
        return 1 - 0.24 * np.exp(-0.5 * (delta / 0.32)**2) + 0.06 * np.sin(3 * angle)
    return np.ones_like(angle)

def curvature_center_x(shape):
    if shape.curvature_origin not in ("canonical","area_centroid"):
        raise ValueError("Unknown physical chip curvature origin")
    if shape.curvature_origin=="canonical" or shape.family not in ("eccentric","notched"):
        return 0.
    dense=np.linspace(-np.pi,np.pi,4096,endpoint=False)
    boundary=np.column_stack((np.cos(dense),np.sin(dense)))*_radial_factor(dense,shape)[:,None]
    half_extent=np.ptp(boundary,axis=0)/2
    angles=np.arange(24)*2*np.pi/24
    outline=np.column_stack((np.cos(angles),np.sin(angles)))
    outline=outline*_radial_factor(angles,shape)[:,None]/half_extent*(np.asarray(shape.dimensions[:2])/2)
    following=np.roll(outline,-1,axis=0)
    cross=outline[:,0]*following[:,1]-following[:,0]*outline[:,1]
    return float(((outline[:,0]+following[:,0])*cross).sum()/(3*cross.sum()))

def warp_chip_points(points, shape):
    """Map canonical curved ellipse points without changing mesh connectivity."""
    if shape.family not in FAMILIES:
        raise ValueError(f"Unknown chip family: {shape.family}")
    dimensions = np.asarray(shape.dimensions, dtype=float)
    if (dimensions.shape != (3,) or not np.all(np.isfinite(dimensions))
            or np.any(dimensions <= 0) or not np.isfinite(shape.rise_m) or shape.rise_m <= 0):
        raise ValueError("Shape lengths must be finite and positive")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Rest points must be finite [N,3]")
    uv = points[:, :2] / (BASE_DIMENSIONS[:2] / 2)
    angle = np.arctan2(uv[:, 1], uv[:, 0])
    outlined = uv * _radial_factor(angle, shape)[:, None]
    # Normalize from a dense common outline, never from fragment-local bounds.
    angles = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
    boundary = np.column_stack((np.cos(angles), np.sin(angles))) * _radial_factor(angles, shape)[:, None]
    half_extent = np.ptp(boundary, axis=0) / 2
    xy = outlined / half_extent * (dimensions[:2] / 2)
    original_bottom = CHIP_CURVE_RISE_M * (points[:, 0] / (BASE_DIMENSIONS[0] / 2))**2
    height_in_shell = (points[:, 2] - original_bottom) * dimensions[2] / BASE_DIMENSIONS[2]
    if shape.curvature_origin not in ("canonical", "area_centroid"):
        raise ValueError("Unknown physical chip curvature origin")
    center_x = curvature_center_x(shape)
    z = shape.rise_m * ((xy[:, 0]-center_x) / (dimensions[0] / 2))**2 + height_in_shell
    return np.column_stack((xy, z))


# Chip shape visuals

class ChipShapeVisuals:
    def __init__(self, stage, actors):
        self.records = {}
        for actor in actors:
            meshes = []
            for path in actor.cfg.visual_prim_paths:
                for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                    if prim.IsA(UsdGeom.Mesh):
                        mesh = UsdGeom.Mesh(prim)
                        meshes.append((mesh, np.asarray(mesh.GetPointsAttr().Get(), dtype=float).copy()))
            self.records[id(actor)] = (actor, meshes)
        self.rest_points = {}

    def reset(self, shape):
        self.rest_points = {
            id(actor): [(mesh, warp_chip_points(original, shape)) for mesh, original in meshes]
            for actor, meshes in self.records.values()
        }
        for meshes in self.rest_points.values():
            for mesh, points in meshes:
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], dtype=np.float32)))
                mesh.GetNormalsAttr().Clear()

    def sync(self, active, fragments, fractured, seam_scale):
        for actor in (fragments if fractured else [active]):
            linear, translation = estimate_affine_deformation(actor.origin_surf_pts, actor.vertices)
            pose = actor.get_pose()
            for mesh, points in self.rest_points[id(actor)]:
                local = points * seam_scale if fractured else points
                deformed = (local @ linear.T + translation - pose.p) @ pose.R
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(deformed.astype(np.float32)))
                mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([deformed.min(0), deformed.max(0)], dtype=np.float32)))


# Chip shapes

@dataclass(frozen=True)
class ChipShape:
    family: str
    dimensions: tuple[float, float, float]
    rise_m: float
    asymmetry: float = 0.15
    notch_angle_rad: float = 0.75
    curvature_origin: str = "canonical"

@dataclass(frozen=True)
class FinalChipEpisode:
    shape: ChipShape
    yaw_rad: float
    # One shared material for the first calibration; no independent hidden
    # lottery of failure thresholds. Not a calibrated pressure in Pa.
    fracture_strain: float = 0.0008
    secure_strain: float = 0.000011
    variant: str = "medium"
    scale: float = 1.0

    @property
    def dimensions(self):
        return self.shape.dimensions

def sample_final_chip_episode(rng, *, geometry_range_scale=1.0, family=None, curvature_origin="canonical"):
    if not np.isfinite(geometry_range_scale) or not 1 <= geometry_range_scale <= 1.3:
        raise ValueError("Geometry range scale must be in [1, 1.3]")
    family = str(rng.choice(FAMILIES)) if family is None else family
    if family not in FAMILIES:
        raise ValueError(f"Unknown chip family: {family}")
    # Long-axis width directly affects fixed world-yaw jaw opening.
    ranges = {
        "ellipse": ((0.055, 0.071), (0.042, 0.053)),
        "rounded_rectangle": ((0.049, 0.066), (0.038, 0.050)),
        "slender": ((0.067, 0.075), (0.028, 0.037)),
        "eccentric": ((0.052, 0.069), (0.040, 0.052)),
        "notched": ((0.055, 0.071), (0.041, 0.054)),
    }
    length, width = (float(rng.uniform(*bounds)) for bounds in ranges[family])
    thickness = float(rng.uniform(0.0034, 0.0048))
    # Natural curvature supplies finger clearance above the actual tabletop.
    shape = ChipShape(family, (length, width, thickness),
                      float(rng.uniform(0.021, 0.025)),
                      float(rng.uniform(0.10, 0.20)), float(rng.uniform(0.4, 1.0)), curvature_origin)
    return FinalChipEpisode(shape, float(rng.uniform(-np.deg2rad(18), np.deg2rad(18))))

def shape_bottom_height(x,shape):
    return shape.rise_m*((np.asarray(x)-curvature_center_x(shape))/(shape.dimensions[0]/2))**2


# Chip tactile controller

@dataclass(frozen=True)
class ChipTactileControlLimits:
    close_target_depth_mm: float = 27.6
    weaker_pad_contact_mm: float = 28.5
    pad_depth_floor_mm: float = 26.7
    fast_step_m: float = 0.00035
    contact_step_m: float = 0.00005
    baseline_ticks: int = 8
    contact_confirm_ticks: int = 2
    contact_onset_min_rise_px: float = 0.012
    contact_onset_window: int = 4
    hold_ticks: int = 24
    max_marker_step_px: float = 0.35

def plate_support_flow_directions(shifts):
    """Separate upward support shear from lateral relaxation on vertical GSmini pads.

    Pixel x follows world -z on the left pad and +z on the right pad. A
    supporting plate unloads downward shear, giving left -x / right +x flow.
    Each pad is checked independently; unequal responses are allowed.
    """
    if set(shifts) != {"left_tactile", "right_tactile"}:
        raise ValueError("Both named tactile pads are required")
    result = {}
    for name, sign in (("left_tactile", -1.0), ("right_tactile", 1.0)):
        vector = np.asarray(shifts[name].get("median_vector_px", []), dtype=float)
        valid = vector.shape == (2,) and bool(np.all(np.isfinite(vector)))
        axial = float(sign * vector[0]) if valid else None
        lateral = float(abs(vector[1])) if valid else None
        result[name] = dict(valid=valid, axial_support_px=axial,
                            lateral_px=lateral,
                            support_direction=bool(valid and axial > lateral))
    return result

class ChipTactileController:
    def __init__(self, limits=None, *, directional_plate_contact=False):
        if not isinstance(directional_plate_contact, bool):
            raise ValueError("directional_plate_contact must be a boolean")
        self.directional_plate_contact = directional_plate_contact
        self.limits = limits or ChipTactileControlLimits()
        self.reference = None
        self.previous = None
        self.background = {"left_tactile": [], "right_tactile": []}
        self.thresholds = None
        self.confirm_ticks = 0
        self.previous_contact_shifts = None
        self.previous_contact_vectors = None
        self.directional_confirm_ticks = {}
        self.contact_onsets = {}
        self.contact_onset_diagnostic = {}

    def close_decision(self, depths):
        depths = np.asarray(depths, dtype=float)
        if depths.shape != (2,) or not np.all(np.isfinite(depths)):
            return {"action": "abort", "reason": "invalid_tactile"}
        lower, upper = float(depths.min()), float(depths.max())
        if lower <= self.limits.pad_depth_floor_mm:
            return {"action": "abort", "reason": "pad_depth_floor"}
        if lower <= self.limits.close_target_depth_mm and upper < self.limits.weaker_pad_contact_mm:
            return {"action": "hold", "reason": "tactile_contact_acquired"}
        step = self.limits.fast_step_m if lower >= 33.0 else self.limits.contact_step_m
        if upper < 28.3:
            step /= 2
        return {"action": "close", "step_m": step}

    def begin_placement(self, markers):
        self.reference = {key: np.array(value, copy=True) for key, value in markers.items()}
        self.previous = self.reference
        self.background = {key: [] for key in markers}
        self.thresholds = None
        self.confirm_ticks = 0
        self.previous_contact_shifts = None
        self.previous_contact_vectors = None
        self.directional_confirm_ticks = {}
        self.contact_onsets = {}
        self.contact_onset_diagnostic = {}

    def placement_shifts(self, markers):
        return {key: marker_contact_shift(self.reference[key], value) for key, value in markers.items()}

    def baseline(self, markers):
        shifts = self.placement_shifts(markers)
        for key, shift in shifts.items():
            self.background[key].append(shift["p90_px"])
        self.previous_contact_shifts = {key: shift["p90_px"] for key,shift in shifts.items()}
        self.previous_contact_vectors = {
            key: np.asarray(shift["median_vector_px"], dtype=float).copy()
            for key, shift in shifts.items()
        }
        if min(map(len, self.background.values())) >= self.limits.baseline_ticks:
            self.thresholds = marker_detection_thresholds(self.background)
        return shifts

    def contact_decision(self, markers):
        if self.thresholds is None:
            raise RuntimeError("Record the pre-contact tactile baseline before lowering")
        shifts = self.placement_shifts(markers)
        detected = self.contact_from_shifts(shifts)
        self.previous = {key: np.array(value, copy=True) for key, value in markers.items()}
        return detected, shifts

    def contact_from_shifts(self, shifts):
        """Recognize a load onset; slow accumulated free-space drift is insufficient."""
        previous = self.previous_contact_shifts or {key: shift["p90_px"] for key,shift in shifts.items()}
        diagnostics = {}
        for key,shift in shifts.items():
            rise = float(shift["p90_px"])-float(previous[key])
            history = self.contact_onsets.setdefault(key, [])
            history.append(bool(np.isfinite(rise) and rise >= self.limits.contact_onset_min_rise_px))
            del history[:-self.limits.contact_onset_window]
            diagnostics[key] = dict(rise_px=rise,recent_onset=any(history))
        level = marker_signal_meets(shifts,self.thresholds,min_active_pads=1)
        if self.directional_plate_contact:
            previous_vectors = self.previous_contact_vectors or {}
            increments = {}
            current_vectors = {}
            for key, shift in shifts.items():
                vector = np.asarray(shift.get("median_vector_px", []), dtype=float)
                prior = np.asarray(previous_vectors.get(key, [0.0, 0.0]), dtype=float)
                valid = (vector.shape == prior.shape == (2,)
                         and bool(np.all(np.isfinite(vector)))
                         and bool(np.all(np.isfinite(prior))))
                increments[key] = {"median_vector_px": vector - prior if valid else []}
                current_vectors[key] = vector.copy()
            directional = plate_support_flow_directions(increments)
            level = level and all(value["valid"] for value in directional.values())
            for key, value in directional.items():
                eligible = bool(level and value["support_direction"]
                    and value["axial_support_px"] >= self.limits.contact_onset_min_rise_px
                    and shifts[key]["p90_px"] >= self.thresholds[key])
                self.directional_confirm_ticks[key] = (
                    self.directional_confirm_ticks.get(key, 0) + 1 if eligible else 0)
                diagnostics[key].update(value)
                diagnostics[key].update(
                    direction_measure="increment_between_observations",
                    consecutive_support_onsets=self.directional_confirm_ticks[key])
            self.previous_contact_vectors = current_vectors
            self.confirm_ticks = max(self.directional_confirm_ticks.values(), default=0)
        else:
            detected = level and any(
                shift["p90_px"] >= self.thresholds[key] and diagnostics[key]["recent_onset"]
                for key,shift in shifts.items())
            self.confirm_ticks = self.confirm_ticks+1 if detected else 0
        self.contact_onset_diagnostic = diagnostics
        self.previous_contact_shifts = {key: shift["p90_px"] for key,shift in shifts.items()}
        return self.confirm_ticks >= self.limits.contact_confirm_ticks

    def hold_decision(self, markers, depths):
        changes = {key: marker_contact_shift(self.previous[key], value) for key, value in markers.items()}
        self.previous = {key: np.array(value, copy=True) for key, value in markers.items()}
        depths = np.asarray(depths, dtype=float)
        safe = (depths.shape == (2,) and np.all(np.isfinite(depths))
                and np.all(depths > self.limits.pad_depth_floor_mm)
                and np.all(depths < 31.0)
                and all(value["p90_px"] <= self.limits.max_marker_step_px for value in changes.values()))
        return bool(safe), changes

    def description(self):
        return {"version": 6 if self.directional_plate_contact else 4,
                "directional_plate_contact": self.directional_plate_contact, "inputs": ["raw_tactile_depth_mm", "raw_marker_coordinates"],
                "limits": asdict(self.limits), "calibration_status": "candidate_pending_prospective_matrix",
                "plate_contact_rule": (
                    "same valid pad exceeds its magnitude threshold with two consecutive upward marker increments; lateral cumulative drift is excluded; bilateral grip retained"
                    if self.directional_plate_contact else
                    "either valid pad exceeds its baseline threshold on two observations with a recent raw-marker load onset; slow drift alone is insufficient; bilateral grip retained")}

def edge_contact_needs_closing(patches, *, minimum_interior_area_px=500, edge_fraction_limit=.30):
    """Small contact at a pad edge needs more wrap before transport.

    Raw contact masks supply this local coverage test. The two pad areas and
    depths may remain different; no material value, force or actor pose is used.
    """
    if set(patches)!={"left_tactile","right_tactile"}:
        raise ValueError("Both raw tactile contact patches are required")
    needs=False
    for patch in patches.values():
        area=float(patch["area_px"])
        edge=float(patch["edge_fraction"]) if patch["edge_fraction"] is not None else float("nan")
        if not np.isfinite(area) or area<=0 or not np.isfinite(edge) or not 0<=edge<=1:
            raise ValueError("Invalid raw tactile contact patch")
        needs |= edge>=edge_fraction_limit and area*(1-edge)<minimum_interior_area_px
    return bool(needs)

def maintain_grasp_decision(depths, reference, *, deadband_mm=0.08,
                            pad_depth_floor_mm=26.7, step_m=0.00001):
    """Restore bilateral unloading relative to each pad's own settled grasp.

    Asymmetric reference depths remain asymmetric. A pad that is already more
    loaded blocks additional closing; physical damage remains independently
    monitored. This rule is only used before the placement marker baseline.
    """
    current=np.asarray(depths,dtype=float)
    reference=np.asarray(reference,dtype=float)
    if (current.shape!=(2,) or reference.shape!=(2,)
            or not np.all(np.isfinite(current)) or not np.all(np.isfinite(reference))):
        return dict(action="abort",reason="invalid_tactile_grip_maintenance")
    if np.min(current)<=pad_depth_floor_mm+0.02:
        return dict(action="hold",reason="pad_depth_margin")
    if np.all(current-reference>deadband_mm):
        return dict(action="close",step_m=step_m,reason="bilateral_grasp_unloading")
    return dict(action="hold",reason="grasp_reference_retained")


# Grasp fragile chip

ASSET_ROOT = "task_assets/tactile_first_suite"

CHIP_APPROACH_ACCELERATION_M_S2 = 1.80

CHIP_APPROACH_SERVO_TOLERANCE_M = 0.0010

CHIP_APPROACH_SPEED_M_S = 0.30

CHIP_CARRY_SPEED_M_S = 0.32

CHIP_DEPTH_CRUSH_FLOOR_MM = 25.2

CHIP_FRACTURE_AFTERMATH_STEPS = 90

CHIP_FREE_DEPTH_MM = 31.0

CHIP_GRIP_FRICTION_RATIO = 20.0

CHIP_GRIP_TARGET_DEPTH_MM = 27.6

CHIP_IMPACT_SPEED_M_S = 0.09  # calibration candidate, shared by all scenarios

CHIP_LIFT_SPEED_M_S = 0.22

CHIP_LIFT_Z = 0.105

CHIP_LOWER_SPEED_M_S = 0.16

CHIP_PLACEMENT_APPROACH_CLEARANCE_M = 0.0015

CHIP_PLACEMENT_CONTACT_DEPTH_MAX_MM = 32.0

CHIP_PLACEMENT_CONTACT_DEPTH_MIN_MM = 27.0

CHIP_PLACEMENT_CONTACT_MIN_PADS = 2

CHIP_PLACEMENT_MARKER_SHIFT_PX = 1.25

CHIP_PLACEMENT_MAX_OVERTRAVEL_M = 0.0006

CHIP_PLACEMENT_MAX_STRAIN_MARGIN = 0.60

CHIP_PLACEMENT_TOUCH_ACCELERATION_M_S2 = 0.05

CHIP_PLACEMENT_TOUCH_SPEED_M_S = 0.006

CHIP_PRESENTATION_YAW_SPIN_RAD = np.deg2rad(-90.0)

CHIP_START_XY = np.array([0.475, -0.075], dtype=np.float64)

CHIP_TRANSPORT_ABORT_ROTATION_RAD = np.deg2rad(25.0)

CHIP_TRANSPORT_ABORT_TRANSLATION_M = 0.015

CHIP_TRANSPORT_ACCELERATION_M_S2 = 0.90

CHIP_TRANSPORT_MAX_JOINT_STEP_RAD = 0.025

CHIP_TRANSPORT_MAX_ORIENTATION_ERROR_RAD = np.deg2rad(3.0)

CHIP_TRANSPORT_MAX_PATH_ERROR_M = 0.006

CHIP_TRANSPORT_MAX_ROTATION_RAD = np.deg2rad(13.0)

CHIP_TRANSPORT_MAX_TRANSLATION_M = 0.006

CHIP_TRANSPORT_MIN_FORCE_PROXY = 0.0

CHIP_TRANSPORT_MIN_SECURE_SAMPLES = 10

CHIP_TRANSPORT_SETTLE_STEPS = 4

CHIP_VISUAL_COLOR = (0.86, 0.57, 0.20)

CHIP_XY_NOISE_M = 0.030

FRACTURE_CLOSE_STRAIN_MARGIN = 0.000005

FRAGMENT_STANDBY_Y = (1.05, 1.20, 1.35, 1.50, 1.65, 1.80)

PEDESTAL_PARK_POSE = Pose([0.46, -1.65, 0.45], [1, 0, 0, 0])

RECENTER_MIN_STEPS_BETWEEN = 4     # close steps between nudges

STANDBY_X = 0.46

STANDBY_Y = {"small": -1.05, "medium": -1.20, "large": -1.35}

STANDBY_Z = 0.45

TABLE_START_CLEARANCE_M = 0.0004

TABLE_TOP_Z = 0.0025

TASK_INITIAL_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
}

TRAY_HEIGHT_NOISE_M = 0.015

TRAY_POSE = Pose([0.555, 0.155, TABLE_TOP_Z + TABLE_START_CLEARANCE_M], [1, 0, 0, 0])

TRAY_TARGET = TRAY_POSE.p + np.array([0.0, 0.0, 0.0065])

TRAY_TARGET_Z_OFFSET_M = 0.0065

TRAY_TARGET_Z_OFFSET_MAX_M = 0.0085

TRAY_TARGET_Z_OFFSET_MIN_M = 0.0045

TRAY_VISUAL_COLOR = (0.72, 0.78, 0.84)

TRAY_XY_NOISE_M = 0.030

@configclass
class TaskCfg(BaseTaskCfg):
    video_size = (1120, 320)
    step_lim = 1600
    max_save_frames = 650
    reset_time_limit = 1500.0
    reset_first_frame_steps = 5
    reset_after_actor_steps = 20
    reset_final_steps = 16
    reset_render_warmup_steps = 8
    use_adaptive_grasp = True
    # Firmer transport hold (A/B): a config-gated second-stage close after the
    # strain-gated secure stop. The staged close stops on chip strain, which
    # leaves the shallow-seated seeds (small scale -> thin chip edge) with a
    # small gel contact patch even though the gripper is already at its
    # calibrated closed pose (measured: seed-0 weaker pad ~860 px vs seed 3
    # ~6.4k px at the same joint). This press keeps tightening until the
    # weaker pad's contact area reaches firm_press_target_area, so the shallow
    # seeds hold firmer during transport while a deep-seated seed already
    # above the target presses ~0 (no crack risk). Safety: aborts cleanly at
    # FRACTURE_CLOSE_STRAIN_MARGIN below fracture_strain, at the gel crush
    # floor, or after firm_press_max_steps.
    chip_randomization_scale: float = 1.0
    uniform_policy_recording: bool = False
    chip_edge_contact_guard: bool = False
    firm_transport_press: bool = False
    firm_press_target_area: int = 6000        # min-pad px below the threshold
    firm_press_area_threshold_mm: float = 33.0  # depth < this = contact (rest 34.0)
    firm_press_step: float = 0.00005          # m per close step (matches the close)
    firm_press_max_steps: int = 300
    planner_ignore_actors: tuple[str, ...] = (
        "fragile_chip_small",
        "fragile_chip_medium",
        "fragile_chip_large",
        "chip_fragment_0",
        "chip_fragment_1",
        "chip_fragment_2",
        "chip_fragment_3",
        "chip_fragment_4",
        "chip_fragment_5",
        "chip_presentation_pedestal",
        "chip_landing_tray",
    )
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.88, -0.03, 0.36),
                rot=(0.676210, 0.206402, 0.206402, 0.676210),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.1,
                focus_distance=0.7,
                horizontal_aperture=3.0,
                clipping_range=(0.08, 100.0),
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]


# Tactile first utils

CHIP_VARIANTS = {
    "small": (0.058, 0.044, 0.0042),
    "medium": (0.064, 0.049, 0.0046),
    "large": (0.070, 0.054, 0.0050),
}

def classify_fragile_strain(
    strain: float,
    *,
    secure_strain: float,
    fracture_strain: float,
) -> str:
    """Classify the chip's reshape strain against per-episode thresholds.

    strain >= fracture_strain -> 'fractured'; secure_strain <= strain <
    fracture_strain -> 'secure'; strain < secure_strain -> 'loose'. Strain
    grows with load (the inverse of depth, which shrinks), so the thresholds
    must be strictly ordered fracture > secure. Raises ValueError on inverted
    ordering or a non-finite strain sample (fail closed).
    """
    if fracture_strain <= secure_strain:
        raise ValueError("fracture strain must exceed the secure-contact strain")
    if strain is None or not np.isfinite(float(strain)):
        raise ValueError("strain must be a finite scalar")
    if float(strain) >= fracture_strain:
        return "fractured"
    if float(strain) >= secure_strain:
        return "secure"
    return "loose"

def transport_force_contact_maintained(
    force_values: list[float] | None,
    *,
    min_secure_samples: int,
    min_force_proxy: float = 0.0,
) -> bool:
    """Force-proxy transport gate: sustained UIPC contact on the carried actor.

    True iff at least `min_secure_samples` force-proxy samples exist and at
    least that many are above `min_force_proxy`. The force proxy
    (`contact_force_on_actor`) reads > 0 on ANY UIPC contact, edge or center,
    while the GelSight depth band far-planes for a chip gripped at the gel edge
    (outside the internal camera's optical region). Its magnitude is an
    uncalibrated sum of vertex gradient norms that varies ~30x across otherwise
    identical runs (large chip: 0.0006-0.018; medium: 0.0014), so gate on
    PRESENCE (> 0), not scale. A dropped actor has no contact vertices -> exact
    0.0. None/empty/malformed input -> False (drop = fail).
    """
    if not force_values:
        return False
    present = sum(
        1 for f in force_values if f is not None and f > min_force_proxy
    )
    return (
        len(force_values) >= min_secure_samples
        and present >= min_secure_samples
    )

def transport_strain_maintained(
    strain_trace,
    tracking_errors,
    *,
    secure_strain: float,
    fracture_strain: float,
    max_translation_error_m: float,
    max_rotation_error_rad: float,
    min_secure_samples: int,
) -> bool:
    """Strain-based transport gate: sustained secure reshape + bounded drift.

    Mirrors transport_contact_maintained but classifies the chip's OWN
    deformation strain (chip-side reshape evidence, dimensionless) instead of
    pad depth. True iff: >= min_secure_samples strain samples classify 'secure'
    via classify_fragile_strain, zero classify 'fractured', and the p95 of
    tracking_errors columns stays under the two limits. tracking_errors is
    (N, 2): col 0 = translation error (m), col 1 = rotation error (rad).
    Missing/malformed input -> False (drop = fail). Threshold ordering is a
    programming error -> ValueError; per-sample bad data -> fail closed False.
    """
    if fracture_strain <= secure_strain:
        raise ValueError("fracture strain must exceed the secure-contact strain")
    if strain_trace is None or len(strain_trace) == 0:
        return False
    if tracking_errors is None:
        return False
    errors = np.asarray(tracking_errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] < 2:
        return False
    if errors.shape[0] == 0 or not np.all(np.isfinite(errors)):
        return False
    secure_samples = 0
    for sample in strain_trace:
        try:
            state = classify_fragile_strain(
                sample,
                secure_strain=secure_strain,
                fracture_strain=fracture_strain,
            )
        except ValueError:
            return False
        if state == "fractured":
            return False
        secure_samples += int(state == "secure")
    if secure_samples < min_secure_samples:
        return False
    if float(np.percentile(errors[:, 0], 95)) > max_translation_error_m:
        return False
    if float(np.percentile(errors[:, 1], 95)) > max_rotation_error_rad:
        return False
    return True

def von_mises_strain(F: np.ndarray) -> float:
    """Scalar von Mises (equivalent) strain from an affine deformation gradient.

    B = F.T @ F; E = 0.5 (B - I); returns sqrt(2/3 * sum(E**2)). Invariant to
    rigid rotation (F.T @ F strips the orthogonal part) so it measures
    stretch-only reshape and is BLIND to whole-body rotation (pair it with an
    in-hand rotation gate). Raises ValueError if F is not 3x3 or non-finite.
    """
    F = np.asarray(F, dtype=np.float64)
    if F.shape != (3, 3):
        raise ValueError("F must be a 3x3 deformation gradient")
    if not np.all(np.isfinite(F)):
        raise ValueError("F must be finite")
    B = F.T @ F
    E = 0.5 * (B - np.eye(3))
    return float(np.sqrt(2.0 / 3.0 * np.sum(E * E)))


# Tactile goals

def record_tactile_sample(
    task: BaseTask,
    *,
    tag: str,
    depths_mm: np.ndarray | None = None,
    force_proxy: float | None = None,
    inhand_translation_m: float | None = None,
    inhand_rotation_rad: float | None = None,
) -> dict:
    """Append one per-step tactile sample to task.tactile_traces[tag].

    Creates task.tactile_traces lazily (a dict of lists keyed by phase tag) and
    appends a sample dict {tag, depth_mm (list|None, mm), force_proxy,
    inhand_translation_m, inhand_rotation_rad}. Returns the sample so callers can
    extend it with mission-specific keys. Unifies the chip's flat
    tactile_depth_trace and the peeler's per-step dict into one recorder.
    """
    traces = getattr(task, "tactile_traces", None)
    if traces is None:
        traces = {}
        task.tactile_traces = traces
    sample = {
        "tag": tag,
        "depth_mm": (
            None
            if depths_mm is None
            else np.asarray(depths_mm, dtype=np.float64).reshape(-1).tolist()
        ),
        "force_proxy": (
            None
            if force_proxy is None
            else float(np.asarray(force_proxy, dtype=np.float64))
        ),
        "inhand_translation_m": (
            None if inhand_translation_m is None else float(inhand_translation_m)
        ),
        "inhand_rotation_rad": (
            None if inhand_rotation_rad is None else float(inhand_rotation_rad)
        ),
    }
    traces.setdefault(tag, []).append(sample)
    return sample


# Grasp fragile chip

class Task(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode=None,
        **kwargs,
    ):
        self.chip_randomization_scale = float(task_parameters(cfg, require_nonnegative_seed=False).get("chip_randomization_scale", cfg.chip_randomization_scale))
        if not np.isfinite(self.chip_randomization_scale) or not 1.0 <= self.chip_randomization_scale <= 1.3:
            raise ValueError("Chip randomization scale must be finite and in [1.0, 1.3]")
        # User-selected support model: one global plane at the visible tabletop.
        cfg.uipc_sim.ground_height = TABLE_TOP_Z
        cfg.sim.physics_material.dynamic_friction = 1.9
        cfg.sim.physics_material.static_friction = 2.2
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        cfg.uipc_sim.contact.d_hat = 0.00035
        # First-solve Newton cap (diagnosed 2026-08-26): the post-teleport settle
        # in _reset_actors stages 9 UIPC bodies into dense contact (~29k contacts)
        # that genuinely needs ~1000 Newton iterations to reach velocity_tol. At
        # the default max_iter=1024 that is ~44 s/step, which trips the reset
        # step-section guard and fails every seed. Capping at 64 lets the
        # line-search-robust solve traverse (~3 s/step); the settle completes
        # across the 35 reset steps and steady-state converges at Newton iter 3-5
        # (verified: seed 0 success in 152 s, converges at iter 0-5 by mid-episode).
        solver_parameters = task_parameters(cfg, require_nonnegative_seed=False)
        edge_guard=solver_parameters.get('chip_edge_contact_guard',cfg.chip_edge_contact_guard)
        if not isinstance(edge_guard,bool):
            raise ValueError('chip_edge_contact_guard must be a boolean task parameter')
        adaptive = solver_parameters.get('chip_edge_adaptive_control', False)
        if not isinstance(adaptive, bool):
            raise ValueError('chip_edge_adaptive_control must be a boolean task parameter')
        if adaptive and cfg.tactile_sensor_type != 'gsmini':
            raise ValueError('Adaptive edge grasp currently requires the vertical GSmini mounts')
        self.chip_edge_adaptive_control = adaptive
        cfg.chip_edge_contact_guard = edge_guard or adaptive
        maintenance=solver_parameters.get('chip_transport_grip_maintenance',False)
        if not isinstance(maintenance,bool):
            raise ValueError('chip_transport_grip_maintenance must be a boolean task parameter')
        self.chip_transport_grip_maintenance=maintenance
        self._transport_grip_depth_reference=None
        directional = solver_parameters.get('chip_directional_plate_contact', False)
        if not isinstance(directional, bool):
            raise ValueError('chip_directional_plate_contact must be a boolean task parameter')
        if directional and cfg.tactile_sensor_type != 'gsmini':
            raise ValueError('Directional plate contact currently requires the vertical GSmini mounts')
        self.chip_directional_plate_contact = directional
        self.chip_placement_approach_clearance_m = float(solver_parameters.get(
            'chip_placement_approach_clearance_m', CHIP_PLACEMENT_APPROACH_CLEARANCE_M))
        if (not np.isfinite(self.chip_placement_approach_clearance_m)
                or not CHIP_PLACEMENT_APPROACH_CLEARANCE_M <= self.chip_placement_approach_clearance_m <= 0.04):
            raise ValueError('Chip pre-contact approach clearance must be finite and in [0.0015, 0.04] m')
        iteration_limit = solver_parameters.get('newton_max_iter', 64)
        if int(iteration_limit) != iteration_limit or iteration_limit < 1:
            raise ValueError('Newton iteration limit must be a positive integer')
        cfg.uipc_sim.newton.max_iter = int(iteration_limit)
        # The default0.05m/s tolerance permits0.417mm residual motion per
        # physics tick, comparable to the0.6mm plate overtravel limit.
        # Resolve slow contact at8.33um/tick as in the validated bulb solver.
        cfg.uipc_sim.newton.velocity_tol = float(solver_parameters.get('newton_velocity_tol_m_s', 0.001))
        if not np.isfinite(cfg.uipc_sim.newton.velocity_tol) or cfg.uipc_sim.newton.velocity_tol <= 0:
            raise ValueError('Newton velocity tolerance must be finite and positive')
        solver_log = solver_parameters.get('solver_log_level', 'Error')
        if solver_log not in ('Error', 'Info'):
            raise ValueError('Unsupported UIPC diagnostic log level')
        cfg.uipc_sim.logger_level = solver_log
        self.monitor_chip_contact = False
        self.fractured = False
        self._fragment_release_pending = False
        self._chip_damage_armed = False
        self._chip_previous_center = None
        self._chip_previous_velocity_z = 0.0
        self._chip_impact_tracker = SupportImpactTracker(CHIP_IMPACT_SPEED_M_S)
        self.tactile_strain_trace = []
        self._chip_strain_reference_verts = None
        self._chip_strain_baseline = None
        self.tray_pose = TRAY_POSE
        self.tray_target = TRAY_TARGET.copy()
        self.tray_target_z_offset = TRAY_TARGET_Z_OFFSET_M
        self.placement_contact_trace = []
        from ._force_task_utils import configure_final_task
        configure_final_task(cfg, task_parameters(cfg, require_nonnegative_seed=False), max_policy_seconds=60)
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._chip_rest_geometry = ChipRestGeometry(
            self.uipc_sim, list(self._actor_manager.actors.values())
        )

    def seed(self, seed=-1):
        super().seed(seed)
        self.episode = sample_final_chip_episode(
            self.rng, geometry_range_scale=self.chip_randomization_scale,
            family=task_parameters(self.cfg, require_nonnegative_seed=False).get("chip_shape_family"),
            curvature_origin=task_parameters(self.cfg, require_nonnegative_seed=False).get("chip_curvature_origin","area_centroid"))
        self.monitor_chip_contact = False
        self._chip_damage_armed = False
        self._fragment_release_pending = False
        self.fractured = False
        self.first_frame = None
        scale = np.asarray(self.episode.dimensions) * self.episode.scale / np.asarray(CHIP_VARIANTS['medium'])
        self._chip_rest_geometry.rebuild(
            [self.chips['medium'], *self.fragments], scale,
            warp=lambda points: warp_chip_points(points, self.episode.shape))

    def reset(self, *args, **kwargs):
        self._action_monitor = None
        result = super().reset(*args, **kwargs)
        self.metadata['robot_initial_joint_pos'] = self._robot_manager.robot.data.joint_pos.detach().cpu().numpy().tolist()
        self.metadata['robot_initial_gripper_pose'] = self._robot_manager.get_gripper_center_pose().tolist()
        self.metadata["chip_randomization_scale"] = self.chip_randomization_scale
        self.metadata["chip_randomization_ranges"] = {
            "xy_half_range_m": CHIP_XY_NOISE_M * self.chip_randomization_scale,
            "target_height_noise_m": 0.0,
            "tray_height_range_m": [0.0, TRAY_HEIGHT_NOISE_M * self.chip_randomization_scale],
            "yaw_half_range_deg": 18.0,
            "shape_family": self.episode.shape.family,
            "hidden_fracture_threshold_changed": False,
            "grasp_anchor_is_fixed": False,
        }
        self._record_chip_dimensions('initial')
        self.metadata['free_reset_support_trace'] = self._reset_support_trace
        self.metadata['chip_settled_pose'] = self.active_chip.get_pose().tolist()
        self.metadata['chip_settled_min_z_m'] = float(self.active_chip.vertices[:, 2].min())
        self.metadata['chip_settled_xy_drift_m'] = float(np.linalg.norm(
            self.active_chip.get_pose().p[:2] - self.chip_start_pose.p[:2]))
        from uipc import builtin
        constraints = self.active_chip.geo_slot_list[0].geometry().instances().find(builtin.is_constrained)
        self.metadata['chip_constrained_at_policy_start'] = bool(np.any(constraints.view()))
        self._chip_strain_reference_verts = self.active_chip.vertices.copy()
        self._chip_strain_baseline = self._chip_raw_strain()
        self._chip_damage_armed = True
        self.monitor_chip_contact = True
        self._execution_reason = ''
        self._action_monitor = ChipPolicyMonitor(self)
        self._chip_tactile_controller = ChipTactileController(
            directional_plate_contact=self.chip_directional_plate_contact)
        if self.chip_edge_adaptive_control:
            self.chip_transport_grip_maintenance = False
            self._chip_tactile_controller.directional_plate_contact = False
        self.metadata['tactile_controller'] = self._chip_tactile_controller.description()
        return result

    def _record_chip_dimensions(self, phase):
        pose = self.active_chip.get_pose()
        local = (self.active_chip.vertices - pose.p) @ pose.R
        self.metadata[f'chip_measured_{phase}_extents_m'] = np.ptp(local, axis=0).tolist()

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        cfg.robot.robot.init_state.joint_pos["panda_finger.*"] = (
            cfg.robot.gripper_max_qpos
        )
        return cfg

    def create_actors(self):
        fixed = UipcObjectCfg.AffineBodyConstitutionCfg(
            m_kappa=260.0,
            # NOT kinematic: a kinematic (is_fixed) affine body stays pinned to
            # its authored init pose and ignores set_pose's aim_transform, so the
            # per-episode pedestal/tray height+xy randomization would never move
            # it and the chip would float above an unmoved pedestal (falls during
            # the close -> pads misaligned -> miss/slam). These fixtures are never
            # unconstrained in this task, so the kinematic "hold pose when freed"
            # property is unused here.
            kinematic=False,
        )
        self.tray = self._actor_manager.add_from_usd_file(
            name="chip_landing_tray",
            asset_path=f"{ASSET_ROOT}/chip_landing_tray.usd",
            visual_asset_path=f"{ASSET_ROOT}/chip_serving_plate_visual.usda",
            pose=TRAY_POSE,
            constitution_cfg=fixed,
            density=1800.0,
            show_physics_mesh=False,
            keep_constrained=True,
        )
        self.pedestal = self._actor_manager.add_from_usd_file(
            name="chip_presentation_pedestal",
            asset_path=f"{ASSET_ROOT}/chip_presentation_pedestal.usd",
            visual_asset_path=(
                f"{ASSET_ROOT}/chip_spill_stack_visual.usda"
            ),
            pose=PEDESTAL_PARK_POSE,
            constitution_cfg=fixed,
            density=1800.0,
            show_physics_mesh=False,
            keep_constrained=True,
        )

        self._spawn_chip_box_visual()
        set_actor_visible(self._kitchen_scene.stage, self.pedestal, False)
        self._kitchen_scene.set_plate_support(TRAY_POSE.p, TABLE_TOP_Z)

        self.chips = {}
        for variant in CHIP_VARIANTS:
            standby = self._chip_standby_pose(variant)
            self.chips[variant] = self._actor_manager.add_from_usd_file(
                name=f"fragile_chip_{variant}",
                asset_path=f"{ASSET_ROOT}/chip_curved_{variant}.usda",
                visual_asset_path=(
                    f"{ASSET_ROOT}/chip_curved_{variant}_visual.usda"
                ),
                pose=standby,
                constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                    # Long-diagonal torsional grip: a stiffer affine body twists
                    # less under the transport's lateral loads -> less in-hand
                    # yaw and less transport depth-dive (the fracture trigger).
                    # Raised from 72 (soft end of the task family's 90-130 band).
                    m_kappa=150.0
                ),
                # Potato-chip mass: real chip material is ~1 g/mL = 1000 kg/m^3
                # (user-confirmed; supersedes the earlier 165 / 100 values).
                # At realistic mass the gripper-acceleration inertial flex loads
                # the chip more heavily, so the fracture band is loosened to sit
                # above the transport/placement strain envelope (calibrated from
                # the density-1000 collect; m_kappa stays at the validated 150).
                density=1000.0,
                show_physics_mesh=False,
                keep_constrained=True,
            )

        self.fragments = []
        for index, standby_y in enumerate(FRAGMENT_STANDBY_Y):
            fragment = self._actor_manager.add_from_usd_file(
                name=f"chip_fragment_{index}",
                asset_path=f"{ASSET_ROOT}/chip_curved_shard_{index}.usda",
                visual_asset_path=(
                    f"{ASSET_ROOT}/chip_curved_shard_{index}_visual.usda"
                ),
                pose=Pose([STANDBY_X, standby_y, STANDBY_Z], [1, 0, 0, 0]),
                constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                    m_kappa=58.0
                ),
                density=1000.0,
                show_physics_mesh=False,
                keep_constrained=True,
            )
            self.fragments.append(fragment)
        self._setup_chip_visual_scale_ops()

    def _spawn_chip_box_visual(self):
        import omni.usd

        self._kitchen_scene = ChipKitchenScene(
            omni.usd.get_context().get_stage(),
            getattr(self.scene, "env_prim_paths", ["/World/envs/env_0"]),
            OBJECTS_ROOT / ASSET_ROOT,
        )

    @staticmethod
    def _chip_standby_pose(variant: str) -> Pose:
        return Pose(
            [STANDBY_X, STANDBY_Y[variant], STANDBY_Z],
            [1, 0, 0, 0],
        )

    def _setup_chip_visual_scale_ops(self):
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        self._chip_visual_scale_ops = {}
        for variant, chip in self.chips.items():
            ops = []
            for prim_path in chip.cfg.visual_prim_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                scale_op = UsdGeom.Xformable(prim).AddScaleOp(
                    UsdGeom.XformOp.PrecisionDouble
                )
                scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*CHIP_VISUAL_COLOR)])
                ops.append(scale_op)
            self._chip_visual_scale_ops[variant] = ops

    def _pose_active_chip_scaled(self):
        from pxr import Gf, UsdGeom
        self.active_chip.set_pose(self.chip_start_pose)
        if not hasattr(self, '_shape_visuals'):
            self._shape_visuals = ChipShapeVisuals(
                self._kitchen_scene.stage, [self.chips['medium'], *self.fragments])
        self._shape_visuals.reset(self.episode.shape)
        for actor in [self.active_chip, *self.fragments]:
            for path in actor.cfg.visual_prim_paths:
                xform = UsdGeom.Xformable(self._kitchen_scene.stage.GetPrimAtPath(path))
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        op.Set(Gf.Vec3d(1, 1, 1))

    def _reset_actors(self):
        import omni.usd
        from pxr import Gf, UsdGeom

        self.fractured = False
        self.monitor_chip_contact = False
        self._fragment_release_pending = False
        self.grasp_state = "loose"
        self.secure_grasp_seen = False
        self.placed_in_tray = False
        self._action_monitor = None
        self._reset_support_trace = None
        self.tactile_depth_trace = []
        self.grasp_lost = False
        self.tactile_traces = {}
        self.placement_contact_trace = []
        self._chip_inhand_pose = None
        self._placement_marker_reference = None
        self._placement_marker_confirm_steps = 0
        self._placement_marker_background = {'left_tactile': [], 'right_tactile': []}
        self._placement_marker_detection_thresholds = None
        self._placement_contact_ee_z = None
        self._chip_damage_armed = False
        self._chip_previous_center = None
        self._chip_previous_velocity_z = 0.0
        self._chip_impact_tracker = SupportImpactTracker(CHIP_IMPACT_SPEED_M_S)
        self.fracture_step = None
        self._recenter_count = 0
        self._last_recenter_step = -RECENTER_MIN_STEPS_BETWEEN
        self._chip_freed_early = False
        self.tactile_strain_trace = []
        self._chip_strain_reference_verts = None
        self._chip_strain_baseline = None

        shared_xy_noise = self.rng.uniform(-CHIP_XY_NOISE_M, CHIP_XY_NOISE_M, size=2) * self.chip_randomization_scale
        pedestal_z_noise = 0.0
        yaw_q = t3d.euler.euler2quat(
            0.0,
            0.0,
            self.episode.yaw_rad + CHIP_PRESENTATION_YAW_SPIN_RAD,
        )
        self.chip_start_pose = Pose(
            [
                CHIP_START_XY[0] + shared_xy_noise[0],
                CHIP_START_XY[1] + shared_xy_noise[1],
                TABLE_TOP_Z + TABLE_START_CLEARANCE_M,
            ],
            yaw_q,
        )
        for variant, chip in self.chips.items():
            chip.set_pose(self._chip_standby_pose(variant))
        self.active_chip = self.chips[self.episode.variant]
        self._pose_active_chip_scaled()
        shape_scale = np.asarray(self.episode.dimensions) * self.episode.scale / np.asarray(CHIP_VARIANTS['medium'])
        self.metadata["chip_curve_rise_m"] = float(self.episode.shape.rise_m)
        self.metadata["chip_collision_thickness_m"] = float(self.episode.dimensions[2] * self.episode.scale)
        local_vertices = self.active_chip.origin_surf_pts
        half_length = self.episode.dimensions[0] * self.episode.scale / 2
        support_center_x=curvature_center_x(self.episode.shape)*self.episode.scale
        self._chip_support_mask = (
            (np.abs(local_vertices[:, 0]-support_center_x) < half_length * 0.4)
            & (local_vertices[:, 2] <= shape_bottom_height(
                local_vertices[:, 0]/self.episode.scale,self.episode.shape)*self.episode.scale + 0.00005)
        )
        # Force gradients index all physical vertices, independently of render surfaces.
        physical_local = (self.active_chip.vertex_positions - self.active_chip.init_pose.p) @ self.active_chip.init_pose.R
        self._chip_load_mask = (
            (np.abs(physical_local[:, 0]-support_center_x) < half_length * 0.4)
            & (physical_local[:, 2] <= shape_bottom_height(
                physical_local[:, 0]/self.episode.scale,self.episode.shape)*self.episode.scale + 0.00005)
        )
        self.metadata['physical_chip_support_center_x_m']=support_center_x
        stage = omni.usd.get_context().get_stage()
        for chip in self.chips.values():
            set_actor_visible(stage, chip, True)
        self._kitchen_scene.reset(self.chip_start_pose.p[:2], TABLE_TOP_Z)
        # The old presentation collider is parked outside the workspace.
        # Only the real tabletop supports the single target.
        self.pedestal_pose = PEDESTAL_PARK_POSE
        set_actor_visible(stage, self.pedestal, False)

        for index, fragment in enumerate(self.fragments):
            set_actor_visible(stage, fragment, False)
            fragment.set_pose(
                Pose(
                    [STANDBY_X, FRAGMENT_STANDBY_Y[index], STANDBY_Z],
                    [1, 0, 0, 0],
                )
            )
        tray_xy_noise = self.rng.uniform(-TRAY_XY_NOISE_M, TRAY_XY_NOISE_M, size=2) * self.chip_randomization_scale
        tray_z_noise = float(self.rng.uniform(0.0, TRAY_HEIGHT_NOISE_M)) * self.chip_randomization_scale
        self.tray_target_z_offset = float(self.rng.uniform(
            TRAY_TARGET_Z_OFFSET_MIN_M, TRAY_TARGET_Z_OFFSET_MAX_M
        ))
        self.tray_pose = Pose(
            [
                TRAY_POSE.p[0] + tray_xy_noise[0],
                TRAY_POSE.p[1] + tray_xy_noise[1],
                TRAY_POSE.p[2] + tray_z_noise,
            ],
            [1, 0, 0, 0],
        )
        self.tray_target = np.array(
            [
                self.tray_pose.p[0],
                self.tray_pose.p[1],
                self.tray_pose.p[2] + self.tray_target_z_offset,
            ]
        )
        self.tray.set_pose(self.tray_pose)
        self.pedestal.set_pose(self.pedestal_pose)
        self._kitchen_scene.set_plate_support(self.tray_pose.p, TABLE_TOP_Z)
        self.metadata['reset_ground_clearance_m'] = staged_ground_clearances(
            self._actor_manager.actors, TABLE_TOP_Z)
        initialize_staged_actor_poses(self.uipc_sim, list(self._actor_manager.actors.values()))
        self.metadata['reset_pose_initialization'] = 'exact_staged_transforms_zero_velocity_before_settle'
        self.metadata['solver_settings'] = {
            'newton_velocity_tol_m_s': float(self.cfg.uipc_sim.newton.velocity_tol),
            'newton_max_iter': int(self.cfg.uipc_sim.newton.max_iter),
            'logger_level': self.cfg.uipc_sim.logger_level,
            'dt_s': float(self.cfg.sim.dt),
            'contact_distance_m': float(self.cfg.uipc_sim.contact.d_hat),
        }
        self.metadata['table_visual_top_z_m'] = TABLE_TOP_Z
        self.metadata['table_collision_top_z_m'] = self.cfg.uipc_sim.ground_height
        self.metadata['table_support_model'] = 'global_plane_aligned_with_visible_table'
        self.metadata['table_heights_match'] = bool(self.cfg.uipc_sim.ground_height == TABLE_TOP_Z)
        self.metadata['table_start_clearance_m'] = TABLE_START_CLEARANCE_M

        self.metadata.update(
            {
                "chip_variant": self.episode.variant,
                "chip_geometry_sampling": "shared_outline_warp_collision_visual_fragments",
                "chip_shape_family": self.episode.shape.family,
                "chip_curvature_origin": self.episode.shape.curvature_origin,
                "target_presentation": "single_free_chip_on_table",
                "chip_scale": float(self.episode.scale),
                "chip_dimensions_m": [
                    float(d * self.episode.scale)
                    for d in self.episode.dimensions
                ],
                "chip_yaw_rad": float(self.episode.yaw_rad),
                "chip_yaw_deg": float(np.rad2deg(self.episode.yaw_rad)),
                "chip_start_pose": self.chip_start_pose.tolist(),
                "pedestal_z_noise": float(pedestal_z_noise),
                "tray_xy_noise": tray_xy_noise.tolist(),
                "tray_z_noise": float(tray_z_noise),
                "tray_target_z_offset_m": float(self.tray_target_z_offset),
                "chip_initial_grasp_height_m": float(self.chip_start_pose.p[2]),
                "chip_visual_color_rgb": list(CHIP_VISUAL_COLOR),
                "tray_visual_color_rgb": list(TRAY_VISUAL_COLOR),
                "fracture_strain": float(self.episode.fracture_strain),
                "secure_strain": float(self.episode.secure_strain),
                "material_model": "shared_affine_reshape_strain_proxy_pending_calibration",
                "latent_strength_independent_of_geometry": False,
            }
        )

    def _setup_scene(self):
        super()._setup_scene()
        # High-friction gel-pad contact element: the chip is carried by UIPC
        # contact alone during transport, so the pads need a strong grip.
        # Applied to the gel pads only (not the chip meshes), so the chip can
        # still leave the pedestal / tray cleanly.
        contact_tabular = self.uipc_sim.scene.contact_tabular()
        default_contact = contact_tabular.default_element()
        grip_contact = contact_tabular.create("grasp_fragile_chip_grip")
        contact_tabular.insert(
            grip_contact,
            default_contact,
            friction_rate=CHIP_GRIP_FRICTION_RATIO,
            resistance=self.cfg.uipc_sim.contact.default_contact_resistance * GPa,
        )
        for tactile in self._tactile_manager.tactiles.values():
            for mesh in tactile.gelpad.uipc_meshes:
                grip_contact.apply_to(mesh)

    def _release_reset_constraints(self):
        self._reset_support_trace = []
        self.active_chip.remove_animate(force=True)
        self._chip_freed_early = True
        self.metadata['target_constraints_released_step'] = int(self.step_count)
        self.metadata['target_pinned_during_grasp'] = False

    def _read_tactile_depth(self):
        try:
            values = (
                self._tactile_manager.get_min_depth().detach().cpu().numpy()
            )
        except (AttributeError, RuntimeError):
            return None
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size < 2 or not np.all(np.isfinite(values)):
            return None
        return values

    def _chip_raw_strain(self):
        """Raw affine strain of the active chip (origin_surf_pts -> live vertices).

        Calibration/baseline only: for a single affine element the raw fit and
        the reference-relative fit give identical strain (trace(E^2) is
        invariant under orthogonal similarity), so this measures the same
        reshape anchored to the chip's init frame rather than the per-episode
        rest snapshot. None on a bad read (fail closed).
        """
        try:
            F, _t = estimate_affine_deformation(
                self.active_chip.origin_surf_pts, self.active_chip.vertices
            )
            return float(von_mises_strain(F))
        except ValueError:
            return None

    def _chip_strain(self):
        """Reference-relative von Mises strain of the chip's affine reshape.

        Fits the per-episode rest snapshot (captured while the chip is still
        pinned at the rigid start pose, before any pad contact) to the live
        vertices, so strain is exactly 0 at rest and immune to any residual
        deformation carried across episodes on the keep_constrained affine body.
        None on a bad read (fail closed).
        """
        if self._chip_strain_reference_verts is None:
            return None
        try:
            F, _t = estimate_affine_deformation(
                self._chip_strain_reference_verts, self.active_chip.vertices
            )
            return float(von_mises_strain(F))
        except ValueError:
            return None

    def _ensure_chip_strain_reference(self):
        """Capture the rest snapshot at the first monitored _step (chip pinned).

        Also records the raw baseline (origin_surf_pts -> rest_verts) for the
        calibration log. Idempotent per episode.
        """
        if self._chip_strain_reference_verts is None:
            self._chip_strain_reference_verts = self.active_chip.vertices.copy()
            self._chip_strain_baseline = self._chip_raw_strain()

    def _fracture_chip(self, depths, strain=None, reason="squeeze", impact_speed=None):
        if self.fractured:
            return
        self.fractured = True
        self.monitor_chip_contact = False
        self.fracture_step = int(self.step_count)
        break_pose = self.active_chip.get_pose()
        self.metadata['fracture_reason'] = reason
        # Score the real body before the fracture handoff parks its intact
        # geometry off-scene. Parking is presentation, not physical velocity.
        observer = getattr(self, '_action_monitor', None)
        if observer is not None and not observer.scorer.failure:
            if self.last_render != self.step_count:
                self._robot_manager.robot.update(dt=self.cfg.sim.dt * self.cfg.decimation)
            observer.advance()
        self.metadata['fracture_handoff'] = replace_squeezed_chip(
            self.uipc_sim, self.active_chip, self.fragments,
            self._chip_standby_pose(self.episode.variant),
            self._fracture_previous_geometry, float(self.cfg.sim.dt * self.cfg.decimation))
        stage = self._kitchen_scene.stage
        set_actor_visible(stage, self.active_chip, False)
        for fragment in self.fragments:
            set_actor_visible(stage, fragment, True)
        self._actor_manager.update(dt=0.0)
        self._fragment_release_pending = False
        self.metadata['fragment_release_step'] = int(self.step_count)
        self.metadata['fragment_settle_steps'] = 0
        self.metadata['fracture_event'] = {
            'step': int(self.step_count),
            'tactile_depth_mm': None if depths is None else np.asarray(depths).tolist(),
            'chip_strain': strain, 'fracture_strain': float(self.episode.fracture_strain),
            'chip_pose': break_pose.tolist(), 'reason': reason,
            'impact_speed_m_s': impact_speed,
            'model': 'threshold fracture with free fragments; no injected kick'}

    def _advance_chip_physics(self, is_save: bool = True):
        if not self.fractured and (self.monitor_chip_contact or self._chip_damage_armed):
            self._fracture_previous_geometry = capture_geometry_state(self.uipc_sim)
        previous_step = self.step_count
        super()._step(is_save=is_save)
        if self.step_count <= previous_step:
            return
        if self._fragment_release_pending:
            current_vertices = [fragment.vertices.copy() for fragment in self.fragments]
            self._fragment_settle_steps += 1
            if self._fragment_previous_vertices is not None:
                displacement = max(float(np.linalg.norm(current - previous, axis=1).max())
                                   for current, previous in zip(current_vertices, self._fragment_previous_vertices))
                self._fragment_stable_steps = self._fragment_stable_steps + 1 if displacement < 0.0005 else 0
            self._fragment_previous_vertices = current_vertices
            if self._fragment_stable_steps >= 2:
                for fragment in self.fragments:
                    fragment.remove_animate(force=True)
                self._actor_manager.update(dt=0.0)
                self._fragment_release_pending = False
                self.metadata["fragment_release_step"] = int(self.step_count)
                self.metadata["fragment_settle_steps"] = self._fragment_settle_steps

        if self.fractured:
            return
        if self._chip_damage_armed:
            vertices = self.active_chip.vertices
            center = vertices.mean(axis=0)
            dt = float(self.cfg.sim.dt * self.cfg.decimation)
            velocity_z = 0.0 if self._chip_previous_center is None else float(
                (center[2] - self._chip_previous_center[2]) / dt
            )
            tray_pose = self.tray.get_pose()
            above_tray = bool(np.all(np.abs(center[:2] - tray_pose.p[:2]) < [0.060, 0.050]))
            support_z = float(tray_pose.p[2] + 0.006) if above_tray else TABLE_TOP_Z
            gap=float(vertices[:,2].min()-support_z)
            support_N=self._plate_support_resultant()['support_force_N']
            upward_contact=bool(gap<=.002 and support_N>0.)
            incoming_speed=max(0.,-self._chip_previous_velocity_z)
            armed_before=self._chip_impact_tracker.flight_armed
            impact=self._chip_impact_tracker.advance(incoming_speed,upward_contact,gap)
            self.metadata['impact_detection_model']='flight_to_upward_surface_contact; supported settling is not a speed gate'
            if armed_before and upward_contact:
                self.metadata.setdefault('physical_landing_events',[]).append(dict(
                    step=self.step_count,incoming_speed_m_s=incoming_speed,support_force_N=support_N,
                    gap_m=gap,fractures=impact,critical_speed_m_s=CHIP_IMPACT_SPEED_M_S))
            if impact:
                self._fracture_chip(self._read_tactile_depth(), reason="impact", impact_speed=incoming_speed)
                return
            self._chip_previous_center = center.copy()
            self._chip_previous_velocity_z = velocity_z
        if not self.monitor_chip_contact and not self._chip_damage_armed:
            return
        self._ensure_chip_strain_reference()
        self.metadata.setdefault("chip_strain_baseline", self._chip_strain_baseline)
        depths = self._read_tactile_depth()
        if depths is None:
            return
        strain = self._chip_strain()
        self.tactile_depth_trace.append(depths.tolist())
        if strain is not None:
            self.tactile_strain_trace.append(float(strain))
        # Fracture on the chip's reshape strain (all deformation directions:
        # pad squeeze, any press, transport shear/crush). Fail-open on a bad
        # affine read (strain None -> the transport gate still fails the
        # episode via min_secure_samples).
        if strain is not None and strain >= self.episode.fracture_strain:
            self._fracture_chip(depths, strain=strain)


    @staticmethod
    def _pose_error(reference: Pose, current: Pose):
        delta = np.linalg.inv(reference.to_transformation_matrix()) @ (
            current.to_transformation_matrix()
        )
        translation = float(np.linalg.norm(delta[:3, 3]))
        cosine = np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        return translation, float(np.arccos(cosine))

    def _record_chip_tracking(self, tag):
        actual_pose = self.active_chip.get_pose()
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        relative_pose = actual_pose.rebase(to_coord=gripper_pose)
        translation_error, rotation_error = self._pose_error(
            self._chip_inhand_pose, relative_pose
        )
        if (
            translation_error > CHIP_TRANSPORT_ABORT_TRANSLATION_M
            or rotation_error > CHIP_TRANSPORT_ABORT_ROTATION_RAD
        ):
            self.grasp_lost = True
        depths = self._read_tactile_depth()
        sample = record_tactile_sample(
            self,
            tag="transport",
            depths_mm=depths,
            force_proxy=contact_force_on_actor(self, self.active_chip, norm="sum"),
            inhand_translation_m=translation_error,
            inhand_rotation_rad=rotation_error,
        )
        sample["phase"] = tag
        sample["actual_chip_pose"] = actual_pose.tolist()
        sample["gripper_center_pose"] = gripper_pose.tolist()
        sample["inhand_pose"] = relative_pose.tolist()
        sample["grasp_lost"] = bool(self.grasp_lost)
        sample["strain"] = self._chip_strain()
        return sample

    def _move_held_chip_linear(
        self,
        target_position,
        tag,
        *,
        max_speed,
        stop_on_support_contact=False,
        allow_grip_maintenance=False,
        max_acceleration=CHIP_TRANSPORT_ACCELERATION_M_S2,
    ):
        target_position = np.asarray(target_position, dtype=np.float64)
        if self._chip_inhand_pose is None:
            raise RuntimeError("Chip motion requires a verified tactile grasp")
        actor_target = Pose(target_position, self._grasped_chip_pose.q)
        center_target = Pose.from_matrix(
            actor_target.to_transformation_matrix()
            @ np.linalg.inv(self._chip_inhand_pose.to_transformation_matrix())
        )
        robot_manager = self._robot_manager
        ee_target = robot_manager.gripper_center_to_ee(center_target)
        ee_start = robot_manager.get_ee_pose()
        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        positions = smooth_translation(
            ee_start.p,
            ee_target.p,
            dt=dt,
            max_speed=max_speed,
            max_acceleration=max_acceleration,
        )
        target_quaternion = torch.as_tensor(
            ee_target.q, dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        self.atom_id += 1
        self.atom_tag = tag
        command_positions = np.concatenate(
            [positions[1:], np.repeat(positions[-1:], CHIP_TRANSPORT_SETTLE_STEPS, axis=0)]
        )
        for target_index, target_ee_position in enumerate(command_positions):
            if self.check_early_stop() or not self.plan_success:
                self.metadata.setdefault('abort_reason', self._action_monitor.failure or 'execution_failure')
                return False
            if self.chip_transport_grip_maintenance and allow_grip_maintenance and not stop_on_support_contact:
                depths=self._read_tactile_depth()
                decision=maintain_grasp_decision(depths,self._transport_grip_depth_reference)
                if decision['action']=='abort':
                    self.metadata['abort_reason']=decision['reason']
                    self.plan_success=False
                    return False
                if decision['action']=='close':
                    current_q=float(robot_manager.get_gripper_qpos())
                    target_q=max(float(robot_manager.gripper_percent2qpos(0.0)),
                                 current_q-decision['step_m'])
                    position=torch.full((len(robot_manager._gripper_ids),),target_q,
                                        device=robot_manager.device)
                    robot_manager.set_gripper(position,torch.zeros_like(position))
                    self.metadata.setdefault('transport_grip_adjustments',[]).append(dict(
                        step=int(self.step_count),motion=tag,depths_mm=np.asarray(depths).tolist(),
                        reference_mm=self._transport_grip_depth_reference.tolist(),
                        from_q_m=current_q,target_q_m=target_q))
            if stop_on_support_contact and self._placement_contact_ee_z is not None:
                target_ee_position = target_ee_position.copy()
                minimum_z = self._placement_contact_ee_z - CHIP_PLACEMENT_MAX_OVERTRAVEL_M + 0.00001
                if self._placement_marker_confirm_steps:
                    minimum_z = max(minimum_z, float(robot_manager.get_ee_pose().p[2]))
                target_ee_position[2] = max(float(target_ee_position[2]), minimum_z)
            ee_position, ee_quaternion = robot_manager.get_ee_pose_tensor()
            joint_position = robot_manager.robot.data.joint_pos[:, robot_manager._arm_ids]
            command = torch.cat(
                [
                    torch.as_tensor(
                        target_ee_position, dtype=torch.float32, device=self.device
                    ).reshape(1, 3),
                    target_quaternion,
                ],
                dim=-1,
            )
            robot_manager._ik_controller.set_command(command)
            joint_target = robot_manager._ik_controller.compute(
                ee_position,
                ee_quaternion,
                robot_manager.jacobian_b[:, :, robot_manager._arm_ids],
                joint_position,
            )
            joint_delta = joint_target - joint_position
            joint_limits = robot_manager.robot.data.soft_joint_pos_limits[:, robot_manager._arm_ids]
            if (
                not bool(torch.all(torch.isfinite(joint_target)))
                or float(torch.max(torch.abs(joint_delta))) > CHIP_TRANSPORT_MAX_JOINT_STEP_RAD
                or bool(torch.any(joint_target < joint_limits[..., 0]))
                or bool(torch.any(joint_target > joint_limits[..., 1]))
            ):
                self.metadata["abort_reason"] = "chip_transport_ik_discontinuity"
                self.plan_success = False
                return False
            robot_manager.set_arm(
                joint_target[0], torch.zeros_like(joint_target[0])
            )
            self._step(is_save=True)
            if self.cfg.uniform_policy_recording and self.last_render != self.step_count:
                robot_manager.robot.update(dt=dt)
            if self.fractured:
                self.metadata["abort_reason"] = "chip_fractured_during_transport"
                self.plan_success = False
                self.delay(CHIP_FRACTURE_AFTERMATH_STEPS, is_save=True)
                return False
            sample = self._record_chip_tracking(tag)
            desired_ee_pose = Pose(target_ee_position, ee_target.q)
            position_error, rotation_error = self._pose_error(
                desired_ee_pose, robot_manager.get_ee_pose()
            )
            sample["command_ee_pose"] = desired_ee_pose.tolist()
            sample["path_error_m"] = position_error
            sample["orientation_error_rad"] = rotation_error
            if (
                position_error > CHIP_TRANSPORT_MAX_PATH_ERROR_M
                or rotation_error > CHIP_TRANSPORT_MAX_ORIENTATION_ERROR_RAD
            ):
                self.metadata["abort_reason"] = "chip_transport_path_tracking_error"
                self.plan_success = False
                return False
            if self.grasp_lost:
                self.metadata["grasp_loss"] = self.tactile_traces["transport"][-1]
                self.metadata["abort_reason"] = "chip_grasp_lost"
                self.plan_success = False
                return False
            if stop_on_support_contact:
                if self._observe_plate_touch(allow_weak_signal=target_index >= len(positions) - 2):
                    break
                if not self.plan_success:
                    return False
        final_pose = self.active_chip.get_pose()
        self.tactile_traces["transport"][-1]["motion_target_pose"] = (
            actor_target.tolist()
        )
        self.tactile_traces["transport"][-1][
            "motion_target_position_error_m"
        ] = float(np.linalg.norm(final_pose.p - actor_target.p))
        self._update_render()
        if stop_on_support_contact and not self.metadata.get("placement_marker_contact_verified", False):
            self.metadata["placement_abort_reason"] = "plate_touch_marker_signal_missing"
            self.plan_success = False
            return False
        return True


    def _select_edge_grasp_control(self):
        """Select feedback from the raw edge-contact acquisition, never geometry IDs."""
        if not self.chip_edge_adaptive_control:
            return
        wrapped = self.metadata.get('edge_grasp_wrap_steps', 0) > 0
        self.chip_transport_grip_maintenance = wrapped
        self._chip_tactile_controller.directional_plate_contact = wrapped
        self.metadata['chip_edge_adaptive_control'] = dict(
            enabled=True, extra_wrap_applied=wrapped,
            transport_maintenance=wrapped, incremental_support=wrapped,
            decision_input='raw_tactile_edge_contact_acquisition')
        self.metadata['chip_transport_grip_maintenance_enabled'] = wrapped
        self.metadata['tactile_controller'] = self._chip_tactile_controller.description()

    def _staged_tactile_close(self, max_steps=400):
        self.metadata['edge_grasp_wrap_steps'] = 0
        device = self._robot_manager.device
        dimensions = len(self._robot_manager._gripper_ids)
        stop_reason = 'max_steps'
        for _ in range(max_steps):
            if self.check_early_stop() or not self.plan_success:
                stop_reason = 'physical_failure'
                break
            depths = self._read_tactile_depth()
            decision = self._chip_tactile_controller.close_decision(depths)
            if decision['action']=='hold' and self.cfg.chip_edge_contact_guard:
                obs=self._tactile_manager.get_observations(["depth"])
                try:
                    patches={name:contact_patch(obs[name]["depth"])
                             for name in ("left_tactile","right_tactile")}
                    needs_more=edge_contact_needs_closing(patches)
                    self.metadata.setdefault("edge_grasp_contact_checks",[]).append(
                        dict(step=self.step_count,patches=patches,needs_more=needs_more))
                    if needs_more:
                        decision=dict(action="close",
                            step_m=self._chip_tactile_controller.limits.contact_step_m/2,
                            reason="edge_contact_needs_wrap")
                except (ValueError,KeyError,TypeError):
                    decision=dict(action="abort",reason="invalid_edge_contact")

            if decision['action'] != 'close':
                stop_reason = decision['reason']
                break
            current = float(self._robot_manager.get_gripper_qpos())
            target = max(float(self._robot_manager.gripper_percent2qpos(0.0)), current - decision['step_m'])
            position = torch.full((dimensions,), target, device=device)
            velocity = torch.full_like(position, (target - current) / self.cfg.sim.dt)
            self._robot_manager.set_gripper(position, velocity)
            if decision.get('reason') == 'edge_contact_needs_wrap':
                self.metadata['edge_grasp_wrap_steps'] += 1
            self._step(is_save=True)
        position = torch.full((dimensions,), float(self._robot_manager.get_gripper_qpos()), device=device)
        self._robot_manager.set_gripper(position, torch.zeros_like(position))
        self.metadata['close_stop_reason'] = stop_reason
        self.metadata['close_stop_strain'] = self._chip_strain()  # diagnostic only
        edge_checks=self.metadata.get('edge_grasp_contact_checks',[])
        if self.cfg.chip_edge_contact_guard and (
                stop_reason=='invalid_edge_contact' or
                (edge_checks and edge_checks[-1]['needs_more'])):
            self.metadata['abort_reason']='edge_contact_not_acquired_before_close_stop'
            return None
        return self._read_tactile_depth()

    def _min_pad_contact_area_px(self, depth_threshold_mm: float):
        """Per-pad gel contact area: pixel count below the depth threshold on
        EACH pad, returned as the min (the weaker pad bounds the hold's
        visibility). Uses the exact obs-depth path the collector saves to h5
        (envs/_base_task.py get_observations), so the metric matches offline
        analysis. None on a missing frame (fail closed)."""
        obs = self._tactile_manager.get_observations(["depth"])
        areas = []
        for name in ("left_tactile", "right_tactile"):
            frame = obs.get(name, {}).get("depth")
            if frame is None:
                return None
            areas.append(float((frame < depth_threshold_mm).sum()))
        return min(areas)

    def _firm_transport_press(self, max_steps: int = None):
        """Second-stage close: tighten past the secure stop so the gel wraps
        more of the chip face during transport.

        The staged close stops on chip strain, which leaves the shallow-seated
        seeds (small scale -> thin chip edge) with a small contact patch even
        though the gripper already reached its calibrated closed pose (seed-0
        weaker pad ~860 px vs seed 3 ~6.4k px at the same joint). This press
        keeps closing in small steps until the WEAKER pad's contact area
        reaches firm_press_target_area, so the shallow seeds hold firmer
        during transport while a deep-seated seed already above the target
        presses ~0. Safety: aborts at FRACTURE_CLOSE_STRAIN_MARGIN below
        fracture_strain (clean, no crack), at the gel crush floor, or after
        firm_press_max_steps; the _step fracture gate is the final backstop.
        Leaves the gripper holding the tighter position for transport.
        Config-gated (TaskCfg.firm_transport_press); only runs after a
        strain_grip_secure close on a freed chip.
        """
        target = int(getattr(self.cfg, "firm_press_target_area", 0))
        if target <= 0 or self.fractured:
            return
        device = self._robot_manager.device
        cmd_dim = len(self._robot_manager._gripper_ids)
        thr = float(getattr(self.cfg, "firm_press_area_threshold_mm", 33.0))
        step = float(getattr(self.cfg, "firm_press_step", 0.00005))
        if max_steps is None:
            max_steps = int(getattr(self.cfg, "firm_press_max_steps", 300))
        fracture_floor = (
            self.episode.fracture_strain - FRACTURE_CLOSE_STRAIN_MARGIN
        )
        stop_reason = "max_steps"
        final_area = None
        for _ in range(max_steps):
            if self.fractured:
                return
            depths = self._read_tactile_depth()
            if depths is None:
                stop_reason = "no_depth"
                break
            mn = float(depths.min())
            if mn <= CHIP_DEPTH_CRUSH_FLOOR_MM:
                stop_reason = "crush_floor"
                break
            area = self._min_pad_contact_area_px(thr)
            if area is None:
                stop_reason = "no_frame"
                break
            final_area = area
            if area >= target:
                stop_reason = "area_satisfied"
                break
            strain = self._chip_strain()
            if strain is not None and strain >= fracture_floor:
                stop_reason = "strain_fracture_floor"
                break
            current_qpos = float(self._robot_manager.get_gripper_qpos())
            next_qpos = current_qpos - np.abs(step)
            position = torch.full((cmd_dim,), next_qpos, device=device)
            velocity = torch.full_like(
                position, -np.abs(step) / self.cfg.sim.dt
            )
            self._robot_manager.set_gripper(position, velocity)
            self._step(is_save=True)
        # Hold the tighter position with zero velocity (mirrors the close).
        current_qpos = float(self._robot_manager.get_gripper_qpos())
        hold_position = torch.full((cmd_dim,), current_qpos, device=device)
        self._robot_manager.set_gripper(
            hold_position, torch.zeros_like(hold_position)
        )
        self.metadata["firm_press_stop"] = stop_reason
        self.metadata["firm_press_area_px"] = final_area
        self.metadata["firm_press_post_depth_mm"] = (
            self._read_tactile_depth().tolist()
        )
        self.metadata["firm_press_post_strain"] = self._chip_strain()

    def _move_chip_approach_linear(self, target_pose: Pose, tag: str):
        """Follow a straight gripper-center path with gradual orientation alignment."""
        if not self.plan_success:
            return False
        robot_manager = self._robot_manager
        start_pose = robot_manager.get_gripper_center_pose()
        control_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        positions = smooth_translation(
            start_pose.p,
            target_pose.p,
            dt=control_dt,
            max_speed=CHIP_APPROACH_SPEED_M_S,
            max_acceleration=CHIP_APPROACH_ACCELERATION_M_S2,
        )
        progress = np.linspace(0.0, 1.0, len(positions))
        fractions = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
        target_quaternion = target_pose.q.copy()
        if np.dot(start_pose.q, target_quaternion) < 0.0:
            target_quaternion *= -1.0
        center_poses = []
        for position, fraction in zip(positions[1:], fractions[1:]):
            quaternion = start_pose.q + fraction * (target_quaternion - start_pose.q)
            quaternion /= np.linalg.norm(quaternion)
            center_poses.append(Pose(position, quaternion))
        center_poses.extend([target_pose] * CHIP_TRANSPORT_SETTLE_STEPS)
        diagnostics = {
            "start_pose": start_pose.tolist(),
            "target_pose": target_pose.tolist(),
            "max_position_error_m": 0.0,
            "max_orientation_error_rad": 0.0,
            "steps": 0,
            "actual_center_poses": [start_pose.tolist()],
        }
        self.metadata.setdefault("grasp_approach_tracking", {})[tag] = diagnostics
        self.atom_id += 1
        self.atom_tag = tag
        for center_pose in center_poses:
            if self.check_success():
                diagnostics["stopped_at_terminal_step"] = int(self.step_count)
                self._update_render()
                return True
            if self.check_early_stop() or not self.plan_success:
                return False
            ee_target = robot_manager.gripper_center_to_ee(center_pose)
            ee_position, ee_quaternion = robot_manager.get_ee_pose_tensor()
            joint_position = robot_manager.robot.data.joint_pos[:, robot_manager._arm_ids]
            robot_manager._ik_controller.set_command(
                torch.as_tensor(ee_target.tolist(), dtype=torch.float32, device=self.device).reshape(1, 7)
            )
            joint_target = robot_manager._ik_controller.compute(
                ee_position,
                ee_quaternion,
                robot_manager.jacobian_b[:, :, robot_manager._arm_ids],
                joint_position,
            )
            joint_limits = robot_manager.robot.data.soft_joint_pos_limits[:, robot_manager._arm_ids]
            if (
                not bool(torch.all(torch.isfinite(joint_target)))
                or float(torch.max(torch.abs(joint_target - joint_position))) > CHIP_TRANSPORT_MAX_JOINT_STEP_RAD
                or bool(torch.any(joint_target < joint_limits[..., 0]))
                or bool(torch.any(joint_target > joint_limits[..., 1]))
            ):
                self.metadata["abort_reason"] = "chip_approach_ik_discontinuity"
                self.plan_success = False
                return False
            robot_manager.set_arm(joint_target[0], torch.zeros_like(joint_target[0]))
            self._step(is_save=True)
            if self.last_render != self.step_count:
                robot_manager.robot.update(dt=control_dt)
            actual_pose = robot_manager.get_gripper_center_pose()
            diagnostics["actual_center_poses"].append(actual_pose.tolist())
            position_error, rotation_error = self._pose_error(
                center_pose, actual_pose
            )
            diagnostics["steps"] += 1
            diagnostics["max_position_error_m"] = max(
                diagnostics["max_position_error_m"], position_error
            )
            diagnostics["max_orientation_error_rad"] = max(
                diagnostics["max_orientation_error_rad"], rotation_error
            )
            if (
                position_error > CHIP_APPROACH_SERVO_TOLERANCE_M
                or rotation_error > CHIP_TRANSPORT_MAX_ORIENTATION_ERROR_RAD
            ):
                self.metadata["abort_reason"] = "chip_approach_path_tracking_error"
                self.plan_success = False
                return False
        diagnostics["final_error_m"], diagnostics["final_orientation_error_rad"] = self._pose_error(
            target_pose, robot_manager.get_gripper_center_pose()
        )
        self._update_render()
        return True

    def _grasp_chip(self):
        self.metadata['chip_edge_contact_guard_enabled']=bool(self.cfg.chip_edge_contact_guard)
        self.metadata['chip_transport_grip_maintenance_enabled']=self.chip_transport_grip_maintenance
        chip_pose = self.active_chip.get_pose()
        # Ground-truth the gel pad's vertical position relative to the gripper
        # center at grasp time (the gripper is vertically oriented here, so the
        # world-z difference IS the gel-vs-gripper offset). Kept as a diagnostic
        # cross-check of the grasp-height calibration (see below).
        gel_l = self._tactile_manager.tactiles["left_tactile"].get_attach_pose()
        gel_r = self._tactile_manager.tactiles["right_tactile"].get_attach_pose()
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        self.metadata["gelpad_vs_gripper_z_m"] = {
            "left": float(gel_l.p[2] - gripper_pose.p[2]),
            "right": float(gel_r.p[2] - gripper_pose.p[2]),
            "gripper_center_z": float(gripper_pose.p[2]),
            "chip_bottom_z": float(chip_pose.p[2]),
        }
        # Use one world-fixed top-down orientation for every chip presentation.
        # Keep random yaw visible as a change in projected jaw width/contact;
        # do not normalize it away by rotating the wrist with the chip.
        camera_up = CHIP_FIXED_CAMERA_UP.copy()
        jaw_axis = np.cross([0.0, 0.0, -1.0], camera_up)
        vertices = self.active_chip.vertices
        rim_anchor, grasp_width, rim_details = fixed_grasp_anchor(vertices, chip_pose.p, jaw_axis)
        rim_center_z=float(rim_anchor[2])
        self.metadata["fixed_grasp_anchor"]=rim_details
        self.metadata["grasp_fixed_yaw_rad"] = float(np.arctan2(camera_up[1], camera_up[0]))
        grasp_point = rim_anchor.copy()
        grasp_pose = construct_grasp_pose(
            grasp_point,
            [0.0, 0.0, 1.0],
            camera_up,
        )
        pad_offset_local = gripper_pose.R.T @ ((gel_l.p + gel_r.p) / 2 - gripper_pose.p)
        pad_offset_z = float((grasp_pose.R @ pad_offset_local)[2])
        grasp_point[2] -= pad_offset_z
        grasp_pose = Pose(grasp_point, grasp_pose.q)
        self.metadata['grasp_orientation_wxyz'] = grasp_pose.q.tolist()
        self.metadata['chip_projected_grasp_width_m'] = grasp_width
        self.metadata["curved_grasp_geometry"] = {
            "rim_center_z_m": rim_center_z,
            "pad_center_offset_z_m": pad_offset_z,
            "gripper_target_z_m": float(grasp_point[2]),
        }
        pre_pose = grasp_pose.add_bias([0.0, 0.0, -0.060])
        approach_orientation = grasp_pose.q.copy()
        lateral_pose = Pose(
            [grasp_pose.p[0], grasp_pose.p[1], gripper_pose.p[2]],
            approach_orientation,
        )
        pre_pose = Pose(pre_pose.p, approach_orientation)
        self.metadata["grasp_approach_waypoints"] = {
            "lateral": lateral_pose.tolist(),
            "pre_grasp": pre_pose.tolist(),
            "final": grasp_pose.tolist(),
            "orientation_mode": "fixed_world_yaw_vertical_top_down",
            "path_mode": "straight_gripper_center_differential_ik",
        }
        if not self.move(self.atom.open_gripper(0.98), tag="open_for_chip"):
            return False
        if not self._move_chip_approach_linear(
            lateral_pose, tag="approach_fragile_chip_lateral"
        ):
            return False
        if not self._move_chip_approach_linear(
            pre_pose,
            tag="approach_fragile_chip",
        ):
            return False
        if not self._move_chip_approach_linear(
            grasp_pose,
            tag="lower_to_fragile_chip",
        ):
            return False

        self.monitor_chip_contact = True
        depths = self._staged_tactile_close()
        if depths is None or self.fractured:
            return False
        self._select_edge_grasp_control()
        self.delay(6, is_save=True)
        if (
            getattr(self.cfg, "firm_transport_press", False)
            and self.metadata.get("close_stop_reason") == "strain_grip_secure"
        ):
            self._firm_transport_press()
        strain = self._chip_strain()
        if strain is None or self.fractured:
            return False
        self.grasp_state = classify_fragile_strain(
            strain,
            secure_strain=self.episode.secure_strain,
            fracture_strain=self.episode.fracture_strain,
        )
        verified_depths = self._read_tactile_depth()
        bilateral_contact = verified_depths is not None and bool(np.all(verified_depths < CHIP_FREE_DEPTH_MM))
        balanced_contact = verified_depths is not None and bool(
            np.all(verified_depths >= CHIP_DEPTH_CRUSH_FLOOR_MM)
            and np.min(verified_depths) <= CHIP_GRIP_TARGET_DEPTH_MM + 0.1
        )
        self.secure_grasp_seen = bilateral_contact and balanced_contact and not self.check_early_stop()
        if not self.secure_grasp_seen:
            self.metadata.setdefault('abort_reason', self.metadata.get('close_stop_reason', 'grasp_not_acquired'))
        self.metadata["bilateral_grasp_verified"] = bilateral_contact
        self.metadata["per_pad_grasp_safe"] = balanced_contact
        self.metadata["grasp_symmetry_required"] = False
        self.metadata["post_close_chip_strain"] = strain
        self.metadata["post_close_tactile_depth_mm"] = (
            self._read_tactile_depth().tolist()
        )
        self._transport_grip_depth_reference=np.asarray(
            self.metadata["post_close_tactile_depth_mm"],dtype=float)
        self.metadata["post_close_grasp_state"] = self.grasp_state
        depth_observations = self._tactile_manager.get_observations(["depth"])
        self.metadata["post_close_contact_patches"] = {
            name: contact_patch(depth_observations[name]["depth"])
            for name in ("left_tactile", "right_tactile")
        }
        self.metadata["post_close_force_proxy"] = contact_force_on_actor(
            self, self.active_chip, norm="sum"
        )
        if self.secure_grasp_seen:
            self._record_chip_dimensions('grasped')
            self.metadata['grasp_joint_pos'] = self._robot_manager.robot.data.joint_pos.detach().cpu().numpy().tolist()
            self._chip_damage_armed = True
            self.active_chip.remove_animate(force=True)
            self._actor_manager.update(dt=0.0)
            self.delay(4, is_save=True)
            measured_pose = self.active_chip.get_pose()
            self._chip_inhand_pose = measured_pose.rebase(
                to_coord=self._robot_manager.get_gripper_center_pose()
            )
            self._grasped_chip_pose = measured_pose
            self.metadata["chip_grasp_actual_pose"] = measured_pose.tolist()
            self.metadata["chip_inhand_reference_pose"] = (
                self._chip_inhand_pose.tolist()
            )
            self._record_chip_tracking("grasp_verified")
        return self.secure_grasp_seen

    def _chip_support_force(self):
        return contact_force_on_actor(
            self, self.active_chip, build_mask=lambda actor: self._chip_support_mask, norm="sum"
        )

    def _plate_clearance(self, vertices):
        tray_pose = self.tray.get_pose()
        local = (np.asarray(vertices) - tray_pose.p) @ tray_pose.R
        return float(local[:, 2].min() - 0.006)

    def get_frame_shot(self, obs):
        return BaseTask.get_frame_shot(self, obs)

    def _read_placement_markers(self):
        observations = self._tactile_manager.get_observations(["marker"])
        return {
            name: observations[name]["marker"].detach().cpu().numpy().copy()
            for name in ("left_tactile", "right_tactile")
        }

    def _save_plate_touch_observation(self):
        if int(self.step_count) in self.metadata.get("placement_observation_steps", []):
            return
        if self.cfg.uniform_policy_recording:
            self.metadata.setdefault("placement_observation_steps", []).append(int(self.step_count))
            return
        observations = self._get_observations()
        if self.mode == "collect" and self.cfg.save_frequency > 0 and self.step_count % self.cfg.save_frequency:
            self.save_observations(observations)
        if self.cfg.video_frequency > 0 and self.step_count % self.cfg.video_frequency:
            self.video_handler.write(self.get_frame_shot(observations))
        self.metadata.setdefault("placement_observation_steps", []).append(int(self.step_count))

    def _observe_plate_touch(self, *, allow_weak_signal=False):
        markers = self._read_placement_markers()
        detected, shifts = self._chip_tactile_controller.contact_decision(markers)
        self.metadata.setdefault('placement_marker_trace', []).append({
            'step': int(self.step_count), 'marker_shift': shifts, 'signal_detected': bool(detected),
            'raw_marker_onset': self._chip_tactile_controller.contact_onset_diagnostic})
        self._placement_marker_confirm_steps = self._chip_tactile_controller.confirm_ticks
        if not detected:
            return False
        self._placement_contact_ee_z = float(self._robot_manager.get_ee_pose().p[2])
        self.metadata['plate_contact_stop_step'] = int(self.step_count)
        self.metadata['placement_marker_contact_verified'] = True
        self.metadata['placement_marker_shift_px'] = shifts
        self.metadata['placement_marker_quality_met'] = all(
            shift['p90_px'] >= CHIP_PLACEMENT_MARKER_SHIFT_PX for shift in shifts.values())
        self._verify_placement_contact()  # scorer diagnostics, never a controller input
        return True

    def _hold_plate_touch(self):
        self.atom_id += 1
        self.atom_tag = 'confirm_plate_touch'
        self.metadata['placement_contact_hold_start_step'] = int(self.step_count)
        hold_steps = self._chip_tactile_controller.limits.hold_ticks
        self.metadata['placement_demonstration_hold_steps'] = hold_steps
        for _ in range(hold_steps):
            self._step(is_save=True)
            if self.check_early_stop() or not self.plan_success:
                return False
            safe, changes = self._chip_tactile_controller.hold_decision(
                self._read_placement_markers(), self._read_tactile_depth())
            self.metadata.setdefault('placement_contact_hold_trace', []).append({
                'step': int(self.step_count), 'marker_change': changes, 'tactile_stable': safe})
            self._verify_placement_contact()
            if not safe:
                self.metadata['placement_abort_reason'] = 'tactile_hold_unstable'
                self.plan_success = False
                return False
        self.metadata['placement_contact_hold_end_step'] = int(self.step_count)
        return True

    def _verify_placement_contact(self):
        depths = self._read_tactile_depth()
        strain = self._chip_strain()
        force_proxy = contact_force_on_actor(self, self.active_chip, norm="sum")
        support_force = self._chip_support_force()
        sample = {
            "depths_mm": None if depths is None else depths.tolist(),
            "strain": strain,
            "force_proxy": float(force_proxy) if force_proxy is not None else None,
            "phase": "placement_contact",
            "support_force_proxy": support_force,
        }
        self.placement_contact_trace.append(sample)
        self.metadata["placement_contact_last"] = sample
        if depths is None or not np.all(np.isfinite(depths)):
            return False
        in_band = (depths >= CHIP_PLACEMENT_CONTACT_DEPTH_MIN_MM) & (depths <= CHIP_PLACEMENT_CONTACT_DEPTH_MAX_MM)
        pad_count = int(np.count_nonzero(in_band))
        pressure_ok = pad_count >= CHIP_PLACEMENT_CONTACT_MIN_PADS
        support_clearance = self._plate_clearance(self.active_chip.vertices)
        pad_clearances = [
            self._plate_clearance(tactile.gelpad.data.nodal_pos_w.detach().cpu().numpy().reshape(-1, 3))
            for tactile in self._tactile_manager.tactiles.values()
        ]
        support_ok = support_force is not None and support_force > 0.0 and -0.0003 <= support_clearance <= 0.0008
        pressure_ok = pressure_ok and support_ok and min(pad_clearances) >= 0.001
        self.metadata["placement_support_clearance_m"] = support_clearance
        self.metadata["placement_pad_clearances_m"] = pad_clearances
        self.metadata["placement_support_contact_verified"] = bool(support_ok)
        strain_ok = strain is not None and strain < self.episode.fracture_strain * CHIP_PLACEMENT_MAX_STRAIN_MARGIN
        self.metadata["placement_contact_pressure_ok"] = bool(pressure_ok)
        self.metadata["placement_contact_strain_ok"] = bool(strain_ok)
        self.metadata["placement_contact_pad_count"] = pad_count
        self.metadata["placement_contact_depth_band_mm"] = [CHIP_PLACEMENT_CONTACT_DEPTH_MIN_MM, CHIP_PLACEMENT_CONTACT_DEPTH_MAX_MM]
        return bool(pressure_ok and strain_ok)

    def _plate_support_resultant(self):
        from ._force_task_utils import actor_contact_resultant
        measured = actor_contact_resultant(
            self, self.active_chip, self._chip_load_mask, self.active_chip.get_pose().p)
        force = np.asarray(measured['force_N'], dtype=float)
        normal = self.tray.get_pose().R[:, 2]
        upward = max(0.0, float(force @ normal))
        volume = float(np.asarray(self.active_chip.geo_slot_list[0].geometry()
            .instances().find('volume').view()).reshape(-1)[0])
        weight = volume * 1000.0 * abs(float(self.cfg.uipc_sim.gravity[2]))
        return {'support_force_N': upward, 'chip_weight_N': weight,
                'support_load_fraction': upward / weight,
                'support_resultant_N': force.tolist()}

    def _release_chip(self):
        """Use the same gripper-opening action as the other tasks."""
        manager = self._robot_manager
        self.metadata['release_motion'] = {
            'mode': 'shared_gripper_open',
            'target_open_fraction': 0.92,
            'support_controller': 'none',
            'task_specific_speed_limit': False,
            'tactile_only_acceptance': False,
        }
        try:
            moved = self.move(self.atom.open_gripper(0.92),
                tag="release_intact_chip", delay=False)
        finally:
            hold = torch.full((len(manager._gripper_ids),),
                float(manager.get_gripper_qpos()), device=manager.device)
            manager.set_gripper(hold, torch.zeros_like(hold))
        return bool(moved and not self.check_early_stop() and self.plan_success)


    def _play_once(self):
        if not self._grasp_chip():
            if self.fractured:
                self.delay(CHIP_FRACTURE_AFTERMATH_STEPS, is_save=True)
            self.plan_success = False
            return
        if not self._move_held_chip_linear(
            [
                self.active_chip.get_pose().p[0],
                self.active_chip.get_pose().p[1],
                CHIP_LIFT_Z,
            ],
            "lift_intact_chip",
            max_speed=CHIP_LIFT_SPEED_M_S,
            allow_grip_maintenance=True,
        ):
            self.plan_success = False
            return
        if not self._move_held_chip_linear(
            [self.tray_target[0], self.tray_target[1], CHIP_LIFT_Z],
            "carry_intact_chip_over_tray",
            max_speed=CHIP_CARRY_SPEED_M_S,
            allow_grip_maintenance=True,
        ):
            self.plan_success = False
            return
        if not self._move_held_chip_linear(
            [
                self.tray_target[0],
                self.tray_target[1],
                self.active_chip.get_pose().p[2]
                + self.tray.get_pose().p[2] + 0.006 + self.chip_placement_approach_clearance_m
                - float(self.active_chip.vertices[:, 2].min()),
            ],
            "lower_intact_chip_into_tray",
            max_speed=CHIP_LOWER_SPEED_M_S,
        ):
            self.plan_success = False
            return

        self.monitor_chip_contact = True
        self._placement_marker_reference = self._read_placement_markers()
        self.metadata["placement_marker_reference_step"] = int(self.step_count)
        self.metadata["placement_approach_clearance_m"] = self.chip_placement_approach_clearance_m
        self.metadata["placement_marker_quality_target_px"] = CHIP_PLACEMENT_MARKER_SHIFT_PX
        self.metadata["placement_marker_reference"] = {
            name: markers.tolist() for name, markers in self._placement_marker_reference.items()
        }
        self._chip_tactile_controller.begin_placement(self._placement_marker_reference)
        for _ in range(self._chip_tactile_controller.limits.baseline_ticks):
            self._step(is_save=True)
            self._chip_tactile_controller.baseline(self._read_placement_markers())
        self._placement_marker_detection_thresholds = self._chip_tactile_controller.thresholds
        self.metadata['placement_marker_detection_threshold_px'] = self._placement_marker_detection_thresholds
        self._save_plate_touch_observation()
        if not self._move_held_chip_linear(
            [self.tray_target[0], self.tray_target[1],
             self.active_chip.get_pose().p[2] + self.tray.get_pose().p[2] + 0.006
             - float(self.active_chip.vertices[:, 2].min()) - 0.0005],
            "seat_chip_on_plate",
            max_speed=CHIP_PLACEMENT_TOUCH_SPEED_M_S,
            max_acceleration=CHIP_PLACEMENT_TOUCH_ACCELERATION_M_S2,
            stop_on_support_contact=True,
        ):
            self.plan_success = False
            return
        if not self._hold_plate_touch():
            self.plan_success = False
            return
        self.monitor_chip_contact = False
        self.metadata["release_start_step"] = int(self.step_count)
        if not self._release_chip():
            self.plan_success = False
            return
        self.delay(5, is_save=True)
        self.active_chip.remove_animate(force=True)
        self._actor_manager.update(dt=0.0)
        self.delay(12, is_save=True)
        self.delay(24, is_save=True)
        current = self._robot_manager.get_gripper_center_pose()
        withdraw = Pose(current.p + [0.0, 0.0, 0.070], [0, 1, 0, 0])
        if not self._move_chip_approach_linear(withdraw, tag="withdraw_after_placement"):
            self.plan_success = False
            return
        self.delay(24, is_save=True)
        self.placed_in_tray = self.check_success()
        if not self.placed_in_tray:
            self.plan_success = False

    def _diagnostics(self):
        # Stamp the full traces here (always runs via check_success) so aborted
        # episodes — e.g. a mid-transport fracture — still persist them for
        # diagnosis; _play_once's success path alone drops them.
        self.metadata["tactile_depth_trace_mm"] = self.tactile_depth_trace
        self.metadata["chip_strain_trace"] = self.tactile_strain_trace
        self.metadata["transport_trace"] = self.tactile_traces.get(
            "transport", []
        )
        self.metadata["placement_contact_trace"] = self.placement_contact_trace
        chip_pose = self.active_chip.get_pose()
        tray_pose = self.tray.get_pose()
        xy_error = float(np.linalg.norm(chip_pose.p[:2] - self.tray_target[:2]))
        z_error = float(abs(chip_pose.p[2] - self.tray_target[2]))
        secure_samples = 0
        for s in self.tactile_strain_trace:
            if (
                s is not None
                and self.episode.secure_strain <= s < self.episode.fracture_strain
            ):
                secure_samples += 1

        transport_trace = self.tactile_traces.get("transport", [])
        transport_strain_trace = []
        transport_errors = []
        for sample in transport_trace:
            if (
                sample.get("strain") is not None
                and sample.get("inhand_translation_m") is not None
                and sample.get("inhand_rotation_rad") is not None
            ):
                transport_strain_trace.append(sample["strain"])
                transport_errors.append(
                    [
                        sample["inhand_translation_m"],
                        sample["inhand_rotation_rad"],
                    ]
                )
        transport_errors = np.asarray(
            transport_errors, dtype=np.float64
        ).reshape(-1, 2)
        transport_strain_ok = transport_strain_maintained(
            transport_strain_trace,
            transport_errors,
            secure_strain=self.episode.secure_strain,
            fracture_strain=self.episode.fracture_strain,
            max_translation_error_m=CHIP_TRANSPORT_MAX_TRANSLATION_M,
            max_rotation_error_rad=CHIP_TRANSPORT_MAX_ROTATION_RAD,
            min_secure_samples=CHIP_TRANSPORT_MIN_SECURE_SAMPLES,
        )

        translation_vals = [
            s["inhand_translation_m"]
            for s in transport_trace
            if s.get("inhand_translation_m") is not None
        ]
        rotation_vals = [
            s["inhand_rotation_rad"]
            for s in transport_trace
            if s.get("inhand_rotation_rad") is not None
        ]
        # Transport-contact evidence, two ANDed gates. (1) The contact-gradient
        # force proxy on the chip (sum of vertex gradient norms) reads on ANY
        # UIPC contact, so it is the transport-contact presence gate; gate on
        # presence (> 0) because its scale varies ~30x across runs (see
        # CHIP_TRANSPORT_MIN_FORCE_PROXY). (2) transport_strain_maintained
        # gates on the chip's von Mises RESHAPE strain staying in the secure
        # band (no fracture, still gripped) with bounded in-hand drift — the
        # chip-side gate reads all deformation directions, no far-plane.
        force_vals = [
            s["force_proxy"]
            for s in transport_trace
            if s.get("force_proxy") is not None
        ]
        transport_force_ok = transport_force_contact_maintained(
            force_vals,
            min_secure_samples=CHIP_TRANSPORT_MIN_SECURE_SAMPLES,
            min_force_proxy=CHIP_TRANSPORT_MIN_FORCE_PROXY,
        )
        return {
            "chip_variant": self.episode.variant,
            "fractured": bool(self.fractured),
            "fracture_step": self.fracture_step,
            "fracture_strain": float(self.episode.fracture_strain),
            "secure_strain": float(self.episode.secure_strain),
            "chip_strain_baseline": self.metadata.get("chip_strain_baseline"),
            "secure_grasp_seen": bool(self.secure_grasp_seen),
            "secure_tactile_samples": int(secure_samples),
            "tactile_sample_count": len(self.tactile_strain_trace),
            "transport_peak_strain": float(
                max(transport_strain_trace, default=0.0)
            ),
            "transport_p95_strain": float(
                np.percentile(transport_strain_trace, 95)
                if transport_strain_trace
                else 0.0
            ),
            "grasp_lost": bool(self.grasp_lost),
            "transport_tracking_count": int(len(transport_trace)),
            "max_transport_translation_error_m": float(
                max(translation_vals, default=0.0)
            ),
            "p95_transport_translation_error_m": float(
                np.percentile(translation_vals, 95) if translation_vals else 0.0
            ),
            "max_transport_rotation_error_rad": float(
                max(rotation_vals, default=0.0)
            ),
            "p95_transport_rotation_error_rad": float(
                np.percentile(rotation_vals, 95) if rotation_vals else 0.0
            ),
            "transport_force_proxy_min": float(min(force_vals)) if force_vals else None,
            "transport_force_proxy_max": float(max(force_vals)) if force_vals else None,
            "transport_force_proxy_mean": float(np.mean(force_vals)) if force_vals else None,
            "transport_force_proxy_contact_count": int(
                sum(1 for f in force_vals if f > 0.0)
            ),
            "transport_strain_maintained": bool(transport_strain_ok),
            "transport_force_contact_maintained": bool(transport_force_ok),
            "placed_in_tray": bool(self.placed_in_tray),
            "placement_contact_pressure_ok": bool(self.metadata.get("placement_contact_pressure_ok", False)),
            "placement_contact_strain_ok": bool(self.metadata.get("placement_contact_strain_ok", False)),
            "placement_marker_contact_verified": bool(self.metadata.get("placement_marker_contact_verified", False)),
            "chip_pose": chip_pose.tolist(),
            "tray_pose": tray_pose.tolist(),
            "tray_xy_error_m": xy_error,
            "tray_z_error_m": z_error,
            "gripper_open_percentage": float(
                self._robot_manager.get_gripper_percentage()
            ),
        }

    def take_action(self, action, *args, **kwargs):
        if getattr(self, "_action_monitor", None) is None:
            self._action_monitor = ChipPolicyMonitor(self)
        action_type = kwargs.get("action_type", args[0] if args else "qpos")
        self._action_monitor.register_command(action, action_type)
        return super().take_action(action, *args, **kwargs)

    def check_early_stop(self):
        observer = getattr(self, "_action_monitor", None)
        return bool(self.fractured or (observer is not None and observer.failure))

    def check_success(self):
        observer = getattr(self, '_action_monitor', None)
        return bool(observer is not None and observer.scorer.success and not self.fractured)

    def _update_render(self):
        if hasattr(self, '_shape_visuals') and hasattr(self, 'active_chip'):
            self._shape_visuals.sync(self.active_chip, self.fragments, self.fractured, FRAGMENT_SEAM_SCALE)
        super()._update_render()

    def _step(self, is_save=True):
        observer = getattr(self, '_action_monitor', None)
        if observer is not None and observer.scorer.success:
            return
        previous_step = self.step_count
        self._advance_chip_physics(is_save=is_save)
        if (self.step_count > previous_step and observer is None
                and getattr(self, '_reset_support_trace', None) is not None):
            vertices = self.active_chip.vertices
            self._reset_support_trace.append({
                'step': int(self.step_count),
                'center_m': vertices.mean(axis=0).tolist(),
                'min_z_m': float(vertices[:, 2].min()),
                'pose': self.active_chip.get_pose().tolist(),
            })
        if self.step_count > previous_step and observer is not None:
            if self.last_render != self.step_count:
                self._robot_manager.robot.update(dt=self.cfg.sim.dt * self.cfg.decimation)
            observer.advance()
            if observer.scorer.failure and not self.metadata.get('chip_physical_terminal_captured'):
                from ._force_task_utils import record_terminal_observation
                self.metadata['chip_physical_terminal_captured']=True
                self.metadata['chip_physical_verdict_step']=int(self.step_count)
                self._set_phase(self.PHASE_TERMINAL,terminal_reason=observer.scorer.failure)
                if self.cfg.save_frequency>0 and self.mode!='eval_test':
                    record_terminal_observation(self,observer.scorer.failure)

    def save_privileged_sidecar(self, reason=""):
        if reason in ("timeout", "error"):
            self._execution_reason = reason
            observer = getattr(self, '_action_monitor', None)
            if observer is not None:
                observer.scorer.finish(reason)
        return None  # Physical failure is recorded by the collector; no extra sidecar.

    def save_to_hdf5(self):
        from ._force_task_utils import record_terminal_observation
        observer = getattr(self, '_action_monitor', None)
        if observer is not None:
            observer.scorer.finish(getattr(self, '_execution_reason', '') or
                                   self.metadata.get('abort_reason') or self.metadata.get('placement_abort_reason') or 'incomplete')
            diagnostics = observer.scorer.snapshot()
        else:
            diagnostics = {'success': False, 'outcome': 'invalid', 'terminal_reason': 'error'}
        self.metadata['success_diagnostics'] = diagnostics
        self.metadata['physical_result'] = diagnostics['outcome']
        self.metadata['terminal_reason'] = diagnostics['terminal_reason']
        if not self.metadata.get('chip_physical_terminal_captured'):
            self.metadata['chip_physical_verdict_step']=int(self.step_count)
            record_terminal_observation(self, diagnostics['terminal_reason'])
        self._save_metadata()
        super().save_to_hdf5()

    def _get_observations(self):
        observation = super()._get_observations()
        if self.phase_id != self.PHASE_PRE_MOVE and self.policy_start_step is not None:
            observation['phase']['policy_step'] = int(self.step_count - self.policy_start_step)
        return observation

