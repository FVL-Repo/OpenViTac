from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

import openvitac_sim as sim
from openvitac_cli.run_log import run_logger
from openvitac_cli.tee import tee_output
from openvitac_cli.video import render_video
from openvitac_tactile.config import TactileCfg
from openvitac_tactile.observation import TactileObservation
from openvitac_tasks import TASKS, EpisodeRandomization, Task, TaskName

if TYPE_CHECKING:
    from isaaclab.sim import SimulationCfg

    from openvitac_data import ObjectFrame, ObjectMeta, PadTactileMeta, TactileFrame
    from openvitac_sim import DepthCameraCfg
    from openvitac_sim.tactile_bridge import PadSurface
    from openvitac_tasks import (
        Action,
        ExpertStateMachine,
        SuccessFn,
    )
    from openvitac_tasks.expert import ArmHandle, ExpertWorld


def _scene_implements_expert_world(scene: sim.Scene) -> ExpertWorld:
    """Static conformance assertion: ``sim.Scene`` structurally satisfies the
    tasks-side ``ExpertWorld`` protocol, and ``scene.arm`` satisfies ``ArmHandle``
    (arm motion/sensing plus the capability-gated gripper handle). sim must not
    import tasks, so the check lives here; it is never called at runtime."""
    world: ExpertWorld = scene
    arm: ArmHandle = world.arm
    _ = arm.gripper
    return world


if TYPE_CHECKING:
    _ = _scene_implements_expert_world

type _TaskLogger = logging.LoggerAdapter[logging.Logger]

_DT = 1 / 120
_SIM_DT_US = int(_DT * 1e6)
_DEFAULT_STEPS = 10
# Physics/control run at 120 Hz; recording defaults to every 2nd physics step
# (60 Hz). --record-every 1 records all 120 Hz steps, 4 records 30 Hz.
_DEFAULT_RECORD_EVERY = 2
# Pre-rollout physics settling: real PhysX+UIPC steps after reset and DR, before
# the episode clock starts. 0 skips settling.
_DEFAULT_SETTLE_STEPS = 20
# Opt-in convergence criterion: pose-delta over this many real settle steps.
# Off by default; fixed 20 steps remains the episode default.
_SETTLE_WINDOW = 5
_SETTLE_POS_EPS_M = 1e-4
_SETTLE_ROT_EPS_RAD = 1e-3
# Safety cap for a scripted expert (insert_usb's current path is ~1600 steps, so
# 2000 leaves headroom); hitting it records outcome="timeout" instead of
# truncating silently.
_DEFAULT_EXPERT_STEPS = 2000


def _run_directory(out: Path, task: TaskName) -> Path:
    """Create one directory per run: ``<out>/<task>_<YYYYMMDD_HHMMSS>[_N]/``.

    Creation is exclusive, so two runs starting in the same second never share
    (and overwrite) an episode file; the suffix only appears on collision.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in itertools.count():
        name = f"{task}_{stamp}" if suffix == 0 else f"{task}_{stamp}_{suffix}"
        candidate = out / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise AssertionError("unreachable: itertools.count is infinite")


def _task_has_tactile(task: TaskName) -> bool:
    """Whether the task's scene spec declares tactile sensor bindings."""
    return bool(TASKS[task].scene_spec().sensors)


def _effective_settle_steps(task: Task, override: int | None) -> int:
    """CLI override when given, otherwise the task's pre-rollout default."""
    steps = task.settle_steps if override is None else override
    if steps < 0:
        raise ValueError(f"--settle-steps must be non-negative, got {steps}")
    return steps


def _apply_action(scene: sim.Scene, action: Action) -> None:
    """Stage the action's explicit EE and gripper targets; do not step."""
    scene.arm.set_ee_target(action.ee_pos, action.ee_quat)
    scene.arm.gripper.set_target(action.gripper_qpos)


def _tick_success(
    success: SuccessFn, world: ExpertWorld
) -> Callable[[], bool]:
    """Return a lazy per-episode success reader.

    The iterator is created on the first read, after the first physics step,
    and advanced on every later read, so the success generator keeps its local
    verdict state across ticks.  Creation and every read stay inside the
    rollout's try block, so failures are logged and the sink is cleaned up.
    """
    source: Iterator[bool] | None = None

    def read() -> bool:
        nonlocal source
        if source is None:
            result = success(world)
            if not isinstance(result, Iterator):
                raise TypeError(
                    "per-tick success must return an Iterator[bool]"
                )
            source = result
        return bool(next(source, False))

    return read


def _run_state_rollout(
    scene: sim.Scene,
    machine: ExpertStateMachine[Any],
    success: SuccessFn,
    *,
    steps: int,
    sink: Callable[[sim.SceneObservation], None],
    record_every: int,
    logger: _TaskLogger | None = None,
) -> Literal["success", "fail", "timeout", "policy_error"]:
    """Drive an explicit per-tick state machine: apply action, then step once.

    ``success`` runs after every physics step; the first True latches the
    outcome, records the 1-based success step, and stops the rollout.  ``steps``
    is the safety cap and maps to ``timeout``.  Exceptions from ``init``/``step``
    are policy errors: the traceback is logged and the episode is finalized
    with ``outcome=policy_error``; success-reader failures keep propagating.
    """
    if logger is not None:
        logger.info(
            "expert start steps=%d mode=state-machine record_every=%d",
            steps,
            record_every,
        )
    read_success = _tick_success(success, scene)
    scene.set_step_sink(sink, every=record_every)
    scene.set_step_limit(steps)
    timed_out = False
    policy_error = False
    success_step: int | None = None
    tick = 0
    try:
        try:
            state = machine.init()
        except Exception:
            if logger is not None:
                logger.exception("policy error in init")
            policy_error = True
        else:
            while True:
                try:
                    result = machine.step(scene, state)
                except Exception:
                    if logger is not None:
                        logger.exception("policy error in step")
                    policy_error = True
                    break
                if result is None:
                    break
                state, action = result
                _apply_action(scene, action)
                scene.step()
                tick += 1
                try:
                    done = read_success()
                except Exception:
                    if logger is not None:
                        logger.exception("success evaluator failed; re-raising")
                    raise
                if done:
                    success_step = tick
                    break
    except sim.StepLimitExceeded:
        timed_out = True
        if logger is not None:
            logger.warning("expert hit step limit steps=%d", steps)
    except Exception:
        if logger is not None:
            logger.exception("rollout failed; re-raising")
        raise
    finally:
        scene.set_step_sink(None)
    if policy_error:
        if logger is not None:
            logger.warning("outcome=policy_error reason=expert-exception")
        return "policy_error"
    if success_step is not None:
        if logger is not None:
            logger.info(
                "outcome=success reason=per-step-success step=%d", success_step
            )
        return "success"
    if timed_out:
        if logger is not None:
            logger.warning("outcome=timeout reason=step-limit")
        return "timeout"
    if logger is not None:
        logger.warning("outcome=fail reason=per-step-success-never-true")
    return "fail"


def _tactile_config(task: Task) -> TactileCfg:
    """The same tactile plan ``sim.Scene`` rebuilds from the task spec.

    The cli keeps this copy for meta and object bookkeeping only; the scene
    owns the worker's actual config.  Patching sync knobs here would therefore
    affect nothing but this copy, so they must be passed to the scene instead.
    """
    from openvitac_sim.tactile_plan import tactile_plan_from_scene

    return tactile_plan_from_scene(task.scene_spec())


def _world_to_pad_local(
    world: npt.NDArray[np.floating[Any]],
    pad_frame: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Map world-space worker vertices into the pad frame (world <- pad)."""
    inverse = np.linalg.inv(np.asarray(pad_frame, dtype=np.float64))
    points = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    return points @ inverse[:3, :3].T + inverse[:3, 3]


def _tactile_meta(
    cfg: TactileCfg,
    *,
    pad_world_frames: Mapping[str, Any],
    pad_surfaces: Mapping[str, PadSurface],
    depth_specs: Mapping[str, DepthCameraCfg] | None = None,
) -> tuple[tuple[PadTactileMeta, ...], tuple[ObjectMeta, ...]]:
    """Build the v3 /meta/tactile and /meta/objects dataclasses.

    ``rest_vertices`` come from the worker's published surface (same order as
    per-frame ``vertices``), expressed in the pad frame.  The frame stream stays
    world-space, so ``pad_frame @ rest_vertices`` recovers the worker rest pose.
    """
    from openvitac_data import DepthCameraMeta, ObjectMeta, PadTactileMeta
    from openvitac_tactile.contact_metrics import select_contact_index

    uipc_objects = [obj for obj in cfg.objects if obj.authority == "uipc"]
    contact_index = select_contact_index(
        [(obj.name, obj.fixed, obj.contact_object) for obj in uipc_objects]
    )
    contact_object = (
        uipc_objects[contact_index].name if contact_index is not None else None
    )

    pads: list[PadTactileMeta] = []
    for pad in cfg.pads:
        pad_frame = pad_world_frames.get(pad.name, pad_world_frames.get(pad.side))
        if pad_frame is None:
            raise ValueError(f"missing calibrated world frame for pad {pad.name!r}")
        surface = pad_surfaces.get(pad.name)
        if surface is None:
            raise ValueError(f"missing worker surface for pad {pad.name!r}")
        pad_frame_array = np.asarray(pad_frame, dtype=np.float64)
        rest_vertices = _world_to_pad_local(
            surface.rest_vertices, pad_frame_array
        ).astype(np.float32)
        max_vertices = min(
            cfg.sync.max_pad_vertices,
            pad.geometry.max_surface_vertices,
        )
        if rest_vertices.shape[0] > max_vertices:
            raise ValueError(
                f"worker surface for pad {pad.name!r} has "
                f"{rest_vertices.shape[0]} vertices, above {max_vertices}"
            )
        depth_spec = depth_specs.get(pad.side) if depth_specs is not None else None
        pads.append(
            PadTactileMeta(
                side=pad.side,
                rest_vertices=rest_vertices,
                surface_triangles=np.asarray(surface.triangles, dtype=np.int32),
                pad_frame=pad_frame_array,
                youngs_mpa=pad.material.youngs_mpa,
                strength=pad.material.constraint_strength,
                prim_path=pad.gelpad_prim_path,
                attachment_body=pad.frame.body_name,
                max_vertices=max_vertices,
                width=pad.image_width,
                height=pad.image_height,
                marker_rows=pad.marker_rows,
                marker_cols=pad.marker_cols,
                contact_object=contact_object,
                image_streams=tuple(pad.render_kinds),
                depth_camera=(
                    DepthCameraMeta(
                        width=depth_spec.width,
                        height=depth_spec.height,
                        focal_length=depth_spec.focal_length,
                        horizontal_aperture=depth_spec.horizontal_aperture,
                        vertical_aperture=depth_spec.vertical_aperture,
                        clipping_range=depth_spec.clipping_range,
                        semantics=depth_spec.data_type,
                    )
                    if depth_spec is not None
                    else None
                ),
            )
        )
    objects: tuple[ObjectMeta, ...] = tuple(
        ObjectMeta(
            name=obj.name,
            prim_path=obj.prim_path,
            authority=obj.authority,
            constitution=obj.constitution,
            usd=obj.usd_path,
        )
        for obj in cfg.objects
    )
    return tuple(pads), objects


def _pose_is_valid(pose: Any) -> bool:
    array = np.asarray(pose, dtype=np.float64)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        return False
    quat_norm = float(np.linalg.norm(array[3:7]))
    return quat_norm > 1e-12


def _tactile_payload(
    observation: TactileObservation,
    *,
    timestamp_us: int,
    object_prim_paths: Mapping[str, str] | None = None,
    object_authorities: Mapping[str, str] | None = None,
    pose_reader: Callable[
        [str], tuple[tuple[float, float, float], tuple[float, float, float, float]]
    ]
    | None = None,
) -> tuple[dict[str, TactileFrame], dict[str, ObjectFrame]]:
    """Map the sim-side tactile projection to v3 Frame payloads.

    The pad set follows the observation, so a single-pad scene writes one pad
    instead of failing on a hard-coded left/right pair.

    PhysX-authority objects are not simulated by the UIPC worker, so their
    tactile object slot is invalid; read those poses from the live scene.  A
    UIPC object has only a kinematic PhysX proxy, whose pose is not ground
    truth, so it is never substituted for an invalid worker pose.
    """
    from openvitac_data import ObjectFrame, TactileFrame

    pads = observation.pads
    if not pads:
        raise RuntimeError("tactile observation contains no pads")
    tactile: dict[str, TactileFrame] = {}
    for side, pad in pads.items():
        tactile[side] = TactileFrame(
            side=side,
            vertices=pad.vertices,
            n_vertices=pad.n_vertices,
            contact_gradient=pad.contact_gradient,
            force=pad.force,
            force_kind=pad.force_kind,
            contact_count=pad.contact_count,
            min_sd_proxy_mm=pad.min_sd_proxy_mm,
            min_sd_full_mm=pad.min_sd_full_mm,
            tracking_error_mm=pad.tracking_error_mm,
            seq=pad.seq,
            age_ms=pad.age_ms,
            stale=pad.stale,
            valid=pad.valid,
            finger_seq=observation.finger_seq,
            physx_step=observation.physx_step,
            finger_age_ms=observation.finger_age_ms,
            finger_stale=observation.finger_stale,
            timestamp_us=timestamp_us,
            pad_usb_min_sd_mm=pad.pad_usb_min_sd_mm,
            pad_usb_contact_vertices=pad.pad_usb_contact_vertices,
            depth=observation.tactile_depth.get(side),
            rgb=observation.tactile_rgb.get(side),
            marker_rgb=observation.tactile_marker_rgb.get(side),
        )
    objects: dict[str, ObjectFrame] = {}
    for name, obj in observation.objects.items():
        pose = np.asarray(obj.pose_xyzw, dtype=np.float32)
        valid = bool(obj.valid and _pose_is_valid(pose))
        authority = (
            object_authorities.get(name) if object_authorities is not None else None
        )
        if (
            not valid
            and authority == "physx"
            and pose_reader is not None
            and object_prim_paths is not None
        ):
            prim_path = object_prim_paths.get(name)
            if prim_path is not None:
                pos, quat = pose_reader(prim_path)
                pose = np.asarray([*pos, *quat], dtype=np.float32)
                # Re-check the scene read: a failed PhysX query must stay
                # invalid instead of being recorded as ground truth.
                valid = _pose_is_valid(pose)
        objects[name] = ObjectFrame(
            pose_xyzw=pose,
            seq=observation.seq,
            valid=valid,
        )
    return tactile, objects


def _validate_run_parameters(
    *,
    steps: int | None,
    seed: int,
    no_video: bool,
    expert: bool,
    record_every: int,
) -> int:
    """Validate CLI run parameters before the app launches; return the step budget.

    Rejecting bad input here avoids a Kit startup and ensures no Scene or
    tactile worker is ever created for a run that cannot produce its output.
    """
    if record_every <= 0:
        raise ValueError(f"--record-every must be positive, got {record_every}")
    if seed < 0:
        raise ValueError(f"--seed must be non-negative, got {seed}")
    limit = (
        steps
        if steps is not None
        else (_DEFAULT_EXPERT_STEPS if expert else _DEFAULT_STEPS)
    )
    if steps is not None and steps < 1:
        raise ValueError(f"--steps must be at least 1, got {steps}")
    if not no_video:
        # render_video needs at least two frame timestamps to infer fps.
        recorded_frames = (limit + record_every - 1) // record_every
        if recorded_frames < 2:
            raise ValueError(
                f"video needs at least two recorded frames, but --steps {limit} "
                f"at --record-every {record_every} records {recorded_frames}; "
                "increase --steps (or use --record-every 1) or pass --no-video"
            )
    return limit


def _seed_run(seed: int) -> int:
    """Seed every RNG the rollout touches once the sim app is up.

    ``isaaclab`` (including ``isaaclab.utils.seed``) must not be imported
    before ``sim.launch``: importing its modules/configs early poisons the Kit
    launch.  The app is already running here, and this still precedes task
    setup and any rollout sampling.
    """
    from isaaclab.utils.seed import configure_seed

    return configure_seed(seed, torch_deterministic=True)


def _open_scene(
    task: TaskName,
    sim_cfg: SimulationCfg,
    *,
    tactile: bool,
    optical_backend: Literal["taxim", "procedural"],
    taxim_device: str,
    seed: int = 0,
    logger: _TaskLogger | None = None,
) -> tuple[sim.Scene, TactileCfg | None]:
    """Build the scene and its cli-side tactile plan copy.

    If post-construction capability validation fails, close the scene before
    re-raising so its tactile worker (spawned in ``Scene.__init__``) cannot
    outlive the failed run.
    """
    spec = TASKS[task]
    tactile_cfg = _tactile_config(spec) if tactile else None
    scene = sim.Scene(
        spec.scene_spec(),
        sim_cfg,
        tactile_enabled=tactile,
        optical_backend=optical_backend,
        taxim_device=taxim_device,
        seed=seed,
    )
    try:
        if logger is not None:
            scene.set_logger(logger)
        missing = spec.required_capabilities - scene.arm.gripper.capabilities
        if missing:
            raise RuntimeError(
                f"task {task!r} requires gripper capabilities {sorted(missing)} "
                f"not provided by {spec.embodiment.name!r}"
            )
    except BaseException:
        scene.close()
        raise
    return scene, tactile_cfg


def collect(
    task: TaskName,
    out: Path = Path("output"),
    steps: int | None = None,
    seed: int = 0,
    no_video: bool = False,
    video_fps: int | None = None,
    expert: bool = False,
    record_every: int = _DEFAULT_RECORD_EVERY,
    settle_steps: int | None = None,
    settle_converge: bool = False,
    optical_backend: Literal["taxim", "procedural"] = "taxim",
    taxim_device: str = "cuda:0",
) -> None:
    """Collect one episode: spec -> sim.Scene -> data writer.

    Writes ``episode.h5``, ``collect.log``, and (unless ``--no-video``)
    ``video.mp4`` under a fresh timestamped directory. Without ``--expert``
    this runs ``steps`` idle steps with ``outcome="unknown"``; with
    ``--expert`` the task's expert drives and ``steps`` is only a safety cap
    (default 2000). Video is post-processed from the finished HDF5 while the
    app is still up, so it must run before the launch context exits; its frame
    rate is inferred from the recorded frame timestamps and ``--video-fps``
    overrides it.  A video run that cannot record two frames (for example
    ``--steps 1``) is rejected before the app launches unless ``--no-video``
    is set; negative seeds and non-positive step counts are rejected likewise.
    The scene is closed after video rendering, and on any failure, so the
    tactile worker and its shared memory never outlive the run.

    Tactile is derived from the task's sensor bindings, not a CLI switch: a
    tactile task always enables the bridge.

    ``--settle-steps`` runs that many real physics steps (PhysX only before the
    tactile bridge starts) after reset/DR and before the rollout clock starts;
    it overrides ``Task.settle_steps`` (default 20), and 0 skips settling.
    ``--settle-converge`` opts into an early stop once every
    rigid actor's pose moved less than the configured epsilons over a 5-step
    window; it is off by default, so the plain fixed-step behavior is unchanged.
    """
    if settle_steps is not None and settle_steps < 0:
        raise ValueError(f"--settle-steps must be non-negative, got {settle_steps}")
    limit = _validate_run_parameters(
        steps=steps,
        seed=seed,
        no_video=no_video,
        expert=expert,
        record_every=record_every,
    )
    tactile = _task_has_tactile(task)
    settle_window = _SETTLE_WINDOW if settle_converge else None

    run_dir = _run_directory(out, task)
    episode = run_dir / "episode.h5"
    # Order matters: tee wraps the app's output, and post-processing runs while
    # the app is still up (Kit's fast shutdown ends the process on context exit).
    # The scene must close before that exit, but only after video rendering:
    # its tactile worker owns the shared-memory channel.
    with (
        tee_output(run_dir / "collect.log"),
        sim.launch(dt=_DT) as sim_cfg,
        run_logger(task=task, seed=seed, run_id=run_dir.name) as log,
    ):
        seed = _seed_run(seed)
        scene, tactile_cfg = _open_scene(
            task,
            sim_cfg,
            tactile=tactile,
            optical_backend=optical_backend,
            taxim_device=taxim_device,
            seed=seed,
            logger=log.task,
        )
        try:
            frames, cameras = _record(
                task,
                scene,
                tactile_cfg,
                episode,
                limit,
                seed,
                expert,
                record_every=record_every,
                settle_steps=settle_steps,
                settle_window=settle_window,
                logger=log.task,
            )
            if not no_video:
                render_video(episode, run_dir / "video.mp4", video_fps)
            log.run.info(
                "wrote %s (%d frames, %d cameras, %d bytes%s)",
                run_dir.resolve(),
                frames,
                cameras,
                episode.stat().st_size,
                ", video.mp4" if not no_video else ", no video",
            )
        finally:
            scene.close()


def _prepare_episode(
    scene: sim.Scene,
    randomization: EpisodeRandomization,
    settle_steps: int,
    settle_window: int | None,
) -> dict[str, float | str]:
    """Reset, settle, then snapshot actual poses before rollout.

    Lifecycle contract: ``reset`` applies the DR poses, ``settle`` advances
    pre-rollout physics, and ``start_tactile`` captures the tactile t0
    reference from the settled state.  Only then are ``actual.*`` poses read
    and cameras warmed, so episode timestamp 0 is the first post-settle frame.
    """
    scene.reset(actor_poses=randomization.actor_poses)
    result = scene.settle(
        settle_steps,
        window=settle_window,
        pos_eps=_SETTLE_POS_EPS_M,
        rot_eps=_SETTLE_ROT_EPS_RAD,
    )
    scene.start_tactile()
    scene.set_expert_params(randomization.expert_context())
    randomization_meta = randomization.to_metadata()
    # Pre-rollout steps are not episode frames; record what actually ran.
    randomization_meta["settle.mode"] = (
        "converged" if settle_window is not None else "fixed"
    )
    randomization_meta["settle.steps"] = float(result.executed_steps)
    randomization_meta["settle.steps_requested"] = float(result.requested_steps)
    if result.converged is None:
        randomization_meta["settle.converged"] = "n/a"
    else:
        randomization_meta["settle.converged"] = "true" if result.converged else "false"
        if result.max_pos_delta_m is not None:
            randomization_meta["settle.max_pos_delta_m"] = result.max_pos_delta_m
        if result.max_rot_delta_rad is not None:
            randomization_meta["settle.max_rot_delta_rad"] = result.max_rot_delta_rad
        if not result.converged:
            print(
                "openvitac: warning: settle did not converge within "
                f"{result.executed_steps} steps "
                f"(pos={result.max_pos_delta_m}, rot={result.max_rot_delta_rad})"
            )
    for prim_path in randomization.actor_poses:
        pos, quat = scene.object_pose(prim_path)
        leaf = prim_path.rsplit("/", 1)[-1]
        for axis, value in zip(("x", "y", "z"), pos, strict=True):
            randomization_meta[f"actual.{leaf}.pos_{axis}"] = float(value)
        for axis, value in zip(("x", "y", "z", "w"), quat, strict=True):
            randomization_meta[f"actual.{leaf}.quat_{axis}"] = float(value)
    scene.warmup(5)
    return randomization_meta


def _record(
    task: TaskName,
    scene: sim.Scene,
    tactile_cfg: TactileCfg | None,
    episode: Path,
    steps: int,
    seed: int,
    expert: bool,
    *,
    record_every: int,
    settle_steps: int | None,
    settle_window: int | None,
    logger: _TaskLogger | None = None,
) -> tuple[int, int]:
    """Run the episode with an already built scene; return (frames, cameras).

    The HDF5 writer is closed before returning so the caller can render video
    from the finished file; that must happen before ``sim.launch`` exits because
    Kit's fast shutdown ends the process there.  The caller owns ``scene`` and
    closes it after video rendering, on success and failure alike.
    """
    from openvitac_data import (
        SCHEMA_VERSION,
        ActionSpec,
        CameraMeta,
        EpisodeExtra,
        EpisodeMeta,
        Frame,
        Hdf5EpisodeWriter,
        PinholeIntrinsics,
    )

    spec = TASKS[task]
    tactile = tactile_cfg is not None
    effective_settle_steps = _effective_settle_steps(spec, settle_steps)
    try:
        randomization = spec.setup(seed, logger=logger)
    except Exception:
        if logger is not None:
            logger.exception("task setup failed; re-raising")
        raise
    # SceneObservation.rgb is keyed by prim_path; the schema keys images by role.
    roles = {camera.prim_path: camera.role for camera in spec.cameras}
    randomization_meta = _prepare_episode(
        scene, randomization, effective_settle_steps, settle_window
    )
    tactile_meta: tuple[PadTactileMeta, ...] = ()
    object_meta: tuple[ObjectMeta, ...] = ()
    if tactile_cfg is not None:
        pad_surfaces = scene.tactile_pad_surfaces()
        if not pad_surfaces:
            raise RuntimeError("tactile bridge is unavailable after Scene.reset()")
        tactile_meta, object_meta = _tactile_meta(
            tactile_cfg,
            pad_world_frames=scene.tactile_pad_world_frames(),
            pad_surfaces=pad_surfaces,
            depth_specs=scene.tactile_depth_specs,
        )
    # Resolved at runtime from the articulation; public order = joint_pos order.
    joint_names = scene.arm.joint_names
    cameras = tuple(
        CameraMeta(
            role=camera.role,
            prim_path=camera.prim_path,
            width=camera.width,
            height=camera.height,
            pos=camera.pos,
            rot=camera.rot,
            intrinsics=(
                None
                if camera.intrinsics is None
                else PinholeIntrinsics(
                    focal_length=camera.intrinsics.focal_length,
                    focus_distance=camera.intrinsics.focus_distance,
                    horizontal_aperture=camera.intrinsics.horizontal_aperture,
                    clipping_range=camera.intrinsics.clipping_range,
                )
            ),
            mount=camera.mount,
        )
        for camera in spec.cameras
    )
    sim_dt_us = _SIM_DT_US
    meta = EpisodeMeta(
        schema_version=SCHEMA_VERSION,
        task_id=spec.name,
        sim_dt=sim_dt_us,
        joint_names=joint_names,
        cameras=cameras,
        action=ActionSpec(representation="abs_joint_pos", joint_names=joint_names),
        extra=EpisodeExtra(
            seed=seed,
            instruction=randomization.instruction,
            # Non-expert runs have no termination predicate; the expert branch
            # rewrites this once success/timeout is known (set_outcome).
            outcome="unknown",
            # Flat scalar attrs in v3 /meta/randomization; no schema bump.
            randomization=randomization_meta,
        ),
        episode_id=episode.parent.name,
        tactile_status="ok" if tactile else "disabled",
        tactile=tactile_meta,
        objects=object_meta,
    )

    with Hdf5EpisodeWriter(episode, meta) as writer:

        def sink(obs: sim.SceneObservation) -> None:
            # Real physics time, not the recorded-frame number: with every=2 the
            # two differ by the recording cadence. Observed steps are 0, 2, 4, ...
            timestamp_us = obs.physics_step * sim_dt_us
            tactile_frames = None
            object_frames = None
            if tactile:
                assert tactile_cfg is not None
                if obs.tactile is None:
                    raise RuntimeError("tactile observation is unavailable")
                tactile_frames, object_frames = _tactile_payload(
                    obs.tactile,
                    timestamp_us=timestamp_us,
                    object_prim_paths={
                        obj.name: obj.prim_path for obj in tactile_cfg.objects
                    },
                    object_authorities={
                        obj.name: obj.authority for obj in tactile_cfg.objects
                    },
                    pose_reader=scene.object_pose,
                )
            frame = Frame(
                images={
                    roles[prim_path]: image for prim_path, image in obs.rgb.items()
                },
                joint_pos=obs.joint_pos,
                joint_vel=obs.joint_vel,
                action=obs.action,
                tactile=tactile_frames,
                objects=object_frames,
                timestamp_us=timestamp_us,
            )
            writer.append(frame)

        if not expert:
            scene.collect(sink, steps=steps, every=record_every)
            return writer.frame_count, len(cameras)

        machine = spec.expert
        outcome = _run_state_rollout(
            scene,
            machine,
            spec.success,
            steps=steps,
            sink=sink,
            record_every=record_every,
            logger=logger,
        )
        # Outcome is only known after the rollout, inside the open writer.
        writer.set_outcome(outcome)
        return writer.frame_count, len(cameras)
