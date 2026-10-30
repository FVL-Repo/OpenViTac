"""Task spec: a frozen, pure-data description of a task's nominal contents.

Tasks hold numbers only — USD paths, quaternions, camera parameters, sensor
bindings — and produce the pure :class:`openvitac_spec.SceneSpec` consumed by
``sim.Scene``.  The sim owns mechanism and translates specs into isaaclab cfgs at
build time.  This package imports on any machine; it must not import
``openvitac_sim``, isaaclab, pxr, or pyuipc.

The scripted policy is callable task content: ``expert`` is an explicit
per-tick :class:`ExpertStateMachine` and ``success`` is a per-tick verdict
generator, both typed against :class:`ExpertWorld`.  The collector applies each
Action, steps physics once, and advances the success iterator; the first True
ends the episode.  The capability-gated gripper handle does not mention a sensor
variant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from openvitac_spec import (
    Actor,
    ActorPose,
    CameraSpec,
    EmbodimentSpec,
    GripperCapability,
    Joint,
    LightSpec,
    SceneSpec,
    SensorBinding,
)

from .expert import (
    ExpertStateMachine,
    SuccessFn,
    TaskLogger,
)


@dataclass(frozen=True)
class EpisodeRandomization:
    """Per-episode sampled content: actor pose overrides plus flat metadata.

    ``actor_poses`` keys are full prim paths.  The maps stay pure Python data so
    ``Task.setup`` can run before the sim app is launched.  ``metadata`` and
    ``to_metadata`` values are restricted to HDF5-attribute scalar types.
    """

    seed: int
    instruction: str
    actor_poses: Mapping[str, ActorPose] = field(
        default_factory=dict[str, ActorPose]
    )
    expert_params: Mapping[str, float] = field(default_factory=dict[str, float])
    contact_object: str | None = None
    metadata: Mapping[str, float | str] = field(
        default_factory=dict[str, float | str]
    )

    def to_metadata(self) -> dict[str, float | str]:
        """Flatten to the ``/meta/randomization`` attribute vocabulary."""
        out: dict[str, float | str] = dict(self.metadata)
        for prim_path, pose in self.actor_poses.items():
            leaf = prim_path.rsplit("/", 1)[-1]
            for axis, value in zip(
                ("x", "y", "z"), pose.position, strict=True
            ):
                out[f"actor.{leaf}.pos_{axis}"] = float(value)
            for axis, value in zip(
                ("x", "y", "z", "w"), pose.quaternion_xyzw, strict=True
            ):
                out[f"actor.{leaf}.quat_{axis}"] = float(value)
        for name, value in self.expert_params.items():
            out[f"expert.{name}"] = float(value)
        return out

    def __post_init__(self) -> None:
        duplicates = set(self.expert_params) & set(self.metadata)
        if duplicates:
            raise ValueError(
                "expert_params must not reuse metadata keys: "
                + ", ".join(sorted(duplicates))
            )

    def expert_context(self) -> dict[str, float | str]:
        """Merge metadata and expert-only params for ``ExpertWorld`` lookup."""
        return {**self.metadata, **self.expert_params}


@dataclass(frozen=True)
class Task:
    """One task's nominal spec, registered in ``openvitac_tasks.TASKS`` by name."""

    name: str
    embodiment: EmbodimentSpec
    sensors: tuple[SensorBinding, ...]
    cameras: tuple[CameraSpec, ...]  # every entry is recorded, keyed by ``role``
    actors: tuple[Actor, ...]  # includes the infrastructure plate explicitly
    instruction: str
    expert: ExpertStateMachine[Any]
    success: SuccessFn
    light: LightSpec = field(default_factory=LightSpec)
    joints: tuple[Joint, ...] = ()  # mechanism links between actors
    required_capabilities: frozenset[GripperCapability] = frozenset()
    joint_pos: dict[str, float] | None = None  # None = embodiment default
    settle_steps: int = 20  # pre-rollout physics steps; 0 = legacy no-settle
    randomizer: Callable[[Task, int], EpisodeRandomization] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.settle_steps < 0:
            raise ValueError(
                f"task {self.name!r} settle_steps must be non-negative, "
                f"got {self.settle_steps}"
            )

    def scene_spec(self) -> SceneSpec:
        """Scene content for ``sim.Scene``; run params (steps/dt) stay with the cli."""
        light = self.light
        if light.texture_file is None:
            light = replace(light, texture_file="base5.exr")
        return SceneSpec(
            embodiment=self.embodiment,
            sensors=self.sensors,
            cameras=self.cameras,
            actors=self.actors,
            light=light,
            joints=self.joints,
            joint_pos=self.joint_pos,
        )

    def setup(
        self, seed: int, *, logger: TaskLogger | None = None
    ) -> EpisodeRandomization:
        """Return this episode's sampled content, preserving the instruction.

        ``logger`` is runtime-only and never stored on this frozen spec.  The
        randomizer stays pure ``(Task, seed)``; this method logs the seed and the
        returned sampling summary so task authors do not need to inject logging
        into every sampler.
        """
        if logger is not None:
            logger.info("task setup start seed=%d task=%s", seed, self.name)
        if self.randomizer is None:
            randomization = EpisodeRandomization(
                seed=seed, instruction=self.instruction
            )
        else:
            randomization = self.randomizer(self, seed)
        if logger is not None:
            logger.info(
                "task setup sampled instruction=%r metadata=%s "
                "actor_poses=%d expert_params=%d",
                randomization.instruction,
                dict(randomization.metadata),
                len(randomization.actor_poses),
                len(randomization.expert_params),
            )
        return randomization
