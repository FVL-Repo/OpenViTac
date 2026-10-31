"""Runtime sensor abstraction: attach, bind once, sync and observe each tick."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from numpy.typing import NDArray

from openvitac_spec.sensor import StreamKind

from .ids import BodyId, CameraId, Pose
from .snapshot import Requires, TickSnapshot


class SensorObservation(Protocol):
    """What the writer consumes: named streams plus optional extras."""

    def streams(self) -> Mapping[StreamKind, NDArray[Any]]: ...

    def extras(self) -> Mapping[str, Any]: ...


class ProxyHandle(Protocol):
    """A cosmetic mesh the sensor may rewrite at bind and per rendered frame."""

    def update(self, vertices: NDArray[Any], triangles: NDArray[Any]) -> None: ...


class SoftBodyHandle(Protocol):
    """A soft body the sensor owns: its id plus this tick's constraint write."""

    @property
    def id(self) -> BodyId: ...

    def stage(self, pose: Pose) -> None: ...


@dataclass(frozen=True)
class ContactObject:
    """A scene body whose surface may touch a sensor: an imaging candidate.

    The scene resolves these from its actor declarations; a sensor never names
    an object itself.  ``id`` is the actor's body id, ``prim_path`` the spawned
    prim and ``usd_path`` the authored asset whose surface is z-buffered.
    """

    id: BodyId
    prim_path: str
    usd_path: str


@dataclass(frozen=True)
class PadCameraRequest:
    """The authored pad camera whose calibration a sensor consumes.

    The camera is not spawned as a render product: the sensor needs its
    intrinsics and its fixed pose relative to the pad, and the recorded depth
    stream is analytic object depth, not a render.
    """

    side: str
    width: int
    height: int


@dataclass(frozen=True)
class Handles:
    """Opaque world handles returned by ``Sensor.attach``."""

    body: BodyId
    soft_body: SoftBodyHandle
    camera: CameraId | None
    proxy: ProxyHandle
    objects: tuple[ContactObject, ...] = ()


class WorldBuilder(Protocol):
    """Narrow build-time API available to a sensor's ``attach``."""

    def body(self, name: str) -> BodyId | None: ...

    def add_soft_body(self, pad_key: str) -> SoftBodyHandle: ...

    def add_pad_camera(self, prim_path: str, spec: PadCameraRequest) -> CameraId: ...

    def add_visual_proxy(self, prim_path: str) -> ProxyHandle: ...

    def remove_pad_proxy(self, body: BodyId) -> None: ...

    def contact_objects(self) -> tuple[ContactObject, ...]:
        """Bodies whose USD surface may touch this sensor; never sensor-owned."""
        ...

    def frames(self) -> Mapping[str, Pose]: ...


class World(Protocol):
    """Bind-time: resolve the world-side handles a sensor needs once."""

    def pad_surface(
        self, body: BodyId
    ) -> tuple[NDArray[Any], NDArray[Any]]: ...

    def pad_frame(self, body: BodyId) -> NDArray[Any]:
        """World <- pad calibration frame of the gelpad at the bind instant."""
        ...

    def rest_camera_pose(self, body: BodyId) -> NDArray[Any]:
        """Rest calibration pose used to bind the sensor (not the live pose)."""
        ...

    def camera_intrinsics(self, body: BodyId) -> NDArray[Any]: ...


class BoundSensor(Protocol):
    """A sensor calibrated against one world, ready to drive ticks."""

    def sync(self, snap: TickSnapshot) -> None: ...

    def publish(self, snap: TickSnapshot) -> None: ...

    def observe(self, snap: TickSnapshot) -> SensorObservation: ...


class Sensor(Protocol):
    """A declarative sensor: what it emits, what it needs, and how to build it."""

    @property
    def name(self) -> str: ...

    @property
    def outputs(self) -> frozenset[StreamKind]: ...

    def requires(self) -> Requires: ...

    def attach(self, builder: WorldBuilder) -> Handles: ...

    def bind(self, world: World) -> BoundSensor: ...
