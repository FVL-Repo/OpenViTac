"""Sensor kind registry: which sim adapter implements each sensor kind."""

from __future__ import annotations

from typing import Protocol, cast

from openvitac_spec.sensor import SensorBinding, SensorKind

from .base import Sensor
from .gsmini import GsMiniSensor

SENSOR_CLASSES: dict[SensorKind, type[Sensor]] = {
    "gsmini": GsMiniSensor,
    "neote": GsMiniSensor,
}


class _SensorCtor(Protocol):
    """The construction contract every registered adapter shares."""

    def __call__(
        self,
        *,
        binding: SensorBinding,
        pad_key: str,
        optical_backend: str,
        taxim_device: str,
        finger_body: str | None = None,
    ) -> Sensor: ...


def sensor_class(kind: SensorKind) -> type[Sensor] | None:
    """Adapter class for ``kind``, or ``None`` when it has no sim-side pass."""
    return SENSOR_CLASSES.get(kind)


def sensor_has_pad_proxy(kind: SensorKind) -> bool:
    """Whether the registered sensor's case USD carries ``pad_proxy`` cubes."""
    sensor = SENSOR_CLASSES.get(kind)
    return bool(getattr(sensor, "pad_proxy", False))


def make_sensor(
    kind: SensorKind,
    *,
    binding: SensorBinding,
    pad_key: str,
    optical_backend: str,
    taxim_device: str,
    finger_body: str | None = None,
) -> Sensor:
    """Construct the adapter for ``kind``; fail loud when unregistered."""
    cls = SENSOR_CLASSES.get(kind)
    if cls is None:
        raise ValueError(f"sensor kind {kind!r} has no sim-side adapter")
    factory = cast(_SensorCtor, cls)
    return factory(
        binding=binding,
        pad_key=pad_key,
        optical_backend=optical_backend,
        taxim_device=taxim_device,
        finger_body=finger_body,
    )
