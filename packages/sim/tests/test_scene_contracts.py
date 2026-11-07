"""Scene build contracts: tactile render streams and UIPC pad proxies."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

from openvitac_sim.scene import (
    _require_supported_render_streams,
    _requires_pad_proxy_disabling,
    _unsupported_render_requests,
)


def _spec(sensor_kind: str, *, dynamic_authority: str | None = None):
    sensors = (
        SimpleNamespace(pad_key="left", sensor=SimpleNamespace(kind=sensor_kind)),
    )
    actors = (
        ()
        if dynamic_authority is None
        else (SimpleNamespace(dynamic=True, authority=dynamic_authority),)
    )
    return SimpleNamespace(sensors=sensors, actors=actors)


def _tactile(render_kinds: tuple[str, ...]):
    pad = SimpleNamespace(name="left", render_kinds=render_kinds)
    return SimpleNamespace(pads=(pad,))


class RenderStreamContractTests(unittest.TestCase):
    def test_gsmini_rendering_is_supported(self) -> None:
        spec = _spec("gsmini")
        tactile = _tactile(("rgb", "depth", "marker_rgb"))
        self.assertEqual(_unsupported_render_requests(spec, tactile), [])
        _require_supported_render_streams(spec, tactile)

    def test_xense_rgb_is_rejected(self) -> None:
        spec = _spec("xense_ws")
        tactile = _tactile(("rgb",))
        self.assertEqual(
            _unsupported_render_requests(spec, tactile),
            ["left (xense_ws): ['rgb']"],
        )
        with self.assertRaises(NotImplementedError):
            _require_supported_render_streams(spec, tactile)

    def test_xense_without_render_streams_is_allowed(self) -> None:
        self.assertEqual(
            _unsupported_render_requests(
                _spec("xense_ws"),
                _tactile(()),
            ),
            [],
        )


class PadProxyContractTests(unittest.TestCase):
    def test_gsmini_uipc_scene_requires_pad_proxies(self) -> None:
        self.assertTrue(
            _requires_pad_proxy_disabling(
                _spec("gsmini", dynamic_authority="uipc"),
                cast(Any, object()),
            )
        )

    def test_xense_uipc_scene_does_not_require_pad_proxies(self) -> None:
        self.assertFalse(
            _requires_pad_proxy_disabling(
                _spec("xense_ws", dynamic_authority="uipc"),
                cast(Any, object()),
            )
        )

    def test_physx_and_missing_bridge_do_not_require_pad_proxies(self) -> None:
        self.assertFalse(
            _requires_pad_proxy_disabling(
                _spec("gsmini", dynamic_authority="physx"),
                cast(Any, object()),
            )
        )
        self.assertFalse(
            _requires_pad_proxy_disabling(
                _spec("gsmini", dynamic_authority="uipc"),
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
