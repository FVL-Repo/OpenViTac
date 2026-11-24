"""A ``uipc`` actor without a tactile worker is refused, not silently downgraded."""

from __future__ import annotations

import pytest

from openvitac_sim import Actor, require_tactile_worker


def _actor(prim: str, authority: str, *, dynamic: bool = True) -> Actor:
    return Actor(usd="x.usd", prim_path=prim, authority=authority, dynamic=dynamic)


def test_uipc_actor_without_a_worker_is_refused() -> None:
    actors = [
        _actor("/World/Table", "physx", dynamic=False),
        _actor("/World/BlockA", "uipc"),
        _actor("/World/BlockB", "uipc"),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        require_tactile_worker(actors, tactile_worker=False)
    message = str(excinfo.value)
    assert "/World/BlockA" in message
    assert "/World/BlockB" in message
    # The static fixture is not the offender, and neither is an unrelated name.
    assert "/World/Table" not in message


def test_physx_only_scene_passes_without_a_worker() -> None:
    actors = [_actor("/World/Block", "physx")]
    require_tactile_worker(actors, tactile_worker=False)


def test_uipc_actor_is_fine_when_the_worker_runs() -> None:
    actors = [_actor("/World/Block", "uipc")]
    require_tactile_worker(actors, tactile_worker=True)


def test_empty_scene_passes_without_a_worker() -> None:
    require_tactile_worker((), tactile_worker=False)
