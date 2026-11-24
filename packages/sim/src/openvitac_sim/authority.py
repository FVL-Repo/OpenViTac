"""Who simulates each dynamic actor, and refusing scenes that cannot honour it.

``Actor.authority`` is the task-side declaration of which world a dynamic prop
lives in.  ``"physx"`` means PhysX simulates the spawned USD body.  ``"uipc"``
means the tactile worker's UIPC world owns it: PhysX only carries a kinematic,
collision-free proxy whose pose the worker writes every step.

That proxy is meaningful only while the worker runs, because the worker is its
only writer.  A scene without one would spawn a body that nothing moves and
nothing collides with -- it cannot be grasped, cannot fall, and every success
predicate waiting on it stays false for the whole episode, while the scene still
renders, still steps and still writes an episode.  That silent ghost is why a
declared ``"uipc"`` actor without a tactile worker is refused here rather than
quietly handed to PhysX: the declaration is the task saying this prop must be
tactile-simulated, and a run that cannot do that is a broken configuration, not
a different configuration.

A worker that is present but fails belongs to the worker; it must surface as its
own error, never as a rerouted actor.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

__all__ = ["require_tactile_worker"]


class _Actor(Protocol):
    """The part of an actor this check reads (a task spec ``Actor``)."""

    @property
    def prim_path(self) -> str: ...

    @property
    def authority(self) -> str: ...

    @property
    def dynamic(self) -> bool: ...


def require_tactile_worker(actors: Iterable[_Actor], *, tactile_worker: bool) -> None:
    """Refuse dynamic ``uipc`` actors in a scene that runs no tactile worker.

    ``tactile_worker`` is whether this scene will actually run one, not whether
    the task asked for one.  Static actors are never UIPC (the spec rejects
    that), so they cannot trip this.
    """
    if tactile_worker:
        return
    orphaned = sorted(
        actor.prim_path
        for actor in actors
        if actor.dynamic and actor.authority == "uipc"
    )
    if not orphaned:
        return
    raise RuntimeError(
        "these dynamic actors declare authority='uipc' but this scene runs no "
        f"tactile worker, so nothing would ever simulate them: {', '.join(orphaned)}.  "
        "A 'uipc' actor is owned by the tactile worker; without one it would be a "
        "kinematic, collision-free ghost that cannot be grasped or fall, and no "
        "success predicate could fire.  Bind tactile sensor pads to this "
        "embodiment, or declare these actors as authority='physx'."
    )
