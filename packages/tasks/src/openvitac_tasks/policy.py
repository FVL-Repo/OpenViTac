"""Composable per-tick expert steps: the policy object is the state.

A policy builds a :class:`Sequence` of :class:`Step` objects on its first tick,
and each step then drives the arm until it reports itself finished, so a task
needs no phase enum, no phase transition table and no per-tick
``(state, action)`` tuples.

A step overrides only the ``Action`` fields it drives; the rest come from the
:class:`Command` the sequence hands it.  A hold therefore re-issues the pose the
previous move ended on, and a move keeps the gripper target it was last given.
The command has one owner, the outermost policy: it starts as the live arm state
and every emitted action replaces it, so a step that takes over mid-tick is
handed the command the finished step was driving rather than the measured pose.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple, Protocol

from .expert import Action, ExpertWorld, Quat, Vec3
from .motion import (
    GRASP_OVERTRAVEL,
    Move,
    advance_move,
    begin_move,
    ee_for_tool_center,
    pad_close_q,
)

# A target point in the frame the move's own class declares (a tool center or an
# EE point), or a zero-argument callable producing one.  The callable is read on
# the step's first tick, which is when a target derived from a moving object's
# pose has to be sampled.
type Point = Vec3 | Callable[[], Vec3]


class Command(NamedTuple):
    """The pose the policy currently commands, or the live arm before any."""

    ee_pos: Vec3
    ee_quat: Quat
    gripper_qpos: float


class Step(Protocol):
    """One bounded stretch of control ticks."""

    finished: bool

    def step(self, world: ExpertWorld, cmd: Command) -> Action | None:
        """This tick's action, or ``None`` on the tick the step finishes.

        The step also sets ``finished`` on that tick; calling ``step`` again
        afterwards raises ``RuntimeError``.
        """
        ...


class Sequence:
    """Steps in order; a handover spends no tick.

    ``step`` asks the active step for an action and, when that step reports it
    has finished, moves on to the next one *within the same tick*: a moved-on
    step never costs a control tick.  ``None`` comes back only once every step
    has finished.  ``cmd`` is forwarded unchanged, so the step that takes over
    sees the command the finished step was handed.
    """

    def __init__(self, *steps: Step) -> None:
        self._steps = tuple(steps)
        self._index = 0
        self.finished = False

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    @property
    def index(self) -> int:
        """Position of the active step; ``len(self)`` once everything ran."""
        return self._index

    @property
    def current(self) -> Step | None:
        """The active step, or ``None`` once everything ran."""
        return self._steps[self._index] if self._index < len(self._steps) else None

    def step(self, world: ExpertWorld, cmd: Command) -> Action | None:
        if self.finished:
            raise RuntimeError("sequence already finished")
        while self._index < len(self._steps):
            action = self._steps[self._index].step(world, cmd)
            if action is not None:
                return action
            self._index += 1
        self.finished = True
        return None


class Policy:
    """A per-tick policy: owns the command and the step sequence built from it.

    The first tick builds the plan and seeds the command from the live arm,
    which is the only time the measured pose enters it; every later tick
    forwards the command to the active step and takes that step's action as the
    next command.
    """

    def __init__(self) -> None:
        self.steps: Sequence | None = None
        self.command: Command | None = None

    def plan(self, world: ExpertWorld) -> Sequence:
        """The steps of one episode, sampled when the policy first runs."""
        raise NotImplementedError

    def step(self, world: ExpertWorld) -> Action | None:
        if self.steps is None:
            steps = self.plan(world)
            pos, quat = world.arm.ee_pose()
            self.steps = steps
            self.command = Command(pos, quat, world.arm.gripper.joint_pos())
        command = self.command
        if command is None:
            raise AssertionError("policy ran its plan without a command")
        action = self.steps.step(world, command)
        if action is None:
            return None
        self.command = Command(action.ee_pos, action.ee_quat, action.gripper_qpos)
        return action


def _point(target: Point) -> Vec3:
    return target() if callable(target) else target


class _DriveTo:
    """Drive the arm to the EE point a subclass derives, then settle there.

    Both moves below share this one per-tick loop; they differ only in the frame
    of the ``target`` they were handed, which is all ``ee_goal`` has to say: it
    reads that point and returns the EE (arm link) point ``begin_move``
    commands.  ``ticks`` are the interpolation steps to the goal and
    ``settle_ticks`` the re-command from the pose reached, which is what let the
    servo converge.
    """

    def __init__(
        self,
        target: Point,
        quat: Quat,
        *,
        ticks: int = 60,
        settle_ticks: int = 60,
    ) -> None:
        self.target = target
        self.quat = quat
        self.ticks = ticks
        self.settle_ticks = settle_ticks
        self.finished = False
        self._move: Move | None = None

    def ee_goal(self, world: ExpertWorld) -> Vec3:
        """The EE point to reach, read once on the step's first tick."""
        raise NotImplementedError

    def step(self, world: ExpertWorld, cmd: Command) -> Action | None:
        if self.finished:
            raise RuntimeError("move already finished")
        if self._move is None:
            self._move, action = begin_move(
                world,
                self.ee_goal(world),
                self.quat,
                ticks=self.ticks,
                settle_ticks=self.settle_ticks,
                gripper_qpos=cmd.gripper_qpos,
            )
            return action
        advanced = advance_move(world, self._move, cmd.gripper_qpos)
        if advanced is None:
            self.finished = True
            return None
        self._move, action = advanced
        return action


class MoveToolTo(_DriveTo):
    """Drive the arm until the tool center reaches ``target``, then settle.

    ``target`` is a *tool-center* point -- the pad midpoint the robot asset
    declares -- so ``ee_goal`` places it with ``ee_for_tool_center``: the EE
    point it commands depends on the hand orientation and the asset's own
    offset, and ``clearance`` backs the EE off further along the hand's axis.
    No gripper constant lives here.

    :class:`MoveEeTo` takes the same arguments in the other frame: an *EE*
    point, the arm link's own origin, which it commands unchanged.
    """

    def __init__(
        self,
        target: Point,
        quat: Quat,
        *,
        clearance: float = 0.0,
        ticks: int = 60,
        settle_ticks: int = 60,
    ) -> None:
        super().__init__(target, quat, ticks=ticks, settle_ticks=settle_ticks)
        self.clearance = clearance

    def ee_goal(self, world: ExpertWorld) -> Vec3:
        return ee_for_tool_center(
            _point(self.target),
            self.quat,
            world.arm.tool,
            clearance=self.clearance,
        )


class MoveEeTo(_DriveTo):
    """Drive the arm until its EE link reaches ``target``, then settle.

    ``target`` is an *EE* point -- the arm link's origin, the point
    ``begin_move`` commands -- so it goes to the solver as given and there is no
    clearance to give: this is the move for a goal stated in the arm's frame,
    such as a retreat along world axes rather than along the hand.

    :class:`MoveToolTo` takes the same arguments in the other frame: a
    *tool-center* point, which it places through the asset's hand offset.
    """

    def ee_goal(self, world: ExpertWorld) -> Vec3:
        return _point(self.target)


class _Hold:
    """Re-issue the current command with one gripper target for n ticks.

    Only the gripper changes: the EE target stays the pose the previous step
    last commanded, so the hold pins the arm where it already is.
    """

    def __init__(self, hold_ticks: int) -> None:
        if hold_ticks < 0:
            raise ValueError(f"hold_ticks must be non-negative, got {hold_ticks}")
        self.hold_ticks = hold_ticks
        self.finished = False
        self._qpos: float | None = None
        self._left = 0

    def gripper_qpos(self, world: ExpertWorld, cmd: Command) -> float:
        """The target the hold commands; read once, on the first tick."""
        raise NotImplementedError

    def step(self, world: ExpertWorld, cmd: Command) -> Action | None:
        if self.finished:
            raise RuntimeError("hold already finished")
        if self._qpos is None:
            self._qpos = self.gripper_qpos(world, cmd)
            self._left = self.hold_ticks
        if self._left <= 0:
            self.finished = True
            return None
        self._left -= 1
        return Action(cmd.ee_pos, cmd.ee_quat, self._qpos)


class CloseTo(_Hold):
    """Close onto ``width``, the object's own span, through the close law."""

    def __init__(
        self,
        width: float,
        *,
        overtravel: float = GRASP_OVERTRAVEL,
        hold_ticks: int = 60,
    ) -> None:
        super().__init__(hold_ticks)
        self.width = width
        self.overtravel = overtravel

    def gripper_qpos(self, world: ExpertWorld, cmd: Command) -> float:
        return pad_close_q(world.arm.tool, self.width, self.overtravel)


class OpenGripper(_Hold):
    """Command the embodiment's fully open gripper target."""

    def __init__(self, *, hold_ticks: int = 60) -> None:
        super().__init__(hold_ticks)

    def gripper_qpos(self, world: ExpertWorld, cmd: Command) -> float:
        return world.arm.gripper.open_qpos


class Wait(_Hold):
    """Hold the current command unchanged."""

    def __init__(self, ticks: int) -> None:
        super().__init__(ticks)

    def gripper_qpos(self, world: ExpertWorld, cmd: Command) -> float:
        return cmd.gripper_qpos
