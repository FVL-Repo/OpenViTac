"""RFCL building blocks for privileged-state data generation.

This module deliberately contains no Isaac/vision dependency.  It converts
successful InsertUSB HDF5 demonstrations into a low-dimensional transition
format and implements the two pieces of RFCL that are independent of the
underlying off-policy learner:

* a per-demonstration reverse curriculum frontier;
* a replay buffer with an explicit demo/online sampling ratio.

The Isaac/UIPC environment adapter and persistent snapshot implementation live
in :mod:`policy.RL.rfcl_env` and :mod:`policy.RL.rfcl_snapshot`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

from policy.RL.action import target_qpos_to_action


DEFAULT_INSERTION_START_TAG = "move_usb_to_pre_insert"
DEFAULT_INSERTION_TAGS = (
    "move_usb_to_pre_insert",
    "move_usb_to_play_pre_insert",
    "insert_usb_into_slot",
    "delay",
    "open_gripper_after_insert",
    "",
)


@dataclass(frozen=True)
class PrivilegedStateLayout:
    """Feature layout for the state-only RFCL actor.

    ``joint_delta`` is a finite-difference proxy because the existing HDF5
    schema does not store simulator qvel.  The live adapter must replace it
    with ``robot.data.joint_vel`` before training an online policy.
    """

    joint: slice
    joint_delta: slice
    ee_pose: slice
    controlled_pose: slice
    target_pose: slice
    controlled_in_target: slice
    dim: int


def _slice(start: int, size: int) -> slice:
    return slice(start, start + size)


PRIVILEGED_STATE_LAYOUT = PrivilegedStateLayout(
    joint=_slice(0, 9),
    joint_delta=_slice(9, 9),
    ee_pose=_slice(18, 7),
    controlled_pose=_slice(25, 7),
    target_pose=_slice(32, 7),
    controlled_in_target=_slice(39, 7),
    dim=46,
)


# ``embodiment/ee`` is the wrist/hand body pose.  USB alignment is defined at
# the gripper centre, which is offset along the EE local +Z axis.
DEFAULT_GRIPPER_OFFSET = 0.131


def _as_float_array(value: object, *, name: str, ndim: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(first, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zero = np.zeros((*vector.shape[:-1], 1), dtype=np.float64)
    vector_quaternion = np.concatenate((zero, vector.astype(np.float64)), axis=-1)
    rotated = _quat_multiply(
        _quat_multiply(quaternion, vector_quaternion),
        _quat_conjugate(quaternion),
    )
    return rotated[..., 1:]


def relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return ``pose`` expressed in ``reference`` coordinates.

    Both inputs use the repository's ``[x, y, z, qw, qx, qy, qz]`` format.
    The implementation is vectorized and does not import Isaac Sim.
    """

    pose = _as_float_array(pose, name="pose")
    reference = _as_float_array(reference, name="reference")
    if pose.shape != reference.shape or pose.shape[-1] != 7:
        raise ValueError(
            "pose and reference must have the same shape (..., 7), "
            f"got {pose.shape} and {reference.shape}"
        )

    reference_quaternion = reference[..., 3:7].astype(np.float64)
    reference_quaternion /= np.maximum(
        np.linalg.norm(reference_quaternion, axis=-1, keepdims=True),
        1e-12,
    )
    pose_quaternion = pose[..., 3:7].astype(np.float64)
    pose_quaternion /= np.maximum(
        np.linalg.norm(pose_quaternion, axis=-1, keepdims=True),
        1e-12,
    )
    reference_inverse = _quat_conjugate(reference_quaternion)
    translation = pose[..., :3].astype(np.float64) - reference[..., :3]
    relative_translation = _quat_rotate(reference_inverse, translation)
    relative_quaternion = _quat_multiply(reference_inverse, pose_quaternion)
    relative_quaternion /= np.maximum(
        np.linalg.norm(relative_quaternion, axis=-1, keepdims=True),
        1e-12,
    )
    return np.concatenate(
        (relative_translation, relative_quaternion),
        axis=-1,
    ).astype(np.float32)


def gripper_center_position_from_ee(
    ee_pose: np.ndarray,
    *,
    gripper_offset: float = DEFAULT_GRIPPER_OFFSET,
) -> np.ndarray:
    """Convert wrist/EE poses to gripper-centre positions."""

    pose = np.asarray(ee_pose, dtype=np.float64)
    if pose.shape[-1] != 7:
        raise ValueError(f"ee_pose must end in 7 values, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("ee_pose contains NaN or infinite values")
    offset = float(gripper_offset)
    if not np.isfinite(offset) or offset <= 0.0:
        raise ValueError("gripper_offset must be finite and positive")
    quaternion = pose[..., 3:7].copy()
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
    )
    local_offset = np.zeros((*pose.shape[:-1], 3), dtype=np.float64)
    local_offset[..., 2] = offset
    return (pose[..., :3] + _quat_rotate(quaternion, local_offset)).astype(
        np.float32
    )


def handoff_xy_error(
    ee_pose: np.ndarray,
    usb_pose: np.ndarray,
    *,
    gripper_offset: float = DEFAULT_GRIPPER_OFFSET,
) -> float:
    """Return gripper-centre to USB XY distance in metres."""

    ee = np.asarray(ee_pose, dtype=np.float64).reshape(-1)
    usb = np.asarray(usb_pose, dtype=np.float64).reshape(-1)
    if ee.shape != (7,) or usb.shape != (7,):
        raise ValueError("ee_pose and usb_pose must each contain 7 values")
    center = gripper_center_position_from_ee(
        ee, gripper_offset=gripper_offset
    )
    return float(np.linalg.norm(center[:2] - usb[:2]))


def build_privileged_states(
    *,
    joint: np.ndarray,
    ee_pose: np.ndarray,
    usb_pose: np.ndarray,
    slot_pose: np.ndarray,
    sim_dt: float = 1.0 / 120.0,
) -> np.ndarray:
    """Build state-only observations from one HDF5 episode.

    The returned dimension is 46: 9 joint positions, 9 finite-difference
    joint velocities, and four 7D poses (EE, USB, slot, USB-in-slot).
    ``sim_dt`` is only used to label the finite-difference proxy; the saved
    frame stride is handled by the caller before this function is called.
    """

    if not np.isfinite(sim_dt) or float(sim_dt) <= 0.0:
        raise ValueError("sim_dt must be finite and positive")
    joint = _as_float_array(joint, name="joint")
    ee_pose = _as_float_array(ee_pose, name="ee_pose")
    usb_pose = _as_float_array(usb_pose, name="usb_pose")
    slot_pose = _as_float_array(slot_pose, name="slot_pose")
    count = joint.shape[0]
    if joint.shape[1] < 9:
        raise ValueError(f"joint must contain at least 9 values, got {joint.shape}")
    for name, value in (
        ("ee_pose", ee_pose),
        ("usb_pose", usb_pose),
        ("slot_pose", slot_pose),
    ):
        if value.shape != (count, 7):
            raise ValueError(
                f"{name} must have shape ({count}, 7), got {value.shape}"
            )

    joint = joint[:, :9]
    joint_delta = np.zeros_like(joint)
    if count > 1:
        # Keep this as a finite-difference feature rather than calling it qvel:
        # HDF5 only stores observations every save_frequency frames.
        joint_delta[1:] = np.diff(joint, axis=0) / float(sim_dt)
    usb_in_slot = relative_pose(usb_pose, slot_pose)
    return np.concatenate(
        (joint, joint_delta, ee_pose, usb_pose, slot_pose, usb_in_slot),
        axis=-1,
    ).astype(np.float32)


def build_pair_privileged_state(
    *,
    joint: np.ndarray,
    joint_velocity: np.ndarray,
    ee_pose: np.ndarray,
    controlled_pose: np.ndarray,
    target_pose: np.ndarray,
) -> np.ndarray:
    """Build one online RFCL observation using the simulator's true qvel."""

    values = {}
    for name, value, width in (
        ("joint", joint, 9),
        ("joint_velocity", joint_velocity, 9),
        ("ee_pose", ee_pose, 7),
        ("controlled_pose", controlled_pose, 7),
        ("target_pose", target_pose, 7),
    ):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape[0] < width:
            raise ValueError(f"{name} must contain at least {width} values")
        array = array[:width]
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        values[name] = array

    controlled_in_target = relative_pose(
        values["controlled_pose"][None, :],
        values["target_pose"][None, :],
    )[0]
    state = np.concatenate(
        (
            values["joint"],
            values["joint_velocity"],
            values["ee_pose"],
            values["controlled_pose"],
            values["target_pose"],
            controlled_in_target,
        )
    ).astype(np.float32)
    if state.shape != (PRIVILEGED_STATE_LAYOUT.dim,):
        raise RuntimeError(f"Unexpected privileged state shape {state.shape}")
    return state


@dataclass(frozen=True)
class RFCLTransition:
    """A learner-independent transition used by the RFCL replay sampler."""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    terminated: bool
    demo_id: str | None
    timestep: int
    source: str

    def __post_init__(self) -> None:
        for name in ("state", "action", "next_state"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 1D array")
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.source not in ("demo", "online"):
            raise ValueError(f"Unsupported transition source {self.source!r}")


@dataclass(frozen=True)
class DemoTrajectory:
    """A suffix trajectory and its corresponding state/action transitions."""

    demo_id: str
    path: Path
    frame_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    tags: tuple[str, ...]
    velocity_source: str

    def __post_init__(self) -> None:
        state_count = len(self.frame_indices)
        if self.states.shape[0] != state_count:
            raise ValueError("states and frame_indices have inconsistent lengths")
        if self.actions.shape[0] != max(0, state_count - 1):
            raise ValueError("actions must contain one item per state transition")
        if self.rewards.shape != self.actions.shape[:1]:
            raise ValueError("rewards must contain one item per transition")
        if self.terminated.shape != self.actions.shape[:1]:
            raise ValueError("terminated must contain one item per transition")

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])

    def transitions(self) -> list[RFCLTransition]:
        return [
            RFCLTransition(
                state=self.states[index],
                action=self.actions[index],
                reward=float(self.rewards[index]),
                next_state=self.states[index + 1],
                terminated=bool(self.terminated[index]),
                demo_id=self.demo_id,
                timestep=index,
                source="demo",
            )
            for index in range(self.transition_count)
        ]


def _resolve_suffix_indices(
    tags: np.ndarray,
    *,
    start_tag: str,
    allowed_tags: Sequence[str],
) -> np.ndarray:
    tags = np.asarray(tags).astype(str)
    start_matches = np.flatnonzero(tags == start_tag)
    if not len(start_matches):
        raise ValueError(f"Missing start tag {start_tag!r}")
    start = int(start_matches[-1])
    allowed = set(str(tag) for tag in allowed_tags)
    suffix = np.arange(start, len(tags), dtype=np.int64)
    invalid = [int(index) for index in suffix if tags[index] not in allowed and tags[index] != ""]
    if invalid:
        raise ValueError(
            "Suffix contains tags outside allowed insertion stages; "
            f"first invalid frame={invalid[0]} tag={tags[invalid[0]]!r}"
        )
    return suffix


def load_insert_usb_demo(
    path: str | Path,
    *,
    action_scale: Sequence[float],
    start_tag: str = DEFAULT_INSERTION_START_TAG,
    allowed_tags: Sequence[str] = DEFAULT_INSERTION_TAGS,
    sim_dt: float = 1.0 / 120.0,
    episode_success: bool = True,
) -> DemoTrajectory:
    """Load one successful Motion Plan suffix as RFCL demo transitions."""

    path = Path(path)
    with h5py.File(path, "r") as hdf5_file:
        required = (
            "atom/tag",
            "embodiment/joint",
            "embodiment/ee",
            "actor/prism",
            "actor/slot",
        )
        missing = [key for key in required if key not in hdf5_file]
        if missing:
            raise KeyError(f"Missing required HDF5 datasets in {path}: {missing}")
        tags = hdf5_file["atom/tag"][()].astype(str)
        frame_indices = _resolve_suffix_indices(
            tags,
            start_tag=start_tag,
            allowed_tags=allowed_tags,
        )
        joint = hdf5_file["embodiment/joint"][frame_indices]
        ee_pose = hdf5_file["embodiment/ee"][frame_indices]
        usb_pose = hdf5_file["actor/prism"][frame_indices]
        slot_pose = hdf5_file["actor/slot"][frame_indices]
        save_frequency = int(hdf5_file["phase"].attrs.get("save_frequency", 1))

    states = build_privileged_states(
        joint=joint,
        ee_pose=ee_pose,
        usb_pose=usb_pose,
        slot_pose=slot_pose,
        sim_dt=float(sim_dt) * save_frequency,
    )
    joint_arm = np.asarray(joint[:, :7], dtype=np.float32)
    actions = np.stack(
        [
            target_qpos_to_action(
                joint_arm[index],
                joint_arm[index + 1],
                action_scale,
            )
            for index in range(max(0, len(joint_arm) - 1))
        ],
        axis=0,
    ) if len(joint_arm) > 1 else np.empty((0, 7), dtype=np.float32)
    rewards = np.zeros(actions.shape[0], dtype=np.float32)
    terminated = np.zeros(actions.shape[0], dtype=bool)
    if len(rewards) and episode_success:
        rewards[-1] = 1.0
        terminated[-1] = True
    return DemoTrajectory(
        demo_id=path.stem,
        path=path,
        frame_indices=frame_indices,
        states=states,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        tags=tuple(tags[frame_indices].tolist()),
        velocity_source="finite_difference_saved_qpos",
    )


def load_insert_usb_demos(
    data_dir: str | Path,
    *,
    action_scale: Sequence[float],
    pattern: str = "*.hdf5",
    max_demos: int | None = None,
    **kwargs,
) -> list[DemoTrajectory]:
    paths = sorted(Path(data_dir).glob(pattern), key=lambda path: int(path.stem))
    if max_demos is not None:
        if int(max_demos) <= 0:
            raise ValueError("max_demos must be positive when provided")
        paths = paths[: int(max_demos)]
    if not paths:
        raise FileNotFoundError(f"No HDF5 demos found under {data_dir!s}/{pattern}")
    return [
        load_insert_usb_demo(path, action_scale=action_scale, **kwargs)
        for path in paths
    ]


class ReverseCurriculum:
    """Per-demo reverse curriculum from RFCL Stage 1.

    The frontier is initialized at the terminal end of every demo.  A sampled
    checkpoint just after the frontier is easier; a successful rollout moves
    the frontier backwards by ``reverse_step_size``.  This is the only state
    machine needed by the learner, so it can be used with SAC, TD3, REDQ, or a
    custom off-policy implementation.
    """

    def __init__(
        self,
        demos: Sequence[DemoTrajectory],
        *,
        reverse_step_size: int = 2,
        geometric_p: float = 0.5,
        per_demo_buffer_size: int = 3,
        demo_horizon_to_max_steps_ratio: float = 1.25,
        minimum_episode_horizon: int = 16,
        seed: int = 0,
    ) -> None:
        if not demos:
            raise ValueError("At least one demo is required")
        if int(reverse_step_size) <= 0:
            raise ValueError("reverse_step_size must be positive")
        if not 0.0 < float(geometric_p) <= 1.0:
            raise ValueError("geometric_p must be in (0, 1]")
        if int(per_demo_buffer_size) <= 0:
            raise ValueError("per_demo_buffer_size must be positive")
        if float(demo_horizon_to_max_steps_ratio) <= 0.0:
            raise ValueError("demo_horizon_to_max_steps_ratio must be positive")
        if int(minimum_episode_horizon) <= 0:
            raise ValueError("minimum_episode_horizon must be positive")
        if any(demo.states.shape[0] <= 0 for demo in demos):
            raise ValueError("Every demo must contain at least one state")
        self.demos = tuple(demos)
        self.reverse_step_size = int(reverse_step_size)
        self.geometric_p = float(geometric_p)
        self.per_demo_buffer_size = int(per_demo_buffer_size)
        self.demo_horizon_to_max_steps_ratio = float(
            demo_horizon_to_max_steps_ratio
        )
        self.minimum_episode_horizon = int(minimum_episode_horizon)
        self.rng = np.random.default_rng(seed)
        self.frontiers = np.asarray(
            [demo.states.shape[0] - 1 for demo in self.demos],
            dtype=np.int64,
        )
        self.success_counts = np.zeros(len(self.demos), dtype=np.int64)
        self.frontier_results = [
            deque([False] * self.per_demo_buffer_size, maxlen=self.per_demo_buffer_size)
            for _ in self.demos
        ]
        self.solved = np.zeros(len(self.demos), dtype=bool)

    def _validate_demo_index(self, demo_index: int) -> int:
        demo_index = int(demo_index)
        if not 0 <= demo_index < len(self.demos):
            raise IndexError(f"demo_index out of range: {demo_index}")
        return demo_index

    def demo_probabilities(self) -> np.ndarray:
        """Return RFCL's frontier-weighted demo sampling distribution."""

        weights = np.asarray(
            [
                float(frontier) / float(demo.states.shape[0])
                if int(frontier) > 0
                else 1e-6
                for frontier, demo in zip(self.frontiers, self.demos)
            ],
            dtype=np.float64,
        )
        return weights / weights.sum()

    def checkpoint_distribution(
        self, demo_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the official five-point geometric reset distribution."""

        demo_index = self._validate_demo_index(demo_index)
        frontier = int(self.frontiers[demo_index])
        final_state = int(self.demos[demo_index].states.shape[0] - 1)
        state_indices = np.minimum(
            frontier + np.arange(5, dtype=np.int64),
            final_state,
        )
        failure_probability = 1.0 - self.geometric_p
        probabilities = np.asarray(
            [
                self.geometric_p * failure_probability**offset
                for offset in range(4)
            ]
            + [failure_probability**4],
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()
        return state_indices, probabilities

    def sample_checkpoint(self, demo_id: int | None = None) -> tuple[int, int]:
        """Return ``(demo_index, state_index)`` for a curriculum reset."""
        if demo_id is None:
            demo_index = int(
                self.rng.choice(len(self.demos), p=self.demo_probabilities())
            )
        else:
            demo_index = self._validate_demo_index(demo_id)
        state_indices, probabilities = self.checkpoint_distribution(demo_index)
        state_index = int(self.rng.choice(state_indices, p=probabilities))
        return demo_index, state_index

    def episode_horizon(self, demo_index: int, state_index: int) -> int:
        """Return RFCL's dynamic time limit for a sampled checkpoint."""

        demo_index = self._validate_demo_index(demo_index)
        state_index = int(state_index)
        state_count = int(self.demos[demo_index].states.shape[0])
        if not 0 <= state_index < state_count:
            raise IndexError(f"state_index out of range: {state_index}")
        remaining_demo_states = state_count - state_index
        return self.minimum_episode_horizon + int(
            remaining_demo_states // self.demo_horizon_to_max_steps_ratio
        )

    def progress(self) -> np.ndarray:
        """Return normalized reverse-curriculum progress for every demo.

        A value of zero is the terminal demonstration state where RFCL starts;
        one means that the frontier has reached state zero.
        """

        final_states = np.asarray(
            [demo.states.shape[0] - 1 for demo in self.demos],
            dtype=np.float64,
        )
        progress = np.ones(len(self.demos), dtype=np.float64)
        nonempty = final_states > 0
        progress[nonempty] = 1.0 - (
            self.frontiers[nonempty].astype(np.float64) / final_states[nonempty]
        )
        return np.clip(progress, 0.0, 1.0)

    def progress_status(
        self,
        *,
        target_progress: float,
        target_demo_fraction: float = 1.0,
    ) -> dict[str, object]:
        """Summarize whether enough demos reached a requested progress level."""

        target_progress = float(target_progress)
        target_demo_fraction = float(target_demo_fraction)
        if not 0.0 <= target_progress <= 1.0:
            raise ValueError("target_progress must be in [0, 1]")
        if not 0.0 < target_demo_fraction <= 1.0:
            raise ValueError("target_demo_fraction must be in (0, 1]")
        progress = self.progress()
        reached = progress >= target_progress
        required_demos = int(np.ceil(len(self.demos) * target_demo_fraction))
        reached_demos = int(reached.sum())
        return {
            "target_progress": target_progress,
            "target_demo_fraction": target_demo_fraction,
            "required_demos": required_demos,
            "reached_demos": reached_demos,
            "reached": reached.copy(),
            "progress": progress.copy(),
            "complete": reached_demos >= required_demos,
        }

    def record_result(
        self,
        demo_index: int,
        state_index: int,
        *,
        success: bool,
    ) -> int:
        demo_index = self._validate_demo_index(demo_index)
        state_index = int(state_index)
        if not 0 <= state_index < self.demos[demo_index].states.shape[0]:
            raise IndexError(f"state_index out of range: {state_index}")
        # Easier samples t_i+1 ... t_i+4 train the policy but must not advance
        # the frontier.  This matches RFCL's steps_back equality check.
        if state_index != int(self.frontiers[demo_index]):
            return int(self.frontiers[demo_index])

        success = bool(success)
        self.frontier_results[demo_index].append(success)
        if success:
            self.success_counts[demo_index] += 1
        if all(self.frontier_results[demo_index]):
            self.frontier_results[demo_index] = deque(
                [False] * self.per_demo_buffer_size,
                maxlen=self.per_demo_buffer_size,
            )
            if int(self.frontiers[demo_index]) > 0:
                self.frontiers[demo_index] = max(
                    0,
                    int(self.frontiers[demo_index]) - self.reverse_step_size,
                )
            else:
                self.solved[demo_index] = True
        return int(self.frontiers[demo_index])

    def state(self) -> dict[str, object]:
        return {
            "frontiers": self.frontiers.copy(),
            "success_counts": self.success_counts.copy(),
            "frontier_results": np.asarray(
                [list(results) for results in self.frontier_results],
                dtype=bool,
            ),
            "solved": self.solved.copy(),
            "reverse_step_size": self.reverse_step_size,
            "geometric_p": self.geometric_p,
            "per_demo_buffer_size": self.per_demo_buffer_size,
            "demo_horizon_to_max_steps_ratio": self.demo_horizon_to_max_steps_ratio,
            "minimum_episode_horizon": self.minimum_episode_horizon,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a curriculum checkpoint after validating the dataset/config."""

        expected = {
            "reverse_step_size": self.reverse_step_size,
            "geometric_p": self.geometric_p,
            "per_demo_buffer_size": self.per_demo_buffer_size,
            "demo_horizon_to_max_steps_ratio": self.demo_horizon_to_max_steps_ratio,
            "minimum_episode_horizon": self.minimum_episode_horizon,
        }
        for name, value in expected.items():
            if name not in state or state[name] != value:
                raise ValueError(
                    f"Curriculum {name} mismatch: checkpoint={state.get(name)!r}, "
                    f"current={value!r}"
                )
        demo_count = len(self.demos)
        frontiers = np.asarray(state["frontiers"], dtype=np.int64)
        success_counts = np.asarray(state["success_counts"], dtype=np.int64)
        frontier_results = np.asarray(state["frontier_results"], dtype=bool)
        solved = np.asarray(state["solved"], dtype=bool)
        if frontiers.shape != (demo_count,):
            raise ValueError("Curriculum frontier count does not match the demos")
        if success_counts.shape != (demo_count,) or np.any(success_counts < 0):
            raise ValueError("Invalid curriculum success_counts")
        if frontier_results.shape != (demo_count, self.per_demo_buffer_size):
            raise ValueError("Invalid curriculum frontier_results shape")
        if solved.shape != (demo_count,):
            raise ValueError("Invalid curriculum solved shape")
        final_states = np.asarray(
            [demo.states.shape[0] - 1 for demo in self.demos], dtype=np.int64
        )
        if np.any(frontiers < 0) or np.any(frontiers > final_states):
            raise ValueError("Curriculum frontiers are outside the current demos")
        self.frontiers[...] = frontiers
        self.success_counts[...] = success_counts
        self.frontier_results = [
            deque(row.tolist(), maxlen=self.per_demo_buffer_size)
            for row in frontier_results
        ]
        self.solved[...] = solved
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


class RoundRobinDemoScheduler:
    """Visit every unsolved demo in fixed-size blocks without starvation."""

    def __init__(self, demo_count: int, *, block_size: int = 1) -> None:
        if int(demo_count) <= 0:
            raise ValueError("demo_count must be positive")
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive")
        self.demo_count = int(demo_count)
        self.block_size = int(block_size)
        self.cursor = 0
        self.active_demo: int | None = None
        self.episodes_in_block = 0
        self.visit_counts = np.zeros(self.demo_count, dtype=np.int64)

    def select_demo(self, solved: Sequence[bool]) -> tuple[int, bool]:
        solved_array = np.asarray(solved, dtype=bool)
        if solved_array.shape != (self.demo_count,):
            raise ValueError("solved must contain one item per demo")
        if solved_array.all():
            raise StopIteration("all demos are solved")
        if (
            self.active_demo is not None
            and self.episodes_in_block < self.block_size
            and not solved_array[self.active_demo]
        ):
            return self.active_demo, False
        for offset in range(self.demo_count):
            candidate = (self.cursor + offset) % self.demo_count
            if not solved_array[candidate]:
                self.active_demo = candidate
                self.cursor = (candidate + 1) % self.demo_count
                self.episodes_in_block = 0
                return candidate, True
        raise StopIteration("all demos are solved")

    def record_episode(self, demo_index: int) -> None:
        demo_index = int(demo_index)
        if demo_index != self.active_demo:
            raise ValueError(
                f"Cannot record demo {demo_index}; active demo is {self.active_demo}"
            )
        self.episodes_in_block += 1
        self.visit_counts[demo_index] += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "demo_count": self.demo_count,
            "block_size": self.block_size,
            "cursor": self.cursor,
            "active_demo": self.active_demo,
            "episodes_in_block": self.episodes_in_block,
            "visit_counts": self.visit_counts.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["demo_count"]) != self.demo_count:
            raise ValueError("Scheduler demo_count does not match the dataset")
        if int(state["block_size"]) != self.block_size:
            raise ValueError("Scheduler block_size does not match the command line")
        cursor = int(state["cursor"])
        active_demo = state["active_demo"]
        active_demo = None if active_demo is None else int(active_demo)
        episodes_in_block = int(state["episodes_in_block"])
        visit_counts = np.asarray(state["visit_counts"], dtype=np.int64)
        if not 0 <= cursor < self.demo_count:
            raise ValueError("Invalid scheduler cursor")
        if active_demo is not None and not 0 <= active_demo < self.demo_count:
            raise ValueError("Invalid scheduler active_demo")
        if not 0 <= episodes_in_block <= self.block_size:
            raise ValueError("Invalid scheduler episodes_in_block")
        if visit_counts.shape != (self.demo_count,) or np.any(visit_counts < 0):
            raise ValueError("Invalid scheduler visit_counts")
        self.cursor = cursor
        self.active_demo = active_demo
        self.episodes_in_block = episodes_in_block
        self.visit_counts[...] = visit_counts


class MixedReplayBuffer:
    """Replay buffer with an explicit demo/online batch composition."""

    def __init__(self, capacity: int = 50_000, *, seed: int = 0) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        # Demonstrations are permanent.  Only online data is capacity-limited;
        # otherwise a long run eventually evicts every demo and silently loses
        # the requested 50/50 replay composition.
        self._demo_buffer: list[RFCLTransition] = []
        self._online_buffer: deque[RFCLTransition] = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, transition: RFCLTransition) -> None:
        if not isinstance(transition, RFCLTransition):
            raise TypeError("transition must be an RFCLTransition")
        if transition.source == "demo":
            self._demo_buffer.append(transition)
        else:
            self._online_buffer.append(transition)

    def extend(self, transitions: Iterable[RFCLTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def __len__(self) -> int:
        return len(self._demo_buffer) + len(self._online_buffer)

    def source_counts(self) -> dict[str, int]:
        return {
            "demo": len(self._demo_buffer),
            "online": len(self._online_buffer),
        }

    def sample(self, batch_size: int, *, demo_fraction: float = 0.5) -> list[RFCLTransition]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= float(demo_fraction) <= 1.0:
            raise ValueError("demo_fraction must be in [0, 1]")
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        demo_pool = self._demo_buffer
        online_pool = list(self._online_buffer)
        requested_demo = int(round(int(batch_size) * float(demo_fraction)))
        requested_online = int(batch_size) - requested_demo

        def draw(pool: list[RFCLTransition], count: int) -> list[RFCLTransition]:
            if count <= 0 or not pool:
                return []
            indices = self.rng.integers(0, len(pool), size=count)
            return [pool[int(index)] for index in indices]

        batch = draw(demo_pool, requested_demo) + draw(online_pool, requested_online)
        if len(batch) < int(batch_size):
            combined = self._demo_buffer + list(self._online_buffer)
            batch.extend(draw(combined, int(batch_size) - len(batch)))
        self.rng.shuffle(batch)
        return batch

    @staticmethod
    def _pack(transitions: Sequence[RFCLTransition]) -> dict[str, object]:
        if not transitions:
            return {"count": 0}
        return {
            "count": len(transitions),
            "states": np.stack([item.state for item in transitions]).astype(np.float32),
            "actions": np.stack([item.action for item in transitions]).astype(np.float32),
            "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32),
            "next_states": np.stack([item.next_state for item in transitions]).astype(np.float32),
            "terminated": np.asarray([item.terminated for item in transitions], dtype=bool),
            "demo_ids": [item.demo_id for item in transitions],
            "timesteps": np.asarray([item.timestep for item in transitions], dtype=np.int64),
            "sources": [item.source for item in transitions],
        }

    @staticmethod
    def _unpack(payload: dict[str, object]) -> list[RFCLTransition]:
        count = int(payload["count"])
        if count == 0:
            return []
        fields = (
            payload["states"], payload["actions"], payload["rewards"],
            payload["next_states"], payload["terminated"], payload["demo_ids"],
            payload["timesteps"], payload["sources"],
        )
        if any(len(field) != count for field in fields):
            raise ValueError("Replay checkpoint contains inconsistent field lengths")
        return [
            RFCLTransition(
                state=payload["states"][index],
                action=payload["actions"][index],
                reward=float(payload["rewards"][index]),
                next_state=payload["next_states"][index],
                terminated=bool(payload["terminated"][index]),
                demo_id=payload["demo_ids"][index],
                timestep=int(payload["timesteps"][index]),
                source=str(payload["sources"][index]),
            )
            for index in range(count)
        ]

    def state_dict(self) -> dict[str, object]:
        """Save online replay and RNG; demos are deterministically reloaded."""

        return {
            "schema": "rfcl_mixed_replay_v1",
            "capacity": self.capacity,
            "demo_count": len(self._demo_buffer),
            "online": self._pack(list(self._online_buffer)),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("schema") != "rfcl_mixed_replay_v1":
            raise ValueError(f"Unsupported replay schema: {state.get('schema')!r}")
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"Replay capacity mismatch: checkpoint={state['capacity']}, "
                f"current={self.capacity}"
            )
        if int(state["demo_count"]) != len(self._demo_buffer):
            raise ValueError("Replay demo count does not match the loaded dataset")
        online = self._unpack(state["online"])
        if any(item.source != "online" for item in online):
            raise ValueError("Replay online checkpoint contains non-online data")
        self._online_buffer.clear()
        self._online_buffer.extend(online)
        self.rng.bit_generator.state = state["rng_state"]
