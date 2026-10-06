from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


ARM_ACTION_DIM = 7
TARGET_QPOS_VELOCITY_ACTION_DIM = 2 * ARM_ACTION_DIM
TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM = TARGET_QPOS_VELOCITY_ACTION_DIM + 1
ACTION_LOW = -1.0
ACTION_HIGH = 1.0


def _as_arm_array(value: ArrayLike, *, name: str) -> NDArray[np.float32]:
    """Convert input to float32 and validate its final arm dimension."""
    array = np.asarray(value, dtype=np.float32)

    if array.ndim == 0 or array.shape[-1] != ARM_ACTION_DIM:
        raise ValueError(
            f"{name} must have trailing dimension {ARM_ACTION_DIM}, "
            f"got shape {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")

    return array


def _as_action_scale(action_scale: ArrayLike) -> NDArray[np.float32]:
    """Validate the per-joint scale used to normalize delta qpos."""
    scale = _as_arm_array(action_scale, name="action_scale")

    if scale.shape != (ARM_ACTION_DIM,):
        raise ValueError(
            f"action_scale must have shape ({ARM_ACTION_DIM},), "
            f"got {scale.shape}"
        )

    if np.any(scale <= 0.0):
        raise ValueError("action_scale must contain only positive values")

    return scale


def _require_same_shape(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    *,
    first_name: str,
    second_name: str,
) -> None:
    if first.shape != second.shape:
        raise ValueError(
            f"{first_name} and {second_name} must have the same shape, "
            f"got {first.shape} and {second.shape}"
        )


def clip_action(action: ArrayLike) -> NDArray[np.float32]:
    """Clip a normalized 7-DoF arm action into [-1, 1]."""
    action_array = _as_arm_array(action, name="action")
    return np.clip(
        action_array,
        ACTION_LOW,
        ACTION_HIGH,
    ).astype(np.float32, copy=False)


def target_qpos_to_action(
    current_qpos: ArrayLike,
    target_qpos: ArrayLike,
    action_scale: ArrayLike,
) -> NDArray[np.float32]:
    """Encode an absolute arm target as a normalized delta action.

    Formula:
        action = clip(
            (target_qpos - current_qpos) / action_scale,
            -1,
            1,
        )
    """
    current = _as_arm_array(current_qpos, name="current_qpos")
    target = _as_arm_array(target_qpos, name="target_qpos")
    scale = _as_action_scale(action_scale)

    _require_same_shape(
        current,
        target,
        first_name="current_qpos",
        second_name="target_qpos",
    )

    normalized_delta = (target - current) / scale
    return clip_action(normalized_delta)


def action_to_target_qpos(
    current_qpos: ArrayLike,
    action: ArrayLike,
    action_scale: ArrayLike,
) -> NDArray[np.float32]:
    """Decode a normalized delta action into an absolute arm target.

    Formula:
        target_qpos = current_qpos + action_scale * clip(action, -1, 1)
    """
    current = _as_arm_array(current_qpos, name="current_qpos")
    normalized_action = _as_arm_array(action, name="action")
    scale = _as_action_scale(action_scale)

    _require_same_shape(
        current,
        normalized_action,
        first_name="current_qpos",
        second_name="action",
    )

    clipped_action = clip_action(normalized_action)
    target = current + scale * clipped_action
    return target.astype(np.float32, copy=False)


def target_qpos_velocity_to_action(
    current_qpos: ArrayLike,
    target_qpos: ArrayLike,
    target_velocity: ArrayLike,
    qpos_scale: ArrayLike,
    velocity_scale: ArrayLike,
) -> NDArray[np.float32]:
    """Encode a position/velocity target as one normalized action.

    The existing Motion Plan controller supplies both targets to PhysX.  A
    position-only action cannot reproduce its contact trajectory, so RFCL can
    use this 14-D representation while retaining the same normalized action
    convention as the BC policy.
    """
    current = _as_arm_array(current_qpos, name="current_qpos")
    target = _as_arm_array(target_qpos, name="target_qpos")
    velocity = _as_arm_array(target_velocity, name="target_velocity")
    qpos = _as_action_scale(qpos_scale)
    vel = _as_action_scale(velocity_scale)
    _require_same_shape(current, target, first_name="current_qpos", second_name="target_qpos")
    _require_same_shape(current, velocity, first_name="current_qpos", second_name="target_velocity")
    normalized = np.concatenate(
        ((target - current) / qpos, velocity / vel), axis=-1
    )
    return np.clip(normalized, ACTION_LOW, ACTION_HIGH).astype(np.float32, copy=False)


def action_to_target_qpos_velocity(
    current_qpos: ArrayLike,
    action: ArrayLike,
    qpos_scale: ArrayLike,
    velocity_scale: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Decode a normalized 14-D position/velocity target action."""
    current = _as_arm_array(current_qpos, name="current_qpos")
    normalized = np.asarray(action, dtype=np.float32)
    if normalized.ndim == 0 or normalized.shape[-1] != TARGET_QPOS_VELOCITY_ACTION_DIM:
        raise ValueError(
            "action must have trailing dimension "
            f"{TARGET_QPOS_VELOCITY_ACTION_DIM}, got shape {normalized.shape}"
        )
    if not np.isfinite(normalized).all():
        raise ValueError("action contains NaN or infinite values")
    if current.shape != normalized.shape[:-1] + (ARM_ACTION_DIM,):
        raise ValueError(
            "current_qpos must match action batch shape, got "
            f"{current.shape} and {normalized.shape}"
        )
    qpos = _as_action_scale(qpos_scale)
    vel = _as_action_scale(velocity_scale)
    clipped = np.clip(normalized, ACTION_LOW, ACTION_HIGH)
    target = current + clipped[..., :ARM_ACTION_DIM] * qpos
    velocity = clipped[..., ARM_ACTION_DIM:] * vel
    return target.astype(np.float32, copy=False), velocity.astype(np.float32, copy=False)


def target_qpos_velocity_force_to_action(
    current_qpos: ArrayLike,
    target_qpos: ArrayLike,
    target_velocity: ArrayLike,
    force_position_write: bool,
    qpos_scale: ArrayLike,
    velocity_scale: ArrayLike,
) -> NDArray[np.float32]:
    """Encode a Motion Plan command, including its position-write phase."""
    action = target_qpos_velocity_to_action(
        current_qpos,
        target_qpos,
        target_velocity,
        qpos_scale,
        velocity_scale,
    )
    flag = np.ones((*action.shape[:-1], 1), dtype=np.float32)
    if not bool(force_position_write):
        flag[...] = -1.0
    return np.concatenate((action, flag), axis=-1).astype(np.float32, copy=False)


def action_to_target_qpos_velocity_force(
    current_qpos: ArrayLike,
    action: ArrayLike,
    qpos_scale: ArrayLike,
    velocity_scale: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32], bool]:
    """Decode a Motion Plan command and its position-write phase."""
    normalized = np.asarray(action, dtype=np.float32)
    if normalized.ndim == 0 or normalized.shape[-1] != TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM:
        raise ValueError(
            "action must have trailing dimension "
            f"{TARGET_QPOS_VELOCITY_FORCE_ACTION_DIM}, got shape {normalized.shape}"
        )
    if not np.isfinite(normalized).all():
        raise ValueError("action contains NaN or infinite values")
    target, velocity = action_to_target_qpos_velocity(
        current_qpos,
        normalized[..., :TARGET_QPOS_VELOCITY_ACTION_DIM],
        qpos_scale,
        velocity_scale,
    )
    return target, velocity, bool(float(normalized[..., -1].reshape(-1)[0]) >= 0.0)
