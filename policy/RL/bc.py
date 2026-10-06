from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn.functional as F

from policy.RL.actor import GaussianActor


@dataclass(frozen=True)
class BCLossOutput:
    loss: torch.Tensor
    action_mae: torch.Tensor
    per_joint_mae: torch.Tensor


@dataclass(frozen=True)
class BCUpdateOutput:
    loss: torch.Tensor
    action_mae: torch.Tensor
    per_joint_mae: torch.Tensor
    grad_norm: torch.Tensor

def compute_bc_loss(
    actor: GaussianActor,
    observation: dict[str, torch.Tensor],
    target_action: torch.Tensor,
) -> BCLossOutput:
    predicted_action = actor.deterministic_action(
        observation
    )

    if predicted_action.shape != target_action.shape:
        raise ValueError(
            "predicted_action and target_action must have "
            f"the same shape, got {tuple(predicted_action.shape)} "
            f"and {tuple(target_action.shape)}"
        )

    if not torch.is_floating_point(target_action):
        raise TypeError("target_action must be floating point")

    if not torch.isfinite(target_action).all():
        raise ValueError(
            "target_action contains NaN or infinite values"
        )

    action_error = predicted_action - target_action
    loss = F.mse_loss(
        predicted_action,
        target_action,
    )

    absolute_error = action_error.detach().abs()

    return BCLossOutput(
        loss=loss,
        action_mae=absolute_error.mean(),
        per_joint_mae=absolute_error.mean(dim=0),
    )

def bc_update(
    actor: GaussianActor,
    optimizer: torch.optim.Optimizer,
    observation: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    max_grad_norm: float = 1.0,
) -> BCUpdateOutput:
    max_grad_norm = float(max_grad_norm)

    if (
        not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0.0
    ):
        raise ValueError(
            "max_grad_norm must be finite and positive"
        )

    actor.train()
    optimizer.zero_grad(set_to_none=True)

    loss_output = compute_bc_loss(
        actor=actor,
        observation=observation,
        target_action=target_action,
    )

    if not torch.isfinite(loss_output.loss):
        raise FloatingPointError("BC loss is not finite")

    loss_output.loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(),
        max_norm=max_grad_norm,
    )

    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            "BC gradient norm is not finite"
        )

    optimizer.step()

    return BCUpdateOutput(
        loss=loss_output.loss.detach(),
        action_mae=loss_output.action_mae,
        per_joint_mae=loss_output.per_joint_mae,
        grad_norm=grad_norm.detach(),
    )