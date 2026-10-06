import pytest
import torch
from torch import nn

from policy.RL.bc import bc_update, compute_bc_loss


class StubActor(nn.Module):
    def __init__(self, predicted_action):
        super().__init__()
        self.predicted_action = nn.Parameter(
            predicted_action.clone()
        )

    def deterministic_action(self, observation):
        return self.predicted_action


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_compute_bc_loss_matches_manual_metrics():
    predicted_action = torch.tensor(
        [
            [0.2, -0.4, 0.6, -0.8, 1.0, 0.0, -0.2],
            [0.1, 0.3, -0.5, 0.7, -0.9, 0.2, 0.4],
        ],
        dtype=torch.float32,
    )
    target_action = torch.zeros_like(predicted_action)
    actor = StubActor(predicted_action)

    output = compute_bc_loss(
        actor=actor,
        observation={},
        target_action=target_action,
    )

    error = predicted_action - target_action
    torch.testing.assert_close(output.loss, error.square().mean())
    torch.testing.assert_close(output.action_mae, error.abs().mean())
    torch.testing.assert_close(
        output.per_joint_mae,
        error.abs().mean(dim=0),
    )
    assert output.per_joint_mae.shape == (7,)


def test_compute_bc_loss_is_zero_for_exact_prediction():
    target_action = torch.randn(3, 7)
    actor = StubActor(target_action)

    output = compute_bc_loss(actor, {}, target_action)

    torch.testing.assert_close(output.loss, torch.tensor(0.0))
    torch.testing.assert_close(
        output.action_mae,
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        output.per_joint_mae,
        torch.zeros(7),
    )


def test_compute_bc_loss_keeps_only_loss_gradient():
    predicted_action = torch.randn(2, 7)
    target_action = torch.zeros_like(predicted_action)
    actor = StubActor(predicted_action)

    output = compute_bc_loss(actor, {}, target_action)

    assert output.loss.requires_grad
    assert not output.action_mae.requires_grad
    assert not output.per_joint_mae.requires_grad

    output.loss.backward()

    assert actor.predicted_action.grad is not None
    assert torch.isfinite(actor.predicted_action.grad).all()


def test_compute_bc_loss_rejects_mismatched_shapes():
    actor = StubActor(torch.zeros(2, 7))
    target_action = torch.zeros(2, 6)

    with pytest.raises(ValueError, match="same shape"):
        compute_bc_loss(actor, {}, target_action)


def test_compute_bc_loss_rejects_integer_target():
    actor = StubActor(torch.zeros(2, 7))
    target_action = torch.zeros(2, 7, dtype=torch.int64)

    with pytest.raises(TypeError, match="floating point"):
        compute_bc_loss(actor, {}, target_action)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_compute_bc_loss_rejects_nonfinite_target(invalid_value):
    actor = StubActor(torch.zeros(2, 7))
    target_action = torch.zeros(2, 7)
    target_action[0, 0] = invalid_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        compute_bc_loss(actor, {}, target_action)


def test_bc_update_changes_actor_toward_target():
    actor = StubActor(torch.full((2, 7), 0.5))
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    target_action = torch.zeros(2, 7)
    before = actor.predicted_action.detach().clone()

    output = bc_update(
        actor=actor,
        optimizer=optimizer,
        observation={},
        target_action=target_action,
    )

    after = actor.predicted_action.detach()
    assert torch.linalg.vector_norm(after) < torch.linalg.vector_norm(before)
    assert actor.training
    assert not output.loss.requires_grad
    assert not output.action_mae.requires_grad
    assert not output.per_joint_mae.requires_grad
    assert not output.grad_norm.requires_grad
    assert torch.isfinite(output.grad_norm)


def test_bc_update_clips_gradient_norm():
    actor = StubActor(torch.ones(2, 7))
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    target_action = -torch.ones(2, 7)
    max_grad_norm = 0.01

    output = bc_update(
        actor=actor,
        optimizer=optimizer,
        observation={},
        target_action=target_action,
        max_grad_norm=max_grad_norm,
    )

    assert output.grad_norm > max_grad_norm
    assert actor.predicted_action.grad is not None
    clipped_grad_norm = torch.linalg.vector_norm(
        actor.predicted_action.grad
    ).item()
    assert clipped_grad_norm <= max_grad_norm + 1e-6


@pytest.mark.parametrize(
    "max_grad_norm",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_bc_update_rejects_invalid_max_grad_norm(max_grad_norm):
    actor = StubActor(torch.zeros(2, 7))
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="finite and positive"):
        bc_update(
            actor=actor,
            optimizer=optimizer,
            observation={},
            target_action=torch.zeros(2, 7),
            max_grad_norm=max_grad_norm,
        )


def test_bc_update_does_not_step_on_nonfinite_loss():
    actor = StubActor(torch.full((2, 7), float("inf")))
    optimizer = CountingSGD(actor.parameters(), lr=0.1)

    with pytest.raises(FloatingPointError, match="loss is not finite"):
        bc_update(
            actor=actor,
            optimizer=optimizer,
            observation={},
            target_action=torch.zeros(2, 7),
        )

    assert optimizer.step_calls == 0
