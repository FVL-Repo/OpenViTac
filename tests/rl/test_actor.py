import pytest
import torch
from torch import nn

import policy.RL.actor as actor_module
from policy.RL.actor import GaussianActor


class StubEncoder(nn.Module):
    """Keep Actor unit tests independent from the ResNet computation."""

    def __init__(self, feature_dim=512, **kwargs):
        super().__init__()
        del kwargs
        self.feature_dim = feature_dim

    def forward(self, observation):
        return observation["feature"]


@pytest.fixture
def actor(monkeypatch):
    monkeypatch.setattr(actor_module, "MultiModalEncoder", StubEncoder)
    return GaussianActor()


def test_actor_forward_output_contract(actor):
    observation = {
        "feature": torch.randn(3, 512),
    }

    mu, log_std = actor(observation)

    assert mu.shape == (3, 7)
    assert log_std.shape == (3, 7)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_std).all()


def test_actor_initial_log_std_is_minus_two(actor):
    observation = {
        "feature": torch.randn(2, 512),
    }

    _, log_std = actor(observation)

    torch.testing.assert_close(
        log_std,
        torch.full((2, 7), -2.0),
    )


@pytest.mark.parametrize(
    ("bias", "expected"),
    [
        (100.0, 2.0),
        (-100.0, -5.0),
    ],
)
def test_actor_clamps_log_std(actor, bias, expected):
    nn.init.constant_(actor.log_std_head.bias, bias)
    observation = {
        "feature": torch.randn(2, 512),
    }

    _, log_std = actor(observation)

    torch.testing.assert_close(
        log_std,
        torch.full((2, 7), expected),
    )


def test_deterministic_action_is_tanh_of_mu_and_is_bounded(actor):
    observation = {
        "feature": torch.randn(3, 512),
    }

    mu, _ = actor(observation)
    action = actor.deterministic_action(observation)

    torch.testing.assert_close(action, torch.tanh(mu))
    assert action.shape == (3, 7)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)


def test_deterministic_action_keeps_bc_gradient_path(actor):
    feature = torch.randn(2, 512, requires_grad=True)
    observation = {
        "feature": feature,
    }

    action = actor.deterministic_action(observation)
    loss = action.square().mean()
    loss.backward()

    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert actor.mu_head.weight.grad is not None
    assert torch.isfinite(actor.mu_head.weight.grad).all()
    assert actor.log_std_head.weight.grad is None


def test_sample_output_contract(actor):
    observation = {
        "feature": torch.randn(4, 512),
    }

    action, log_probability = actor.sample(observation)

    assert action.shape == (4, 7)
    assert log_probability.shape == (4, 1)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_probability).all()


def test_sample_is_stochastic(actor):
    observation = {
        "feature": torch.randn(2, 512),
    }

    torch.manual_seed(0)
    first_action, _ = actor.sample(observation)
    torch.manual_seed(1)
    second_action, _ = actor.sample(observation)

    assert not torch.allclose(first_action, second_action)


def test_sample_keeps_reparameterized_gradient_path(actor):
    feature = torch.randn(2, 512, requires_grad=True)
    observation = {
        "feature": feature,
    }

    action, log_probability = actor.sample(observation)
    loss = action.square().mean() + log_probability.mean()
    loss.backward()

    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert actor.mu_head.weight.grad is not None
    assert torch.isfinite(actor.mu_head.weight.grad).all()
    assert actor.log_std_head.weight.grad is not None
    assert torch.isfinite(actor.log_std_head.weight.grad).all()


def test_sample_log_probability_is_finite_for_saturated_action(actor):
    nn.init.zeros_(actor.mu_head.weight)
    nn.init.constant_(actor.mu_head.bias, 100.0)
    nn.init.constant_(actor.log_std_head.bias, actor.log_std_max)
    observation = {
        "feature": torch.randn(2, 512),
    }

    action, log_probability = actor.sample(observation)

    torch.testing.assert_close(action, torch.ones_like(action))
    assert torch.isfinite(log_probability).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_dim": 0},
        {"hidden_dim": 0},
        {"log_std_min": 2.0, "log_std_max": 2.0},
        {"log_std_min": 3.0, "log_std_max": 2.0},
    ],
)
def test_actor_rejects_invalid_constructor_arguments(
    monkeypatch,
    kwargs,
):
    monkeypatch.setattr(actor_module, "MultiModalEncoder", StubEncoder)

    with pytest.raises(ValueError):
        GaussianActor(**kwargs)
