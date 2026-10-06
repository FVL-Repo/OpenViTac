import numpy as np
import torch
from gymnasium import spaces
from torch import nn

import policy.RL.actor as actor_module
from policy.RL.actor import GaussianActor
from policy.RL.checkpoint import (
    BC_CHECKPOINT_FORMAT_VERSION,
    BC_CHECKPOINT_KIND,
    build_actor_checkpoint_payload,
)
from policy.RL.sb3_policy import BCGaussianSACPolicy


class StubEncoder(nn.Module):
    def __init__(
        self,
        qpos_dim=7,
        feature_dim=8,
        camera_keys=(),
        tactile_keys=(),
        visual_backbone="resnet18",
        visual_pretrained_path=None,
        freeze_visual_backbone=False,
        tactile_backbone="resnet18",
        tactile_normalization="group_norm",
        tactile_output_projection=False,
        freeze_tactile_backbone=False,
    ):
        super().__init__()
        self.qpos_dim = int(qpos_dim)
        self.feature_dim = int(feature_dim)
        self.camera_keys = tuple(camera_keys)
        self.tactile_keys = tuple(tactile_keys)
        self.visual_backbone = visual_backbone
        self.visual_pretrained_path = visual_pretrained_path
        self.freeze_visual_backbone = bool(freeze_visual_backbone)
        self.tactile_backbone = tactile_backbone
        self.tactile_normalization = tactile_normalization
        self.tactile_output_projection = bool(tactile_output_projection)
        self.freeze_tactile_backbone = bool(freeze_tactile_backbone)
        self.projection = nn.Linear(self.qpos_dim, self.feature_dim)

    def forward(self, observation):
        return self.projection(observation["qpos"].float())


def _save_bc_checkpoint(path, actor):
    torch.save(
        {
            "kind": BC_CHECKPOINT_KIND,
            "format_version": BC_CHECKPOINT_FORMAT_VERSION,
            "actor": build_actor_checkpoint_payload(actor),
            "action_statistics": {},
            "action_contract": {},
            "observation_contract": {},
            "data_split": {},
            "training_state": {},
        },
        path,
    )


def test_sac_actor_is_restored_bc_gaussian_actor(monkeypatch, tmp_path):
    monkeypatch.setattr(actor_module, "MultiModalEncoder", StubEncoder)
    torch.manual_seed(123)
    bc_actor = GaussianActor(
        hidden_dim=16,
        camera_keys=(),
        tactile_keys=(),
    ).eval()
    checkpoint_path = tmp_path / "bc.pt"
    _save_bc_checkpoint(checkpoint_path, bc_actor)

    observation_space = spaces.Dict(
        {
            "qpos": spaces.Box(
                -np.inf,
                np.inf,
                shape=(7,),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(7,),
        dtype=np.float32,
    )
    policy = BCGaussianSACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=[16],
        bc_checkpoint=checkpoint_path,
        freeze_actor_encoder=True,
    )

    observation = {"qpos": torch.randn(4, 7)}
    with torch.no_grad():
        expected_action = bc_actor.deterministic_action(observation)
        restored_action = policy.actor(observation, deterministic=True)

    torch.testing.assert_close(restored_action, expected_action)
    assert all(
        not parameter.requires_grad
        for parameter in policy.actor.gaussian_actor.encoder.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in policy.actor.gaussian_actor.trunk.parameters()
    )


def test_sac_actor_action_log_prob_is_finite(monkeypatch, tmp_path):
    monkeypatch.setattr(actor_module, "MultiModalEncoder", StubEncoder)
    actor = GaussianActor(
        hidden_dim=16,
        camera_keys=(),
        tactile_keys=(),
    )
    checkpoint_path = tmp_path / "bc.pt"
    _save_bc_checkpoint(checkpoint_path, actor)
    observation_space = spaces.Dict(
        {
            "qpos": spaces.Box(
                -np.inf,
                np.inf,
                shape=(7,),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(7,),
        dtype=np.float32,
    )

    policy = BCGaussianSACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=[16],
        bc_checkpoint=checkpoint_path,
    )

    action, log_probability = policy.actor.action_log_prob(
        {"qpos": torch.randn(3, 7)}
    )

    assert action.shape == (3, 7)
    assert log_probability.shape == (3,)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_probability).all()


def test_sac_actor_log_std_override_keeps_bc_mean(monkeypatch, tmp_path):
    monkeypatch.setattr(actor_module, "MultiModalEncoder", StubEncoder)
    torch.manual_seed(123)
    bc_actor = GaussianActor(
        hidden_dim=16,
        camera_keys=(),
        tactile_keys=(),
    ).eval()
    checkpoint_path = tmp_path / "bc.pt"
    _save_bc_checkpoint(checkpoint_path, bc_actor)
    observation_space = spaces.Dict(
        {
            "qpos": spaces.Box(
                -np.inf,
                np.inf,
                shape=(7,),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(7,),
        dtype=np.float32,
    )

    policy = BCGaussianSACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=[16],
        bc_checkpoint=checkpoint_path,
        bc_actor_log_std_override=-4.0,
    )

    observation = {"qpos": torch.randn(4, 7)}
    with torch.no_grad():
        expected_mu, _ = bc_actor(observation)
        restored_mu, restored_log_std = policy.actor.gaussian_actor(observation)
        expected_action = bc_actor.deterministic_action(observation)
        restored_action = policy.actor(observation, deterministic=True)

    torch.testing.assert_close(restored_mu, expected_mu)
    torch.testing.assert_close(restored_action, expected_action)
    torch.testing.assert_close(
        restored_log_std,
        torch.full((4, 7), -4.0),
    )
