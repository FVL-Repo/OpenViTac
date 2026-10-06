from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from policy.RL.encoders import MultiModalEncoder


class GaussianActor(nn.Module):
    """Single-step Gaussian actor shared by BC and SAC."""

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 256,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        qpos_dim: int = 7,
        camera_keys: Sequence[str] = (
            "cam_high",
            "cam_wrist",
        ),
        tactile_keys: Sequence[str] = (
            "tac_left",
            "tac_right",
        ),
        visual_backbone: str = "resnet18",
        visual_pretrained_path: str | Path | None = None,
        freeze_visual_backbone: bool = False,
        tactile_backbone: str = "resnet18",
        tactile_normalization: str = "group_norm",
        tactile_output_projection: bool = False,
        freeze_tactile_backbone: bool = False,
    ):
        super().__init__()

        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if log_std_min >= log_std_max:
            raise ValueError(
                "log_std_min must be smaller than log_std_max"
            )

        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.encoder = MultiModalEncoder(
            qpos_dim=qpos_dim,
            feature_dim=512,
            camera_keys=camera_keys,
            tactile_keys=tactile_keys,
            visual_backbone=visual_backbone,
            visual_pretrained_path=visual_pretrained_path,
            freeze_visual_backbone=freeze_visual_backbone,
            tactile_backbone=tactile_backbone,
            tactile_normalization=tactile_normalization,
            tactile_output_projection=(
                tactile_output_projection
            ),
            freeze_tactile_backbone=(
                freeze_tactile_backbone
            ),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.mu_head = nn.Linear(
            hidden_dim,
            self.action_dim,
        )
        self.log_std_head = nn.Linear(
            hidden_dim,
            self.action_dim,
        )

        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -2.0)

    def forward(
        self,
        observation: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encoder(observation)
        hidden = self.trunk(feature)

        mu = self.mu_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(
            log_std,
            min=self.log_std_min,
            max=self.log_std_max,
        )
        return mu, log_std

    def deterministic_action(
        self,
        observation: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mu, _ = self(observation)
        return torch.tanh(mu)

    def sample(
        self,
        observation: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(observation)
        std = torch.exp(log_std)

        distribution = Normal(mu, std)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)

        log_probability = distribution.log_prob(
            raw_action
        )

        log_squash_derivative = 2.0 * (
            math.log(2.0)
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )

        log_probability = (
            log_probability - log_squash_derivative
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return action, log_probability
