from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from policy.RL.checkpoint import load_bc_checkpoint, restore_actor_from_bc_checkpoint


class BCFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        bc_checkpoint: str | Path,
        freeze: bool = True,
    ):
        checkpoint = load_bc_checkpoint(bc_checkpoint, map_location="cpu")
        actor = restore_actor_from_bc_checkpoint(checkpoint, device="cpu")
        super().__init__(
            observation_space,
            features_dim=int(actor.encoder.feature_dim),
        )
        self.encoder = actor.encoder
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            return self.encoder(observations)
        with torch.no_grad():
            return self.encoder(observations)
