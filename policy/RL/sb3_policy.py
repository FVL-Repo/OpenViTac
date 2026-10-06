from __future__ import annotations

from pathlib import Path
from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
)
from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy

from policy.RL.checkpoint import load_bc_checkpoint, restore_actor_from_bc_checkpoint


class BCGaussianSACActor(BasePolicy):
    """SAC actor that is exactly the BC GaussianActor architecture."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        *,
        bc_checkpoint: str | Path,
        freeze_encoder: bool = True,
        normalize_images: bool = True,
        bc_actor_log_std_override: float | None = None,
        use_sde: bool = False,
        **_: Any,
    ):
        if use_sde:
            raise ValueError("BCGaussianSACActor does not support gSDE")
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            squash_output=True,
            normalize_images=normalize_images,
        )

        checkpoint_path = Path(bc_checkpoint).expanduser().resolve()
        checkpoint = load_bc_checkpoint(checkpoint_path, map_location="cpu")
        self.gaussian_actor = restore_actor_from_bc_checkpoint(
            checkpoint,
            device="cpu",
        )
        self.bc_checkpoint = str(checkpoint_path)
        self.freeze_encoder = bool(freeze_encoder)
        self.bc_actor_log_std_override = (
            None
            if bc_actor_log_std_override is None
            else float(bc_actor_log_std_override)
        )
        self.action_dist = SquashedDiagGaussianDistribution(
            self.gaussian_actor.action_dim,
        )

        action_dim = get_action_dim(self.action_space)
        if action_dim != self.gaussian_actor.action_dim:
            raise ValueError(
                "Action space and BC actor dimensions differ: "
                f"{action_dim} != {self.gaussian_actor.action_dim}"
            )

        if self.freeze_encoder:
            for parameter in self.gaussian_actor.encoder.parameters():
                parameter.requires_grad_(False)
        if self.bc_actor_log_std_override is not None:
            if not (
                self.gaussian_actor.log_std_min
                <= self.bc_actor_log_std_override
                <= self.gaussian_actor.log_std_max
            ):
                raise ValueError(
                    "bc_actor_log_std_override must be within the restored "
                    "actor clamp range "
                    f"[{self.gaussian_actor.log_std_min}, "
                    f"{self.gaussian_actor.log_std_max}], got "
                    f"{self.bc_actor_log_std_override}"
                )
            with th.no_grad():
                self.gaussian_actor.log_std_head.weight.zero_()
                self.gaussian_actor.log_std_head.bias.fill_(
                    self.bc_actor_log_std_override
                )

    def _preprocess_observation(
        self,
        observation: dict[str, th.Tensor],
    ) -> dict[str, th.Tensor]:
        preprocessed = preprocess_obs(
            observation,
            self.observation_space,
            normalize_images=self.normalize_images,
        )
        if not isinstance(preprocessed, dict):
            raise TypeError("BCGaussianSACActor requires dict observations")
        return preprocessed

    def get_action_dist_params(
        self,
        observation: dict[str, th.Tensor],
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        processed_observation = self._preprocess_observation(observation)
        mean_actions, log_std = self.gaussian_actor(processed_observation)
        return mean_actions, log_std, {}

    def forward(
        self,
        observation: dict[str, th.Tensor],
        deterministic: bool = False,
    ) -> th.Tensor:
        mean_actions, log_std, _ = self.get_action_dist_params(observation)
        return self.action_dist.actions_from_params(
            mean_actions,
            log_std,
            deterministic=deterministic,
        )

    def action_log_prob(
        self,
        observation: dict[str, th.Tensor],
    ) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, _ = self.get_action_dist_params(observation)
        return self.action_dist.log_prob_from_params(mean_actions, log_std)

    def _predict(
        self,
        observation: dict[str, th.Tensor],
        deterministic: bool = False,
    ) -> th.Tensor:
        return self(observation, deterministic=deterministic)

    def reset_noise(self, batch_size: int = 1) -> None:
        del batch_size
        raise RuntimeError("BCGaussianSACActor does not support gSDE")

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "bc_checkpoint": self.bc_checkpoint,
                "freeze_encoder": self.freeze_encoder,
                "bc_actor_log_std_override": self.bc_actor_log_std_override,
            }
        )
        return data


class BCGaussianSACPolicy(SACMultiInputPolicy):
    """SAC policy whose actor is initialized from the BC checkpoint."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[th.nn.Module] = th.nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        bc_checkpoint: str | Path = "",
        freeze_actor_encoder: bool = True,
        bc_actor_log_std_override: float | None = None,
    ):
        if not bc_checkpoint:
            raise ValueError("bc_checkpoint is required for BCGaussianSACPolicy")
        self.bc_checkpoint = str(Path(bc_checkpoint).expanduser().resolve())
        self.freeze_actor_encoder = bool(freeze_actor_encoder)
        self.bc_actor_log_std_override = (
            None
            if bc_actor_log_std_override is None
            else float(bc_actor_log_std_override)
        )

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> BCGaussianSACActor:
        del features_extractor
        actor_kwargs = dict(self.actor_kwargs)
        for key in (
            "observation_space",
            "action_space",
            "normalize_images",
        ):
            actor_kwargs.pop(key, None)
        return BCGaussianSACActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            bc_checkpoint=self.bc_checkpoint,
            freeze_encoder=self.freeze_actor_encoder,
            bc_actor_log_std_override=self.bc_actor_log_std_override,
            normalize_images=self.normalize_images,
            **actor_kwargs,
        ).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "bc_checkpoint": self.bc_checkpoint,
                "freeze_actor_encoder": self.freeze_actor_encoder,
                "bc_actor_log_std_override": self.bc_actor_log_std_override,
            }
        )
        return data
