"""Small PyTorch SAC learner used by the privileged RFCL pilot.

This is deliberately self-contained.  It keeps the upstream RFCL choices
visible in one place: replay always contains the demonstrations, online data
is mixed at a configured ratio, and the critic uses a clipped Q ensemble.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from policy.RL.rfcl import MixedReplayBuffer, RFCLTransition


def _mlp(
    input_dim: int, output_dim: int, *, layer_norm: bool = False
) -> nn.Sequential:
    layers: list[nn.Module] = []
    hidden_dim = 256
    for index in range(3):
        layers.append(nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, initial_log_std: float) -> None:
        super().__init__()
        # RFCL's actor is a plain MLP; normalization is kept in the critic.
        self.net = _mlp(state_dim, 2 * action_dim, layer_norm=False)
        self.action_dim = int(action_dim)
        self.initial_log_std = float(initial_log_std)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std + self.initial_log_std, -5.0, 1.0)
        return mean, log_std

    def sample(
        self, state: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + std * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_prob = torch.zeros(
                (state.shape[0],), dtype=state.dtype, device=state.device
            )
        else:
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(pre_tanh).sum(-1)
            log_prob -= torch.log(1.0 - action.square() + 1e-6).sum(-1)
        return action, log_prob


class QEnsemble(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_qs: int) -> None:
        super().__init__()
        self.qs = nn.ModuleList(
            [
                _mlp(state_dim + action_dim, 1, layer_norm=True)
                for _ in range(int(num_qs))
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((state, action), dim=-1)
        return torch.cat([q(inputs) for q in self.qs], dim=-1)


@dataclass
class SACMetrics:
    critic_loss: float
    actor_loss: float
    alpha: float
    q_mean: float


class RFCLSACTrainer:
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int = 7,
        device: str = "cuda",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.02,
        auto_alpha: bool = False,
        backup_entropy: bool = False,
        target_entropy: float | None = None,
        num_qs: int = 2,
        num_min_qs: int | None = None,
        initial_log_std: float = -3.0,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        if int(num_qs) <= 0:
            raise ValueError("num_qs must be positive")
        if num_min_qs is None:
            num_min_qs = int(num_qs)
        if int(num_min_qs) <= 0 or int(num_min_qs) > int(num_qs):
            raise ValueError("num_min_qs must be in [1, num_qs]")
        self.num_qs = int(num_qs)
        self.num_min_qs = int(num_min_qs)
        self.backup_entropy = bool(backup_entropy)
        self.initial_log_std = float(initial_log_std)
        self.target_entropy = float(
            -action_dim if target_entropy is None else target_entropy
        )
        self.actor = GaussianPolicy(
            self.state_dim, self.action_dim, initial_log_std
        ).to(self.device)
        self.critic = QEnsemble(
            self.state_dim, self.action_dim, num_qs
        ).to(self.device)
        self.target_critic = QEnsemble(
            self.state_dim, self.action_dim, num_qs
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(learning_rate)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(learning_rate)
        )
        self.auto_alpha = bool(auto_alpha)
        if self.auto_alpha:
            self.log_alpha = nn.Parameter(
                torch.tensor(np.log(max(float(alpha), 1e-6)), device=self.device)
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(learning_rate))
        else:
            self.log_alpha = torch.tensor(
                np.log(max(float(alpha), 1e-6)), device=self.device
            )
            self.alpha_optimizer = None
        self.state_mean = torch.as_tensor(
            np.zeros(self.state_dim, dtype=np.float32)
            if state_mean is None
            else np.asarray(state_mean, dtype=np.float32),
            device=self.device,
        )
        self.state_std = torch.as_tensor(
            np.ones(self.state_dim, dtype=np.float32)
            if state_std is None
            else np.maximum(np.asarray(state_std, dtype=np.float32), 1e-4),
            device=self.device,
        )
        self.update_count = 0
        self.last_batch_source_counts = {"demo": 0, "online": 0}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _states(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return (tensor - self.state_mean) / self.state_std

    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actor.sample(
                self._states(np.asarray(state)[None, :]),
                deterministic=deterministic,
            )
        return action[0].cpu().numpy().astype(np.float32)

    def update(
        self,
        replay: MixedReplayBuffer,
        *,
        batch_size: int = 256,
        demo_fraction: float = 0.5,
        update_actor: bool = True,
    ) -> SACMetrics:
        batch = replay.sample(batch_size, demo_fraction=demo_fraction)
        self.last_batch_source_counts = {
            "demo": sum(item.source == "demo" for item in batch),
            "online": sum(item.source == "online" for item in batch),
        }
        states = self._states(np.stack([x.state for x in batch]))
        actions = torch.as_tensor(
            np.stack([x.action for x in batch]), dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            np.asarray([x.reward for x in batch], dtype=np.float32),
            device=self.device,
        )
        next_states = self._states(np.stack([x.next_state for x in batch]))
        terminated = torch.as_tensor(
            np.asarray([x.terminated for x in batch], dtype=np.float32),
            device=self.device,
        )
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            next_qs = self.target_critic(next_states, next_action)
            if self.num_min_qs < self.num_qs:
                indices = torch.randperm(self.num_qs, device=self.device)[: self.num_min_qs]
                next_qs = next_qs.index_select(1, indices)
            next_q = next_qs.min(dim=1).values
            target = rewards + self.gamma * (1.0 - terminated) * next_q
            if self.backup_entropy:
                target -= (
                    self.gamma
                    * (1.0 - terminated)
                    * self.alpha.detach()
                    * next_log_prob
                )
        q_values = self.critic(states, actions)
        critic_loss = 0.5 * ((q_values - target[:, None]) ** 2).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        if update_actor:
            policy_action, log_prob = self.actor.sample(states)
            # The upstream RFCL SAC actor uses the ensemble mean; only the
            # target backup is clipped to a sampled min-Q subset.
            policy_q = self.critic(states, policy_action).mean(dim=1)
            actor_loss = (self.alpha.detach() * log_prob - policy_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            if self.auto_alpha:
                alpha_loss = -(
                    self.log_alpha
                    * (log_prob.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()

        with torch.no_grad():
            for target, source in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target.mul_(1.0 - self.tau).add_(self.tau * source)
        self.update_count += 1
        return SACMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=actor_loss_value,
            alpha=float(self.alpha.detach().cpu()),
            q_mean=float(q_values.detach().mean().cpu()),
        )

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "rfcl_sac_checkpoint_v2",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "auto_alpha": self.auto_alpha,
            "backup_entropy": self.backup_entropy,
            "num_qs": self.num_qs,
            "num_min_qs": self.num_min_qs,
            "target_entropy": self.target_entropy,
            "initial_log_std": self.initial_log_std,
            "update_count": self.update_count,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "state_mean": self.state_mean.detach().cpu(),
            "state_std": self.state_std.detach().cpu(),
            "extra": extra or {},
        }
        if self.alpha_optimizer is not None:
            payload["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Restore all SAC parameters/optimizers and return runner metadata."""

        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") not in (
            "rfcl_sac_checkpoint_v1",
            "rfcl_sac_checkpoint_v2",
        ):
            raise ValueError(f"Unsupported RFCL checkpoint: {payload.get('schema')!r}")
        expected = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_qs": self.num_qs,
            "num_min_qs": self.num_min_qs,
            "auto_alpha": self.auto_alpha,
            "backup_entropy": self.backup_entropy,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(
                    f"Checkpoint {name} mismatch: checkpoint={payload.get(name)!r}, "
                    f"current={value!r}"
                )
        float_expected = {
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "initial_log_std": self.initial_log_std,
        }
        for name, value in float_expected.items():
            if name not in payload or not np.isclose(float(payload[name]), value):
                raise ValueError(
                    f"Checkpoint {name} mismatch: checkpoint={payload.get(name)!r}, "
                    f"current={value!r}"
                )
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
            self.state_mean.copy_(payload["state_mean"].to(self.device))
            self.state_std.copy_(payload["state_std"].to(self.device))
        if self.alpha_optimizer is not None:
            if "alpha_optimizer" not in payload:
                raise ValueError("Auto-alpha checkpoint is missing alpha_optimizer")
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.update_count = int(payload["update_count"])
        return payload


def demo_state_statistics(demos: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    states = np.concatenate([np.asarray(demo.states, dtype=np.float32) for demo in demos])
    return states.mean(axis=0).astype(np.float32), np.maximum(
        states.std(axis=0), 1e-4
    ).astype(np.float32)


def add_demo_transitions(
    replay: MixedReplayBuffer, demos: list[Any]
) -> None:
    for demo in demos:
        replay.extend(demo.transitions())
