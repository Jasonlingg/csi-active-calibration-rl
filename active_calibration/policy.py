"""Small CPU actor/critic for choosing which target example to label.

The same candidate encoder and actor are applied to every row. Global context
is computed using masked means, so the network does not learn a pool row's
position or a receiver identifier. Hidden reward-set data is never an input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PolicyConfig:
    feature_dim: int = 27
    n_classes: int = 8
    hidden_dim: int = 64


class SetActorCritic(nn.Module):
    """Permutation-equivariant candidate logits and an invariant value head."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
        if min(self.config.feature_dim, self.config.hidden_dim) < 1:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if self.config.n_classes < 2:
            raise ValueError("n_classes must be at least two")
        hidden = self.config.hidden_dim
        input_dim = self.config.feature_dim + 2 * self.config.n_classes + 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(4 * hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(3 * hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(values.dtype).unsqueeze(1)
        return (values * weight).sum(0) / weight.sum().clamp_min(1.0)

    def forward(self, observation: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits ``(pool_size,)`` and a scalar state value."""
        device = next(self.parameters()).device
        features = torch.as_tensor(observation["features"], dtype=torch.float32, device=device)
        scores = torch.as_tensor(observation["scores"], dtype=torch.float32, device=device)
        labels = torch.as_tensor(observation["labels"], dtype=torch.long, device=device)
        queried = torch.as_tensor(observation["queried_mask"], dtype=torch.bool, device=device)
        available = torch.as_tensor(observation["action_mask"], dtype=torch.bool, device=device)
        budget = torch.as_tensor(observation["budget"], dtype=torch.float32, device=device).reshape(-1)

        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("observation features have the wrong shape")
        count = features.shape[0]
        if scores.shape != (count, self.config.n_classes):
            raise ValueError("observation scores have the wrong shape")
        if any(tensor.shape != (count,) for tensor in (labels, queried, available)):
            raise ValueError("labels and masks must have one entry per candidate")
        if budget.numel() != 1:
            raise ValueError("budget must contain one scalar")
        if not bool(available.any()):
            raise ValueError("the observation has no legal query action")
        if bool((queried & available).any()):
            raise ValueError("queried candidates cannot remain legal actions")
        if bool(((labels[queried] < 0) | (labels[queried] >= self.config.n_classes)).any()):
            raise ValueError("queried candidates need a valid observed label")
        if bool((labels[~queried] != -1).any()):
            raise ValueError("unqueried labels must remain hidden as -1")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in (features, scores, budget)):
            raise ValueError("policy inputs must be finite")

        onehot = F.one_hot(labels.clamp_min(0), self.config.n_classes).to(features.dtype)
        onehot = onehot * queried.unsqueeze(1)
        encoded = self.encoder(torch.cat([
            features, scores, onehot,
            queried.to(features.dtype).unsqueeze(1),
            available.to(features.dtype).unsqueeze(1),
        ], dim=1))
        # queried | available excludes padding if an environment pads the pool.
        context = torch.cat([
            self._masked_mean(encoded, queried | available),
            self._masked_mean(encoded, queried),
            self._masked_mean(encoded, available),
            budget,
        ])
        actor_input = torch.cat([encoded, context.unsqueeze(0).expand(count, -1)], dim=1)
        logits = self.actor(actor_input).squeeze(1).masked_fill(~available, -torch.inf)
        return logits, self.critic(context).squeeze(0)

    def checkpoint(self) -> dict[str, Any]:
        """Return tensor/primitive-only checkpoint data for safe Torch loading."""
        return {
            "format_version": 1,
            "model_config": asdict(self.config),
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
        }


class NeuralPolicy:
    """Load a local checkpoint and choose a deterministic legal action."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported policy checkpoint version")
        self.model = SetActorCritic(PolicyConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def __call__(self, observation: Mapping[str, np.ndarray]) -> int:
        with torch.no_grad():
            logits, _ = self.model(observation)
        return int(logits.argmax().item())
