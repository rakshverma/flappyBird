from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 5, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value
