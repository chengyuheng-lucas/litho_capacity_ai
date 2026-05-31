from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelOutput:
    inv_mu: torch.Tensor
    inv_logvar: torch.Tensor
    ratio_mu: torch.Tensor
    ratio_logvar: torch.Tensor

    def confidence(self) -> torch.Tensor:
        inv_std = torch.exp(0.5 * self.inv_logvar.clamp(-8.0, 2.0))
        ratio_std = torch.exp(0.5 * self.ratio_logvar.clamp(-10.0, 0.0))
        score = 1.0 / (1.0 + inv_std + 10.0 * ratio_std)
        return score.clamp(0.0, 1.0)


class CapacityNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        max_ratio: float,
    ):
        super().__init__()
        self.max_ratio = float(max_ratio)

        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
        )

        self.inv_head = nn.Linear(hidden_size // 2, 2)
        self.ratio_head = nn.Linear(hidden_size // 2, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        h = self.ln(h)
        z = self.mlp(h)

        inv = self.inv_head(z)
        ratio = self.ratio_head(z)

        inv_mu_raw, inv_logvar = inv[:, :1], inv[:, 1:2]
        inv_mu = F.softplus(inv_mu_raw) + 1e-3

        ratio_mu_raw, ratio_logvar = ratio[:, :1], ratio[:, 1:2]
        ratio_mu = torch.tanh(ratio_mu_raw) * self.max_ratio

        inv_logvar = inv_logvar.clamp(-8.0, 2.0)
        ratio_logvar = ratio_logvar.clamp(-10.0, 0.0)

        return ModelOutput(inv_mu=inv_mu, inv_logvar=inv_logvar, ratio_mu=ratio_mu, ratio_logvar=ratio_logvar)


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar)
