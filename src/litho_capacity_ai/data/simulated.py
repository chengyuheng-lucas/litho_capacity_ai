from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from litho_capacity_ai.config import TrainConfig


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    eps = rng.normal(0.0, sigma, size=n).astype(np.float32)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


@dataclass(frozen=True)
class SimulatedBatch:
    x: np.ndarray
    y_inv_weeks: np.ndarray
    y_adj_ratio: np.ndarray


def make_simulated(cfg: TrainConfig, n_samples: int, seed_offset: int = 0) -> SimulatedBatch:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    total_len = cfg.lookback_weeks + cfg.horizon_weeks

    x = np.zeros((n_samples, cfg.lookback_weeks, cfg.num_features), dtype=np.float32)
    y_inv = np.zeros((n_samples,), dtype=np.float32)
    y_ratio = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        demand = 1.0 + 0.15 * _ar1(rng, total_len, phi=0.92, sigma=0.4)
        tight = 0.0 + 0.20 * _ar1(rng, total_len, phi=0.88, sigma=0.5)
        macro = 0.0 + 0.10 * _ar1(rng, total_len, phi=0.97, sigma=0.2)

        price = 1.0 + 0.6 * demand + 0.5 * tight + 0.2 * macro + rng.normal(0.0, 0.15, size=total_len)
        price = price.astype(np.float32)

        ret = np.diff(price, prepend=price[0])
        vol = np.sqrt(_rolling_mean(ret * ret, window=8))
        lead = 8.0 + 6.0 * np.maximum(0.0, tight) + 2.0 * vol + rng.normal(0.0, 0.4, size=total_len)
        lead = lead.astype(np.float32)

        inv_proxy = 10.0 + 3.0 * np.maximum(0.0, -demand) + rng.normal(0.0, 0.6, size=total_len)
        inv_proxy = inv_proxy.astype(np.float32)

        feats = np.stack(
            [
                demand,
                tight,
                macro,
                price,
                ret.astype(np.float32),
                vol.astype(np.float32),
                lead,
                inv_proxy,
            ],
            axis=-1,
        ).astype(np.float32)

        if cfg.num_features > feats.shape[-1]:
            extra = rng.normal(0.0, 1.0, size=(total_len, cfg.num_features - feats.shape[-1])).astype(np.float32)
            feats = np.concatenate([feats, extra], axis=-1)
        else:
            feats = feats[:, : cfg.num_features]

        x[i] = feats[: cfg.lookback_weeks]

        cur_demand = float(demand[cfg.lookback_weeks - 1])
        fut_demand = float(demand[cfg.lookback_weeks :].mean())
        growth = (fut_demand - cur_demand) / max(1e-6, abs(cur_demand))

        fut_vol = float(vol[cfg.lookback_weeks :].mean())
        fut_tight = float(np.maximum(0.0, tight[cfg.lookback_weeks :]).mean())

        base_inv = 8.0
        target_inv = base_inv + 6.0 * max(0.0, -growth) + 2.5 * fut_vol + 1.5 * fut_tight
        target_inv = float(np.clip(target_inv, 2.0, 26.0))

        ratio = 0.22 * growth - 0.05 * fut_vol + 0.03 * (tight[cfg.lookback_weeks - 1])
        ratio = float(np.clip(ratio, -cfg.max_ratio, cfg.max_ratio))

        y_inv[i] = target_inv
        y_ratio[i] = ratio

    return SimulatedBatch(x=x, y_inv_weeks=y_inv, y_adj_ratio=y_ratio)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    acc = 0.0
    q = []
    for i, v in enumerate(x.astype(np.float32)):
        q.append(float(v))
        acc += float(v)
        if len(q) > window:
            acc -= q.pop(0)
        out[i] = acc / len(q)
    return out


class SimulatedDataset(Dataset):
    def __init__(self, batch: SimulatedBatch):
        self.x = torch.from_numpy(batch.x)
        self.y_inv = torch.from_numpy(batch.y_inv_weeks).unsqueeze(-1)
        self.y_ratio = torch.from_numpy(batch.y_adj_ratio).unsqueeze(-1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_inv[idx], self.y_ratio[idx]
