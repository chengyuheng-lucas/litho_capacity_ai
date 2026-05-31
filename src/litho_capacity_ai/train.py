from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from litho_capacity_ai.config import TrainConfig
from litho_capacity_ai.data.simulated import SimulatedDataset, make_simulated
from litho_capacity_ai.data.supervised import SupervisedDataset
from litho_capacity_ai.data.public_sources import DEFAULT_SERIES, SeriesDef, default_public_end, make_public_samples
from litho_capacity_ai.io import StandardScaler, save_artifacts
from litho_capacity_ai.model import CapacityNet, gaussian_nll


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="simulated", choices=["simulated", "public"])
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="data_cache")
    p.add_argument("--ttl_hours", type=float, default=6.0)
    p.add_argument("--timeout_s", type=float, default=15.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--offline_dir", type=str, default=None)
    p.add_argument("--public_profile", type=str, default="full", choices=["full", "fred_fast"])
    p.add_argument("--public_target", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


@torch.no_grad()
def _eval(model: CapacityNet, loader: DataLoader) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    inv_mae = 0.0
    ratio_mae = 0.0

    for x, y_inv, y_ratio in loader:
        out = model(x)
        loss = gaussian_nll(y_inv, out.inv_mu, out.inv_logvar).mean() + gaussian_nll(y_ratio, out.ratio_mu, out.ratio_logvar).mean()
        loss_sum += float(loss.item()) * x.shape[0]

        inv_mae += float(torch.abs(y_inv - out.inv_mu).mean().item()) * x.shape[0]
        ratio_mae += float(torch.abs(y_ratio - out.ratio_mu).mean().item()) * x.shape[0]
        total += x.shape[0]

    return {
        "loss": loss_sum / max(1, total),
        "inv_mae": inv_mae / max(1, total),
        "ratio_mae": ratio_mae / max(1, total),
    }


def main() -> None:
    args = _parse_args()
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    if args.batch_size is not None:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.lr is not None:
        cfg = replace(cfg, lr=args.lr)

    _set_seed(cfg.seed)

    if args.data == "simulated":
        train_batch = make_simulated(cfg, n_samples=cfg.train_samples, seed_offset=1)
        val_batch = make_simulated(cfg, n_samples=cfg.val_samples, seed_offset=2)
        scaler = StandardScaler.fit(train_batch.x.reshape(-1, cfg.num_features))
        x_train = scaler.transform(train_batch.x.reshape(-1, cfg.num_features)).reshape(train_batch.x.shape)
        x_val = scaler.transform(val_batch.x.reshape(-1, cfg.num_features)).reshape(val_batch.x.shape)

        train_ds = SimulatedDataset(batch=type(train_batch)(x=x_train, y_inv_weeks=train_batch.y_inv_weeks, y_adj_ratio=train_batch.y_adj_ratio))
        val_ds = SimulatedDataset(batch=type(val_batch)(x=x_val, y_inv_weeks=val_batch.y_inv_weeks, y_adj_ratio=val_batch.y_adj_ratio))

        feature_names = [f"f{i}" for i in range(cfg.num_features)]
        extra_meta: dict[str, object] = {"data": "simulated"}
    else:
        end = args.end or default_public_end()
        series = DEFAULT_SERIES
        if args.public_profile == "fred_fast":
            allow = {"SP500", "VIX", "US10Y", "US2Y", "10Y2Y", "FEDFUNDS", "USD_CNY", "WTI"}
            series = [s for s in series if s.source == "fred" and s.name in allow]
        if args.offline_dir:
            base = Path(args.offline_dir)
            series = [SeriesDef(name=s.name, source="file", key=str(base / f"{s.name}.csv"), kind=s.kind) for s in series]
        x, y_inv, y_ratio, feature_names, target, sample_index = make_public_samples(
            series=series,
            lookback_weeks=cfg.lookback_weeks,
            horizon_weeks=cfg.horizon_weeks,
            max_ratio=cfg.max_ratio,
            start=args.start,
            end=end,
            cache_dir=args.cache_dir,
            ttl_hours=args.ttl_hours,
            timeout_s=args.timeout_s,
            retries=args.retries,
            target_col=args.public_target,
        )

        n = x.shape[0]
        split = int(n * 0.85)
        x_train_np, x_val_np = x[:split], x[split:]
        y_inv_train_np, y_inv_val_np = y_inv[:split], y_inv[split:]
        y_ratio_train_np, y_ratio_val_np = y_ratio[:split], y_ratio[split:]

        num_features = int(x.shape[-1])
        cfg = replace(cfg, num_features=num_features)

        scaler = StandardScaler.fit(x_train_np.reshape(-1, num_features))
        x_train_np = scaler.transform(x_train_np.reshape(-1, num_features)).reshape(x_train_np.shape)
        x_val_np = scaler.transform(x_val_np.reshape(-1, num_features)).reshape(x_val_np.shape)

        x_train_t = torch.from_numpy(x_train_np.astype(np.float32))
        x_val_t = torch.from_numpy(x_val_np.astype(np.float32))
        y_inv_train_t = torch.from_numpy(y_inv_train_np.astype(np.float32)).unsqueeze(-1)
        y_inv_val_t = torch.from_numpy(y_inv_val_np.astype(np.float32)).unsqueeze(-1)
        y_ratio_train_t = torch.from_numpy(y_ratio_train_np.astype(np.float32)).unsqueeze(-1)
        y_ratio_val_t = torch.from_numpy(y_ratio_val_np.astype(np.float32)).unsqueeze(-1)

        train_ds = SupervisedDataset(x=x_train_t, y_inv=y_inv_train_t, y_ratio=y_ratio_train_t)
        val_ds = SupervisedDataset(x=x_val_t, y_inv=y_inv_val_t, y_ratio=y_ratio_val_t)

        extra_meta = {
            "data": "public",
            "public": {
                "series": [s.to_dict() for s in series],
                "feature_names": feature_names,
                "target_col": target,
                "start": args.start,
                "end": end,
                "sample_index_tail": [str(ts) for ts in sample_index[-5:]],
            },
        }

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = CapacityNet(
        num_features=cfg.num_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_ratio=cfg.max_ratio,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y_inv, y_ratio in train_loader:
            out = model(x)
            loss = gaussian_nll(y_inv, out.inv_mu, out.inv_logvar).mean() + gaussian_nll(y_ratio, out.ratio_mu, out.ratio_logvar).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        metrics = _eval(model, val_loader)
        metrics["epoch"] = epoch
        history.append(metrics)

        if metrics["loss"] < best:
            best = metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(json.dumps(metrics, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(args.out_dir)
    meta = {"train_config": asdict(cfg), "val_best_loss": best, "history_tail": history[-5:], "feature_names": feature_names, **extra_meta}
    save_artifacts(out_dir=out_dir, model=model, scaler=scaler, meta=meta)
    print(str((out_dir / "model.pt").resolve()))


if __name__ == "__main__":
    main()
