from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from litho_capacity_ai.config import TrainConfig
from litho_capacity_ai.data.simulated import make_simulated
from litho_capacity_ai.data.public_sources import SeriesDef, latest_public_window
from litho_capacity_ai.io import load_meta, load_scaler
from litho_capacity_ai.model import CapacityNet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="artifacts")
    p.add_argument("--data", type=str, default="simulated", choices=["simulated", "public"])
    p.add_argument("--cache_dir", type=str, default="data_cache")
    p.add_argument("--ttl_hours", type=float, default=6.0)
    p.add_argument("--timeout_s", type=float, default=15.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--offline_dir", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir)

    meta = load_meta(model_dir)
    train_cfg = TrainConfig(**meta["train_config"])
    scaler = load_scaler(model_dir)

    model = CapacityNet(
        num_features=train_cfg.num_features,
        hidden_size=train_cfg.hidden_size,
        num_layers=train_cfg.num_layers,
        dropout=train_cfg.dropout,
        max_ratio=train_cfg.max_ratio,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    if args.data == "simulated":
        batch = make_simulated(replace(train_cfg, train_samples=1, val_samples=1), n_samples=1, seed_offset=99)
        x = batch.x[0]
        x = scaler.transform(x.reshape(-1, train_cfg.num_features)).reshape(1, train_cfg.lookback_weeks, train_cfg.num_features)
        x_t = torch.from_numpy(x.astype(np.float32))
    else:
        public_meta = meta.get("public")
        if not isinstance(public_meta, dict):
            raise ValueError("missing public meta")
        series_defs = [SeriesDef.from_dict(d) for d in public_meta["series"]]
        if args.offline_dir:
            base = Path(args.offline_dir)
            series_defs = [SeriesDef(name=s.name, source="file", key=str(base / f"{s.name}.csv"), kind=s.kind) for s in series_defs]
        window, ts = latest_public_window(
            series=series_defs,
            lookback_weeks=train_cfg.lookback_weeks,
            start=public_meta.get("start"),
            end=public_meta.get("end"),
            cache_dir=args.cache_dir,
            ttl_hours=args.ttl_hours,
            timeout_s=args.timeout_s,
            retries=args.retries,
        )
        x_np = window.to_numpy(dtype=np.float32)
        x_np = scaler.transform(x_np.reshape(-1, train_cfg.num_features)).reshape(1, train_cfg.lookback_weeks, train_cfg.num_features)
        x_t = torch.from_numpy(x_np.astype(np.float32))

    out = model(x_t)
    payload = {
        "target_inventory_weeks": float(out.inv_mu.item()),
        "capacity_adjust_ratio": float(out.ratio_mu.item()),
        "confidence": float(out.confidence().item()),
    }
    if args.data == "public":
        payload["asof_week"] = str(ts)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
