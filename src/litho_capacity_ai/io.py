from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "StandardScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.maximum(std, eps)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


def save_artifacts(out_dir: str | Path, model: torch.nn.Module, scaler: StandardScaler, meta: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "scaler.json").write_text(
        json.dumps(
            {
                "mean": scaler.mean.tolist(),
                "std": scaler.std.tolist(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_scaler(model_dir: str | Path) -> StandardScaler:
    model_dir = Path(model_dir)
    payload = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
    return StandardScaler(mean=np.array(payload["mean"], dtype=np.float32), std=np.array(payload["std"], dtype=np.float32))


def load_meta(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    return json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
