from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    lookback_weeks: int = 52
    horizon_weeks: int = 12
    num_features: int = 16
    train_samples: int = 8000
    val_samples: int = 1000
    batch_size: int = 128
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 1e-4
    hidden_size: int = 48
    num_layers: int = 1
    dropout: float = 0.15
    max_ratio: float = 0.3


@dataclass(frozen=True)
class InferConfig:
    lookback_weeks: int = 52
    max_ratio: float = 0.3
