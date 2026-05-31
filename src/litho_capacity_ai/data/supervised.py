from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SupervisedDataset(Dataset):
    def __init__(self, x: torch.Tensor, y_inv: torch.Tensor, y_ratio: torch.Tensor):
        self.x = x
        self.y_inv = y_inv
        self.y_ratio = y_ratio

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_inv[idx], self.y_ratio[idx]
