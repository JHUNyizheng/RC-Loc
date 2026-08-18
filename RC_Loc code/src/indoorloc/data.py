from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


N_APS = 520
MISSING_RSS = 100.0
RSS_FLOOR = -110.0


@dataclass
class UJISplit:
    x_train: np.ndarray
    y_train: np.ndarray
    floor_train: np.ndarray
    building_train: np.ndarray
    phone_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    floor_test: np.ndarray
    building_test: np.ndarray
    phone_test: np.ndarray


def normalize_rss(rss: np.ndarray) -> np.ndarray:
    """Map observed dBm values to (0, 1] and missing value 100 to exactly zero."""
    rss = np.asarray(rss, dtype=np.float32)
    observed = rss < MISSING_RSS
    clipped = np.clip(rss, RSS_FLOOR, 0.0)
    out = np.zeros_like(clipped, dtype=np.float32)
    out[observed] = (clipped[observed] - RSS_FLOOR) / (-RSS_FLOOR)
    return out


def denormalize_rss(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = x * (-RSS_FLOOR) + RSS_FLOOR
    out[x <= 0] = MISSING_RSS
    return out


def load_uji(data_dir: str | Path) -> UJISplit:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "trainingData.csv")
    test = pd.read_csv(data_dir / "validationData.csv")
    wap_cols = [f"WAP{i:03d}" for i in range(1, N_APS + 1)]

    def unpack(df: pd.DataFrame):
        return (
            normalize_rss(df[wap_cols].to_numpy()),
            df[["LONGITUDE", "LATITUDE"]].to_numpy(np.float32),
            df["FLOOR"].to_numpy(np.int64),
            df["BUILDINGID"].to_numpy(np.int64),
            df["PHONEID"].to_numpy(np.int64),
        )

    return UJISplit(*unpack(train), *unpack(test))


def inner_train_val_indices(
    floor: np.ndarray,
    building: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(floor))
    strata = np.char.add(building.astype(str), np.char.add("_", floor.astype(str)))
    return train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=strata,
    )


def corrupt_rss(
    x: np.ndarray,
    drop_prob: float = 0.0,
    noise_db: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(x, dtype=np.float32).copy()
    active = out > 0
    if drop_prob > 0:
        drop = rng.random(out.shape) < drop_prob
        out[active & drop] = 0.0
    if noise_db > 0:
        noise = rng.normal(0.0, noise_db / (-RSS_FLOOR), size=out.shape).astype(np.float32)
        out = np.where(out > 0, np.clip(out + noise, 1e-4, 1.0), 0.0)
    return out


class FingerprintDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        floor: np.ndarray,
        building: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.floor = torch.from_numpy(np.asarray(floor, dtype=np.int64))
        self.building = torch.from_numpy(np.asarray(building, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.floor[index], self.building[index]
