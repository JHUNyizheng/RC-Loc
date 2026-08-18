from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_uji


@dataclass
class RadioMapSplit:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    floor_train: np.ndarray
    building_train: np.ndarray
    group_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    floor_test: np.ndarray
    building_test: np.ndarray
    test_domain: np.ndarray

    @property
    def n_aps(self) -> int:
        return self.x_train.shape[1]

    @property
    def n_floors(self) -> int:
        # Output dimensionality must be determined without consulting test labels.
        return int(self.floor_train.max(initial=0) + 1)

    @property
    def n_buildings(self) -> int:
        return int(self.building_train.max(initial=0) + 1)


def normalize_rss_matrix(rss: np.ndarray, floor_dbm: float = -110.0) -> np.ndarray:
    rss = np.asarray(rss, dtype=np.float32)
    observed = np.isfinite(rss) & (rss < 50.0)
    out = np.zeros_like(rss, dtype=np.float32)
    clipped = np.clip(rss[observed], floor_dbm, 0.0)
    out[observed] = (clipped - floor_dbm) / (-floor_dbm)
    return out


def spatial_groups(y: np.ndarray, floor: np.ndarray, building: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    key = np.column_stack(
        [
            np.round(np.asarray(y)[:, 0] / resolution).astype(np.int64),
            np.round(np.asarray(y)[:, 1] / resolution).astype(np.int64),
            np.asarray(floor, dtype=np.int64),
            np.asarray(building, dtype=np.int64),
        ]
    )
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def _encode_from_train(train_values, test_values) -> tuple[np.ndarray, np.ndarray]:
    """Encode classes from training labels only; unseen test classes map to -1."""
    train_strings = np.asarray(train_values).astype(str)
    test_strings = np.asarray(test_values).astype(str)
    classes = {value: i for i, value in enumerate(sorted(set(train_strings.tolist())))}
    return (
        np.asarray([classes[v] for v in train_strings], dtype=np.int64),
        np.asarray([classes.get(v, -1) for v in test_strings], dtype=np.int64),
    )


def load_uji_twc(root: str | Path) -> RadioMapSplit:
    split = load_uji(root)
    floor_train, floor_test = _encode_from_train(split.floor_train, split.floor_test)
    building_train, building_test = _encode_from_train(split.building_train, split.building_test)
    return RadioMapSplit(
        name="UJIIndoorLoc",
        x_train=split.x_train,
        y_train=split.y_train,
        floor_train=floor_train,
        building_train=building_train,
        group_train=spatial_groups(split.y_train, floor_train, building_train),
        x_test=split.x_test,
        y_test=split.y_test,
        floor_test=floor_test,
        building_test=building_test,
        test_domain=np.full(len(split.x_test), "official_3_month", dtype=object),
    )


def load_uts(root: str | Path) -> RadioMapSplit:
    root = Path(root)
    train = pd.read_csv(root / "UTS_training.csv")
    test = pd.read_csv(root / "UTS_test.csv")
    wap = [c for c in train.columns if c.startswith("WAP")]
    floor_train, floor_test = _encode_from_train(train["Floor_ID"], test["Floor_ID"])
    building_train, building_test = _encode_from_train(train["Building_ID"], test["Building_ID"])
    y_train = train[["Pos_x", "Pos_y"]].to_numpy(np.float32)
    y_test = test[["Pos_x", "Pos_y"]].to_numpy(np.float32)
    return RadioMapSplit(
        name="UTSIndoorLoc",
        x_train=normalize_rss_matrix(train[wap].to_numpy()),
        y_train=y_train,
        floor_train=floor_train,
        building_train=building_train,
        group_train=spatial_groups(y_train, floor_train, building_train),
        x_test=normalize_rss_matrix(test[wap].to_numpy()),
        y_test=y_test,
        floor_test=floor_test,
        building_test=building_test,
        test_domain=np.full(len(test), "official_route", dtype=object),
    )


def load_tampere(root: str | Path) -> RadioMapSplit:
    root = Path(root)
    x_train_raw = pd.read_csv(root / "Training_rss.csv", header=None).to_numpy(np.float32)
    x_test_raw = pd.read_csv(root / "Test_rss.csv", header=None).to_numpy(np.float32)
    c_train = pd.read_csv(root / "Training_coordinates.csv", header=None).to_numpy(np.float32)
    c_test = pd.read_csv(root / "Test_coordinates.csv", header=None).to_numpy(np.float32)
    floor_train, floor_test = _encode_from_train(c_train[:, 2], c_test[:, 2])
    building_train = np.zeros(len(c_train), dtype=np.int64)
    building_test = np.zeros(len(c_test), dtype=np.int64)
    return RadioMapSplit(
        name="Tampere",
        x_train=normalize_rss_matrix(x_train_raw),
        y_train=c_train[:, :2],
        floor_train=floor_train,
        building_train=building_train,
        group_train=spatial_groups(c_train[:, :2], floor_train, building_train, resolution=1.0),
        x_test=normalize_rss_matrix(x_test_raw),
        y_test=c_test[:, :2],
        floor_test=floor_test,
        building_test=building_test,
        test_domain=np.full(len(c_test), "official_trajectory", dtype=object),
    )


def _load_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy(np.float32)


def load_uji_library(root: str | Path) -> RadioMapSplit:
    """Strict forward-time protocol: month 1 training, months 2--25 testing."""
    db = Path(root) / "db"
    train_x, train_y = [], []
    for rss_path in sorted((db / "01").glob("trn*rss.csv")):
        prefix = rss_path.name[:-7]
        train_x.append(_load_csv_matrix(rss_path))
        train_y.append(_load_csv_matrix(rss_path.with_name(prefix + "crd.csv"))[:, :2])

    test_x, test_y, domains = [], [], []
    for month in range(2, 26):
        month_dir = db / f"{month:02d}"
        for rss_path in sorted(month_dir.glob("tst*rss.csv")):
            prefix = rss_path.name[:-7]
            x = _load_csv_matrix(rss_path)
            y = _load_csv_matrix(rss_path.with_name(prefix + "crd.csv"))[:, :2]
            test_x.append(x)
            test_y.append(y)
            domains.extend([f"month_{month:02d}"] * len(x))

    x_train = np.concatenate(train_x)
    y_train = np.concatenate(train_y)
    x_test = np.concatenate(test_x)
    y_test = np.concatenate(test_y)
    floor_train = np.zeros(len(x_train), dtype=np.int64)
    floor_test = np.zeros(len(x_test), dtype=np.int64)
    building_train = np.zeros(len(x_train), dtype=np.int64)
    building_test = np.zeros(len(x_test), dtype=np.int64)
    return RadioMapSplit(
        name="UJI-Library-25M",
        x_train=normalize_rss_matrix(x_train),
        y_train=y_train,
        floor_train=floor_train,
        building_train=building_train,
        group_train=spatial_groups(y_train, floor_train, building_train, resolution=0.25),
        x_test=normalize_rss_matrix(x_test),
        y_test=y_test,
        floor_test=floor_test,
        building_test=building_test,
        test_domain=np.asarray(domains, dtype=object),
    )


def load_twc_datasets(research_root: str | Path) -> dict[str, RadioMapSplit]:
    root = Path(research_root) / "data" / "raw"
    return {
        "uji": load_uji_twc(root / "UJIIndoorLoc" / "UJIndoorLoc"),
        "uts": load_uts(root / "UTSIndoorLoc" / "UTSIndoorLoc"),
        "tampere": load_tampere(root / "Tampere"),
        "uji_library": load_uji_library(root / "UJI_Library" / "extracted"),
    }
