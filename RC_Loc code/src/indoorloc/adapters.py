from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HuaweiSchema:
    sample_id: str = "sample_id"
    timestamp: str = "timestamp"
    anchor_id: str = "anchor_id"
    rssi_dbm: str = "rssi_dbm"
    x: str = "x_m"
    y: str = "y_m"
    z: str = "z_m"
    floor: str = "floor"
    building: str = "building"
    device: str = "device_id"


@dataclass
class HuaweiFingerprintData:
    sample_ids: np.ndarray
    anchor_ids: list[str]
    rssi_dbm: np.ndarray
    x: np.ndarray | None
    y: np.ndarray | None
    z: np.ndarray | None
    floor: np.ndarray | None
    building: np.ndarray | None
    device: np.ndarray | None
    timestamp: np.ndarray | None


def _first_per_sample(df: pd.DataFrame, sample_col: str, column: str) -> np.ndarray | None:
    if column not in df.columns:
        return None
    values = df.groupby(sample_col, sort=True)[column].first()
    return values.to_numpy()


def load_huawei_long_csv(
    path: str | Path,
    *,
    schema: HuaweiSchema = HuaweiSchema(),
    anchor_order: Iterable[str] | None = None,
    require_coordinates: bool = True,
    rssi_min_dbm: float = -120.0,
    rssi_max_dbm: float = 0.0,
) -> HuaweiFingerprintData:
    """Load the recommended Huawei long-form fingerprint CSV.

    Repeated readings for the same (sample, anchor) are aggregated by median.
    Missing anchors remain NaN so that the caller can apply a training-specific
    missing-value policy without confusing a missing reading with weak RSSI.
    """
    path = Path(path)
    df = pd.read_csv(path)
    required = {schema.sample_id, schema.anchor_id, schema.rssi_dbm}
    if require_coordinates:
        required |= {schema.x, schema.y}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required Huawei columns: {missing}")
    if df.empty:
        raise ValueError("Huawei fingerprint file is empty")
    if df[[schema.sample_id, schema.anchor_id]].isna().any().any():
        raise ValueError("sample_id and anchor_id must not contain null values")

    df = df.copy()
    df[schema.anchor_id] = df[schema.anchor_id].astype(str)
    df[schema.rssi_dbm] = pd.to_numeric(df[schema.rssi_dbm], errors="coerce")
    if df[schema.rssi_dbm].isna().any():
        raise ValueError("rssi_dbm contains non-numeric or null values")
    if not df[schema.rssi_dbm].between(rssi_min_dbm, rssi_max_dbm).all():
        raise ValueError(f"rssi_dbm must be within [{rssi_min_dbm}, {rssi_max_dbm}] dBm")

    pivot = df.pivot_table(
        index=schema.sample_id,
        columns=schema.anchor_id,
        values=schema.rssi_dbm,
        aggfunc="median",
        sort=True,
    )
    if anchor_order is None:
        anchors = sorted(map(str, pivot.columns))
    else:
        anchors = list(map(str, anchor_order))
        unknown = sorted(set(map(str, pivot.columns)) - set(anchors))
        if unknown:
            raise ValueError(f"Input contains anchors absent from anchor_order: {unknown[:8]}")
    pivot = pivot.reindex(columns=anchors)

    def numeric_meta(column: str) -> np.ndarray | None:
        values = _first_per_sample(df, schema.sample_id, column)
        if values is None:
            return None
        converted = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
        if require_coordinates and column in {schema.x, schema.y} and np.isnan(converted).any():
            raise ValueError(f"{column} contains non-numeric or null values")
        return converted

    return HuaweiFingerprintData(
        sample_ids=pivot.index.to_numpy(),
        anchor_ids=anchors,
        rssi_dbm=pivot.to_numpy(np.float32),
        x=numeric_meta(schema.x),
        y=numeric_meta(schema.y),
        z=numeric_meta(schema.z),
        floor=_first_per_sample(df, schema.sample_id, schema.floor),
        building=_first_per_sample(df, schema.sample_id, schema.building),
        device=_first_per_sample(df, schema.sample_id, schema.device),
        timestamp=_first_per_sample(df, schema.sample_id, schema.timestamp),
    )


def normalize_huawei_rssi(rssi_dbm: np.ndarray, floor_dbm: float = -120.0) -> np.ndarray:
    """Normalize observed Huawei RSSI to (0, 1], preserving NaN as missing=0."""
    x = np.asarray(rssi_dbm, dtype=np.float32)
    out = np.zeros_like(x)
    observed = np.isfinite(x)
    clipped = np.clip(x[observed], floor_dbm, 0.0)
    out[observed] = (clipped - floor_dbm) / (-floor_dbm)
    return out
