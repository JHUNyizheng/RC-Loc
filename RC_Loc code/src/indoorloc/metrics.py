from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def localization_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(y_true) - np.asarray(y_pred), axis=1)


def localization_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor_true: np.ndarray | None = None,
    floor_pred: np.ndarray | None = None,
    building_true: np.ndarray | None = None,
    building_pred: np.ndarray | None = None,
) -> dict[str, float]:
    err = localization_errors(y_true, y_pred)
    result = {
        "n": float(len(err)),
        "mean_m": float(np.mean(err)),
        "median_m": float(np.median(err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "p75_m": float(np.percentile(err, 75)),
        "p90_m": float(np.percentile(err, 90)),
        "p95_m": float(np.percentile(err, 95)),
        "within_3m": float(np.mean(err <= 3.0)),
    }
    if floor_true is not None and floor_pred is not None:
        truth = np.asarray(floor_true)
        pred = np.asarray(floor_pred)
        known = truth >= 0
        result["floor_accuracy"] = float(np.mean(truth[known] == pred[known])) if known.any() else float("nan")
        result["floor_unknown_fraction"] = float(np.mean(~known))
    if building_true is not None and building_pred is not None:
        truth = np.asarray(building_true)
        pred = np.asarray(building_pred)
        known = truth >= 0
        result["building_accuracy"] = float(np.mean(truth[known] == pred[known])) if known.any() else float("nan")
        result["building_unknown_fraction"] = float(np.mean(~known))
    return result


def bootstrap_difference_ci(
    reference_errors: np.ndarray,
    candidate_errors: np.ndarray,
    statistic: str = "median",
    n_boot: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    reference_errors = np.asarray(reference_errors)
    candidate_errors = np.asarray(candidate_errors)
    if len(reference_errors) != len(candidate_errors):
        raise ValueError("Paired errors must have equal length")
    fn = np.median if statistic == "median" else np.mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(reference_errors), size=(n_boot, len(reference_errors)))
    diffs = fn(candidate_errors[indices], axis=1) - fn(reference_errors[indices], axis=1)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def paired_wilcoxon(reference_errors: np.ndarray, candidate_errors: np.ndarray) -> float:
    delta = np.asarray(candidate_errors) - np.asarray(reference_errors)
    if np.allclose(delta, 0):
        return 1.0
    return float(wilcoxon(delta, alternative="two-sided").pvalue)
