from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indoorloc.metrics import localization_metrics  # noqa: E402
from indoorloc.twc_data import load_twc_datasets  # noqa: E402
from indoorloc.twc_training import select_pca_graph, weighted_neighbor_predict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts", "tampere", "uji_library"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    args = parser.parse_args()

    out = ROOT / "results" / "twc_extended"
    pred_root = out / "predictions" / "classical"
    model_root = out / "models" / "classical"
    for directory in (out, pred_root, model_root):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "classical_replay_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    complete = {(r["dataset"], int(r["seed"]), r["method"]) for r in rows}
    datasets = load_twc_datasets(ROOT)

    for key in args.datasets:
        split = datasets[key]
        for seed in args.seeds:
            fit_idx, selection_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed).split(
                    split.x_train, groups=split.group_train
                )
            )
            if (split.name, seed, "PCA-WKNN") not in complete:
                start = time.perf_counter()
                best, pack = select_pca_graph(
                    split.x_train[fit_idx], split.y_train[fit_idx],
                    split.x_train[selection_idx], split.y_train[selection_idx], seed,
                )
                train_seconds = time.perf_counter() - start
                _, _, _, k, power, _ = best
                pca, nn, _ = pack
                start = time.perf_counter()
                prediction, dispersion, distance = weighted_neighbor_predict(
                    nn, split.y_train[fit_idx], pca.transform(split.x_test), k, power
                )
                latency = (time.perf_counter() - start) * 1000 / len(split.x_test)
                rows.append({
                    "dataset": split.name, "seed": seed, "method": "PCA-WKNN", "scenario": "clean",
                    **localization_metrics(split.y_test, prediction),
                    "train_seconds": train_seconds, "latency_ms_per_sample": latency,
                })
                joblib.dump((best, pack, fit_idx), model_root / f"pca_{key}_seed{seed}.joblib")
                np.savez_compressed(
                    pred_root / f"pca_{key}_seed{seed}.npz",
                    y_true=split.y_test, prediction=prediction,
                    risk=dispersion + 0.25 * distance, domain=split.test_domain,
                )
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                print(f"PCA-WKNN {split.name} seed={seed}: p90={rows[-1]['p90_m']:.3f}", flush=True)

            if (split.name, seed, "ExtraTrees") not in complete:
                start = time.perf_counter()
                trees = ExtraTreesRegressor(
                    n_estimators=600, max_features=0.75, min_samples_leaf=1,
                    n_jobs=-1, random_state=seed,
                ).fit(split.x_train[fit_idx], split.y_train[fit_idx])
                train_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction = trees.predict(split.x_test)
                latency = (time.perf_counter() - start) * 1000 / len(split.x_test)
                rows.append({
                    "dataset": split.name, "seed": seed, "method": "ExtraTrees", "scenario": "clean",
                    **localization_metrics(split.y_test, prediction),
                    "train_seconds": train_seconds, "latency_ms_per_sample": latency,
                })
                joblib.dump((trees, fit_idx), model_root / f"extratrees_{key}_seed{seed}.joblib")
                np.savez_compressed(
                    pred_root / f"extratrees_{key}_seed{seed}.npz",
                    y_true=split.y_test, prediction=prediction, domain=split.test_domain,
                )
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                print(f"ExtraTrees {split.name} seed={seed}: p90={rows[-1]['p90_m']:.3f}", flush=True)

    (out / "classical_replay_manifest.json").write_text(json.dumps({
        "datasets": args.datasets, "seeds": args.seeds,
        "selection": "18% spatial reference-point groups",
        "PCA_WKNN": "validation-selected components/metric/k/power",
        "ExtraTrees": {"n_estimators": 600, "max_features": 0.75, "min_samples_leaf": 1},
        "official_test_used_for_selection": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
