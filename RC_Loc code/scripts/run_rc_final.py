from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indoorloc.metrics import localization_metrics  # noqa: E402
from indoorloc.twc_data import load_twc_datasets  # noqa: E402
from indoorloc.twc_models import MaskTopoLoc  # noqa: E402
from indoorloc.twc_training import (  # noqa: E402
    fused_graph_predict,
    neural_predict,
    seed_all,
    select_fused_graph,
    select_pca_graph,
    train_neural,
)


def corrupt_numpy(x: np.ndarray, drop_prob: float, noise_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(x, dtype=np.float32).copy()
    active = out > 0
    if drop_prob:
        out[active & (rng.random(out.shape) < drop_prob)] = 0
    if noise_db:
        noise = rng.normal(0, noise_db / 110.0, out.shape).astype(np.float32)
        out = np.where(out > 0, np.clip(out + noise, 1e-4, 1.0), 0)
    return out


def metric_row(split, seed, method, scenario, prediction, fit, latency, floor=None, building=None):
    return {
        "dataset": split.name,
        "seed": seed,
        "method": method,
        "scenario": scenario,
        **localization_metrics(
            split.y_test, prediction, split.floor_test, floor,
            split.building_test, building,
        ),
        "train_seconds": fit.train_seconds,
        "latency_ms_per_sample": latency,
        "best_epoch": fit.best_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts", "tampere", "uji_library"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    out = ROOT / "results" / "twc_revision"
    model_dir = out / "models" / "rc_final"
    history_dir = out / "histories" / "rc_final"
    prediction_dir = out / "predictions" / "rc_final"
    for directory in (out, model_dir, history_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "rc_final_metrics.csv"
    tuning_path = out / "rc_final_tuning.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    tuning = pd.read_csv(tuning_path).to_dict("records") if tuning_path.exists() else []
    complete = {
        (row["dataset"], int(row["seed"])) for row in rows
        if row["method"] == "RC-Loc" and row["scenario"] == "clean"
    }
    datasets = load_twc_datasets(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key in args.datasets:
        split = datasets[key]
        for seed in args.seeds:
            if (split.name, seed) in complete:
                print(f"RC {split.name} seed={seed}: already complete", flush=True)
                continue
            seed_all(seed)
            fit_idx, selection_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed).split(
                    split.x_train, groups=split.group_train
                )
            )
            _, pca_pack = select_pca_graph(
                split.x_train[fit_idx], split.y_train[fit_idx],
                split.x_train[selection_idx], split.y_train[selection_idx], seed,
            )
            model = MaskTopoLoc(split.n_aps, split.n_floors, split.n_buildings)
            fit = train_neural(
                model,
                split.x_train,
                split.y_train,
                split.floor_train,
                split.building_train,
                fit_idx,
                selection_idx,
                device,
                proposed=True,
                max_epochs=args.max_epochs,
                pretrain_epochs=0,
                patience=args.patience,
                topology_weight=0.0,
            )
            fit_rep = neural_predict(fit.model, split.x_train[fit_idx], device, fit.coord_mean, fit.coord_std)
            selection_rep = neural_predict(
                fit.model, split.x_train[selection_idx], device, fit.coord_mean, fit.coord_std
            )
            test_rep = neural_predict(fit.model, split.x_test, device, fit.coord_mean, fit.coord_std)
            fusion = select_fused_graph(
                pca_pack[0],
                split.x_train[fit_idx],
                split.y_train[fit_idx],
                fit_rep["embedding"],
                split.x_train[selection_idx],
                split.y_train[selection_idx],
                selection_rep["embedding"],
                selection_rep["coord"],
            )
            start = time.perf_counter()
            prediction, risk = fused_graph_predict(
                fusion,
                pca_pack[0],
                split.x_train[fit_idx],
                split.y_train[fit_idx],
                fit_rep["embedding"],
                split.x_test,
                test_rep["embedding"],
                test_rep["coord"],
            )
            latency = (time.perf_counter() - start) * 1000.0 / len(split.x_test)
            rows.append(metric_row(
                split, seed, "RC-Loc", "clean", prediction, fit, latency,
                test_rep["floor"], test_rep["building"],
            ))
            rows.append(metric_row(
                split, seed, "RC-Direct", "clean", test_rep["coord"], fit, latency,
                test_rep["floor"], test_rep["building"],
            ))
            tuning.append({
                "dataset": split.name,
                "seed": seed,
                "gamma": fusion[1],
                "k": fusion[2],
                "power": fusion[3],
                "direct_blend": repr(fusion[4]),
                **{f"val_{name}": value for name, value in fusion[5].items()},
            })

            if key in {"uji", "uts"}:
                for drop, noise in ((0.2, 0), (0.4, 0), (0.6, 0), (0.2, 2), (0.2, 4), (0.2, 6)):
                    corrupted = corrupt_numpy(split.x_test, drop, noise, seed + 1000)
                    corrupted_rep = neural_predict(
                        fit.model, corrupted, device, fit.coord_mean, fit.coord_std
                    )
                    robust_prediction, _ = fused_graph_predict(
                        fusion,
                        pca_pack[0],
                        split.x_train[fit_idx],
                        split.y_train[fit_idx],
                        fit_rep["embedding"],
                        corrupted,
                        corrupted_rep["embedding"],
                        corrupted_rep["coord"],
                    )
                    rows.append(metric_row(
                        split, seed, "RC-Loc", f"drop={drop},noise={noise}",
                        robust_prediction, fit, latency,
                    ))

            if key == "uji_library":
                for month in sorted(set(split.test_domain.tolist())):
                    mask = split.test_domain == month
                    month_metrics = localization_metrics(split.y_test[mask], prediction[mask])
                    rows.append({
                        "dataset": split.name, "seed": seed, "method": "RC-Loc",
                        "scenario": month, **month_metrics,
                        "train_seconds": fit.train_seconds,
                        "latency_ms_per_sample": latency,
                        "best_epoch": fit.best_epoch,
                    })

            torch.save({
                "state_dict": fit.model.state_dict(),
                "coord_mean": fit.coord_mean,
                "coord_std": fit.coord_std,
                "best_epoch": fit.best_epoch,
                "n_aps": split.n_aps,
                "n_floors": split.n_floors,
                "n_buildings": split.n_buildings,
            }, model_dir / f"{key}_seed{seed}.pt")
            joblib.dump((pca_pack[0], fusion), model_dir / f"{key}_fusion_seed{seed}.joblib")
            pd.DataFrame(fit.history).to_csv(history_dir / f"{key}_seed{seed}.csv", index=False)
            np.savez_compressed(
                prediction_dir / f"{key}_seed{seed}.npz",
                y_true=split.y_test,
                prediction=prediction,
                direct=test_rep["coord"],
                risk=risk,
                floor_true=split.floor_test,
                floor_prediction=test_rep["floor"],
                building_true=split.building_test,
                building_prediction=test_rep["building"],
                domain=split.test_domain,
            )
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            pd.DataFrame(tuning).to_csv(tuning_path, index=False)
            clean = localization_metrics(split.y_test, prediction)
            print(
                f"RC {split.name} seed={seed}: mean={clean['mean_m']:.3f} "
                f"p90={clean['p90_m']:.3f} epoch={fit.best_epoch}",
                flush=True,
            )

    manifest = {
        "method": "RC-Loc",
        "removed_after_review": ["masked pretraining", "fixed topology loss", "conformal claim from selection set"],
        "retained": ["explicit AP mask", "corrupted-view supervision", "representation consistency", "anchor/direct fusion"],
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "pretrain_epochs": 0,
        "patience": args.patience,
        "test_labels_used_for_selection": False,
    }
    (out / "rc_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
