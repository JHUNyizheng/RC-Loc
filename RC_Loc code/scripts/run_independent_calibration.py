from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indoorloc.metrics import localization_errors, localization_metrics  # noqa: E402
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


def three_way_group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_index = np.arange(len(groups))
    remain_local, calibration_local = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(
            all_index, groups=groups
        )
    )
    remain_index = all_index[remain_local]
    calibration_index = all_index[calibration_local]
    # 15% of the original groups for selection, 70% for fitting.
    fit_local, selection_local = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.15 / 0.85, random_state=seed + 100_003
        ).split(remain_index, groups=groups[remain_index])
    )
    return remain_index[fit_local], remain_index[selection_local], calibration_index


def finite_sample_quantile(scores: np.ndarray, coverage: float = 0.90) -> tuple[float, int]:
    values = np.sort(np.asarray(scores, dtype=float))
    rank = min(len(values), math.ceil((len(values) + 1) * coverage))
    return float(values[rank - 1]), rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts", "tampere", "uji_library"])
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111],
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output-file", default="independent_calibration_metrics.csv")
    parser.add_argument("--method-name", default="MRC-Loc-independent-calibration")
    parser.add_argument("--artifact-subdir", default="independent_calibration")
    args = parser.parse_args()

    out = ROOT / "results" / "twc_revision"
    prediction_dir = out / "predictions" / args.artifact_subdir
    history_dir = out / "histories" / args.artifact_subdir
    out.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out / args.output_file
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    complete = {(row["dataset"], int(row["seed"])) for row in rows}

    datasets = load_twc_datasets(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.datasets:
        split = datasets[key]
        for seed in args.seeds:
            if (split.name, seed) in complete:
                print(f"CAL {split.name} seed={seed}: already complete", flush=True)
                continue
            seed_all(seed)
            fit_idx, selection_idx, calibration_idx = three_way_group_split(split.group_train, seed)
            fit_groups = set(split.group_train[fit_idx].tolist())
            selection_groups = set(split.group_train[selection_idx].tolist())
            calibration_groups = set(split.group_train[calibration_idx].tolist())
            if fit_groups & selection_groups or fit_groups & calibration_groups or selection_groups & calibration_groups:
                raise RuntimeError("Three-way spatial groups are not disjoint")

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
                pretrain_epochs=args.pretrain_epochs,
                patience=args.patience,
                topology_weight=0.0,
            )
            fit_rep = neural_predict(fit.model, split.x_train[fit_idx], device, fit.coord_mean, fit.coord_std)
            select_rep = neural_predict(
                fit.model, split.x_train[selection_idx], device, fit.coord_mean, fit.coord_std
            )
            calibration_rep = neural_predict(
                fit.model, split.x_train[calibration_idx], device, fit.coord_mean, fit.coord_std
            )
            test_rep = neural_predict(fit.model, split.x_test, device, fit.coord_mean, fit.coord_std)
            fusion = select_fused_graph(
                pca_pack[0],
                split.x_train[fit_idx],
                split.y_train[fit_idx],
                fit_rep["embedding"],
                split.x_train[selection_idx],
                split.y_train[selection_idx],
                select_rep["embedding"],
                select_rep["coord"],
            )
            calibration_pred, calibration_risk = fused_graph_predict(
                fusion,
                pca_pack[0],
                split.x_train[fit_idx],
                split.y_train[fit_idx],
                fit_rep["embedding"],
                split.x_train[calibration_idx],
                calibration_rep["embedding"],
                calibration_rep["coord"],
            )
            test_pred, test_risk = fused_graph_predict(
                fusion,
                pca_pack[0],
                split.x_train[fit_idx],
                split.y_train[fit_idx],
                fit_rep["embedding"],
                split.x_test,
                test_rep["embedding"],
                test_rep["coord"],
            )
            calibration_error = localization_errors(split.y_train[calibration_idx], calibration_pred)
            scores = calibration_error / np.maximum(calibration_risk, 1e-4)
            q90, rank = finite_sample_quantile(scores, 0.90)
            test_radius = q90 * test_risk
            test_error = localization_errors(split.y_test, test_pred)
            metrics = localization_metrics(split.y_test, test_pred)
            rows.append({
                "dataset": split.name,
                "seed": seed,
                "method": args.method_name,
                **metrics,
                "coverage_q90": float(np.mean(test_error <= test_radius)),
                "median_radius_q90_m": float(np.median(test_radius)),
                "conformal_scale": q90,
                "finite_sample_rank": rank,
                "n_fit": len(fit_idx),
                "n_selection": len(selection_idx),
                "n_calibration": len(calibration_idx),
                "fit_groups": len(fit_groups),
                "selection_groups": len(selection_groups),
                "calibration_groups": len(calibration_groups),
                "best_epoch": fit.best_epoch,
                "train_seconds": fit.train_seconds,
                "selected_gamma": fusion[1],
                "selected_k": fusion[2],
                "selected_power": fusion[3],
                "selected_blend": repr(fusion[4]),
            })
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            pd.DataFrame(fit.history).to_csv(history_dir / f"{key}_seed{seed}.csv", index=False)
            np.savez_compressed(
                prediction_dir / f"{key}_seed{seed}.npz",
                y_true=split.y_test,
                prediction=test_pred,
                error=test_error,
                risk=test_risk,
                radius_q90=test_radius,
                calibration_scores=scores,
            )
            print(
                f"CAL {split.name} seed={seed}: coverage={np.mean(test_error <= test_radius):.3f} "
                f"median={metrics['median_m']:.3f} ncal={len(calibration_idx)} rank={rank}",
                flush=True,
            )

    manifest = {
        "split": "70% fit / 15% model-selection / 15% calibration by disjoint spatial group",
        "nominal_coverage": 0.90,
        "quantile_rank": "ceil((n_cal+1)*0.90), clipped at n_cal",
        "test_labels_used": False,
        "exchangeability_caveat": (
            "Finite-sample marginal coverage requires calibration/test exchangeability; official temporal/device "
            "shifts are therefore reported as coverage stress tests, not guaranteed coverage."
        ),
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "patience": args.patience,
    }
    (out / f"{args.artifact_subdir}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
