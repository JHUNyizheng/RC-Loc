from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indoorloc.metrics import localization_metrics  # noqa: E402
from indoorloc.twc_data import load_twc_datasets  # noqa: E402
from indoorloc.twc_models import ConvRSSLoc, DNNBNLoc  # noqa: E402
from indoorloc.twc_training import neural_predict, seed_all, train_neural  # noqa: E402


MODEL_FACTORIES = {
    "DNNBN-reimpl": DNNBNLoc,
    "CNN-RSS-reimpl": ConvRSSLoc,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts", "tampere", "uji_library"])
    parser.add_argument("--methods", nargs="+", choices=sorted(MODEL_FACTORIES), default=sorted(MODEL_FACTORIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    out = ROOT / "results" / "twc_extended"
    model_root = out / "models" / "modern_baselines"
    history_root = out / "histories" / "modern_baselines"
    prediction_root = out / "predictions" / "modern_baselines"
    for directory in (out, model_root, history_root, prediction_root):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "modern_baseline_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    complete = {(r["dataset"], int(r["seed"]), r["method"]) for r in rows if r["scenario"] == "clean"}
    datasets = load_twc_datasets(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key in args.datasets:
        split = datasets[key]
        for seed in args.seeds:
            fit_idx, selection_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed).split(
                    split.x_train, groups=split.group_train
                )
            )
            for method in args.methods:
                if (split.name, seed, method) in complete:
                    print(f"{method} {split.name} seed={seed}: already complete", flush=True)
                    continue
                seed_all(seed)
                model = MODEL_FACTORIES[method](split.n_aps, split.n_floors, split.n_buildings)
                fit = train_neural(
                    model,
                    split.x_train,
                    split.y_train,
                    split.floor_train,
                    split.building_train,
                    fit_idx,
                    selection_idx,
                    device,
                    proposed=False,
                    max_epochs=args.max_epochs,
                    pretrain_epochs=0,
                    patience=args.patience,
                    batch_size=256,
                    lr=5e-4,
                    weight_decay=2e-4,
                )
                pred = neural_predict(fit.model, split.x_test, device, fit.coord_mean, fit.coord_std)
                metrics = localization_metrics(
                    split.y_test,
                    pred["coord"],
                    split.floor_test,
                    pred["floor"],
                    split.building_test,
                    pred["building"],
                )
                rows.append({
                    "dataset": split.name,
                    "seed": seed,
                    "method": method,
                    "scenario": "clean",
                    **metrics,
                    "train_seconds": fit.train_seconds,
                    "best_epoch": fit.best_epoch,
                    "trained_epochs": len(fit.history),
                    "parameters": sum(p.numel() for p in fit.model.parameters()),
                    "fit_fingerprints": len(fit_idx),
                    "selection_fingerprints": len(selection_idx),
                })

                method_key = method.lower().replace("-", "_")
                model_dir = model_root / method_key
                history_dir = history_root / method_key
                prediction_dir = prediction_root / method_key
                for directory in (model_dir, history_dir, prediction_dir):
                    directory.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": fit.model.state_dict(),
                    "coord_mean": fit.coord_mean,
                    "coord_std": fit.coord_std,
                    "best_epoch": fit.best_epoch,
                    "n_aps": split.n_aps,
                    "n_floors": split.n_floors,
                    "n_buildings": split.n_buildings,
                    "method": method,
                }, model_dir / f"{key}_seed{seed}.pt")
                pd.DataFrame(fit.history).to_csv(history_dir / f"{key}_seed{seed}.csv", index=False)
                np.savez_compressed(
                    prediction_dir / f"{key}_seed{seed}.npz",
                    y_true=split.y_test,
                    prediction=pred["coord"],
                    floor_true=split.floor_test,
                    floor_prediction=pred["floor"],
                    building_true=split.building_test,
                    building_prediction=pred["building"],
                    domain=split.test_domain,
                )
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                print(
                    f"{method} {split.name} seed={seed}: mean={metrics['mean_m']:.3f} "
                    f"p90={metrics['p90_m']:.3f} epoch={fit.best_epoch}/{len(fit.history)}",
                    flush=True,
                )

    manifest = {
        "methods": args.methods,
        "qualification": "independent protocol-matched reimplementations, not official author code",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection": "18% spatial reference-point groups",
        "official_test_used_for_selection": False,
    }
    (out / "modern_baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
