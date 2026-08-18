from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indoorloc.metrics import localization_metrics  # noqa: E402
from indoorloc.twc_data import load_twc_datasets  # noqa: E402
from indoorloc.twc_models import Anchor2VecTransformer  # noqa: E402
from indoorloc.twc_training import neural_predict, seed_all, train_neural  # noqa: E402


METHOD = "AaT-base-reimpl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts", "tampere", "uji_library"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out = ROOT / "results" / "twc_revision"
    model_dir = out / "models" / "aat_base"
    history_dir = out / "histories" / "aat_base"
    prediction_dir = out / "predictions" / "aat_base"
    for directory in (out, model_dir, history_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "anchor_transformer_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    complete = {(row["dataset"], int(row["seed"])) for row in rows if row["scenario"] == "clean"}

    datasets = load_twc_datasets(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.datasets:
        split = datasets[key]
        for seed in args.seeds:
            if (split.name, seed) in complete:
                print(f"AAT {split.name} seed={seed}: already complete", flush=True)
                continue
            seed_all(seed)
            fit_idx, select_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed).split(
                    split.x_train, groups=split.group_train
                )
            )
            model = Anchor2VecTransformer(split.n_aps, split.n_floors, split.n_buildings)
            parameters = int(sum(parameter.numel() for parameter in model.parameters()))
            fit = train_neural(
                model,
                split.x_train,
                split.y_train,
                split.floor_train,
                split.building_train,
                fit_idx,
                select_idx,
                device,
                proposed=False,
                max_epochs=args.epochs,
                pretrain_epochs=0,
                patience=args.epochs,
                batch_size=args.batch_size,
                lr=1e-4,
                weight_decay=0.0,
                coordinate_loss="l1",
            )
            start = time.perf_counter()
            pred = neural_predict(fit.model, split.x_test, device, fit.coord_mean, fit.coord_std)
            latency = (time.perf_counter() - start) * 1000.0 / len(split.x_test)
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
                "method": METHOD,
                "scenario": "clean",
                **metrics,
                "best_epoch": fit.best_epoch,
                "trained_epochs": args.epochs,
                "train_seconds": fit.train_seconds,
                "latency_ms_per_sample": latency,
                "parameters": parameters,
                "fit_fingerprints": len(fit_idx),
                "selection_fingerprints": len(select_idx),
            })
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            pd.DataFrame(fit.history).to_csv(history_dir / f"{key}_seed{seed}.csv", index=False)
            torch.save({
                "state_dict": fit.model.state_dict(),
                "coord_mean": fit.coord_mean,
                "coord_std": fit.coord_std,
                "best_epoch": fit.best_epoch,
                "n_aps": split.n_aps,
                "n_floors": split.n_floors,
                "n_buildings": split.n_buildings,
            }, model_dir / f"{key}_seed{seed}.pt")
            np.savez_compressed(
                prediction_dir / f"{key}_seed{seed}.npz",
                y_true=split.y_test,
                prediction=pred["coord"],
                floor_true=split.floor_test,
                floor_prediction=pred["floor"],
                building_true=split.building_test,
                building_prediction=pred["building"],
            )
            print(
                f"AAT {split.name} seed={seed}: mean={metrics['mean_m']:.3f} "
                f"p90={metrics['p90_m']:.3f} best={fit.best_epoch}/{args.epochs}",
                flush=True,
            )

    manifest = {
        "method": METHOD,
        "paper_mapping": "base Anchor-agnostic Transformer (bAaT) architecture",
        "qualification": "protocol-matched independent reimplementation; not official eAaT code",
        "tokens": 64,
        "d_model": 128,
        "encoder_layers": 3,
        "heads": 8,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW with zero weight decay (equivalent decoupled decay disabled)",
        "learning_rate": 1e-4,
        "coordinate_loss": "L1",
        "seeds": args.seeds,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out / "anchor_transformer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
