from __future__ import annotations

import argparse
import json
import sys
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


VARIANTS = {
    "RC-NoMask": {"use_mask": False},
    "RC-NoAugCoord": {"augmentation_weight": 0.0},
    "RC-NoConsistency": {"consistency_weight": 0.0},
    "RC-NoReconstruction": {"reconstruction_weight": 0.0, "use_decoder": False},
    "RC-NoCoarseHeads": {"floor_weight": 0.0, "building_weight": 0.0},
    "RC-NoCorruption": {
        "augmentation_weight": 0.0,
        "consistency_weight": 0.0,
        "corruption_drop": 0.0,
        "corruption_noise_db": 0.0,
    },
}


def shifted_inputs(x: np.ndarray, fit_x: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    active_frequency = np.mean(fit_x > 0, axis=0)
    observed = np.flatnonzero(active_frequency > 0)
    ranked = observed[np.argsort(active_frequency[observed], kind="stable")[::-1]]
    top_count = max(1, int(np.ceil(0.20 * len(observed))))
    top = ranked[:top_count]
    rng = np.random.default_rng(seed + 7000)

    outage = x.copy()
    outage[:, top] = 0.0

    bias_db = rng.normal(0.0, 8.0, size=x.shape[1]).astype(np.float32)
    bias = np.where(x > 0, np.clip(x + bias_db[None, :] / 110.0, 1e-4, 1.0), 0.0).astype(np.float32)
    offset = np.where(x > 0, np.clip(x - 6.0 / 110.0, 1e-4, 1.0), 0.0).astype(np.float32)
    combined = np.where(outage > 0, np.clip(outage + bias_db[None, :] / 220.0, 1e-4, 1.0), 0.0).astype(np.float32)
    return {
        "clean": x,
        "top20_outage": outage,
        "persistent_bias_8db": bias,
        "device_offset_-6db": offset,
        "top20_outage_plus_bias4db": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output-name", default="twc_extended")
    args = parser.parse_args()

    out = ROOT / "results" / args.output_name
    model_root = out / "models" / "component_ablations"
    history_root = out / "histories" / "component_ablations"
    prediction_root = out / "predictions" / "component_ablations"
    for directory in (out, model_root, history_root, prediction_root):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "component_ablation_metrics.csv"
    selection_path = out / "component_ablation_selection.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    selections = pd.read_csv(selection_path).to_dict("records") if selection_path.exists() else []
    complete = {
        (r["dataset"], int(r["seed"]), r["method"])
        for r in rows if r["scenario"] == "clean" and r["estimator"] == "fused"
    }
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
            _, pca_pack = select_pca_graph(
                split.x_train[fit_idx], split.y_train[fit_idx],
                split.x_train[selection_idx], split.y_train[selection_idx], seed,
            )
            for variant in args.variants:
                if (split.name, seed, variant) in complete:
                    print(f"{variant} {split.name} seed={seed}: already complete", flush=True)
                    continue
                seed_all(seed)
                cfg = VARIANTS[variant]
                model = MaskTopoLoc(
                    split.n_aps,
                    split.n_floors,
                    split.n_buildings,
                    use_mask=cfg.get("use_mask", True),
                    use_decoder=cfg.get("use_decoder", True),
                )
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
                    augmentation_weight=cfg.get("augmentation_weight", 0.35),
                    consistency_weight=cfg.get("consistency_weight", 0.12),
                    reconstruction_weight=cfg.get("reconstruction_weight", 0.03),
                    corruption_drop=cfg.get("corruption_drop", 0.18),
                    corruption_noise_db=cfg.get("corruption_noise_db", 2.0),
                    floor_weight=cfg.get("floor_weight", 0.10),
                    building_weight=cfg.get("building_weight", 0.10),
                )
                fit_rep = neural_predict(fit.model, split.x_train[fit_idx], device, fit.coord_mean, fit.coord_std)
                selection_rep = neural_predict(
                    fit.model, split.x_train[selection_idx], device, fit.coord_mean, fit.coord_std
                )
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

                scenario_predictions: dict[str, np.ndarray] = {}
                scenario_direct: dict[str, np.ndarray] = {}
                for scenario, x_query in shifted_inputs(split.x_test, split.x_train[fit_idx], seed).items():
                    rep = neural_predict(fit.model, x_query, device, fit.coord_mean, fit.coord_std)
                    prediction, _ = fused_graph_predict(
                        fusion,
                        pca_pack[0],
                        split.x_train[fit_idx],
                        split.y_train[fit_idx],
                        fit_rep["embedding"],
                        x_query,
                        rep["embedding"],
                        rep["coord"],
                    )
                    scenario_predictions[scenario] = prediction
                    scenario_direct[scenario] = rep["coord"]
                    for estimator, pred in (("fused", prediction), ("direct", rep["coord"])):
                        rows.append({
                            "dataset": split.name,
                            "seed": seed,
                            "method": variant,
                            "estimator": estimator,
                            "scenario": scenario,
                            **localization_metrics(split.y_test, pred),
                            "parameters": sum(p.numel() for p in fit.model.parameters()),
                            "train_seconds": fit.train_seconds,
                            "best_epoch": fit.best_epoch,
                            "trained_epochs": len(fit.history),
                        })
                selections.append({
                    "dataset": split.name,
                    "seed": seed,
                    "method": variant,
                    "gamma": fusion[1],
                    "k": fusion[2],
                    "power": fusion[3],
                    "blend": repr(fusion[4]),
                    **{f"val_{k}": v for k, v in fusion[5].items()},
                })

                variant_key = variant.lower().replace("-", "_")
                model_dir = model_root / variant_key
                history_dir = history_root / variant_key
                prediction_dir = prediction_root / variant_key
                for directory in (model_dir, history_dir, prediction_dir):
                    directory.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": fit.model.state_dict(),
                    "coord_mean": fit.coord_mean,
                    "coord_std": fit.coord_std,
                    "best_epoch": fit.best_epoch,
                    "config": cfg,
                    "n_aps": split.n_aps,
                    "n_floors": split.n_floors,
                    "n_buildings": split.n_buildings,
                }, model_dir / f"{key}_seed{seed}.pt")
                joblib.dump((pca_pack[0], fusion), model_dir / f"{key}_fusion_seed{seed}.joblib")
                pd.DataFrame(fit.history).to_csv(history_dir / f"{key}_seed{seed}.csv", index=False)
                np.savez_compressed(
                    prediction_dir / f"{key}_seed{seed}.npz",
                    y_true=split.y_test,
                    **{f"fused__{name}": value for name, value in scenario_predictions.items()},
                    **{f"direct__{name}": value for name, value in scenario_direct.items()},
                )
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                pd.DataFrame(selections).to_csv(selection_path, index=False)
                clean = localization_metrics(split.y_test, scenario_predictions["clean"])
                print(
                    f"{variant} {split.name} seed={seed}: mean={clean['mean_m']:.3f} "
                    f"p90={clean['p90_m']:.3f} epoch={fit.best_epoch}/{len(fit.history)}",
                    flush=True,
                )

    manifest = {
        "variants": {name: cfg for name, cfg in VARIANTS.items() if name in args.variants},
        "datasets": args.datasets,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection": "18% spatial reference-point groups",
        "structured_shifts": [
            "top20_outage", "persistent_bias_8db", "device_offset_-6db", "top20_outage_plus_bias4db"
        ],
        "test_labels_used_for_selection": False,
    }
    (out / "component_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
