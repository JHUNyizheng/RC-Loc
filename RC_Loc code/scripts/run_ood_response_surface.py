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
from indoorloc.twc_models import Anchor2VecTransformer, MaskTopoLoc  # noqa: E402
from indoorloc.twc_training import (  # noqa: E402
    fused_graph_predict,
    neural_predict,
    select_pca_graph,
    weighted_neighbor_predict,
)


def make_shift(x: np.ndarray, fit_x: np.ndarray, outage: float, bias_sd: float, seed: int) -> tuple[np.ndarray, list[int]]:
    frequency = np.mean(fit_x > 0, axis=0)
    observed = np.flatnonzero(frequency > 0)
    ranked = observed[np.argsort(frequency[observed], kind="stable")[::-1]]
    count = int(np.ceil(outage * len(observed)))
    removed = ranked[:count]
    shifted = np.asarray(x, dtype=np.float32).copy()
    if count:
        shifted[:, removed] = 0.0
    if bias_sd:
        rng = np.random.default_rng(seed + int(outage * 10_000) + int(bias_sd * 100) + 910_001)
        delta = rng.normal(0.0, bias_sd, size=(1, x.shape[1])).astype(np.float32) / 110.0
        active = shifted > 0
        candidate = shifted + delta
        shifted[active] = np.clip(np.broadcast_to(candidate, shifted.shape)[active], 1e-4, 1.0)
    return shifted, removed.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    args = parser.parse_args()

    revision = ROOT / "results" / "twc_revision"
    out = ROOT / "results" / "twc_extended"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "ood_response_surface_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    complete = {
        (r["dataset"], int(r["seed"]), float(r["outage_fraction"]), float(r["bias_sd_db"]), r["method"])
        for r in rows
    }
    datasets = load_twc_datasets(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = []

    for key in args.datasets:
        split = datasets[key]
        for seed in args.seeds:
            fit_idx, selection_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed).split(
                    split.x_train, groups=split.group_train
                )
            )
            pca_best, pca_pack = select_pca_graph(
                split.x_train[fit_idx], split.y_train[fit_idx],
                split.x_train[selection_idx], split.y_train[selection_idx], seed,
            )
            rc_record = torch.load(revision / "models" / "rc_final" / f"{key}_seed{seed}.pt", map_location=device)
            rc = MaskTopoLoc(rc_record["n_aps"], rc_record["n_floors"], rc_record["n_buildings"]).to(device)
            rc.load_state_dict(rc_record["state_dict"]); rc.eval()
            rc_pca, fusion = joblib.load(revision / "models" / "rc_final" / f"{key}_fusion_seed{seed}.joblib")

            aat_record = torch.load(revision / "models" / "aat_base" / f"{key}_seed{seed}.pt", map_location=device)
            aat = Anchor2VecTransformer(
                aat_record["n_aps"], aat_record["n_floors"], aat_record["n_buildings"]
            ).to(device)
            aat.load_state_dict(aat_record["state_dict"]); aat.eval()

            for outage in (0.0, 0.10, 0.20, 0.30):
                for bias_sd in (0.0, 4.0, 8.0):
                    x_query, removed = make_shift(split.x_test, split.x_train[fit_idx], outage, bias_sd, seed)
                    manifest.append({
                        "dataset": split.name, "seed": seed, "outage_fraction": outage,
                        "bias_sd_db": bias_sd, "removed_ap_indices": removed,
                    })
                    key_base = (split.name, seed, outage, bias_sd)
                    if (*key_base, "PCA-WKNN") not in complete:
                        pred, _, _ = weighted_neighbor_predict(
                            pca_pack[1], split.y_train[fit_idx], pca_pack[0].transform(x_query),
                            pca_best[3], pca_best[4],
                        )
                        rows.append({
                            "dataset": split.name, "seed": seed, "outage_fraction": outage,
                            "bias_sd_db": bias_sd, "method": "PCA-WKNN",
                            **localization_metrics(split.y_test, pred),
                        })
                    if (*key_base, "AaT-base-reimpl") not in complete:
                        rep = neural_predict(aat, x_query, device, aat_record["coord_mean"], aat_record["coord_std"])
                        rows.append({
                            "dataset": split.name, "seed": seed, "outage_fraction": outage,
                            "bias_sd_db": bias_sd, "method": "AaT-base-reimpl",
                            **localization_metrics(split.y_test, rep["coord"]),
                        })
                    rc_rep = neural_predict(rc, x_query, device, rc_record["coord_mean"], rc_record["coord_std"])
                    if (*key_base, "RC-Direct") not in complete:
                        rows.append({
                            "dataset": split.name, "seed": seed, "outage_fraction": outage,
                            "bias_sd_db": bias_sd, "method": "RC-Direct",
                            **localization_metrics(split.y_test, rc_rep["coord"]),
                        })
                    if (*key_base, "RC-Loc") not in complete:
                        pred, _ = fused_graph_predict(
                            fusion, rc_pca, split.x_train[fit_idx], split.y_train[fit_idx],
                            np.empty((len(fit_idx), 0)), x_query, rc_rep["embedding"], rc_rep["coord"],
                        )
                        rows.append({
                            "dataset": split.name, "seed": seed, "outage_fraction": outage,
                            "bias_sd_db": bias_sd, "method": "RC-Loc",
                            **localization_metrics(split.y_test, pred),
                        })
                    pd.DataFrame(rows).to_csv(metrics_path, index=False)
                    print(f"surface {split.name} seed={seed} outage={outage:.2f} bias={bias_sd:.0f}", flush=True)

    (out / "ood_response_surface_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
