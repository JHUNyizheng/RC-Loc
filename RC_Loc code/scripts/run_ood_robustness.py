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


def add_db_delta(x: np.ndarray, delta_db: np.ndarray | float) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    active = out > 0
    shifted = out + np.asarray(delta_db, dtype=np.float32) / 110.0
    out[active] = np.clip(np.broadcast_to(shifted, out.shape)[active], 1e-4, 1.0)
    return out


def scenarios(x: np.ndarray, x_fit: np.ndarray, seed: int):
    frequency = np.mean(x_fit > 0, axis=0)
    observed_ids = np.flatnonzero(frequency > 0)
    ranked = observed_ids[np.argsort(frequency[observed_ids])[::-1]]
    for fraction in (0.10, 0.20, 0.30):
        count = max(1, int(np.ceil(fraction * len(observed_ids))))
        removed = ranked[:count]
        changed = x.copy()
        changed[:, removed] = 0.0
        yield f"top_ap_outage={fraction:.2f}", changed, removed.tolist()

    rng = np.random.default_rng(seed + 500_003)
    for sd in (4.0, 8.0):
        bias = rng.normal(0.0, sd, size=(1, x.shape[1])).astype(np.float32)
        yield f"persistent_ap_bias_sd={sd:.0f}dB", add_db_delta(x, bias), []
    for offset in (-6.0, 6.0):
        yield f"device_offset={offset:+.0f}dB", add_db_delta(x, offset), []

    removed = ranked[:max(1, int(np.ceil(0.20 * len(observed_ids))))]
    mixed = x.copy()
    mixed[:, removed] = 0.0
    bias = rng.normal(0.0, 4.0, size=(1, x.shape[1])).astype(np.float32)
    yield "top_ap_outage=0.20+bias_sd=4dB", add_db_delta(mixed, bias), removed.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    args = parser.parse_args()

    revision = ROOT / "results" / "twc_revision"
    output_path = revision / "ood_robustness_metrics.csv"
    revision.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(output_path).to_dict("records") if output_path.exists() else []
    complete = {(row["dataset"], int(row["seed"]), row["scenario"], row["method"]) for row in rows}
    dataset_map = load_twc_datasets(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outage_manifest: list[dict[str, object]] = []
    for key in args.datasets:
        split = dataset_map[key]
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

            mrc_record = torch.load(
                revision / "models" / "rc_final" / f"{key}_seed{seed}.pt",
                map_location=device,
            )
            mrc = MaskTopoLoc(
                mrc_record["n_aps"], mrc_record["n_floors"], mrc_record["n_buildings"]
            ).to(device)
            mrc.load_state_dict(mrc_record["state_dict"])
            mrc.eval()
            mrc_pca, fusion = joblib.load(
                revision / "models" / "rc_final" / f"{key}_fusion_seed{seed}.joblib"
            )

            aat_record = torch.load(
                revision / "models" / "aat_base" / f"{key}_seed{seed}.pt",
                map_location=device,
            )
            aat = Anchor2VecTransformer(
                aat_record["n_aps"], aat_record["n_floors"], aat_record["n_buildings"]
            ).to(device)
            aat.load_state_dict(aat_record["state_dict"])
            aat.eval()

            for scenario, x_query, removed in scenarios(split.x_test, split.x_train[fit_idx], seed):
                outage_manifest.append({
                    "dataset": split.name,
                    "seed": seed,
                    "scenario": scenario,
                    "removed_ap_indices": removed,
                })
                if (split.name, seed, scenario, "PCA-WKNN") not in complete:
                    pca_prediction, _, _ = weighted_neighbor_predict(
                        pca_pack[1],
                        split.y_train[fit_idx],
                        pca_pack[0].transform(x_query),
                        pca_best[3],
                        pca_best[4],
                    )
                    rows.append({
                        "dataset": split.name, "seed": seed, "method": "PCA-WKNN",
                        "scenario": scenario, **localization_metrics(split.y_test, pca_prediction),
                    })

                if (split.name, seed, scenario, "AaT-base-reimpl") not in complete:
                    aat_prediction = neural_predict(
                        aat, x_query, device, aat_record["coord_mean"], aat_record["coord_std"]
                    )
                    rows.append({
                        "dataset": split.name, "seed": seed, "method": "AaT-base-reimpl",
                        "scenario": scenario,
                        **localization_metrics(split.y_test, aat_prediction["coord"]),
                    })

                mrc_prediction = neural_predict(
                    mrc, x_query, device, mrc_record["coord_mean"], mrc_record["coord_std"]
                )
                if (split.name, seed, scenario, "RC-Direct") not in complete:
                    rows.append({
                        "dataset": split.name, "seed": seed, "method": "RC-Direct",
                        "scenario": scenario,
                        **localization_metrics(split.y_test, mrc_prediction["coord"]),
                    })
                if (split.name, seed, scenario, "RC-Loc") not in complete:
                    fused_prediction, _ = fused_graph_predict(
                        fusion,
                        mrc_pca,
                        split.x_train[fit_idx],
                        split.y_train[fit_idx],
                        np.empty((len(fit_idx), 0)),
                        x_query,
                        mrc_prediction["embedding"],
                        mrc_prediction["coord"],
                    )
                    rows.append({
                        "dataset": split.name, "seed": seed, "method": "RC-Loc",
                        "scenario": scenario,
                        **localization_metrics(split.y_test, fused_prediction),
                    })
                pd.DataFrame(rows).to_csv(output_path, index=False)
                print(f"OOD {split.name} seed={seed} {scenario}", flush=True)

    (revision / "ood_scenario_manifest.json").write_text(
        json.dumps(outage_manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
