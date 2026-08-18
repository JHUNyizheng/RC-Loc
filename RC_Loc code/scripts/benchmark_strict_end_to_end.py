from __future__ import annotations

import argparse
import json
import platform
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

from indoorloc.twc_data import load_twc_datasets  # noqa: E402
from indoorloc.twc_models import (  # noqa: E402
    Anchor2VecTransformer,
    ConvRSSLoc,
    DNNBNLoc,
    MaskTopoLoc,
)
from indoorloc.twc_training import fused_graph_predict, neural_predict, weighted_neighbor_predict  # noqa: E402


def neural_from_checkpoint(method: str, record: dict):
    args = (record["n_aps"], record["n_floors"], record["n_buildings"])
    if method == "AaT-base-reimpl":
        return Anchor2VecTransformer(*args)
    if method == "DNNBN-reimpl":
        return DNNBNLoc(*args)
    if method == "CNN-RSS-reimpl":
        return ConvRSSLoc(*args)
    if method in {"RC-Direct", "RC-Loc"}:
        return MaskTopoLoc(*args)
    raise KeyError(method)


def percentile_row(values: np.ndarray) -> dict[str, float]:
    return {
        "latency_median_ms_per_sample": float(np.median(values)),
        "latency_p05_ms_per_sample": float(np.quantile(values, 0.05)),
        "latency_p95_ms_per_sample": float(np.quantile(values, 0.95)),
        "throughput_samples_per_s": float(1000.0 / np.median(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uji", "uts"])
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    extended = ROOT / "results" / "twc_extended"
    revision = ROOT / "results" / "twc_revision"
    out_path = extended / "strict_end_to_end_latency.csv"
    datasets = load_twc_datasets(ROOT)
    rows = []
    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    for key in args.datasets:
        split = datasets[key]
        fit_idx, _ = next(
            GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=args.seed).split(
                split.x_train, groups=split.group_train
            )
        )
        for execution in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            device = torch.device(execution)
            for method in [
                "PCA-WKNN", "ExtraTrees", "DNNBN-reimpl", "CNN-RSS-reimpl",
                "AaT-base-reimpl", "RC-Direct", "RC-Loc",
            ]:
                if method in {"PCA-WKNN", "ExtraTrees"} and execution == "cuda":
                    continue
                parameters = 0
                model_size = 0
                if method == "PCA-WKNN":
                    model_path = extended / "models" / "classical" / f"pca_{key}_seed{args.seed}.joblib"
                    if not model_path.exists():
                        continue
                    best, pack, fit_idx = joblib.load(model_path)
                    _, _, _, k_neighbors, power, _ = best
                    pca, nn, _ = pack

                    def predict(batch):
                        return weighted_neighbor_predict(
                            nn, split.y_train[fit_idx], pca.transform(batch), k_neighbors, power
                        )[0]

                elif method == "ExtraTrees":
                    model_path = extended / "models" / "classical" / f"extratrees_{key}_seed{args.seed}.joblib"
                    if not model_path.exists():
                        continue
                    trees, _ = joblib.load(model_path)

                    def predict(batch):
                        return trees.predict(batch)

                else:
                    if method in {"RC-Direct", "RC-Loc"}:
                        model_path = revision / "models" / "rc_final" / f"{key}_seed{args.seed}.pt"
                    elif method == "AaT-base-reimpl":
                        model_path = revision / "models" / "aat_base" / f"{key}_seed{args.seed}.pt"
                    else:
                        model_key = method.lower().replace("-", "_")
                        model_path = extended / "models" / "modern_baselines" / model_key / f"{key}_seed{args.seed}.pt"
                    if not model_path.exists():
                        continue
                    record = torch.load(model_path, map_location=device)
                    model = neural_from_checkpoint(method, record).to(device)
                    model.load_state_dict(record["state_dict"])
                    model.eval()
                    parameters = sum(p.numel() for p in model.parameters())
                    if method == "RC-Loc":
                        pca, fusion = joblib.load(
                            revision / "models" / "rc_final" / f"{key}_fusion_seed{args.seed}.joblib"
                        )

                        def predict(batch):
                            rep = neural_predict(model, batch, device, record["coord_mean"], record["coord_std"], batch_size=len(batch))
                            return fused_graph_predict(
                                fusion, pca, split.x_train[fit_idx], split.y_train[fit_idx],
                                np.empty((len(fit_idx), 0)), batch, rep["embedding"], rep["coord"]
                            )[0]

                    else:

                        def predict(batch):
                            return neural_predict(
                                model, batch, device, record["coord_mean"], record["coord_std"], batch_size=len(batch)
                            )["coord"]

                model_size = model_path.stat().st_size
                for batch_size in (1, 32, 256):
                    batch = np.asarray(
                        split.x_test[np.arange(batch_size) % len(split.x_test)], dtype=np.float32
                    )
                    for _ in range(args.warmup):
                        predict(batch)
                    if device.type == "cuda":
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
                    repeats = 300 if batch_size == 1 else (150 if batch_size == 32 else 80)
                    samples = []
                    for _ in range(repeats):
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        predict(batch)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        samples.append((time.perf_counter_ns() - start) / 1e6 / batch_size)
                    peak_gpu = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                    rows.append({
                        "dataset": split.name, "seed": args.seed, "method": method,
                        "execution": execution, "batch_size": batch_size,
                        "warmup_repeats": args.warmup, "timed_repeats": repeats,
                        **percentile_row(np.asarray(samples)),
                        "parameters": parameters, "serialized_model_bytes": model_size,
                        "peak_gpu_memory_bytes": peak_gpu,
                        "boundary": "in-memory normalized RSS to final coordinate; exact retrieval included for RC-Loc",
                    })
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                    print(
                        f"latency {split.name} {method} {execution} b={batch_size}: "
                        f"{rows[-1]['latency_median_ms_per_sample']:.4f} ms/sample",
                        flush=True,
                    )
    (extended / "strict_latency_environment.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
