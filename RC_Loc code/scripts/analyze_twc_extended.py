from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indoorloc.twc_data import load_twc_datasets  # noqa: E402

REV = ROOT / "results" / "twc_revision"
EXT = ROOT / "results" / "twc_extended"
FIG = EXT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 9.0, "legend.fontsize": 7.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

DATASETS = ["uji", "uts", "tampere", "uji_library"]
DATA_NAMES = {
    "uji": "UJIIndoorLoc", "uts": "UTSIndoorLoc", "tampere": "Tampere",
    "uji_library": "UJI-Library-25M",
}
SHORT = {"uji": "UJI", "uts": "UTS", "tampere": "Tampere", "uji_library": "UJI-25M"}
SEEDS = [11, 22, 33, 44, 55]
METHODS = [
    "PCA-WKNN", "ExtraTrees", "DNNBN-reimpl", "CNN-RSS-reimpl",
    "AaT-base-reimpl", "RC-Direct", "RC-Loc",
]
COLORS = {
    "PCA-WKNN": "#7f7f7f", "ExtraTrees": "#9467bd", "DNNBN-reimpl": "#8c564b",
    "CNN-RSS-reimpl": "#e377c2", "AaT-base-reimpl": "#d99032",
    "RC-Direct": "#64a86b", "RC-Loc": "#2878b5",
}


def prediction_path(method: str, dataset: str, seed: int) -> Path:
    if method == "PCA-WKNN":
        return EXT / "predictions" / "classical" / f"pca_{dataset}_seed{seed}.npz"
    if method == "ExtraTrees":
        return EXT / "predictions" / "classical" / f"extratrees_{dataset}_seed{seed}.npz"
    if method == "AaT-base-reimpl":
        return REV / "predictions" / "aat_base" / f"{dataset}_seed{seed}.npz"
    if method in {"RC-Loc", "RC-Direct"}:
        return REV / "predictions" / "rc_final" / f"{dataset}_seed{seed}.npz"
    key = method.lower().replace("-", "_")
    return EXT / "predictions" / "modern_baselines" / key / f"{dataset}_seed{seed}.npz"


def load_prediction(method: str, dataset: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = prediction_path(method, dataset, seed)
    z = np.load(path, allow_pickle=True)
    pred_key = "direct" if method == "RC-Direct" else "prediction"
    return np.asarray(z["y_true"]), np.asarray(z[pred_key]), {k: np.asarray(z[k]) for k in z.files}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    error = np.linalg.norm(residual, axis=1)
    p90, p95 = np.quantile(error, [0.90, 0.95])
    return {
        "mean_m": error.mean(), "median_m": np.median(error),
        "rmse_m": np.sqrt(np.mean(error ** 2)), "p75_m": np.quantile(error, 0.75),
        "p90_m": p90, "p95_m": p95, "p99_m": np.quantile(error, 0.99),
        "cvar90_m": error[error >= p90].mean(), "cvar95_m": error[error >= p95].mean(),
        "max_m": error.max(), "mae_x_m": np.abs(residual[:, 0]).mean(),
        "mae_y_m": np.abs(residual[:, 1]).mean(),
        "within_1m": np.mean(error <= 1), "within_3m": np.mean(error <= 3),
        "within_5m": np.mean(error <= 5), "within_10m": np.mean(error <= 10),
    }


def collect_clean() -> pd.DataFrame:
    rows = []
    for dataset, seed, method in itertools.product(DATASETS, SEEDS, METHODS):
        path = prediction_path(method, dataset, seed)
        if not path.exists():
            continue
        y, pred, extra = load_prediction(method, dataset, seed)
        row = {"dataset": DATA_NAMES[dataset], "dataset_key": dataset, "seed": seed, "method": method, **metrics(y, pred)}
        if "floor_true" in extra and "floor_prediction" in extra:
            valid = extra["floor_true"] >= 0
            row["floor_accuracy"] = np.mean(extra["floor_true"][valid] == extra["floor_prediction"][valid]) if valid.any() else np.nan
        if "building_true" in extra and "building_prediction" in extra:
            valid = extra["building_true"] >= 0
            row["building_accuracy"] = np.mean(extra["building_true"][valid] == extra["building_prediction"][valid]) if valid.any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(EXT / "extended_clean_seed_metrics.csv", index=False)
    summary = out.groupby(["dataset", "method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in [
            "mean_m", "median_m", "rmse_m", "p75_m", "p90_m", "p95_m", "p99_m",
            "cvar90_m", "cvar95_m", "within_1m", "within_3m", "within_5m", "within_10m",
        ]},
        **{f"{m}_std": (m, "std") for m in ["mean_m", "median_m", "p90_m", "p95_m", "cvar95_m"]},
    )
    summary.to_csv(EXT / "extended_clean_summary.csv", index=False)
    return out


def plot_multimetric_rank(clean: pd.DataFrame) -> None:
    lower = ["mean_m", "median_m", "rmse_m", "p90_m", "p95_m", "p99_m", "cvar95_m"]
    upper = ["within_3m", "within_5m", "within_10m"]
    averaged = clean.groupby(["dataset", "method"], as_index=False)[lower + upper].mean()
    rank_rows = []
    for dataset, g in averaged.groupby("dataset"):
        for metric in lower:
            rank = g[metric].rank(method="average", ascending=True)
            for method, value in zip(g.method, rank):
                rank_rows.append((dataset, method, metric, value))
        for metric in upper:
            rank = g[metric].rank(method="average", ascending=False)
            for method, value in zip(g.method, rank):
                rank_rows.append((dataset, method, metric, value))
    ranks = pd.DataFrame(rank_rows, columns=["dataset", "method", "metric", "rank"])
    table = (
        ranks.groupby(["method", "metric"])["rank"]
        .mean()
        .unstack()
        .reindex(METHODS)
        .dropna(how="all")
    )
    table["mean_rank"] = table.mean(axis=1)
    table.to_csv(EXT / "multimetric_average_ranks.csv")
    shown = table.drop(columns="mean_rank")
    fig, ax = plt.subplots(figsize=(7.15, 3.15))
    im = ax.imshow(shown.to_numpy(), aspect="auto", cmap="YlGnBu_r", vmin=1, vmax=max(3, np.nanmax(shown.to_numpy())))
    ax.set_xticks(range(len(shown.columns)), [c.replace("_m", "").replace("within_", "≤") for c in shown.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(shown.index)), shown.index)
    for i in range(len(shown.index)):
        for j in range(len(shown.columns)):
            if np.isfinite(shown.iloc[i, j]):
                ax.text(j, i, f"{shown.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.8)
    ax.set_title("Average rank across four official tests (lower is better)")
    fig.colorbar(im, ax=ax, label="rank", fraction=0.025, pad=0.02)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "extended_multimetric_rank.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(clean: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.3))
    selected = [m for m in METHODS if m in clean.method.unique()]
    for ax, dataset in zip(axes.flat, DATASETS):
        all_errors = {}
        cap = 0
        for method in selected:
            seed_errors = []
            for seed in SEEDS:
                if not prediction_path(method, dataset, seed).exists():
                    continue
                y, pred, _ = load_prediction(method, dataset, seed)
                seed_errors.append(np.linalg.norm(pred - y, axis=1))
            if seed_errors:
                all_errors[method] = seed_errors
                cap = max(cap, int(np.ceil(np.mean([np.quantile(e, 0.97) for e in seed_errors]))))
        grid = np.linspace(0, max(cap, 1), 250)
        for method, seed_errors in all_errors.items():
            curves = np.asarray([[np.mean(e <= threshold) for threshold in grid] for e in seed_errors])
            ax.plot(grid, curves.mean(axis=0), color=COLORS[method], label=method, linewidth=1.25)
            ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=COLORS[method], alpha=0.08)
        ax.set_title(SHORT[dataset]); ax.set_xlabel("Horizontal error (m)"); ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0, 1.01); ax.grid(linestyle=":", linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.55)
    fig.savefig(FIG / "extended_error_ecdf.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_label_overlays() -> None:
    methods = ["PCA-WKNN", "AaT-base-reimpl", "RC-Loc"]
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.9))
    for row, dataset in enumerate(["uts", "tampere"]):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            y, pred, _ = load_prediction(method, dataset, 33)
            ax.plot(y[:, 0], y[:, 1], color="#999999", linewidth=0.75, alpha=0.75, label="label")
            ax.plot(pred[:, 0], pred[:, 1], color=COLORS[method], linewidth=0.75, alpha=0.78, label="prediction")
            step = max(1, len(y) // 45)
            for idx in range(0, len(y), step):
                ax.plot([y[idx, 0], pred[idx, 0]], [y[idx, 1], pred[idx, 1]], color="#333333", alpha=0.22, linewidth=0.35)
            err = np.linalg.norm(pred - y, axis=1)
            ax.set_title(f"{SHORT[dataset]}: {method}\nmedian={np.median(err):.2f} m, P90={np.quantile(err,.9):.2f} m")
            ax.set_aspect("equal", adjustable="datalim"); ax.grid(linestyle=":", linewidth=0.4)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.5)
    fig.savefig(FIG / "extended_label_prediction_overlays.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_improvement() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0))
    for ax, dataset in zip(axes.flat, DATASETS):
        differences = []
        y_ref = None
        for seed in SEEDS:
            if not prediction_path("PCA-WKNN", dataset, seed).exists():
                continue
            y, pca, _ = load_prediction("PCA-WKNN", dataset, seed)
            _, rc, _ = load_prediction("RC-Loc", dataset, seed)
            y_ref = y
            differences.append(np.linalg.norm(pca - y, axis=1) - np.linalg.norm(rc - y, axis=1))
        if not differences:
            continue
        delta = np.mean(differences, axis=0)
        limit = np.quantile(np.abs(delta), 0.95)
        sc = ax.scatter(y_ref[:, 0], y_ref[:, 1], c=np.clip(delta, -limit, limit), s=8, cmap="RdBu", vmin=-limit, vmax=limit, alpha=0.75)
        ax.set_title(f"{SHORT[dataset]}: PCA error - RC-Loc error")
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(sc, ax=ax, label="improvement (m)", fraction=0.04, pad=0.02)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "extended_spatial_error_improvement.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_visibility_strata(dataset_map) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.9))
    methods = ["PCA-WKNN", "AaT-base-reimpl", "RC-Loc"]
    for ax, dataset in zip(axes.flat, DATASETS):
        visible = np.sum(dataset_map[dataset].x_test > 0, axis=1)
        edges = np.unique(np.quantile(visible, np.linspace(0, 1, 6)).astype(int))
        if len(edges) < 3:
            edges = np.unique(np.linspace(visible.min(), visible.max() + 1, 4).astype(int))
        centers = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            centers.append((lo + hi) / 2)
        for method in methods:
            curves = []
            for seed in SEEDS:
                y, pred, _ = load_prediction(method, dataset, seed)
                error = np.linalg.norm(pred - y, axis=1)
                values = []
                for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                    mask = (visible >= lo) & (visible <= hi if i == len(edges) - 2 else visible < hi)
                    values.append(np.quantile(error[mask], 0.90) if mask.any() else np.nan)
                curves.append(values)
            curves = np.asarray(curves)
            ax.plot(centers, np.nanmean(curves, axis=0), marker="o", color=COLORS[method], label=method)
            ax.fill_between(centers, np.nanmin(curves, axis=0), np.nanmax(curves, axis=0), color=COLORS[method], alpha=0.10)
        ax.set_title(SHORT[dataset]); ax.set_xlabel("Visible AP count"); ax.set_ylabel("P90 error (m)")
        ax.grid(linestyle=":", linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.5)
    fig.savefig(FIG / "extended_visible_ap_strata.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_long_term(dataset_map) -> None:
    methods = ["PCA-WKNN", "AaT-base-reimpl", "RC-Loc"]
    domain = dataset_map["uji_library"].test_domain.astype(str)
    months = sorted(np.unique(domain), key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8))
    for ax, stat in zip(axes, ["median", "p90"]):
        for method in methods:
            curves = []
            for seed in SEEDS:
                y, pred, _ = load_prediction(method, "uji_library", seed)
                error = np.linalg.norm(pred - y, axis=1)
                curves.append([
                    np.median(error[domain == month]) if stat == "median" else np.quantile(error[domain == month], 0.90)
                    for month in months
                ])
            curves = np.asarray(curves)
            x = np.arange(2, 2 + len(months))
            ax.plot(x, curves.mean(axis=0), color=COLORS[method], label=method)
            ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=COLORS[method], alpha=0.10)
        ax.set_title(f"Monthly {stat} error"); ax.set_xlabel("Test month"); ax.set_ylabel("Horizontal error (m)")
        ax.grid(linestyle=":", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.91), pad=0.5)
    fig.savefig(FIG / "extended_monthly_forward_time.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_calibration() -> None:
    q_grid = np.arange(0.50, 1.00, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    rows = []
    for dataset in DATASETS:
        coverage_curves, radius_curves = [], []
        for seed in [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]:
            path = REV / "predictions" / "rc_independent_calibration" / f"{dataset}_seed{seed}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            scores, risk, error = z["calibration_scores"], z["risk"], z["error"]
            coverages, radii = [], []
            ordered = np.sort(scores)
            for q in q_grid:
                rank = min(len(ordered), int(np.ceil((len(ordered) + 1) * q)))
                scale = ordered[rank - 1]
                radius = scale * risk
                coverages.append(np.mean(error <= radius)); radii.append(np.median(radius))
                rows.append({"dataset": DATA_NAMES[dataset], "seed": seed, "nominal": q,
                             "coverage": coverages[-1], "median_radius_m": radii[-1]})
            coverage_curves.append(coverages); radius_curves.append(radii)
        coverage_curves, radius_curves = np.asarray(coverage_curves), np.asarray(radius_curves)
        axes[0].plot(q_grid, coverage_curves.mean(axis=0), marker="o", label=SHORT[dataset])
        axes[0].fill_between(q_grid, coverage_curves.min(axis=0), coverage_curves.max(axis=0), alpha=0.08)
        axes[1].plot(coverage_curves.mean(axis=0), radius_curves.mean(axis=0), marker="o", label=SHORT[dataset])
    pd.DataFrame(rows).to_csv(EXT / "calibration_reliability_metrics.csv", index=False)
    axes[0].plot([0.5, 1], [0.5, 1], "k--", linewidth=0.8); axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage"); axes[0].set_title("Reliability")
    axes[1].set_xlabel("Empirical coverage"); axes[1].set_ylabel("Median radius (m)")
    axes[1].set_title("Coverage-radius efficiency")
    for ax in axes: ax.grid(linestyle=":", linewidth=0.5)
    axes[0].legend(frameon=False, ncol=2); axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "extended_calibration_reliability.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selective_risk() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.8))
    retain = np.arange(0.1, 1.01, 0.1)
    rows = []
    for ax, dataset in zip(axes.flat, DATASETS):
        curves = []
        for seed in SEEDS:
            y, pred, extra = load_prediction("RC-Loc", dataset, seed)
            error, risk = np.linalg.norm(pred - y, axis=1), extra["risk"]
            order = np.argsort(risk)
            values = []
            for fraction in retain:
                selected = order[:max(1, int(np.ceil(fraction * len(order))))]
                values.append(np.quantile(error[selected], 0.90))
                rows.append({"dataset": DATA_NAMES[dataset], "seed": seed, "retain_fraction": fraction,
                             "p90_m": values[-1]})
            curves.append(values)
        curves = np.asarray(curves)
        ax.plot(100 * retain, curves.mean(axis=0), marker="o", color=COLORS["RC-Loc"])
        ax.fill_between(100 * retain, curves.min(axis=0), curves.max(axis=0), color=COLORS["RC-Loc"], alpha=0.12)
        ax.set_title(SHORT[dataset]); ax.set_xlabel("Retained queries (%)"); ax.set_ylabel("Retained P90 (m)")
        ax.grid(linestyle=":", linewidth=0.5)
    pd.DataFrame(rows).to_csv(EXT / "selective_risk_metrics.csv", index=False)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "extended_selective_risk.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ood_surface() -> None:
    path = EXT / "ood_response_surface_metrics.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    methods = ["PCA-WKNN", "AaT-base-reimpl", "RC-Loc"]
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.5))
    for row, dataset in enumerate(["UJIIndoorLoc", "UTSIndoorLoc"]):
        for col, method in enumerate(methods):
            g = d[(d.dataset == dataset) & (d.method == method)].groupby(
                ["bias_sd_db", "outage_fraction"]
            ).p90_m.mean().unstack().sort_index().sort_index(axis=1)
            im = axes[row, col].imshow(g.to_numpy(), origin="lower", aspect="auto", cmap="magma")
            axes[row, col].set_xticks(range(len(g.columns)), [f"{100*x:.0f}%" for x in g.columns])
            axes[row, col].set_yticks(range(len(g.index)), [f"{x:.0f}" for x in g.index])
            axes[row, col].set_title(f"{dataset.replace('IndoorLoc','')}: {method}")
            axes[row, col].set_xlabel("Top-frequency AP outage"); axes[row, col].set_ylabel("AP bias SD (dB)")
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    axes[row, col].text(j, i, f"{g.iloc[i,j]:.1f}", color="white", ha="center", va="center", fontsize=6.5)
            fig.colorbar(im, ax=axes[row, col], fraction=0.04, pad=0.02, label="P90 (m)")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "extended_ood_response_surface.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation() -> None:
    path = EXT / "component_ablation_metrics.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    d = d[d.estimator == "fused"]
    full = pd.read_csv(REV / "rc_final_metrics.csv")
    full = full[(full.method == "RC-Loc") & (full.scenario == "clean")]
    rows = []
    for dataset in ["UJIIndoorLoc", "UTSIndoorLoc"]:
        for seed in SEEDS:
            ref = full[(full.dataset == dataset) & (full.seed == seed)].iloc[0]
            rows.append({"dataset": dataset, "seed": seed, "method": "RC-Loc", "scenario": "clean",
                         "p90_m": ref.p90_m, "mean_m": ref.mean_m})
    all_data = pd.concat([d[["dataset", "seed", "method", "scenario", "p90_m", "mean_m"]], pd.DataFrame(rows)])
    clean = all_data[all_data.scenario == "clean"]
    methods = ["RC-Loc"] + sorted(m for m in clean.method.unique() if m != "RC-Loc")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), sharey=True)
    for ax, dataset in zip(axes, ["UJIIndoorLoc", "UTSIndoorLoc"]):
        means = clean[clean.dataset == dataset].groupby("method").p90_m.agg(["mean", "std"]).reindex(methods)
        y = np.arange(len(methods))
        ax.errorbar(means["mean"], y, xerr=means["std"], fmt="o", color="#2878b5", capsize=2)
        ax.set_yticks(y, methods); ax.invert_yaxis(); ax.set_title(dataset.replace("IndoorLoc", ""))
        ax.set_xlabel("Clean P90 (m)"); ax.grid(axis="x", linestyle=":", linewidth=0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "extended_component_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    all_data.to_csv(EXT / "component_ablation_combined.csv", index=False)


def plot_latency() -> None:
    path = EXT / "strict_end_to_end_latency.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    d = d[(d.execution == "cuda") & (d.batch_size == 1) & (d.dataset == "UJIIndoorLoc")]
    clean_path = EXT / "extended_clean_summary.csv"
    if not clean_path.exists():
        return
    clean = pd.read_csv(clean_path)
    clean = clean[clean.dataset == "UJIIndoorLoc"][["method", "p90_m_mean"]]
    g = d.merge(clean, on="method")
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for _, row in g.iterrows():
        ax.scatter(row.latency_median_ms_per_sample, row.p90_m_mean,
                   s=30 + 80 * row.parameters / max(g.parameters.max(), 1), color=COLORS.get(row.method, "#555555"))
        ax.annotate(row.method.replace("-reimpl", ""), (row.latency_median_ms_per_sample, row.p90_m_mean),
                    xytext=(3, 3), textcoords="offset points", fontsize=6.7)
    ax.set_xscale("log"); ax.set_xlabel("End-to-end GPU batch-1 latency (ms/sample)")
    ax.set_ylabel("UJI P90 error (m)"); ax.grid(linestyle=":", linewidth=0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "extended_accuracy_latency_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    clean = collect_clean()
    dataset_map = load_twc_datasets(ROOT)
    plot_multimetric_rank(clean)
    plot_ecdf(clean)
    plot_label_overlays()
    plot_spatial_improvement()
    plot_visibility_strata(dataset_map)
    plot_long_term(dataset_map)
    plot_calibration()
    plot_selective_risk()
    plot_ood_surface()
    plot_ablation()
    plot_latency()
    print(f"Generated extended summaries for {len(clean)} seed-method-dataset rows and figures in {FIG}")


if __name__ == "__main__":
    main()
