from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import localization_metrics
from .twc_models import pairwise_topology_loss


@dataclass
class NeuralFit:
    model: nn.Module
    coord_mean: np.ndarray
    coord_std: np.ndarray
    best_epoch: int
    pretrain_epochs: int
    train_seconds: float
    history: list[dict[str, float]]


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(x, y, floor, building, indices, batch_size, shuffle):
    index = np.asarray(indices, dtype=np.int64)
    ds = TensorDataset(
        torch.from_numpy(np.asarray(x[index], dtype=np.float32)),
        torch.from_numpy(np.asarray(y[index], dtype=np.float32)),
        torch.from_numpy(np.asarray(floor[index], dtype=np.int64)),
        torch.from_numpy(np.asarray(building[index], dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def corrupt(x: torch.Tensor, drop_prob: float, noise_db: float) -> torch.Tensor:
    out = x.clone()
    active = out > 0
    if drop_prob > 0:
        out = out.masked_fill(active & (torch.rand_like(out) < drop_prob), 0.0)
    if noise_db > 0:
        out = torch.where(out > 0, (out + torch.randn_like(out) * (noise_db / 110.0)).clamp(1e-4, 1.0), out)
    return out


@torch.no_grad()
def neural_predict(model: nn.Module, x: np.ndarray, device: torch.device, coord_mean, coord_std,
                   batch_size: int = 1024) -> dict[str, np.ndarray]:
    model.eval()
    coords, embeddings, floors, buildings = [], [], [], []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(device)
        out = model(xb)
        coords.append(out["coord"].cpu().numpy())
        embeddings.append(out["embedding"].cpu().numpy())
        floors.append(out["floor"].argmax(1).cpu().numpy())
        buildings.append(out["building"].argmax(1).cpu().numpy())
    return {
        "coord": np.concatenate(coords) * coord_std + coord_mean,
        "embedding": np.concatenate(embeddings),
        "floor": np.concatenate(floors),
        "building": np.concatenate(buildings),
    }


def train_neural(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    floor: np.ndarray,
    building: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    *,
    proposed: bool,
    max_epochs: int = 200,
    pretrain_epochs: int = 50,
    patience: int = 30,
    batch_size: int = 256,
    lr: float = 7e-4,
    weight_decay: float = 2e-4,
    topology_weight: float = 0.12,
    consistency_weight: float = 0.12,
    augmentation_weight: float = 0.35,
    reconstruction_weight: float = 0.03,
    coordinate_loss: str = "smooth_l1",
    corruption_drop: float = 0.18,
    corruption_noise_db: float = 2.0,
    coordinate_consistency_scale: float = 1.0,
    embedding_consistency_scale: float = 0.25,
    floor_weight: float = 0.10,
    building_weight: float = 0.10,
) -> NeuralFit:
    coord_mean = y[train_idx].mean(axis=0).astype(np.float32)
    coord_std = np.maximum(y[train_idx].std(axis=0), 1.0).astype(np.float32)
    y_z = (y - coord_mean) / coord_std
    train_loader = _loader(x, y_z, floor, building, train_idx, batch_size, True)
    model.to(device)
    start_time = time.perf_counter()

    if proposed and pretrain_epochs > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
        model.train()
        for pre_epoch in range(1, pretrain_epochs + 1):
            for xb, _, _, _ in train_loader:
                xb = xb.to(device)
                observed = xb > 0
                masked = observed & (torch.rand_like(xb) < 0.35)
                x_in = xb.masked_fill(masked, 0.0)
                out = model(x_in)
                if masked.any():
                    loss = F.mse_loss(out["reconstruction"][masked], xb[masked])
                else:
                    loss = F.mse_loss(out["reconstruction"], xb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            if pre_epoch % 25 == 0 or pre_epoch == pretrain_epochs:
                print(f"      pretrain epoch {pre_epoch}/{pretrain_epochs}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch, stale = 0, 0
    history = []
    n_floors = int(max(floor.max(initial=0) + 1, 1))
    n_buildings = int(max(building.max(initial=0) + 1, 1))

    if coordinate_loss == "smooth_l1":
        coord_loss_fn = F.smooth_l1_loss
    elif coordinate_loss == "l1":
        coord_loss_fn = F.l1_loss
    else:
        raise ValueError(f"Unsupported coordinate_loss={coordinate_loss!r}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb, fb, bb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            fb, bb = fb.to(device), bb.to(device)
            if proposed:
                xa = corrupt(xb, corruption_drop, corruption_noise_db)
                clean = model(xb)
                aug = model(xa)
                loss = coord_loss_fn(clean["coord"], yb)
                if augmentation_weight > 0:
                    loss = loss + augmentation_weight * coord_loss_fn(aug["coord"], yb)
                if n_floors > 1:
                    loss = loss + floor_weight * F.cross_entropy(clean["floor"], fb)
                if n_buildings > 1:
                    loss = loss + building_weight * F.cross_entropy(clean["building"], bb)
                loss = loss + topology_weight * pairwise_topology_loss(clean["embedding"], yb)
                if consistency_weight > 0:
                    loss = loss + consistency_weight * (
                        coordinate_consistency_scale * F.smooth_l1_loss(clean["coord"], aug["coord"])
                        + embedding_consistency_scale
                        * (1 - F.cosine_similarity(clean["embedding"], aug["embedding"], dim=1).mean())
                    )
                active = xb > 0
                if reconstruction_weight > 0 and active.any() and "reconstruction" in clean:
                    loss = loss + reconstruction_weight * F.mse_loss(clean["reconstruction"][active], xb[active])
            else:
                out = model(xb)
                loss = coord_loss_fn(out["coord"], yb)
                if n_floors > 1:
                    loss = loss + floor_weight * F.cross_entropy(out["floor"], fb)
                if n_buildings > 1:
                    loss = loss + building_weight * F.cross_entropy(out["building"], bb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(xb)
            seen += len(xb)
        scheduler.step()

        val = neural_predict(model, x[val_idx], device, coord_mean, coord_std)
        err = np.linalg.norm(val["coord"] - y[val_idx], axis=1)
        median, p90 = float(np.median(err)), float(np.quantile(err, 0.9))
        score = median + 0.20 * p90
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_median_m": median,
                        "val_p90_m": p90, "selection_score": score})
        if epoch % 25 == 0 or epoch == 1:
            print(f"      supervised epoch {epoch}/{max_epochs}: val median={median:.3f} p90={p90:.3f}", flush=True)
        if score < best_score - 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if epoch >= 50 and stale >= patience:
                break
    model.load_state_dict(best_state)
    return NeuralFit(model, coord_mean, coord_std, best_epoch, pretrain_epochs if proposed else 0,
                     time.perf_counter() - start_time, history)


def weighted_neighbor_predict(nn: NearestNeighbors, y_train: np.ndarray, z: np.ndarray, k: int,
                              power: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance, index = nn.kneighbors(z, n_neighbors=k)
    weight = 1.0 / np.maximum(distance, 1e-5) ** power
    weight /= weight.sum(axis=1, keepdims=True)
    neighbors = y_train[index]
    pred = np.sum(neighbors * weight[..., None], axis=1)
    dispersion = np.sqrt(np.sum(weight * np.sum((neighbors - pred[:, None, :]) ** 2, axis=2), axis=1))
    mean_distance = np.sum(weight * distance, axis=1)
    return pred, dispersion, mean_distance


def select_pca_graph(x_train, y_train, x_val, y_val, seed: int):
    best, best_pack = None, None
    max_comp = min(x_train.shape[0] - 1, x_train.shape[1])
    candidates = sorted(set(min(max_comp, n) for n in [16, 32, 64, 96, 128] if min(max_comp, n) >= 2))
    for n_comp in candidates:
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
        z_train = pca.fit_transform(x_train)
        z_val = pca.transform(x_val)
        for metric in ["euclidean", "manhattan"]:
            nn = NearestNeighbors(n_neighbors=min(17, len(x_train)), metric=metric, n_jobs=-1).fit(z_train)
            for k in [3, 5, 7, 11, 17]:
                if k > len(x_train):
                    continue
                for power in [1.0, 2.0]:
                    pred, _, _ = weighted_neighbor_predict(nn, y_train, z_val, k, power)
                    m = localization_metrics(y_val, pred)
                    score = m["median_m"] + 0.20 * m["p90_m"]
                    if best is None or score < best[0]:
                        best = (score, n_comp, metric, k, power, m)
                        best_pack = (pca, nn, z_train)
    return best, best_pack


def select_fused_graph(
    pca: PCA,
    x_train: np.ndarray,
    y_train: np.ndarray,
    emb_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    emb_val: np.ndarray,
    direct_val: np.ndarray,
):
    z_train = pca.transform(x_train)
    z_val = pca.transform(x_val)
    pca_scale = np.maximum(z_train.std(axis=0, keepdims=True), 1e-4)
    emb_mean = emb_train.mean(axis=0, keepdims=True)
    emb_scale = np.maximum(emb_train.std(axis=0, keepdims=True), 1e-4)
    z_train = z_train / pca_scale
    z_val = z_val / pca_scale
    e_train = (emb_train - emb_mean) / emb_scale
    e_val = (emb_val - emb_mean) / emb_scale
    best = None
    for gamma in [0.0, 0.25, 0.5, 1.0, 2.0]:
        train_feature = np.concatenate([z_train, gamma * e_train], axis=1)
        val_feature = np.concatenate([z_val, gamma * e_val], axis=1)
        nn = NearestNeighbors(n_neighbors=min(17, len(x_train)), metric="euclidean", n_jobs=-1).fit(train_feature)
        for k in [3, 5, 7, 11, 17]:
            if k > len(x_train):
                continue
            for power in [1.0, 2.0]:
                graph, disp, dist = weighted_neighbor_predict(nn, y_train, val_feature, k, power)
                risk = disp + 0.25 * dist
                blend_specs: list[float | tuple[str, float, float]] = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]
                for base in [0.0, 0.15, 0.30]:
                    for q in [0.25, 0.50, 0.75, 0.90]:
                        blend_specs.append(("adaptive", base, float(np.quantile(risk, q))))
                for blend in blend_specs:
                    if isinstance(blend, tuple):
                        _, base, threshold = blend
                        alpha = base + (1 - base) * np.clip(risk / max(threshold, 1e-4), 0, 1)
                    else:
                        alpha = np.full(len(graph), blend)
                    pred = (1 - alpha[:, None]) * graph + alpha[:, None] * direct_val
                    m = localization_metrics(y_val, pred)
                    score = m["median_m"] + 0.20 * m["p90_m"]
                    if best is None or score < best[0]:
                        best = (score, gamma, k, power, blend, m, nn, pca_scale, emb_mean, emb_scale, disp, dist)
    return best


def fused_graph_predict(best, pca, x_train, y_train, emb_train, x_query, emb_query, direct_query):
    _, gamma, k, power, blend, _, nn, pca_scale, emb_mean, emb_scale, _, _ = best
    z_query = pca.transform(x_query) / pca_scale
    e_query = (emb_query - emb_mean) / emb_scale
    feature = np.concatenate([z_query, gamma * e_query], axis=1)
    graph, dispersion, distance = weighted_neighbor_predict(nn, y_train, feature, k, power)
    uncertainty = dispersion + 0.25 * distance
    if isinstance(blend, tuple):
        _, base, threshold = blend
        alpha = base + (1 - base) * np.clip(uncertainty / max(threshold, 1e-4), 0, 1)
    else:
        alpha = np.full(len(graph), blend)
    pred = (1 - alpha[:, None]) * graph + alpha[:, None] * direct_query
    return pred, uncertainty
