from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _corrupt_tensor(x: torch.Tensor, drop_prob: float, noise_db: float) -> torch.Tensor:
    out = x.clone()
    active = out > 0
    if drop_prob > 0:
        out = out.masked_fill(active & (torch.rand_like(out) < drop_prob), 0.0)
    if noise_db > 0:
        noise = torch.randn_like(out) * (noise_db / 110.0)
        out = torch.where(out > 0, (out + noise).clamp(1e-4, 1.0), out)
    return out


def _loss(
    output: dict[str, torch.Tensor],
    y: torch.Tensor,
    floor: torch.Tensor,
    building: torch.Tensor,
    use_uncertainty: bool,
) -> torch.Tensor:
    if use_uncertainty:
        sq = (output["coord"] - y) ** 2
        coord_loss = 0.5 * torch.mean(torch.exp(-output["logvar"]) * sq + output["logvar"])
    else:
        coord_loss = F.smooth_l1_loss(output["coord"], y)
    return coord_loss + 0.15 * F.cross_entropy(output["floor"], floor) + 0.15 * F.cross_entropy(
        output["building"], building
    )


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
) -> dict[str, np.ndarray]:
    model.eval()
    coords, logvars, floors, buildings = [], [], [], []
    for x, _, _, _ in loader:
        out = model(x.to(device, non_blocking=True))
        coords.append(out["coord"].cpu().numpy())
        logvars.append(out["logvar"].cpu().numpy())
        floors.append(out["floor"].argmax(1).cpu().numpy())
        buildings.append(out["building"].argmax(1).cpu().numpy())
    coord_z = np.concatenate(coords)
    return {
        "coord": coord_z * coord_std + coord_mean,
        "logvar": np.concatenate(logvars),
        "floor": np.concatenate(floors),
        "building": np.concatenate(buildings),
    }


@torch.no_grad()
def inference_latency_ms(model: nn.Module, x: torch.Tensor, device: torch.device, repeats: int = 100) -> float:
    model.eval()
    x = x.to(device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats / len(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    epochs: int = 15,
    lr: float = 8e-4,
    weight_decay: float = 1e-4,
    drop_prob: float = 0.15,
    noise_db: float = 2.0,
    consistency_weight: float = 0.10,
    use_uncertainty: bool = True,
    patience: int = 4,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for x, y, floor, building in train_loader:
            x, y = x.to(device), y.to(device)
            floor, building = floor.to(device), building.to(device)
            view1 = _corrupt_tensor(x, drop_prob, noise_db)
            optimizer.zero_grad(set_to_none=True)
            out1 = model(view1)
            loss = _loss(out1, y, floor, building, use_uncertainty)
            if consistency_weight > 0:
                view2 = _corrupt_tensor(x, drop_prob, noise_db)
                out2 = model(view2)
                loss = loss + consistency_weight * F.mse_loss(out1["coord"], out2["coord"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(x)
            seen += len(x)
        scheduler.step()

        pred = predict_model(model, val_loader, device, coord_mean, coord_std)["coord"]
        truth = np.concatenate([batch[1].numpy() for batch in val_loader]) * coord_std + coord_mean
        val_median = float(np.median(np.linalg.norm(pred - truth, axis=1)))
        history.append({"epoch": float(epoch), "train_loss": total / seen, "val_median_m": val_median})
        if val_median < best_val - 1e-4:
            best_val = val_median
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(model=model, history=history, best_epoch=best_epoch)

