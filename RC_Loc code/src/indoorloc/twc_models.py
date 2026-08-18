from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GatedResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        return x + self.gate(z) * self.value(z)


class MaskTopoLoc(nn.Module):
    """Mask-aware radio-map encoder with reconstruction and multi-task heads."""

    def __init__(
        self,
        n_aps: int,
        n_floors: int,
        n_buildings: int,
        hidden: int = 384,
        embedding_dim: int = 96,
        blocks: int = 3,
        dropout: float = 0.12,
        use_mask: bool = True,
        use_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.use_mask = use_mask
        mask_hidden = max(64, hidden // 3)
        self.rss_stem = nn.Sequential(nn.Linear(n_aps, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.mask_stem = (
            nn.Sequential(nn.Linear(n_aps, mask_hidden), nn.LayerNorm(mask_hidden), nn.GELU())
            if use_mask else None
        )
        fuse_dim = hidden + mask_hidden if use_mask else hidden
        self.fuse = nn.Sequential(nn.Linear(fuse_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[GatedResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.embedding = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))
        self.coord_head = nn.Sequential(nn.Linear(embedding_dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.floor_head = nn.Linear(embedding_dim, max(1, n_floors))
        self.building_head = nn.Linear(embedding_dim, max(1, n_buildings))
        self.decoder = (
            nn.Sequential(nn.Linear(embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, n_aps), nn.Sigmoid())
            if use_decoder else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rss = self.rss_stem(x)
        if self.use_mask:
            mask = (x > 0).to(x.dtype)
            h = self.fuse(torch.cat([rss, self.mask_stem(mask)], dim=1))
        else:
            h = self.fuse(rss)
        h = self.blocks(h)
        emb = self.embedding(h)
        output = {
            "embedding": emb,
            "coord": self.coord_head(emb),
            "floor": self.floor_head(emb),
            "building": self.building_head(emb),
        }
        if self.decoder is not None:
            output["reconstruction"] = self.decoder(emb)
        return output


class DNNBNLoc(nn.Module):
    """Batch-normalized RSS regression baseline inspired by recent DNN pipelines."""

    def __init__(self, n_aps: int, n_floors: int, n_buildings: int, dropout: float = 0.15) -> None:
        super().__init__()
        dims = [n_aps, 768, 384, 192, 128]
        layers: list[nn.Module] = []
        for source, target in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(source, target), nn.BatchNorm1d(target), nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)
        self.coord_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))
        self.floor_head = nn.Linear(128, max(1, n_floors))
        self.building_head = nn.Linear(128, max(1, n_buildings))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.net(x)
        return {
            "embedding": embedding,
            "coord": self.coord_head(embedding),
            "floor": self.floor_head(embedding),
            "building": self.building_head(embedding),
        }


class ConvRSSLoc(nn.Module):
    """Lightweight one-dimensional CNN baseline over the fixed AP vector."""

    def __init__(self, n_aps: int, n_floors: int, n_buildings: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, padding=2), nn.BatchNorm1d(96), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8, 192), nn.GELU(), nn.Dropout(dropout))
        self.coord_head = nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 2))
        self.floor_head = nn.Linear(192, max(1, n_floors))
        self.building_head = nn.Linear(192, max(1, n_buildings))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.embedding(self.features(x[:, None, :]))
        return {
            "embedding": embedding,
            "coord": self.coord_head(embedding),
            "floor": self.floor_head(embedding),
            "building": self.building_head(embedding),
        }


class DenseLoc(nn.Module):
    """Capacity-matched direct-regression neural baseline."""

    def __init__(self, n_aps: int, n_floors: int, n_buildings: int, hidden: int = 384, dropout: float = 0.12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_aps, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.GELU(),
        )
        self.coord_head = nn.Linear(128, 2)
        self.floor_head = nn.Linear(128, max(1, n_floors))
        self.building_head = nn.Linear(128, max(1, n_buildings))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        return {"embedding": h, "coord": self.coord_head(h), "floor": self.floor_head(h), "building": self.building_head(h)}


class MatchedMLP(nn.Module):
    """Parameter-matched residual MLP without mask, reconstruction, or topology learning."""

    def __init__(
        self,
        n_aps: int,
        n_floors: int,
        n_buildings: int,
        hidden: int = 384,
        embedding_dim: int = 96,
        blocks: int = 3,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(n_aps, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.Sequential(*[GatedResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.embedding = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))
        self.coord_head = nn.Sequential(nn.Linear(embedding_dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.floor_head = nn.Linear(embedding_dim, max(1, n_floors))
        self.building_head = nn.Linear(embedding_dim, max(1, n_buildings))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.embedding(self.blocks(self.stem(x)))
        return {
            "embedding": emb,
            "coord": self.coord_head(emb),
            "floor": self.floor_head(emb),
            "building": self.building_head(emb),
        }


class Anchor2VecTransformer(nn.Module):
    """Protocol-matched base AaT-style baseline for sparse RSS vectors.

    The implementation follows the published bAaT ingredients that can be
    reproduced unambiguously: an Anchor2Vec linear projector, 64 learned RSS
    tokens with positional embeddings, a class token, and three pre-norm
    Transformer encoder blocks.  It deliberately omits the paper's adaptive
    random multi-task weighting, so results are reported as a reimplementation
    of the base architecture rather than as official eAaT numbers.
    """

    def __init__(
        self,
        n_aps: int,
        n_floors: int,
        n_buildings: int,
        tokens: int = 64,
        d_model: int = 128,
        n_heads: int = 8,
        layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.tokens = tokens
        self.d_model = d_model
        self.anchor2vec = nn.Linear(n_aps, tokens * d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, tokens + 1, d_model))
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.coord_head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 2))
        self.floor_head = nn.Linear(d_model, max(1, n_floors))
        self.building_head = nn.Linear(d_model, max(1, n_buildings))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.anchor2vec(x).reshape(len(x), self.tokens, self.d_model)
        cls = self.cls_token.expand(len(x), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1) + self.position)
        embedding = self.norm(encoded[:, 0])
        return {
            "embedding": embedding,
            "coord": self.coord_head(embedding),
            "floor": self.floor_head(embedding),
            "building": self.building_head(embedding),
        }


def pairwise_topology_loss(embedding: torch.Tensor, coordinate: torch.Tensor, max_points: int = 64) -> torch.Tensor:
    if len(embedding) > max_points:
        index = torch.randperm(len(embedding), device=embedding.device)[:max_points]
        embedding, coordinate = embedding[index], coordinate[index]
    e = F.normalize(embedding, dim=1)
    de = torch.cdist(e, e)
    dy = torch.cdist(coordinate, coordinate)
    dy = dy / dy.detach().mean().clamp_min(1e-4)
    de = de / de.detach().mean().clamp_min(1e-4)
    upper = torch.triu(torch.ones_like(de, dtype=torch.bool), diagonal=1)
    return F.smooth_l1_loss(de[upper], dy[upper])
