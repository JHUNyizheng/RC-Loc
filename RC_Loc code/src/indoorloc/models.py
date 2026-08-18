from __future__ import annotations

import torch
from torch import nn


class SparseAnchorUQLoc(nn.Module):
    """SAU-Loc: sparse-anchor set encoder with uncertainty-aware regression."""

    def __init__(
        self,
        n_aps: int = 520,
        max_tokens: int = 32,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
        n_floors: int = 5,
        n_buildings: int = 3,
    ) -> None:
        super().__init__()
        self.n_aps = n_aps
        self.max_tokens = min(max_tokens, n_aps)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.ap_embedding = nn.Embedding(n_aps + 1, d_model, padding_idx=0)
        self.rss_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool_score = nn.Linear(d_model, 1)
        self.post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.coord_head = nn.Linear(d_model * 2, 2)
        self.logvar_head = nn.Linear(d_model * 2, 2)
        self.floor_head = nn.Linear(d_model * 2, n_floors)
        self.building_head = nn.Linear(d_model * 2, n_buildings)

    def _tokenize(self, rss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = torch.topk(rss, k=self.max_tokens, dim=1, largest=True, sorted=True)
        valid = values > 0
        ap_ids = (indices + 1) * valid.long()
        tokens = self.ap_embedding(ap_ids) + self.rss_embedding(values.unsqueeze(-1))
        tokens = tokens * valid.unsqueeze(-1)
        # The always-valid CLS token represents an empty or heavily corrupted fingerprint.
        cls = self.cls_token.expand(len(rss), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        padding_mask = torch.cat(
            [torch.zeros((len(rss), 1), dtype=torch.bool, device=rss.device), ~valid], dim=1
        )
        return tokens, padding_mask

    def forward(self, rss: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, padding_mask = self._tokenize(rss)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        scores = self.pool_score(encoded).squeeze(-1).masked_fill(padding_mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        hidden = self.post(pooled)
        return {
            "coord": self.coord_head(hidden),
            "logvar": self.logvar_head(hidden).clamp(-6.0, 4.0),
            "floor": self.floor_head(hidden),
            "building": self.building_head(hidden),
            "embedding": hidden,
        }


class FlatMLPLoc(nn.Module):
    def __init__(self, n_aps: int = 520, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_aps, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.coord_head = nn.Linear(hidden // 2, 2)
        self.floor_head = nn.Linear(hidden // 2, 5)
        self.building_head = nn.Linear(hidden // 2, 3)

    def forward(self, rss: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(rss)
        return {
            "coord": self.coord_head(hidden),
            "logvar": torch.zeros_like(self.coord_head(hidden)),
            "floor": self.floor_head(hidden),
            "building": self.building_head(hidden),
            "embedding": hidden,
        }
