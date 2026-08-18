import numpy as np
import torch
import pandas as pd

from indoorloc.adapters import load_huawei_long_csv, normalize_huawei_rssi
from indoorloc.data import corrupt_rss, normalize_rss
from indoorloc.metrics import localization_metrics
from indoorloc.models import SparseAnchorUQLoc
from indoorloc.twc_data import _encode_from_train, normalize_rss_matrix, spatial_groups
from indoorloc.twc_models import (
    Anchor2VecTransformer,
    ConvRSSLoc,
    DNNBNLoc,
    MaskTopoLoc,
    pairwise_topology_loss,
)


def test_normalize_rss_preserves_missing_mask():
    x = np.array([[100, -110, -55, 0]], dtype=np.float32)
    y = normalize_rss(x)
    assert y[0, 0] == 0
    assert y[0, 1] == 0
    assert 0 < y[0, 2] < 1
    assert y[0, 3] == 1


def test_corruption_is_reproducible():
    x = np.full((3, 8), 0.5, dtype=np.float32)
    a = corrupt_rss(x, drop_prob=0.5, noise_db=2, seed=7)
    b = corrupt_rss(x, drop_prob=0.5, noise_db=2, seed=7)
    assert np.array_equal(a, b)


def test_sparse_model_shapes_and_finite_values():
    model = SparseAnchorUQLoc(n_aps=16, max_tokens=4, d_model=16, n_heads=4, n_layers=1)
    x = torch.zeros(3, 16)
    x[:, :6] = torch.rand(3, 6)
    out = model(x)
    assert out["coord"].shape == (3, 2)
    assert out["logvar"].shape == (3, 2)
    assert torch.isfinite(out["coord"]).all()
    empty_out = model(torch.zeros(2, 16))
    assert torch.isfinite(empty_out["coord"]).all()
    assert torch.isfinite(empty_out["logvar"]).all()


def test_metrics_known_error():
    y = np.array([[0, 0], [0, 0]], dtype=float)
    pred = np.array([[3, 4], [0, 0]], dtype=float)
    m = localization_metrics(y, pred)
    assert m["mean_m"] == 2.5
    assert m["p90_m"] == 4.5


def test_huawei_long_adapter_aggregates_and_preserves_missing(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": [1, 1, 1, 2],
            "anchor_id": ["A", "A", "B", "A"],
            "rssi_dbm": [-60, -62, -80, -70],
            "x_m": [0, 0, 0, 2],
            "y_m": [1, 1, 1, 3],
        }
    )
    path = tmp_path / "huawei.csv"
    frame.to_csv(path, index=False)
    data = load_huawei_long_csv(path)
    assert data.anchor_ids == ["A", "B"]
    assert data.rssi_dbm.shape == (2, 2)
    assert data.rssi_dbm[0, 0] == -61
    assert np.isnan(data.rssi_dbm[1, 1])
    normalized = normalize_huawei_rssi(data.rssi_dbm)
    assert normalized[1, 1] == 0
    assert normalized[0, 0] > 0


def test_twc_normalization_groups_and_model():
    raw = np.array([[100, -110, -55], [-80, 100, -40]], dtype=np.float32)
    x = normalize_rss_matrix(raw)
    assert np.array_equal(x[:, 0] == 0, np.array([True, False]))
    y = np.array([[0, 0], [0.1, 0.1], [3, 4]], dtype=np.float32)
    groups = spatial_groups(y, np.zeros(3), np.zeros(3), resolution=0.5)
    assert groups[0] == groups[1] and groups[2] != groups[0]
    model = MaskTopoLoc(n_aps=3, n_floors=1, n_buildings=1, hidden=32, embedding_dim=8, blocks=1)
    out = model(torch.from_numpy(x))
    assert out["coord"].shape == (2, 2)
    loss = pairwise_topology_loss(out["embedding"], torch.from_numpy(y[:2]))
    assert torch.isfinite(loss)


def test_extended_baseline_shapes():
    x = torch.rand(4, 32)
    models = (
        DNNBNLoc(32, 3, 2),
        ConvRSSLoc(32, 3, 2),
        MaskTopoLoc(32, 3, 2, hidden=48, embedding_dim=12, blocks=1, use_mask=False),
    )
    for model in models:
        model.eval()
        out = model(x)
        assert out["coord"].shape == (4, 2)
        assert torch.isfinite(out["coord"]).all()


def test_test_only_classes_do_not_change_training_encoding():
    train, test = _encode_from_train(["floor_b", "floor_a", "floor_b"], ["floor_a", "floor_c"])
    assert np.array_equal(train, np.array([1, 0, 1]))
    assert np.array_equal(test, np.array([0, -1]))


def test_unknown_class_accuracy_is_reported_separately():
    y = np.zeros((2, 2), dtype=float)
    m = localization_metrics(y, y, floor_true=np.array([0, -1]), floor_pred=np.array([0, 0]))
    assert m["floor_accuracy"] == 1.0
    assert m["floor_unknown_fraction"] == 0.5


def test_anchor2vec_transformer_shapes():
    model = Anchor2VecTransformer(
        n_aps=12, n_floors=3, n_buildings=2,
        tokens=4, d_model=16, n_heads=4, layers=1,
    )
    out = model(torch.rand(5, 12))
    assert out["coord"].shape == (5, 2)
    assert out["floor"].shape == (5, 3)
    assert out["building"].shape == (5, 2)
    assert torch.isfinite(out["embedding"]).all()
