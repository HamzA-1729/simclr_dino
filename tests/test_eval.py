"""Unit tests for chunked weighted k-NN and frozen Linear Probe evaluation."""

from __future__ import annotations

import torch

from simclr_dino.eval import (
    evaluate_linear_probe,
    train_linear_probe,
    weighted_knn_accuracy,
)


def _make_separable_clusters(n_per_class: int = 30, dim: int = 16, num_classes: int = 4):
    torch.manual_seed(0)
    feats = []
    labels = []
    for c in range(num_classes):
        center = torch.zeros(dim)
        center[c] = 5.0
        pts = center.unsqueeze(0) + 0.1 * torch.randn(n_per_class, dim)
        feats.append(pts)
        labels.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def test_weighted_knn_perfect_on_separable_clusters():
    train_x, train_y = _make_separable_clusters(n_per_class=25)
    query_x, query_y = _make_separable_clusters(n_per_class=10)

    acc_single = weighted_knn_accuracy(
        train_x, train_y, query_x, query_y, k=5, num_classes=4, chunk_size=100
    )
    acc_chunked = weighted_knn_accuracy(
        train_x, train_y, query_x, query_y, k=5, num_classes=4, chunk_size=7
    )
    assert acc_single == 1.0
    assert acc_chunked == 1.0


def test_linear_probe_perfect_on_separable_clusters():
    train_x, train_y = _make_separable_clusters(n_per_class=30)
    val_x, val_y = _make_separable_clusters(n_per_class=10)

    res = train_linear_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        num_classes=4,
        lr=0.1,
        epochs=50,
        weight_decay_grid=(0.0, 1e-4),
    )
    assert res["val_acc"] == 1.0
    test_acc = evaluate_linear_probe(
        res["probe"], val_x, val_y, mean=res["mean"], std=res["std"]
    )
    assert test_acc == 1.0
