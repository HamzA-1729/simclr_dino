"""Unit tests for stratified CIFAR-10 splitting and zero-leakage guarantees."""

from __future__ import annotations

from collections import Counter

import torch

from simclr_dino.data import TensorImageDataset, build_stratified_indices


def test_stratified_indices_zero_leakage_and_class_balance():
    # Synthetic 50,000-label CIFAR-10-like target list (5,000 per class).
    targets = [c for c in range(10) for _ in range(5000)]

    train_idx, val_idx, full_train_idx = build_stratified_indices(
        targets=targets,
        val_size=5000,
        train_subset=5000,
        split_seed=42,
    )

    assert len(val_idx) == 5000
    assert len(full_train_idx) == 45000
    assert len(train_idx) == 5000

    assert set(val_idx).isdisjoint(set(full_train_idx))
    assert set(train_idx).issubset(set(full_train_idx))
    assert set(train_idx).isdisjoint(set(val_idx))

    val_counts = Counter(targets[i] for i in val_idx)
    sub_counts = Counter(targets[i] for i in train_idx)
    assert all(val_counts[c] == 500 for c in range(10))
    assert all(sub_counts[c] == 500 for c in range(10))


def test_stratified_indices_deterministic_under_seed():
    targets = [c for c in range(10) for _ in range(100)]
    t1, v1, _ = build_stratified_indices(targets, val_size=100, train_subset=200, split_seed=7)
    t2, v2, _ = build_stratified_indices(targets, val_size=100, train_subset=200, split_seed=7)
    assert t1 == t2
    assert v1 == v2


def test_tensor_image_dataset_scales_to_unit_float():
    imgs = torch.full((4, 3, 32, 32), 255, dtype=torch.uint8)
    ds = TensorImageDataset(imgs, [0, 1, 2, 3])
    img, label = ds[1]
    assert img.dtype == torch.float32
    assert torch.allclose(img, torch.ones(3, 32, 32))
    assert label.item() == 1
