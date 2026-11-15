"""Unit tests for batched tensor augmentations."""

from __future__ import annotations

import torch

from simclr_dino.augmentations import (
    SSLAugmentor,
    denormalize_cifar10,
    normalize_cifar10,
    random_color_jitter,
    random_grayscale,
)


def test_simclr_and_dino_view_shapes():
    aug = SSLAugmentor(n_local_crops=4)
    x = torch.rand(6, 3, 32, 32)

    v1, v2 = aug.simclr_views(x)
    assert v1.shape == (6, 3, 32, 32)
    assert v2.shape == (6, 3, 32, 32)
    assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
    assert not torch.allclose(v1, v2)

    globals_, locals_ = aug.dino_views(x)
    assert len(globals_) == 2
    assert len(locals_) == 4
    assert all(g.shape == (6, 3, 32, 32) for g in globals_)
    assert all(loc.shape == (6, 3, 16, 16) for loc in locals_)


def test_seeded_determinism():
    aug = SSLAugmentor()
    x = torch.rand(4, 3, 32, 32)

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    v1_a, v2_a = aug.simclr_views(x, generator=g1)
    v1_b, v2_b = aug.simclr_views(x, generator=g2)

    assert torch.allclose(v1_a, v1_b)
    assert torch.allclose(v2_a, v2_b)


def test_normalize_denormalize_roundtrip_and_color_bounds():
    x = torch.rand(4, 3, 32, 32)
    rt = denormalize_cifar10(normalize_cifar10(x))
    assert torch.allclose(x, rt, atol=1e-6)

    jittered = random_color_jitter(x, p=1.0, strength=0.5)
    assert jittered.min() >= 0.0 and jittered.max() <= 1.0

    gray = random_grayscale(x, p=1.0)
    assert torch.allclose(gray[:, 0], gray[:, 1])
    assert torch.allclose(gray[:, 1], gray[:, 2])
