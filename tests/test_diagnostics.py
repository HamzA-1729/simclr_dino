"""Unit tests for representation collapse diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from simclr_dino.diagnostics import (
    compute_representation_diagnostics,
    normalized_feature_std,
    rankme_effective_rank,
)


def test_constant_embeddings_collapse_diagnostics():
    feats = torch.ones(64, 128)
    assert normalized_feature_std(feats) == pytest.approx(0.0, abs=1e-5)
    assert rankme_effective_rank(feats, eps=1e-7) == pytest.approx(1.0, abs=1e-2)


def test_identity_embeddings_full_effective_rank():
    d = 128
    feats = torch.eye(d)
    assert rankme_effective_rank(feats, eps=1e-7) == pytest.approx(float(d), abs=1e-2)


def test_isotropic_random_embeddings_near_one_over_sqrt_d():
    torch.manual_seed(0)
    d = 128
    feats = torch.randn(1024, d)
    diag = compute_representation_diagnostics(feats, eps=1e-7)
    assert diag["feature_std"] == pytest.approx(1.0 / math.sqrt(d), abs=5e-3)
    assert diag["rankme"] > 0.8 * d
