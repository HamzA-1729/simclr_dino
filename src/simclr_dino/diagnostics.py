"""Representation collapse diagnostics: per-dimension std and RankMe effective rank."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def normalized_feature_std(features: torch.Tensor) -> float:
    """
    Mean per-dimension standard deviation of L2-normalized embeddings.

    Spread-out isotropic embeddings approach ``1 / sqrt(D)``; constant collapse gives ``0.0``.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(f"Expected 2D tensor with N >= 2, got {features.shape}.")
    normed = F.normalize(features.float(), dim=-1)
    return normed.std(dim=0, unbiased=False).mean().item()


@torch.no_grad()
def rankme_effective_rank(features: torch.Tensor, eps: float = 1e-7) -> float:
    """
    Compute RankMe effective rank ``exp(-sum(p * log(p + eps)))`` on ``features`` ``(N, D)``.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(f"Expected 2D tensor with N >= 2, got {features.shape}.")
    s = torch.linalg.svdvals(features.float())
    p = s / (s.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def compute_representation_diagnostics(
    features: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Compute both ``feature_std`` and ``rankme`` on backbone ``[CLS]`` features ``h``."""
    return {
        "feature_std": normalized_feature_std(features),
        "rankme": rankme_effective_rank(features, eps=eps),
    }
