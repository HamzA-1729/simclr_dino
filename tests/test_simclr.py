"""Unit tests for SimCLR projection head and NT-Xent loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from simclr_dino.simclr import SimCLRModel, nt_xent_loss
from simclr_dino.vit import TinyViT


def _naive_nt_xent_loop(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    n = z1.shape[0]
    total = z.new_tensor(0.0)
    for i in range(2 * n):
        pos_idx = (i + n) % (2 * n)
        sims = []
        for k in range(2 * n):
            if k != i:
                sims.append(torch.dot(z[i], z[k]) / temperature)
        denom = torch.logsumexp(torch.stack(sims), dim=0)
        pos_sim = torch.dot(z[i], z[pos_idx]) / temperature
        total = total + (denom - pos_sim)
    return total / (2 * n)


def test_nt_xent_identical_embeddings_equals_log_2n_minus_1():
    n = 256
    z = torch.ones(n, 16)
    loss = nt_xent_loss(z, z.clone(), temperature=0.2)
    assert loss.item() == pytest.approx(math.log(2 * n - 1), rel=1e-4)


def test_nt_xent_matches_naive_loop():
    torch.manual_seed(42)
    z1 = torch.randn(8, 32)
    z2 = torch.randn(8, 32)
    tau = 0.2

    loss_vec = nt_xent_loss(z1, z2, temperature=tau)
    loss_loop = _naive_nt_xent_loop(z1, z2, temperature=tau)
    assert torch.allclose(loss_vec, loss_loop, atol=1e-5)


def test_simclr_model_gradients_flow_through_both_views():
    backbone = TinyViT(patch_size=4, embed_dim=64, depth=2, num_heads=4)
    model = SimCLRModel(backbone=backbone, embed_dim=64, head_hidden_dim=128, proj_dim=64)

    v1 = torch.randn(4, 3, 32, 32, requires_grad=True)
    v2 = torch.randn(4, 3, 32, 32, requires_grad=True)
    _, _, z1, z2 = model(v1, v2)
    loss = nt_xent_loss(z1, z2, temperature=0.2)
    loss.backward()

    assert v1.grad is not None and v1.grad.abs().sum() > 0
    assert v2.grad is not None and v2.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
