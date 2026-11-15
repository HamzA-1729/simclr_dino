"""Unit tests for TinyViT backbone and bidirectional attention parity."""

from __future__ import annotations

import torch

from simclr_dino.vit import TinyViT


def test_vit_shapes_global_and_local():
    model = TinyViT(patch_size=4, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0)
    x_global = torch.randn(3, 3, 32, 32)
    x_local = torch.randn(5, 3, 16, 16)

    h_global = model(x_global)
    h_local = model(x_local)

    assert h_global.shape == (3, 128)
    assert h_local.shape == (5, 128)

    n_params = sum(p.numel() for p in model.parameters())
    assert 750_000 < n_params < 850_000


def test_manual_attention_matches_sdpa_and_sums_to_one():
    torch.manual_seed(0)
    model = TinyViT(patch_size=4, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0)
    model.eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        h_sdpa = model(x)
        h_manual, attn = model.forward_with_attention(x)

    assert torch.allclose(h_sdpa, h_manual, atol=1e-5)
    assert attn.shape == (2, 4, 65, 65)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4, 65), atol=1e-5)
