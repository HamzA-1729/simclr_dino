"""Unit tests for DINO head, Student-Teacher EMA, centering, and loss."""

from __future__ import annotations

import math

import pytest
import torch

from simclr_dino.dino import DINOHead, DINOLoss, DINOModel
from simclr_dino.vit import TinyViT


def test_dino_head_weight_norm_g_is_one_and_unit_rows():
    head = DINOHead(in_dim=64, hidden_dim=128, bottleneck_dim=32, num_prototypes=64, freeze_g=True)
    g = head.last_layer.parametrizations.weight.original0
    assert torch.allclose(g, torch.ones_like(g))
    assert g.requires_grad is False

    row_norms = head.last_layer.weight.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)


@pytest.mark.parametrize("n_local,expected_terms", [(0, 2), (2, 6), (4, 10)])
def test_dino_loss_term_count_and_teacher_no_grad(n_local: int, expected_terms: int):
    loss_fn = DINOLoss(num_prototypes=32)
    teacher_outs = [torch.randn(4, 32) for _ in range(2)]
    student_outs = [torch.randn(4, 32, requires_grad=True) for _ in range(2 + n_local)]

    loss, metrics = loss_fn(student_outs, teacher_outs, step=0)
    assert metrics["n_terms"] == float(expected_terms)
    assert not loss_fn.center.requires_grad

    loss.backward()
    assert all(s.grad is not None for s in student_outs)


def test_dino_ema_covers_backbone_and_head_and_teacher_frozen():
    backbone = TinyViT(patch_size=4, embed_dim=64, depth=2, num_heads=4)
    model = DINOModel(backbone=backbone, embed_dim=64, head_hidden_dim=128, bottleneck_dim=32, num_prototypes=64)

    g1 = torch.randn(3, 3, 32, 32)
    g2 = torch.randn(3, 3, 32, 32)
    loc = torch.randn(3, 3, 16, 16)

    with torch.no_grad():
        t_out = model.teacher.forward_views([g1, g2])
    s_out = model.student.forward_views([g1, g2, loc])
    loss, _ = DINOLoss(num_prototypes=64)(s_out, t_out, step=0)
    loss.backward()

    assert all(p.grad is None for p in model.teacher.parameters())

    # m=1.0 freezes teacher
    teacher_before = [p.detach().clone() for p in model.teacher.parameters()]
    with torch.no_grad():
        for p in model.student.parameters():
            p.add_(0.5)
    model.update_teacher_ema(momentum=1.0)
    for p_before, p_after in zip(teacher_before, model.teacher.parameters()):
        assert torch.allclose(p_before, p_after)

    # m=0.0 copies student (including head) to teacher
    model.update_teacher_ema(momentum=0.0)
    for p_s, p_t in zip(model.student.parameters(), model.teacher.parameters()):
        assert torch.allclose(p_s, p_t)


def test_cancel_last_layer_gradients_freezes_prototypes_with_adamw():
    backbone = TinyViT(patch_size=4, embed_dim=64, depth=2, num_heads=4)
    model = DINOModel(backbone=backbone, embed_dim=64, head_hidden_dim=128, bottleneck_dim=32, num_prototypes=64)
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-2, weight_decay=0.1)

    before = [p.detach().clone() for p in model.student.head.last_layer.parameters()]
    g1, g2 = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    loss, _ = DINOLoss(num_prototypes=64)(
        model.student.forward_views([g1, g2]),
        model.teacher.forward_views([g1, g2]),
        step=0,
    )
    optimizer.zero_grad()
    loss.backward()
    model.cancel_last_layer_gradients()
    optimizer.step()

    for p_b, p_a in zip(before, model.student.head.last_layer.parameters()):
        assert torch.allclose(p_b, p_a)


def test_dino_uniform_logits_give_log_k_entropy():
    k = 64
    loss_fn = DINOLoss(num_prototypes=k)
    t_outs = [torch.zeros(4, k), torch.zeros(4, k)]
    s_outs = [torch.zeros(4, k), torch.zeros(4, k)]
    _, metrics = loss_fn(s_outs, t_outs, step=0)
    assert metrics["teacher_entropy"] == pytest.approx(math.log(k), rel=1e-4)
