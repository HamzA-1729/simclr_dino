"""DINO head, Student-Teacher EMA, centering/sharpening loss, and schedules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def cosine_ema_momentum(
    step: int,
    total_steps: int,
    base_momentum: float = 0.985,
) -> float:
    """Cosine-anneal teacher EMA momentum from ``base_momentum`` up to ``1.0``."""
    progress = min(max(step, 0), max(1, total_steps)) / max(1, total_steps)
    return 1.0 - (1.0 - base_momentum) * 0.5 * (1.0 + math.cos(math.pi * progress))


class DINOHead(nn.Module):
    """3-layer MLP + L2 normalization + weight-normalized prototype linear layer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
        num_prototypes: int = 1024,
        freeze_g: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = weight_norm(
            nn.Linear(bottleneck_dim, num_prototypes, bias=False),
            name="weight",
            dim=0,
        )
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(1.0)
        if freeze_g:
            self.last_layer.parametrizations.weight.original0.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.mlp(h)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOBranch(nn.Module):
    """Pairs a shared backbone with a DINO projection head."""

    def __init__(self, backbone: nn.Module, head: DINOHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward_views(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run views grouped by spatial resolution for efficient batched forward passes."""
        outputs: list[torch.Tensor | None] = [None] * len(views)
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, view in enumerate(views):
            key = (view.shape[-2], view.shape[-1])
            groups.setdefault(key, []).append(idx)

        for indices in groups.values():
            batch = torch.cat([views[i] for i in indices], dim=0)
            feats = self.backbone(batch)
            logits = self.head(feats)
            chunks = torch.chunk(logits, len(indices), dim=0)
            for idx, chunk in zip(indices, chunks):
                outputs[idx] = chunk

        return [out for out in outputs if out is not None]


class DINOModel(nn.Module):
    """Student-Teacher DINO wrapper with momentum EMA covering backbone and head."""

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        head_hidden_dim: int = 512,
        bottleneck_dim: int = 128,
        num_prototypes: int = 1024,
        freeze_g: bool = True,
    ) -> None:
        super().__init__()
        student_head = DINOHead(
            in_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_prototypes=num_prototypes,
            freeze_g=freeze_g,
        )
        self.student = DINOBranch(backbone, student_head)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update_teacher_ema(self, momentum: float) -> None:
        """Update teacher backbone and head parameters via EMA."""
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

    def cancel_last_layer_gradients(self) -> None:
        """Set prototype-layer grads to ``None`` so AdamW skips weight decay and momentum."""
        for param in self.student.head.last_layer.parameters():
            param.grad = None


class DINOLoss(nn.Module):
    """DINO cross-view self-distillation loss with teacher centering and sharpening."""

    def __init__(
        self,
        num_prototypes: int,
        student_temp: float = 0.1,
        teacher_temp_start: float = 0.04,
        teacher_temp_end: float = 0.07,
        teacher_temp_warmup_steps: int = 900,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_end = teacher_temp_end
        self.teacher_temp_warmup_steps = max(1, teacher_temp_warmup_steps)
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, num_prototypes))

    def get_teacher_temp(self, step: int) -> float:
        if step >= self.teacher_temp_warmup_steps:
            return self.teacher_temp_end
        frac = max(step, 0) / self.teacher_temp_warmup_steps
        return self.teacher_temp_start + frac * (self.teacher_temp_end - self.teacher_temp_start)

    def forward(
        self,
        student_outputs: list[torch.Tensor],
        teacher_outputs: list[torch.Tensor],
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if len(teacher_outputs) != 2:
            raise ValueError(f"Expected 2 global teacher outputs, got {len(teacher_outputs)}.")
        if len(student_outputs) < 2:
            raise ValueError(f"Expected at least 2 student outputs, got {len(student_outputs)}.")

        tau_t = self.get_teacher_temp(step)
        teacher_probs = [
            F.softmax((t.detach() - self.center) / tau_t, dim=-1)
            for t in teacher_outputs
        ]
        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_outputs
        ]

        total_loss = student_outputs[0].new_tensor(0.0)
        total_kl = student_outputs[0].new_tensor(0.0)
        n_terms = 0

        for iq, q in enumerate(teacher_probs):
            for iv, log_p in enumerate(student_log_probs):
                if iq == iv:
                    continue
                ce = -(q * log_p).sum(dim=-1).mean()
                q_ent = -(q * torch.log(q.clamp(min=1e-12))).sum(dim=-1).mean()
                total_loss = total_loss + ce
                total_kl = total_kl + (ce - q_ent)
                n_terms += 1

        loss = total_loss / n_terms
        mean_kl = (total_kl / n_terms).item()

        with torch.no_grad():
            t_cat = torch.cat([t.detach() for t in teacher_outputs], dim=0)
            batch_center = t_cat.mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )
            q_stack = torch.cat(teacher_probs, dim=0)
            teacher_entropy = -(q_stack * torch.log(q_stack.clamp(min=1e-12))).sum(dim=-1).mean().item()

        metrics = {
            "n_terms": float(n_terms),
            "teacher_entropy": teacher_entropy,
            "kl_div": mean_kl,
            "teacher_temp": tau_t,
        }
        return loss, metrics
