"""SimCLR projection head and NT-Xent (InfoNCE) contrastive loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRProjectionHead(nn.Module):
    """2-layer MLP projection head for SimCLR with optional BatchNorm."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        head_norm: str = "none",
    ) -> None:
        super().__init__()
        if head_norm not in {"none", "bn"}:
            raise ValueError(f"Unsupported head_norm '{head_norm}'; expected 'none' or 'bn'.")

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if head_norm == "bn":
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Compute vectorized symmetric NT-Xent (InfoNCE) loss over two views ``(N, D)``."""
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError(f"Expected matching 2D tensors, got {z1.shape} and {z2.shape}.")
    bsz = z1.shape[0]
    if bsz < 2:
        raise ValueError("NT-Xent requires batch size >= 2 to form negative pairs.")

    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)  # (2N, D)
    logits = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    self_mask = torch.eye(2 * bsz, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    # Positive pair for index i is (i + N) mod 2N.
    targets = torch.arange(2 * bsz, device=z.device)
    targets = (targets + bsz) % (2 * bsz)
    return F.cross_entropy(logits, targets)


class SimCLRModel(nn.Module):
    """Shared backbone paired with SimCLR's nonlinear projection head."""

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        head_hidden_dim: int = 256,
        proj_dim: int = 128,
        head_norm: str = "none",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = SimCLRProjectionHead(
            in_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            out_dim=proj_dim,
            head_norm=head_norm,
        )

    def forward(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = v1.shape[0]
        h_all = self.backbone(torch.cat([v1, v2], dim=0))
        z_all = self.head(h_all)
        h1, h2 = h_all[:bsz], h_all[bsz:]
        z1, z2 = z_all[:bsz], z_all[bsz:]
        return h1, h2, z1, z2
