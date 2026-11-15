"""Batched tensor augmentations for SimCLR and DINO (no PIL DataLoader bottleneck)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def normalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    """Normalize ``[0, 1]`` RGB images ``(B, 3, H, W)`` with CIFAR-10 channel stats."""
    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def denormalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    """Invert CIFAR-10 normalization back to ``[0, 1]`` for visualization."""
    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def random_resized_crop(
    images: torch.Tensor,
    size: int,
    scale: tuple[float, float] = (0.3, 1.0),
    ratio: tuple[float, float] = (0.75, 1.3333),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-sample batched RandomResizedCrop using ``affine_grid`` + ``grid_sample``."""
    bsz, channels, _, _ = images.shape
    device = images.device

    areas = torch.empty(bsz, device=device).uniform_(scale[0], scale[1], generator=generator)
    log_ratio_min = torch.log(torch.tensor(ratio[0], device=device))
    log_ratio_max = torch.log(torch.tensor(ratio[1], device=device))
    aspect = torch.exp(
        torch.empty(bsz, device=device).uniform_(log_ratio_min.item(), log_ratio_max.item(), generator=generator)
    )

    # Fractional width and height in [0, 1] clamped so the crop fits inside the image.
    w_frac = torch.sqrt(areas * aspect).clamp(max=1.0)
    h_frac = torch.sqrt(areas / aspect).clamp(max=1.0)

    # In [-1, 1] normalized coordinates, half-width is w_frac and center lies in [-(1-w_frac), (1-w_frac)].
    max_tx = 1.0 - w_frac
    max_ty = 1.0 - h_frac
    cx = (torch.rand(bsz, device=device, generator=generator) * 2.0 - 1.0) * max_tx
    cy = (torch.rand(bsz, device=device, generator=generator) * 2.0 - 1.0) * max_ty

    theta = torch.zeros(bsz, 2, 3, device=device, dtype=images.dtype)
    theta[:, 0, 0] = w_frac
    theta[:, 1, 1] = h_frac
    theta[:, 0, 2] = cx
    theta[:, 1, 2] = cy

    grid = F.affine_grid(theta, size=(bsz, channels, size, size), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def random_horizontal_flip(
    images: torch.Tensor,
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-sample horizontal flip with probability ``p``."""
    if p <= 0.0:
        return images
    mask = (torch.rand(images.shape[0], device=images.device, generator=generator) < p).view(-1, 1, 1, 1)
    return torch.where(mask, images.flip(-1), images)


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, 3, H, W)`` RGB in ``[0, 1]`` to HSV in ``[0, 1]``."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc, max_idx = rgb.max(dim=1)
    minc = rgb.min(dim=1).values
    v = maxc
    diff = maxc - minc
    s = torch.where(maxc > 0, diff / maxc.clamp(min=1e-8), torch.zeros_like(maxc))

    safe_diff = diff.clamp(min=1e-8)
    rc = (maxc - r) / safe_diff
    gc = (maxc - g) / safe_diff
    bc = (maxc - b) / safe_diff

    h = torch.zeros_like(maxc)
    h = torch.where(max_idx == 0, bc - gc, h)
    h = torch.where(max_idx == 1, 2.0 + rc - bc, h)
    h = torch.where(max_idx == 2, 4.0 + gc - rc, h)
    h = torch.where(diff > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, 3, H, W)`` HSV in ``[0, 1]`` to RGB in ``[0, 1]``."""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = torch.floor(h * 6.0).to(torch.long)
    f = (h * 6.0) - i.to(hsv.dtype)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6

    # Stack candidates (6, B, H, W) and gather by i_mod for clean vectorized selection.
    r_cand = torch.stack([v, q, p, p, t, v], dim=0)
    g_cand = torch.stack([t, v, v, q, p, p], dim=0)
    b_cand = torch.stack([p, p, t, v, v, q], dim=0)
    idx = i_mod.unsqueeze(0)
    r = torch.gather(r_cand, 0, idx).squeeze(0)
    g = torch.gather(g_cand, 0, idx).squeeze(0)
    b = torch.gather(b_cand, 0, idx).squeeze(0)
    return torch.stack([r, g, b], dim=1)


def _luma(images: torch.Tensor) -> torch.Tensor:
    """Compute ITU-R BT.601 grayscale luma ``(B, 1, H, W)``."""
    weights = images.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def random_color_jitter(
    images: torch.Tensor,
    p: float = 0.8,
    strength: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-sample color jitter (brightness, contrast, saturation, hue) on ``[0, 1]`` tensors."""
    if p <= 0.0 or strength <= 0.0:
        return images

    bsz = images.shape[0]
    device = images.device
    apply_mask = (torch.rand(bsz, device=device, generator=generator) < p).view(-1, 1, 1, 1)
    if not apply_mask.any():
        return images

    b_lim = 0.8 * strength
    c_lim = 0.8 * strength
    s_lim = 0.8 * strength
    h_lim = 0.2 * strength

    out = images
    order = torch.randperm(4, generator=generator).tolist()
    for op in order:
        if op == 0:
            bf = torch.empty(bsz, 1, 1, 1, device=device).uniform_(1.0 - b_lim, 1.0 + b_lim, generator=generator)
            out = (out * bf).clamp(0.0, 1.0)
        elif op == 1:
            cf = torch.empty(bsz, 1, 1, 1, device=device).uniform_(1.0 - c_lim, 1.0 + c_lim, generator=generator)
            mean_val = _luma(out).mean(dim=(-2, -1), keepdim=True)
            out = (out * cf + mean_val * (1.0 - cf)).clamp(0.0, 1.0)
        elif op == 2:
            sf = torch.empty(bsz, 1, 1, 1, device=device).uniform_(1.0 - s_lim, 1.0 + s_lim, generator=generator)
            gray = _luma(out).expand_as(out)
            out = (out * sf + gray * (1.0 - sf)).clamp(0.0, 1.0)
        else:
            hf = torch.empty(bsz, 1, 1, device=device).uniform_(-h_lim, h_lim, generator=generator)
            hsv = _rgb_to_hsv(out)
            hsv = torch.stack([(hsv[:, 0] + hf) % 1.0, hsv[:, 1], hsv[:, 2]], dim=1)
            out = _hsv_to_rgb(hsv).clamp(0.0, 1.0)

    return torch.where(apply_mask, out, images)


def random_grayscale(
    images: torch.Tensor,
    p: float = 0.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-sample conversion to 3-channel grayscale with probability ``p``."""
    if p <= 0.0:
        return images
    mask = (torch.rand(images.shape[0], device=images.device, generator=generator) < p).view(-1, 1, 1, 1)
    gray = _luma(images).expand_as(images)
    return torch.where(mask, gray, images)


@dataclass(frozen=True)
class SSLAugmentor:
    """Shared batched augmentation pipeline for SimCLR and DINO (no Gaussian blur)."""

    global_size: int = 32
    global_scale: tuple[float, float] = (0.3, 1.0)
    local_size: int = 16
    local_scale: tuple[float, float] = (0.05, 0.3)
    n_local_crops: int = 4
    ratio: tuple[float, float] = (0.75, 1.3333)
    flip_p: float = 0.5
    jitter_p: float = 0.8
    jitter_strength: float = 0.5
    grayscale_p: float = 0.2

    def augment_view(
        self,
        images: torch.Tensor,
        size: int,
        scale: Sequence[float],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Apply crop -> flip -> color jitter -> grayscale -> CIFAR-10 normalization."""
        x = random_resized_crop(
            images,
            size=size,
            scale=(float(scale[0]), float(scale[1])),
            ratio=(float(self.ratio[0]), float(self.ratio[1])),
            generator=generator,
        )
        x = random_horizontal_flip(x, p=self.flip_p, generator=generator)
        x = random_color_jitter(x, p=self.jitter_p, strength=self.jitter_strength, generator=generator)
        x = random_grayscale(x, p=self.grayscale_p, generator=generator)
        return normalize_cifar10(x)

    def simclr_views(
        self,
        images: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate two identically configured global views for SimCLR."""
        v1 = self.augment_view(images, self.global_size, self.global_scale, generator=generator)
        v2 = self.augment_view(images, self.global_size, self.global_scale, generator=generator)
        return v1, v2

    def dino_views(
        self,
        images: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generate 2 global views and ``n_local_crops`` local views for DINO."""
        globals_ = [
            self.augment_view(images, self.global_size, self.global_scale, generator=generator)
            for _ in range(2)
        ]
        locals_ = [
            self.augment_view(images, self.local_size, self.local_scale, generator=generator)
            for _ in range(self.n_local_crops)
        ]
        return globals_, locals_
