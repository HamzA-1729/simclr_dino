"""Tiny Vision Transformer (ViT) backbone with dynamic 2D sin-cos positional embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_pos_embed(
    grid_h: int,
    grid_w: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute fixed 2D sinusoidal positional embeddings of shape ``(1, grid_h * grid_w, embed_dim)``."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin-cos pos embed, got {embed_dim}.")

    half_dim = embed_dim // 2
    omega = torch.arange(half_dim // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (half_dim // 2)))

    y = torch.arange(grid_h, device=device, dtype=torch.float32)
    x = torch.arange(grid_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    out_y = grid_y.reshape(-1, 1) * omega.unsqueeze(0)
    out_x = grid_x.reshape(-1, 1) * omega.unsqueeze(0)

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    pos = torch.cat([pos_y, pos_x], dim=1).unsqueeze(0)
    return pos.to(dtype=dtype)


class BidirectionalSelfAttention(nn.Module):
    """Multi-head bidirectional self-attention with optional manual attention-weight return."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights: torch.Tensor | None = None
        if return_attention:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)
        out = self.proj(out)
        if return_attention:
            assert attn_weights is not None
            return out, attn_weights
        return out


class TransformerBlock(nn.Module):
    """Pre-norm ViT encoder block with GELU MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = BidirectionalSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x), return_attention=False)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """Tiny Vision Transformer backbone returning final-LayerNorm ``[CLS]`` features."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Input spatial dims ({height}, {width}) must be divisible by patch_size ({self.patch_size})."
            )
        patches = self.patch_embed(x)  # (B, D, H/p, W/p)
        grid_h, grid_w = patches.shape[-2], patches.shape[-1]
        patch_tokens = patches.flatten(2).transpose(1, 2)  # (B, N_p, D)

        patch_pos = build_2d_sincos_pos_embed(
            grid_h, grid_w, self.embed_dim, device=x.device, dtype=x.dtype
        )
        patch_tokens = patch_tokens + patch_pos

        cls_tokens = (self.cls_token + self.cls_pos_embed).expand(bsz, -1, -1)
        return torch.cat([cls_tokens, patch_tokens], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._prepare_tokens(x)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cls_feature, last_block_attention_weights)``."""
        tokens = self._prepare_tokens(x)
        attn_weights: torch.Tensor | None = None
        for idx, block in enumerate(self.blocks):
            if idx == len(self.blocks) - 1:
                tokens, attn_weights = block(tokens, return_attention=True)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        assert attn_weights is not None
        return tokens[:, 0], attn_weights
