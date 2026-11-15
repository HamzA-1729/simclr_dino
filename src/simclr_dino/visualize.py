"""Visualizations for augmentations, training/collapse curves, and [CLS] attention maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from simclr_dino.augmentations import SSLAugmentor, denormalize_cifar10, normalize_cifar10
from simclr_dino.data import load_cifar10_splits
from simclr_dino.vit import TinyViT


def save_augmentation_grid(
    images: torch.Tensor,
    augmentor: SSLAugmentor,
    out_path: str | Path = "outputs/augmentation_grid.png",
    num_images: int = 4,
    seed: int = 42,
) -> Path:
    """Save a grid of original images alongside 2 global (32x32) and 4 local (16x16) crops."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    batch = images[:num_images]
    gen = torch.Generator(device=batch.device).manual_seed(seed)
    globals_, locals_ = augmentor.dino_views(batch, generator=gen)

    cols = 1 + len(globals_) + len(locals_)
    fig, axes = plt.subplots(num_images, cols, figsize=(2.0 * cols, 2.0 * num_images))
    if num_images == 1:
        axes = axes[None, :]

    titles = ["Original"] + [f"Global {i+1}" for i in range(len(globals_))] + [
        f"Local {i+1}" for i in range(len(locals_))
    ]
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=9)

    for r in range(num_images):
        row_imgs = [batch[r]] + [denormalize_cifar10(g)[r] for g in globals_] + [
            denormalize_cifar10(loc)[r] for loc in locals_
        ]
        for c, img in enumerate(row_imgs):
            axes[r, c].imshow(img.permute(1, 2, 0).cpu().numpy())
            axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_training_curves(
    simclr_metrics_path: str | Path,
    dino_metrics_path: str | Path,
    out_path: str | Path = "outputs/training_curves.png",
) -> Path:
    """Plot val k-NN, feature std, RankMe, and DINO teacher entropy/KL curves."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(simclr_metrics_path, "r", encoding="utf-8") as f:
        sim_data = json.load(f)["history"]
    with open(dino_metrics_path, "r", encoding="utf-8") as f:
        dino_data = json.load(f)["history"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 1. Val k-NN Accuracy
    axes[0, 0].plot(
        [d["step"] for d in sim_data],
        [d["val_knn_acc"] * 100 for d in sim_data],
        marker="o",
        label="SimCLR",
    )
    axes[0, 0].plot(
        [d["step"] for d in dino_data],
        [d["teacher_val_knn_acc"] * 100 for d in dino_data],
        marker="s",
        label="DINO (Teacher)",
    )
    axes[0, 0].plot(
        [d["step"] for d in dino_data],
        [d["student_val_knn_acc"] * 100 for d in dino_data],
        linestyle="--",
        label="DINO (Student)",
    )
    axes[0, 0].set_title("Validation Weighted k-NN (%)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Per-dim Feature Std
    axes[0, 1].plot([d["step"] for d in sim_data], [d["feature_std"] for d in sim_data], label="SimCLR")
    axes[0, 1].plot([d["step"] for d in dino_data], [d["feature_std"] for d in dino_data], label="DINO")
    axes[0, 1].set_title("Normalized [CLS] Feature Std")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. RankMe Effective Rank
    axes[1, 0].plot([d["step"] for d in sim_data], [d["rankme"] for d in sim_data], label="SimCLR")
    axes[1, 0].plot([d["step"] for d in dino_data], [d["rankme"] for d in dino_data], label="DINO")
    axes[1, 0].set_title("RankMe Effective Rank (h)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. DINO Entropy & KL
    axes[1, 1].plot(
        [d["step"] for d in dino_data],
        [d["teacher_entropy"] for d in dino_data],
        label="Teacher Entropy",
    )
    axes[1, 1].plot(
        [d["step"] for d in dino_data],
        [d["kl_div"] for d in dino_data],
        label="KL(Teacher || Student)",
    )
    axes[1, 1].set_title("DINO Collapse Diagnostics")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def save_cls_attention_grid(
    images: torch.Tensor,
    simclr_ckpt: str | Path,
    dino_ckpt: str | Path,
    cfg: dict,
    out_path: str | Path = "outputs/cls_attention_grid.png",
    num_images: int = 4,
    seed: int = 42,
) -> Path:
    """Compare mean [CLS] -> patch attention maps across Random-Init, SimCLR, and DINO."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    bb_cfg = cfg["backbone"]

    def _build_vit(ckpt_path: str | Path | None, key: str | None) -> TinyViT:
        torch.manual_seed(seed)
        m = TinyViT(
            patch_size=bb_cfg["patch_size"],
            embed_dim=bb_cfg["embed_dim"],
            depth=bb_cfg["depth"],
            num_heads=bb_cfg["num_heads"],
            mlp_ratio=bb_cfg["mlp_ratio"],
        ).to(device)
        if ckpt_path is not None and key is not None:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            m.load_state_dict(ckpt[key])
        m.eval()
        return m

    models = [
        ("Random-Init", _build_vit(None, None)),
        ("SimCLR", _build_vit(simclr_ckpt, "backbone_state_dict")),
        ("DINO (Teacher)", _build_vit(dino_ckpt, "teacher_backbone_state_dict")),
    ]

    raw_batch = images[:num_images].to(device)
    norm_batch = normalize_cifar10(raw_batch)
    grid_side = 32 // bb_cfg["patch_size"]  # 8

    fig, axes = plt.subplots(num_images, 4, figsize=(8.5, 2.1 * num_images))
    if num_images == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("Input", fontsize=10)
    for c, (name, _) in enumerate(models, start=1):
        axes[0, c].set_title(name, fontsize=10)

    for r in range(num_images):
        axes[r, 0].imshow(raw_batch[r].permute(1, 2, 0).cpu().numpy())
        axes[r, 0].axis("off")

    for c, (_, model) in enumerate(models, start=1):
        _, attn = model.forward_with_attention(norm_batch)
        # Take [CLS] -> patch attention (drop CLS -> CLS at index 0), average across heads.
        cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)  # (B, 64)
        maps = cls_to_patch.reshape(num_images, 1, grid_side, grid_side)
        upsampled = F.interpolate(maps, size=(32, 32), mode="bilinear", align_corners=False)
        for r in range(num_images):
            m_np = upsampled[r, 0].cpu().numpy()
            axes[r, c].imshow(raw_batch[r].permute(1, 2, 0).cpu().numpy())
            axes[r, c].imshow(m_np, cmap="magma", alpha=0.55)
            axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SimCLR vs. DINO visualizations.")
    parser.add_argument("--config", type=str, default="configs/dino.yaml")
    parser.add_argument("--simclr-metrics", type=str, default="checkpoints/simclr/metrics_seed42.json")
    parser.add_argument("--dino-metrics", type=str, default="checkpoints/dino/metrics_seed42.json")
    parser.add_argument("--simclr-ckpt", type=str, default="checkpoints/simclr/best_seed42.pt")
    parser.add_argument("--dino-ckpt", type=str, default="checkpoints/dino/best_seed42.pt")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = load_cifar10_splits(
        data_dir=cfg["data"]["data_dir"],
        val_size=cfg["data"]["val_size"],
        train_subset=cfg["data"]["train_subset"],
        split_seed=cfg["data"]["split_seed"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
    )
    val_imgs, _ = next(iter(splits.val_loader))
    aug_cfg = cfg["augmentations"]
    augmentor = SSLAugmentor(
        global_size=aug_cfg["global_size"],
        global_scale=tuple(aug_cfg["global_scale"]),
        local_size=aug_cfg["local_size"],
        local_scale=tuple(aug_cfg["local_scale"]),
        n_local_crops=int(aug_cfg["n_local_crops"]),
        ratio=tuple(aug_cfg["ratio"]),
        flip_p=aug_cfg["flip_p"],
        jitter_p=aug_cfg["jitter_p"],
        jitter_strength=aug_cfg["jitter_strength"],
        grayscale_p=aug_cfg["grayscale_p"],
    )

    aug_out = save_augmentation_grid(val_imgs, augmentor)
    print(f"Saved {aug_out}")

    if Path(args.simclr_metrics).exists() and Path(args.dino_metrics).exists():
        curves_out = save_training_curves(args.simclr_metrics, args.dino_metrics)
        print(f"Saved {curves_out}")

    if Path(args.simclr_ckpt).exists() and Path(args.dino_ckpt).exists():
        attn_out = save_cls_attention_grid(val_imgs, args.simclr_ckpt, args.dino_ckpt, cfg)
        print(f"Saved {attn_out}")


if __name__ == "__main__":
    main()
