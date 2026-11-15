"""SimCLR training loop with step-based budget, cosine warmup, and val k-NN monitor."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from simclr_dino.augmentations import SSLAugmentor
from simclr_dino.data import load_cifar10_splits
from simclr_dino.diagnostics import compute_representation_diagnostics
from simclr_dino.eval import extract_features, weighted_knn_accuracy
from simclr_dino.simclr import SimCLRModel, nt_xent_loss
from simclr_dino.vit import TinyViT


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Apply weight decay to 2D+ weights only; keep biases, norms, and tokens at 0.0."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_warmup_lr(
    step: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_frac: float,
) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    warmup_steps = max(1, int(warmup_frac * max_steps))
    if step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_simclr(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    splits = load_cifar10_splits(
        data_dir=data_cfg["data_dir"],
        val_size=data_cfg["val_size"],
        train_subset=data_cfg["train_subset"],
        split_seed=data_cfg["split_seed"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    aug_cfg = cfg["augmentations"]
    augmentor = SSLAugmentor(
        global_size=aug_cfg["global_size"],
        global_scale=tuple(aug_cfg["global_scale"]),
        ratio=tuple(aug_cfg["ratio"]),
        flip_p=aug_cfg["flip_p"],
        jitter_p=aug_cfg["jitter_p"],
        jitter_strength=aug_cfg["jitter_strength"],
        grayscale_p=aug_cfg["grayscale_p"],
    )

    bb_cfg = cfg["backbone"]
    backbone = TinyViT(
        patch_size=bb_cfg["patch_size"],
        embed_dim=bb_cfg["embed_dim"],
        depth=bb_cfg["depth"],
        num_heads=bb_cfg["num_heads"],
        mlp_ratio=bb_cfg["mlp_ratio"],
    )
    sim_cfg = cfg["simclr"]
    model = SimCLRModel(
        backbone=backbone,
        embed_dim=bb_cfg["embed_dim"],
        head_hidden_dim=sim_cfg["head_hidden_dim"],
        proj_dim=sim_cfg["proj_dim"],
        head_norm=sim_cfg["head_norm"],
    ).to(device)

    tr_cfg = cfg["train"]
    max_steps = int(tr_cfg["max_steps"])
    optimizer = torch.optim.AdamW(
        build_param_groups(model, float(tr_cfg["weight_decay"])),
        lr=float(tr_cfg["lr"]),
    )

    ckpt_dir = Path(tr_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg["eval"]
    eval_interval = int(tr_cfg["eval_interval_steps"])
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1000)

    history: list[dict[str, float]] = []
    best_val_knn = -1.0
    train_iter = iter(splits.train_loader)
    t_start = time.perf_counter()

    for step in range(1, max_steps + 1):
        try:
            images, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(splits.train_loader)
            images, _ = next(train_iter)

        lr = cosine_warmup_lr(
            step=step,
            max_steps=max_steps,
            base_lr=float(tr_cfg["lr"]),
            min_lr=float(tr_cfg["min_lr"]),
            warmup_frac=float(tr_cfg["warmup_frac"]),
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        images = images.to(device)
        v1, v2 = augmentor.simclr_views(images, generator=aug_gen)

        model.train()
        optimizer.zero_grad()
        _, _, z1, z2 = model(v1, v2)
        loss = nt_xent_loss(z1, z2, temperature=float(sim_cfg["temperature"]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg["grad_clip"]))
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == max_steps:
            elapsed = time.perf_counter() - t_start
            print(
                f"[SimCLR seed={seed}] step {step}/{max_steps} | loss={loss.item():.4f} "
                f"| lr={lr:.2e} | elapsed={elapsed:.1f}s"
            )

        if eval_interval > 0 and (step % eval_interval == 0 or step == max_steps):
            bank_feats, bank_labels = extract_features(model.backbone, splits.bank_loader, device)
            val_feats, val_labels = extract_features(model.backbone, splits.val_loader, device)
            val_knn = weighted_knn_accuracy(
                bank_feats,
                bank_labels,
                val_feats,
                val_labels,
                k=eval_cfg["knn_k"],
                temperature=eval_cfg["knn_temperature"],
                chunk_size=eval_cfg["knn_chunk_size"],
            )
            diag = compute_representation_diagnostics(val_feats)
            record = {
                "step": float(step),
                "loss": loss.item(),
                "lr": lr,
                "val_knn_acc": val_knn,
                **diag,
            }
            history.append(record)
            print(
                f"  -> Val k-NN={val_knn * 100:.2f}% | std={diag['feature_std']:.4f} "
                f"| rankme={diag['rankme']:.2f}"
            )
            if val_knn > best_val_knn:
                best_val_knn = val_knn
                torch.save(
                    {
                        "step": step,
                        "seed": seed,
                        "val_knn_acc": val_knn,
                        "backbone_state_dict": model.backbone.state_dict(),
                        "config": cfg,
                    },
                    ckpt_dir / f"best_seed{seed}.pt",
                )

    total_time = time.perf_counter() - t_start
    torch.save(
        {
            "step": max_steps,
            "seed": seed,
            "val_knn_acc": best_val_knn,
            "backbone_state_dict": model.backbone.state_dict(),
            "config": cfg,
        },
        ckpt_dir / f"final_seed{seed}.pt",
    )
    summary = {
        "method": "simclr",
        "seed": seed,
        "max_steps": max_steps,
        "total_time_sec": total_time,
        "sec_per_step": total_time / max_steps,
        "best_val_knn_acc": best_val_knn,
        "history": history,
    }
    with open(ckpt_dir / f"metrics_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SimCLR on CIFAR-10.")
    parser.add_argument("--config", type=str, default="configs/simclr.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.train_subset is not None:
        cfg["data"]["train_subset"] = args.train_subset
    if args.temperature is not None:
        cfg["simclr"]["temperature"] = args.temperature
    if args.eval_interval is not None:
        cfg["train"]["eval_interval_steps"] = args.eval_interval
    if args.checkpoint_dir is not None:
        cfg["train"]["checkpoint_dir"] = args.checkpoint_dir

    train_simclr(cfg)


if __name__ == "__main__":
    main()
