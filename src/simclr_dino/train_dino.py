"""DINO training loop with Student-Teacher EMA, centering, sharpening, and val k-NN monitor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from simclr_dino.augmentations import SSLAugmentor
from simclr_dino.data import load_cifar10_splits
from simclr_dino.diagnostics import compute_representation_diagnostics
from simclr_dino.dino import DINOLoss, DINOModel, cosine_ema_momentum
from simclr_dino.eval import extract_features, weighted_knn_accuracy
from simclr_dino.train_simclr import build_param_groups, cosine_warmup_lr
from simclr_dino.vit import TinyViT


def train_dino(cfg: dict[str, Any]) -> dict[str, Any]:
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
        local_size=aug_cfg["local_size"],
        local_scale=tuple(aug_cfg["local_scale"]),
        n_local_crops=int(aug_cfg["n_local_crops"]),
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
    dino_cfg = cfg["dino"]
    model = DINOModel(
        backbone=backbone,
        embed_dim=bb_cfg["embed_dim"],
        head_hidden_dim=dino_cfg["head_hidden_dim"],
        bottleneck_dim=dino_cfg["bottleneck_dim"],
        num_prototypes=dino_cfg["num_prototypes"],
        freeze_g=bool(dino_cfg["freeze_g"]),
    ).to(device)

    tr_cfg = cfg["train"]
    max_steps = int(tr_cfg["max_steps"])
    criterion = DINOLoss(
        num_prototypes=int(dino_cfg["num_prototypes"]),
        student_temp=float(dino_cfg["student_temp"]),
        teacher_temp_start=float(dino_cfg["teacher_temp_start"]),
        teacher_temp_end=float(dino_cfg["teacher_temp_end"]),
        teacher_temp_warmup_steps=int(float(dino_cfg["teacher_temp_warmup_frac"]) * max_steps),
        center_momentum=float(dino_cfg["center_momentum"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        build_param_groups(model.student, float(tr_cfg["weight_decay"])),
        lr=float(tr_cfg["lr"]),
    )

    ckpt_dir = Path(tr_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg["eval"]
    eval_interval = int(tr_cfg["eval_interval_steps"])
    freeze_last_steps = int(dino_cfg["freeze_last_layer_steps"])
    ema_base = float(dino_cfg["ema_momentum_base"])
    aug_gen = torch.Generator(device=device).manual_seed(seed + 2000)

    history: list[dict[str, float]] = []
    best_teacher_knn = -1.0
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
        globals_, locals_ = augmentor.dino_views(images, generator=aug_gen)

        model.student.train()
        model.teacher.eval()
        with torch.no_grad():
            teacher_outs = model.teacher.forward_views(globals_)
        student_outs = model.student.forward_views(globals_ + locals_)

        loss, loss_metrics = criterion(student_outs, teacher_outs, step=step - 1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), float(tr_cfg["grad_clip"]))
        if step <= freeze_last_steps:
            model.cancel_last_layer_gradients()
        optimizer.step()

        ema_m = cosine_ema_momentum(step=step, total_steps=max_steps, base_momentum=ema_base)
        model.update_teacher_ema(ema_m)

        if step == 1 or step % 50 == 0 or step == max_steps:
            elapsed = time.perf_counter() - t_start
            print(
                f"[DINO seed={seed}] step {step}/{max_steps} | loss={loss.item():.4f} "
                f"| H_t={loss_metrics['teacher_entropy']:.3f} | KL={loss_metrics['kl_div']:.3f} "
                f"| m={ema_m:.4f} | elapsed={elapsed:.1f}s"
            )

        if eval_interval > 0 and (step % eval_interval == 0 or step == max_steps):
            t_bank_f, t_bank_y = extract_features(model.teacher.backbone, splits.bank_loader, device)
            t_val_f, t_val_y = extract_features(model.teacher.backbone, splits.val_loader, device)
            teacher_knn = weighted_knn_accuracy(
                t_bank_f,
                t_bank_y,
                t_val_f,
                t_val_y,
                k=eval_cfg["knn_k"],
                temperature=eval_cfg["knn_temperature"],
                chunk_size=eval_cfg["knn_chunk_size"],
            )

            s_bank_f, s_bank_y = extract_features(model.student.backbone, splits.bank_loader, device)
            s_val_f, s_val_y = extract_features(model.student.backbone, splits.val_loader, device)
            student_knn = weighted_knn_accuracy(
                s_bank_f,
                s_bank_y,
                s_val_f,
                s_val_y,
                k=eval_cfg["knn_k"],
                temperature=eval_cfg["knn_temperature"],
                chunk_size=eval_cfg["knn_chunk_size"],
            )

            diag = compute_representation_diagnostics(t_val_f)
            record = {
                "step": float(step),
                "loss": loss.item(),
                "lr": lr,
                "teacher_val_knn_acc": teacher_knn,
                "student_val_knn_acc": student_knn,
                "teacher_entropy": loss_metrics["teacher_entropy"],
                "kl_div": loss_metrics["kl_div"],
                **diag,
            }
            history.append(record)
            print(
                f"  -> Teacher k-NN={teacher_knn * 100:.2f}% | Student k-NN={student_knn * 100:.2f}% "
                f"| std={diag['feature_std']:.4f} | rankme={diag['rankme']:.2f}"
            )
            if teacher_knn > best_teacher_knn:
                best_teacher_knn = teacher_knn
                torch.save(
                    {
                        "step": step,
                        "seed": seed,
                        "teacher_val_knn_acc": teacher_knn,
                        "student_val_knn_acc": student_knn,
                        "teacher_backbone_state_dict": model.teacher.backbone.state_dict(),
                        "student_backbone_state_dict": model.student.backbone.state_dict(),
                        "config": cfg,
                    },
                    ckpt_dir / f"best_seed{seed}.pt",
                )

    total_time = time.perf_counter() - t_start
    torch.save(
        {
            "step": max_steps,
            "seed": seed,
            "teacher_val_knn_acc": best_teacher_knn,
            "teacher_backbone_state_dict": model.teacher.backbone.state_dict(),
            "student_backbone_state_dict": model.student.backbone.state_dict(),
            "config": cfg,
        },
        ckpt_dir / f"final_seed{seed}.pt",
    )
    summary = {
        "method": "dino",
        "seed": seed,
        "max_steps": max_steps,
        "total_time_sec": total_time,
        "sec_per_step": total_time / max_steps,
        "best_teacher_val_knn_acc": best_teacher_knn,
        "history": history,
    }
    with open(ckpt_dir / f"metrics_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINO on CIFAR-10.")
    parser.add_argument("--config", type=str, default="configs/dino.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--n-local-crops", type=int, default=None)
    parser.add_argument("--num-prototypes", type=int, default=None)
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
    if args.n_local_crops is not None:
        cfg["augmentations"]["n_local_crops"] = args.n_local_crops
    if args.num_prototypes is not None:
        cfg["dino"]["num_prototypes"] = args.num_prototypes
    if args.eval_interval is not None:
        cfg["train"]["eval_interval_steps"] = args.eval_interval
    if args.checkpoint_dir is not None:
        cfg["train"]["checkpoint_dir"] = args.checkpoint_dir

    train_dino(cfg)


if __name__ == "__main__":
    main()
