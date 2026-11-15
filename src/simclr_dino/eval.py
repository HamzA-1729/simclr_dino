"""Chunked weighted k-NN, frozen Linear Probe, and multi-seed random-init baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from simclr_dino.augmentations import normalize_cifar10
from simclr_dino.data import load_cifar10_splits
from simclr_dino.diagnostics import compute_representation_diagnostics
from simclr_dino.vit import TinyViT


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract unaugmented ``[CLS]`` features ``h`` and labels on CPU."""
    was_training = backbone.training
    backbone.eval()
    feats_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    try:
        for images, labels in loader:
            images = normalize_cifar10(images.to(device))
            feats = backbone(images).cpu()
            feats_list.append(feats)
            labels_list.append(labels.cpu())
    finally:
        backbone.train(was_training)
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def weighted_knn_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = 10,
    chunk_size: int = 512,
) -> float:
    """Chunked cosine-similarity weighted k-NN accuracy (DINO protocol: k=20, tau=0.07)."""
    train_norm = F.normalize(train_features.float(), dim=-1)
    query_norm = F.normalize(query_features.float(), dim=-1)
    k_eff = min(k, train_norm.shape[0])

    correct = 0
    total = query_norm.shape[0]
    for start in range(0, total, chunk_size):
        q_chunk = query_norm[start : start + chunk_size]
        y_chunk = query_labels[start : start + chunk_size]
        sim = torch.matmul(q_chunk, train_norm.T)
        topk_sim, topk_idx = sim.topk(k_eff, dim=-1, largest=True, sorted=True)
        weights = torch.exp(topk_sim / temperature)
        retrieved_labels = train_labels[topk_idx]
        one_hot = F.one_hot(retrieved_labels, num_classes=num_classes).to(weights.dtype)
        votes = (one_hot * weights.unsqueeze(-1)).sum(dim=1)
        preds = votes.argmax(dim=-1)
        correct += int((preds == y_chunk).sum().item())

    return correct / max(1, total)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int = 10,
    lr: float = 1e-2,
    epochs: int = 100,
    weight_decay_grid: tuple[float, ...] = (0.0, 1e-4, 1e-2),
    standardize: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a linear classifier on cached, unaugmented frozen ``[CLS]`` features,
    selecting ``weight_decay`` strictly on the validation split.
    """
    x_tr = train_features.float()
    x_va = val_features.float()
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    if standardize:
        mean = x_tr.mean(dim=0, keepdim=True)
        std = x_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
        x_tr = (x_tr - mean) / std
        x_va = (x_va - mean) / std

    in_dim = x_tr.shape[1]
    best_val_acc = -1.0
    best_wd = weight_decay_grid[0]
    best_state: dict[str, torch.Tensor] | None = None

    for wd in weight_decay_grid:
        torch.manual_seed(seed)
        probe = nn.Linear(in_dim, num_classes)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            logits = probe(x_tr)
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            opt.step()

        with torch.no_grad():
            val_acc = (probe(x_va).argmax(dim=-1) == val_labels).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wd = wd
            best_state = copy.deepcopy(probe.state_dict())

    best_probe = nn.Linear(in_dim, num_classes)
    assert best_state is not None
    best_probe.load_state_dict(best_state)
    return {
        "probe": best_probe,
        "val_acc": best_val_acc,
        "best_weight_decay": best_wd,
        "mean": mean,
        "std": std,
    }


@torch.no_grad()
def evaluate_linear_probe(
    probe: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> float:
    """Evaluate a trained linear probe on frozen features."""
    x = features.float()
    if mean is not None and std is not None:
        x = (x - mean) / std
    return (probe(x).argmax(dim=-1) == labels).float().mean().item()


def evaluate_random_init_baseline(
    backbone_fn: Callable[[int], nn.Module],
    bank_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    seeds: tuple[int, ...] = (42, 1337),
    k: int = 20,
    temperature: float = 0.07,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Compute multi-seed random-init k-NN baseline on validation split."""
    per_seed: dict[int, float] = {}
    for seed in seeds:
        backbone = backbone_fn(seed).to(device)
        bank_feats, bank_labels = extract_features(backbone, bank_loader, device)
        val_feats, val_labels = extract_features(backbone, val_loader, device)
        acc = weighted_knn_accuracy(
            bank_feats,
            bank_labels,
            val_feats,
            val_labels,
            k=k,
            temperature=temperature,
            chunk_size=chunk_size,
        )
        per_seed[seed] = acc
    mean_acc = sum(per_seed.values()) / len(per_seed)
    return {"mean_knn_acc": mean_acc, "per_seed": per_seed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSL checkpoints or random-init baseline.")
    parser.add_argument("--config", type=str, default="configs/simclr.yaml")
    parser.add_argument("--baseline", action="store_true", help="Run multi-seed random-init baseline.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to saved model checkpoint.")
    parser.add_argument("--branch", type=str, default="teacher", choices=["backbone", "teacher", "student"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--full-bank", action="store_true", help="Use full 45k train bank instead of train_subset.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_cifar10_splits(
        data_dir=cfg["data"]["data_dir"],
        val_size=cfg["data"]["val_size"],
        train_subset=cfg["data"]["train_subset"],
        split_seed=cfg["data"]["split_seed"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    bank_loader = splits.full_bank_loader if args.full_bank else splits.bank_loader
    query_loader = splits.val_loader if args.split == "val" else splits.test_loader

    bb_cfg = cfg["backbone"]
    eval_cfg = cfg["eval"]

    def _make_backbone(seed: int) -> TinyViT:
        torch.manual_seed(seed)
        return TinyViT(
            patch_size=bb_cfg["patch_size"],
            embed_dim=bb_cfg["embed_dim"],
            depth=bb_cfg["depth"],
            num_heads=bb_cfg["num_heads"],
            mlp_ratio=bb_cfg["mlp_ratio"],
        )

    if args.baseline:
        res = evaluate_random_init_baseline(
            backbone_fn=_make_backbone,
            bank_loader=bank_loader,
            val_loader=query_loader,
            device=device,
            seeds=tuple(cfg.get("seeds", [42, 1337])),
            k=eval_cfg["knn_k"],
            temperature=eval_cfg["knn_temperature"],
            chunk_size=eval_cfg["knn_chunk_size"],
        )
        print(json.dumps({"baseline_random_init": res, "split": args.split}, indent=2))
        return

    if args.checkpoint is None:
        raise ValueError("Provide either --baseline or --checkpoint.")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    backbone = _make_backbone(0).to(device)
    if "backbone_state_dict" in ckpt:
        backbone.load_state_dict(ckpt["backbone_state_dict"])
    elif args.branch == "teacher" and "teacher_backbone_state_dict" in ckpt:
        backbone.load_state_dict(ckpt["teacher_backbone_state_dict"])
    elif args.branch == "student" and "student_backbone_state_dict" in ckpt:
        backbone.load_state_dict(ckpt["student_backbone_state_dict"])
    else:
        raise KeyError(f"Could not find matching backbone state dict in {args.checkpoint}.")

    bank_feats, bank_labels = extract_features(backbone, bank_loader, device)
    val_feats, val_labels = extract_features(backbone, splits.val_loader, device)
    query_feats, query_labels = (
        (val_feats, val_labels)
        if args.split == "val"
        else extract_features(backbone, query_loader, device)
    )

    knn_acc = weighted_knn_accuracy(
        bank_feats,
        bank_labels,
        query_feats,
        query_labels,
        k=eval_cfg["knn_k"],
        temperature=eval_cfg["knn_temperature"],
        chunk_size=eval_cfg["knn_chunk_size"],
    )
    probe_res = train_linear_probe(
        bank_feats,
        bank_labels,
        val_feats,
        val_labels,
        lr=eval_cfg["probe_lr"],
        epochs=eval_cfg["probe_epochs"],
        weight_decay_grid=tuple(eval_cfg["probe_weight_decay_grid"]),
    )
    probe_acc = evaluate_linear_probe(
        probe_res["probe"],
        query_feats,
        query_labels,
        mean=probe_res["mean"],
        std=probe_res["std"],
    )
    diag = compute_representation_diagnostics(query_feats)

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "split": args.split,
        "knn_acc": knn_acc,
        "linear_probe_acc": probe_acc,
        "probe_best_wd": probe_res["best_weight_decay"],
        **diag,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
