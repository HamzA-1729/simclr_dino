"""CIFAR-10 dataset splitting (45k train / 5k val / 10k test) with zero leakage."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


def build_stratified_indices(
    targets: Sequence[int],
    val_size: int = 5000,
    train_subset: int | None = None,
    split_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Create deterministic, class-balanced ``(train_indices, val_indices, full_train_indices)``.

    ``val_indices`` is carved first from the full training set; ``train_subset`` (if set)
    is sampled strictly from ``full_train_indices`` so validation is never leaked.
    """
    rng = random.Random(split_seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(int(label), []).append(idx)

    num_classes = len(by_class)
    if val_size % num_classes != 0:
        raise ValueError(f"val_size ({val_size}) must be divisible by num_classes ({num_classes}).")
    val_per_class = val_size // num_classes

    val_indices: list[int] = []
    full_train_indices: list[int] = []
    train_by_class: dict[int, list[int]] = {}

    for cls in sorted(by_class):
        cls_indices = list(by_class[cls])
        rng.shuffle(cls_indices)
        if len(cls_indices) <= val_per_class:
            raise ValueError(f"Class {cls} has only {len(cls_indices)} items, cannot take {val_per_class} for val.")
        val_indices.extend(cls_indices[:val_per_class])
        rem = cls_indices[val_per_class:]
        full_train_indices.extend(rem)
        train_by_class[cls] = rem

    if train_subset is None:
        train_indices = list(full_train_indices)
    else:
        if train_subset % num_classes != 0:
            raise ValueError(
                f"train_subset ({train_subset}) must be divisible by num_classes ({num_classes})."
            )
        sub_per_class = train_subset // num_classes
        train_indices = []
        for cls in sorted(train_by_class):
            if len(train_by_class[cls]) < sub_per_class:
                raise ValueError(
                    f"Class {cls} has only {len(train_by_class[cls])} train items, cannot take {sub_per_class}."
                )
            train_indices.extend(train_by_class[cls][:sub_per_class])

    rng.shuffle(val_indices)
    rng.shuffle(full_train_indices)
    rng.shuffle(train_indices)
    return train_indices, val_indices, full_train_indices


class TensorImageDataset(Dataset):
    """In-memory uint8 ``(N, 3, 32, 32)`` dataset returning float ``[0, 1]`` images and labels."""

    def __init__(self, images_uint8: torch.Tensor, labels: Sequence[int]) -> None:
        self.images = images_uint8
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].to(torch.float32).div_(255.0)
        return img, self.labels[idx]


@dataclass(frozen=True)
class CIFAR10Splits:
    """Container for train, k-NN bank, val, and test DataLoaders."""

    train_loader: DataLoader
    bank_loader: DataLoader
    full_bank_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    val_size: int
    test_size: int


def _make_dataset_from_cifar(base: CIFAR10, indices: Sequence[int]) -> TensorImageDataset:
    raw = torch.from_numpy(base.data[list(indices)]).permute(0, 3, 1, 2).contiguous()
    labels = [int(base.targets[i]) for i in indices]
    return TensorImageDataset(raw, labels)


def load_cifar10_splits(
    data_dir: str | Path = "data",
    val_size: int = 5000,
    train_subset: int | None = 5000,
    split_seed: int = 42,
    batch_size: int = 256,
    num_workers: int = 2,
) -> CIFAR10Splits:
    """Load CIFAR-10 and return stratified ``train / bank / val / test`` DataLoaders."""
    root = str(data_dir)
    base_train = CIFAR10(root=root, train=True, download=True)
    base_test = CIFAR10(root=root, train=False, download=True)

    train_idx, val_idx, full_train_idx = build_stratified_indices(
        targets=base_train.targets,
        val_size=val_size,
        train_subset=train_subset,
        split_seed=split_seed,
    )

    train_ds = _make_dataset_from_cifar(base_train, train_idx)
    full_bank_ds = _make_dataset_from_cifar(base_train, full_train_idx)
    val_ds = _make_dataset_from_cifar(base_train, val_idx)
    test_ds = _make_dataset_from_cifar(base_test, list(range(len(base_test))))

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    bank_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    full_bank_loader = DataLoader(
        full_bank_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return CIFAR10Splits(
        train_loader=train_loader,
        bank_loader=bank_loader,
        full_bank_loader=full_bank_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )
