"""Dataset builders for linear-probe evaluation.

Provides :func:`build_dataloaders` which returns a ``(train_loader, val_loader)``
pair for any supported dataset.  All datasets are resized to 224×224 and
normalised with ImageNet statistics, matching the pretraining pipeline.
"""

import torch
import torchvision.datasets as tvd

import stable_pretraining as spt
from stable_pretraining.data import transforms

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet100": 100,
}

AVAILABLE_DATASETS = list(NUM_CLASSES)


# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------
def _build_transforms():
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.ImageNet),
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )
    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_dataloaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> tuple:
    """Return ``(train_loader, val_loader)`` for *dataset*.

    Args:
        dataset:     One of ``'cifar10'``, ``'cifar100'``, ``'imagenet100'``.
        data_root:   Root directory passed to torchvision / HF datasets.
        batch_size:  Batch size per GPU.
        num_workers: DataLoader workers.
    """
    if dataset not in NUM_CLASSES:
        raise ValueError(f"dataset must be one of {AVAILABLE_DATASETS}, got '{dataset}'")

    train_tf, val_tf = _build_transforms()

    if dataset == "cifar10":
        train_ds = spt.data.FromTorchDataset(
            tvd.CIFAR10(data_root, train=True, download=True),
            names=["image", "label"],
            transform=train_tf,
        )
        val_ds = spt.data.FromTorchDataset(
            tvd.CIFAR10(data_root, train=False, download=True),
            names=["image", "label"],
            transform=val_tf,
        )
    elif dataset == "cifar100":
        train_ds = spt.data.FromTorchDataset(
            tvd.CIFAR100(data_root, train=True, download=True),
            names=["image", "label"],
            transform=train_tf,
        )
        val_ds = spt.data.FromTorchDataset(
            tvd.CIFAR100(data_root, train=False, download=True),
            names=["image", "label"],
            transform=val_tf,
        )
    elif dataset == "imagenet100":
        train_ds = spt.data.HFDataset(
            "clane9/imagenet-100",
            split="train",
            transform=train_tf,
        )
        val_ds = spt.data.HFDataset(
            "clane9/imagenet-100",
            split="validation",
            transform=val_tf,
        )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_ds, **loader_kwargs)

    return train_loader, val_loader
