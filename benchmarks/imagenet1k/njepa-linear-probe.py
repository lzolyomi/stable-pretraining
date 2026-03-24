"""Linear probing benchmark for NJEPA checkpoint.

Trains a frozen NJEPA backbone + linear classifier (BN + Linear) on one of:
  CIFAR-10, CIFAR-100, Oxford-IIIT-Pets, ImageNet-100.

Usage:
    DATASET=cifar10      python benchmarks/imagenet1k/njepa-linear-probe.py
    DATASET=cifar100     python benchmarks/imagenet1k/njepa-linear-probe.py
    DATASET=pets         python benchmarks/imagenet1k/njepa-linear-probe.py
    DATASET=imagenet100  python benchmarks/imagenet1k/njepa-linear-probe.py

Environment variables:
    NJEPA_CKPT          Path to .ckpt file (required)
    NJEPA_ENCODER_NAME  timm model name (default: vit_base_patch16_224)
                        Used to determine embed_dim automatically.
    DATASET             One of: cifar10, cifar100, pets, imagenet100 (default: cifar10)
    DATA_ROOT           Root dir for torchvision datasets (default: /nfs-gpu/datasets)
    LP_EPOCHS           Training epochs (default: 100)
    LP_BATCH_SIZE       Batch size per GPU (default: 256)
    LP_NUM_WORKERS      DataLoader workers (default: 8)
    LP_LR               Learning rate for linear head (default: 1e-3)
    LP_USE_WANDB        Set to "1" to enable W&B logging (default: 0)
"""

import os
import types
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision.datasets as tvd
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.njepa import NJEPA

from loguru import logger as _logger
_logger.disable("stable_pretraining.callbacks.cpu_offload")
_logger.disable("stable_pretraining.callbacks.checkpoint_sklearn")

# ---------------------------------------------------------------------------
# Per-size embed dim lookup (mirrors njepa.py)
# ---------------------------------------------------------------------------
EMBED_DIM_DEFAULTS = {
    "vit_small_patch16_224": 384,
    "vit_base_patch16_224": 768,
    "vit_large_patch16_224": 1024,
}
PREDICTOR_DEFAULTS = {
    "vit_small_patch16_224": {"predictor_embed_dim": 192, "predictor_depth": 6},
    "vit_base_patch16_224": {"predictor_embed_dim": 384, "predictor_depth": 6},
    "vit_large_patch16_224": {"predictor_embed_dim": 384, "predictor_depth": 12},
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

CKPT_PATH = os.environ.get("NJEPA_CKPT")
if CKPT_PATH is None:
    raise ValueError(
        "NJEPA_CKPT environment variable must point to a .ckpt file. "
        "Example: NJEPA_CKPT=checkpoints/njepa-vitb/njepa-vitb-epoch=299.ckpt"
    )

encoder_name = os.environ.get("NJEPA_ENCODER_NAME", "vit_base_patch16_224")
if encoder_name not in EMBED_DIM_DEFAULTS:
    raise ValueError(
        f"NJEPA_ENCODER_NAME='{encoder_name}' not in the defaults table. "
        f"Known keys: {list(EMBED_DIM_DEFAULTS)}"
    )

EMBED_DIM = EMBED_DIM_DEFAULTS[encoder_name]
pred_cfg = PREDICTOR_DEFAULTS[encoder_name]

DATASET = os.environ.get("DATASET", "cifar10")
DATA_ROOT = os.environ.get("DATA_ROOT", "/nfs-gpu/datasets")
EPOCHS = int(os.environ.get("LP_EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("LP_BATCH_SIZE", "256"))
NUM_WORKERS = int(os.environ.get("LP_NUM_WORKERS", "8"))
LR = float(os.environ.get("LP_LR", "1e-3"))

NUM_CLASSES_MAP = {"cifar10": 10, "cifar100": 100, "pets": 37, "imagenet100": 100}
if DATASET not in NUM_CLASSES_MAP:
    raise ValueError(f"DATASET must be one of {list(NUM_CLASSES_MAP)}, got '{DATASET}'")
num_classes = NUM_CLASSES_MAP[DATASET]
probe_name = f"linear_probe_{DATASET}"

print(
    "NJEPA linear probe config:",
    {
        "ckpt": CKPT_PATH,
        "encoder": encoder_name,
        "embed_dim": EMBED_DIM,
        "dataset": DATASET,
        "num_classes": num_classes,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    },
)

# ---------------------------------------------------------------------------
# Load NJEPA backbone from checkpoint
# ---------------------------------------------------------------------------
njepa = NJEPA(
    encoder_name=encoder_name,
    predictor_embed_dim=pred_cfg["predictor_embed_dim"],
    predictor_depth=pred_cfg["predictor_depth"],
    num_targets=4,
)

print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
state_dict = ckpt.get("state_dict", ckpt)
njepa.load_state_dict(state_dict, strict=False)

njepa.requires_grad_(False)
njepa.optim = None


def probe_forward(self, batch, stage):
    with torch.no_grad():
        self.encoder.student.eval()
        out = self.encoder.student(batch["image"])
    return {
        "embedding": out.encoded[:, 0, :].detach(),  # CLS token [B, D]
        "label": batch["label"].long(),
    }


njepa.forward = types.MethodType(probe_forward, njepa)

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
if DATASET == "cifar10":
    train_ds = spt.data.FromTorchDataset(
        tvd.CIFAR10(DATA_ROOT, train=True, download=True),
        names=["image", "label"],
        transform=train_transform,
    )
    val_ds = spt.data.FromTorchDataset(
        tvd.CIFAR10(DATA_ROOT, train=False, download=True),
        names=["image", "label"],
        transform=val_transform,
    )

elif DATASET == "cifar100":
    train_ds = spt.data.FromTorchDataset(
        tvd.CIFAR100(DATA_ROOT, train=True, download=True),
        names=["image", "label"],
        transform=train_transform,
    )
    val_ds = spt.data.FromTorchDataset(
        tvd.CIFAR100(DATA_ROOT, train=False, download=True),
        names=["image", "label"],
        transform=val_transform,
    )

elif DATASET == "pets":
    train_ds = spt.data.FromTorchDataset(
        tvd.OxfordIIITPet(DATA_ROOT, split="trainval", target_types="category", download=True),
        names=["image", "label"],
        transform=train_transform,
    )
    val_ds = spt.data.FromTorchDataset(
        tvd.OxfordIIITPet(DATA_ROOT, split="test", target_types="category", download=True),
        names=["image", "label"],
        transform=val_transform,
    )

elif DATASET == "imagenet100":
    train_ds = spt.data.HFDataset(
        "clane9/imagenet-100",
        split="train",
        transform=train_transform,
    )
    val_ds = spt.data.HFDataset(
        "clane9/imagenet-100",
        split="validation",
        transform=val_transform,
    )

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=NUM_WORKERS > 0,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
)
data = spt.data.DataModule(train=train_loader, val=val_loader)

# ---------------------------------------------------------------------------
# Linear probe callback
# ---------------------------------------------------------------------------
probe = spt.callbacks.OnlineProbe(
    njepa,
    name=probe_name,
    input="embedding",
    target="label",
    probe=nn.Sequential(
        nn.BatchNorm1d(EMBED_DIM),
        nn.Linear(EMBED_DIM, num_classes),
    ),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
        "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5),
    },
    optimizer={"type": "AdamW", "lr": LR, "weight_decay": 1e-4},
    scheduler={"type": "CosineAnnealingLR", "T_max": EPOCHS},
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
wandb_logger = None
if os.environ.get("LP_USE_WANDB", "0") == "1":
    from lightning.pytorch.loggers import WandbLogger
    wandb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=f"njepa-lp-{DATASET}",
        log_model=False,
    )

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
num_gpus = torch.cuda.device_count() or 1
ckpt_dir = SCRIPT_DIR / "checkpoints" / f"njepa-lp-{DATASET}"

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    num_sanity_val_steps=0,
    callbacks=[
        probe,
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"njepa-lp-{DATASET}-{{epoch:03d}}",
            save_last=True,
            save_top_k=1,
            monitor=f"eval/{probe_name}_top1",
            mode="max",
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ],
    precision="16-mixed" if torch.cuda.is_available() else 32,
    logger=wandb_logger,
    devices=num_gpus,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
)

manager = spt.Manager(trainer=trainer, module=njepa, data=data)
manager()
