"""VICReg pretraining on ImageNet-100 with ViT-Base.

Uses 2-view augmentation following the original VICReg paper.
The backbone is a timm ViT-Base returning the 768-dim CLS token.

Usage (defaults):
    python benchmarks/imagenet100/vicreg-vit-base.py

Environment variables:
    VICREG_LR                   Learning rate (default: 5e-4)
    VICREG_WEIGHT_DECAY         Weight decay (default: 0.05)
    VICREG_BATCH_SIZE           Per-device batch size (default: 256)
    VICREG_EPOCHS               Training epochs (default: 300)
    VICREG_NUM_WORKERS          DataLoader workers (default: 16)
    VICREG_PRECISION            Lightning precision (default: 16-mixed)
    VICREG_CKPT_EVERY           Save checkpoint every N epochs (default: 50)
    VICREG_SEED                 Random seed for reproducibility (default: 42)
    VICREG_SIM_COEFF            Invariance loss coefficient (default: 25.0)
    VICREG_STD_COEFF            Variance loss coefficient (default: 25.0)
    VICREG_COV_COEFF            Covariance loss coefficient (default: 1.0)
    VICREG_RUN_NAME             Base name for the run (default: vicreg)
    VICREG_USE_WANDB            Set to "1" to enable W&B logging (default: 1)
    HF_IN100_CACHE_DIR          ImageNet-100 HuggingFace cache dir
    SLURM_NNODES                Number of nodes (set automatically by SLURM)
    WANDB_ENTITY                W&B entity
    WANDB_PROJECT               W&B project (default: stable-pretraining)
"""

import os
from datetime import datetime
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
lr = float(os.environ.get("VICREG_LR", "5e-4"))
weight_decay = float(os.environ.get("VICREG_WEIGHT_DECAY", "0.05"))
batch_size = int(os.environ.get("VICREG_BATCH_SIZE", "256"))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
num_workers = int(os.environ.get("VICREG_NUM_WORKERS", "16"))
max_epochs = int(os.environ.get("VICREG_EPOCHS", "300"))
precision = os.environ.get("VICREG_PRECISION", "16-mixed")
ckpt_every = int(os.environ.get("VICREG_CKPT_EVERY", "50"))
seed = int(os.environ.get("VICREG_SEED", "42"))

sim_coeff = float(os.environ.get("VICREG_SIM_COEFF", "25.0"))
std_coeff = float(os.environ.get("VICREG_STD_COEFF", "25.0"))
cov_coeff = float(os.environ.get("VICREG_COV_COEFF", "1.0"))

data_dir = Path(os.environ.get("HF_IN100_CACHE_DIR", "/nfs-gpu/users_home/levizolyomi/hf-in100"))
data_dir.mkdir(parents=True, exist_ok=True)

print(
    "VICReg config:",
    {
        "encoder": "vit_base_patch16_224",
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size_per_device": batch_size,
        "num_nodes": num_nodes,
        "max_epochs": max_epochs,
        "sim_coeff": sim_coeff,
        "std_coeff": std_coeff,
        "cov_coeff": cov_coeff,
    },
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
vicreg_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

train_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="train",
    cache_dir=str(data_dir),
    transform=vicreg_transform,
)
val_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
    persistent_workers=num_workers > 0,
    pin_memory=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    pin_memory=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# num_classes=0 makes timm return the 768-dim CLS token directly
backbone = spt.backbone.from_timm("vit_base_patch16_224", num_classes=0, pretrained=False)

projector = nn.Sequential(
    nn.Linear(768, 8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(inplace=True),
    nn.Linear(8192, 8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(inplace=True),
    nn.Linear(8192, 8192, bias=False),
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.vicreg_forward,
    vicreg_loss=spt.losses.VICRegLoss(
        sim_coeff=sim_coeff,
        std_coeff=std_coeff,
        cov_coeff=cov_coeff,
    ),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(768, 100),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
    },
    optimizer={"type": "AdamW", "lr": 3e-3, "weight_decay": 1e-4},
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"top1": torchmetrics.classification.MulticlassAccuracy(100)},
    input_dim=768,
    k=20,
)

rankme = spt.callbacks.RankMe(
    name="rankme",
    target="embedding",
    queue_length=2048,
    target_shape=768,
)

_dt_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
predefined_run_name = os.environ.get("VICREG_RUN_NAME", "vicreg")
run_name = f"{predefined_run_name}-vitb-{_dt_suffix}"
ckpt_dir = Path(__file__).parent / "checkpoints" / run_name

print(f">>>>> CKPT_DIR: {ckpt_dir}")

wandb_logger = False
if os.environ.get("VICREG_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=run_name,
        log_model=False,
    )
    wandb_logger.log_hyperparams({
        "encoder": "vit_base_patch16_224",
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size_per_device": batch_size,
        "num_nodes": num_nodes,
        "max_epochs": max_epochs,
        "sim_coeff": sim_coeff,
        "std_coeff": std_coeff,
        "cov_coeff": cov_coeff,
        "precision": precision,
    })

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = pl.Trainer(
    max_epochs=max_epochs,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=5,
    callbacks=[
        spt.callbacks.StepTimer(),
        linear_probe,
        knn_probe,
        rankme,
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{run_name}-{{epoch:03d}}",
            save_top_k=-1,
            every_n_epochs=ckpt_every,
            save_last=True,
            save_on_train_epoch_end=True,
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
    precision=precision,
    logger=wandb_logger,
    devices="auto",
    num_nodes=num_nodes,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    sync_batchnorm=True,
)

manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
manager()
