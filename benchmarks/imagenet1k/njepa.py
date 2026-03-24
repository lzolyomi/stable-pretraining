"""NJEPA pretraining on ImageNet-1K with 3-phase noising target encoder update.

Supports ViT-Small, ViT-Base, and ViT-Large via a single env var; all other
hyperparameters are also configurable via environment variables.

Usage (ViT-Base, defaults):
    python benchmarks/imagenet1k/njepa.py

Usage (ViT-Small):
    NJEPA_ENCODER_NAME=vit_small_patch16_224 python benchmarks/imagenet1k/njepa.py

Usage (ViT-Large):
    NJEPA_ENCODER_NAME=vit_large_patch16_224 python benchmarks/imagenet1k/njepa.py

Usage (vanilla I-JEPA, no noising):
    NJEPA_PHASE1_END=0 NJEPA_PHASE2_END=0 python benchmarks/imagenet1k/njepa.py

Environment variables:
    NJEPA_ENCODER_NAME          timm model name (default: vit_base_patch16_224)
    NJEPA_PREDICTOR_EMBED_DIM   Override predictor hidden dim (default: per-size table)
    NJEPA_PREDICTOR_DEPTH       Override predictor depth (default: per-size table)
    NJEPA_PHASE1_END            Last step of Phase 1 / noise-only (default: 1000)
    NJEPA_PHASE2_END            Last step of Phase 2 / EMA+noise (default: 10000)
    NJEPA_NOISE_SCALE           Phase-1 noise amplitude relative to param RMS (default: 0.05)
    NJEPA_NOISE_FLOOR_SCALE     Phase-2 noise amplitude relative to param RMS (default: 0.02)
    NJEPA_EMA_START             Initial EMA decay (default: 0.996)
    NJEPA_EMA_END               Final EMA decay (default: 1.0)
    NJEPA_BATCH_SIZE            Per-device batch size (default: 256)
    NJEPA_LR                    Base learning rate (default: 5e-4)
    NJEPA_EPOCHS                Training epochs (default: 300)
    NJEPA_NUM_WORKERS           DataLoader workers (default: 16)
    NJEPA_PRECISION             Lightning precision (default: 16-mixed)
    NJEPA_CKPT_EVERY            Save checkpoint every N epochs (default: 50)
    NJEPA_USE_WANDB             Set to "1" to enable W&B logging (default: 1)
    HF_IN1K_CACHE_DIR           ImageNet-1K HuggingFace cache dir
    HF_TOKEN                    HuggingFace access token
    HF_DATASET_REVISION         Optional dataset revision pin
    WANDB_ENTITY                W&B entity
    WANDB_PROJECT               W&B project (default: stable-pretraining)
"""

import os
import time
import types
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.data import transforms
from stable_pretraining.methods.njepa import NJEPA

# ---------------------------------------------------------------------------
# Per-size predictor defaults (matching I-JEPA paper)
# ---------------------------------------------------------------------------
PREDICTOR_DEFAULTS = {
    "vit_small_patch16_224": {
        "embed_dim": 384,
        "predictor_embed_dim": 192,
        "predictor_depth": 6,
        "short_name": "vits",
    },
    "vit_base_patch16_224": {
        "embed_dim": 768,
        "predictor_embed_dim": 384,
        "predictor_depth": 6,
        "short_name": "vitb",
    },
    "vit_large_patch16_224": {
        "embed_dim": 1024,
        "predictor_embed_dim": 384,
        "predictor_depth": 12,
        "short_name": "vitl",
    },
}


def build_hf_dataset(split, cache_dir, transform):
    kwargs = {
        "path": "ILSVRC/imagenet-1k",
        "split": split,
        "cache_dir": str(cache_dir),
        "token": os.environ.get("HF_TOKEN") or True,
        "transform": transform,
    }
    revision = os.environ.get("HF_DATASET_REVISION")
    if revision:
        kwargs["revision"] = revision
    return spt.data.HFDataset(**kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
encoder_name = os.environ.get("NJEPA_ENCODER_NAME", "vit_base_patch16_224")
if encoder_name not in PREDICTOR_DEFAULTS:
    raise ValueError(
        f"NJEPA_ENCODER_NAME='{encoder_name}' is not in the defaults table. "
        f"Known keys: {list(PREDICTOR_DEFAULTS)}. "
        "Set NJEPA_PREDICTOR_EMBED_DIM and NJEPA_PREDICTOR_DEPTH manually if needed."
    )

size_cfg = PREDICTOR_DEFAULTS[encoder_name]
embed_dim = size_cfg["embed_dim"]
predictor_embed_dim = int(os.environ.get("NJEPA_PREDICTOR_EMBED_DIM", size_cfg["predictor_embed_dim"]))
predictor_depth = int(os.environ.get("NJEPA_PREDICTOR_DEPTH", size_cfg["predictor_depth"]))
short_name = size_cfg["short_name"]

phase1_end = int(os.environ.get("NJEPA_PHASE1_END", "1000"))
phase2_end = int(os.environ.get("NJEPA_PHASE2_END", "10000"))
noise_scale = float(os.environ.get("NJEPA_NOISE_SCALE", "0.05"))
noise_floor_scale = float(os.environ.get("NJEPA_NOISE_FLOOR_SCALE", "0.02"))
ema_start = float(os.environ.get("NJEPA_EMA_START", "0.996"))
ema_end = float(os.environ.get("NJEPA_EMA_END", "1.0"))

batch_size = int(os.environ.get("NJEPA_BATCH_SIZE", "256"))
lr = float(os.environ.get("NJEPA_LR", "5e-4"))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))

num_workers = int(os.environ.get("NJEPA_NUM_WORKERS", "16"))
max_epochs = int(os.environ.get("NJEPA_EPOCHS", "300"))
precision = os.environ.get("NJEPA_PRECISION", "16-mixed")
ckpt_every = int(os.environ.get("NJEPA_CKPT_EVERY", "50"))

data_dir = Path(os.environ.get("HF_IN1K_CACHE_DIR", "/nfs-gpu/users_home/levizolyomi/hf-in1k"))
data_dir.mkdir(parents=True, exist_ok=True)

print(
    "NJEPA config:",
    {
        "encoder": encoder_name,
        "predictor_embed_dim": predictor_embed_dim,
        "predictor_depth": predictor_depth,
        "phase1_end": phase1_end,
        "phase2_end": phase2_end,
        "noise_scale": noise_scale,
        "noise_floor_scale": noise_floor_scale,
        "ema_start": ema_start,
        "ema_end": ema_end,
        "batch_size_per_device": batch_size,
        "num_nodes": num_nodes,
        "lr": lr,
        "max_epochs": max_epochs,
    },
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(**spt.data.static.ImageNet),
)
val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

train_dataset = build_hf_dataset("train", data_dir, train_transform)
val_dataset = build_hf_dataset("validation", data_dir, val_transform)

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
module = NJEPA(
    encoder_name=encoder_name,
    predictor_embed_dim=predictor_embed_dim,
    predictor_depth=predictor_depth,
    num_targets=4,
    ema_decay_start=ema_start,
    ema_decay_end=ema_end,
    phase1_end=phase1_end,
    phase2_end=phase2_end,
    noise_scale=noise_scale,
    noise_floor_scale=noise_floor_scale,
)


def njepa_forward(self, batch, stage):
    output = NJEPA.forward(self, batch["image"])
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    # Mean-pool across patch dimension: [B, N, D] -> [B, D]
    embedding = output.embedding.mean(dim=1)
    return {
        "loss": output.loss,
        "embedding": embedding,
        **({"label": batch["label"].long()} if "label" in batch else {}),
    }


module.forward = types.MethodType(njepa_forward, module)
module.optim = {
    "main": {
        "modules": "encoder.student|predictor",
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
}

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(embed_dim, 1000),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(1000),
        "top5": torchmetrics.classification.MulticlassAccuracy(1000, top_k=5),
    },
    optimizer={
        "type": "AdamW",
        "lr": 3e-3,
        "weight_decay": 1e-4,
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=10000,
    metrics={"top1": torchmetrics.classification.MulticlassAccuracy(1000)},
    input_dim=embed_dim,
    k=20,
)

ckpt_dir = Path(__file__).parent / "checkpoints" / f"njepa-{short_name}"

wandb_logger = False
if os.environ.get("NJEPA_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=f"njepa-{short_name}-{time.time():.0f}",
        log_model=False,
    )

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = pl.Trainer(
    max_epochs=max_epochs,
    num_sanity_val_steps=0,
    callbacks=[
        TeacherStudentCallback(),
        spt.callbacks.StepTimer(),
        linear_probe,
        knn_probe,
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"njepa-{short_name}-{{epoch:03d}}",
            save_top_k=-1,
            every_n_epochs=ckpt_every,
            save_last=True,
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
    precision=precision,
    logger=wandb_logger,
    devices="auto",
    num_nodes=num_nodes,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
