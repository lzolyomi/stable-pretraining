"""SALT Stage 2 (student-teacher JEPA pretraining) on ImageNet-100 with ViT-Base.

Loads a Stage 1 MAE checkpoint as the teacher and trains a student encoder
with a cross-attention predictor using masked patch prediction.

Usage (with teacher checkpoint):
    SALT_TEACHER_CKPT=path/to/stage1.ckpt python benchmarks/imagenet100/salt-stage2-vit-base.py

Usage (no teacher, random init):
    python benchmarks/imagenet100/salt-stage2-vit-base.py

Environment variables:
    SALT_TEACHER_CKPT                   Path to Stage 1 checkpoint (optional; random init if unset)
    SALT_BATCH_SIZE                     Per-device batch size (default: 256)
    SALT_LR                             Learning rate (default: 5e-4)
    SALT_NUM_WORKERS                    DataLoader workers (default: 16)
    SALT_PREDICTOR_EMBED_DIM            Predictor hidden dim (default: 384)
    SALT_PREDICTOR_DEPTH                Predictor depth (default: 12)
    SALT_PREDICTOR_NUM_HEADS            Predictor attention heads (default: 16)
    SALT_NUM_TARGETS                    Number of masking targets (default: 4)
    SALT_STAGE2_EPOCHS                  Training epochs (default: 400)
    SALT_STAGE2_RUN_NAME                Base name for the run (default: salt-stage2)
    SALT_STAGE2_CKPT_EVERY              Save checkpoint every N epochs (default: 50)
    SALT_SEED                           Random seed for reproducibility (default: 42)
    SALT_PRECISION                      Lightning precision (default: 16-mixed)
    SALT_USE_WANDB                      Set to "1" to enable W&B logging (default: 1)
    HF_IN100_CACHE_DIR                  ImageNet-100 HuggingFace cache dir
    SLURM_NNODES                        Number of nodes (set by SLURM; default: 1)
    WANDB_ENTITY                        W&B entity
    WANDB_PROJECT                       W&B project (default: stable-pretraining)
"""

import os
import types
from datetime import datetime
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.salt import SALT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
batch_size = int(os.environ.get("SALT_BATCH_SIZE", "256"))
lr = float(os.environ.get("SALT_LR", "5e-4"))
num_workers = int(os.environ.get("SALT_NUM_WORKERS", "16"))
seed = int(os.environ.get("SALT_SEED", "42"))

data_dir = Path(os.environ.get("HF_IN100_CACHE_DIR", "/nfs-gpu/users_home/levizolyomi/hf-in100"))
data_dir.mkdir(parents=True, exist_ok=True)

print(
    "Stage 2 config:",
    {
        "encoder": "vit_base_patch16_224",
        "cache_dir": str(data_dir),
        "num_nodes": num_nodes,
        "batch_size_per_device": batch_size,
        "lr": lr,
        "teacher_ckpt": os.environ.get("SALT_TEACHER_CKPT"),
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

train_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="train",
    cache_dir=str(data_dir),
    transform=train_transform,
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
def salt_forward(self, batch, stage):
    output = SALT.forward(self, batch["image"])
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return {
        "loss": output.loss,
        "embedding": output.embedding,
        **({"label": batch["label"].long()} if "label" in batch else {}),
    }


ckpt_path = os.environ.get("SALT_TEACHER_CKPT")
if ckpt_path is not None:
    module = SALT.from_checkpoint(
        ckpt_path,
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=int(os.environ.get("SALT_PREDICTOR_EMBED_DIM", "384")),
        predictor_depth=int(os.environ.get("SALT_PREDICTOR_DEPTH", "12")),
        predictor_num_heads=int(os.environ.get("SALT_PREDICTOR_NUM_HEADS", "16")),
        num_targets=int(os.environ.get("SALT_NUM_TARGETS", "4")),
    )
else:
    module = SALT(
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=int(os.environ.get("SALT_PREDICTOR_EMBED_DIM", "384")),
        predictor_depth=int(os.environ.get("SALT_PREDICTOR_DEPTH", "12")),
        predictor_num_heads=int(os.environ.get("SALT_PREDICTOR_NUM_HEADS", "16")),
        num_targets=int(os.environ.get("SALT_NUM_TARGETS", "4")),
    )

module.forward = types.MethodType(salt_forward, module)
module.optim = {
    "main": {
        "modules": "student|predictor",
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
    probe=nn.Linear(768, 100),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
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
    queue_length=20000,
    metrics={"top1": torchmetrics.classification.MulticlassAccuracy(100)},
    input_dim=768,
    k=20,
)

_dt_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
predefined_run_name = os.environ.get("SALT_STAGE2_RUN_NAME", "salt-stage2")
run_name = f"{predefined_run_name}-vitb-{_dt_suffix}"
ckpt_dir = Path(__file__).parent / "checkpoints" / run_name

print(f">>>>> CKPT_DIR: {ckpt_dir}")

wandb_logger = False
if os.environ.get("SALT_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=run_name,
        log_model=False,
    )

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = pl.Trainer(
    max_epochs=int(os.environ.get("SALT_STAGE2_EPOCHS", "400")),
    num_sanity_val_steps=0,
    check_val_every_n_epoch=5,
    callbacks=[
        spt.callbacks.StepTimer(),
        linear_probe,
        knn_probe,
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{run_name}-{{epoch:03d}}",
            save_top_k=-1,
            every_n_epochs=int(os.environ.get("SALT_STAGE2_CKPT_EVERY", "50")),
            save_last=True,
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
    precision=os.environ.get("SALT_PRECISION", "16-mixed"),
    logger=wandb_logger,
    devices="auto",
    num_nodes=num_nodes,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
)

manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
manager()
