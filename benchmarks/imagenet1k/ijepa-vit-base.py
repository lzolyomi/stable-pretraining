"""I-JEPA pretraining on ImageNet-1K with ViT-Base.

Usage (defaults):
    python benchmarks/imagenet1k/ijepa-vit-base.py

Environment variables:
    IJEPA_ENCODER_NAME              timm model name (default: vit_base_patch16_224)
    IJEPA_PREDICTOR_EMBED_DIM       Predictor hidden dim (default: 384)
    IJEPA_PREDICTOR_DEPTH           Predictor depth (default: 6)
    IJEPA_NUM_TARGETS               Number of target blocks (default: 4)
    IJEPA_TARGET_SCALE_MIN          Min target block scale (default: 0.15)
    IJEPA_TARGET_SCALE_MAX          Max target block scale (default: 0.2)
    IJEPA_TARGET_ASPECT_MIN         Min target aspect ratio (default: 0.75)
    IJEPA_TARGET_ASPECT_MAX         Max target aspect ratio (default: 1.5)
    IJEPA_CONTEXT_SCALE_MIN         Min context block scale (default: 0.85)
    IJEPA_CONTEXT_SCALE_MAX         Max context block scale (default: 1.0)
    IJEPA_EMA_START                 Initial EMA decay (default: 0.996)
    IJEPA_EMA_END                   Final EMA decay (default: 1.0)
    IJEPA_BATCH_SIZE                Per-device batch size (default: 256)
    IJEPA_LR                        Base learning rate (default: 5e-4)
    IJEPA_EPOCHS                    Training epochs (default: 300)
    IJEPA_NUM_WORKERS               DataLoader workers (default: 16)
    IJEPA_PRECISION                 Lightning precision (default: 16-mixed)
    IJEPA_CKPT_EVERY                Save checkpoint every N epochs (default: 50)
    IJEPA_RUN_NAME                  Base name for the run (default: ijepa)
    IJEPA_USE_WANDB                 Set to "1" to enable W&B logging (default: 1)
    HF_IN1K_CACHE_DIR               ImageNet-1K HuggingFace cache dir
    HF_TOKEN                        HuggingFace access token
    HF_DATASET_REVISION             Optional dataset revision pin
    WANDB_ENTITY                    W&B entity
    WANDB_PROJECT                   W&B project (default: stable-pretraining)
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
from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
encoder_name = os.environ.get("IJEPA_ENCODER_NAME", "vit_base_patch16_224")
predictor_embed_dim = int(os.environ.get("IJEPA_PREDICTOR_EMBED_DIM", "384"))
predictor_depth = int(os.environ.get("IJEPA_PREDICTOR_DEPTH", "6"))
num_targets = int(os.environ.get("IJEPA_NUM_TARGETS", "4"))
target_scale = (
    float(os.environ.get("IJEPA_TARGET_SCALE_MIN", "0.15")),
    float(os.environ.get("IJEPA_TARGET_SCALE_MAX", "0.2")),
)
target_aspect_ratio = (
    float(os.environ.get("IJEPA_TARGET_ASPECT_MIN", "0.75")),
    float(os.environ.get("IJEPA_TARGET_ASPECT_MAX", "1.5")),
)
context_scale = (
    float(os.environ.get("IJEPA_CONTEXT_SCALE_MIN", "0.85")),
    float(os.environ.get("IJEPA_CONTEXT_SCALE_MAX", "1.0")),
)
ema_start = float(os.environ.get("IJEPA_EMA_START", "0.996"))
ema_end = float(os.environ.get("IJEPA_EMA_END", "1.0"))

batch_size = int(os.environ.get("IJEPA_BATCH_SIZE", "256"))
lr = float(os.environ.get("IJEPA_LR", "5e-4"))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
num_workers = int(os.environ.get("IJEPA_NUM_WORKERS", "16"))
max_epochs = int(os.environ.get("IJEPA_EPOCHS", "300"))
precision = os.environ.get("IJEPA_PRECISION", "16-mixed")
ckpt_every = int(os.environ.get("IJEPA_CKPT_EVERY", "50"))

data_dir = Path(os.environ.get("HF_IN1K_CACHE_DIR", "/nfs-gpu/users_home/levizolyomi/hf-in1k"))
data_dir.mkdir(parents=True, exist_ok=True)

print(
    "IJEPA config:",
    {
        "encoder": encoder_name,
        "predictor_embed_dim": predictor_embed_dim,
        "predictor_depth": predictor_depth,
        "num_targets": num_targets,
        "target_scale": target_scale,
        "target_aspect_ratio": target_aspect_ratio,
        "context_scale": context_scale,
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
module = IJEPA(
    encoder_name=encoder_name,
    predictor_embed_dim=predictor_embed_dim,
    predictor_depth=predictor_depth,
    num_targets=num_targets,
    target_scale=target_scale,
    target_aspect_ratio=target_aspect_ratio,
    context_scale=context_scale,
    ema_decay_start=ema_start,
    ema_decay_end=ema_end,
    pretrained=False,
)


def ijepa_forward(self, batch, stage):
    output = IJEPA.forward(self, batch["image"])
    embedding = output.embedding.mean(dim=1)
    if self.training:
        embedding = embedding.detach()

    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)

    return {
        "loss": output.loss,
        "embedding": embedding,
        **({"label": batch["label"].long()} if "label" in batch else {}),
    }


module.forward = types.MethodType(ijepa_forward, module)
module.optim = {
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
}

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(768, 1000),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(1000),
        "top5": torchmetrics.classification.MulticlassAccuracy(1000, top_k=5),
    },
    optimizer={"type": "AdamW", "lr": 3e-3, "weight_decay": 1e-4},
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=10000,
    metrics={"top1": torchmetrics.classification.MulticlassAccuracy(1000)},
    input_dim=768,
    k=20,
)

_dt_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
predefined_run_name = os.environ.get("IJEPA_RUN_NAME", "ijepa")
run_name = f"{predefined_run_name}-vitb-{_dt_suffix}"
ckpt_dir = Path(__file__).parent / "checkpoints" / run_name

wandb_logger = False
if os.environ.get("IJEPA_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=run_name,
        log_model=False,
    )
    wandb_logger.log_hyperparams({
        "encoder": encoder_name,
        "predictor_embed_dim": predictor_embed_dim,
        "predictor_depth": predictor_depth,
        "num_targets": num_targets,
        "target_scale": target_scale,
        "target_aspect_ratio": target_aspect_ratio,
        "context_scale": context_scale,
        "ema_start": ema_start,
        "ema_end": ema_end,
        "batch_size_per_device": batch_size,
        "num_nodes": num_nodes,
        "lr": lr,
        "max_epochs": max_epochs,
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
        TeacherStudentCallback(
            update_frequency=1,
            update_after_backward=False,
        ),
        spt.callbacks.StepTimer(),
        linear_probe,
        knn_probe,
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
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
