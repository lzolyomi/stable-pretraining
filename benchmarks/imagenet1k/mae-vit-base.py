"""MAE pretraining on ImageNet-1K with ViT-Base.

Usage (defaults):
    python benchmarks/imagenet1k/mae-vit-base.py

Environment variables:
    MAE_BATCH_SIZE          Per-device batch size (default: 256)
    MAE_BASE_LR             Base learning rate; scaled by effective batch / 2048 (default: 5e-4)
    MAE_EPOCHS              Training epochs (default: 300)
    MAE_NUM_WORKERS         DataLoader workers (default: 16)
    MAE_PRECISION           Lightning precision (default: 16-mixed)
    MAE_CKPT_EVERY          Save checkpoint every N epochs (default: 50)
    MAE_SEED                Random seed for reproducibility (default: 42)
    MAE_RUN_NAME            Base name for the run (default: mae)
    MAE_USE_WANDB           Set to "1" to enable W&B logging (default: 1)
    HF_IN1K_CACHE_DIR       ImageNet-1K HuggingFace cache dir
    HF_TOKEN                HuggingFace access token
    HF_DATASET_REVISION     Optional dataset revision pin
    WANDB_ENTITY            W&B entity
    WANDB_PROJECT           W&B project (default: stable-pretraining)
"""

import os
import types
from datetime import datetime
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size = int(os.environ.get("MAE_BATCH_SIZE", "256"))
base_lr = float(os.environ.get("MAE_BASE_LR", "5e-4"))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count() or 1))
scaled_lr = base_lr * (batch_size * num_gpus * num_nodes / 2048)

num_workers = int(os.environ.get("MAE_NUM_WORKERS", "16"))
max_epochs = int(os.environ.get("MAE_EPOCHS", "300"))
precision = os.environ.get("MAE_PRECISION", "16-mixed")
ckpt_every = int(os.environ.get("MAE_CKPT_EVERY", "50"))
seed = int(os.environ.get("MAE_SEED", "42"))

data_dir = Path(os.environ.get("HF_IN1K_CACHE_DIR", "/leonardo_work/AIFAC_F01_019/spt-data/hf-in1k"))
data_dir.mkdir(parents=True, exist_ok=True)

print(
    "MAE config:",
    {
        "encoder": "vit_base_patch16_224",
        "base_lr": base_lr,
        "scaled_lr": scaled_lr,
        "batch_size_per_device": batch_size,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
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
    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
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
module = MAE(
    encoder_name="vit_base_patch16_224",
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mask_ratio=0.75,
    block_size=1,
    norm_pix_loss=True,
    loss_type="mse",
    pretrained=False,
)


def mae_forward(self, batch, stage):
    output = MAE.forward(self, batch["image"])
    with torch.no_grad():
        features = self.encoder.forward_features(batch["image"])
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return {
        "loss": output.loss,
        "embedding": features[:, 1:].mean(dim=1).detach(),  # skip cls, mean-pool patches
        **({"label": batch["label"].long()} if "label" in batch else {}),
    }


module.forward = types.MethodType(mae_forward, module)
module.optim = {
    "optimizer": {
        "type": "AdamW",
        "lr": scaled_lr,
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

rankme = spt.callbacks.RankMe(
    name="rankme",
    target="embedding",
    queue_length=1000,
    target_shape=768,
)

_dt_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
predefined_run_name = os.environ.get("MAE_RUN_NAME", "mae")
run_name = f"{predefined_run_name}-vitb-{_dt_suffix}"
ckpt_dir = Path(__file__).parent / "checkpoints" / run_name

print(f">>>>> CKPT_DIR: {ckpt_dir}")

wandb_logger = False
if os.environ.get("MAE_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=run_name,
        log_model=False,
    )
    wandb_logger.log_hyperparams({
        "encoder": "vit_base_patch16_224",
        "base_lr": base_lr,
        "scaled_lr": scaled_lr,
        "batch_size_per_device": batch_size,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
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
)

manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
manager()
