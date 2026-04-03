"""LeJEPA pretraining on ImageNet-1K with ViT-Base.

Multi-view invariance + Epps-Pulley goodness-of-fit (SIGReg).
Uses 2 global views (224x224) + 6 local views (96x96) matching
the official LeJEPA augmentation strategy.

Usage (defaults):
    python benchmarks/imagenet1k/lejepa-vit-base.py

Environment variables:
    LEJEPA_ENCODER_NAME         timm model name (default: vit_base_patch16_224)
    LEJEPA_LAMB                 SIGReg weight λ (default: 0.02)
    LEJEPA_N_SLICES             Random projection directions (default: 1024)
    LEJEPA_N_POINTS             EP quadrature nodes (default: 17)
    LEJEPA_GLOBAL_VIEWS         Number of global views (default: 1)
    LEJEPA_LOCAL_VIEWS          Number of local views (default: 3)
    LEJEPA_BATCH_SIZE           Per-device batch size (default: 256)
    LEJEPA_LR                   Base learning rate (default: 5e-4)
    LEJEPA_EPOCHS               Training epochs (default: 300)
    LEJEPA_NUM_WORKERS          DataLoader workers (default: 16)
    LEJEPA_PRECISION            Lightning precision (default: 16-mixed)
    LEJEPA_CKPT_EVERY           Save checkpoint every N epochs (default: 50)
    LEJEPA_SEED                 Random seed for reproducibility (default: 42)
    LEJEPA_RUN_NAME             Base name for the run (default: lejepa)
    LEJEPA_USE_WANDB            Set to "1" to enable W&B logging (default: 1)
    HF_IN1K_CACHE_DIR           ImageNet-1K HuggingFace cache dir
    HF_TOKEN                    HuggingFace access token
    HF_DATASET_REVISION         Optional dataset revision pin
    WANDB_ENTITY                W&B entity
    WANDB_PROJECT               W&B project (default: stable-pretraining)
"""

import os
from datetime import datetime
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
encoder_name = os.environ.get("LEJEPA_ENCODER_NAME", "vit_base_patch16_224")
lamb = float(os.environ.get("LEJEPA_LAMB", "0.02"))
n_slices = int(os.environ.get("LEJEPA_N_SLICES", "1024"))
n_points = int(os.environ.get("LEJEPA_N_POINTS", "17"))
n_global_views = int(os.environ.get("LEJEPA_GLOBAL_VIEWS", "1"))
n_local_views = int(os.environ.get("LEJEPA_LOCAL_VIEWS", "3"))

batch_size = int(os.environ.get("LEJEPA_BATCH_SIZE", "256"))
lr = float(os.environ.get("LEJEPA_LR", "5e-4"))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
num_workers = int(os.environ.get("LEJEPA_NUM_WORKERS", "16"))
max_epochs = int(os.environ.get("LEJEPA_EPOCHS", "300"))
precision = os.environ.get("LEJEPA_PRECISION", "16-mixed")
ckpt_every = int(os.environ.get("LEJEPA_CKPT_EVERY", "50"))
seed = int(os.environ.get("LEJEPA_SEED", "42"))

data_dir = Path(os.environ.get("HF_IN1K_CACHE_DIR", "/nfs-gpu/users_home/levizolyomi/hf-in1k"))
data_dir.mkdir(parents=True, exist_ok=True)

print(
    "LeJEPA config:",
    {
        "encoder": encoder_name,
        "lamb": lamb,
        "n_slices": n_slices,
        "n_points": n_points,
        "n_global_views": n_global_views,
        "n_local_views": n_local_views,
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


def _photometric_transforms() -> list:
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.2),
    ]


def _global_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def _local_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.3)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


train_transform = transforms.MultiViewTransform(
    {
        **{f"global_{i}": _global_transform() for i in range(n_global_views)},
        **{f"local_{i}": _local_transform() for i in range(n_local_views)},
    }
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
num_gpus = torch.cuda.device_count() or 1
total_steps = (len(train_dataloader) // (num_gpus * num_nodes)) * max_epochs


def lejepa_forward(self, batch, stage):
    out = {}

    images = batch.get("image")
    if stage == "fit":
        global_views = [
            batch[key]["image"] for key in batch if key.startswith("global")
        ]
        local_views = [batch[key]["image"] for key in batch if key.startswith("local")]
        labels = next(
            batch[key]["label"]
            for key in batch
            if key.startswith("global") or key.startswith("local")
        )

        output: LeJEPAOutput = self.model.forward(
            global_views=global_views, local_views=local_views, images=images
        )
        out["label"] = labels.repeat(len(global_views))
    else:
        output: LeJEPAOutput = self.model.forward(images=images)
        out["label"] = batch["label"].long()

    out["loss"] = output.loss
    out["embedding"] = output.embedding

    self.log(f"{stage}/sigreg", output.sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/inv", output.inv_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


model = LeJEPA(
    encoder_name=encoder_name,
    lamb=lamb,
    n_slices=n_slices,
    n_points=n_points,
)

module = spt.Module(
    model=model,
    forward=lejepa_forward,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 10 / max_epochs,
            "start_factor": 0.01,
            "end_lr": lr / 1000,
            "total_steps": total_steps,
        },
        "interval": "step",
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
    probe=nn.Linear(model.embed_dim, 1000),
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
    input_dim=model.embed_dim,
    k=20,
)

_dt_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
predefined_run_name = os.environ.get("LEJEPA_RUN_NAME", "lejepa")
run_name = f"{predefined_run_name}-vitb-{_dt_suffix}"
ckpt_dir = Path(__file__).parent / "checkpoints" / run_name

wandb_logger = False
if os.environ.get("LEJEPA_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=run_name,
        log_model=False,
    )
    wandb_logger.log_hyperparams({
        "encoder": encoder_name,
        "lamb": lamb,
        "n_slices": n_slices,
        "n_points": n_points,
        "n_global_views": n_global_views,
        "n_local_views": n_local_views,
        "batch_size_per_device": batch_size,
        "num_nodes": num_nodes,
        "lr": lr,
        "max_epochs": max_epochs,
        "precision": precision,
        "total_steps": total_steps,
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
