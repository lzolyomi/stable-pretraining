"""DINO pretraining on ImageNet-100 with ViT-Base.

Uses 2 global crops (224x224) + 6 local crops (96x96) following the original
DINO paper. The backbone and projector are both wrapped with TeacherStudentWrapper
for EMA-based teacher updates.

Usage (defaults):
    python benchmarks/imagenet100/dino-vit-base.py

Environment variables:
    DINO_LR                         Base learning rate (default: 5e-3)
    DINO_BATCH_SIZE                 Per-device batch size (default: 256)
    DINO_EPOCHS                     Training epochs (default: 300)
    DINO_NUM_WORKERS                DataLoader workers (default: 16)
    DINO_PRECISION                  Lightning precision (default: 16-mixed)
    DINO_CKPT_EVERY                 Save checkpoint every N epochs (default: 50)
    DINO_SEED                       Random seed for reproducibility (default: 42)
    DINO_EMA_START                  Initial EMA decay (default: 0.9995)
    DINO_EMA_END                    Final EMA decay (default: 1.0)
    DINO_TEMP_TEACHER               Final teacher temperature (default: 0.07)
    DINO_TEMP_TEACHER_WARMUP        Warmup start teacher temperature (default: 0.04)
    DINO_TEMP_TEACHER_WARMUP_EPOCHS Epochs to warm up teacher temperature (default: 50)
    DINO_RUN_NAME                   Base name for the run (default: dino)
    DINO_USE_WANDB                  Set to "1" to enable W&B logging (default: 1)
    HF_IN100_CACHE_DIR              ImageNet-100 HuggingFace cache dir
    SLURM_NNODES                    Number of nodes (set automatically by SLURM)
    WANDB_ENTITY                    W&B entity
    WANDB_PROJECT                   W&B project (default: stable-pretraining)
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
from stable_pretraining.forward import dino_forward
from stable_pretraining.data import transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
lr = float(os.environ.get("DINO_LR", "5e-3"))
batch_size = int(os.environ.get("DINO_BATCH_SIZE", "256"))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
num_workers = int(os.environ.get("DINO_NUM_WORKERS", "16"))
max_epochs = int(os.environ.get("DINO_EPOCHS", "300"))
precision = os.environ.get("DINO_PRECISION", "16-mixed")
ckpt_every = int(os.environ.get("DINO_CKPT_EVERY", "50"))
seed = int(os.environ.get("DINO_SEED", "42"))

ema_start = float(os.environ.get("DINO_EMA_START", "0.9995"))
ema_end = float(os.environ.get("DINO_EMA_END", "1.0"))
temperature_teacher = float(os.environ.get("DINO_TEMP_TEACHER", "0.07"))
warmup_temperature_teacher = float(os.environ.get("DINO_TEMP_TEACHER_WARMUP", "0.04"))
warmup_epochs_temperature_teacher = int(os.environ.get("DINO_TEMP_TEACHER_WARMUP_EPOCHS", "50"))

data_dir = Path(os.environ.get("HF_IN100_CACHE_DIR", "/nfs-gpu/users_home/levizolyomi/hf-in100"))
data_dir.mkdir(parents=True, exist_ok=True)

print(
    "DINO config:",
    {
        "encoder": "vit_base_patch16_224",
        "lr": lr,
        "batch_size_per_device": batch_size,
        "num_nodes": num_nodes,
        "max_epochs": max_epochs,
        "ema_start": ema_start,
        "ema_end": ema_end,
        "temperature_teacher": temperature_teacher,
        "warmup_temperature_teacher": warmup_temperature_teacher,
        "warmup_epochs_temperature_teacher": warmup_epochs_temperature_teacher,
    },
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dino_transform = transforms.MultiViewTransform(
    {
        "global_1": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "global_2": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_1": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_2": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_3": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_4": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_5": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_6": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    }
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
    transform=dino_transform,
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
backbone = spt.backbone.vit_hf(
    size="base",
    patch_size=16,
    image_size=224,
    pretrained=False,
)

wrapped_backbone = spt.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=ema_start,
    final_ema_coefficient=ema_end,
)

projector = nn.Sequential(
    nn.Linear(768, 2048),
    nn.BatchNorm1d(2048),
    nn.GELU(),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.GELU(),
    nn.Linear(2048, 256),
    spt.utils.nn_modules.L2Norm(),
    nn.Linear(256, 65536, bias=False),  # Prototypes layer
)

wrapped_projector = spt.TeacherStudentWrapper(
    projector,
    warm_init=True,
    base_ema_coefficient=ema_start,
    final_ema_coefficient=ema_end,
)

module = spt.Module(
    backbone=wrapped_backbone,
    projector=wrapped_projector,
    forward=dino_forward,
    dino_loss=spt.losses.DINOv1Loss(
        temperature_student=0.1,
        center_momentum=0.9,
    ),
    warmup_temperature_teacher=warmup_temperature_teacher,
    temperature_teacher=temperature_teacher,
    warmup_epochs_temperature_teacher=warmup_epochs_temperature_teacher,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 1e-4,
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
teacher_student_callback = spt.callbacks.TeacherStudentCallback(
    update_frequency=1,
    update_after_backward=False,
)

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
predefined_run_name = os.environ.get("DINO_RUN_NAME", "dino")
run_name = f"{predefined_run_name}-vitb-{_dt_suffix}"
ckpt_dir = Path(__file__).parent / "checkpoints" / run_name

print(f">>>>> CKPT_DIR: {ckpt_dir}")

wandb_logger = False
if os.environ.get("DINO_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=run_name,
        log_model=False,
    )
    wandb_logger.log_hyperparams({
        "encoder": "vit_base_patch16_224",
        "lr": lr,
        "batch_size_per_device": batch_size,
        "num_nodes": num_nodes,
        "max_epochs": max_epochs,
        "ema_start": ema_start,
        "ema_end": ema_end,
        "temperature_teacher": temperature_teacher,
        "warmup_temperature_teacher": warmup_temperature_teacher,
        "warmup_epochs_temperature_teacher": warmup_epochs_temperature_teacher,
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
        teacher_student_callback,
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
    sync_batchnorm=True,
)

manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
manager()
