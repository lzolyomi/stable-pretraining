"""SALT Stage 1 (MAE pretraining) on ImageNet-1K with ViT-Base.

Runs masked autoencoder pretraining as the first stage of SALT.
The resulting checkpoint is used as the teacher for Stage 2.

Usage (defaults):
    python benchmarks/imagenet1k/salt-stage1-vit-base.py

Environment variables:
    SALT_BATCH_SIZE                     Per-device batch size (default: 256)
    SALT_LR                             Learning rate (default: 5e-4)
    SALT_NUM_WORKERS                    DataLoader workers (default: 16)
    SALT_STAGE1_DECODER_EMBED_DIM       MAE decoder hidden dim (default: 512)
    SALT_STAGE1_DECODER_DEPTH           MAE decoder depth (default: 8)
    SALT_STAGE1_DECODER_NUM_HEADS       MAE decoder attention heads (default: 16)
    SALT_NUM_TARGETS                    Number of masking targets for MultiBlockMasking (default: 4)
    SALT_STAGE1_EPOCHS                  Training epochs (default: 100)
    SALT_STAGE1_RUN_NAME                Base name for the run (default: salt-stage1-vitb)
    SALT_STAGE1_CKPT_EVERY              Save checkpoint every N epochs (default: 25)
    SALT_SEED                           Random seed for reproducibility (default: 42)
    SALT_PRECISION                      Lightning precision (default: 16-mixed)
    SALT_USE_WANDB                      Set to "1" to enable W&B logging (default: 1)
    HF_IN1K_CACHE_DIR                   ImageNet-1K HuggingFace cache dir
    HF_TOKEN                            HuggingFace access token
    HF_DATASET_REVISION                 Optional dataset revision pin
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
from stable_pretraining.backbone import MultiBlockMasking
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE


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

data_dir = Path(
    os.environ.get("HF_IN1K_CACHE_DIR", "/nfs-gpu/users_home/levizolyomi/hf-in1k")
)
data_dir.mkdir(parents=True, exist_ok=True)

num_gpus = torch.cuda.device_count() or 1
batch_size = int(os.environ.get("SALT_BATCH_SIZE", "256"))
lr = float(os.environ.get("SALT_LR", "5e-4"))
num_workers = int(os.environ.get("SALT_NUM_WORKERS", "16"))
seed = int(os.environ.get("SALT_SEED", "42"))

print(
    "Stage 1 config:",
    {
        "cache_dir": str(data_dir),
        "num_gpus": num_gpus,
        "batch_size_per_device": batch_size,
        "lr": lr,
    },
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


def mae_forward(self, batch, stage):
    output = MAE.forward(self, batch["image"])
    with torch.no_grad():
        features = self.encoder.forward_features(batch["image"])

    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)

    return {
        "loss": output.loss,
        "embedding": features[:, 1:].mean(dim=1).detach(),
        **({"label": batch["label"].long()} if "label" in batch else {}),
    }


module = MAE(
    encoder_name="vit_base_patch16_224",
    decoder_embed_dim=int(os.environ.get("SALT_STAGE1_DECODER_EMBED_DIM", "512")),
    decoder_depth=int(os.environ.get("SALT_STAGE1_DECODER_DEPTH", "8")),
    decoder_num_heads=int(os.environ.get("SALT_STAGE1_DECODER_NUM_HEADS", "16")),
    norm_pix_loss=True,
    loss_type="mse",
    masking=MultiBlockMasking(num_targets=int(os.environ.get("SALT_NUM_TARGETS", "4"))),
)

module.forward = types.MethodType(mae_forward, module)
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
    input_dim=768,
    k=20,
)

_dt_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
predefined_run_name = os.environ.get("SALT_STAGE1_RUN_NAME", "salt-stage1")
run_name = f"{predefined_run_name}-vitb-{_dt_suffix}"
ckpt_dir = Path(__file__).parent / "checkpoints" / run_name

wandb_logger = False
if os.environ.get("SALT_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "stable-pretraining"),
        name=run_name,
        log_model=False,
    )

trainer = pl.Trainer(
    max_epochs=int(os.environ.get("SALT_STAGE1_EPOCHS", "100")),
    num_sanity_val_steps=0,
    callbacks=[
        linear_probe,
        knn_probe,
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{run_name}-{{epoch:03d}}",
            save_top_k=-1,
            every_n_epochs=int(os.environ.get("SALT_STAGE1_CKPT_EVERY", "25")),
            save_last=True,
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
    precision=os.environ.get("SALT_PRECISION", "16-mixed"),
    logger=wandb_logger,
    devices=num_gpus,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
)

manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
manager()
