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
from stable_pretraining.backbone import MultiBlockMasking
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE


def resolve_batch_config(target_effective_batch_size, max_batch_size_per_device, num_gpus):
    if target_effective_batch_size < 1:
        raise ValueError("SALT_EFFECTIVE_BATCH_SIZE must be positive.")
    if max_batch_size_per_device < 1:
        raise ValueError("SALT_MAX_BATCH_SIZE_PER_DEVICE must be positive.")

    upper = min(max_batch_size_per_device, target_effective_batch_size)
    for batch_size in range(upper, 0, -1):
        per_step_batch_size = batch_size * num_gpus
        if target_effective_batch_size % per_step_batch_size == 0:
            accumulate_grad_batches = target_effective_batch_size // per_step_batch_size
            return batch_size, accumulate_grad_batches

    raise ValueError(
        "Could not match the requested effective batch size exactly. "
        f"GPU count={num_gpus}, target={target_effective_batch_size}, "
        f"max_per_device={max_batch_size_per_device}. "
        "Adjust SALT_MAX_BATCH_SIZE_PER_DEVICE or the GPU count."
    )


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
effective_batch_size = int(os.environ.get("SALT_EFFECTIVE_BATCH_SIZE", "2048"))
max_batch_size_per_device = int(os.environ.get("SALT_MAX_BATCH_SIZE_PER_DEVICE", "128"))
batch_size, accumulate_grad_batches = resolve_batch_config(
    effective_batch_size,
    max_batch_size_per_device,
    num_gpus,
)
scaled_lr = 1.5e-4 * (effective_batch_size / 4096)
num_workers = int(os.environ.get("SALT_NUM_WORKERS", "16"))

print(
    "Stage 1 config:",
    {
        "cache_dir": str(data_dir),
        "num_gpus": num_gpus,
        "batch_size_per_device": batch_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "effective_batch_size": effective_batch_size,
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
        "lr": scaled_lr,
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

wandb_logger = False
if os.environ.get("SALT_USE_WANDB", "1") == "1":
    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "levi-temp"),
        name=f"salt-stage1-vit-base-{time.time()}",
        log_model=False,
    )

trainer = pl.Trainer(
    max_epochs=int(os.environ.get("SALT_STAGE1_EPOCHS", "100")),
    num_sanity_val_steps=0,
    callbacks=[
        linear_probe,
        knn_probe,
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(Path(__file__).parent / "checkpoints" / "salt-stage1-vitb"),
            filename="salt-stage1-vitb-{epoch:03d}",
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
    accumulate_grad_batches=accumulate_grad_batches,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
