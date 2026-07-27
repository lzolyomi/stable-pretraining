"""Multi-dataset linear probing for NJEPA / IJEPA checkpoints.

Loads a frozen pretrained backbone and sequentially trains a linear head
(BN + Linear) on each requested dataset.  Each dataset evaluation starts from
scratch – fresh head weights, optimizer, and trainer.  All results land in a
single W&B run for easy side-by-side comparison.

Usage:
    CKPT_PATH=.../epoch=299.ckpt MODEL_TYPE=njepa python benchmarks/linear_probing/run.py
    CKPT_PATH=.../epoch=299.ckpt MODEL_TYPE=ijepa python benchmarks/linear_probing/run.py
    CKPT_PATH=.../salt-stage2-vitb-epoch=399.ckpt MODEL_TYPE=salt python benchmarks/linear_probing/run.py

    # Only evaluate on CIFAR datasets:
    DATASETS=cifar10,cifar100 CKPT_PATH=... MODEL_TYPE=njepa python benchmarks/linear_probing/run.py

Environment variables:
    CKPT_PATH           Path to .ckpt file (required)
                        For SALT, point at a Stage 2 checkpoint.
    MODEL_TYPE          'njepa', 'ijepa', 'mae', or 'salt' (required)
    ENCODER_NAME        timm model name (default: vit_base_patch16_224)
    DATASETS            Comma-separated datasets to evaluate
                        Options: cifar10, cifar100, imagenet100
                        (default: cifar10,cifar100,imagenet100)
    DATA_ROOT           Root dir for CIFAR torchvision datasets (default: /nfs-gpu/datasets)
    LP_EPOCHS           Epochs per dataset (default: 50)
    LP_BATCH_SIZE       Batch size per GPU (default: 2048)
    LP_NUM_WORKERS      DataLoader workers (default: 16)
    LP_LR               Learning rate for linear head (default: 1e-3)
    LP_WEIGHT_DECAY     Weight decay (default: 1e-4)
    LP_SEED             Random seed for reproducibility (default: 42)
    LP_ENCODER          Which encoder to probe: 'teacher' (EMA, default) or 'student'
                        For SALT: 'teacher' = frozen Stage 1, 'student' = Stage 2 trainable.
    LP_POOLING          How to build the embedding vector (default: cls):
                          cls     – CLS token of the final layer  [D]
                          mean    – mean of all patch tokens of the final layer  [D]
                          last-N  – mean-pool patches from the last N layers,
                                    concatenated  [N*D]  (e.g. last-4)
                        Note: 'teacher'/'student' are ignored for MAE (single encoder)
    LP_USE_WANDB        '1' to enable W&B logging (default: 1)
    WANDB_ENTITY        W&B entity
    WANDB_PROJECT       W&B project (default: stable-pretraining)
    WANDB_RUN_NAME      Override auto-generated run name
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
from torch import nn

import stable_pretraining as spt

from loguru import logger as _logger
_logger.disable("stable_pretraining.callbacks.cpu_offload")
_logger.disable("stable_pretraining.callbacks.checkpoint_sklearn")

# Make sibling modules importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from _loaders import EMBED_DIM_DEFAULTS, load_backbone
from _datasets import AVAILABLE_DATASETS, NUM_CLASSES, build_dataloaders

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CKPT_PATH = os.environ.get("CKPT_PATH")
if not CKPT_PATH:
    raise ValueError(
        "CKPT_PATH must point to a .ckpt file.\n"
        "  Example: CKPT_PATH=checkpoints/njepa-vitb/epoch=299.ckpt"
    )

MODEL_TYPE = os.environ.get("MODEL_TYPE", "").lower()
if MODEL_TYPE not in ("njepa", "ijepa", "mae", "salt"):
    raise ValueError("MODEL_TYPE must be 'njepa', 'ijepa', 'mae', or 'salt'.")

encoder_name = os.environ.get("ENCODER_NAME", "vit_base_patch16_224")
if encoder_name not in EMBED_DIM_DEFAULTS:
    raise ValueError(
        f"ENCODER_NAME='{encoder_name}' not in the defaults table. "
        f"Known: {list(EMBED_DIM_DEFAULTS)}"
    )

datasets_str = os.environ.get("DATASETS", "cifar10,cifar100,imagenet100")
datasets = [d.strip() for d in datasets_str.split(",")]
unknown = [d for d in datasets if d not in AVAILABLE_DATASETS]
if unknown:
    raise ValueError(f"Unknown dataset(s): {unknown}. Options: {AVAILABLE_DATASETS}")

data_root    = os.environ.get("DATA_ROOT", "/nfs-gpu/datasets")
epochs       = int(os.environ.get("LP_EPOCHS", "50"))
batch_size   = int(os.environ.get("LP_BATCH_SIZE", "2048"))
num_workers  = int(os.environ.get("LP_NUM_WORKERS", "16"))
lr           = float(os.environ.get("LP_LR", "1e-3"))
weight_decay = float(os.environ.get("LP_WEIGHT_DECAY", "1e-4"))
use_wandb    = os.environ.get("LP_USE_WANDB", "1") == "1"

_lp_encoder = os.environ.get("LP_ENCODER", "teacher").lower()
if _lp_encoder not in ("teacher", "student"):
    raise ValueError(f"LP_ENCODER must be 'teacher' or 'student', got '{_lp_encoder}'")
use_teacher = _lp_encoder == "teacher"

pooling = os.environ.get("LP_POOLING", "cls")
seed    = int(os.environ.get("LP_SEED", "42"))

pl.seed_everything(seed, workers=True)

num_gpus = torch.cuda.device_count() or 1

ckpt_stem    = Path(CKPT_PATH).stem
dt_suffix    = datetime.now().strftime("%Y%m%d_%H%M%S")
_vit_short   = encoder_name.split("_patch")[0].replace("vit_", "vit")
_ckpt_folder = Path(CKPT_PATH).parent.name
run_name     = os.environ.get(
    "WANDB_RUN_NAME",
    f"lp-{_ckpt_folder}-{_vit_short}-{dt_suffix}",
)

hparams = {
    "ckpt":          CKPT_PATH,
    "ckpt_name":     ckpt_stem,
    "model_type":    MODEL_TYPE,
    "encoder":       encoder_name,
    "probe_encoder": _lp_encoder,
    "pooling":       pooling,
    "datasets":      datasets,
    "epochs":        epochs,
    "batch_size":    batch_size,
    "num_gpus":      num_gpus,
    "lr":            lr,
    "weight_decay":  weight_decay,
    "seed":          seed,
}

print("Linear probe config:", hparams)

# ---------------------------------------------------------------------------
# W&B – single run that accumulates metrics for all datasets
# ---------------------------------------------------------------------------
# We call wandb.init() once so that hyperparams are recorded as config params
# (visible in the W&B "Config" panel) and all per-dataset metrics land in one
# run.  WandbLogger receives the existing `experiment` object so Lightning
# never calls wandb.finish() between datasets.
wandb_run = None
if use_wandb:
    import wandb
    wandb_run = wandb.init(
        entity="tirex",
        project="stable-pretraining-lp",
        name=run_name,
        config=hparams,
    )
    print(f"W&B run: {wandb_run.url}")

# ---------------------------------------------------------------------------
# Load backbone once; it stays frozen for all datasets
# ---------------------------------------------------------------------------
model, probe_embed_dim = load_backbone(
    model_type=MODEL_TYPE,
    ckpt_path=CKPT_PATH,
    encoder_name=encoder_name,
    use_teacher=use_teacher,
    pooling=pooling,
)

# ---------------------------------------------------------------------------
# Sequential per-dataset evaluation
# ---------------------------------------------------------------------------
ckpt_base = Path(__file__).parent / "checkpoints" / run_name

# Save original methods before any probe wraps them, so each dataset
# iteration starts from a clean slate (no chained wrappers from prior runs).
_orig_configure_model = model.configure_model
_orig_configure_optimizers = model.configure_optimizers
_orig_forward = model.forward

for dataset_name in datasets:
    # Reset all three monkey-patched methods and callback state accumulated
    # by the previous iteration so the new probe registers cleanly.
    # forward must be reset too: its wrapper also chains, and the old
    # cifar10 wrapper would try to read callbacks_metrics['linear_probe_cifar10']
    # which no longer exists after the reset below.
    model.configure_model = _orig_configure_model
    model.configure_optimizers = _orig_configure_optimizers
    model.forward = _orig_forward
    model.callbacks_modules = torch.nn.ModuleDict()
    model.callbacks_metrics = torch.nn.ModuleDict()
    model._optimizer_index_to_name = {}
    model._optimizer_frequencies = {}
    model._optimizer_gradient_clip_val = {}
    model._optimizer_gradient_clip_algorithm = {}
    num_classes = NUM_CLASSES[dataset_name]
    probe_name  = f"linear_probe_{dataset_name}"

    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()}  ({num_classes} classes)")
    print(f"{'='*60}")

    # -- Data ----------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(
        dataset=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    # -- Fresh probe (new head weights + optimizer) --------------------------
    probe = spt.callbacks.OnlineProbe(
        model,
        name=probe_name,
        input="embedding",
        target="label",
        probe=nn.Sequential(
            nn.BatchNorm1d(probe_embed_dim),
            nn.Linear(probe_embed_dim, num_classes),
        ),
        loss=nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
            "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5),
        },
        optimizer={"type": "AdamW", "lr": lr, "weight_decay": weight_decay},
        scheduler={"type": "CosineAnnealingLR", "T_max": epochs},
    )

    # -- Logger (re-use the existing W&B run) --------------------------------
    if wandb_run is not None:
        from lightning.pytorch.loggers import WandbLogger
        # Pass experiment= so Lightning does NOT call wandb.finish() on teardown
        logger = WandbLogger(experiment=wandb_run, log_model=False)
    else:
        logger = None

    # -- Trainer -------------------------------------------------------------
    ckpt_dir = ckpt_base / dataset_name
    trainer = pl.Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=0,
        callbacks=[
            probe,
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename=f"{probe_name}-{{epoch:03d}}",
                save_last=True,
                save_top_k=1,
                monitor=f"eval/{probe_name}_top1",
                mode="max",
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=logger,
        devices=num_gpus,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
    )

    manager = spt.Manager(trainer=trainer, module=model, data=data, seed=seed)
    manager()

# ---------------------------------------------------------------------------
# Finish W&B run
# ---------------------------------------------------------------------------
if wandb_run is not None:
    wandb_run.finish()
    print(f"\nW&B run finished: {wandb_run.url}")

print("\nAll datasets evaluated. Done.")
