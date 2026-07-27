"""Backbone loading helpers for NJEPA / IJEPA / MAE / SALT linear-probe evaluation.

NJEPA / IJEPA expose a TeacherStudentWrapper encoder:
    model.encoder.teacher(image) -> EMA-smoothed encoder  (preferred for probing)
    model.encoder.student(image) -> online encoder

MAE exposes a plain MaskedEncoder:
    model.encoder  ->  MaskedEncoder  (single encoder, no teacher)
    model.encoder.forward_features(image) -> runs in eval/no-mask mode

SALT exposes student + frozen teacher as direct attributes (Stage 2 checkpoint):
    model.student(image) -> trainable encoder used during Stage 2 fine-tuning
    model.teacher(image) -> frozen Stage 1 encoder (wrapped in EvalOnly)

Output shape for all: [B, num_prefix + N_patches, D]; CLS token is always at index 0.

Pooling modes
-------------
cls     : CLS token only  [B, D]
mean    : mean over all patch tokens of the final layer  [B, D]
last-N  : mean-pool patches of each of the last N layers, then concatenate  [B, N*D]
          The final LayerNorm is applied to every intermediate layer before pooling,
          which is the standard practice (DINOv2 multi-layer probing).
"""

import types

import torch

from stable_pretraining.methods.ijepa import IJEPA
from stable_pretraining.methods.mae import MAE
from stable_pretraining.methods.njepa import NJEPA
from stable_pretraining.methods.salt import SALT

# ---------------------------------------------------------------------------
# Lookup tables (shared with run.py via import)
# ---------------------------------------------------------------------------
EMBED_DIM_DEFAULTS = {
    "vit_small_patch16_224": 384,
    "vit_base_patch16_224": 768,
    "vit_large_patch16_224": 1024,
}

PREDICTOR_DEFAULTS = {
    "vit_small_patch16_224": {"predictor_embed_dim": 192, "predictor_depth": 6},
    "vit_base_patch16_224": {"predictor_embed_dim": 384, "predictor_depth": 6},
    "vit_large_patch16_224": {"predictor_embed_dim": 384, "predictor_depth": 12},
}

VALID_POOLING_PREFIX = ("cls", "mean", "last-")


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------
def _parse_pooling(pooling: str, num_blocks: int) -> tuple[str, int]:
    """Validate and parse the pooling string.

    Returns:
        (mode, n_layers) where mode is ``'cls'``, ``'mean'``, or ``'last_n'``.
    """
    if pooling == "cls":
        return "cls", 1
    if pooling == "mean":
        return "mean", 1
    if pooling.startswith("last-"):
        try:
            n = int(pooling[5:])
        except ValueError:
            raise ValueError(f"pooling '{pooling}': expected an integer after 'last-'")
        if n < 1 or n > num_blocks:
            raise ValueError(
                f"pooling '{pooling}': N={n} out of range "
                f"(model has {num_blocks} transformer blocks)"
            )
        return "last_n", n
    raise ValueError(
        f"pooling must be 'cls', 'mean', or 'last-N' (e.g. 'last-4'), got '{pooling}'"
    )


# ---------------------------------------------------------------------------
# Forward factories
# ---------------------------------------------------------------------------
def _make_probe_forward_jepa(use_teacher: bool, mode: str, n_layers: int, num_prefix: int):
    """Forward for NJEPA / IJEPA (TeacherStudentWrapper encoder)."""

    def _run_encoder(self, images):
        if use_teacher:
            self.encoder.teacher.eval()
            return self.encoder.forward_teacher(images)
        else:
            with torch.no_grad():
                self.encoder.student.eval()
                return self.encoder.student(images)

    if mode == "cls":
        def _forward(self, batch, stage):
            out = _run_encoder(self, batch["image"])
            return {
                "embedding": out.encoded[:, 0, :].detach(),
                "label": batch["label"].long(),
            }

    elif mode == "mean":
        def _forward(self, batch, stage):
            out = _run_encoder(self, batch["image"])
            patches = out.encoded[:, num_prefix:, :]
            return {
                "embedding": patches.mean(dim=1).detach(),
                "label": batch["label"].long(),
            }

    else:  # last_n
        def _forward(self, batch, stage):
            enc = self.encoder.teacher if use_teacher else self.encoder.student
            vit = enc.vit
            blocks = vit.blocks if hasattr(vit, "blocks") else vit.layers

            captured = []

            def _hook(module, inp, out):
                captured.append(out)

            handles = [
                block.register_forward_hook(_hook)
                for block in list(blocks)[-n_layers:]
            ]

            enc.eval()
            with torch.no_grad():
                enc(batch["image"])

            for h in handles:
                h.remove()

            norm = vit.norm
            layer_embs = []
            for layer_out in captured:
                normed = norm(layer_out)
                patches = normed[:, num_prefix:, :]
                layer_embs.append(patches.mean(dim=1))

            return {
                "embedding": torch.cat(layer_embs, dim=-1).detach(),
                "label": batch["label"].long(),
            }

    return _forward


def _make_probe_forward_mae(mode: str, n_layers: int, num_prefix: int):
    """Forward for MAE (plain MaskedEncoder, no teacher)."""

    if mode == "cls":
        def _forward(self, batch, stage):
            # forward_features() runs eval + no_grad internally
            out = self.encoder.forward_features(batch["image"])
            return {
                "embedding": out[:, 0, :].detach(),
                "label": batch["label"].long(),
            }

    elif mode == "mean":
        def _forward(self, batch, stage):
            out = self.encoder.forward_features(batch["image"])
            patches = out[:, num_prefix:, :]
            return {
                "embedding": patches.mean(dim=1).detach(),
                "label": batch["label"].long(),
            }

    else:  # last_n
        def _forward(self, batch, stage):
            vit = self.encoder.vit
            blocks = vit.blocks if hasattr(vit, "blocks") else vit.layers

            captured = []

            def _hook(module, inp, out):
                captured.append(out)

            handles = [
                block.register_forward_hook(_hook)
                for block in list(blocks)[-n_layers:]
            ]

            self.encoder.eval()
            with torch.no_grad():
                self.encoder(batch["image"])

            for h in handles:
                h.remove()

            norm = vit.norm
            layer_embs = []
            for layer_out in captured:
                normed = norm(layer_out)
                patches = normed[:, num_prefix:, :]
                layer_embs.append(patches.mean(dim=1))

            return {
                "embedding": torch.cat(layer_embs, dim=-1).detach(),
                "label": batch["label"].long(),
            }

    return _forward


def _make_probe_forward_salt(use_teacher: bool, mode: str, n_layers: int, num_prefix: int):
    """Forward for SALT (student + frozen teacher as direct attributes)."""

    def _run_encoder(self, images):
        enc = self.teacher if use_teacher else self.student
        with torch.no_grad():
            if not use_teacher:
                self.student.eval()
            return enc(images)

    if mode == "cls":
        def _forward(self, batch, stage):
            out = _run_encoder(self, batch["image"])
            return {
                "embedding": out.encoded[:, 0, :].detach(),
                "label": batch["label"].long(),
            }

    elif mode == "mean":
        def _forward(self, batch, stage):
            out = _run_encoder(self, batch["image"])
            patches = out.encoded[:, num_prefix:, :]
            return {
                "embedding": patches.mean(dim=1).detach(),
                "label": batch["label"].long(),
            }

    else:  # last_n
        def _forward(self, batch, stage):
            enc = self.teacher if use_teacher else self.student
            vit = enc.vit
            blocks = vit.blocks if hasattr(vit, "blocks") else vit.layers

            captured = []

            def _hook(module, inp, out):
                captured.append(out)

            handles = [
                block.register_forward_hook(_hook)
                for block in list(blocks)[-n_layers:]
            ]

            if not use_teacher:
                self.student.eval()
            with torch.no_grad():
                enc(batch["image"])

            for h in handles:
                h.remove()

            norm = vit.norm
            layer_embs = []
            for layer_out in captured:
                normed = norm(layer_out)
                patches = normed[:, num_prefix:, :]
                layer_embs.append(patches.mean(dim=1))

            return {
                "embedding": torch.cat(layer_embs, dim=-1).detach(),
                "label": batch["label"].long(),
            }

    return _forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_backbone(
    model_type: str,
    ckpt_path: str,
    encoder_name: str = "vit_base_patch16_224",
    use_teacher: bool = True,
    pooling: str = "cls",
    predictor_embed_dim: int = None,
    predictor_depth: int = None,
    num_targets: int = 4,
) -> tuple:
    """Instantiate and load a frozen backbone from a checkpoint.

    Args:
        model_type:          ``'njepa'``, ``'ijepa'``, ``'mae'``, or ``'salt'``.
        ckpt_path:           Path to the ``.ckpt`` file saved by Lightning.
                             For SALT this is a Stage 2 checkpoint.
        encoder_name:        timm model name – must be in :data:`EMBED_DIM_DEFAULTS`.
        use_teacher:         Probe the EMA teacher (True, default) or online student (False).
                             Ignored for MAE (has no teacher). For SALT this selects
                             the frozen Stage 1 teacher vs. the trainable student.
        pooling:             How to produce the embedding vector:

                             * ``'cls'``    – CLS token of the final layer  **[D]**
                             * ``'mean'``   – mean of patch tokens of the final layer  **[D]**
                             * ``'last-N'`` – mean-pool patches of the last N layers,
                               concatenated  **[N×D]**  (e.g. ``'last-4'``)

        predictor_embed_dim: Override predictor hidden dim (NJEPA/IJEPA only).
        predictor_depth:     Override predictor depth (NJEPA/IJEPA only).
        num_targets:         Number of masking targets (NJEPA/IJEPA only).

    Returns:
        ``(model, probe_embed_dim)`` where *probe_embed_dim* is the dimension of the
        resulting embedding vector (``D`` for cls/mean, ``N×D`` for last-N).
    """
    model_type = model_type.lower()

    if encoder_name not in EMBED_DIM_DEFAULTS:
        raise ValueError(
            f"encoder_name='{encoder_name}' not in the defaults table. "
            f"Known: {list(EMBED_DIM_DEFAULTS)}"
        )

    embed_dim = EMBED_DIM_DEFAULTS[encoder_name]

    # -- Instantiate model ---------------------------------------------------
    if model_type in ("njepa", "ijepa"):
        pred_cfg = PREDICTOR_DEFAULTS[encoder_name]
        pred_embed_dim = predictor_embed_dim or pred_cfg["predictor_embed_dim"]
        pred_depth = predictor_depth or pred_cfg["predictor_depth"]

        cls = NJEPA if model_type == "njepa" else IJEPA
        model = cls(
            encoder_name=encoder_name,
            predictor_embed_dim=pred_embed_dim,
            predictor_depth=pred_depth,
            num_targets=num_targets,
        )
        # Resolve pooling against actual block count
        enc_ref = model.encoder.teacher if use_teacher else model.encoder.student
        vit = enc_ref.vit
        blocks = vit.blocks if hasattr(vit, "blocks") else vit.layers
        num_blocks = len(blocks)
        num_prefix = enc_ref.num_prefix_tokens

    elif model_type == "mae":
        model = MAE(encoder_name=encoder_name)
        vit = model.encoder.vit
        blocks = vit.blocks if hasattr(vit, "blocks") else vit.layers
        num_blocks = len(blocks)
        num_prefix = model.encoder.num_prefix_tokens
        if use_teacher:
            print("  Note: MAE has no teacher encoder; probing the (only) encoder.")

    elif model_type == "salt":
        # Predictor args match the Stage 2 training script (imagenet1k/salt-stage2-vit-base.py).
        # Loading uses strict=False, so a predictor-shape mismatch would only show up as
        # extra missing/unexpected keys — but we keep the defaults in sync for cleanliness.
        model = SALT(encoder_name=encoder_name)
        enc_ref = model.teacher if use_teacher else model.student
        vit = enc_ref.vit
        blocks = vit.blocks if hasattr(vit, "blocks") else vit.layers
        num_blocks = len(blocks)
        num_prefix = enc_ref.num_prefix_tokens

    else:
        raise ValueError(
            f"model_type must be 'njepa', 'ijepa', 'mae', or 'salt', got '{model_type}'"
        )

    mode, n_layers = _parse_pooling(pooling, num_blocks)
    probe_embed_dim = embed_dim * n_layers

    # -- Load checkpoint -----------------------------------------------------
    print(f"Loading {model_type.upper()} checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if model_type == "salt":
        # SALT teacher is wrapped in EvalOnly: state-dict keys live under teacher.backbone.*
        probe_prefix = "teacher.backbone" if use_teacher else "student"
    elif model_type == "mae":
        probe_prefix = "encoder"
    else:
        probe_prefix = "encoder.teacher" if use_teacher else "encoder.student"

    probe_missing = [k for k in missing if k.startswith(probe_prefix)]
    if probe_missing:
        print(
            f"  WARNING: {len(probe_missing)} missing key(s) in {probe_prefix} "
            f"(first 5): {probe_missing[:5]}"
        )
    else:
        if model_type == "mae":
            which = "encoder"
        elif model_type == "salt":
            which = "teacher (frozen Stage 1)" if use_teacher else "student (Stage 2)"
        else:
            which = "teacher (EMA)" if use_teacher else "student"
        print(
            f"  Checkpoint loaded OK – probing {which}, "
            f"pooling={pooling}, probe_embed_dim={probe_embed_dim} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # -- Freeze & bind forward -----------------------------------------------
    model.requires_grad_(False)
    model.optim = None

    if model_type in ("njepa", "ijepa"):
        model.forward = types.MethodType(
            _make_probe_forward_jepa(use_teacher, mode, n_layers, num_prefix), model
        )
    elif model_type == "salt":
        model.forward = types.MethodType(
            _make_probe_forward_salt(use_teacher, mode, n_layers, num_prefix), model
        )
    else:
        model.forward = types.MethodType(
            _make_probe_forward_mae(mode, n_layers, num_prefix), model
        )

    return model, probe_embed_dim
