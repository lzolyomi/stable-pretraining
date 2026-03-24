"""NJEPA: I-JEPA with 3-phase noising target encoder update.

Identical to I-JEPA in architecture (context encoder + EMA target encoder + predictor)
but the target encoder update follows a 3-phase noising schedule instead of plain EMA:

- Phase 1 (step <= phase1_end): EMA toward scale-invariant noise
- Phase 2 (phase1_end < step <= phase2_end): EMA toward student + tiny noise
- Phase 3 (step > phase2_end): standard EMA (vanilla I-JEPA)

Setting phase1_end=0 and phase2_end=0 recovers vanilla I-JEPA exactly.

References:
    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023.
    https://arxiv.org/abs/2301.08243

Example::

    from stable_pretraining.methods.njepa import NJEPA
    from stable_pretraining.callbacks import TeacherStudentCallback
    import lightning as pl

    # 3-phase noising (default)
    model = NJEPA(
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=6,
        num_targets=4,
        phase1_end=1000,
        phase2_end=10000,
    )

    # Vanilla I-JEPA (disable noising phases)
    model = NJEPA("vit_base_patch16_224", phase1_end=0, phase2_end=0)

    trainer = pl.Trainer(
        max_epochs=300,
        callbacks=[TeacherStudentCallback()],
    )
    trainer.fit(model, dataloader)

    # Access trained encoder
    encoder = model.encoder.student
"""

from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn.functional as F

from stable_pretraining.backbone import (
    FlexibleTransformer,
    IJEPAMasking,
    MaskedEncoder,
    NoisingTeacherStudentWrapper,
)
from stable_pretraining import Module


@dataclass
class NJEPAOutput:
    """Output from NJEPA forward pass.

    :ivar loss: Prediction loss (0 in eval mode)
    :ivar embedding: Patch embeddings [B, N, D] for downstream use
    :ivar predictions: Predicted representations [B, N_tgt, D] (or context in eval)
    :ivar targets: Target representations [B, N_tgt, D] (or context in eval)
    :ivar num_targets: Number of target patches (0 in eval)
    :ivar num_context: Number of context patches (all patches in eval)
    """

    loss: torch.Tensor
    embedding: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor
    num_targets: int
    num_context: int


class NJEPA(Module):
    """NJEPA: I-JEPA with 3-phase noising target encoder update.

    Architecture is identical to I-JEPA:
        - **Context Encoder** (student): Encodes visible/context patches
        - **Target Encoder** (teacher): Updated via 3-phase noising schedule
        - **Predictor**: Lightweight transformer predicting targets from context

    The target encoder update follows three phases based on global optimizer steps:

    - **Phase 1** (``step <= phase1_end``): push target toward scale-invariant noise,
      preventing the target encoder from immediately collapsing to the student.
    - **Phase 2** (``phase1_end < step <= phase2_end``): EMA toward student + small noise,
      gradually transitioning the target toward the student representation.
    - **Phase 3** (``step > phase2_end``): standard EMA — identical to vanilla I-JEPA.

    Setting ``phase1_end=0`` and ``phase2_end=0`` disables noising and recovers vanilla I-JEPA.

    :param encoder_name: timm model name (e.g., ``"vit_base_patch16_224"``)
    :param predictor_embed_dim: Predictor hidden dimension (default: 384)
    :param predictor_depth: Number of predictor blocks (default: 6)
    :param num_targets: Number of target blocks to sample (default: 4)
    :param target_scale: (min, max) fraction of patches per target block
    :param target_aspect_ratio: (min, max) aspect ratio of target blocks
    :param context_scale: (min, max) fraction of non-target patches as context
    :param ema_decay_start: Initial EMA decay (default: 0.996)
    :param ema_decay_end: Final EMA decay (default: 1.0)
    :param phase1_end: Last step of Phase 1 (noise-only). Set 0 to disable.
    :param phase2_end: Last step of Phase 2 (EMA + noise). Must be >= phase1_end.
    :param noise_scale: Noise amplitude relative to param RMS in Phase 1 (default: 0.05)
    :param noise_floor_scale: Noise amplitude relative to param RMS in Phase 2 (default: 0.02)
    :param pretrained: Load pretrained encoder weights

    Example::

        # ViT-Small with default 3-phase schedule
        model = NJEPA("vit_small_patch16_224", predictor_embed_dim=192)

        # ViT-Base
        model = NJEPA("vit_base_patch16_224", predictor_embed_dim=384)

        # ViT-Large
        model = NJEPA(
            "vit_large_patch16_224",
            predictor_embed_dim=384,
            predictor_depth=12,
        )

        # Vanilla I-JEPA (no noising)
        model = NJEPA("vit_base_patch16_224", phase1_end=0, phase2_end=0)

    Note:
        - Use :class:`TeacherStudentCallback` for automatic teacher updates.
        - Access the trained encoder via ``model.encoder.student``.
        - ``model.encoder.global_step`` tracks optimizer steps for phase transitions.
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        phase1_end: int = 1000,
        phase2_end: int = 10000,
        noise_scale: float = 0.05,
        noise_floor_scale: float = 0.02,
        pretrained: bool = False,
    ):
        super().__init__()

        base_encoder = MaskedEncoder(
            encoder_name,
            masking=None,
            pretrained=pretrained,
        )
        self.encoder = NoisingTeacherStudentWrapper(
            base_encoder,
            phase1_end=phase1_end,
            phase2_end=phase2_end,
            noise_scale=noise_scale,
            noise_floor_scale=noise_floor_scale,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        embed_dim = base_encoder.embed_dim
        num_patches = base_encoder.default_grid_h * base_encoder.default_grid_w

        self.predictor = FlexibleTransformer(
            input_dim=embed_dim,
            hidden_dim=predictor_embed_dim,
            output_dim=embed_dim,
            num_patches=num_patches,
            depth=predictor_depth,
            num_heads=max(1, predictor_embed_dim // 64),
            self_attn=True,
            cross_attn=False,
            add_mask_token=True,
            use_adaln=False,
            num_prefix_tokens=0,
            zero_init_output=False,
        )

        self.masking = IJEPAMasking(
            num_targets=num_targets,
            target_scale=target_scale,
            target_aspect_ratio=target_aspect_ratio,
            context_scale=context_scale,
        )

        self.embed_dim = embed_dim
        self._fix_init_weight()

    def _encode(
        self,
        patches: torch.Tensor,
        indices: torch.Tensor,
        grid_h: int,
        grid_w: int,
        encoder: MaskedEncoder,
    ) -> torch.Tensor:
        B, _, D = patches.shape
        _, pos = encoder._get_pos_embed(grid_h, grid_w)
        x = patches + pos.expand(B, -1, -1)
        x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        x = encoder.vit.pos_drop(x)
        x = encoder.vit.blocks(x)
        return encoder.vit.norm(x)

    def forward(
        self, images: torch.Tensor, embedding_source: str = "teacher"
    ) -> NJEPAOutput:
        """Forward pass.

        Training mode: masks image, encodes context (student) and targets (teacher),
        predicts targets, returns smooth L1 loss.

        Eval mode: encodes all patches through student, returns zero loss.

        :param images: Input images [B, C, H, W]
        :param embedding_source: ``"teacher"`` (default) or ``"student"`` for the
            embedding output used by downstream probes. Only affects training mode.
        :return: :class:`NJEPAOutput`
        """
        if embedding_source not in ("teacher", "student"):
            raise ValueError(
                f"embedding_source must be 'teacher' or 'student', got '{embedding_source}'"
            )

        B = images.shape[0]
        grid_h, grid_w = self.encoder.student._get_grid_size(images)
        student_patches = self.encoder.student.patch_embed(images)
        teacher_patches = self.encoder.teacher.patch_embed(images)

        mask_out = self.masking(student_patches, grid_h, grid_w)

        if self.training:
            context = self._encode(
                student_patches,
                mask_out.context_idx,
                grid_h,
                grid_w,
                self.encoder.student,
            )

            with torch.no_grad():
                all_idx = (
                    torch.arange(grid_h * grid_w, device=images.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
                teacher_full = self._encode(
                    teacher_patches, all_idx, grid_h, grid_w, self.encoder.teacher
                )
                teacher_full_normed = F.layer_norm(
                    teacher_full,
                    [teacher_full.size(-1)],
                    weight=None,
                    bias=None,
                )
                D = teacher_full.size(-1)
                targets = torch.gather(
                    teacher_full_normed,
                    1,
                    mask_out.target_idx.unsqueeze(-1).expand(-1, -1, D),
                )

                if embedding_source == "teacher":
                    embedding = teacher_full
                else:
                    embedding = self._encode(
                        student_patches, all_idx, grid_h, grid_w, self.encoder.student
                    )

            N_tgt = mask_out.target_idx.shape[1]
            queries = torch.zeros(
                B, N_tgt, self.embed_dim, device=images.device, dtype=context.dtype
            )
            query_mask = torch.ones(B, N_tgt, device=images.device, dtype=torch.bool)
            predictions = self.predictor(
                context=context,
                queries=queries,
                context_idx=mask_out.context_idx,
                query_idx=mask_out.target_idx,
                query_mask=query_mask,
            )

            loss = F.smooth_l1_loss(predictions, targets, beta=1.0)
        else:
            with torch.no_grad():
                context = self._encode(
                    student_patches,
                    mask_out.context_idx,
                    grid_h,
                    grid_w,
                    self.encoder.student,
                )
            predictions = context
            targets = context
            embedding = context
            loss = torch.tensor(0.0, device=images.device)

        return NJEPAOutput(
            loss=loss,
            embedding=embedding,
            predictions=predictions,
            targets=targets,
            num_targets=mask_out.target_idx.shape[1],
            num_context=mask_out.context_idx.shape[1],
        )

    def _fix_init_weight(self):
        """Rescale attention proj and MLP output weights by depth (I-JEPA init)."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for encoder in (self.encoder.student, self.encoder.teacher):
            for layer_id, block in enumerate(encoder.vit.blocks):
                rescale(block.attn.proj.weight.data, layer_id + 1)
                rescale(block.mlp.fc2.weight.data, layer_id + 1)

        for layer_id, block in enumerate(self.predictor.blocks):
            rescale(block.attn.proj.weight.data, layer_id + 1)
            rescale(block.mlp.fc2.weight.data, layer_id + 1)
