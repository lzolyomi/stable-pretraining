"""Quick smoke test for NJEPA — run this to verify the implementation.

Expected output (all lines should print without error):
    imports OK
    wrapper step tracking OK
    train forward+backward OK, loss: <some value>
    eval forward OK
    vanilla JEPA recovery (phase1=0, phase2=0) OK
    ALL OK

Usage:
    python benchmarks/imagenet1k/test_njepa_smoke.py
"""

import torch
import torch.nn as nn

from stable_pretraining.backbone import NoisingTeacherStudentWrapper
from stable_pretraining.methods.njepa import NJEPA

print("imports OK")

# ------------------------------------------------------------------ #
# 1. Wrapper: step tracking and phase transitions                     #
# ------------------------------------------------------------------ #
student = nn.Linear(4, 4)
w = NoisingTeacherStudentWrapper(
    student, phase1_end=2, phase2_end=4, noise_scale=0.05, noise_floor_scale=0.02
)
w.train()

w.update_teacher()
assert int(w.global_step.item()) == 1, f"expected 1, got {w.global_step.item()}"
w.update_teacher()
assert int(w.global_step.item()) == 2, f"expected 2, got {w.global_step.item()}"
w.update_teacher()  # should enter Phase 2
assert int(w.global_step.item()) == 3, f"expected 3, got {w.global_step.item()}"

print("wrapper step tracking OK")

# ------------------------------------------------------------------ #
# 2. Full forward + backward (tiny ViT, CPU)                         #
# ------------------------------------------------------------------ #
model = NJEPA(
    "vit_tiny_patch16_224",
    predictor_embed_dim=64,
    predictor_depth=2,
    phase1_end=2,
    phase2_end=5,
)
imgs = torch.randn(2, 3, 224, 224)

model.train()
out = model(imgs)
out.loss.backward()
print(f"train forward+backward OK, loss: {out.loss.item():.4f}")

# ------------------------------------------------------------------ #
# 3. Eval mode (no loss)                                              #
# ------------------------------------------------------------------ #
model.eval()
with torch.no_grad():
    out_eval = model(imgs)
assert out_eval.loss.item() == 0.0, f"eval loss should be 0, got {out_eval.loss.item()}"
assert out_eval.embedding.shape[0] == 2, "embedding batch dim mismatch"
print(f"eval forward OK, embedding shape: {out_eval.embedding.shape}")

# ------------------------------------------------------------------ #
# 4. Vanilla I-JEPA recovery: phase1=0, phase2=0 → standard EMA     #
# ------------------------------------------------------------------ #
model2 = NJEPA(
    "vit_tiny_patch16_224",
    predictor_embed_dim=64,
    predictor_depth=2,
    phase1_end=0,
    phase2_end=0,
)
model2.train()
# Snapshot a teacher param before the first update
param_name = next(iter(dict(model2.encoder.teacher.named_parameters())))
t_before = dict(model2.encoder.teacher.named_parameters())[param_name].data.clone()
s = dict(model2.encoder.student.named_parameters())[param_name].data.clone()
model2.encoder.update_teacher()
t_after = dict(model2.encoder.teacher.named_parameters())[param_name].data.clone()
m = float(model2.encoder.ema_coefficient.item())
expected = m * t_before + (1.0 - m) * s
assert torch.allclose(t_after, expected, atol=1e-5), "vanilla EMA update mismatch"
print("vanilla JEPA recovery (phase1=0, phase2=0) OK")

print("\nALL OK")
