from __future__ import annotations

from enum import StrEnum


class LossType(StrEnum):
  MMD = "mmd"     # PPO-clipped policy gradient on the sampled action
  RNAD = "rnad"   # NeuRD on the all-actions regret vector


class MagnetUpdateType(StrEnum):
  PERIODIC = "periodic"        # hard copy of params every magnet_interval steps
  INCREMENTAL = "incremental"  # Polyak: magnet ← τ·params + (1−τ)·magnet ("moving magnet")


class KLDirection(StrEnum):
  FORWARD = "forward"   # KL(magnet ‖ current) — mode-covering
  REVERSE = "reverse"   # KL(current ‖ magnet) — mode-seeking (default)


# Hyperparameters that may be driven by an optax schedule of the step counter.
MMD_SCHEDULABLE = frozenset({
  "clip_eps", "vf_coef", "ent_coef",
  "magnet_coef", "old_policy_coef",
  "target_update_rate", "magnet_update_rate",
  "neurd_clip", "neurd_threshold",
})
