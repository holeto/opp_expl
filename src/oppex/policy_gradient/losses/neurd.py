"""Neural Replicator Dynamics (NeuRD) loss and the all-actions regret estimate."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def neurd_loss(
  logits: jax.Array,
  legal_actions: jax.Array,
  advantages: jax.Array,
  clip: float = 10000,
  threshold: float = 2.0,
) -> jax.Array:
  """NeuRD loss (Hennes et al., 2019, https://arxiv.org/abs/1906.00190).

  Replicator dynamics in logit space, with the update for a logit switched off
  once it leaves ``[-threshold, threshold]`` in the direction of the update —
  this bounds the spread between any two logits by ``2·threshold``.
  Returns the quantity to *maximise*; callers negate it.
  """
  logits_centered = logits * legal_actions
  advantages = jnp.clip(advantages, -clip, clip)

  logit_can_decrease = logits_centered > -threshold
  logit_can_increase = logits_centered < threshold
  advantage_weights = (
    logit_can_decrease * jnp.minimum(0.0, advantages)
    + logit_can_increase * jnp.maximum(0.0, advantages)
  )
  loss = logits_centered * jax.lax.stop_gradient(advantage_weights)
  return jnp.sum(loss * legal_actions, axis=-1)


def estimate_baseline_regrets(
  value: jax.Array,              # (A,) per-action baseline values
  advantage: jax.Array,          # (1,) sampled-action advantage
  strategy: jax.Array,           # (A,) current π
  sampling_strategy: jax.Array,  # (A,) rollout μ
  action: jax.Array,             # ()   sampled action
) -> jax.Array:
  """All-actions regrets from one sampled action (Schmid et al., 2018,
  https://arxiv.org/abs/1809.03057): q = baseline + 1[a]·(u − baseline)/μ(a),
  centred by E_π[q]."""
  q_values = value + jax.nn.one_hot(action, value.shape[-1]) * advantage / jnp.take_along_axis(
    sampling_strategy, action[None], axis=-1
  )
  baseline = jnp.sum(jnp.where(strategy > 0, q_values * strategy, 0.0), axis=-1, keepdims=True)
  return q_values - baseline
