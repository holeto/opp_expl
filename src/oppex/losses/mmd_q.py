"""MMD-Q loss: policy head + Q-head, no V-head. Per (timestep, player) → scalar.

The policy gradient uses an all-actions advantage built from the Polyak
target network's Q-vector (stop-gradiented), with the sampled action replaced
by its multi-step Retrace(λ) target:

    q_vec(a) = Q_target(s, a),   q_vec(a_taken) = q_target

The advantage is centred by the baseline E_π[q_vec] over the SAME vector, so
Σ_a π(a)·A(s, a) = 0. Using one consistent (target-net) Q-vector for both the
per-action values and the baseline is what keeps it centred; mixing the live
Q-head, a target-net baseline and a multi-step target across actions would
leave a systematic offset that biases the update.

For NeuRD (``loss_type=RNAD``) the per-action regret is reconstructed from the
sampled-action advantage with the all-actions baseline trick; the magnet
regularisation is a separate KL penalty rather than being folded into the
Q-values.

The live Q-head is supervised only via the Q-loss (live Q(s, a_taken) →
q_target, PPO-style clipped) and never enters the advantage.

Ported from RLG ``losses/mmd_q.py`` minus the α-entmax path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .neurd import estimate_baseline_regrets, neurd_loss
from .ppo import ppo_policy_loss, ppo_value_loss
from ..algorithms.types import KLDirection, LossType
from ..policy import policy_entropy_loss, policy_kl, policy_log_probs, policy_probs


def mmd_q_loss(
  q_values: jax.Array,         # (A,) — current Q(s, :)
  logits: jax.Array,           # (A,) — current policy logits
  legal_actions: jax.Array,    # (A,) — boolean action mask
  actions: jax.Array,          # ()   — action taken during rollout
  sample_logits: jax.Array,    # (A,) — rollout policy logits
  sample_q_values: jax.Array,  # (A,) — rollout-time Q(s, :), for Q-loss clipping
  magnet_logits: jax.Array,    # (A,) — magnet policy logits
  q_target: jax.Array,         # ()   — Retrace(λ) target for Q(s, a_taken) (stop-grad)
  target_q_values: jax.Array,  # (A,) — target-net Q(s, :) (stop-grad)
  clip_eps: float,
  vf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  old_policy_coef: float = 0.0,
  neurd_clip: float = 5.0,
  neurd_threshold: float = 2.0,
  kl_direction: KLDirection = KLDirection.REVERSE,
  loss_type: LossType = LossType.MMD,
) -> tuple[jax.Array, dict]:
  pi = policy_probs(logits, legal_actions)
  log_probs_all = policy_log_probs(logits, legal_actions)
  sample_pi = policy_probs(sample_logits, legal_actions)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions)

  if kl_direction == KLDirection.FORWARD:
    _kl = lambda lp, lq: policy_kl(lq, lp)
  else:
    _kl = lambda lp, lq: policy_kl(lp, lq)

  if loss_type == LossType.RNAD:
    advantage = q_target - jnp.take_along_axis(q_values, actions[None], axis=-1)
    regrets = estimate_baseline_regrets(q_values, advantage, pi, sample_pi, actions)
    policy_loss = -neurd_loss(logits, legal_actions, regrets, neurd_clip, neurd_threshold)
  else:
    q_vec = target_q_values.at[actions].set(q_target)
    advantage = q_target - jnp.dot(q_vec, pi)
    policy_loss = ppo_policy_loss(
      log_probs_all[actions], sample_log_probs_all[actions], advantage, clip_eps
    )

  # ── Q-loss: live Q(s, a_taken) → Retrace target, PPO-clipped ───────────────
  q_loss = ppo_value_loss(q_values[actions], sample_q_values[actions], q_target, clip_eps)

  # ── Regularisers ───────────────────────────────────────────────────────────
  magnet_loss = _kl(log_probs_all, magnet_log_probs_all)
  old_kl_loss = _kl(log_probs_all, sample_log_probs_all)
  entropy_loss = policy_entropy_loss(pi, log_probs_all)

  total = (
    policy_loss
    + vf_coef * q_loss
    + ent_coef * entropy_loss
    + magnet_coef * magnet_loss
    + old_policy_coef * old_kl_loss
  )
  return total, {
    "policy_loss": policy_loss,
    "q_loss": q_loss,
    "entropy_loss": entropy_loss,
    "magnet_loss": magnet_loss,
    "old_kl_loss": old_kl_loss,
  }
