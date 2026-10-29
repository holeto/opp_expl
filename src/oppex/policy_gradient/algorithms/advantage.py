"""Multi-step return estimators. Pure array functions, jit/vmap-safe."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _compose_affine(
  transform_i: tuple[jax.Array, jax.Array], transform_j: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
  """Compose two affine maps x ↦ bias + scale·x (i applied first, then j)."""
  bias_i, scale_i = transform_i
  bias_j, scale_j = transform_j
  return bias_j + scale_j * bias_i, scale_j * scale_i


def retrace(
  rewards: jax.Array,     # (T,)
  q_taken: jax.Array,     # (T,) — Q_target(s_t, a_t)
  v_target: jax.Array,    # (T,) — E_{a~μ}[Q_target(s_t, a)]
  discount: jax.Array,    # (T,) — γ·(1 − done_t)
  lambda_: float | jax.Array = 1.0,
  importance_sampling: float | jax.Array = 1.0,
  clip: float | jax.Array = 1.0,
  bootstrap_v: float | jax.Array = 0.0,
) -> jax.Array:           # (T,) — Retrace targets Q^ret(s_t, a_t)
  """Retrace(λ) targets (Munos et al., 2016) as a backward associative scan.

      Q^ret_t = r_t + γ_t·( V_{t+1} − c_{t+1}·Q_{t+1} + c_{t+1}·Q^ret_{t+1} ),
      c_t     = λ·min(clip, π/μ)

  On-policy (π = μ) the IS ratio is 1 and c_t = λ: Expected SARSA(λ).

  ``bootstrap_v`` is V(s_T) for the state *after* the last step. RLG fixed it
  at 0, which is right only when every row ends on a terminal. Rows of a
  fixed-length, auto-resetting rollout usually end mid-episode, so the caller
  passes E_μ[Q_target(s_T)] here. The trace coefficient past the window is 0,
  so the last target is the one-step ``r + γ_{T-1}·bootstrap_v``; when the last
  step is terminal ``γ_{T-1} = 0`` and ``bootstrap_v`` has no effect.

  Episode boundaries *inside* the row need no special handling: ``discount = 0``
  at a terminal cuts both the bootstrap and the trace, so the next episode's
  values never leak backwards.
  """
  c = jnp.broadcast_to(lambda_ * jnp.minimum(clip, importance_sampling), rewards.shape)
  bootstrap_v = jnp.broadcast_to(bootstrap_v, v_target.shape[1:])

  # Shift one step forward: index t holds the t+1 quantity.
  next_v = jnp.concatenate([v_target[1:], bootstrap_v[None]])
  next_q = jnp.concatenate([q_taken[1:], jnp.zeros_like(q_taken[:1])])
  next_c = jnp.concatenate([c[1:], jnp.zeros_like(c[:1])])

  bias = rewards + discount * (next_v - next_c * next_q)
  scale = discount * next_c

  rev_q_ret, _ = jax.lax.associative_scan(_compose_affine, (bias[::-1], scale[::-1]))
  return rev_q_ret[::-1]
