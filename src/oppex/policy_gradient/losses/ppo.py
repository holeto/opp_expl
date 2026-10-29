"""PPO clipped surrogate pieces, per (timestep, player) sample → scalar."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax


def ppo_policy_loss(
  log_prob: jax.Array,         # () — log π(a) under the current policy
  sample_log_prob: jax.Array,  # () — log μ(a) under the rollout policy
  advantage: jax.Array,        # ()
  clip_eps: float = 0.2,
) -> jax.Array:
  ratio = jnp.exp(log_prob - sample_log_prob)
  advantage = lax.stop_gradient(advantage)
  return -jnp.minimum(
    ratio * advantage,
    jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage,
  )


def ppo_value_loss(
  values: jax.Array,         # () — current estimate
  sample_values: jax.Array,  # () — estimate at rollout time
  returns: jax.Array,        # () — regression target
  clip_eps: float = 0.2,
) -> jax.Array:
  v_clipped = sample_values + jnp.clip(values - sample_values, -clip_eps, clip_eps)
  return 0.5 * jnp.maximum((values - returns) ** 2, (v_clipped - returns) ** 2)
