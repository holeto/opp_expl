from __future__ import annotations

import jax
import flax.linen as nn

from .base import Head


class CategoricalHead(Head):
  """Raw logits over n_actions; masking and softmax happen at the point of use."""

  n_actions: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return nn.Dense(self.n_actions)(x)


class QHead(Head):
  """Q-values for each action: shape (..., n_actions)."""

  n_actions: int
  # Small init keeps early Q-values near zero so Retrace targets stay in the
  # reward range at the start of training (avoids bootstrap instability).
  output_scale: float = 0.01

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return nn.Dense(
      self.n_actions,
      kernel_init=nn.initializers.variance_scaling(
        self.output_scale, "fan_in", "truncated_normal"
      ),
    )(x)
