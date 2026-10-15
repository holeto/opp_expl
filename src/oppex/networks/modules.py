from __future__ import annotations

import jax
import jax.nn as jnn
import jax.numpy as jnp
import flax.linen as nn


_ACTIVATIONS = {
  "none": lambda x: x,
  "relu": jnn.relu,
  "gelu": jnn.gelu,
  "tanh": jnp.tanh,
  "sigmoid": jnn.sigmoid,
  "silu": jnn.silu,
  "elu": jnn.elu,
  "leaky_relu": jnn.leaky_relu,
  "softplus": jnn.softplus,
}


class Activation(nn.Module):
  """Stateless activation selected by kind.

  The field is ``kind`` rather than ``name`` because ``flax.linen.Module.name``
  is Flax's scope identifier and must not be shadowed.
  """

  kind: str = "relu"

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.kind not in _ACTIVATIONS:
      raise ValueError(f"Unknown activation '{self.kind}'. Options: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[self.kind](x)


class Normalization(nn.Module):
  """Normalization layer selected by kind: 'layer', 'rms', 'group' or 'none'."""

  kind: str = "none"
  num_groups: int = 32

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    if self.kind == "none":
      return x
    if self.kind == "layer":
      return nn.LayerNorm()(x)
    if self.kind == "rms":
      return nn.RMSNorm()(x)
    if self.kind == "group":
      return nn.GroupNorm(self.num_groups)(x)
    raise ValueError(f"Unknown normalization '{self.kind}'. Options: layer, rms, group, none")
