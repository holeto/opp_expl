from __future__ import annotations
from typing import Any

import jax
import flax.linen as nn

from .base import Torso
from .modules import Activation, Normalization


class MLPTorso(Torso):
  """MLP feature extractor: hidden layers (Dense → activation → norm), then a
  projection to ``feature_dim``."""

  hidden: tuple[int, ...]
  activation: Activation = Activation(kind="relu")
  norm: Normalization = Normalization(kind="none")

  @nn.compact
  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    for h in self.hidden:
      x = Normalization(kind=self.norm.kind, num_groups=self.norm.num_groups)(
        self.activation(nn.Dense(h)(x))
      )
    return nn.Dense(self.feature_dim)(x), state
