"""Network building blocks: Torso (features + recurrent carry) and Head.

Every network maps ``(x, state) -> (outputs, new_state)``. Stateless torsos
return ``state`` unchanged and use ``None`` as their carry; the signature still
threads it so recurrent torsos can be added without touching the agents or the
rollout.
"""

from __future__ import annotations
from typing import Any

import jax
import flax.linen as nn


class Torso(nn.Module):
  """Feature extractor: (x, state) -> (features, new_state).

  Stateless torsos ignore state and return it unchanged.
  Recurrent torsos carry meaningful state between steps; their initial state is
  a learnable parameter obtained via ``init_state(params)``.
  """

  feature_dim: int

  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    raise NotImplementedError

  def _zero_state(self) -> Any:
    """Dummy zero state used only for tracing inside init_params."""
    return None

  def init_state(self, _params: Any) -> Any:
    """Return the (learnable) initial state extracted from params."""
    return None

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self._zero_state())["params"]


class Head(nn.Module):
  """Maps features to a task-specific output."""

  def __call__(self, x: jax.Array) -> Any:
    raise NotImplementedError

  def init_params(self, key: jax.Array, x: jax.Array) -> Any:
    return self.init(key, x)["params"]
