from __future__ import annotations
from typing import Any

import jax
import flax.linen as nn

from .base import Head, Torso


class TwinHead(nn.Module):
  """Shared torso whose features feed two independent heads.

  Returns ((out1, out2), new_state). For QMMD: head1 = CategoricalHead
  (policy logits), head2 = QHead.
  """

  torso: Torso
  head1: Head
  head2: Head

  def __call__(self, x: Any, state: Any) -> tuple[tuple[Any, Any], Any]:
    features, new_state = self.torso(x, state)
    return (self.head1(features), self.head2(features)), new_state

  def _zero_state(self) -> Any:
    return self.torso._zero_state()

  def init_state(self, params: Any) -> Any:
    return self.torso.init_state(params["torso"])

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self._zero_state())["params"]


class SplitTwinHead(nn.Module):
  """Two independent (torso, head) pathways fed from different observations.

  Input x is a tuple (actor_obs, critic_obs): the actor sees the information
  set, the critic sees privileged ground truth (``state_representation``).
  The pathways share no parameters and may have different input sizes.

  Returns ((actor_out, critic_out), (new_actor_state, new_critic_state)).
  """

  actor_torso: Torso
  actor_head: Head
  critic_torso: Torso
  critic_head: Head

  def __call__(
    self, x: tuple[Any, Any], state: tuple[Any, Any]
  ) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    actor_obs, critic_obs = x
    state_actor, state_critic = state
    actor_features, new_state_actor = self.actor_torso(actor_obs, state_actor)
    critic_features, new_state_critic = self.critic_torso(critic_obs, state_critic)
    return (
      self.actor_head(actor_features),
      self.critic_head(critic_features),
    ), (new_state_actor, new_state_critic)

  def _zero_state(self) -> tuple[Any, Any]:
    return self.actor_torso._zero_state(), self.critic_torso._zero_state()

  def init_state(self, params: Any) -> tuple[Any, Any]:
    return (
      self.actor_torso.init_state(params["actor_torso"]),
      self.critic_torso.init_state(params["critic_torso"]),
    )

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self._zero_state())["params"]
