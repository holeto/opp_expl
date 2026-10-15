"""Policy + Q-value agents for QMMD (no V-head).

The state value is never predicted directly; QMMD derives it on the fly as
``V(s) = E_π[Q(s, :)]``.
"""

from __future__ import annotations
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import Agent


class PolicyQOutput(NamedTuple):
  """logits: raw action logits (..., A).  q_values: Q(s, :) (..., A)."""

  logits: jax.Array
  q_values: jax.Array


class PolicyQAgent(Agent):
  """TwinHead network: head1 = CategoricalHead (logits), head2 = QHead.

  Actor and critic share a torso and both see the information set.
  """

  def __init__(self, network: nn.Module) -> None:
    self.network = network

  def init_params(self, key: jax.Array, dummy_obs: Any) -> Any:
    return self.network.init_params(key, dummy_obs)

  def init_state(self, params: Any) -> Any:
    return self.network.init_state(params)

  def player_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[PolicyQOutput, Any]:
    (logits, q_values), new_state = self.network.apply({"params": params}, obs, state)
    return PolicyQOutput(logits=logits, q_values=q_values), new_state

  def build_eval_fn(self, env: Any):
    network = self.network
    zero_state = network._zero_state()

    def _apply(params, infoset_batch):
      def single(obs):
        (logits, _), _ = network.apply({"params": params}, obs, zero_state)
        return logits

      return jax.vmap(single)(infoset_batch)

    return jax.jit(_apply)


class PrivilegedPolicyQAgent(Agent):
  """Q-critic on ground truth, actor on the information set.

  Requires a ``SplitTwinHead``: the actor pathway reads the player's infoset,
  the critic pathway reads ``state_representation`` (both hole cards in HUNL,
  the full prize order in random Goofspiel). The critic is only used to build
  training targets and advantages, never to act, so the learned policy remains
  a function of the information set alone.

  The observation passed to the network is the tuple ``(infosets, state_reps)``
  with the single state representation broadcast across the P players.
  """

  privileged = True

  def __init__(self, network: nn.Module) -> None:
    self.network = network

  def init_params(self, key: jax.Array, dummy_obs: Any) -> Any:
    return self.network.init_params(key, dummy_obs)

  def init_state(self, params: Any) -> Any:
    return self.network.init_state(params)

  def dummy_obs(self, env: Any, env_state: Any, key: jax.Array) -> Any:
    return (env.information_set(env_state, 0, key), env.state_representation(env_state, key))

  def make_obs_step(self, infosets: Any, state_rep: Any) -> Any:
    P = infosets.shape[0]
    return (infosets, jnp.broadcast_to(state_rep[None], (P,) + state_rep.shape))

  def eval_obs(self, rollout: Any) -> Any:
    infosets, state_reps = rollout.infosets, rollout.state_reps   # (B,T,P,d), (B,T,d_s)
    P = infosets.shape[2]
    state_reps = jnp.broadcast_to(
      state_reps[:, :, None], infosets.shape[:2] + (P,) + state_reps.shape[2:]
    )
    return (infosets, state_reps)

  def player_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[PolicyQOutput, Any]:
    (logits, q_values), new_state = self.network.apply({"params": params}, obs, state)
    return PolicyQOutput(logits=logits, q_values=q_values), new_state

  def build_eval_fn(self, env: Any):
    """Actor-only logits; the critic pathway gets a dummy state rep.

    The SplitTwinHead pathways are independent, so the dummy critic input has
    no effect on the logits — it only satisfies the network's input structure.
    """
    network = self.network
    zero_state = network._zero_state()
    dummy_state_rep = jnp.asarray(
      env.state_representation(env.init_state(jax.random.key(0)), jax.random.key(0))
    )

    def _apply(params, infoset_batch):
      def single(obs):
        (logits, _), _ = network.apply({"params": params}, (obs, dummy_state_rep), zero_state)
        return logits

      return jax.vmap(single)(infoset_batch)

    return jax.jit(_apply)
