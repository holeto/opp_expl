"""Agent — the bridge between an Algorithm and its network(s).

An Agent owns network construction and exposes ``player_evaluate`` (plus
``public_evaluate`` / ``state_evaluate`` for the other observation rows of the
``Env`` table). Each returns ``(output_pytree, new_agent_state)``; the output
pytree carries whatever the algorithm's loss needs (logits, Q-values, …), so
the algorithm never imports a concrete network class.

Stateless agents return ``None`` from ``init_state`` and accept ``None`` back.
"""

from __future__ import annotations
import abc
from typing import Any

import jax


class Agent(abc.ABC):
  """Bridge between an Algorithm and one or more networks."""

  # True when the network also consumes ``state_representation`` (a privileged
  # critic). The rollout stores state_reps only for such agents: at HUNL batch
  # sizes they are a sizeable fraction of rollout memory.
  privileged: bool = False

  @abc.abstractmethod
  def init_params(self, key: jax.Array, dummy_obs: Any) -> Any:
    """Initialise and return all network parameters."""

  @abc.abstractmethod
  def init_state(self, params: Any) -> Any:
    """Return the initial recurrent carry (None for stateless agents)."""

  def dummy_obs(self, env: Any, env_state: Any, key: jax.Array) -> Any:
    """Single observation used to initialise network parameters."""
    return env.information_set(env_state, 0, key)

  def make_obs_step(self, infosets: Any, state_rep: Any) -> Any:
    """Observation for ``player_evaluate`` during one rollout step.

    infosets:  (P, ...) — information set of each player.
    state_rep: (...) or None — ground-truth state representation.
    """
    return infosets

  def eval_obs(self, rollout: Any) -> Any:
    """Batched observation for re-evaluating a stored rollout, (B, T, P, ...)."""
    return rollout.infosets

  @abc.abstractmethod
  def build_eval_fn(self, env: Any):
    """Return a jit-compiled ``(params, infoset_batch) -> logits_batch``.

    Used by exact exploitability evaluation, which only has information sets
    (no ground-truth state), so privileged agents must not depend on it for the
    actor.
    """

  @abc.abstractmethod
  def player_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[Any, Any]:
    """Forward pass on player-private information (infoset). obs may be batched."""

  def public_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[Any, Any]:
    """Forward pass on public information. Same network unless overridden."""
    return self.player_evaluate(params, state, obs)

  def state_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[Any, Any]:
    """Forward pass on privileged ground truth. Same network unless overridden."""
    return self.player_evaluate(params, state, obs)
