"""Fixed-length, auto-resetting rollout collection.

Each of the B rows is a window of exactly T steps over a *continuing* stream of
games: when a game ends, a fresh one is dealt in place and the row carries on.
The env state at the end of the window is handed back and becomes the start of
the next window, so a game cut by the window boundary resumes in the next
training iteration rather than being restarted or discarded.

Why not RLG's scheme. RLG ran every row from a fresh ``init_state`` for exactly
``env.max_length`` steps and masked whatever came after the terminal. That
costs nothing for Goofspiel (every game lasts ``n_cards`` steps) but for HUNL
``max_length`` is a tail bound (~124 steps at 200 BB) while typical hands end in
a handful of steps — most of every row would be masked padding.

What it costs. A row now usually ends mid-game, so its last Retrace target
must bootstrap from V(s_T) instead of assuming 0 (see ``advantage.retrace``).
``collect_rollout`` therefore also returns the observation at s_T.

Chance. Each reset draws a fresh ``init_state`` key, so all chance for a game is
still sampled once at its start, matching the ``Env`` contract.

Truncation. A game reaching ``env.max_length`` steps without terminating is
force-reset (history buffers are sized to ``max_length``) and recorded as a
zero-reward terminal. This never fires for Goofspiel and is ~0.1% of HUNL hands
under realistic play; ``truncated_frac`` in the metrics keeps an eye on it.
"""

from __future__ import annotations
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ...agents.base import Agent
from ...envs.base import Env


class Rollout(NamedTuple):
  """One row of T steps. Leading axis T; per-player fields add a P axis."""

  agent_states: Any       # (T, ...)      carry fed to the agent at each step
  infosets: jax.Array     # (T, P, d)
  state_reps: Any         # (T, d_s), or None unless agent.privileged
  legal_actions: jax.Array  # (T, P, A)
  actions: jax.Array      # (T, P)
  rewards: jax.Array      # (T, P)
  dones: jax.Array        # (T,) bool — the game ended after this step (terminal or truncated)
  truncated: jax.Array    # (T,) bool — subset of dones: cut at max_length, not a real terminal
  agent_output: Any       # (T, P, ...) — e.g. PolicyQOutput at rollout time


class RolloutCarry(NamedTuple):
  """Per-row state that persists between windows."""

  env_state: Any
  agent_state: Any
  ep_len: jax.Array  # () int32


class Bootstrap(NamedTuple):
  """What the algorithm needs to evaluate V(s_T) after the window."""

  obs: Any                  # agent observation at s_T, (P, ...)
  legal_actions: jax.Array  # (P, A)
  agent_state: Any


def _observe(env: Env, agent: Agent, env_state: Any, key: jax.Array):
  P = env.num_players
  pids = jnp.arange(P)
  obs_key, state_key = jax.random.split(key)
  infosets = jax.vmap(env.information_set, in_axes=(None, 0, 0))(
    env_state, pids, jax.random.split(obs_key, P)
  )
  state_rep = env.state_representation(env_state, state_key) if agent.privileged else None
  legal = jax.vmap(env.legal_actions, in_axes=(None, 0))(env_state, pids)
  return infosets, state_rep, legal


def _select(pred: jax.Array, on_true: Any, on_false: Any) -> Any:
  return jax.tree.map(lambda a, b: jnp.where(pred, a, b), on_true, on_false)


def collect_rollout(
  env: Env,
  agent: Agent,
  params: Any,
  carry: RolloutCarry,
  rng: jax.Array,
  rollout_len: int,
  normalize_rewards: bool = True,
) -> tuple[RolloutCarry, Rollout, Bootstrap]:
  """Collect ``rollout_len`` self-play steps for one row (vmap over rows).

  All players are evaluated every step in one batched forward pass; in a
  sequential game the non-acting player has a single legal action, so its
  sampled action is forced and ignored by the env.
  """
  initial_agent_state = agent.init_state(params)

  def scan_step(carry: RolloutCarry, key: jax.Array):
    obs_key, act_key, reset_key = jax.random.split(key, 3)
    infosets, state_rep, legal = _observe(env, agent, carry.env_state, obs_key)

    obs = agent.make_obs_step(infosets, state_rep)
    agent_out, next_agent_state = agent.player_evaluate(params, carry.agent_state, obs)

    logits = jnp.where(legal, agent_out.logits, -jnp.inf)
    actions = jax.vmap(jax.random.categorical)(
      jax.random.split(act_key, env.num_players), logits
    )

    next_env_state, rewards, done, _ = env.apply_action(carry.env_state, actions)
    if normalize_rewards:
      rewards = rewards / env.max_reward

    truncated = (~done) & (carry.ep_len + 1 >= env.max_length)
    reset = done | truncated
    new_carry = RolloutCarry(
      env_state=_select(reset, env.init_state(reset_key), next_env_state),
      agent_state=_select(reset, initial_agent_state, next_agent_state),
      ep_len=jnp.where(reset, 0, carry.ep_len + 1).astype(jnp.int32),
    )
    step = Rollout(
      agent_states=carry.agent_state,
      infosets=infosets,
      state_reps=state_rep,
      legal_actions=legal,
      actions=actions,
      rewards=rewards,
      dones=reset,
      truncated=truncated,
      agent_output=agent_out,
    )
    return new_carry, step

  carry, rollout = jax.lax.scan(scan_step, carry, jax.random.split(rng, rollout_len))

  obs_key = jax.random.fold_in(rng, rollout_len)
  infosets, state_rep, legal = _observe(env, agent, carry.env_state, obs_key)
  bootstrap = Bootstrap(
    obs=agent.make_obs_step(infosets, state_rep),
    legal_actions=legal,
    agent_state=carry.agent_state,
  )
  return carry, rollout, bootstrap


def collect_rollouts(
  env: Env,
  agent: Agent,
  params: Any,
  carry: RolloutCarry,  # batched over B
  rng: jax.Array,
  rollout_len: int,
  normalize_rewards: bool = True,
) -> tuple[RolloutCarry, Rollout, Bootstrap]:
  """``collect_rollout`` vmapped over the B rows of ``carry``."""
  B = carry.ep_len.shape[0]
  return jax.vmap(
    lambda c, k: collect_rollout(env, agent, params, c, k, rollout_len, normalize_rewards)
  )(carry, jax.random.split(rng, B))
