"""BFS layered game-tree extraction (history-unique rows, sparse transitions)."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from ..envs.base import Env
from ..envs.goofspiel import Goofspiel
from .types import (
  EK_CHANCE,
  EK_PLAYER0,
  EK_PLAYER1,
  EK_SIMULTANEOUS,
  GameTreeLayer,
  LayerTransition,
  LayeredGameTree,
  NK_CHANCE,
  NK_PLAYER,
  NK_SIMULTANEOUS,
  NK_TERMINAL,
)


def _batch_dim(stacked: Any) -> int:
  """Leading batch size of a stacked env-state pytree."""
  for leaf in jax.tree.leaves(stacked):
    if hasattr(leaf, "shape") and len(leaf.shape) > 0:
      return int(leaf.shape[0])
  raise ValueError("empty or scalar-only state batch")


def _stack_states(states: list[Any]) -> Any:
  return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def _make_vm_tree_obs(env: Env):
  """Single fused vmapped kernel for frontier metadata (one compile per batch size)."""

  def row(state):
    return (
      env.current_player(state),
      env.legal_actions(state, jnp.int32(0)),
      env.legal_actions(state, jnp.int32(1)),
      env.information_set(state, jnp.int32(0), jax.random.key(0)),
      env.information_set(state, jnp.int32(1), jax.random.key(0)),
    )

  return jax.jit(jax.vmap(row))


def _make_vm_tree_apply(env: Env):
  return jax.jit(jax.vmap(env.apply_action))


def _enumerate_initial_states(env: Env) -> list[tuple[Any, float]]:
  """All initial states with their probabilities.

  Chance is resolved entirely in ``init_state`` (see ``oppex.envs.base``), so
  enumerating the initial states enumerates all chance outcomes; the tree gets
  one artificial chance root over them when there is more than one.
  """
  if isinstance(env, Goofspiel):
    return _goofspiel_inits(env)
  raise ValueError(
    f"Cannot enumerate initial states for {type(env).__name__}. "
    "Add a case to _enumerate_initial_states."
  )


def _goofspiel_inits(env: Goofspiel) -> list[tuple[Any, float]]:
  base = env.init_state(jax.random.PRNGKey(0))
  if env.prize_order != "random":
    return [(base, 1.0)]
  perms = list(permutations(range(1, env.n_cards + 1)))
  prob = 1.0 / len(perms)
  return [(base._replace(prize_deck=jnp.array(p, dtype=jnp.int32)), prob) for p in perms]


def _chance_layer(depth: int, num_actions: int) -> GameTreeLayer:
  z = np.zeros((1, num_actions), dtype=bool)
  return GameTreeLayer(
    depth=depth,
    n_nodes=1,
    node_kind=np.array([NK_CHANCE], dtype=np.uint8),
    acting_player=np.array([-1], dtype=np.int8),
    legal_mask_p0=z.copy(),
    legal_mask_p1=z.copy(),
    info_set_id_p0=(b"",),
    info_set_id_p1=(b"",),
    reward_p0=np.zeros(1, dtype=np.float32),
    reward_p1=np.zeros(1, dtype=np.float32),
    state_batch=None,
  )


def _layer_from_states(
  env: Env,
  depth: int,
  states: list[Any],
  *,
  vm_obs: Any,
) -> GameTreeLayer:
  """Build metadata rows for non-terminal states (batched vmapped)."""
  if not states:
    raise ValueError("empty state list")
  stacked = _stack_states(states)
  N = len(states)

  cur_arr, legal0, legal1, iset0, iset1 = vm_obs(stacked)
  cur_arr = np.asarray(cur_arr)
  legal0 = np.asarray(legal0).astype(bool)
  legal1 = np.asarray(legal1).astype(bool)
  iset0 = np.asarray(iset0)
  iset1 = np.asarray(iset1)

  node_kind = np.zeros(N, dtype=np.uint8)
  acting = np.full(N, -1, dtype=np.int8)
  for i in range(N):
    cp = int(cur_arr[i])
    if cp == -1:
      node_kind[i] = NK_SIMULTANEOUS
    else:
      node_kind[i] = NK_PLAYER
      acting[i] = cp

  info_p0 = tuple(iset0[i].tobytes() for i in range(N))
  info_p1 = tuple(iset1[i].tobytes() for i in range(N))

  return GameTreeLayer(
    depth=depth,
    n_nodes=N,
    node_kind=node_kind,
    acting_player=acting,
    legal_mask_p0=legal0,
    legal_mask_p1=legal1,
    info_set_id_p0=info_p0,
    info_set_id_p1=info_p1,
    reward_p0=np.zeros(N, dtype=np.float32),
    reward_p1=np.zeros(N, dtype=np.float32),
    state_batch=stacked,
  )


def extract_game_tree(env: Env) -> LayeredGameTree:
  """BFS layered game-tree extraction (no cross-branch merging).

  Uses fused ``jit(vmap)`` for frontier observations and for ``apply_action``,
  and batched gathers for the next frontier.
  """
  layers: list[GameTreeLayer] = []
  transitions: list[LayerTransition] = []

  vm_obs = _make_vm_tree_obs(env)
  vm_apply = _make_vm_tree_apply(env)

  init_states = _enumerate_initial_states(env)
  multi_root = not (len(init_states) == 1 and init_states[0][1] >= 1.0 - 1e-9)

  depth_cursor = 0
  frontier_stacked: Any | None

  if multi_root:
    layers.append(_chance_layer(depth_cursor, env.num_actions))
    depth_cursor += 1
    init_list = [s for s, _ in init_states]
    probs = np.array([float(p) for _, p in init_states], dtype=np.float64)
    layers.append(
      _layer_from_states(env, depth_cursor, init_list, vm_obs=vm_obs),
    )
    K = len(init_list)
    transitions.append(
      LayerTransition(
        parent_idx=np.zeros(K, dtype=np.int32),
        child_idx=np.arange(K, dtype=np.int32),
        edge_prob=probs,
        edge_kind=np.full(K, EK_CHANCE, dtype=np.uint8),
        action_p0=np.full(K, -1, dtype=np.int32),
        action_p1=np.full(K, -1, dtype=np.int32),
      )
    )
    depth_cursor += 1
    frontier_stacked = _stack_states(init_list)
    frontier_rows = np.arange(K, dtype=np.int32)
  else:
    s0 = init_states[0][0]
    layers.append(
      _layer_from_states(env, depth_cursor, [s0], vm_obs=vm_obs),
    )
    depth_cursor += 1
    frontier_stacked = jax.tree.map(lambda x: jnp.expand_dims(x, 0), s0)
    frontier_rows = np.array([0], dtype=np.int32)

  while frontier_stacked is not None:
    N = _batch_dim(frontier_stacked)
    assert N == len(frontier_rows), "frontier_rows must align with stacked batch"

    cur_layer = layers[-1]
    cur_arr = cur_layer.acting_player[frontier_rows]
    legal0 = cur_layer.legal_mask_p0[frontier_rows]
    legal1 = cur_layer.legal_mask_p1[frontier_rows]

    pair_sidx: list[int] = []
    pair_acts: list[Any] = []
    meta: list[dict[str, Any]] = []

    for i in range(N):
      cp = int(cur_arr[i])
      la0 = [a for a in range(env.num_actions) if legal0[i, a]]
      la1 = [a for a in range(env.num_actions) if legal1[i, a]]
      keys: list[Any] = []
      avecs: list[Any] = []

      if cp == -1:
        for a0 in la0:
          for a1 in la1:
            avecs.append(np.array([a0, a1], dtype=np.int32))
            keys.append((a0, a1))
      else:
        for a in la0 if cp == 0 else la1:
          avec = np.zeros(2, dtype=np.int32)
          avec[cp] = a
          avecs.append(avec)
          keys.append(a)

      start = len(pair_sidx)
      pair_sidx.extend([i] * len(avecs))
      pair_acts.extend(avecs)

      meta.append(
        {
          "cp": cp,
          "keys": keys,
          "sl": (start, start + len(avecs)),
        }
      )

    idx_arr = jnp.asarray(np.array(pair_sidx, dtype=np.int32))
    acts_arr = jnp.asarray(np.stack(pair_acts))
    batch_s = jax.tree.map(lambda x: x[idx_arr], frontier_stacked)
    ns_batch, rw_batch, done_batch, _ = vm_apply(batch_s, acts_arr)

    rw_np = np.asarray(rw_batch)
    done_np = np.asarray(done_batch).astype(bool)
    M = int(rw_np.shape[0])

    cur_n, legal0_n, legal1_n, iset0_n, iset1_n = vm_obs(ns_batch)
    cur_n = np.asarray(cur_n)
    legal0_n = np.asarray(legal0_n).astype(bool)
    legal1_n = np.asarray(legal1_n).astype(bool)
    iset0_n = np.asarray(iset0_n)
    iset1_n = np.asarray(iset1_n)

    parent_idx = np.zeros(M, dtype=np.int32)
    edge_kind = np.zeros(M, dtype=np.uint8)
    action_p0 = np.full(M, -1, dtype=np.int32)
    action_p1 = np.full(M, -1, dtype=np.int32)
    edge_prob = np.ones(M, dtype=np.float64)

    node_kind = np.zeros(M, dtype=np.uint8)
    acting = np.full(M, -1, dtype=np.int8)
    lm0 = np.zeros((M, env.num_actions), dtype=bool)
    lm1 = np.zeros((M, env.num_actions), dtype=bool)
    rew0 = np.zeros(M, dtype=np.float32)
    rew1 = np.zeros(M, dtype=np.float32)
    info0: list[bytes] = [b""] * M
    info1: list[bytes] = [b""] * M

    pending_rows: list[int] = []

    for i in range(N):
      m = meta[i]
      cp = m["cp"]
      s0, _ = m["sl"]
      ek = EK_SIMULTANEOUS if cp == -1 else (EK_PLAYER0 if cp == 0 else EK_PLAYER1)

      for j_loc, key in enumerate(m["keys"]):
        j = s0 + j_loc
        parent_idx[j] = int(frontier_rows[i])
        edge_kind[j] = ek
        if cp == -1:
          a0, a1 = key  # type: ignore[misc]
          action_p0[j] = int(a0)
          action_p1[j] = int(a1)
        else:
          action_p0[j] = int(key) if cp == 0 else -1
          action_p1[j] = int(key) if cp == 1 else -1

        if done_np[j]:
          node_kind[j] = NK_TERMINAL
          rew0[j] = float(rw_np[j, 0])
          rew1[j] = float(rw_np[j, 1])
        else:
          pending_rows.append(j)
          cp_j = int(cur_n[j])
          if cp_j == -1:
            node_kind[j] = NK_SIMULTANEOUS
          else:
            node_kind[j] = NK_PLAYER
            acting[j] = cp_j
          lm0[j] = legal0_n[j]
          lm1[j] = legal1_n[j]
          info0[j] = iset0_n[j].tobytes()
          info1[j] = iset1_n[j].tobytes()

    child_idx = np.arange(M, dtype=np.int32)
    state_batch_next = ns_batch

    layers.append(
      GameTreeLayer(
        depth=depth_cursor,
        n_nodes=M,
        node_kind=node_kind,
        acting_player=acting,
        legal_mask_p0=lm0,
        legal_mask_p1=lm1,
        info_set_id_p0=tuple(info0),
        info_set_id_p1=tuple(info1),
        reward_p0=rew0,
        reward_p1=rew1,
        state_batch=state_batch_next,
      )
    )
    transitions.append(
      LayerTransition(
        parent_idx=parent_idx,
        child_idx=child_idx,
        edge_prob=edge_prob,
        edge_kind=edge_kind,
        action_p0=action_p0,
        action_p1=action_p1,
      )
    )
    depth_cursor += 1

    if pending_rows:
      pr = jnp.array(pending_rows, dtype=jnp.int32)
      frontier_stacked = jax.tree.map(lambda x: x[pr], ns_batch)
      frontier_rows = np.array(pending_rows, dtype=np.int32)
    else:
      frontier_stacked = None

  return LayeredGameTree(layers=layers, transitions=transitions)
