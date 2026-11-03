"""The blueprint's preflop strategy on the public tree of ``oppex.cfr``.

Fixed-opponent PSCFR needs the opponent as a behaviour strategy
``(n_nodes, n_hands, n_actions)`` over the public tree. This module reads it
off the network: for every node of the player and every one of the 1326 hands,
it replays the node's public line in the env with that hand dealt and evaluates
the policy. The tree must be built from the env's own rules (``env.rules``), or
node paths and env states would disagree.

**Duplicate atoms.** The tree builder strikes a bet atom that lands on the same
betting state as a sibling, e.g. a bin that clips to the all-in, but the env
keeps it legal and the network may play it. Its mass belongs to the surviving
sibling, which ``DecisionNode.alias`` names. Anything else the network plays
that the tree does not have is an error, not something to renormalise away.

``replay`` and ``where_b`` are shared with the leaf sampler, which replays the
same kind of padded paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..cfr.cards import HAND_CARDS, N_HANDS
from ..policy import policy_probs
from .blueprint import Blueprint


def where_b(mask, new, old):
  """Per-game select between two batched pytrees, ``mask`` of shape ``(B,)``."""
  return jax.tree_util.tree_map(
    lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - 1)), n, o), new, old
  )


def pad_paths(paths, min_width: int = 1) -> np.ndarray:
  """``(n, width)`` int32, each path left-aligned and padded with ``-1``.

  The width is a power of two of at least ``min_width``, so that the jitted
  callers see few distinct shapes.
  """
  longest = max([len(p) for p in paths] + [min_width, 1])
  width = 1 << (longest - 1).bit_length()
  out = np.full((len(paths), width), -1, np.int32)
  for i, p in enumerate(paths):
    out[i, : len(p)] = p
  return out


def replay(env, s, paths):
  """Apply each game's padded ``(B, D)`` path of atoms to the batched state ``s``.

  Padding entries leave the game untouched, so paths of different lengths share
  one trace. Both seats' entries of the action carry the atom; the env reads
  only the acting player's. A loop rather than unrolling keeps the graph small
  for long histories.
  """
  apply_b = jax.vmap(env.apply_action)

  def body(j, s):
    atom = paths[:, j]
    new = apply_b(s, jnp.stack([atom, atom], -1).astype(jnp.int32))[0]
    return where_b(atom >= 0, new, s)

  return jax.lax.fori_loop(0, paths.shape[1], body, s)


def _other_cards() -> np.ndarray:
  # Two cards disjoint from each hand, for the seat whose cards nobody reads.
  out = np.zeros((N_HANDS, 2), np.int32)
  for h, (c0, c1) in enumerate(HAND_CARDS):
    out[h] = [c for c in range(4) if c not in (c0, c1)][:2]
  return out


def blueprint_tree_strategy(
  bp: Blueprint, tree, player: int, chunk_nodes: int = 16,
) -> np.ndarray:
  """The blueprint's ``(n_nodes, n_hands, n_actions)`` strategy for ``player``.

  Rows of ``player``'s nodes hold the network policy, with duplicate atoms
  folded into their survivor. The other player's rows are uniform over legal
  actions, so the result is always a full profile.
  """
  env = bp.env
  mine = [i for i, n in enumerate(tree.nodes) if n.player == player]
  n_act = tree.n_actions
  if env.num_actions != n_act:
    raise ValueError(f"env has {env.num_actions} atoms, the tree {n_act}")

  legal = np.array([n.legal for n in tree.nodes], np.float32)
  out = legal[:, None, :] / legal.sum(-1)[:, None, None]
  out = np.repeat(out, N_HANDS, 1)
  alias = np.array([n.alias for n in tree.nodes], np.int32)

  hole = np.zeros((N_HANDS, 2, 2), np.int32)
  hole[:, player] = HAND_CARDS
  hole[:, 1 - player] = _other_cards()
  hole = jnp.asarray(hole)
  key = jax.random.PRNGKey(0)
  infoset_b = jax.vmap(env.information_set, in_axes=(0, None, None))
  legal_b = jax.vmap(env.legal_actions, in_axes=(0, None))

  @jax.jit
  def probs(paths):
    # paths (k, D) -> (k, n_hands, n_actions), every hand at every node.
    k = paths.shape[0]
    s0 = env.init_state(key)
    s = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (k * N_HANDS,) + x.shape), s0)
    s = s._replace(hole_cards=jnp.tile(hole, (k, 1, 1)))
    s = replay(env, s, jnp.repeat(paths, N_HANDS, 0))
    seat = jnp.int32(player)
    p = policy_probs(bp.eval_fn(bp.params, infoset_b(s, seat, key)), legal_b(s, seat))
    ok = (s.cur_player == seat) & ~s.done & (s.street == 0)
    return p.reshape(k, N_HANDS, -1), ok.reshape(k, N_HANDS).all(-1)

  padded = pad_paths([tree.nodes[i].path for i in mine])
  for lo in range(0, len(mine), chunk_nodes):
    idx = mine[lo: lo + chunk_nodes]
    # A fixed chunk shape keeps it to one compile; the tail is padded.
    rows = np.zeros((chunk_nodes, padded.shape[1]), np.int32) - 1
    rows[: len(idx)] = padded[lo: lo + len(idx)]
    p, ok = probs(jnp.asarray(rows))
    p, ok = np.asarray(p, np.float64)[: len(idx)], np.asarray(ok)[: len(idx)]
    if not ok.all():
      bad = [idx[j] for j in np.flatnonzero(~ok)]
      raise AssertionError(f"replaying the paths of nodes {bad} does not reach "
                           f"player {player}'s preflop decision")
    for j, i in enumerate(idx):
      row = np.zeros((N_HANDS, n_act))
      for a in range(n_act):
        if alias[i, a] >= 0:
          row[:, alias[i, a]] += p[j, :, a]
      lost = 1.0 - row.sum(-1)
      if np.abs(lost).max() > 1e-4:
        raise AssertionError(
          f"node {i}: the blueprint puts up to {np.abs(lost).max():.2e} mass on "
          "atoms the tree does not have (is the tree built with allow_limp=True?)"
        )
      out[i] = row / row.sum(-1, keepdims=True)
  return out.astype(np.float32)
