"""Matrix-valued leaves for depth-limited search, sampled from the blueprint.

At a depth-limit leaf each player chooses one of ``N_TRANSFORMS`` continuations
(see ``blueprint``), so the leaf is worth ``V[t0, t1, h0, h1]`` chips to player
0. This module estimates that tensor by playing the rest of the hand out with
the biased blueprints.

**Sampling.** For each pair ``(t0, t1)`` the same ``n_games`` deals are played.
Every action is drawn by inverse CDF from one uniform per (game, env step).
These common random numbers make the 16 cells differ by the transformations
rather than by the luck of the deal. Each game replays the leaf's preflop
``path`` through ``env.apply_action`` first, so the action history the network
sees is one it met in training.

**From games to hand pairs.** A few thousand games cannot cover 1326 x 1326
hand pairs, so hands are grouped into ``K`` buckets of similar preflop equity
against a random hand. Whole isomorphism classes go into one bucket, so suit
relabelling never moves a hand between buckets. The leaf value of a hand pair
is the mean payoff of its bucket pair:

    V[t0, t1, h0, h1] = Vb[t0, t1, b(h0), b(h1)] * MASK[h0, h1]

Like ``EV`` in ``oppex.cfr.equity``, ``V`` is stored already multiplied by the
card-removal mask. A solver must therefore never mask it again. The estimator
is coarse, but it defines one fixed game, so an exact solver comparison on it
is still exact.

Payoffs are net chips from the start of the hand, the same convention as the
fold constants and ``stake * EV`` of the preflop tree. A game still running at
``env.max_length`` is truncated with zero reward, as in training.

**Every leaf at once.** The blueprint's full preflop tree has over a thousand
leaves, so ``sample_leaves`` gives each game its own padded path and samples a
fixed-size chunk of leaves per call: one compile for the whole tree. Its
result is ``cfr.leaves.BucketedValues``, since the dense tensor would not fit.
With ``fixed_player=p``, player ``p`` only ever plays the blueprint, so only
the ``T`` cells with ``t_p = BLUEPRINT`` are sampled and that axis has size one.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..cfr.cards import HAND_CLASS, HAND_INDEX, MASK_NP, N_HANDS, N_OPP
from ..cfr.leaves import BucketedValues
from ..policy import policy_probs, sample_atom
from .blueprint import (
  BLUEPRINT, N_TRANSFORMS, Blueprint, bias_multipliers, transform_policy,
)
from .tree_policy import pad_paths, replay, where_b


class LeafSamples(NamedTuple):
  """Raw games of one leaf, every array shaped ``(T0, T1, n_games)``.

  ``sample_leaves`` returns the same fields with a leading leaf axis.
  """

  h0: np.ndarray          # player 0's hand index
  h1: np.ndarray
  payoff: np.ndarray      # chips to player 0
  truncated: np.ndarray   # hit env.max_length before the end of the hand


# ── Rollouts ─────────────────────────────────────────────────────────────────


def play_out(bp: Blueprint, s, game_keys, t0, t1, mult):
  """Play the batched states ``s`` to the end with biased blueprints.

  Player ``p`` plays the blueprint under transformation ``t_p``, a scalar or a
  per-game ``(B,)`` array. Every action is drawn by inverse CDF from a uniform
  fixed by (game key, env step), which gives common random numbers across
  calls that share keys. Returns the final states and the reward to player 0
  in the env's normalised units; a game cut at ``env.max_length`` gets none.
  """
  env, eval_fn = bp.env, bp.eval_fn
  obs_key = jax.random.PRNGKey(0)
  L = env.max_length
  apply_b = jax.vmap(env.apply_action)
  infoset_b = jax.vmap(env.information_set, in_axes=(0, 0, None))
  legal_b = jax.vmap(env.legal_actions)

  def active(s):
    return ~s.done & (s.step < L)

  def body(carry):
    s, payoff = carry
    cur = s.cur_player
    legal = legal_b(s, cur)
    probs = policy_probs(eval_fn(bp.params, infoset_b(s, cur, obs_key)), legal)
    probs = transform_policy(probs, legal, jnp.where(cur == 0, t0, t1), mult)

    u = jax.vmap(lambda k, i: jax.random.uniform(jax.random.fold_in(k, i)))(game_keys, s.step)
    atom = sample_atom(probs, u)

    live = active(s)
    new_s, rewards, _, _ = apply_b(s, jnp.stack([atom, atom], -1).astype(jnp.int32))
    return where_b(live, new_s, s), payoff + jnp.where(live, rewards[:, 0], 0.0)

  n = game_keys.shape[0]
  return jax.lax.while_loop(
    lambda c: active(c[0]).any(), body, (s, jnp.zeros(n, jnp.float32))
  )


def _build_rollout(bp: Blueprint, n_games: int, factor: float):
  # Returns a jitted (key, paths, t0, t1) -> (h0, h1, payoff, truncated) over
  # n_games, where game i first replays the padded path paths[i].
  env = bp.env
  mult = bias_multipliers(env.num_actions, factor)
  hand_index = jnp.asarray(HAND_INDEX)

  @jax.jit
  def rollout(key, paths, t0, t1):
    deal_key, act_key = jax.random.split(key)
    s = jax.vmap(env.init_state)(jax.random.split(deal_key, n_games))
    # Replay the public preflop line that leads to the leaf.
    s = replay(env, s, paths)
    s, payoff = play_out(bp, s, jax.random.split(act_key, n_games), t0, t1, mult)
    h0 = hand_index[s.hole_cards[:, 0, 0], s.hole_cards[:, 0, 1]]
    h1 = hand_index[s.hole_cards[:, 1, 0], s.hole_cards[:, 1, 1]]
    return h0, h1, payoff * env.starting_stack, ~s.done

  return rollout


def sample_leaf(
  bp: Blueprint, path: tuple[int, ...], n_games: int = 10_000,
  factor: float = 5.0, seed: int = 0,
) -> LeafSamples:
  """Play ``n_games`` from the leaf reached by ``path`` for every ``(t0, t1)``."""
  rollout = _build_rollout(bp, n_games, factor)
  paths = jnp.asarray(np.repeat(pad_paths([tuple(int(a) for a in path)]), n_games, 0))
  key = jax.random.PRNGKey(seed)
  out = np.zeros((4, N_TRANSFORMS, N_TRANSFORMS, n_games))
  for t0 in range(N_TRANSFORMS):
    for t1 in range(N_TRANSFORMS):
      # The same key for every pair: identical deals and action uniforms.
      res = rollout(key, paths, jnp.int32(t0), jnp.int32(t1))
      for i, r in enumerate(res):
        out[i, t0, t1] = np.asarray(r)
  return LeafSamples(
    h0=out[0].astype(np.int32), h1=out[1].astype(np.int32),
    payoff=out[2], truncated=out[3].astype(bool),
  )


def continuation_cells(fixed_player: int | None) -> tuple[list[int], list[int]]:
  """The ``t0`` and ``t1`` values sampled; a fixed player keeps the blueprint."""
  every = list(range(N_TRANSFORMS))
  return ([BLUEPRINT] if fixed_player == 0 else every,
          [BLUEPRINT] if fixed_player == 1 else every)


def sample_leaves(
  bp: Blueprint, paths, n_games: int = 2000, factor: float = 5.0, seed: int = 0,
  fixed_player: int | None = None, chunk_leaves: int = 16, progress: bool = True,
) -> LeafSamples:
  """Play ``n_games`` from every leaf in ``paths`` for every sampled ``(t0, t1)``.

  Returns ``LeafSamples`` with arrays ``(n_leaves, T0, T1, n_games)``, where the
  fixed player's axis, if any, has size one. Each chunk of leaves uses its own
  key, shared by all its cells, so cells differ only by the continuations.
  """
  t0s, t1s = continuation_cells(fixed_player)
  n_leaves = len(paths)
  rollout = _build_rollout(bp, chunk_leaves * n_games, factor)
  padded = pad_paths([tuple(int(a) for a in p) for p in paths])
  out = np.zeros((4, n_leaves, len(t0s), len(t1s), n_games))
  base = jax.random.PRNGKey(seed)
  for c, lo in enumerate(range(0, n_leaves, chunk_leaves)):
    hi = min(lo + chunk_leaves, n_leaves)
    # A fixed chunk shape keeps it to one compile; the tail repeats leaf `lo`.
    rows = np.repeat(padded[lo: lo + 1], chunk_leaves, 0)
    rows[: hi - lo] = padded[lo:hi]
    rows = jnp.asarray(np.repeat(rows, n_games, 0))
    key = jax.random.fold_in(base, c)
    for i, t0 in enumerate(t0s):
      for j, t1 in enumerate(t1s):
        res = rollout(key, rows, jnp.int32(t0), jnp.int32(t1))
        for k, r in enumerate(res):
          out[k, lo:hi, i, j] = np.asarray(r).reshape(chunk_leaves, n_games)[: hi - lo]
    if progress:
      print(f"    leaves {hi}/{n_leaves}", flush=True)
  return LeafSamples(
    h0=out[0].astype(np.int32), h1=out[1].astype(np.int32),
    payoff=out[2], truncated=out[3].astype(bool),
  )


# ── Bucketing ────────────────────────────────────────────────────────────────


def bucket_hands(ev: np.ndarray, k: int) -> np.ndarray:
  """``(N_HANDS,)`` bucket per hand, 0 = weakest, of roughly equal combo counts.

  Buckets are cut on class-averaged equity against a random hand and assigned
  by cumulative combo count, so an isomorphism class is never split.
  """
  eq = np.asarray(ev, np.float64).sum(1) / N_OPP
  cls_eq = np.bincount(HAND_CLASS, eq, 169) / np.bincount(HAND_CLASS, minlength=169)
  order = np.argsort(cls_eq, kind="stable")
  combos = np.bincount(HAND_CLASS, minlength=169)[order]
  # A class goes to the bucket containing the middle of its combo range.
  mid = np.cumsum(combos) - combos / 2.0
  cls_bucket = np.empty(169, np.int32)
  cls_bucket[order] = np.minimum((mid * k / N_HANDS).astype(np.int32), k - 1)
  return cls_bucket[HAND_CLASS]


def bucket_values(samples: LeafSamples, buckets: np.ndarray, k: int):
  """Bucket-pair means ``(T0, T1, k, k)`` and the per-cell sample counts.

  An empty cell falls back to the mean of its ``(t0, t1)``, which keeps every
  cell defined without inventing structure.
  """
  T0, T1 = samples.payoff.shape[:2]
  vb = np.zeros((T0, T1, k, k))
  counts = np.zeros((T0, T1, k, k), np.int64)
  for t0 in range(T0):
    for t1 in range(T1):
      cell = buckets[samples.h0[t0, t1]] * k + buckets[samples.h1[t0, t1]]
      n = np.bincount(cell, minlength=k * k)
      s = np.bincount(cell, samples.payoff[t0, t1], minlength=k * k)
      mean = samples.payoff[t0, t1].mean()
      vb[t0, t1] = np.where(n > 0, s / np.maximum(n, 1), mean).reshape(k, k)
      counts[t0, t1] = n.reshape(k, k)
  return vb, counts


def bucketed_leaf_values(samples: LeafSamples, buckets: np.ndarray, k: int):
  """``BucketedValues`` of a multi-leaf sample, plus the ``(L, T0, T1, k, k)`` counts."""
  per_leaf = [bucket_values(LeafSamples(*(x[l] for x in samples)), buckets, k)
              for l in range(samples.payoff.shape[0])]
  vb = np.stack([v for v, _ in per_leaf]).astype(np.float32)
  counts = np.stack([c for _, c in per_leaf])
  return BucketedValues(jnp.asarray(vb), jnp.asarray(buckets, jnp.int32)), counts


def leaf_value_tensor(vb: np.ndarray, buckets: np.ndarray) -> np.ndarray:
  """Expand bucket values to the mask-folded ``(T, T, N_HANDS, N_HANDS)`` float32."""
  return (vb[:, :, buckets[:, None], buckets[None, :]] * MASK_NP).astype(np.float32)


# ── Cache ────────────────────────────────────────────────────────────────────


def cache_path(
  bp: Blueprint, path: tuple[int, ...], n_games: int, factor: float, k: int, seed: int,
  run_dir: str,
) -> str:
  line = "-".join(str(int(a)) for a in path)
  name = (f"leaf_{line}_{bp.which}_step{bp.step}_n{n_games}_f{factor:g}"
          f"_k{k}_s{seed}.npz")
  return os.path.join(run_dir, "leaf_values", name)


def save_leaf(file: str, samples: LeafSamples, vb, counts, buckets, path, factor) -> None:
  os.makedirs(os.path.dirname(file), exist_ok=True)
  np.savez_compressed(
    file, h0=samples.h0, h1=samples.h1, payoff=samples.payoff,
    truncated=samples.truncated, vb=vb, counts=counts, buckets=buckets,
    path=np.asarray(path, np.int32), factor=np.float64(factor),
  )


def load_leaf(file: str) -> dict:
  """The saved arrays plus the expanded ``values`` tensor."""
  with np.load(file) as z:
    out = {name: z[name] for name in z.files}
  out["values"] = leaf_value_tensor(out["vb"], out["buckets"])
  return out
