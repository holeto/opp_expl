"""Validate depth-limited re-solving against the fixed blueprint, then play it.

  uv run python scripts/solve_fixed_opponent.py checkpoints/qmmd_hunl_560k   # caches
  uv run python scripts/resolve_fixed_opponent.py checkpoints/qmmd_hunl_560k --play 100
  uv run python scripts/policy_gradient/play_slumbot.py checkpoints/qmmd_hunl_560k \\
      --resolve --hands 50

The ladder, cheapest first:

a. ``build_subgame`` from the preflop root with no depth limit reproduces the
   tree of ``build_preflop_tree``, node for node.
b. With the bucketed leaf values of the fixed-opponent PSCFR run and no
   opponent noise, the single-hand best response from the deal
   equals PSCFR's per-hand best-response cfv, for a sample of hands per seat.
   Same game, so any difference is a bug in reaches, card removal, terminals,
   leaves or seats.
c. Re-rooted at a later decision, with the belief computed from the history,
   the best response equals that node's value in the solve from the deal.
d. CFR's average reaches the best-response value (gap logged on every solve).
e. Rollout leaf values, averaged over the hands of a bucket, agree with the
   uniform-deal sample of the same leaf, up to Monte Carlo error.
f. ``--play N`` plays N hands against the blueprint in the env, with a
   blueprint-vs-blueprint control on the same deals, and evaluates every hand
   with AIVAT on per-history subgame values (``resolving_aivat``).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from oppex.cfr import equity, exploit, infoset_cfr, tree
from oppex.cfr.cards import HAND_CARDS, N_HANDS
from oppex.cfr.leaves import BucketedValues
from oppex.cfr.subgame import build_subgame
from oppex.envs import betting
from oppex.policy_gradient import leaf_values as lv
from oppex.policy_gradient.blueprint import N_TRANSFORMS, Blueprint
from oppex.policy_gradient.resolving_aivat import ResolvingAivat, report_aivat
from oppex.policy_gradient.resolving import (
  Resolver, bucketed_leaf_values, play_vs_blueprint, report_play,
)


def load_cache(run_dir, bp, games, buckets, seed=0, factor=5.0):
  # The fixed-opponent PSCFR caches of scripts/solve_fixed_opponent.py.
  d = os.path.join(run_dir, "fixed_opponent")
  tag = f"{bp.which}_step{bp.step}"
  with open(os.path.join(d, "tree.pkl"), "rb") as f:
    tr = pickle.load(f)
  with open(os.path.join(d, f"sigma_{tag}.pkl"), "rb") as f:
    mmd = pickle.load(f)
  vb = {}
  for fixed in (0, 1):
    name = f"leaves_fixed{fixed}_{tag}_n{games}_f{factor:g}_k{buckets}_s{seed}.pkl"
    with open(os.path.join(d, name), "rb") as f:
      vb[1 - fixed] = np.asarray(pickle.load(f)[0])
  return tr, mmd, vb


def check_builder(tr):
  sg = build_subgame(tr.rules, betting.initial_state(tr.rules), (), None)
  same = (len(sg.nodes) == len(tr.nodes) and len(sg.terminals) == len(tr.terminals)
          and all(a.child == b.child and a.path == b.path and a.alias == b.alias
                  and a.legal == b.legal for a, b in zip(sg.nodes, tr.nodes))
          and all(a.kind == b.kind and a.path == b.path and abs(a.const - b.const) < 1e-6
                  and abs(a.stake - b.stake) < 1e-6 for a, b in zip(sg.terminals, tr.terminals)))
  assert same, "build_subgame differs from build_preflop_tree"
  print(f"a. builder: identical to the preflop tree ({len(sg.nodes)} nodes, "
        f"{len(sg.leaves)} leaves)")


def exact_resolver(bp, ev, vb, bucket, tr, seat):
  # No depth limit, no opponent noise, PSCFR's leaf values: PSCFR's own game.
  paths = [tr.terminals[j].path for j in tr.leaves]

  def leaf_fn(t, r_term, cards, board, s):
    return bucketed_leaf_values(t, r_term, cards, s, vb[seat], bucket, paths)

  return Resolver(bp, ev, depth=None, iters=200, leaf_fn=leaf_fn)


def check_exact(bp, ev, tr, mmd, vb, bucket, n_hands, rng):
  owner = np.array([n.player for n in tr.nodes])
  profile = jnp.asarray(np.where((owner == 0)[:, None, None], mmd[0], mmd[1]))
  tau = jnp.zeros((2, len(tr.leaves), N_HANDS, N_TRANSFORMS)).at[..., 0].set(1.0)
  hands = rng.choice(N_HANDS, n_hands, replace=False)
  worst, gaps = 0.0, []
  for seat in (0, 1):
    vals = BucketedValues(jnp.asarray(vb[seat]), jnp.asarray(bucket))
    ref, _ = exploit.best_response(tr, profile, seat, ev, N_HANDS, leaf_values=vals,
                                   leaf_tau=tau, per_hand=True)
    res = exact_resolver(bp, ev, vb, bucket, tr, seat)
    for h in hands:
      res.new_hand(seat, HAND_CARDS[h])
      sol = res.solve((), [])
      br = infoset_cfr.best_response(sol.game)[0]
      worst = max(worst, abs(br - ref[h]) / max(abs(ref[h]), 1.0))
      gaps.append(sol.stats["gap"])
  print(f"b. single-hand BR vs PSCFR per-hand BR, {n_hands} hands x 2 seats: "
        f"max relative error {worst:.1e}")
  print(f"d. CFR (200 iterations) vs BR on those solves: mean gap {np.mean(gaps):.3f}, "
        f"max {np.max(gaps):.3f} chips")
  assert worst < 1e-3, "single-hand re-solve disagrees with PSCFR"


def check_reroot(bp, ev, tr, vb, bucket, rng):
  worst = 0.0
  for seat in (0, 1):
    res = exact_resolver(bp, ev, vb, bucket, tr, seat)
    for h in rng.choice(N_HANDS, 3, replace=False):
      res.new_hand(seat, HAND_CARDS[h])
      sol = res.solve((), [])
      _, node_vals, _, _ = infoset_cfr.best_response(sol.game)
      sg = sol.game.tree
      # Our decisions one or two plies below the deal, a few per hand.
      ours = [i for i, n in enumerate(sg.nodes)
              if n.player == seat and 1 <= len(n.path) <= 3][:4]
      for i in ours:
        sub = res.solve(sg.nodes[i].path, [])
        v = infoset_cfr.best_response(sub.game)[0]
        worst = max(worst, abs(v - node_vals[i]) / max(abs(node_vals[i]), 1.0))
  print(f"c. re-rooted BR vs node value of the solve from the deal: "
        f"max relative error {worst:.1e}")
  assert worst < 1e-3, "re-rooting changes the value: the belief is wrong"


def check_rollouts(bp, ev, bucket, games, leaf_file, buckets=(2, 9)):
  """Rollout leaf values against the uniform-deal sample of the same leaf.

  ``leaf_file`` is a ``sample_leaf_values.py`` sample: games dealt uniformly
  at one leaf, for every pair of continuations. Its mean payoff over games
  where our hand falls in bucket ``b`` and the opponent plays the blueprint
  is exactly what the rollouts estimate when averaged over the hands of ``b``
  with a uniform opponent. A single hand cannot be compared with a bucket
  mean: the strongest hands of a bucket are worth far more than its average.
  """
  from oppex.policy_gradient.resolving import LeafRollouts, compatible
  from oppex.policy_gradient.tree_policy import pad_paths

  old = lv.load_leaf(leaf_file)
  path = tuple(int(a) for a in old["path"])
  chunk = 8
  ro = LeafRollouts(bp, n_games=games, chunk=chunk)
  paths = jnp.asarray(np.repeat(pad_paths([path], 8), chunk, 0))
  filler = jnp.arange(5, dtype=jnp.int32)
  worst = 0.0
  for seat in (0, 1):
    for b in buckets:
      rows = []
      for h in np.flatnonzero(bucket == b):
        q = compatible(HAND_CARDS[h], [])
        q = jnp.asarray(np.repeat((q / q.sum())[None], chunk, 0), jnp.float32)
        rows.append([float(np.asarray(ro._run(
          jax.random.PRNGKey(int(h)), paths, q, jnp.asarray(HAND_CARDS[h]), filler,
          jnp.int32(0), jnp.int32(seat), jnp.int32(t))[0]).mean()) for t in range(N_TRANSFORMS)])
      roll = np.mean(rows, 0)
      if seat == 0:
        cells = [old["payoff"][t, 0][bucket[old["h0"][t, 0]] == b] for t in range(N_TRANSFORMS)]
      else:
        cells = [-old["payoff"][0, t][bucket[old["h1"][0, t]] == b] for t in range(N_TRANSFORMS)]
      ref = np.array([c.mean() for c in cells])
      se = np.array([c.std() / np.sqrt(len(c)) for c in cells])
      z = (roll - ref) / se
      worst = max(worst, float(np.abs(z).max()))
      print(f"e. leaf {list(path)}, seat {seat}, bucket {b}: rollouts {np.round(roll, 1)} "
            f"vs uniform deal {np.round(ref, 1)} chips, z {np.round(z, 1)}")
  # The four t of one cell share games, so they move together.
  assert worst < 4.0, "rollout leaf values disagree with the uniform-deal sample"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir")
  ap.add_argument("--checkpoint")
  ap.add_argument("--params", choices=("params", "target", "magnet"), default="params")
  ap.add_argument("--cache-games", type=int, default=2000,
                  help="games per leaf of the PSCFR leaf cache used by checks b, c, e")
  ap.add_argument("--buckets", type=int, default=10)
  ap.add_argument("--boards", type=int, default=200_000)
  ap.add_argument("--check-hands", type=int, default=10)
  ap.add_argument("--rollout-check-games", type=int, default=64,
                  help="games per hand and leaf slot (8 slots) in check e")
  ap.add_argument("--leaf-file", default=None,
                  help="sample_leaf_values.py sample for check e (default: the 1-1 leaf)")
  ap.add_argument("--checks", default="abce", help="which checks to run, e.g. 'ae' or ''")
  ap.add_argument("--play", type=int, default=0, help="hands against the blueprint")
  ap.add_argument("--depth", type=int, default=4)
  ap.add_argument("--leaf-games", type=int, default=128)
  ap.add_argument("--cfr-iters", type=int, default=500)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--no-aivat", action="store_true")
  ap.add_argument("--no-cover-opponent", action="store_true",
                  help="skip AIVAT terms for opponent decisions outside every played subgame "
                       "instead of solving them when reached")
  args = ap.parse_args()
  if args.leaf_file is None:
    args.leaf_file = os.path.join(args.run_dir, "leaf_values",
                                  "leaf_1-1_params_step560000_n10000_f5_k10_s0.npz")

  bp = Blueprint(args.run_dir, args.checkpoint, args.params)
  ev = equity.preflop_ev_matrix(n_boards=args.boards, progress=False)
  rng = np.random.default_rng(args.seed)
  if args.checks:
    tr, mmd, vb = load_cache(args.run_dir, bp, args.cache_games, args.buckets)
    bucket = lv.bucket_hands(np.asarray(ev), args.buckets)
    if "a" in args.checks:
      check_builder(tr)
    if "b" in args.checks:
      check_exact(bp, ev, tr, mmd, vb, bucket, args.check_hands, rng)
    if "c" in args.checks:
      check_reroot(bp, ev, tr, vb, bucket, rng)
    if "e" in args.checks:
      check_rollouts(bp, ev, bucket, args.rollout_check_games, args.leaf_file)

  if args.play:
    out_dir = os.path.join(args.run_dir, "resolving")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"vs_blueprint_d{args.depth}_n{args.leaf_games}_"
                                f"it{args.cfr_iters}_s{args.seed}.jsonl")
    res = Resolver(bp, ev, depth=args.depth, leaf_games=args.leaf_games,
                   iters=args.cfr_iters, seed=args.seed)
    aivat = None
    if not args.no_aivat:
      aivat = ResolvingAivat(res, cover_opponent=not args.no_cover_opponent)
    print(f"\nf. {args.play} hands against the blueprint -> {out}")
    with open(out, "w") as f:
      recs = play_vs_blueprint(bp, res, args.play, seed=args.seed, log=f, aivat=aivat)
    mbb = 1e3 / bp.env.big_blind
    report_play(recs, mbb)
    if aivat is not None:
      report_aivat(recs, mbb)


if __name__ == "__main__":
  main()
