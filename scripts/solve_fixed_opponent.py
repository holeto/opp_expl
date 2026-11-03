"""Exploit the MMD blueprint preflop: PSCFR against a fixed opponent.

  uv run python scripts/solve_fixed_opponent.py checkpoints/qmmd_hunl_560k \\
      --iters 300 --dcfr --play 20000
  uv run python scripts/policy_gradient/play_slumbot.py checkpoints/qmmd_hunl_560k \\
      --exploiter checkpoints/qmmd_hunl_560k/fixed_opponent/<solution>.npz --hands 50

The tree is the blueprint's own preflop abstraction (its rules, all bet atoms,
limps allowed), with a matrix leaf wherever preflop closes without an all-in.
For each seat ``x`` the other seat is frozen at the blueprint's preflop strategy
and, at every leaf, at the single continuation "keep playing the blueprint".
CFR then updates seat ``x`` only, which converges to a best response.

Steps, each cached under ``<run_dir>/fixed_opponent/``:

1. Build the tree and read the blueprint's strategy off it, per player.
2. Sample the leaf values with the frozen player pinned to the blueprint.
3. Check the bucketed leaf contraction against the dense one on real leaves,
   and report the blueprint-vs-blueprint value of the tree model.
4. Solve each seat. The log compares the exploiter's average-strategy value with
   the exact best-response value against the same frozen opponent. The gap
   must shrink toward zero. At the end the frozen rows must be bit-identical,
   and the zero-sum and reach invariants must hold.
5. Save both seats as an ``ExploiterPolicy``, and with ``--play N`` play N hands
   against the blueprint in the env.

The numbers are a model of the game, not a measurement of HUNL. The leaf
values are coarse Monte Carlo estimates, and the blueprint does not know that
the exploiter changes its postflop play at the leaf.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import jax.numpy as jnp
import numpy as np

from oppex.cfr import equity, exploit, solver, tree
from oppex.cfr.cards import N_HANDS
from oppex.cfr.leaves import BucketedValues, Leaves, leaf_action_cfv
from oppex.policy_gradient import leaf_values as lv
from oppex.policy_gradient.blueprint import N_TRANSFORMS, TRANSFORM_NAMES, Blueprint
from oppex.policy_gradient.exploiter import ExploiterPolicy, play_vs_blueprint
from oppex.policy_gradient.tree_policy import blueprint_tree_strategy


def _cached(file, build, what):
  # Load `file` if it exists, otherwise build and pickle it.
  if os.path.exists(file):
    with open(file, "rb") as f:
      print(f"  {what}: cached {file}", flush=True)
      return pickle.load(f)
  t0 = time.time()
  obj = build()
  with open(file, "wb") as f:
    pickle.dump(obj, f)
  print(f"  {what}: built in {time.time() - t0:.0f}s -> {file}", flush=True)
  return obj


def check_bucketed(values: BucketedValues, n_check: int = 4, tol: float = 1e-4) -> float:
  """Bucketed vs dense ``leaf_action_cfv`` on the first leaves, random reaches."""
  rng = np.random.default_rng(0)
  vb = np.asarray(values.vb[:n_check], np.float64)
  bucket = np.asarray(values.bucket)
  dense = jnp.asarray(np.stack([lv.leaf_value_tensor(v, bucket) for v in vb]))
  sub = BucketedValues(values.vb[:n_check], values.bucket)
  worst = 0.0
  for p in (0, 1):
    r = jnp.asarray(rng.random((len(vb), N_HANDS)), jnp.float32)
    tau = jnp.asarray(rng.dirichlet(np.ones(N_TRANSFORMS), (len(vb), N_HANDS)), jnp.float32)
    a, b = leaf_action_cfv(dense, r, tau, p), leaf_action_cfv(sub, r, tau, p)
    worst = max(worst, float(jnp.abs(a - b).max() / jnp.abs(a).max()))
  assert worst < tol, f"bucketed leaf contraction differs from dense by {worst:.2e}"
  return worst


def seat_values(tr, sigma, ev, leaves):
  # Value of each seat in chips, chance constant applied.
  v0, v1 = solver.root_value(tr, sigma, ev, N_HANDS, leaves=leaves)
  return float(v0), float(v1)


def solve_seat(tr, ev, x, mmd_sigma, values, args, bb):
  """Seat ``x``'s tables against the frozen blueprint in seat ``1 - x``."""
  fixed = 1 - x
  legal = solver.legal_mask(tr)
  discount = solver.Discount(*map(float, args.dcfr.split(","))) if args.dcfr else None
  scheme = dict(plus=args.plus, linear=args.linear, discount=discount)
  tab0 = solver.fix_strategy(tr, solver.init_tables(tr, N_HANDS, n_transforms=N_TRANSFORMS),
                             fixed, mmd_sigma)
  tab = tab0
  print(f"\n  seat {x} exploits the blueprint in seat {fixed}")
  print(f"  {'iter':>6}  {'avg value(mbb)':>15}  {'BR value(mbb)':>14}  {'gap(mbb)':>9}  "
        f"{'s/iter':>7}")
  t0, log_at = time.time(), {1, 10, args.iters} | set(range(0, args.iters + 1, args.log_every))
  for it in range(1, args.iters + 1):
    tab = solver.cfr_step(tr, tab, legal, ev, N_HANDS, leaf_values=values,
                          fixed_player=fixed, **scheme)
    if it in log_at:
      tab.regret.block_until_ready()
      dt = (time.time() - t0) / it
      avg, tau = solver.average_strategy(tab, legal), solver.average_leaf_strategy(tab)
      v = seat_values(tr, avg, ev, Leaves(values, tau))[x]
      br, _ = exploit.best_response(tr, avg, x, ev, N_HANDS, leaf_values=values, leaf_tau=tau)
      print(f"  {it:>6}  {v / bb * 1e3:>15.1f}  {br / bb * 1e3:>14.1f}  "
            f"{(br - v) / bb * 1e3:>9.2f}  {dt:>7.3f}", flush=True)

  # The frozen player's rows must never have been written back.
  mine = jnp.asarray([n.player == fixed for n in tr.nodes])
  frozen = (bool((tab.regret[mine] == tab0.regret[mine]).all())
            and bool((tab.strategy_sum[mine] == tab0.strategy_sum[mine]).all())
            and bool((tab.leaf_regret[fixed] == tab0.leaf_regret[fixed]).all())
            and bool((tab.leaf_strategy_sum[fixed] == tab0.leaf_strategy_sum[fixed]).all()))
  assert frozen, "the fixed player's tables changed"
  avg, tau = solver.average_strategy(tab, legal), solver.average_leaf_strategy(tab)
  zs = solver.check_zero_sum(tr, avg, ev, N_HANDS, leaves=Leaves(values, tau))
  rc = solver.check_reach_conservation(tr, avg, N_HANDS)
  print(f"  frozen rows unchanged | zero-sum {zs:.1e} | joint reach {rc:.1e}")
  return avg, tau


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir")
  ap.add_argument("--checkpoint")
  ap.add_argument("--params", choices=("params", "target", "magnet"), default="params")
  ap.add_argument("--iters", type=int, default=300)
  ap.add_argument("--log-every", type=int, default=50)
  ap.add_argument("--plus", action="store_true")
  ap.add_argument("--linear", action="store_true")
  ap.add_argument("--dcfr", nargs="?", const="1.5,0,2", metavar="A,B,G")
  ap.add_argument("--games", type=int, default=2000, help="leaf rollouts per leaf and cell")
  ap.add_argument("--factor", type=float, default=5.0)
  ap.add_argument("--buckets", type=int, default=10)
  ap.add_argument("--chunk-leaves", type=int, default=16)
  ap.add_argument("--boards", type=int, default=200_000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--seats", type=int, nargs="+", default=[0, 1],
                  help="exploiter seats to solve (the saved policy needs both)")
  ap.add_argument("--play", type=int, default=0, help="hands against the blueprint")
  args = ap.parse_args()

  bp = Blueprint(args.run_dir, args.checkpoint, args.params)
  env = bp.env
  bb = env.big_blind
  if not env._drop_duplicate_allin:
    raise ValueError("the tree drops duplicate all-ins, so the env must too")
  out_dir = os.path.join(args.run_dir, "fixed_opponent")
  os.makedirs(out_dir, exist_ok=True)
  tag = f"{bp.which}_step{bp.step}"
  print(f"{bp.checkpoint} (step {bp.step}, {bp.which})")

  # ── 1. Tree and the blueprint on it ────────────────────────────────────────
  tr = _cached(
    os.path.join(out_dir, "tree.pkl"),
    lambda: tree.build_preflop_tree(env.rules, allow_limp=True, allow_bets=True,
                                    matrix_leaves=True, max_depth=64),
    "tree",
  )
  print(f"  {len(tr.nodes)} decisions, {len(tr.terminals)} terminals, "
        f"{len(tr.leaves)} matrix leaves")
  mmd = _cached(
    os.path.join(out_dir, f"sigma_{tag}.pkl"),
    lambda: [blueprint_tree_strategy(bp, tr, p) for p in (0, 1)],
    "blueprint strategy",
  )
  owner = np.array([n.player for n in tr.nodes])
  # The blueprint's full profile: each node from its owner's extraction.
  mmd_profile = jnp.asarray(np.where((owner == 0)[:, None, None], mmd[0], mmd[1]))
  ev = equity.preflop_ev_matrix(n_boards=args.boards, progress=False)

  # ── 2. Leaf values, the frozen player pinned to the blueprint ──────────────
  buckets = lv.bucket_hands(np.asarray(ev), args.buckets)
  paths = [tr.terminals[j].path for j in tr.leaves]
  values = {}
  for x in args.seats:
    fixed = 1 - x
    name = (f"leaves_fixed{fixed}_{tag}_n{args.games}_f{args.factor:g}"
            f"_k{args.buckets}_s{args.seed}.pkl")

    def build(fixed=fixed):
      samples = lv.sample_leaves(bp, paths, args.games, args.factor, args.seed,
                                 fixed_player=fixed, chunk_leaves=args.chunk_leaves)
      vals, counts = lv.bucketed_leaf_values(samples, buckets, args.buckets)
      print(f"    truncated {samples.truncated.mean():.1e}, "
            f"empty bucket cells {(counts == 0).mean():.1%}")
      return np.asarray(vals.vb), counts

    vb, _ = _cached(os.path.join(out_dir, name), build, f"leaf values, seat {fixed} fixed")
    values[x] = BucketedValues(jnp.asarray(vb), jnp.asarray(buckets, jnp.int32))

  # ── 3. Checks before solving ───────────────────────────────────────────────
  for x, v in values.items():
    print(f"  bucketed vs dense leaf contraction (seat {1 - x} fixed): "
          f"{check_bucketed(v):.1e} relative")
  blueprint_tau = jnp.zeros((2, len(tr.leaves), N_HANDS, N_TRANSFORMS)).at[..., 0].set(1.0)
  for x, v in values.items():
    v0, v1 = seat_values(tr, mmd_profile, ev, Leaves(v, blueprint_tau))
    print(f"  blueprint vs blueprint in the tree model: SB {v0 / bb * 1e3:+.1f}, "
          f"BB {v1 / bb * 1e3:+.1f} mbb/hand (leaves of seat {1 - x} fixed)")

  # ── 4. Solve ───────────────────────────────────────────────────────────────
  sol_sigma, sol_tau, model = np.asarray(mmd_profile).copy(), np.asarray(blueprint_tau).copy(), {}
  for x in args.seats:
    avg, tau = solve_seat(tr, ev, x, mmd_profile, values[x], args, bb)
    sol_sigma[owner == x] = np.asarray(avg)[owner == x]
    sol_tau[x] = np.asarray(tau[x])
    model[x] = seat_values(tr, avg, ev, Leaves(values[x], tau))[x]
    t_freq = np.asarray(tau[x]).mean((0, 1))
    print(f"  model value {model[x] / bb * 1e3:+.1f} mbb/hand; continuation frequency "
          + ", ".join(f"{n} {f:.2f}" for n, f in zip(TRANSFORM_NAMES, t_freq)))
  if len(model) == 2:
    print(f"\n  seat-averaged model value {sum(model.values()) / 2 / bb * 1e3:+.1f} mbb/hand")

  # ── 5. Save and play ───────────────────────────────────────────────────────
  pol = ExploiterPolicy.from_solution(tr, sol_sigma, sol_tau, args.factor)
  file = os.path.join(out_dir, f"solution_{tag}_n{args.games}_it{args.iters}.npz")
  pol.save(file, checkpoint=bp.checkpoint, seats=np.asarray(args.seats))
  print(f"  saved {file}")
  if set(args.seats) != {0, 1}:
    print("  only one seat solved: the other plays the blueprint")
  if args.play:
    t0 = time.time()
    res = play_vs_blueprint(bp, pol, args.play, seed=args.seed)
    mbb = 1e3 / bb
    for name, v in (("exploiter", res["payoff"]), ("control", res["control"]),
                    ("difference", res["payoff"] - res["control"])):
      print(f"  {name:>10}: {v.mean() * mbb:+8.1f} ± {1.96 * v.std() / np.sqrt(v.size) * mbb:.1f}"
            " mbb/hand")
    for x in (0, 1):
      sel = res["seat"] == x
      print(f"  seat {x}: {res['payoff'][sel].mean() * mbb:+8.1f} mbb/hand, "
            f"reached a leaf {(res['leaf'][sel] >= 0).mean():.1%}")
    print(f"  {args.play} hands in {time.time() - t0:.0f}s, left the tree {res['lost'].mean():.1e}, "
          f"truncated {res['truncated'].mean():.1e}")


if __name__ == "__main__":
  main()
