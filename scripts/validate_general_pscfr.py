"""The general public-state CFR must reproduce the poker solver, iteration by iteration.

  uv run python scripts/validate_general_pscfr.py [--iters 30]

The poker tree is written as a ``PublicGame`` by ``cfr.general.from_poker``
and both solvers run side by side. After every iteration the regrets,
strategy sums, leaf tables and averaged leaf values must agree to float32
rounding. At the end the exact best responses of both implementations must
agree too.

Games, all at 10 BB: the push-fold tree; the limp tree; the limp tree with
a synthetic 3x3 matrix leaf; and the subgame after a limp with that leaf,
started from non-uniform random ranges. Schemes: vanilla simultaneous, linear
alternating, CFR+ and alternating DCFR.

Both run in float64. In float32 they agree to ~1e-4 on the trees without a
leaf, but on the leaf tree the summation order of the leaf contraction alone
moves leaf regrets by ~1e-4 relative after one iteration, and regret matching
amplifies that: once a near-zero regret changes sign the averages diverge by
1e-1 within a dozen iterations. In float64 they agree to ~1e-11, which is the
statement worth checking.
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from oppex.cfr import equity, exploit, solver, tree  # noqa: E402
from oppex.cfr.cards import MASK, N_DEALS, N_HANDS  # noqa: E402
from oppex.cfr.general import pscfr  # noqa: E402
from oppex.cfr.general.from_poker import from_poker, to_poker_tables  # noqa: E402
from oppex.cfr.subgame import build_subgame  # noqa: E402
from oppex.envs import betting  # noqa: E402
from oppex.envs.betting import CALL  # noqa: E402

SCHEMES = {
  "vanilla": dict(),
  "linear alt": dict(alternating=True, linear=True),
  "CFR+": dict(alternating=True, plus=True, linear=True),
  "DCFR alt": dict(alternating=True, discount=solver.DCFR),
}
N_T = 3


def synthetic_leaves(tr, ev, seed=0):
  # Checkdown value plus a random masked term per continuation pair.
  rng = np.random.default_rng(seed)
  out = []
  for j in tr.leaves:
    stake = float(tr.terminals[j].betting.committed[1])
    noise = rng.normal(0.0, 0.3 * stake, (N_T, N_T, N_HANDS, N_HANDS))
    out.append(stake * np.asarray(ev)[None, None] + noise * np.asarray(MASK)[None, None])
  return np.stack(out).astype(np.float64)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  scale = max(np.abs(b).max(initial=0.0), 1e-12)
  return np.abs(a - b).max(initial=0.0) / scale


def compare(name, tr, ev, values, root_reach, iters, scheme):
  # Runs both solvers for `iters` iterations; returns the worst disagreement.
  n_t = values.shape[1] if values is not None else 1
  legal = solver.legal_mask(tr)
  ptab = solver.init_tables(tr, N_HANDS, jnp.float64, n_transforms=n_t)
  game = from_poker(tr, ev, values, root_reach, dtype=np.float64)
  gtab = pscfr.init_tables(game, np.float64)
  jv = None if values is None else jnp.asarray(values)
  jr = None if root_reach is None else tuple(jnp.asarray(r) for r in root_reach)
  worst = 0.0
  for _ in range(iters):
    ptab = solver.cfr_step(tr, ptab, legal, ev, N_HANDS, leaf_values=jv, root_reach=jr,
                           **scheme)
    gtab = pscfr.cfr_step(game, gtab, **scheme)
    got = to_poker_tables(game, gtab, tr)
    want = (ptab.regret, ptab.strategy_sum, ptab.leaf_regret, ptab.leaf_strategy_sum,
            ptab.leaf_cfv_sum, ptab.leaf_mass_sum)
    for g, w in zip(got, want):
      worst = max(worst, rel(g, w))

  # Best responses to the average profile, from both implementations.
  avg, tau = solver.average_strategy(ptab, legal), solver.average_leaf_strategy(ptab)
  gsig, gtau = pscfr.average_strategy(game, gtab)
  br_err = 0.0
  for p in (0, 1):
    v_p, _ = exploit.best_response(tr, avg, p, ev, N_HANDS, jnp.float64, leaf_values=jv,
                                   leaf_tau=tau if values is not None else None,
                                   root_reach=jr)
    v_g, _ = pscfr.best_response(game, gsig, gtau, p)
    br_err = max(br_err, abs(v_g / N_DEALS - v_p) / max(abs(v_p), 1e-9))
  print(f"  {name:28s} {iters:>4d} it   tables {worst:.1e}   BR {br_err:.1e}", flush=True)
  return max(worst, br_err)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=30)
  ap.add_argument("--boards", type=int, default=200_000)
  ap.add_argument("--tol", type=float, default=1e-8)
  args = ap.parse_args()

  ev = jnp.asarray(equity.preflop_ev_matrix(n_boards=args.boards), jnp.float64)
  rules = betting.make_rules(20.0, 1.0, 2.0, 2.0, 0)
  shove = tree.build_preflop_tree(rules, allow_limp=False)
  limp = tree.build_preflop_tree(rules, allow_limp=True)
  leafy = tree.build_preflop_tree(rules, allow_limp=True, matrix_leaves=True)
  values = synthetic_leaves(leafy, ev)
  n = next(i for i, nd in enumerate(leafy.nodes) if nd.path == (CALL,))
  sub = build_subgame(rules, leafy.nodes[n].betting, leafy.nodes[n].path, None)
  rng = np.random.default_rng(1)
  ranges = tuple(rng.random(N_HANDS) for _ in range(2))

  games = [
    ("push-fold", shove, None, None),
    ("limp", limp, None, None),
    ("limp + 3x3 leaf", leafy, values, None),
    ("limp subgame + leaf, ranges", sub, values, ranges),
  ]
  worst = 0.0
  t0 = time.time()
  for scheme_name, scheme in SCHEMES.items():
    print(f"{scheme_name}:")
    for name, tr, v, rr in games:
      worst = max(worst, compare(name, tr, ev, v, rr, args.iters, scheme))
  print(f"\nworst relative disagreement {worst:.1e} ({time.time() - t0:.0f}s)")
  assert worst < args.tol, "the general PSCFR does not reproduce the poker solver"
  print("All checks passed.")


if __name__ == "__main__":
  main()
