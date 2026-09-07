"""Scaling benchmark for the public-state CFR solver.

  uv run python scripts/bench_cfr.py [--boards 20000] [--iters 20]

Walks a grid of (stack depth, bet bins) preflop trees and reports what each
costs: tree size, trace+compile time, and steady-state seconds per iteration.

**These trees are not solutions.** With bet atoms every line that closes preflop
without an all-in is valued as a checkdown, which overpays seeing a flop; see
``build_preflop_tree``. The point here is the cost curve, not the strategy. It
shows: at 50 BB with 3 bins the SB min-raises 69.7% of hands and never shoves,
which is the checkdown artifact, not a strategy anyone should read.

Measured on this CPU box, DCFR + alternating, 1326 hands, no limp:

  ====  ====  =====  =====  ==========  ======
    bb  bins  nodes  terms  compile(s)  s/iter
  ====  ====  =====  =====  ==========  ======
    10     0      2      3        0.37  0.0008
    50     3     66     99        4.71  0.0150
    50     4    204    306       13.33  0.0400
   100     3    120    180        7.97  0.0286
   100     4    472    708       32.68  0.1199
  ====  ====  =====  =====  ==========  ======

Both costs are linear and terminals track nodes at ~1.5x, so the whole grid
collapses to two constants — and each is a wall on the way up:

* **~69 ms of compile per node.** The tree is unrolled into one jaxpr by Python
  recursion at trace time, so nodes *are* the graph. Extrapolated: ~11 min at
  10k nodes, ~2 h at 100k. This is the binding constraint, and it is structural
  — fixing it means giving up the recursion for a level-indexed array tree
  driven by ``lax.scan``, so nodes at the same depth become one batched step.
* **~0.16 ms per terminal per iteration**, all of it 1326x1326 matvecs in
  ``terminal_cfv``. Cheaper to fix: every fold terminal multiplies by the same
  ``MASK`` and every showdown by the same ``EV``, so stacking each kind's
  ``r_opp`` into a matrix turns N matvecs into one matmul. Measured directly at
  N=708: 89.6 ms of GEMVs (27.8 GFLOP/s) versus 6.5 ms for the equivalent GEMM
  (384.5 GFLOP/s) — 13.8x, and it also cuts that case's compile from 100 s to
  0.03 s, so it takes a bite out of the first wall too.
"""

from __future__ import annotations

import argparse
import time

import jax

from oppex.cfr import equity, solver, tree
from oppex.cfr.cards import N_HANDS
from oppex.envs import betting


def build(bb: float, k: int, limp: bool, max_depth: int = 24):
  rules = betting.make_rules(bb * 2.0, 1.0, 2.0, 2.0, k)
  return tree.build_preflop_tree(
    rules, allow_limp=limp, allow_bets=k > 0, max_depth=max_depth
  )


def time_case(tr, ev, iters: int, **kw):
  legal = solver.legal_mask(tr)
  tab = solver.init_tables(tr, N_HANDS)
  t0 = time.perf_counter()
  tab = solver.cfr_step(tr, tab, legal, ev, N_HANDS, **kw)
  jax.block_until_ready(tab)
  compile_s = time.perf_counter() - t0

  t0 = time.perf_counter()
  for _ in range(iters):
    tab = solver.cfr_step(tr, tab, legal, ev, N_HANDS, **kw)
  jax.block_until_ready(tab)
  return compile_s, (time.perf_counter() - t0) / iters


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--boards", type=int, default=20_000)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--limp", action="store_true")
  ap.add_argument("--bb", type=float, nargs="+", default=[10, 20, 50, 100])
  ap.add_argument("--bins", type=int, nargs="+", default=[0, 1, 2, 3, 4])
  args = ap.parse_args()

  ev = equity.preflop_ev_matrix(n_boards=args.boards, exact=False, progress=False)
  kw = dict(alternating=True, discount=solver.DCFR)

  print(f"DCFR + alternating, {N_HANDS} hands, limp={args.limp}, "
        f"{args.iters} timed iterations\n")
  print(f"{'bb':>5} {'bins':>5} {'nodes':>7} {'terms':>7} {'atoms':>6} "
        f"{'compile(s)':>11} {'s/iter':>9} {'ms/term':>9}")
  for bb in args.bb:
    for k in args.bins:
      tr = build(bb, k, args.limp)
      compile_s, per = time_case(tr, ev, args.iters, **kw)
      print(f"{bb:>5g} {k:>5} {len(tr.nodes):>7} {len(tr.terminals):>7} "
            f"{tr.n_actions:>6} {compile_s:>11.2f} {per:>9.4f} "
            f"{per / len(tr.terminals) * 1e3:>9.3f}", flush=True)


if __name__ == "__main__":
  main()
