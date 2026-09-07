"""Scaling benchmark for the public-state CFR solver.

  uv run python scripts/bench_cfr.py [--boards 20000] [--iters 20]

Walks a grid of (stack depth, bet bins) preflop trees and reports what each
costs: tree size, trace+compile time, and steady-state seconds per iteration.

**These trees are not solutions.** With bet atoms every line that closes preflop
without an all-in is valued as a checkdown, which overpays seeing a flop; see
``build_preflop_tree``. The point here is the cost curve, not the strategy. It
shows: at 50 BB with 3 bins the SB min-raises 69.7% of hands and never shoves,
which is the checkdown artifact, not a strategy anyone should read.

Measured on this CPU box, DCFR + alternating, 1326 hands, no limp, before and
after ``terminal_cfv`` was batched into one matmul per terminal kind:

  ====  ====  =====  =====  ===============  ==============
    bb  bins  nodes  terms  compile(s) b/a   s/iter b/a
  ====  ====  =====  =====  ===============  ==============
    10     0      2      3    0.37 -> 0.46   0.0008 -> 0.0009
    50     3     66     99    4.71 -> 3.69   0.0150 -> 0.0076
    50     4    204    306   13.33 -> 16.54  0.0400 -> 0.0225
   100     3    120    180    7.97 -> 7.47   0.0286 -> 0.0111
   100     4    472    708   32.68 -> 73.49  0.1199 -> 0.0509
  ====  ====  =====  =====  ===============  ==============

Both costs stay linear and terminals track nodes at ~1.5x, so the grid collapses
to two constants — and each is a wall on the way up:

* **~155 ms of compile per node** (was ~69 ms before batching, which added graph
  at the leaves). The tree is unrolled into one jaxpr by Python recursion at
  trace time, so nodes *are* the graph. Extrapolated: ~26 min at 10k nodes.
  This is the binding constraint and it is structural — fixing it means giving
  up the recursion for a level-indexed array tree driven by ``lax.scan``, so
  nodes at the same depth become one batched step. Until then, node count is the
  number to watch when choosing an abstraction.
* **~0.07 ms per terminal per iteration** (was ~0.16 ms), and no longer rising
  with tree size. This is the part batching fixed, worth 1.8-2.6x on trees with
  99 terminals or more. It is *not* the 13.8x an isolated matmul microbenchmark
  predicts; see ``terminal_cfv`` for why, and for the compile-time price.
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
