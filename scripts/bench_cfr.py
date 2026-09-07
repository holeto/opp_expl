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
after the traversal was batched per depth instead of recursed per node:

  ====  ====  =====  =====  ================  ==================
    bb  bins  nodes  terms  compile(s) b/a    s/iter b/a
  ====  ====  =====  =====  ================  ==================
    10     0      2      3    0.46 ->  0.78   0.0009 -> 0.0035
    50     3     66     99    3.69 ->  1.49   0.0076 -> 0.0090
    50     4    204    306   16.54 ->  1.46   0.0225 -> 0.0224
   100     3    120    180    7.47 ->  1.35   0.0111 -> 0.0127
   100     4    472    708   73.49 ->  1.55   0.0509 -> 0.0578
  ====  ====  =====  =====  ================  ==================

**Compile no longer scales with the tree.** It was ~155 ms per node, because the
recursive traversal emitted one set of ops per node at trace time; nodes *were*
the graph, and 10k nodes extrapolated to ~26 minutes. Batching by depth makes the
graph O(depth) — under ten levels even at 472 nodes — and compile flattens to
~1.5 s everywhere on this grid. That was the binding constraint on scaling up,
and it is gone.

It is not free. Per iteration the level machinery costs a fixed overhead that
small trees cannot amortise: the 2-node push-fold tree is ~4x slower per
iteration, and even 472 nodes is 13% slower. At 2000 iterations the 100 BB /
4-bin case still wins overall (175 s -> 117 s), but by ~10k iterations the
compile saving is amortised away and the two are level. The reason to keep it is
what happens *above* this grid, not on it.

What remains linear:

* **~0.07 ms per terminal per iteration**, the 1326x1326 contractions in
  ``terminal_cfv``. At 708 terminals that is already 80% of an iteration, so it
  is the next thing to look at — and unlike compile it is real arithmetic, so
  the fix is a smaller abstraction or a GPU, not a restructuring.
* **Memory**, at ``(n_nodes, 1326, n_actions)`` per table. 472 nodes with 7 atoms
  is 17 MB per table and there are two, plus the traversal's own ``branch``
  array of the same shape.
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
