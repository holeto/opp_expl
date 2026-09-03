"""Solve the preflop push-fold game and validate against the exact LP.

  uv run python scripts/solve_pushfold.py [--bb 10] [--iters 2000]

Runs vanilla CFR over the shove-fold tree, checks every invariant, and compares
the result against the exact linear-programming solution on the same equity
matrix — so a disagreement is a solver bug rather than sampling noise.
"""

from __future__ import annotations

import argparse
import time

import jax.numpy as jnp
import numpy as np

from oppex.cfr import equity, exploit, lp_reference, report, solver, tree
from oppex.cfr.cards import N_HANDS
from oppex.envs import betting


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--bb", type=float, default=10.0, help="stack depth in big blinds")
  ap.add_argument("--iters", type=int, default=2000)
  ap.add_argument("--boards", type=int, default=200_000)
  ap.add_argument("--exact-equity", action="store_true")
  ap.add_argument("--limp", action="store_true", help="allow the limp (NOT chart-comparable)")
  args = ap.parse_args()

  big_blind, small_blind = 2.0, 1.0
  stack = args.bb * big_blind

  print(f"Loading equity matrix ({'exact' if args.exact_equity else f'MC {args.boards:,}'})…")
  ev = equity.preflop_ev_matrix(
    n_boards=args.boards, exact=args.exact_equity, progress=True
  )

  rules = betting.make_rules(stack, small_blind, big_blind, 2.0, 0)
  tr = tree.build_preflop_tree(rules, allow_limp=args.limp)
  legal = solver.legal_mask(tr)
  print(f"\nTree ({args.bb:g} BB, limp={args.limp}): "
        f"{len(tr.nodes)} decisions, {len(tr.terminals)} terminals")

  # ── CFR ────────────────────────────────────────────────────────────────────
  tab = solver.init_tables(tr, N_HANDS)
  t0 = time.time()
  print(f"\n{'iter':>7}  {'exploitability(bb)':>19}  {'value(bb)':>10}")
  for it in range(1, args.iters + 1):
    tab = solver.cfr_iteration(tr, tab, legal, ev, N_HANDS)
    if it in (1, 10, 100) or it % max(args.iters // 8, 1) == 0:
      avg = solver.average_strategy(tab, legal)
      e, br0, br1 = exploit.exploitability(tr, avg, ev, N_HANDS, big_blind)
      v0, _ = solver.root_value(tr, avg, ev, N_HANDS)
      print(f"{it:>7}  {e:>19.3e}  {float(v0) / big_blind:>10.5f}")
  dt = time.time() - t0
  print(f"  {args.iters} iterations in {dt:.1f}s ({dt / args.iters * 1000:.1f} ms/iter)")

  avg = solver.average_strategy(tab, legal)

  # ── Invariants ─────────────────────────────────────────────────────────────
  zs = solver.check_zero_sum(tr, avg, ev, N_HANDS)
  rc = solver.check_reach_conservation(tr, avg, N_HANDS)
  e, br0, br1 = exploit.exploitability(tr, avg, ev, N_HANDS, big_blind)
  v0, v1 = solver.root_value(tr, avg, ev, N_HANDS)
  print(f"\nInvariants: zero-sum {zs:.2e} | joint reach {rc:.2e} "
        f"| v0+v1 {float(v0 + v1):+.2e}")

  # ── Exact LP reference ─────────────────────────────────────────────────────
  if args.limp:
    print("\nLP reference skipped: it solves the shove-fold game, not the limp tree.")
    return

  print("\nSolving the same game exactly by LP…")
  t0 = time.time()
  lp_v, lp_x, lp_y = lp_reference.solve_shove_fold(
    np.asarray(ev), small_blind=small_blind, big_blind=big_blind, stake=stack
  )
  print(f"  solved in {time.time() - t0:.1f}s")

  ALL_IN, CALL = betting.ALL_IN, betting.CALL
  cfr_x = np.asarray(avg[0][:, ALL_IN])
  cfr_y = np.asarray(avg[1][:, CALL])

  print(f"\n{'':22s}{'CFR':>12s}{'LP (exact)':>12s}{'diff':>10s}")
  print(f"  {'value to SB (bb)':20s}{float(v0) / big_blind:>12.5f}"
        f"{lp_v / big_blind:>12.5f}{abs(float(v0) - lp_v) / big_blind:>10.2e}")
  print(f"  {'SB shove freq':20s}{cfr_x.mean():>12.5f}{lp_x.mean():>12.5f}"
        f"{abs(cfr_x.mean() - lp_x.mean()):>10.2e}")
  print(f"  {'BB call freq':20s}{cfr_y.mean():>12.5f}{lp_y.mean():>12.5f}"
        f"{abs(cfr_y.mean() - lp_y.mean()):>10.2e}")

  lp_reference.check_against_cfr(lp_v, br0 * big_blind, br1 * big_blind)
  print(f"\n  LP sandwich BR0 >= v >= -BR1 holds "
        f"({br0 * big_blind:+.5f} >= {lp_v:+.5f} >= {-br1 * big_blind:+.5f})")
  print(f"  exploitability: {e:.3e} bb/hand")

  # ── Charts ─────────────────────────────────────────────────────────────────
  print()
  print(report.format_grid(cfr_x, f"SB shove range, {args.bb:g} BB (CFR, %)"))
  print()
  print(report.format_grid(cfr_y, f"BB call range, {args.bb:g} BB (CFR, %)"))
  print()
  print("Range summary (CFR):")
  for who, r in (("SB shove", cfr_x), ("BB call", cfr_y)):
    s = report.range_summary(r)
    print(f"  {who}: freq {s['frequency']:.3f}  always {s['classes_always']:>3d}  "
          f"never {s['classes_never']:>3d}  mixed {len(s['classes_mixed'])}")


if __name__ == "__main__":
  main()
