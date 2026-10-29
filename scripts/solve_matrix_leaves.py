"""Solve the limp/shove preflop game with a matrix-valued leaf, and check it by LP.

  uv run python scripts/policy_gradient/sample_leaf_values.py checkpoints/qmmd_hunl_560k
  uv run python scripts/solve_matrix_leaves.py \\
      --leaf checkpoints/qmmd_hunl_560k/leaf_values/leaf_1-1_params_step560000_n10000_f5_k10_s0.npz \\
      --iters 2000 --dcfr --alternating

The game is the ``--limp`` sanity-check tree of ``solve_pushfold.py``: SB folds,
limps or shoves; after a limp BB checks or shoves; after a BB shove SB folds or
calls. It is built at the blueprint's 200 BB, because the network's input size
depends on stack depth. The limp-check line, where preflop closes without an
all-in, is a depth-limit leaf. There both players simultaneously choose one of
four continuations of the QMMD blueprint (itself, or biased toward folding,
calling or raising), and the leaf pays the sampled ``V[t0, t1, a, b]``.

The validation ladder:

1. **Degenerate leaf.** With one continuation worth ``stake * EV``, the leaf is
   exactly the old checkdown terminal, so PSCFR on both trees must agree to
   float rounding. This isolates the leaf plumbing from everything else.
2. **Invariants** on the real leaf: zero-sum at every node (which also checks
   player 1's sign and transpose at the leaf) and joint reach conservation.
3. **Exact reference.** The sequence-form LP of the same tree and leaf tensor,
   compared on value, action frequencies, continuation frequencies, and the
   sandwich ``BR0 >= v_LP >= -BR1``.
"""

from __future__ import annotations

import argparse
import time

import jax.numpy as jnp
import numpy as np

from oppex.cfr import equity, exploit, lp_reference, seqform, solver, tree
from oppex.cfr.cards import N_HANDS
from oppex.cfr.leaves import Leaves
from oppex.envs import betting
from oppex.policy_gradient.blueprint import TRANSFORM_NAMES
from oppex.policy_gradient.leaf_values import load_leaf

_ATOM = {0: "FOLD", 1: "CALL", 2: "ALL_IN"}


def run_cfr(tr, ev, leaf_values, iters, scheme, n_t, log=None):
  # Returns the final tables; prints exploitability at a few checkpoints.
  legal = solver.legal_mask(tr)
  tab = solver.init_tables(tr, N_HANDS, n_transforms=n_t)
  for it in range(1, iters + 1):
    tab = solver.cfr_step(tr, tab, legal, ev, N_HANDS, leaf_values=leaf_values, **scheme)
    if log is not None and (it in (1, 10, 100) or it % max(iters // 8, 1) == 0):
      avg, tau = solver.average_strategy(tab, legal), solver.average_leaf_strategy(tab)
      e, _, _ = exploit.exploitability(tr, avg, ev, N_HANDS, log, leaf_values=leaf_values,
                                       leaf_tau=tau)
      v0, _ = solver.root_value(tr, avg, ev, N_HANDS, leaves=_leaves(tr, leaf_values, tau))
      print(f"{it:>7}  {e:>19.3e}  {float(v0) / log:>10.5f}", flush=True)
  return tab


def _leaves(tr, values, tau):
  return Leaves(values, tau) if tr.leaves else None


def degenerate_check(rules, ev, iters, scheme):
  """PSCFR on the checkdown tree vs the same tree with a one-continuation leaf."""
  plain = tree.build_preflop_tree(rules, allow_limp=True)
  leafy = tree.build_preflop_tree(rules, allow_limp=True, matrix_leaves=True)
  stake = [t.stake for t in plain.terminals if t.kind == tree.CHECKDOWN_T]
  values = jnp.stack([s * ev for s in stake])[:, None, None]      # (n_leaves, 1, 1, H, H)

  tab_p = run_cfr(plain, ev, None, iters, scheme, 1)
  tab_l = run_cfr(leafy, ev, values, iters, scheme, 1)
  legal = solver.legal_mask(plain)
  avg_p, avg_l = solver.average_strategy(tab_p, legal), solver.average_strategy(tab_l, legal)
  v_p, _ = solver.root_value(plain, avg_p, ev, N_HANDS)
  v_l, _ = solver.root_value(leafy, avg_l, ev, N_HANDS,
                             leaves=Leaves(values, solver.average_leaf_strategy(tab_l)))
  d_sig = float(jnp.abs(avg_p - avg_l).max())
  d_val = abs(float(v_p) - float(v_l))
  print(f"  degenerate leaf vs checkdown after {iters} iterations: "
        f"value {d_val:.2e} chips, strategy {d_sig:.2e}")
  assert d_val < 1e-4 and d_sig < 1e-4, "a one-continuation leaf must equal the checkdown"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--leaf", required=True, help="npz from sample_leaf_values.py")
  ap.add_argument("--iters", type=int, default=2000)
  ap.add_argument("--boards", type=int, default=200_000)
  ap.add_argument("--alternating", action="store_true")
  ap.add_argument("--plus", action="store_true")
  ap.add_argument("--linear", action="store_true")
  ap.add_argument("--dcfr", nargs="?", const="1.5,0,2", metavar="A,B,G")
  ap.add_argument("--degenerate-iters", type=int, default=200)
  ap.add_argument("--no-lp", action="store_true")
  ap.add_argument("--lp-method", default="highs-ipm",
                  help="interior point: ~80 s on the 10 BB limp LP, against >17 min for simplex")
  args = ap.parse_args()

  discount = None
  if args.dcfr is not None:
    discount = solver.Discount(*(float(x) for x in args.dcfr.split(",")))
  scheme = dict(alternating=args.alternating, plus=args.plus, linear=args.linear,
                discount=discount)

  # The blueprint's game: 1/2 blinds, 200 BB.
  big_blind, small_blind, stack = 2.0, 1.0, 400.0
  ev = equity.preflop_ev_matrix(n_boards=args.boards)
  rules = betting.make_rules(stack, small_blind, big_blind, 2.0, 0)
  tr = tree.build_preflop_tree(rules, allow_limp=True, matrix_leaves=True)
  leaf = load_leaf(args.leaf)
  (leaf_t,) = tr.leaves
  if tuple(leaf["path"].tolist()) != tr.terminals[leaf_t].path:
    raise ValueError(f"leaf file was sampled at {leaf['path'].tolist()}, "
                     f"the tree's leaf is at {list(tr.terminals[leaf_t].path)}")
  values = jnp.asarray(leaf["values"])[None]
  n_t = values.shape[1]
  print(f"Tree (200 BB, limp, matrix leaf at {list(tr.terminals[leaf_t].path)}): "
        f"{len(tr.nodes)} decisions, {len(tr.terminals)} terminals, {n_t} continuations")
  print(tree.format_tree(tr))

  # ── 1. Degenerate leaf ─────────────────────────────────────────────────────
  print("\nDegenerate check")
  degenerate_check(rules, ev, args.degenerate_iters, scheme)

  # ── 2. PSCFR with the sampled leaf ─────────────────────────────────────────
  print(f"\n{'iter':>7}  {'exploitability(bb)':>19}  {'value(bb)':>10}")
  t0 = time.time()
  tab = run_cfr(tr, ev, values, args.iters, scheme, n_t, log=big_blind)
  dt = time.time() - t0
  print(f"  {args.iters} iterations in {dt:.1f}s ({dt / args.iters * 1000:.1f} ms/iter)")

  legal = solver.legal_mask(tr)
  avg, tau = solver.average_strategy(tab, legal), solver.average_leaf_strategy(tab)
  leaves = Leaves(values, tau)
  zs = solver.check_zero_sum(tr, avg, ev, N_HANDS, leaves=leaves)
  rc = solver.check_reach_conservation(tr, avg, N_HANDS)
  e, br0, br1 = exploit.exploitability(tr, avg, ev, N_HANDS, big_blind,
                                       leaf_values=values, leaf_tau=tau)
  v0, v1 = solver.root_value(tr, avg, ev, N_HANDS, leaves=leaves)
  print(f"\nInvariants: zero-sum {zs:.2e} | joint reach {rc:.2e} "
        f"| v0+v1 {float(v0 + v1):+.2e}")

  # ── 3. Exact reference: sequence-form LP ───────────────────────────────────
  if args.no_lp:
    return
  print(f"\nSolving the same game by sequence-form LP ({args.lp_method})…", flush=True)
  t0 = time.time()
  lp_v, r0, r1, sf = seqform.solve(tr, np.asarray(ev), N_HANDS, method=args.lp_method,
                                   leaf_values=np.asarray(values))
  print(f"  solved in {time.time() - t0:.1f}s "
        f"(SB sequences {sf.n_seq[0]}, BB sequences {sf.n_seq[1]})")
  lp_sig = (seqform.behaviour_strategy(tr, sf, r0, 0),
            seqform.behaviour_strategy(tr, sf, r1, 1))

  print(f"\n{'':26s}{'CFR':>12s}{'LP (exact)':>12s}{'diff':>10s}")
  print(f"  {'value to SB (bb)':24s}{float(v0) / big_blind:>12.5f}"
        f"{lp_v / big_blind:>12.5f}{abs(float(v0) - lp_v) / big_blind:>10.2e}")
  for i, node in enumerate(tr.nodes):
    sig = lp_sig[node.player][i]
    for a in (a for a, c in enumerate(node.child) if c is not None):
      cfr_f, lp_f = float(np.asarray(avg[i])[:, a].mean()), float(sig[:, a].mean())
      print(f"  n{i} P{node.player} {_ATOM.get(a, f'BET_{a - 3}'):<18s}{cfr_f:>12.5f}"
            f"{lp_f:>12.5f}{abs(cfr_f - lp_f):>10.2e}")
  for p in range(2):
    sig = lp_sig[p][("leaf", 0)]
    for t, name in enumerate(TRANSFORM_NAMES[:n_t]):
      cfr_f, lp_f = float(np.asarray(tau[p, 0])[:, t].mean()), float(sig[:, t].mean())
      print(f"  leaf P{p} {name:<17s}{cfr_f:>12.5f}{lp_f:>12.5f}{abs(cfr_f - lp_f):>10.2e}")

  lp_reference.check_against_cfr(lp_v, br0 * big_blind, br1 * big_blind)
  print(f"\n  LP sandwich BR0 >= v >= -BR1 holds "
        f"({br0 * big_blind:+.5f} >= {lp_v:+.5f} >= {-br1 * big_blind:+.5f})")
  print(f"  exploitability: {e:.3e} bb/hand")
  diff = abs(float(v0) - lp_v) / big_blind
  assert diff < 1e-3, f"CFR and LP values differ by {diff:.2e} bb"
  print(f"  CFR value matches the LP to {diff:.2e} bb")


if __name__ == "__main__":
  main()
