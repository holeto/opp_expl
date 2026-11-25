"""Public-state CFR on a subgame with both players free, from given root ranges.

  uv run python scripts/validate_free_subgame.py [--iters 2000]

The game is the 10 BB limp tree with a matrix leaf where preflop closes without
an all-in. Its leaf tensor is synthetic: the checkdown ``stake * EV`` plus a
random mask-folded term per continuation pair, so both players have three
continuations that genuinely differ. Nothing here needs the blueprint.

Checks:

1. **Re-rooting is exact.** For a fixed profile, the subgame below the BB's
   decision after a limp, started from the full tree's reaches there, has the
   full tree's cfvs at its root.
2. **Zero-sum** holds with non-uniform root ranges.
3. **Re-solving from equilibrium ranges** with both players free reproduces the
   full equilibrium's value of that subgame, and its exploitability within the
   subgame goes to 0.
4. **Averaged leaf values** equal the weighted sums of the per-iteration leaf
   cfvs and masses, recomputed by hand, under linear CFR and DCFR.
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from oppex.cfr import equity, exploit, solver, tree
from oppex.cfr.cards import MASK, N_DEALS, N_HANDS
from oppex.cfr.leaves import Leaves
from oppex.cfr.subgame import build_subgame
from oppex.envs import betting
from oppex.envs.betting import CALL

N_T = 3


def synthetic_leaves(tr, ev, seed=0):
  # (n_leaves, T, T, H, H): checkdown value plus a random masked term per
  # continuation pair. Not antisymmetric, like a real leaf tensor.
  rng = np.random.default_rng(seed)
  out = []
  for j in tr.leaves:
    stake = float(tr.terminals[j].betting.committed[1])
    noise = rng.normal(0.0, 0.3 * stake, (N_T, N_T, N_HANDS, N_HANDS)).astype(np.float32)
    out.append(stake * np.asarray(ev)[None, None] + noise * np.asarray(MASK)[None, None])
  return jnp.asarray(np.stack(out))


def solve(tr, ev, values, iters, root_reach=None, scheme=None):
  scheme = scheme or dict(alternating=True, discount=solver.DCFR)
  legal = solver.legal_mask(tr)
  tab = solver.init_tables(tr, N_HANDS, n_transforms=N_T)
  for _ in range(iters):
    tab = solver.cfr_step(tr, tab, legal, ev, N_HANDS, leaf_values=values,
                          root_reach=root_reach, **scheme)
  return tab, solver.average_strategy(tab, legal), solver.average_leaf_strategy(tab)


def subtree_maps(full, sub):
  # Subgame node -> full-tree node and subgame leaf -> full-tree leaf, by path.
  by_path = {n.path: i for i, n in enumerate(full.nodes)}
  leaf_by_path = {full.terminals[j].path: l for l, j in enumerate(full.leaves)}
  nodes = np.array([by_path[n.path] for n in sub.nodes])
  leaves = np.array([leaf_by_path[sub.terminals[j].path] for j in sub.leaves])
  return nodes, leaves


def check_averages(tr, ev, values):
  # Rebuild the leaf-value averages from per-iteration traversals.
  legal = solver.legal_mask(tr)
  ones = jnp.ones(N_HANDS, jnp.float32)
  for name, scheme in (("linear", dict(linear=True)), ("DCFR", dict(discount=solver.DCFR))):
    tab = solver.init_tables(tr, N_HANDS, n_transforms=N_T)
    cfv = np.zeros((2, len(tr.leaves), N_HANDS))
    mass = np.zeros_like(cfv)
    for t in range(1, 6):
      sigma = solver.regret_matching_plus(tab.regret, legal)
      tau = solver.leaf_strategy(tab)
      tv = solver._traverse(tr, sigma, ev, ones, ones, Leaves(values, tau))
      v_t = np.asarray((tv.leaf_q * tau).sum(-1))
      m_t = np.asarray(tv.leaf_reach[::-1] @ MASK)
      if "discount" in scheme:
        d = (t / (t + 1.0)) ** solver.DCFR.gamma
        cfv, mass = (cfv + v_t) * d, (mass + m_t) * d
      else:
        cfv, mass = cfv + t * v_t, mass + t * m_t
      tab = solver.cfr_step(tr, tab, legal, ev, N_HANDS, leaf_values=values, **scheme)
    err_c = np.abs(np.asarray(tab.leaf_cfv_sum) - cfv).max() / np.abs(cfv).max()
    err_m = np.abs(np.asarray(tab.leaf_mass_sum) - mass).max() / np.abs(mass).max()
    alt = np.asarray(solver.average_leaf_cfv(tab))
    err_a = np.abs(alt * mass - cfv).max() / np.abs(cfv).max()
    print(f"  {name:6s}: cfv sum {err_c:.1e}, mass sum {err_m:.1e}, "
          f"per-unit value {err_a:.1e} (relative)")
    assert max(err_c, err_m, err_a) < 1e-4, "averaged leaf values disagree"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=2000)
  ap.add_argument("--boards", type=int, default=200_000)
  args = ap.parse_args()

  ev = equity.preflop_ev_matrix(n_boards=args.boards)
  rules = betting.make_rules(20.0, 1.0, 2.0, 2.0, 0)
  full = tree.build_preflop_tree(rules, allow_limp=True, matrix_leaves=True)
  values = synthetic_leaves(full, ev)
  print(f"10 BB limp tree: {len(full.nodes)} decisions, {len(full.leaves)} leaves, "
        f"{N_T} continuations each")

  print("\n4. Averaged leaf values")
  check_averages(full, ev, values)

  # The full-game equilibrium.
  tab, sigma, tau = solve(full, ev, values, args.iters)
  e, _, _ = exploit.exploitability(full, sigma, ev, N_HANDS, 2.0, leaf_values=values,
                                   leaf_tau=tau)
  print(f"\nFull game: {args.iters} iterations of alternating DCFR, "
        f"exploitability {e:.2e} bb/hand")

  # The subgame below the BB's decision after a limp.
  n = next(i for i, nd in enumerate(full.nodes) if nd.path == (CALL,))
  sub = build_subgame(rules, full.nodes[n].betting, full.nodes[n].path, None)
  node_map, leaf_map = subtree_maps(full, sub)
  ones = jnp.ones(N_HANDS, jnp.float32)
  tv = solver._traverse(full, sigma, ev, ones, ones, Leaves(values, tau))
  r0, r1 = tv.reach0[n], tv.reach1[n]
  print(f"Subgame at path {list(full.nodes[n].path)}: {len(sub.nodes)} decisions, "
        f"{len(sub.leaves)} leaves")

  print("\n1. Re-rooting")
  sig_sub, tau_sub = sigma[node_map], tau[:, leaf_map]
  leaves_sub = Leaves(values[leaf_map], tau_sub)
  ts = solver._traverse(sub, sig_sub, ev, r0, r1, leaves_sub)
  for p, (a, b) in enumerate(((ts.cfv0, tv.cfv0), (ts.cfv1, tv.cfv1))):
    err = float(jnp.abs(a[sub.root] - b[n]).max() / jnp.abs(b[n]).max())
    print(f"  player {p}: root cfvs vs full tree, max relative error {err:.1e}")
    assert err < 1e-5, "the re-rooted subgame does not reproduce the full tree"

  print("\n2. Zero-sum with root ranges")
  zs = solver.check_zero_sum(sub, sig_sub, ev, N_HANDS, leaves=leaves_sub,
                             root_reach=(r0, r1))
  print(f"  worst relative error {zs:.1e}")

  print("\n3. Re-solve from equilibrium ranges, both players free")
  v_full = float((r0 * tv.cfv0[n]).sum()) / N_DEALS
  _, sig_r, tau_r = solve(sub, ev, values[leaf_map], args.iters, root_reach=(r0, r1))
  v_sub, _ = solver.root_value(sub, sig_r, ev, N_HANDS, leaves=Leaves(values[leaf_map], tau_r),
                               root_reach=(r0, r1))
  br0, _ = exploit.best_response(sub, sig_r, 0, ev, N_HANDS, leaf_values=values[leaf_map],
                                 leaf_tau=tau_r, root_reach=(r0, r1))
  br1, _ = exploit.best_response(sub, sig_r, 1, ev, N_HANDS, leaf_values=values[leaf_map],
                                 leaf_tau=tau_r, root_reach=(r0, r1))
  print(f"  subgame value to SB: re-solved {float(v_sub):+.6f}, full equilibrium "
        f"{v_full:+.6f} chips (per deal)")
  print(f"  subgame NashConv of the re-solve: {br0 + br1:.2e} chips "
        f"(BR0 {br0:+.6f}, BR1 {br1:+.6f})")
  assert br0 + br1 < 1e-3, "the re-solve did not converge"
  assert abs(float(v_sub) - v_full) < 1e-3, "re-solve value differs from the full game"
  print("\nAll checks passed.")


if __name__ == "__main__":
  main()
