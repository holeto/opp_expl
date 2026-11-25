"""Safe depth-limited re-solving on Goofspiel 3 (descending prizes), end to end.

  uv run python scripts/resolve_goofspiel.py [--iters 10 30 100 300 1000 3000]

**Leaf checks** (depth-1 cut at the root, exact normal-form leaves):

1. The leaf matrices equal the values computed straight from the rules.
2. Degenerate check: the depth-1 root solve has value 0, and its root
   strategy, with the behaviour its leaf mixtures induce below, has full-game
   NashConv near 0.
3. Against player 0's root mix (1/3, 0, 2/3) playing High below, player 1's
   best response in the depth-limited game is worth 1/3 with exact leaves (its
   full-game value) and 0 with the single highest-card continuation, which is
   why the latter accepts that exploitable mix as an equilibrium.

**Re-solving.** The compound strategy starts uniform. Turns are walked top-
down; every public state whose joint reach under the compound strategy
exceeds ``--threshold`` is re-solved once per player ``p``. Each re-solve cuts
the public state one turn deep with exact leaves. At the root it is a plain
solve; below the root it uses the Resolve gadget, whose chance node weighs
each root history by chance times ``p``'s compound reach, and whose
alternative values are the opponent's per-unit leaf values averaged over
``p``'s solve of the parent. ``p``'s average strategy at the subgame root goes
into the compound strategy, and the opponent's leaf values are kept for ``p``'s
re-solves of the children. Unresolved states keep the uniform start.

For each CFR budget per solve this reports the compound strategy's exact
full-game NashConv, its largest deviation from always playing the highest
card on infosets with reach above the threshold, the re-solved public states,
and the gadget's safety: the opponent's best-response value per unit at each
root infoset, minus its alternative value. The walk runs with the general
PSCFR and again with vanilla CFR on the same gadget games.
"""

from __future__ import annotations

import argparse
import time

import jax

# The shared CFR update rule is JAX, which truncates to float32 without x64.
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from oppex.cfr import solver  # noqa: E402
from oppex.cfr.general import pscfr  # noqa: E402
from oppex.envs import build_env  # noqa: E402
from oppex.tree import best_response_values, extract_game_tree  # noqa: E402
from oppex.tree.history_game import (  # noqa: E402
  DECISION, LEAF, from_layered, reach_probabilities, strategy_dicts,
)
from oppex.tree.public import (  # noqa: E402
  leaf_rows_by_key, public_partition, rows_from_strategies, strategies_from_rows,
  to_public_game,
)
from oppex.tree.resolve import gadget_safety, solve_game  # noqa: E402
from oppex.tree.subgame import (  # noqa: E402
  behaviour_from_tau, cut_subgame, fixed_profile_leaves, normal_form_leaves, resolve_gadget,
)

PRIZES = (3, 2, 1)
SCHEME = dict(alternating=True, discount=solver.DCFR)


def highest_card(full):
  # Always play the highest card left: the pure Nash equilibrium.
  out = ({}, {})
  for n in range(full.n_nodes):
    for p in (0, 1):
      if full.kind[n] == DECISION and full.acting[n][p]:
        m = np.zeros(full.n_actions)
        m[np.flatnonzero(full.legal[n][p]).max()] = 1.0
        out[p][full.keys[n][p]] = m
  return out


def nash_conv(layered, strategies):
  bv = best_response_values(layered, strategies[0], strategies[1])
  return bv.v0_p0_br + bv.v1_p1_br


# ── Leaf checks ──────────────────────────────────────────────────────────────


def check_matrices(full, sub, conts):
  # Every leaf entry against the rules: the turn-1 prize from the history's
  # cards, then each continuation's card at turn 2, turn 3 forced.
  sgn = lambda a, b: (a > b) - (a < b)
  worst = 0.0
  for n in range(sub.n_nodes):
    if sub.kind[n] != LEAF:
      continue
    a, b = (x + 1 for x in full.action[sub.ref[n]])
    hands = [{1, 2, 3} - {a}, {1, 2, 3} - {b}]
    for t0, s0 in enumerate(conts[(0, sub.keys[n][0])]):
      for t1, s1 in enumerate(conts[(1, sub.keys[n][1])]):
        c = [s0[sub.keys[n][0]] + 1, s1[sub.keys[n][1]] + 1]
        last = [(hands[p] - {c[p]}).pop() for p in (0, 1)]
        want = PRIZES[0] * sgn(a, b) + PRIZES[1] * sgn(*c) + PRIZES[2] * sgn(*last)
        worst = max(worst, abs(sub.leaf_values[n][t0, t1] - want))
  print(f"  1. leaf matrices vs the rules: largest error {worst:.1e}")
  assert worst < 1e-12


def leaf_checks(layered, full, iters):
  print("Leaf checks (depth-1 cut at the root)")
  sub = cut_subgame(full, full.roots, 1)
  conts = normal_form_leaves(full, sub)
  check_matrices(full, sub, conts)

  # 2. Degenerate check: solve, then extend the root strategy below the cut.
  pg = to_public_game(sub)
  tab = pscfr.solve(pg, iters, **SCHEME)
  sig, tau = pscfr.average_strategy(pg, tab)
  v0, _ = pscfr.root_value(pg, sig, tau)
  strat = strategies_from_rows(pg, sig)
  rows = leaf_rows_by_key(pg)
  tau_by_key = {k: tau[r] for k, r in rows.items()}
  for p in (0, 1):
    strat[p].update(behaviour_from_tau(full, conts, tau_by_key, p))
  nc_full = nash_conv(layered, strat)
  nc_dl = pscfr.nash_conv(pg, sig, tau)
  print(f"  2. depth-1 root solve, {iters} iterations: value {v0:+.2e}, depth-limited "
        f"NashConv {nc_dl:.2e}, full-game NashConv of the extended strategy {nc_full:.2e}")
  assert abs(v0) < 1e-3 and nc_full < 1e-3

  # 3. The exploitable root mix, against exact and single-continuation leaves.
  root_key = full.keys[full.roots[0]][0]
  mix = np.array([1 / 3, 0.0, 2 / 3])
  high = highest_card(full)
  for name, exact in (("exact leaves", True), ("T=1 highest-card leaves", False)):
    s = cut_subgame(full, full.roots, 1)
    if exact:
      c = normal_form_leaves(full, s)
      # Player 0 plays High below: the continuation that plays the higher card.
      tau0 = {k: np.eye(len(v))[max(range(len(v)), key=lambda t: v[t][k[1]])]
              for k, v in c.items() if k[0] == 0}
    else:
      fixed_profile_leaves(full, s, high)
      tau0 = {}
    g = to_public_game(s)
    sig, tau = rows_from_strategies(g, ({root_key: mix}, {}), tau0)
    br1 = pscfr.best_response(g, sig, tau, 1)[0]
    print(f"  3. {name:24s}: player 1's best response to (1/3, 0, 2/3) is worth {br1:+.4f}")
    assert abs(br1 - (1 / 3 if exact else 0.0)) < 1e-9


# ── Re-solving ───────────────────────────────────────────────────────────────


def resolve_walk(layered, full, iters, threshold, use_pscfr, root_iters=None):
  """One top-down re-solving walk; returns the compound strategy and a log.

  ``root_iters`` gives the root solve its own budget; the log then also holds
  the root solve's full strategy (root average plus the behaviour its leaf
  mixtures induce), the strategy the gadgets are meant to be safe against.
  """
  states, _ = public_partition(full)
  depth = np.zeros(full.n_nodes, int)
  for n in range(full.n_nodes):
    if full.parent[n] >= 0:
      depth[n] = depth[full.parent[n]] + 1
  compound = strategy_dicts(full)
  store = ({}, {})
  cr = full.chance_reach()
  log = dict(resolved=[], safety=[])

  for d in range(depth.max() + 1):
    level = [c for c in states if depth[c[0]] == d and full.kind[c[0]] == DECISION]
    # Reaches under the compound strategy as it stands before this turn.
    reach = reach_probabilities(full, compound)
    for cls in level:
      joint = sum(cr[h] * reach[h, 0] * reach[h, 1] for h in cls)
      if d > 0 and joint <= threshold:
        continue
      log["resolved"].append((d, len(cls), joint))
      for p in (0, 1):
        o = 1 - p
        sub = cut_subgame(full, cls, 1)
        normal_form_leaves(full, sub)
        if d == 0:
          game = sub
        else:
          pi = {r: cr[sub.ref[r]] * reach[sub.ref[r], p] for r in sub.roots}
          alt = {sub.keys[r][o]: store[p][sub.keys[r][o]] for r in sub.roots if pi[r] > 0}
          game = resolve_gadget(sub, p, pi, alt)
        n_it = iters if d > 0 or root_iters is None else root_iters
        strat, values, leaf_tau = solve_game(game, n_it, use_pscfr, SCHEME)
        if d == 0:
          conts = normal_form_leaves(full, sub)
          ext = log.setdefault("root_strategy", ({}, {}))
          ext[p].update(strat[p])
          ext[p].update(behaviour_from_tau(full, conts, leaf_tau, p))
        if d > 0:
          log["safety"].append((d, len(cls), gadget_safety(sub, p, pi, alt, strat, leaf_tau)))
        root_keys = {full.keys[h][p] for h in cls}
        compound[p].update({k: v for k, v in strat[p].items() if k in root_keys})
        store[p].update({k: v for (q, k), v in values.items() if q == o})
  return compound, log


def deviation(full, compound, threshold):
  # Largest probability off the highest card, over infosets reached above the threshold.
  high = highest_card(full)
  reach = reach_probabilities(full, compound)
  joint = full.chance_reach() * reach[:, 0] * reach[:, 1]
  mass = ({}, {})
  for n in range(full.n_nodes):
    for p in (0, 1):
      if full.kind[n] == DECISION and full.acting[n][p]:
        k = full.keys[n][p]
        mass[p][k] = mass[p].get(k, 0.0) + joint[n]
  return max((1.0 - compound[p][k] @ high[p][k]
              for p in (0, 1) for k, m in mass[p].items() if m > threshold), default=0.0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, nargs="+", default=[10, 30, 100, 300, 1000, 3000])
  ap.add_argument("--threshold", type=float, default=0.05)
  ap.add_argument("--leaf-iters", type=int, default=2000)
  ap.add_argument("--root-iters", type=int, nargs="+", default=[1, 2, 3, 5])
  ap.add_argument("--stress-iters", type=int, default=1000)
  args = ap.parse_args()

  env = build_env({"name": "goofspiel", "n_cards": 3, "prize_order": "descending"})
  layered = extract_game_tree(env)
  full = from_layered(layered)
  print(f"Goofspiel 3 descending: {full.n_nodes} histories; highest-card NashConv "
        f"{nash_conv(layered, highest_card(full)):.1e}\n")
  leaf_checks(layered, full, args.leaf_iters)

  print(f"\nRe-solving walk (reach threshold {args.threshold}, alternating DCFR per solve)")
  print(f"  {'iters':>6} {'solver':>8} {'NashConv':>10} {'deviation':>10} "
        f"{'resolved (turn, histories, reach)':<44} {'max safety':>11} {'time':>6}")
  rows = []
  for n_it in args.iters:
    out = {}
    for name, use in (("PSCFR", True), ("vanilla", False)):
      t0 = time.time()
      comp, log = resolve_walk(layered, full, n_it, args.threshold, use)
      nc = nash_conv(layered, comp)
      dev = deviation(full, comp, args.threshold)
      res = " ".join(f"({d},{h},{j:.2f})" for d, h, j in log["resolved"])
      saf = max((v for _, _, v in log["safety"]), default=float("nan"))
      print(f"  {n_it:>6} {name:>8} {nc:>10.2e} {dev:>10.2e} {res:<44} {saf:>+11.2e} "
            f"{time.time() - t0:>5.1f}s", flush=True)
      out[name] = (comp, nc, dev, saf)
    diff = max(np.abs(out["PSCFR"][0][p][k] - out["vanilla"][0][p][k]).max()
               for p in (0, 1) for k in out["PSCFR"][0][p])
    rows.append((n_it, out["PSCFR"][1], out["PSCFR"][2], out["PSCFR"][3], diff))
    print(f"  {'':>6} {'':>8} PSCFR and vanilla compound strategies differ by {diff:.1e}")

  ncs = [r[1] for r in rows]
  print(f"\nNashConv by budget: " + ", ".join(f"{r[0]}: {r[1]:.2e}" for r in rows))
  assert ncs[-1] < 1e-2, "the compound strategy did not converge"
  late = [r[1] for r in rows if r[0] >= 300]
  assert all(b <= a + 1e-6 for a, b in zip(late, late[1:])), "NashConv rose past 300 iterations"

  stress(layered, full, args)
  print("All checks passed.")


def stress(layered, full, args):
  # With a well-converged root solve almost nothing below the root is
  # reached, so the gadget only meets single-history states. Cutting the root
  # solve short keeps the root mixed and puts the gadget on multi-history
  # states. Safety then means: the compound strategy is no more exploitable
  # than the root solve's own full strategy, and every gadget whose parent
  # solve converged keeps the opponent at its alternative values.
  print(f"\nGadget stress test: root solve cut short, {args.stress_iters} iterations "
        f"below the root (PSCFR)")
  print(f"  {'root it':>7} {'root-solve NC':>14} {'compound NC':>12} {'resolved':>9} "
        f"{'max safety, turn 1':>19} {'max safety, turn 2':>19}")
  for ri in args.root_iters:
    comp, log = resolve_walk(layered, full, args.stress_iters, args.threshold, True, ri)
    nc_root, nc = nash_conv(layered, log["root_strategy"]), nash_conv(layered, comp)
    multi = sum(h > 1 for _, h, _ in log["resolved"])
    s1 = max((v for d, _, v in log["safety"] if d == 1), default=float("nan"))
    s2 = max((v for d, _, v in log["safety"] if d == 2), default=float("nan"))
    print(f"  {ri:>7} {nc_root:>14.4e} {nc:>12.4e} {multi:>3} multi {s1:>+19.2e} {s2:>+19.2e}")
    assert nc <= nc_root + 1e-6, "re-solving made the strategy more exploitable"
    assert not s2 > 1e-3, "a gadget below a converged parent let the opponent gain"
  print("  Turn-1 alternative values are averaged over the few root iterations: the\n"
        "  opponent's value under their own unconverged continuation, not a best response,\n"
        "  so no re-solved strategy can hold them, and the turn-1 numbers are not bounded.")


if __name__ == "__main__":
  main()
