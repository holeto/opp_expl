"""Safe depth-limited re-solving on Leduc Hold'em, against an independent Nash.

  uv run python scripts/resolve_leduc.py [--iters 10 30 100 300 1000 3000]

**The reference.** ``src/oppex/envs/leduc_nash.pkl`` is a Nash equilibrium of
OpenSpiel's Leduc, computed by a sequence-form LP outside this code base. It
holds ``(v0, v1, X, S)``: ``X[depth][player]`` lists infoset tensors (seat 2 |
private card 6 | public card, none + 6 | 8 action slots × fold/call/raise) and
``S`` the matching strategies over (unused, fold, call, raise). Cards are
indexed, so a rank is ``card // 2``; rows that differ only in suits are one
infoset of our rank-level env and are merged by averaging (they agree to
4e-7, so this is the suit-symmetric equilibrium).

**Checks** before re-solving:

0. The reference in our env: its value is ``v0`` and its exact NashConv is
   near 0, which validates the env port and the infoset mapping at once.
1. Exact leaves. The root solve stops where round 2 starts, after the public
   card. A player's continuations at a leaf infoset are all their reduced
   pure strategies of the rest of the round (5 for player 0, 9 for player 1).
   Mapped into the leaf game by Kuhn's construction (a pure continuation gets
   the product of the reference's probabilities of its actions), the
   reference makes every leaf pay the full game's value below it.
2. The reference's round-1 strategy with those mixtures is an equilibrium of
   the depth-limited game.
3. Degenerate check: the depth-limited root solve has the game's value, and
   its round-1 strategy with the behaviour its leaf mixtures induce below has
   full-game NashConv near 0.
4. Vanilla CFR takes the same step as the general PSCFR from the same tables,
   on the root game and on a round-2 gadget game. Free-running solves of the
   two drift apart by roundoff amplification (see ``oppex.tree.resolve``), so
   the sweep reports both solvers' numbers side by side instead.

**Re-solving.** One root solve of round 1, with normal-form leaves. Its
average gives the compound strategy in round 1, and the behaviour its leaf
mixtures induce in round 2 (the *root-solve strategy*). Then every round-2
public state (betting line × public card) whose joint reach exceeds
``--threshold`` is re-solved once per player ``p``, to the end of the game,
inside the Resolve gadget: chance weighs each root history by chance times
``p``'s round-1 reach, and the opponent's alternative values are its average
per-unit leaf values from the root solve. ``p``'s re-solved strategy replaces
the root-solve strategy in that public state.

Per CFR budget (the same for the root solve and each re-solve) this reports
the exact full-game NashConv of the root-solve strategy and of the compound,
the gadgets' largest safety violation (the opponent's best-response value per
unit at a root infoset minus its alternative value), how far the compound is
from the reference (reach-weighted total variation per round, and the chips
it loses to the reference over both seats), and runs the walk with the
general PSCFR and with vanilla CFR. Round-2 equilibria are not unique given
the ranges, so the round-2 distance need not go to 0.

**Control.** Round 2 re-solved without a gadget from the reference's own
ranges: a subgame equilibrium, but exploitable in the full game.

**Diagnostic.** With the root solve fixed, the re-solve budget is raised, with
averaged cfvs and with CBVs (the opponent's best-response values against the
root-solve strategy) as alternative values.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax

# The shared CFR update rule is JAX, which truncates to float32 without x64.
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from oppex.cfr import solver  # noqa: E402
from oppex.cfr.general import pscfr  # noqa: E402
from oppex.envs import build_env  # noqa: E402
from oppex.tree import best_response_values, extract_game_tree  # noqa: E402
from oppex.tree.history_game import (  # noqa: E402
  DECISION, LEAF, TERMINAL, expected_value, from_layered, reach_probabilities,
)
from oppex.tree.public import (  # noqa: E402
  leaf_rows_by_key, public_partition, rows_from_strategies, strategies_from_rows,
  to_public_game,
)
from oppex.tree.resolve import (  # noqa: E402
  gadget_safety, opponent_values, solve_game, step_disagreement,
)
from oppex.tree.subgame import (  # noqa: E402
  behaviour_from_tau, cut_subgame, normal_form_leaves, resolve_gadget,
)

NASH_PKL = Path(__file__).resolve().parents[1] / "src/oppex/envs/leduc_nash.pkl"
SCHEME = dict(alternating=True, discount=solver.DCFR)


def nash_conv(layered, strategies):
  bv = best_response_values(layered, strategies[0], strategies[1])
  return bv.v0_p0_br + bv.v1_p1_br


# ── The game and the reference ───────────────────────────────────────────────


class Leduc:
  """The full history game, each node's env state, and the round-2 starts."""

  def __init__(self):
    self.layered = extract_game_tree(build_env({"name": "leduc"}))
    self.full = full = from_layered(self.layered)
    self.state = [None] * full.n_nodes
    for n in range(full.n_nodes):
      L, r = full.ref[n]
      sb = self.layered.layers[L].state_batch
      if sb is not None and full.kind[n] != TERMINAL:
        self.state[n] = jax.tree.map(lambda x: np.asarray(x[r]), sb)
    rnd = [-1 if s is None else int(s.round_idx) for s in self.state]
    # The first decision of round 2, right after the public card.
    self.start2 = np.array([full.kind[n] == DECISION and rnd[n] == 1 and rnd[full.parent[n]] == 0
                            for n in range(full.n_nodes)])
    self.round = np.array(rnd)

  def describe(self, n, p):
    # (player, own rank, public rank or -1, action history) of node n.
    s = self.state[n]
    return (p, int(s.private_cards[p]), int(s.public_card),
            tuple(int(a) for a in s.action_history[: int(s.step)]))


def load_reference(game: Leduc):
  """The reference as ``(strategies, value to player 0)``, and the largest suit asymmetry."""
  v0, _, X, S = pickle.load(open(NASH_PKL, "rb"))
  rows: dict = {}
  for x_d, s_d in zip(X, S):
    for p in (0, 1):
      for x, s in zip(x_d[p], s_d[p]):
        card = int(np.flatnonzero(x[2:8])[0])
        pub = int(np.flatnonzero(x[8:15])[0]) - 1
        hist = tuple(int(np.flatnonzero(slot)[0]) for slot in x[15:39].reshape(8, 3) if slot.any())
        d = (p, card // 2, pub // 2 if pub >= 0 else -1, hist)
        rows.setdefault(d, []).append(np.asarray(s[1:], float))
  full = game.full
  out, seen = ({}, {}), set()
  for n in range(full.n_nodes):
    for p in (0, 1):
      if full.kind[n] == DECISION and full.acting[n][p]:
        d = game.describe(n, p)
        seen.add(d)
        s = np.mean(rows[d], 0)
        out[p][full.keys[n][p]] = s / s.sum()
  assert seen == set(rows), "the reference and the env have different infosets"
  asym = max(np.abs(np.array(r) - np.mean(r, 0)).max() for r in rows.values())
  return out, v0, asym, sum(len(r) for r in rows.values())


def kuhn_tau(conts, strategies):
  # Realization-equivalent mixtures: a pure continuation gets the product of
  # the behaviour probabilities of the actions it specifies.
  out = {}
  for (p, k), strats in conts.items():
    t = np.array([np.prod([strategies[p][key][a] for key, a in s.items()]) for s in strats])
    assert abs(t.sum() - 1) < 1e-9
    out[(p, k)] = t
  return out


# ── Checks ───────────────────────────────────────────────────────────────────


def checks(game: Leduc, ref, v_ref, iters):
  full = game.full
  sub = cut_subgame(full, full.roots, None, stop=lambda n: game.start2[n])
  conts = normal_form_leaves(full, sub)
  leaves = [n for n in range(sub.n_nodes) if sub.kind[n] == LEAF]
  n0 = {len(v) for (p, _), v in conts.items() if p == 0}
  n1 = {len(v) for (p, _), v in conts.items() if p == 1}
  print(f"Depth-limited root game: {sub.n_nodes} histories, {len(leaves)} leaves, "
        f"{len(conts)} leaf infosets, continuations {n0} x {n1}")

  # 1. Every leaf, under the reference's continuation mixtures.
  tau = kuhn_tau(conts, ref)
  v_full = expected_value(full, ref)
  err = max(abs(tau[(0, sub.keys[n][0])] @ sub.leaf_values[n] @ tau[(1, sub.keys[n][1])]
                - v_full[sub.ref[n]]) for n in leaves)
  print(f"  1. leaf values under the reference vs the full game below: largest error {err:.1e}")
  assert err < 1e-12

  # 2. The reference restricted to round 1 solves the depth-limited game.
  pg = to_public_game(sub)
  sig, tau_rows = rows_from_strategies(pg, ref, tau)
  nc_ref = pscfr.nash_conv(pg, sig, tau_rows)
  print(f"  2. reference in the depth-limited game: value {pscfr.root_value(pg, sig, tau_rows)[0]:+.7f}, "
        f"NashConv {nc_ref:.1e}")
  assert nc_ref < 1e-6

  # 3. Solve it, then extend the root strategy below the cut.
  tab = pscfr.solve(pg, iters, **SCHEME)
  sig, tau_rows = pscfr.average_strategy(pg, tab)
  strat = strategies_from_rows(pg, sig)
  rows = leaf_rows_by_key(pg)
  tau_by_key = {k: tau_rows[r] for k, r in rows.items()}
  for p in (0, 1):
    strat[p].update(behaviour_from_tau(full, conts, tau_by_key, p))
  v0 = pscfr.root_value(pg, sig, tau_rows)[0]
  nc = nash_conv(game.layered, strat)
  print(f"  3. depth-limited root solve, {iters} iterations: value {v0:+.7f} (reference "
        f"{v_ref:+.7f}), depth-limited NashConv {pscfr.nash_conv(pg, sig, tau_rows):.1e}, "
        f"full-game NashConv of the extended strategy {nc:.1e}")
  assert abs(v0 - v_ref) < 1e-3 and nc < 1e-3

  # 4. The two solvers' steps, on the root game and on the gadget of the
  # round-2 state with the most histories, re-solved for player 1.
  reach = reach_probabilities(full, strat)
  cr = full.chance_reach()
  states, _ = public_partition(full)
  cls = max((c for c in states if game.start2[c[0]]), key=len)
  r2 = cut_subgame(full, cls, None)
  pi = {r: cr[r2.ref[r]] * reach[r2.ref[r], 1] for r in r2.roots}
  alt = {r2.keys[r][0]: 0.0 for r in r2.roots}
  gadget = resolve_gadget(r2, 1, pi, alt)
  worst = max(step_disagreement(g, 100, sc) for g in (sub, gadget)
              for sc in (SCHEME, dict(alternating=True, plus=True, linear=True), {}))
  print(f"  4. vanilla CFR's step vs PSCFR's, 100 steps of DCFR, CFR+ and vanilla CFR on the "
        f"root game and a {len(cls)}-history gadget: largest relative error {worst:.1e}")
  assert worst < 1e-10


# ── Re-solving ───────────────────────────────────────────────────────────────


def resolve_walk(game: Leduc, iters, threshold, use_pscfr, root_iters=None, alt_kind="avg"):
  """Root solve, then gadget re-solves of round 2; returns (compound, root-solve strategy, log).

  ``root_iters`` gives the root solve its own budget. ``alt_kind`` picks the
  alternative values: ``"avg"`` the opponent's averaged leaf cfvs per unit,
  ``"cbv"`` its best-response values against the root-solve strategy.
  """
  full = game.full
  root = cut_subgame(full, full.roots, None, stop=lambda n: game.start2[n])
  conts = normal_form_leaves(full, root)
  strat, values, leaf_tau = solve_game(root, root_iters or iters, use_pscfr, SCHEME)
  root_strategy = ({}, {})
  for p in (0, 1):
    root_strategy[p].update(strat[p])
    root_strategy[p].update(behaviour_from_tau(full, conts, leaf_tau, p))
  compound = (dict(root_strategy[0]), dict(root_strategy[1]))

  # Round-1 reaches are final once the root solve is done.
  reach = reach_probabilities(full, compound)
  cr = full.chance_reach()
  states, _ = public_partition(full)
  log = dict(resolved=[], skipped=[], safety=[])
  for cls in (c for c in states if game.start2[c[0]]):
    joint = sum(cr[h] * reach[h, 0] * reach[h, 1] for h in cls)
    if joint <= threshold:
      log["skipped"].append((len(cls), joint))
      continue
    log["resolved"].append((len(cls), joint))
    for p in (0, 1):
      o = 1 - p
      sub = cut_subgame(full, cls, None)
      pi = {r: cr[sub.ref[r]] * reach[sub.ref[r], p] for r in sub.roots}
      if alt_kind == "avg":
        alt = {sub.keys[r][o]: values[(o, sub.keys[r][o])] for r in sub.roots if pi[r] > 0}
      else:
        alt = opponent_values(sub, p, pi, root_strategy, None)
      s, _, lt = solve_game(resolve_gadget(sub, p, pi, alt), iters, use_pscfr, SCHEME)
      log["safety"].append(gadget_safety(sub, p, pi, alt, s, lt))
      # The re-solve covers the rest of the game below this public state.
      own = {sub.keys[n][p] for n in range(sub.n_nodes)
             if sub.kind[n] == DECISION and sub.acting[n][p]}
      compound[p].update({k: v for k, v in s[p].items() if k in own})
  return compound, root_strategy, log


def compare(game: Leduc, strat, ref, v_ref):
  """Distance of ``strat`` from the reference.

  Returns the reach-weighted mean total variation per round (weights: the
  reference's joint reach of each infoset), and the chips ``strat`` loses
  to the reference summed over both seats, which is at most its NashConv.
  """
  full = game.full
  r = reach_probabilities(full, ref)
  joint = full.chance_reach() * r[:, 0] * r[:, 1]
  w, tv = [{}, {}], [{}, {}]
  for n in range(full.n_nodes):
    for p in (0, 1):
      if full.kind[n] == DECISION and full.acting[n][p]:
        k, rd = full.keys[n][p], game.round[n]
        w[rd][(p, k)] = w[rd].get((p, k), 0.0) + joint[n]
        tv[rd][(p, k)] = 0.5 * np.abs(strat[p][k] - ref[p][k]).sum()
  mean_tv = [sum(w[i][k] * tv[i][k] for k in w[i]) / sum(w[i].values()) for i in (0, 1)]
  loss = ((v_ref - expected_value(full, (strat[0], ref[1]))[0])
          + (expected_value(full, (ref[0], strat[1]))[0] - v_ref))
  return mean_tv, loss


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, nargs="+", default=[10, 30, 100, 300, 1000, 3000])
  ap.add_argument("--threshold", type=float, default=0.0)
  ap.add_argument("--check-iters", type=int, default=3000)
  ap.add_argument("--control-iters", type=int, default=10000)
  ap.add_argument("--diag-root-iters", type=int, default=1000)
  ap.add_argument("--diag-iters", type=int, nargs="+", default=[1000, 3000, 10000])
  args = ap.parse_args()

  game = Leduc()
  ref, v_ref, asym, n_rows = load_reference(game)
  v_mine = expected_value(game.full, ref)[0]
  nc_ref = nash_conv(game.layered, ref)
  n_inf = sum(len(d) for d in ref)
  print(f"Leduc: {game.full.n_nodes} histories, {n_inf} acting infosets")
  print(f"  0. reference: {n_rows} card-level rows merged into {n_inf} infosets (largest suit "
        f"asymmetry {asym:.1e}); value in our env {v_mine:+.7f} (file {v_ref:+.7f}), "
        f"exact NashConv {nc_ref:.1e}")
  assert abs(v_mine - v_ref) < 1e-6 and nc_ref < 1e-6
  checks(game, ref, v_ref, args.check_iters)

  print(f"\nRe-solving (root solve of round 1, gadget re-solves of round 2 above reach "
        f"{args.threshold}, alternating DCFR)")
  print(f"  {'iters':>6} {'solver':>8} {'root-solve NC':>14} {'compound NC':>12} "
        f"{'resolved':>9} {'max safety':>11} {'TV round 1':>11} {'TV round 2':>11} "
        f"{'loss vs ref':>12} {'time':>6}")
  rows = []
  for n_it in args.iters:
    out = {}
    for name, use in (("PSCFR", True), ("vanilla", False)):
      t0 = time.time()
      comp, root_s, log = resolve_walk(game, n_it, args.threshold, use)
      nc_root, nc = nash_conv(game.layered, root_s), nash_conv(game.layered, comp)
      (tv1, tv2), loss = compare(game, comp, ref, v_ref)
      saf = max(log["safety"], default=float("nan"))
      print(f"  {n_it:>6} {name:>8} {nc_root:>14.3e} {nc:>12.3e} "
            f"{len(log['resolved']):>4}/{len(log['resolved']) + len(log['skipped']):<4} "
            f"{saf:>+11.2e} {tv1:>11.2e} {tv2:>11.2e} {loss:>12.3e} {time.time() - t0:>5.1f}s",
            flush=True)
      out[name] = (comp, nc_root, nc, saf, tv1, tv2, loss)
    rows.append((n_it, *out["PSCFR"][1:]))

  print("\nCompound NashConv by budget (PSCFR): "
        + ", ".join(f"{r[0]}: {r[2]:.2e}" for r in rows))
  assert rows[-1][2] < 1e-2, "the compound strategy did not converge"
  unsafe_control(game, ref, v_ref, args.control_iters)
  diagnose(game, args)
  print("All checks passed.")


def unsafe_control(game: Leduc, ref, v_ref, iters):
  # Unsafe re-solving: each round-2 public state solved with both players
  # free from the reference's own ranges, no gadget. The reference's round 2
  # is one equilibrium of each such subgame; the solve finds another, which
  # is exploitable through round 1.
  full = game.full
  reach = reach_probabilities(full, ref)
  cr = full.chance_reach()
  states, _ = public_partition(full)
  hyb = (dict(ref[0]), dict(ref[1]))
  for cls in (c for c in states if game.start2[c[0]]):
    sub = cut_subgame(full, cls, None, own_reach={h: reach[h, :2] for h in cls}, chance=cr)
    s, _, _ = solve_game(sub, iters, False, SCHEME)
    for p in (0, 1):
      hyb[p].update(s[p])
  (_, tv2), _ = compare(game, hyb, ref, v_ref)
  nc = nash_conv(game.layered, hyb)
  print(f"\nControl: the reference's round 1, round 2 re-solved without a gadget from its "
        f"ranges ({iters} iterations, vanilla CFR): NashConv {nc:.3e}, TV round 2 {tv2:.2e}")
  assert nc > 1e-2, "unsafe re-solving was expected to be exploitable"


def diagnose(game: Leduc, args):
  # Is a compound worse than its root solve short of re-solve iterations, or
  # held back by the alternative values? Fix the root solve, raise the
  # re-solve budget, and compare averaged cfvs with CBVs.
  print(f"\nAlternative values: root solve fixed at {args.diag_root_iters} iterations (PSCFR)")
  print(f"  {'re-solve it':>11} {'alt values':>10} {'root-solve NC':>14} {'compound NC':>12} "
        f"{'max safety':>11} {'time':>6}")
  for n_it in args.diag_iters:
    for kind in ("avg", "cbv"):
      t0 = time.time()
      comp, root_s, log = resolve_walk(game, n_it, args.threshold, True,
                                       args.diag_root_iters, kind)
      nc_root, nc = nash_conv(game.layered, root_s), nash_conv(game.layered, comp)
      print(f"  {n_it:>11} {kind:>10} {nc_root:>14.3e} {nc:>12.3e} "
            f"{max(log['safety']):>+11.2e} {time.time() - t0:>5.1f}s", flush=True)


if __name__ == "__main__":
  main()
