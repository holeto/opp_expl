"""Depth-limited RNR with safe re-solving on Leduc Hold'em, against a fixed model.

  uv run python scripts/rnr_leduc.py --model chump|perturbed [--p 0 0.1 0.5 1]

**RNR.** A chance node picks the opponent model with probability ``p`` or a
free opponent with ``1 - p``; we do not see which (``oppex.tree.rnr``). We
play one seat and the model the other, and each seat is solved separately.

**The walk** follows ``resolve_leduc.py``. A root solve of the RNR game cut at
the start of round 2 (exact normal-form leaves; known-copy leaves value each
of our continuations against the model) gives the *parent*: its round 1 and
the behaviour its leaf mixtures induce below. Every round-2 public state
reached in either branch is then re-solved to the end of the game in
``rnr_gadget``: the known copy of all its histories (weights
``p·c·π_i·π_fix``) and the Resolve gadget over the unknown copy, with root
weights ``(1 - p)·c·π_i`` (``--weights B``) or that times the parent's free
opponent reach (``A``), and the free opponent's CBVs against the parent
(``--alt cbv``) or its averaged leaf cfvs (``avg``) as alternative values.
Only PSCFR is used.

**Metrics,** per seat and their mean: value against the model, exploitability
of our strategy alone (the seat's game value minus our worst-case value; not
the NashConv of us and the model, which would count the model's own
exploitability), and the RNR objective ``F = p·vs_model + (1 - p)·worst``.
The ground truth is the full-game RNR solved by PSCFR without a depth limit.

**Checks** (asserted):

1. Frozen model rows stay bit-identical through every solve (``pscfr.solve``).
2. The RNR games are weighted and labelled correctly: under random strategies
   (Kuhn mixtures at the leaves) the full and the depth-limited RNR game are
   worth ``p·u(ours, model) + (1 - p)·u(ours, free)`` in the original game.
3. ``p = 0``: the compound equals the existing safe re-solve (``resolve_walk``)
   bit for bit, and it is near a Nash equilibrium.
4. ``p = 1``: our value against the model equals the exact best response's,
   for the depth-limited compound and for the full-game RNR.
5. Variant B with CBVs never lowers the objective: ``F(compound) ≥ F(parent)``
   up to the solves' accuracy. Other variants report the same difference.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from resolve_leduc import SCHEME, Leduc, kuhn_tau, load_reference, nash_conv, resolve_walk

from oppex.envs.leduc_holdem import CALL, FOLD
from oppex.tree.history_game import DECISION, expected_value, reach_probabilities
from oppex.tree.public import public_partition
from oppex.tree.resolve import gadget_safety, opponent_values, solve_game
from oppex.tree.rnr import KNOWN, is_known, rnr_gadget, rnr_game, seat_metrics
from oppex.tree.subgame import behaviour_from_tau, cut_subgame, normal_form_leaves

MAIN = ("B", "cbv")
VARIANTS = (("B", "cbv"), ("B", "avg"), ("A", "cbv"))


# ── Opponent models ──────────────────────────────────────────────────────────


def chump(game: Leduc) -> dict:
  """Always call (check when nobody has bet), for both seats."""
  full, out = game.full, {}
  for n in range(full.n_nodes):
    for p in (0, 1):
      if full.kind[n] == DECISION and full.acting[n][p]:
        out[full.keys[n][p]] = np.eye(full.n_actions)[CALL]
  return out


def perturbed(game: Leduc, ref, eps: float) -> dict:
  """The reference Nash, with ``eps`` of its folds turned into calls when facing a round-2 bet."""
  full, out = game.full, {}
  for n in range(full.n_nodes):
    for p in (0, 1):
      if full.kind[n] == DECISION and full.acting[n][p]:
        k = full.keys[n][p]
        s = np.array(ref[p][k], float)
        if game.round[n] == 1 and full.legal[n][p][FOLD]:
          s[CALL] += eps * s[FOLD]
          s[FOLD] *= 1.0 - eps
        out[k] = s
  return out


# ── RNR walk ─────────────────────────────────────────────────────────────────


def _seat(i, ours, other):
  # A strategy pair with ours in seat i.
  return (ours, other) if i == 0 else (other, ours)


def rnr_walk(game: Leduc, i, p, model, root_iters, iters, weights="B", alt_kind="cbv"):
  """Root solve, then RNR re-solves of round 2 for seat ``i``; returns (compound, parent, log)."""
  full, o = game.full, 1 - i
  root = cut_subgame(full, full.roots, None, stop=lambda n: game.start2[n])
  conts = normal_form_leaves(full, root)
  G, fixed = rnr_game(root, i, p, model, full, conts)
  strat, values, leaf_tau = solve_game(G, root_iters, True, SCHEME, fixed)
  parent = dict(strat[i])
  parent.update(behaviour_from_tau(full, conts, leaf_tau, i))
  free = None
  if p < 1:
    free = {k: v for k, v in strat[o].items() if not is_known(k)}
    free.update(behaviour_from_tau(full, conts, leaf_tau, o))
  compound = dict(parent)

  # Round-1 reaches are final once the root solve is done.
  r_model = reach_probabilities(full, _seat(i, parent, model))
  r_free = None if free is None else reach_probabilities(full, _seat(i, parent, free))
  cr = full.chance_reach()
  states, _ = public_partition(full)
  log = dict(resolved=0, skipped=0, exceed=[])
  for cls in (c for c in states if game.start2[c[0]]):
    sub = cut_subgame(full, cls, None)
    pi = {r: cr[sub.ref[r]] * r_model[sub.ref[r], i] for r in sub.roots}
    wk = {r: p * pi[r] * r_model[sub.ref[r], o] for r in sub.roots}
    if weights == "B":
      wu = {r: (1.0 - p) * pi[r] for r in sub.roots}
    else:
      wu = {r: (0.0 if free is None else (1.0 - p) * pi[r] * r_free[sub.ref[r], o])
            for r in sub.roots}
    if not any(w > 0 for w in (*wk.values(), *wu.values())):
      log["skipped"] += 1
      continue
    log["resolved"] += 1
    alt = {}
    if any(w > 0 for w in wu.values()):
      if alt_kind == "avg":
        alt = {sub.keys[r][o]: values[(o, sub.keys[r][o])] for r in sub.roots if wu[r] > 0}
      else:
        alt = opponent_values(sub, i, pi, _seat(i, parent, {}), None)
    g, fx = rnr_gadget(sub, i, wk, wu, alt, model)
    s, _, lt = solve_game(g, iters, True, SCHEME, fx)
    if alt:
      # How far the free opponent's CBVs exceed the alternative values: RNR
      # may trade this for gains against the model.
      log["exceed"].append(gadget_safety(sub, i, pi, alt, s, lt))
    own = {sub.keys[n][i] for n in range(sub.n_nodes)
           if sub.kind[n] == DECISION and sub.acting[n][i]}
    compound.update({k: v for k, v in s[i].items() if k in own})
  return compound, parent, log


def full_rnr(game: Leduc, i, p, model, iters):
  """Our seat-``i`` strategy in the full-game RNR, no depth limit."""
  G, fixed = rnr_game(game.full, i, p, model)
  strat, _, _ = solve_game(G, iters, True, SCHEME, fixed)
  return strat[i]


# ── Checks ───────────────────────────────────────────────────────────────────


def check_values(game: Leduc, model, p=0.3, seed=0):
  # Check 2: the RNR games under random strategies, against the original game.
  rng = np.random.default_rng(seed)
  full = game.full
  rand = ({}, {})
  for n in range(full.n_nodes):
    for q in (0, 1):
      if full.kind[n] == DECISION and full.acting[n][q]:
        m = np.asarray(full.legal[n][q], bool)
        s = np.zeros(full.n_actions)
        s[m] = rng.dirichlet(np.ones(m.sum()))
        rand[q][full.keys[n][q]] = s
  root = cut_subgame(full, full.roots, None, stop=lambda n: game.start2[n])
  conts = normal_form_leaves(full, root)
  tau = kuhn_tau(conts, rand)
  worst = 0.0
  for i in (0, 1):
    o = 1 - i
    want = (p * expected_value(full, _seat(i, rand[i], model))[0]
            + (1 - p) * expected_value(full, _seat(i, rand[i], rand[o]))[0])
    opp = dict(rand[o])
    opp.update({(KNOWN, k): model[k] for k in rand[o]})
    strategies = _seat(i, rand[i], opp)
    G, _ = rnr_game(full, i, p, model)
    R, _ = rnr_game(root, i, p, model, full, conts)
    leaf_tau = ({k: t for (q, k), t in tau.items() if q == 0},
                {k: t for (q, k), t in tau.items() if q == 1})
    got = (expected_value(G, strategies)[0], expected_value(R, strategies, leaf_tau)[0])
    worst = max(worst, *(abs(x - want) for x in got))
  print(f"  2. RNR games (p = {p}) under random strategies vs p·u(model) + (1-p)·u(free): "
        f"largest error {worst:.1e} (full game and depth-limited, both seats)")
  assert worst < 1e-12


def metrics(game, strat_by_seat, model, p, v_ref):
  # Per-seat metrics and their mean.
  per = [seat_metrics(game.layered, game.full, strat_by_seat[i], model, i, p, v_ref)
         for i in (0, 1)]
  return per, {k: 0.5 * (per[0][k] + per[1][k]) for k in per[0]}


def fmt(m):
  return (f"{m['vs_model']:>+10.5f} {m['worst']:>+10.5f} {m['expl']:>10.3e} "
          f"{m['F']:>+10.5f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", choices=["chump", "perturbed"], default="chump")
  ap.add_argument("--perturb", type=float, default=0.5)
  ap.add_argument("--p", type=float, nargs="+", default=[0.0, 1.0, 0.1, 0.5])
  ap.add_argument("--root-iters", type=int, default=1000)
  ap.add_argument("--iters", type=int, default=3000)
  ap.add_argument("--truth-iters", type=int, default=3000)
  ap.add_argument("--br-tol", type=float, default=1e-4)
  ap.add_argument("--monotone-tol", type=float, default=1e-3)
  args = ap.parse_args()

  game = Leduc()
  ref, v_ref, _, _ = load_reference(game)
  model = chump(game) if args.model == "chump" else perturbed(game, ref, args.perturb)
  print(f"Leduc RNR against the {args.model} model"
        + (f" (eps {args.perturb})" if args.model == "perturbed" else "")
        + f"; game value to player 0 {v_ref:+.7f}")
  per, mean = metrics(game, (ref[0], ref[1]), model, 0.0, v_ref)
  print(f"  model: best response wins {per[0]['br_vs_model']:+.5f} (seat 0), "
        f"{per[1]['br_vs_model']:+.5f} (seat 1), mean {mean['br_vs_model']:+.5f}; "
        f"the reference Nash wins {mean['vs_model']:+.5f}")
  print("  1. frozen model rows: asserted bit-identical inside every solve")
  check_values(game, model)

  print(f"\nRNR walk: root solve {args.root_iters} iterations, re-solves {args.iters}, "
        f"full-game RNR {args.truth_iters}; alternating DCFR, PSCFR")
  print(f"  {'p':>4} {'variant':>9} {'seat':>5} {'vs model':>10} {'worst':>10} "
        f"{'expl':>10} {'F':>10} {'F - F_par':>10} {'F_true - F':>10} {'resolved':>8} "
        f"{'max exceed':>10} {'time':>6}")
  for p in args.p:
    t0 = time.time()
    truth = [full_rnr(game, i, p, model, args.truth_iters) for i in (0, 1)]
    t_per, t_mean = metrics(game, truth, model, p, v_ref)
    t_truth = time.time() - t0
    for seat, m in (("0", t_per[0]), ("1", t_per[1]), ("mean", t_mean)):
      print(f"  {p:>4} {'full RNR':>9} {seat:>5} {fmt(m)} {'':>10} {'':>10} {'':>8} {'':>10} "
            + (f"{t_truth:>5.0f}s" if seat == "mean" else ""), flush=True)
    variants = VARIANTS if 0 < p < 1 else (MAIN,)
    for weights, alt_kind in variants:
      t0 = time.time()
      runs = [rnr_walk(game, i, p, model, args.root_iters, args.iters, weights, alt_kind)
              for i in (0, 1)]
      comp = [runs[i][0] for i in (0, 1)]
      c_per, c_mean = metrics(game, comp, model, p, v_ref)
      p_per, p_mean = metrics(game, [runs[i][1] for i in (0, 1)], model, p, v_ref)
      dt = time.time() - t0
      name = f"{weights}+{alt_kind}"
      for seat, m, mp, mt, log in (("0", c_per[0], p_per[0], t_per[0], runs[0][2]),
                                   ("1", c_per[1], p_per[1], t_per[1], runs[1][2]),
                                   ("mean", c_mean, p_mean, t_mean, None)):
        extra = (f"{log['resolved']:>3}/{log['resolved'] + log['skipped']:<4} "
                 f"{max(log['exceed'], default=float('nan')):>+10.2e}" if log
                 else f"{'':>8} {'':>10}")
        print(f"  {p:>4} {name:>9} {seat:>5} {fmt(m)} {m['F'] - mp['F']:>+10.2e} "
              f"{mt['F'] - m['F']:>+10.2e} {extra} "
              + (f"{dt:>5.0f}s" if seat == "mean" else ""), flush=True)

      if (weights, alt_kind) == MAIN:
        # Check 5: variant B with CBVs does not lower the objective.
        for i in (0, 1):
          assert c_per[i]["F"] >= p_per[i]["F"] - args.monotone_tol, \
            f"re-solving lowered F in seat {i}"
      if p == 0:
        # Check 3: the existing safe re-solve, bit for bit, near a Nash.
        base, _, _ = resolve_walk(game, args.iters, 0.0, True, args.root_iters, alt_kind)
        same = all(set(comp[i]) <= set(base[i])
                   and all(np.array_equal(comp[i][k], base[i][k]) for k in comp[i])
                   for i in (0, 1))
        nc = nash_conv(game.layered, (comp[0], comp[1]))
        print(f"       3. p = 0: compound equals resolve_walk bit for bit: {same}; "
              f"exploitability {c_mean['expl']:.2e} per seat (NashConv of our two seats "
              f"{nc:.2e})")
        assert same and nc < 1e-2
      if p == 1:
        # Check 4: the best response to the model.
        gaps = [abs(m["vs_model"] - m["br_vs_model"]) for m in (*c_per, *t_per)]
        print(f"       4. p = 1: |value - best-response value| against the model: compound "
              f"{gaps[0]:.1e} / {gaps[1]:.1e}, full RNR {gaps[2]:.1e} / {gaps[3]:.1e} "
              f"(seat 0 / seat 1)")
        assert max(gaps) < args.br_tol
  print("All checks passed.")


if __name__ == "__main__":
  main()
