"""Validate re-solving against Slumbot (needs the Slumbot data: set ``SLUMBOT_DIR``).

  posterior  Rollouts start from the posterior over Slumbot's translation type.
             In depth-3 subgames, at leaves where Slumbot acts next, its first rollout action for
             hand ``h`` must be distributed as ``SlumbotModel.probs`` at the
             leaf (calibration over the mixed predictions; zero-probability
             actions must never occur).
  solve      ``Resolver`` with ``SlumbotModel`` and ``SlumbotRollouts`` at our
             decisions of random games against Slumbot: the CFR best-response
             gap, refusals, desyncs and time per solve.
  cards      AIVAT's boards valued by rollouts (``board_q``) against Slumbot.
             At leaves where a street closes, with Slumbot's hand fixed, the
             rollout values with the next card(s) fixed must average to the
             leaf's value over the uniform law of those cards (every turn or
             river card, a fresh sample of flops per seed), on the same seed.
             The term is zero-mean only if they do.

  chips      Rollouts scored in real chips against the env's scoring, on the same
             games. Where a game's env and real chips agree at a shared end the two
             payoffs must match; reports how much scoring differs elsewhere and
             how the refusals change.
  terminals  Fold and showdown payoffs from Slumbot's exact chips against the
             tree's. At terminals where every world's chips are the tree's × 50
             the two must match; reports how much they differ elsewhere.
  hands      The cost of per-hand leaf rollouts against Slumbot: at random decisions, how
             many hands reach each leaf, and the games per solve at ``--games`` per
             hand against the ordinary 128 per leaf.

  uv run python scripts/policy_gradient/validate_slumbot_resolving.py posterior RUN_DIR
  uv run python scripts/policy_gradient/validate_slumbot_resolving.py solve RUN_DIR --roots 20
  uv run python scripts/policy_gradient/validate_slumbot_resolving.py cards RUN_DIR --roots 15 --games 1024
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from oppex.cfr import equity
from oppex.cfr.cards import HAND_CARDS
from oppex.cfr.subgame import build_subgame
from oppex.cfr.tree import FOLD_T, MATRIX_T, SHOWDOWN_T
from oppex.policy_gradient.blueprint import N_TRANSFORMS, Blueprint
from oppex.policy_gradient.resolving import (
  Resolver,
  compatible,
  showdown_sign,
  subgame_reach,
  terminal_payoffs,
)
from oppex.policy_gradient.slumbot.model import SlumbotModel
from oppex.policy_gradient.slumbot.rollouts import SlumbotRollouts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_slumbot_model import _calibration, play_game  # noqa: E402


def _roots(rng, rules, n, prob):
  # Our decisions in random games against a uniform-type Slumbot:
  # [(history, visible board, our seat, our cards)].
  out = []
  while len(out) < n:
    game = {}

    def on_ours(path, board, seat, st):
      if rng.random() < prob:
        out.append((path, list(board), seat, game["ours"]))

    play_game(rng, rules, on_ours=on_ours,
              on_start=lambda m, ours: game.update(ours=[int(c) for c in ours]))
  return out[:n]


# ── posterior ────────────────────────────────────────────────────────────────


def run_posterior(args, bp):
  rules = bp.env.rules
  rng = np.random.default_rng(args.seed)
  model = SlumbotModel(rules)
  ro = SlumbotRollouts(bp, model, n_games=args.games, chunk=1)
  preds, obs = [], []
  n_leaves = 0
  for path, board, seat, ours in _roots(rng, rules, args.roots, 0.5):
    # An odd depth limit from our decision leaves Slumbot to act after the leaf.
    tree = build_subgame(rules, model._step(path).bs, path, 3)
    belief, _ = model.belief(rules, path, ours, board, seat)
    if belief.sum() <= 0:
      continue
    _, r_term, _ = subgame_reach(tree, belief, model, ours, board, seat)
    # Leaves at the depth limit where Slumbot acts next, on the same street.
    pick = []
    for j in tree.leaves:
      t = tree.terminals[j]
      st = model._step(t.path)
      if (t.kind == MATRIX_T and r_term[j].sum() > 1e-6 and not bool(st.bs.done)
          and int(st.bs.cur_player) != seat and len(board) == (0, 3, 4, 5)[int(st.bs.street)]):
        pick.append(j)
    # Each leaf separately with its own seed: within one call every leaf's
    # rollout slot i shares its uniforms (common random numbers), which would
    # correlate the samples of different leaves.
    for j in pick:
      r = np.zeros_like(r_term)
      r[j] = r_term[j]
      ro.record = []
      ro.values(tree, r, ours, board, seat, seed=int(rng.integers(1 << 30)),
                t_mask=np.eye(4, dtype=bool)[[0] * len(tree.leaves)])
      leaf_paths, h, first = ro.record[0]
      p = model.probs([leaf_paths[0]], ours, board, seat)[0]
      for hh, a in zip(h, first):
        if a >= 0:
          preds.append(p[hh])
          obs.append(a)
    ro.record = None
    n_leaves += len(pick)
  if not obs:
    print("no leaf with Slumbot to act next")
    return 1
  preds, obs = np.array(preds), np.array(obs)
  p_obs = preds[np.arange(len(obs)), obs]
  print(f"leaves {n_leaves}, first Slumbot actions {len(obs)}, "
        f"mixed {(preds.max(-1) < 1 - 1e-9).sum()}")
  print(f"zero-probability first actions: {(p_obs <= 0).sum()}")
  print(f"rollout stats: {ro.stats}")
  if (preds.max(-1) < 1 - 1e-9).any():
    _calibration(preds, obs)
  return 0 if (p_obs <= 0).sum() == 0 else 1


# ── solve ────────────────────────────────────────────────────────────────────


def run_solve(args, bp):
  rules = bp.env.rules
  rng = np.random.default_rng(args.seed)
  ev = equity.preflop_ev_matrix(n_boards=args.boards, progress=False)
  model = SlumbotModel(rules)
  ro = SlumbotRollouts(bp, model, n_games=args.games, chunk=args.chunk)
  res = Resolver(bp, ev, depth=args.depth, iters=args.iters, model=model, rollouts=ro)
  rows = []
  for path, board, seat, ours in _roots(rng, rules, args.roots, 0.3):
    res.new_hand(seat, ours)
    if (model.belief(rules, path, ours, board, seat)[0] * compatible(ours, board)).sum() <= 0:
      continue
    t = time.time()
    sol = res.solve(path, board)
    st = sol.stats
    rows.append((len(board), st["nodes"], st["leaves"], st["gap"], st["t_reach"],
                 st["t_leaves"], time.time() - t, len(st["model_failures"])))
    print(f"street {len(board)} nodes {st['nodes']:4d} leaves {st['leaves']:4d} "
          f"value {st['value']:+8.2f} gap {st['gap']:.4f} reach {st['t_reach']:.1f}s "
          f"leaves {st['t_leaves']:.1f}s total {time.time() - t:.1f}s", flush=True)
  r = np.array(rows, np.float64)
  print(f"solves {len(rows)}: gap mean {r[:, 3].mean():.4f} max {r[:, 3].max():.4f} chips; "
        f"model failures {int(r[:, 7].sum())}")
  for b in (0, 3, 4, 5):
    sel = r[:, 0] == b
    if sel.any():
      print(f"  board {b}: {sel.sum()} solves, {r[sel, 6].mean():.1f}s each "
            f"(reach {r[sel, 4].mean():.1f}s, leaves {r[sel, 5].mean():.1f}s), "
            f"{r[sel, 2].mean():.0f} leaves")
  print(f"rollout stats: {ro.stats}")
  print(f"model: refusals {len(model.stats['failures'])}, desync {model.stats['desync']}, "
        f"raises clamped {model.stats['raise_clamped']}")
  return 0


# ── cards ────────────────────────────────────────────────────────────────────


def _street_leaves(tree, model, r_term, board):
  # Leaves where the street of `board` closes and the hand goes on.
  out = []
  for k, j in enumerate(tree.leaves):
    t = tree.terminals[j]
    st = model._step(t.path)
    if (t.kind == MATRIX_T and r_term[j].sum() > 1e-6 and not bool(st.bs.done)
        and (0, 3, 4, 5)[int(st.bs.street)] > len(board)):
      out.append((k, j))
  return out


def run_cards(args, bp):
  rules = bp.env.rules
  rng = np.random.default_rng(args.seed)
  model = SlumbotModel(rules)
  # The card rollouts exactly as ResolvingAivat builds them.
  ro = SlumbotRollouts(bp, model, n_games=args.games, chunk=args.chunk).resized(args.games, 1)
  diffs, spreads, scale, rows = [], [], [], []
  t_start = time.time()
  for path, board, seat, ours in _roots(rng, rules, args.roots, 0.5):
    if len(board) == 5:
      continue
    tree = build_subgame(rules, model._step(path).bs, path, args.depth)
    belief, _ = model.belief(rules, path, ours, board, seat)
    if belief.sum() <= 0:
      continue
    _, r_term, _ = subgame_reach(tree, belief, model, ours, board, seat)
    cand = _street_leaves(tree, model, r_term, board)
    for i in rng.permutation(len(cand))[: args.leaves]:
      k, j = cand[i]
      # Slumbot's hand from the leaf's reach, as play would deal it.
      q = r_term[j] / r_term[j].sum()
      h = int(rng.choice(len(q), p=q))
      r = np.zeros_like(r_term)
      r[j, h] = 1.0
      t_mask = np.zeros((len(tree.leaves), N_TRANSFORMS), bool)
      t_mask[k, 0] = True
      used = set(ours) | set(board) | set(int(c) for c in HAND_CARDS[h])
      rest = np.array([c for c in range(52) if c not in used])
      for _ in range(args.seeds):
        # The next cards: every turn or river card, or a fresh sample of
        # flops per seed, so that the seeds' differences are independent.
        if board:
          nexts = [[int(c)] for c in rest]
        else:
          nexts = [[int(c) for c in rng.choice(rest, 3, replace=False)]
                   for _ in range(args.flops)]
        seed = int(rng.integers(1 << 30))
        v_mean = ro.values(tree, r, ours, board, seat, seed=seed, t_mask=t_mask)[0][k, 0]
        v_card = np.array([ro.values(tree, r, ours, board + c, seat, seed=seed,
                                     t_mask=t_mask)[0][k, 0] for c in nexts])
        diffs.append(v_mean - v_card.mean())
        spreads.append(v_card.std())
        scale.append(abs(v_mean))
      rows.append((len(board), len(nexts)))
      d = np.array(diffs[-args.seeds:])
      print(f"board {len(board)} leaf {list(tree.terminals[j].path)} h {h}: "
            f"V {v_mean:+8.2f}, card spread {v_card.std():6.2f}, "
            f"V - mean_c V(c) {d.mean():+6.2f} (per seed {np.round(d, 2).tolist()}) "
            f"[{time.time() - t_start:.0f}s]", flush=True)
  if not diffs:
    print("no street-closing leaf")
    return 1
  d = np.array(diffs)
  se = d.std(ddof=1) / np.sqrt(len(d))
  z = d.mean() / se
  n_b = np.array(rows)
  print(f"leaves {len(rows)} (preflop {(n_b[:, 0] == 0).sum()}, flop {(n_b[:, 0] == 3).sum()}, "
        f"turn {(n_b[:, 0] == 4).sum()}), {args.seeds} seeds each")
  print(f"V - mean_c V(c): {d.mean():+.3f} ± {1.96 * se:.3f} chips (z {z:+.2f}); "
        f"mean |V| {np.mean(scale):.1f}, mean card spread {np.mean(spreads):.1f} chips")
  print(f"rollout stats: {ro.stats}")
  return 0 if abs(z) < 3 else 1


# ── chips ────────────────────────────────────────────────────────────────────


def run_chips(args, bp):
  rules = bp.env.rules
  rng = np.random.default_rng(args.seed)
  model = SlumbotModel(rules)
  env_ro = SlumbotRollouts(bp, model, n_games=args.games, chunk=args.chunk)
  real_ro = SlumbotRollouts(bp, model, n_games=args.games, chunk=args.chunk, real_chips=True)
  real_ro.record_pay = []
  v_env, v_real, z_all = [], [], []
  for path, board, seat, ours in _roots(rng, rules, args.roots, 0.3):
    tree = build_subgame(rules, model._step(path).bs, path, args.depth)
    belief, _ = model.belief(rules, path, ours, board, seat)
    if belief.sum() <= 0:
      continue
    _, r_term, _ = subgame_reach(tree, belief, model, ours, board, seat)
    t_mask = np.zeros((len(tree.leaves), N_TRANSFORMS), bool)
    t_mask[:, 0] = True
    seed = int(rng.integers(1 << 30))
    a, _ = env_ro.values(tree, r_term, ours, board, seat, seed=seed, min_mass=1e-4,
                         t_mask=t_mask)
    b, _ = real_ro.values(tree, r_term, ours, board, seat, seed=seed, min_mass=1e-4,
                          t_mask=t_mask)
    z = r_term[list(tree.leaves)].sum(-1)
    live = z > 1e-4
    v_env.append(a[live, 0] / z[live])
    v_real.append(b[live, 0] / z[live])
    z_all.append(z[live])
  env_pay, real_pay, scored, agree = (np.concatenate(x) for x in zip(*real_ro.record_pay))
  diff = np.abs(env_pay[agree] - real_pay[agree])
  print(f"games {len(env_pay)}, scored from real chips {scored.sum()}")
  print(f"env and real chips agree at a shared end: {agree.sum()} games, "
        f"max |env pay - real pay| {diff.max():.4f} chips")
  sc = real_ro.stats
  print(f"string ended while the env went on: {sc['string_ended']}; env ended first: "
        f"{sc['env_ended']}; desyncs {sc['desync']}")
  print(f"refusals: env scoring {env_ro.stats['refusals']} {env_ro.stats['refused'][:3]}; "
        f"real scoring {sc['refusals']} {sc['refused'][:3]}")
  ve, vr = np.concatenate(v_env), np.concatenate(v_real)
  d = vr - ve
  print(f"leaf values: {len(d)} leaves, real - env mean {d.mean():+.3f}, "
        f"mean |real - env| {np.abs(d).mean():.3f}, max |real - env| {np.abs(d).max():.3f} chips")
  return 0 if diff.max() < 0.01 else 1


# ── terminals ────────────────────────────────────────────────────────────────


def run_terminals(args, bp):
  rules = bp.env.rules
  rng = np.random.default_rng(args.seed)
  ev = equity.preflop_ev_matrix(n_boards=args.boards, progress=False)
  model = SlumbotModel(rules)
  n_exact, n_diff, n_empty, max_exact = 0, 0, 0, 0.0
  errs, weights = [], []
  for path, board, seat, ours in _roots(rng, rules, args.roots, 0.3):
    tree = build_subgame(rules, model._step(path).bs, path, args.depth)
    belief, _ = model.belief(rules, path, ours, board, seat)
    if belief.sum() <= 0:
      continue
    _, r_term, _ = subgame_reach(tree, belief, model, ours, board, seat)
    sign = showdown_sign(ours, board, ev)
    payoff = terminal_payoffs(tree, seat, sign)
    real = model.real_terminal_payoffs(tree, payoff, sign, board, seat)
    z = r_term.sum()
    bkey = tuple(int(c) for c in board)
    for j, t in enumerate(tree.terminals):
      if t.kind not in (FOLD_T, SHOWDOWN_T):
        continue
      seen = r_term[j] > 0
      if not seen.any():
        continue
      ws = model._walk(tuple(int(a) for a in t.path), bkey, seat)
      if not ws:
        n_empty += 1
        continue
      chips = [int(round(float(c))) * model.scale for c in np.asarray(t.betting.committed)]
      d = np.abs(real[j, seen] - payoff[j, seen])
      if all(list(w.game.committed) == chips for w in ws):
        n_exact += 1
        max_exact = max(max_exact, float(d.max()))
      else:
        n_diff += 1
        errs.append(float((r_term[j, seen] * d).sum() / r_term[j, seen].sum()))
        weights.append(float(r_term[j, seen].sum() / z))
  print(f"terminals reached: {n_exact} with every world on the tree's chips, {n_diff} with "
        f"other chips, {n_empty} with no world")
  print(f"on the tree's chips: max |real - tree| {max_exact:.2e} chips")
  if errs:
    e, w = np.array(errs), np.array(weights)
    print(f"on other chips: reach-mean |real - tree| per terminal {e.mean():.2f} chips, max "
          f"{e.max():.2f}; mean reach share {w.mean():.4f}")
  print(f"model: desync {model.stats['desync']}, refusals {len(model.stats['failures'])}")
  return 0 if max_exact < 1e-6 else 1


# ── hands ────────────────────────────────────────────────────────────────────


def run_hands(args, bp):
  rules = bp.env.rules
  rng = np.random.default_rng(args.seed)
  model = SlumbotModel(rules)
  rows = []
  for path, board, seat, ours in _roots(rng, rules, args.roots, 0.3):
    tree = build_subgame(rules, model._step(path).bs, path, args.depth)
    belief, _ = model.belief(rules, path, ours, board, seat)
    if belief.sum() <= 0:
      continue
    _, r_term, _ = subgame_reach(tree, belief, model, ours, board, seat)
    leaves = list(tree.leaves)
    z = r_term[leaves].sum(-1)
    # The rollouts' own cut: leaves below min_mass of the root are skipped.
    live = np.flatnonzero(z > 1e-4 * belief.sum())
    n_hands = (r_term[leaves] > 0).sum(-1)[live]
    rows.append((len(board), len(live), n_hands.mean() if len(live) else 0.0, n_hands.sum()))
  r = np.array(rows, np.float64)
  T = N_TRANSFORMS
  print(f"{len(r)} decisions, per-hand rollouts at {args.games} games per hand, {T} continuations")
  for b in (0, 3, 4, 5):
    sel = r[:, 0] == b
    if sel.any():
      per_hand = r[sel, 3].mean() * args.games * T
      ordinary = r[sel, 1].mean() * 128 * T
      print(f"  board {b}: {sel.sum()} solves, {r[sel, 1].mean():.0f} live leaves, "
            f"{r[sel, 2].mean():.0f} hands per leaf, {per_hand / 1e6:.2f}M games per solve "
            f"({ordinary / 1e3:.0f}k ordinary, x{per_hand / ordinary:.0f})")
  return 0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("posterior", "solve", "cards", "chips", "terminals", "hands"))
  ap.add_argument("run_dir")
  ap.add_argument("--checkpoint")
  ap.add_argument("--roots", type=int, default=20)
  ap.add_argument("--games", type=int, default=128, help="rollouts per leaf (and t)")
  ap.add_argument("--chunk", type=int, default=32)
  ap.add_argument("--depth", type=int, default=4)
  ap.add_argument("--iters", type=int, default=500)
  ap.add_argument("--boards", type=int, default=200_000,
                  help="preflop EV boards (solve; cached in $OPPEX_CACHE)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--leaves", type=int, default=2, help="cards: leaves per root")
  ap.add_argument("--seeds", type=int, default=4, help="cards: seeds per leaf")
  ap.add_argument("--flops", type=int, default=45, help="cards: flops sampled per preflop leaf")
  args = ap.parse_args()
  bp = Blueprint(args.run_dir, args.checkpoint)
  raise SystemExit({"posterior": run_posterior, "solve": run_solve, "cards": run_cards, "chips": run_chips, "terminals": run_terminals, "hands": run_hands}[args.mode](args, bp))


if __name__ == "__main__":
  main()
