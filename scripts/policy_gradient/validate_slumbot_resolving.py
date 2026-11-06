"""Validate re-solving against Slumbot (needs the Slumbot data: set ``SLUMBOT_DIR``).

  posterior  Rollouts start from the posterior over Slumbot's translation type.
             In depth-3 subgames, at leaves where Slumbot acts next, its first rollout action for
             hand ``h`` must be distributed as ``SlumbotModel.probs`` at the
             leaf (calibration over the mixed predictions; zero-probability
             actions must never occur).
  solve      ``Resolver`` with ``SlumbotModel`` and ``SlumbotRollouts`` at our
             decisions of random games against Slumbot: the CFR best-response
             gap, refusals, desyncs and time per solve.

  uv run python scripts/policy_gradient/validate_slumbot_resolving.py posterior RUN_DIR
  uv run python scripts/policy_gradient/validate_slumbot_resolving.py solve RUN_DIR --roots 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from oppex.cfr import equity
from oppex.cfr.subgame import build_subgame
from oppex.cfr.tree import MATRIX_T
from oppex.policy_gradient.blueprint import Blueprint
from oppex.policy_gradient.resolving import Resolver, compatible, subgame_reach
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


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("posterior", "solve"))
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
  args = ap.parse_args()
  bp = Blueprint(args.run_dir, args.checkpoint)
  raise SystemExit({"posterior": run_posterior, "solve": run_solve}[args.mode](args, bp))


if __name__ == "__main__":
  main()
