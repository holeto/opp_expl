"""Re-solving against Slumbot in the env, with a paired blueprint control.

Needs the Slumbot data (set ``SLUMBOT_DIR``). The opponent is Slumbot itself,
its moves from the C++ library with a random translation type (slumbot.com
hand number) per hand; the re-solver models it exactly (``SlumbotModel``) and
rolls leaves out against it (``SlumbotRollouts``). The control replays every
hand with the blueprint in our seat, the same deal and the same Slumbot.

  uv run python scripts/policy_gradient/resolve_vs_slumbot.py checkpoints/qmmd_hunl_560k --hands 100
  uv run python scripts/policy_gradient/resolve_vs_slumbot.py RUN_DIR --report LOG.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from oppex.cfr import equity
from oppex.policy_gradient.blueprint import Blueprint
from oppex.policy_gradient.resolving import Resolver, play_vs_blueprint, report_play
from oppex.policy_gradient.resolving_aivat import ResolvingAivat, report_aivat
from oppex.policy_gradient.slumbot.model import SlumbotModel
from oppex.policy_gradient.slumbot.opponent import SlumbotOpponent
from oppex.policy_gradient.slumbot.rollouts import SlumbotRollouts


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir")
  ap.add_argument("--checkpoint")
  ap.add_argument("--params", choices=("params", "target", "magnet"), default="params")
  ap.add_argument("--hands", type=int, default=100)
  ap.add_argument("--depth", type=int, default=4)
  ap.add_argument("--leaf-games", type=int, default=128)
  ap.add_argument("--cfr-iters", type=int, default=500)
  ap.add_argument("--boards", type=int, default=200_000, help="preflop EV boards (cached)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--no-aivat", action="store_true")
  ap.add_argument("--no-cover-opponent", action="store_true",
                  help="skip AIVAT terms for Slumbot decisions outside every played subgame")
  ap.add_argument("--out", help="JSONL log (default: <run_dir>/resolving/vs_slumbot_...)")
  ap.add_argument("--report", nargs="+", metavar="LOG",
                  help="only pool and report these logs (several seeds)")
  args = ap.parse_args()
  if args.report:
    recs = [json.loads(line) for path in args.report for line in open(path)]
    with open(os.path.join(args.run_dir, "config.yaml")) as f:
      mbb = 1e3 / float(yaml.safe_load(f)["env"]["big_blind"])
    print(f"{len(recs)} hands from {len(args.report)} logs")
    report_play(recs, mbb)
    if any("aivat" in r for r in recs):
      report_aivat(recs, mbb)
    return

  bp = Blueprint(args.run_dir, args.checkpoint, args.params)
  ev = equity.preflop_ev_matrix(n_boards=args.boards, progress=False)
  model = SlumbotModel(bp.env.rules)
  rollouts = SlumbotRollouts(bp, model, n_games=args.leaf_games)
  res = Resolver(bp, ev, depth=args.depth, iters=args.cfr_iters, seed=args.seed, model=model,
                 rollouts=rollouts)
  opponent = SlumbotOpponent(model)
  aivat = None if args.no_aivat else ResolvingAivat(res, cover_opponent=not args.no_cover_opponent)
  out = args.out or os.path.join(args.run_dir, "resolving",
                                 f"vs_slumbot_d{args.depth}_n{args.leaf_games}_"
                                 f"it{args.cfr_iters}_s{args.seed}.jsonl")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  print(f"{args.hands} hands against Slumbot -> {out}", flush=True)
  with open(out, "w") as f:
    recs = play_vs_blueprint(bp, res, args.hands, seed=args.seed, log=f, aivat=aivat,
                             opponent=opponent)
  mbb = 1e3 / bp.env.big_blind
  report_play(recs, mbb)
  if aivat is not None:
    report_aivat(recs, mbb)
  print(f"  Slumbot: {opponent.stats['moves']} moves, {len(opponent.stats['refused'])} refused, "
        f"{opponent.stats['desync']} desyncs; model refusals {len(model.stats['failures'])}, "
        f"desyncs {model.stats['desync']}; rollouts {rollouts.stats['refusals']} refusals "
        f"{rollouts.stats['refused'][:5]}, {rollouts.stats['desync']} desyncs")


if __name__ == "__main__":
  main()
