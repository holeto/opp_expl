"""Local best response against the chumps of Lisý & Bowling's Table 2, in the real game.

The opponents are ACPC example players (``slumbot/lbr.py``, ``ChumpOpponent``):
always call, half call and half a random raise, and a random action type. A
fourth, ``uniform``, plays every legal action with equal probability, each
raise size counting as one action. A raise is uniform over whole chips
(``--raise-step chip``, the example player) or whole big blinds (``bb``), from
the legal minimum raise (``--raise-floor min``) or from one step (``step``).
LBR decides on the real game with an exact model of the chump (``RealLBR``),
so no checkpoint is loaded. The paper's values (BB/hand, ±95%):

  bets  rounds  always call  1/2 call 1/2 raise  random
  fc    1-4     0            7.1 ± 0.5           15.7 ± 0.4
  fc    3-4     0            16.2 ± 0.3           2.2 ± 0.7
  fcpa  1-4     34.0 ± 0.5   23.1 ± 0.5          39.1 ± 0.6
  fcpa  3-4     49.0 ± 0.4   24.4 ± 0.6          80.7 ± 0.7

The paper's random chump is our ``uniform`` with raises in chips, which matches
all four of its cells; our ``random`` draws each action type with probability
1/3. Rounds 3-4 is ``--call-streets 2``.

  uv run python scripts/policy_gradient/lbr_chumps.py --chump random --call-streets 2 \
    --hands 1000 --duplicate --out /tmp/lbr_random.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from oppex.cfr import equity
from oppex.policy_gradient.duplicate import report_duplicate
from oppex.policy_gradient.slumbot.api import BIG_BLIND
from oppex.policy_gradient.slumbot.lbr import (BET_FRACTIONS, CHUMPS, RAISE_FLOORS, RAISE_STEPS,
                                               ChumpOpponent, RealLBR, play_chump_hand)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--chump", choices=sorted(CHUMPS), required=True)
  ap.add_argument("--bets", choices=sorted(BET_FRACTIONS), default="fcpa")
  ap.add_argument("--call-streets", type=int, default=0,
                  help="check/call on the first N streets (the paper's rounds 3-4: 2)")
  ap.add_argument("--raise-step", choices=sorted(RAISE_STEPS), default="chip",
                  help="chips between the chump's raise sizes: one chip or one big blind")
  ap.add_argument("--raise-floor", choices=RAISE_FLOORS, default="min",
                  help="the chump's smallest raise: the legal minimum raise, or one step")
  ap.add_argument("--hands", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--duplicate", action="store_true",
                  help="hands 2k and 2k+1 replay one deal with the seats and cards swapped")
  ap.add_argument("--out", required=True, help="one JSON line per hand")
  args = ap.parse_args()

  # Preflop signs need the EV matrix only when LBR decides preflop.
  ev = (equity.preflop_ev_matrix(n_boards=200_000, progress=False)
        if args.call_streets == 0 else None)
  lbr = RealLBR(ev, bets=args.bets, call_streets=args.call_streets)
  client = ChumpOpponent(CHUMPS[args.chump], seed=args.seed + 1, duplicate=args.duplicate,
                         raise_step=RAISE_STEPS[args.raise_step], raise_floor=args.raise_floor)

  results, t0 = [], time.monotonic()
  with open(args.out, "w") as f:
    for h in range(args.hands):
      hand = play_chump_hand(client, lbr)
      hand.update(hand=h, chump=args.chump, bets=args.bets, call_streets=args.call_streets,
                  raise_step=args.raise_step, raise_floor=args.raise_floor, seed=args.seed)
      if args.duplicate:
        hand["pair"] = h // 2
      f.write(json.dumps(hand) + "\n")
      results.append((h, hand["winnings"]))
  total = sum(w for _, w in results)
  print(f"{args.chump}, {args.bets}, call streets {args.call_streets}, raise step "
        f"{args.raise_step}, floor {args.raise_floor}: {len(results)} hands, "
        f"{total / BIG_BLIND / len(results):+.2f} BB/hand, {time.monotonic() - t0:.0f}s")
  print(lbr.summary())
  if args.duplicate:
    print("duplicate pairs (mbb/hand, ±95%; std in BB/hand):")
    report_duplicate([(h, 1000 * w / BIG_BLIND) for h, w in results], "winnings")


if __name__ == "__main__":
  main()
