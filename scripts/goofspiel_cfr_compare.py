"""Vanilla history CFR against the general public-state CFR on Goofspiel.

  uv run python scripts/goofspiel_cfr_compare.py [--iters 1000]

1. **Same iteration map.** Along a PSCFR trajectory, the tables before each
   step are handed to vanilla CFR (rows matched by infoset key), and its next
   tables must equal PSCFR's, under four CFR schemes. Comparing one step at a
   time, rather than two free-running trajectories, is deliberate: Goofspiel
   has exact ties, and a 1e-16 difference in summation order flips regret
   matching between uniform and pure within a couple of iterations.
2. **Both converge.** Free-running from zero, each solver's average strategy
   is scored by the independent exact best response of
   ``oppex.tree.best_response_values``; NashConv must go to 0.
3. **Speed** per iteration for both, on Goofspiel 3, 4 and 5 (descending).
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
from oppex.tree import cfr as vcfr  # noqa: E402
from oppex.tree.history_game import from_layered  # noqa: E402
from oppex.tree.public import strategies_from_rows, to_public_game  # noqa: E402
from oppex.tree.resolve import step_disagreement  # noqa: E402

SCHEMES = {
  "vanilla": dict(),
  "linear alt": dict(alternating=True, linear=True),
  "CFR+": dict(alternating=True, plus=True, linear=True),
  "DCFR alt": dict(alternating=True, discount=solver.DCFR),
}


def build(n):
  env = build_env({"name": "goofspiel", "n_cards": n, "prize_order": "descending"})
  t0 = time.time()
  layered = extract_game_tree(env)
  game = from_layered(layered)
  pg = to_public_game(game)
  return layered, game, pg, time.time() - t0


def nash_conv(layered, strategies):
  bv = best_response_values(layered, strategies[0], strategies[1])
  return bv.v0_p0_br + bv.v1_p1_br


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=1000)
  ap.add_argument("--step-iters", type=int, default=30)
  ap.add_argument("--time-iters", type=int, default=20)
  args = ap.parse_args()

  games = {}
  for n in (3, 4, 5):
    games[n] = build(n)
    layered, game, pg, dt = games[n]
    n_inf = len(vcfr.layout(game).rows)
    print(f"Goofspiel {n}: {game.n_nodes} histories, {n_inf} acting infosets, "
          f"{len(pg.states)} public states (built in {dt:.1f}s)")

  print("\n1. One step of vanilla CFR from PSCFR's tables")
  worst = 0.0
  for n in (3, 4):
    _, game, pg, _ = games[n]
    for name, scheme in SCHEMES.items():
      w = step_disagreement(game, args.step_iters, scheme, pg)
      worst = max(worst, w)
      print(f"  Goofspiel {n} {name:11s} {args.step_iters} steps: worst relative error {w:.1e}")
  assert worst < 1e-10, "vanilla CFR and PSCFR take different steps"

  print(f"\n2. Free-running convergence on Goofspiel 3, alternating DCFR")
  layered, game, pg, _ = games[3]
  scheme = SCHEMES["DCFR alt"]
  tp, tv = pscfr.init_tables(pg), vcfr.init_tables(game)
  print(f"  {'iter':>6}  {'NashConv PSCFR':>15}  {'NashConv vanilla':>17}")
  for it in range(1, args.iters + 1):
    tp, tv = pscfr.cfr_step(pg, tp, **scheme), vcfr.cfr_step(game, tv, **scheme)
    if it in (1, 10, 100) or it == args.iters:
      sp = strategies_from_rows(pg, pscfr.average_strategy(pg, tp)[0])
      sv = vcfr.strategy_dicts_from_rows(game, vcfr.average_strategy(game, tv)[0])
      print(f"  {it:>6}  {nash_conv(layered, sp):>15.2e}  {nash_conv(layered, sv):>17.2e}")
  assert nash_conv(layered, sp) < 1e-2 and nash_conv(layered, sv) < 1e-2

  print("\n3. Time per iteration (alternating DCFR, numpy, CPU)")
  print(f"  {'game':12s} {'PSCFR ms':>10} {'vanilla ms':>11} {'ratio':>7}")
  for n in (3, 4, 5):
    _, game, pg, _ = games[n]
    times = []
    for mod, g in ((pscfr, pg), (vcfr, game)):
      tab = mod.cfr_step(g, mod.init_tables(g), **scheme)
      t0 = time.perf_counter()
      for _ in range(args.time_iters):
        tab = mod.cfr_step(g, tab, **scheme)
      times.append((time.perf_counter() - t0) / args.time_iters * 1e3)
    print(f"  Goofspiel {n:<2d} {times[0]:>10.2f} {times[1]:>11.2f} {times[0] / times[1]:>7.2f}")
  print("\nAll checks passed.")


if __name__ == "__main__":
  main()
