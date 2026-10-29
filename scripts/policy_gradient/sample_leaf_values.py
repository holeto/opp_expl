"""Sample the matrix-valued leaf values of a checkpoint's biased continuations.

  uv run python scripts/policy_gradient/sample_leaf_values.py checkpoints/qmmd_hunl_560k \\
      --games 10000 --factor 5 --buckets 10

Plays ``--games`` deals per pair of continuations from the leaf reached by
``--path`` (preflop atoms; the default ``1 1`` is SB limp, BB check) and saves
the games and the bucket-pair values to ``<run_dir>/leaf_values/``. See
``oppex.policy_gradient.leaf_values`` for the estimator.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from oppex.cfr import equity
from oppex.policy_gradient import leaf_values as lv
from oppex.policy_gradient.blueprint import TRANSFORM_NAMES, Blueprint


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir")
  ap.add_argument("--checkpoint")
  ap.add_argument("--params", choices=("params", "target", "magnet"), default="params")
  ap.add_argument("--path", type=int, nargs="*", default=[1, 1])
  ap.add_argument("--games", type=int, default=10_000)
  ap.add_argument("--factor", type=float, default=5.0, help="bias multiplier (Pluribus: 5)")
  ap.add_argument("--buckets", type=int, default=10)
  ap.add_argument("--boards", type=int, default=200_000, help="MC boards of the equity matrix")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  bp = Blueprint(args.run_dir, args.checkpoint, args.params)
  path = tuple(args.path)
  out = lv.cache_path(bp, path, args.games, args.factor, args.buckets, args.seed, args.run_dir)
  print(f"{bp.checkpoint} (step {bp.step}, {args.params}), leaf path {list(path)}")

  t0 = time.time()
  samples = lv.sample_leaf(bp, path, args.games, args.factor, args.seed)
  print(f"  {16 * args.games:,} games in {time.time() - t0:.1f}s, "
        f"truncated {samples.truncated.mean():.2e}")

  ev = equity.preflop_ev_matrix(n_boards=args.boards)
  buckets = lv.bucket_hands(np.asarray(ev), args.buckets)
  vb, counts = lv.bucket_values(samples, buckets, args.buckets)
  print(f"  {args.buckets} buckets, empty cells {(counts == 0).mean():.1%}, "
        f"min samples per cell {counts.min()}")

  # Hand-averaged value per continuation pair, the matrix game at this leaf.
  print(f"\n  mean chips to SB {'':6s}" + "".join(f"{n:>10s}" for n in TRANSFORM_NAMES))
  for t0_, name in enumerate(TRANSFORM_NAMES):
    row = samples.payoff[t0_].mean(-1)
    print(f"  SB {name:<20s}" + "".join(f"{v:>10.3f}" for v in row))

  lv.save_leaf(out, samples, vb, counts, buckets, path, args.factor)
  print(f"\n  saved {out}")


if __name__ == "__main__":
  main()
