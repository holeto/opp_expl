"""Duplicate matches: every deal played twice, with the seats and the cards swapped.

Hands ``2k`` and ``2k + 1`` form pair ``k``: the second replays the first's deal
with each player in the other's seat holding the other's cards (the ACPC's
duplicate poker). The card luck of a deal then enters both hands with opposite
signs, so the pair average has less variance than two independent hands.

The duplicate estimate is the mean over pairs of the pair average, with its
interval from the spread of the pair averages. Its per-hand std is the std of
the pair averages × √2, the std of one hand that would give the same interval
with twice as many independent hands, so it compares directly with the
unpaired std. It is lower exactly when the two hands of a pair are negatively
correlated. Both estimates use only complete pairs.

  report_duplicate(recs, "raw", value=lambda r: r["payoff"] * mbb)
  report_duplicate([(hand, value), ...], "raw")
"""

from __future__ import annotations

import numpy as np


def pair_values(items, value=None, hand=None) -> tuple[np.ndarray, np.ndarray]:
  """The values of the first and the second hand of every complete pair.

  ``items`` are ``(hand, value)`` tuples, or records read with ``value(rec)``
  and ``hand(rec)`` (default ``rec["hand"]``). A pair missing a hand is dropped.
  """
  if value is None:
    by_hand = {int(h): float(v) for h, v in items}
  else:
    hand = hand or (lambda r: r["hand"])
    by_hand = {int(hand(r)): float(value(r)) for r in items}
  ks = sorted({h // 2 for h in by_hand if h - h % 2 in by_hand and h - h % 2 + 1 in by_hand})
  return (np.array([by_hand[2 * k] for k in ks]), np.array([by_hand[2 * k + 1] for k in ks]))


def duplicate_stats(first: np.ndarray, second: np.ndarray) -> dict:
  """Unpaired and duplicate means, ±95% half-widths, per-hand stds, correlation.

  Returns a dict: ``pairs``; ``mean``, ``ci`` and ``std`` over the hands;
  ``dup_mean``, ``dup_ci`` over the pair averages and ``dup_std``, their std
  × √2; ``corr`` between the two hands (nan below 2 pairs or with no spread).
  """
  n = len(first)
  both = np.concatenate([first, second])
  avg = (first + second) / 2
  sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else float("nan")
  corr = float("nan")
  if n > 1 and first.std() > 0 and second.std() > 0:
    corr = float(np.corrcoef(first, second)[0, 1])
  return {"pairs": n, "mean": float(both.mean()) if n else float("nan"),
          "ci": 1.96 * sd(both) / np.sqrt(max(2 * n, 1)), "std": sd(both),
          "dup_mean": float(avg.mean()) if n else float("nan"),
          "dup_ci": 1.96 * sd(avg) / np.sqrt(max(n, 1)), "dup_std": sd(avg) * np.sqrt(2),
          "corr": corr}


def report_duplicate(items, label: str, value=None, hand=None, std_unit: float = 1e3) -> dict:
  """Prints one line comparing the unpaired and the duplicate estimate.

  ``items``, ``value`` and ``hand`` as in ``pair_values``. Means are printed in
  the values' unit (mbb/hand for the callers here), stds divided by
  ``std_unit`` (BB/hand). Returns ``duplicate_stats``.
  """
  s = duplicate_stats(*pair_values(items, value, hand))
  print(f"  {label:<22} unpaired {s['mean']:+8.0f} ± {s['ci']:6.0f} std {s['std'] / std_unit:6.2f}"
        f" | duplicate {s['dup_mean']:+8.0f} ± {s['dup_ci']:6.0f} std "
        f"{s['dup_std'] / std_unit:6.2f} over {s['pairs']} pairs, correlation {s['corr']:+.2f}")
  return s


def report_aivat_duplicate(recs, mbb: float) -> None:
  """``report_duplicate`` for the AIVAT estimators of ``resolving_aivat.report_aivat``."""
  from .resolving_aivat import TERMS
  recs = [r for r in recs if "aivat" in r]
  rows = [("raw", ()), ("+ opponent decisions", TERMS)]
  if recs and all("board_q" in r["aivat"] for r in recs):
    rows.append(("boards by rollouts", ("deal", "board_q", "runout", "ours", "opponent")))
  print("\n  AIVAT, duplicate pairs (mbb/hand, ±95%; std in BB/hand):")
  for label, keys in rows:
    report_duplicate(recs, label,
                     value=lambda r, keys=keys: (r["payoff"] + sum(r["aivat"][k] for k in keys))
                     * mbb)
