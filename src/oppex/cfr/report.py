"""Formatting strategies as the 13x13 grid published preflop charts use.

Rows and columns run A down to 2. Pairs sit on the diagonal, suited hands above
it, offsuit below — the layout every poker chart uses, so results can be compared
by eye against published ranges.

The solver works in 1326 combinations; a chart works in 169 suit-isomorphic
classes. Collapsing is a *reporting* step only, and it is an average over the
combinations in each class, weighted equally. A class showing 0.5 can therefore
mean either "half the combos shove" or "every combo shoves half the time" — read
the grid as a summary, never as evidence about a specific combination.
"""

from __future__ import annotations

import numpy as np

from .cards import HAND_CLASS, RANKS, class_label


def to_grid(per_hand: np.ndarray) -> np.ndarray:
  """Collapse a (1326,) per-combo quantity to the (13, 13) class grid."""
  per_hand = np.asarray(per_hand, dtype=np.float64)
  sums = np.bincount(HAND_CLASS, weights=per_hand, minlength=169)
  counts = np.bincount(HAND_CLASS, minlength=169)
  return (sums / np.maximum(counts, 1)).reshape(13, 13)


def format_grid(per_hand: np.ndarray, title: str = "", scale: int = 100) -> str:
  """Render as integer percentages, blanking zeros so the range shape stands out."""
  grid = to_grid(per_hand) * scale
  head = "    " + "".join(f"{RANKS[12 - c]:>4s}" for c in range(13))
  lines = [title, head] if title else [head]
  for r in range(13):
    cells = "".join(
      f"{grid[r, c]:>4.0f}" if grid[r, c] >= 0.5 else "   ." for c in range(13)
    )
    lines.append(f"{RANKS[12 - r]:>3s}" + cells)
  return "\n".join(lines)


def range_summary(per_hand: np.ndarray) -> dict:
  """Combo-weighted frequency, plus the classes at the edge of the range."""
  per_hand = np.asarray(per_hand, dtype=np.float64)
  grid = to_grid(per_hand).reshape(-1)
  mixed = np.where((grid > 0.01) & (grid < 0.99))[0]
  return {
    "frequency": float(per_hand.mean()),          # over all 1326 combos
    "classes_always": int((grid >= 0.99).sum()),
    "classes_never": int((grid <= 0.01).sum()),
    "classes_mixed": [f"{class_label(int(k))}:{grid[k]:.2f}" for k in mixed],
  }
