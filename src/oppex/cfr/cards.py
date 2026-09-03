"""Hole-card indexing and the card-removal mask.

The 1326 two-card combinations are the *hands* CFR keeps regrets for. Everything
here is a small host-side constant built once at import.

**Card removal** is the reason this module exists. The cards you hold are cards
your opponent cannot hold, so their hand distribution is conditioned on yours: if
you hold AsAh, only one AA combination is left to them (AdAc) rather than six —
a 5.5x difference on a quantity that decides whether calling a shove is correct.

``MASK[a, b]`` is 1.0 exactly when hands ``a`` and ``b`` are disjoint. It belongs in **terminal evaluation only** —
never in reach propagation, where ``r_i[h]`` is player *i*'s own action-probability
product and carries no dependence on the opponent's holding. Masking during
propagation double-counts removal and converges to the wrong fixed point.

At a terminal node make sure to have the chance constant as 
``c * (MASK @ r_opp)` rather than ``c * r_opp.sum()``.

Card encoding matches ``oppex.envs.hunl_holdem``: index ``c`` in ``0..51`` decodes
to ``rank = c // 4`` (0=2 … 12=A) and ``suit = c % 4``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

# ── Sizes ────────────────────────────────────────────────────────────────────
N_CARDS = 52
N_HANDS = 1326                 # C(52, 2)
N_OPP = 1225                   # C(50, 2) — opponent combos left once you hold two
N_DEALS = N_HANDS * N_OPP      # 1_624_350 ordered disjoint pairs

RANKS = "23456789TJQKA"
SUITS = "cdhs"


# ── Hand indexing (colex order) ──────────────────────────────────────────────
# index(c0 < c1) = c1*(c1-1)//2 + c0. Recorded in the equity cache sidecar: a
# change to this convention silently invalidates every cached matrix.

HAND_CARDS = np.array(
  [[c0, c1] for c1 in range(N_CARDS) for c0 in range(c1)], dtype=np.int32
)  # (1326, 2)

HAND_INDEX = np.full((N_CARDS, N_CARDS), -1, dtype=np.int32)
HAND_INDEX[HAND_CARDS[:, 0], HAND_CARDS[:, 1]] = np.arange(N_HANDS)
HAND_INDEX[HAND_CARDS[:, 1], HAND_CARDS[:, 0]] = np.arange(N_HANDS)


def hand_index(c0: int, c1: int) -> int:
  """Index of the combo holding cards ``c0`` and ``c1`` (order-independent)."""
  return int(HAND_INDEX[c0, c1])


def hand_label(i: int) -> str:
  """Human-readable combo, e.g. ``'AcKd'`` — high card first."""
  c0, c1 = HAND_CARDS[i]
  lo, hi = (c0, c1) if c0 // 4 < c1 // 4 else (c1, c0)
  return f"{RANKS[hi // 4]}{SUITS[hi % 4]}{RANKS[lo // 4]}{SUITS[lo % 4]}"


# ── Card-removal mask ────────────────────────────────────────────────────────

_INCIDENCE = np.zeros((N_HANDS, N_CARDS), dtype=np.int8)  # (1326, 52)
_INCIDENCE[np.arange(N_HANDS), HAND_CARDS[:, 0]] = 1
_INCIDENCE[np.arange(N_HANDS), HAND_CARDS[:, 1]] = 1

OVERLAP = (_INCIDENCE.astype(np.int16) @ _INCIDENCE.T.astype(np.int16)) > 0  # share a card
MASK_NP = (~OVERLAP).astype(np.float32)
MASK = jnp.asarray(MASK_NP)          # (1326, 1326) float32, 1.0 iff disjoint
INCIDENCE = jnp.asarray(_INCIDENCE.astype(np.float32))


def masked_sum(r: jnp.Array) -> jnp.Array:
  """``MASK @ r`` by inclusion–exclusion — an independent derivation, for tests.

  Subtract, from the unconditional total, the mass of every opponent hand using
  either of your two cards; hands using *both* got subtracted twice, so add one
  copy back. Roughly 7x faster than the matvec, but it relies on cancellation, so
  the solver uses ``MASK @ r`` and this exists to prove the mask is consistent.
  """
  per_card = INCIDENCE.T @ r                       # (52,)
  return r.sum() - per_card[HAND_CARDS[:, 0]] - per_card[HAND_CARDS[:, 1]] + r


# ── 169 isomorphism classes (for chart reporting) ────────────────────────────
# Grid convention matching published preflop charts: pairs on the diagonal,
# suited above it, offsuit below, aces top-left. Class index = row * 13 + col.

_r0, _r1 = HAND_CARDS[:, 0] // 4, HAND_CARDS[:, 1] // 4
_suited = (HAND_CARDS[:, 0] % 4) == (HAND_CARDS[:, 1] % 4)
_hi, _lo = np.maximum(_r0, _r1), np.minimum(_r0, _r1)
_row = np.where(_suited, 12 - _hi, 12 - _lo)
_col = np.where(_suited, 12 - _lo, 12 - _hi)
HAND_CLASS = (_row * 13 + _col).astype(np.int32)  # (1326,)


def class_label(k: int) -> str:
  """Label for one of the 169 classes, e.g. ``'AKs'``, ``'AKo'``, ``'AA'``."""
  row, col = divmod(int(k), 13)
  hi, lo = 12 - min(row, col), 12 - max(row, col)
  if row == col:
    return f"{RANKS[hi]}{RANKS[lo]}"
  return f"{RANKS[hi]}{RANKS[lo]}{'s' if col > row else 'o'}"


# ── Self-checks ──────────────────────────────────────────────────────────────
# Cheap, and each one is a one-line proof of a property the solver silently
# depends on. Failure here is far preferable to a plausible-looking equilibrium.

def _self_check() -> None:
  row_sums = np.unique(MASK_NP.sum(axis=1))
  assert row_sums.tolist() == [float(N_OPP)], (
    f"every hand must be compatible with exactly {N_OPP} others, got {row_sums}"
  )
  assert OVERLAP[np.arange(N_HANDS), np.arange(N_HANDS)].all(), "a hand overlaps itself"
  assert np.array_equal(MASK_NP, MASK_NP.T), "mask must be symmetric"

  counts = np.bincount(HAND_CLASS, minlength=169)
  assert counts.sum() == N_HANDS
  assert set(np.unique(counts).tolist()) == {4, 6, 12}, np.unique(counts).tolist()
  assert (counts == 6).sum() == 13, "13 pairs, 6 combos each"
  assert (counts == 4).sum() == 78, "78 suited classes, 4 combos each"
  assert (counts == 12).sum() == 78, "78 offsuit classes, 12 combos each"


_self_check()
