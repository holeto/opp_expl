"""Exact Nash solution of the shove-fold game by linear programming.

The sharpest available check on the solver. With ``allow_limp=False`` the preflop
game is bilinear zero-sum over two boxes — the small blind's shove probability
per hand and the big blind's call probability per hand — and the big blind's
inner minimisation decomposes per hand, which collapses it to a linear program
small enough for HiGHS to solve exactly.

Crucially this validates the *tree, traversal, chance normalisation and mask*
without depending on the equity matrix being exact: the LP and CFR are fed the
same ``EV``, so any disagreement is a bug in the solver, not sampling noise in
the equity estimate. That makes it a far better instrument than comparing against
published charts, where an equity error and a solver error look identical.

Derivation. Writing ``x_a`` for P(shove | a) and ``y_b`` for P(call | b), the
value to the small blind is

    (1/N_DEALS) Σ_{a,b} MASK[a,b] [ (1-x_a)(-sb)
                                    + x_a ( (1-y_b) bb + y_b · stake · EV[a,b] ) ]

The big blind picks each ``y_b`` independently, so their contribution is
``t_b = min(bb · Σ_a MASK[a,b] x_a,  stake · Σ_a MASK[a,b] EV[a,b] x_a)``, and the
fold term telescopes (``Σ_b MASK[a,b] = N_OPP``) to a constant plus a linear term.
Maximising over ``x`` is then a plain LP in 2652 variables.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import bmat, eye

from .cards import MASK_NP, N_DEALS, N_HANDS, N_OPP


def solve_shove_fold(
  ev: np.ndarray, *, small_blind: float, big_blind: float, stake: float,
  method: str = "highs",
):
  """Exact equilibrium of the shove-fold game.

  Returns ``(value_to_sb_chips, shove_probs (1326,), call_probs (1326,))``.
  ``stake`` is the chips at risk at the all-in showdown (the starting stack).
  """
  ev = np.asarray(ev, dtype=np.float64)
  mask = MASK_NP.astype(np.float64)
  m_ev = mask * ev

  # t_b <= bb * Σ_a MASK[a,b] x_a           (big blind folds)
  # t_b <= stake * Σ_a MASK[a,b] EV[a,b] x_a (big blind calls)
  ident = eye(N_HANDS, format="csr")
  a_ub = bmat([[-big_blind * mask.T, ident],
               [-stake * m_ev.T, ident]], format="csr")
  b_ub = np.zeros(2 * N_HANDS)

  # maximise (sb/N_HANDS) Σ x + (1/N_DEALS) Σ t  ->  minimise its negative
  c = np.concatenate([
    np.full(N_HANDS, -small_blind / N_HANDS),
    np.full(N_HANDS, -1.0 / N_DEALS),
  ])
  bounds = [(0.0, 1.0)] * N_HANDS + [(None, None)] * N_HANDS

  res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method=method)
  if not res.success:
    raise RuntimeError(f"LP failed: {res.message}")

  x = np.clip(res.x[:N_HANDS], 0.0, 1.0)
  value = -small_blind - res.fun   # the folded-every-hand baseline, plus the gain

  # Big blind's best reply, read off which of the two bounds is tight.
  fold_branch = big_blind * (mask.T @ x)
  call_branch = stake * (m_ev.T @ x)
  y = (call_branch < fold_branch).astype(np.float64)   # call when it hurts the SB more
  return float(value), x, y


def check_against_cfr(lp_value, br0, br1, tol=1e-3):
  """The LP value must sit inside the best-response sandwich ``br0 >= v >= -br1``."""
  assert br0 >= lp_value - tol, f"BR0 {br0:.6f} below LP value {lp_value:.6f}"
  assert -br1 <= lp_value + tol, f"-BR1 {-br1:.6f} above LP value {lp_value:.6f}"
  return True
