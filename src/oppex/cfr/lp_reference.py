"""Exact Nash solution of the shove-fold game by linear programming.

The shove-fold game is a minimal poker game, where each player only
acts (at most) once, so the little blind cannot CALL at the root, only
ALL_IN or FOLD. After that, the player big blind player responds accordingly.
Sanity check for minimal PSCFR

**Why this is not the textbook matrix-game LP**, ``max V s.t. V <= Σ_a σ(a) u(·,a)``
with one constraint per opponent pure strategy. Both players here have private
information: 1326 types with 2 actions each, so an opponent *pure strategy* is a
map from hands to actions — 2^1326 of them, and that many constraints cannot be
written down.

What makes it tractable is that a player's strategy space is a **product of
per-type simplices** — the box ``[0,1]^1326``, not a simplex — and the payoff is
bilinear. The inner minimisation therefore separates coordinate-wise: each ``y_b``
independently minimises a linear function. So the single scalar ``V`` splits into a
sum of per-type values ``Σ_b t_b``, and the constraints collapse from one per
opponent pure strategy to **two per opponent type** (one per action available to
that type). That is exactly the standard construction, indexed by (type, action)
rather than by pure strategy.

Note there is deliberately no ``Σ_a x_a = 1``: ``x_a`` is P(shove | hand a), an
independent probability per hand, so the feasible set is a hypercube. Normalising
across hands would assert the small blind shoves one hand's worth in total.

Equivalently this **is** a sequence-form LP at depth 1. The small blind's sequences
are ``(a, shove)`` and ``(a, fold)`` with realisation constraint
``r(a,shove) + r(a,fold) = P(a)``; eliminating ``r(a,fold)`` is free when a type has
two actions, and produces the constant ``-sb`` offset in the returned value. The
limp line defeats that elimination: the small blind then has ``(a, limp, call)`` and
``(a, limp, fold)`` summing to ``r(a, limp)`` — a variable rather than a constant —
so the feasible set is no longer a box and explicit sequence form is required.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_array, eye

from .cards import MASK_NP, N_DEALS, N_HANDS, N_OPP


def solve_shove_fold(
  ev: np.ndarray, *, small_blind: float, big_blind: float, stake: float,
  method: str = "highs",
):
  """Exact equilibrium of the shove-fold matrix game — one move per player.

  Valid only against ``build_preflop_tree(..., allow_limp=False)``. Pointing it at
  the limp tree would be comparing two different games: see the module docstring.

  Returns ``(value_to_sb_chips, shove_probs (1326,), call_probs (1326,))``.
  ``stake`` is the chips at risk at the all-in showdown (the starting stack).
  """
  ev = np.asarray(ev, dtype=np.float64)
  mask = MASK_NP.astype(np.float64)
  #Showdown utility for each hand configuration
  m_ev = mask * ev

  # t_b <= bb * Σ_a MASK[a,b] x_a           (big blind folds)
  # t_b <= stake * Σ_a MASK[a,b] EV[a,b] x_a (big blind calls)
  ident = eye(N_HANDS, format="csr")
  #Player 1 is the small blind player
  a_ub = block_array([[-big_blind * mask.T, ident],
               [-stake * m_ev.T, ident]], format="csr")
  b_ub = np.zeros(2 * N_HANDS)

  # maximise (sb/N_HANDS) Σ x + (1/N_DEALS) Σ t  ->  minimise its negative
  c = np.concatenate([
    #Uniform reach of player hands
    np.full(N_HANDS, -small_blind / N_HANDS),
    #All values have uniform chance reach in the deal options
    np.full(N_HANDS, -1.0 / N_DEALS),
  ])
  #First strategies (as pbt )
  bounds = [(0.0, 1.0)] * N_HANDS + [(None, None)] * N_HANDS

  res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method=method)
  if not res.success:
    raise RuntimeError(f"LP failed: {res.message}")

  x = np.clip(res.x[:N_HANDS], 0.0, 1.0)
  value = -small_blind - res.fun   # the folded-every-hand baseline, plus the gain

  # Big blind's equilibrium strategy comes from the DUALS, not from reading off
  # which bound is tight. Both give a best response, but where the two branches
  # are equal the big blind is indifferent and equilibrium may require a mixture;
  # a tight-bound readout can only ever return a vertex. Stationarity in t_b
  # forces the two multipliers to sum to t_b's objective coefficient, so
  # normalising them recovers the mixture directly. (Measured at 10 BB: exactly
  # one hand of 1326 is indifferent, where the readout says fold and the true
  # value is ~0.495 — small, but wrong for a reference implementation.)
  duals = -np.asarray(res.ineqlin.marginals)          # >= 0 for '<=' rows
  d_fold, d_call = duals[:N_HANDS], duals[N_HANDS:]
  total = d_fold + d_call
  y = np.divide(d_call, total, out=np.zeros(N_HANDS), where=total > 1e-15)
  return float(value), x, np.clip(y, 0.0, 1.0)


def check_against_cfr(lp_value, br0, br1, tol=1e-3):
  """The LP value must sit inside the best-response sandwich ``br0 >= v >= -br1``."""
  assert br0 >= lp_value - tol, f"BR0 {br0:.6f} below LP value {lp_value:.6f}"
  assert -br1 <= lp_value + tol, f"-BR1 {-br1:.6f} above LP value {lp_value:.6f}"
  return True
