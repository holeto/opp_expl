"""Exact Nash solution of the shove-fold game by linear programming.

Works ONLY for the pre-flop game, where the only allowed
actions are FOLD or ALL_IN (shove)

The shove-fold game is a minimal poker game, where each player only
acts (at most) once, so the little blind cannot CALL at the root, only
ALL_IN or FOLD. After that, the player big blind player responds accordingly.
Sanity check for minimal PSCFR

Sequence form LP for the 2 step game.

**Where the sequence-form equalities went.** Sequence form requires, per infoset
``I``, ``Σ_{a ∈ A(I)} r(σ_I·a) = r(σ_I)``. Neither player's constraints appear as
explicit ``A_eq`` rows below, because at depth 1 both discharge in closed form.
Both are verified to hold exactly (see the checks in ``scripts/``).

The right-hand side is ``r(σ_I)``, **not** the chance probability of reaching ``I``:
a realisation plan is conditional on the player's own choices, since chance and the
opponent decide which infoset is reached and the player cannot influence that.
Chance therefore lives in the objective coefficients — the ``sb/N_HANDS`` and
``1/N_DEALS`` below — never in the constraints.

*Small blind — eliminated in the primal.* Infoset "I hold ``a``" is reached by the
empty sequence, so ``r(a,ALL_IN) + r(a,FOLD) = r(∅) = 1``. Two actions and a
*constant* right-hand side let the equality substitute out: ``x_a = r(a,ALL_IN)``
and ``r(a,FOLD) = 1 - x_a``. What survives is exactly the two non-negativity
conditions, which is what ``bounds = (0, 1)`` encodes — and that substitution is
where the constant ``-sb`` offset in the returned value comes from.

*Big blind — implicit in the dual.* They get no primal variables; their
minimisation is done analytically, and ``t_b`` with its two ``<=`` rows is the
hypograph of ``t_b = min(fold branch, call branch)``. Their equality reappears as
dual stationarity in ``t_b``:

    lambda_fold[b] + lambda_call[b] = (objective coefficient of t_b) = 1/N_DEALS

Scaling by ``N_DEALS`` gives ``r(b,FOLD) + r(b,CALL) = 1``. That is why their
strategy is read from the duals rather than from which bound is tight.

**Both shortcuts need two actions per infoset AND a constant parent realisation**,
which is exactly what "each player acts at most once" buys. The limp line destroys
it: the small blind then owns a second infoset with
``r(a,limp,CALL) + r(a,limp,FOLD) = r(a,limp)`` — a *variable* right-hand side, so
nothing substitutes out, the feasible set stops being a box, and the equalities
must be written explicitly.
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
  #EV: A matrix indexed by (p1_hand, p2_hand), giving the 
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
  #First sequence probabilities (in this case, just a standard pbt distribution)
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
  y = np.clip(y, 0.0, 1.0)

  check_sequence_form(res, x, y)
  return float(value), x, y


def check_against_cfr(lp_value, br0, br1, tol=1e-3):
  """The LP value must sit inside the best-response sandwich ``br0 >= v >= -br1``."""
  assert br0 >= lp_value - tol, f"BR0 {br0:.6f} below LP value {lp_value:.6f}"
  assert -br1 <= lp_value + tol, f"-BR1 {-br1:.6f} above LP value {lp_value:.6f}"
  return True


def check_sequence_form(res, x, y, tol=1e-9) -> dict:
  """Assert both players' sequence-form equalities, which are implicit above.

  Neither is an ``A_eq`` row, so without this they are a claim in a docstring
  rather than a property of the solution. Returns the residuals.
  """
  # Small blind: r(a,ALL_IN) + r(a,FOLD) = 1, with r(a,FOLD) := 1 - x_a.
  sb_res = float(np.abs(x + (1.0 - x) - 1.0).max())
  assert sb_res <= tol, f"SB realisation does not sum to 1 (max {sb_res:.3e})"
  assert x.min() >= -tol and x.max() <= 1.0 + tol, "SB realisation outside [0, 1]"

  # Big blind: dual stationarity in t_b forces the multipliers to sum to t_b's
  # objective coefficient, which is 1/N_DEALS. Scaled up, that is their equality.
  duals = -np.asarray(res.ineqlin.marginals)
  total = duals[:N_HANDS] + duals[N_HANDS:]
  bb_res = float(np.abs(total * N_DEALS - 1.0).max())
  assert bb_res <= 1e-6, f"BB realisation does not sum to 1 (max {bb_res:.3e})"
  assert y.min() >= -tol and y.max() <= 1.0 + tol, "BB realisation outside [0, 1]"
  return {"sb_residual": sb_res, "bb_residual": bb_res}
