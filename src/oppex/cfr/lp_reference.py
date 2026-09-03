"""Exact Nash solution of the shove-fold game by linear programming.

**Scope — read this before comparing anything to it.** This solves a *matrix
game*: one move each. The small blind either folds or shoves; the big blind, only
if facing a shove, either folds or calls. That is the tree
``build_preflop_tree(..., allow_limp=False)`` produces, and nothing else.

It is therefore **not** a solution of the full FOLD/CALL/ALL_IN preflop game.
That game contains the line

    SB calls (limps) → BB shoves → SB calls or folds

in which the small blind acts *twice*, so its value is not bilinear in any single
vector of per-hand action probabilities and no LP of this shape can express it.
Solving that needs a sequence-form LP over realisation plans, which this is not.

**How the limp is excluded: the action is deleted, not repriced.** With
``allow_limp=False`` the tree builder strikes any CALL whose successor is
non-terminal from the legal set, so at the root ``legal[CALL] = False`` and
``child[CALL] = None``. The limp branch is not assigned a payoff of zero, is not
run to showdown, and is not reachable — it simply does not exist. Restricting the
small blind this way makes the game strictly worse for them, which is why the LP
value falls with stack depth (+0.056 bb at 5 BB, -0.183 bb at 20 BB): the deeper
the stacks, the more the shove-or-fold straitjacket costs.

Consequently this validates the terminal operator, the card-removal mask and the
chance normalisation, but it exercises **no multi-level traversal** — no player
acts twice anywhere in this tree. That part of the solver is currently checked
only by the structural invariants (zero-sum, joint-reach conservation), not
against an exact reference.

*Side pots do not arise.* Every showdown terminal has ``committed == [S, S]``:
with equal starting stacks an all-in call always matches exactly, so the
``showdown_payoffs`` no-side-pot assumption is never exercised. The terminals with
asymmetric ``committed`` are all folds, which pay a plain chip constant.

Derivation. Writing ``x_a`` for P(shove | a) and ``y_b`` for P(call | b), the
value to the small blind is

    (1/N_DEALS) Σ_{a,b} MASK[a,b] [ (1-x_a)(-sb)
                                    + x_a ( (1-y_b) bb + y_b · stake · EV[a,b] ) ]

The big blind acts at most once and their choice for hand ``b`` touches only terms
containing ``b``, so the inner minimisation *decomposes per hand* — this is the
step the limp line would destroy. Each contribution is

    t_b = min( bb · Σ_a MASK[a,b] x_a,        <- big blind folds
               stake · Σ_a MASK[a,b] EV[a,b] x_a )   <- big blind calls

which enters the LP as two ``<=`` constraint blocks; since ``t`` has a positive
objective coefficient the optimum pushes each ``t_b`` up to the min. The fold term
telescopes, because ``Σ_b MASK[a,b] = N_OPP`` for every ``a``:

    (1/N_DEALS) · (-sb · N_OPP · Σ_a (1 - x_a))  =  -sb + (sb/N_HANDS) · Σ_a x_a

which is the constant offset in the returned value and the ``sb/N_HANDS``
objective coefficient. Maximising over ``x`` is then a plain LP in 2652 variables.
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
  m_ev = mask * ev

  # t_b <= bb * Σ_a MASK[a,b] x_a           (big blind folds)
  # t_b <= stake * Σ_a MASK[a,b] EV[a,b] x_a (big blind calls)
  ident = eye(N_HANDS, format="csr")
  a_ub = block_array([[-big_blind * mask.T, ident],
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
