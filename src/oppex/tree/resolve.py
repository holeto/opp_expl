"""Solving history games and checking a re-solve's safety, with either solver.

The re-solving scripts (``scripts/resolve_goofspiel.py``,
``scripts/resolve_leduc.py``) solve depth-limited subgames and Resolve
gadgets built by ``subgame.py``. ``solve_game`` runs the general PSCFR
(through the public-state closure) or vanilla history CFR on such a game
and returns everything in infoset keys, so the two are interchangeable.

``step_disagreement`` checks that they are: from the same tables, both must
take the same step. Comparing two free-running solves instead does not work.
CFR amplifies a 1e-16 difference in summation order (on the Leduc root game
from 1e-15 after 10 iterations to 3e-4 after 100), not only at exact ties.
"""

from __future__ import annotations

import numpy as np

from ..cfr.general import pscfr
from . import cfr as vcfr
from .history_game import HistoryGame
from .public import leaf_rows_by_key, rows_from_strategies, strategies_from_rows, to_public_game
from .subgame import resolve_gadget


def solve_game(game: HistoryGame, iters: int, use_pscfr: bool, scheme: dict, fixed=None):
  """Solve ``game`` for ``iters`` iterations of ``scheme``.

  Returns ``(strategies, values, leaf_tau)``: the average behaviour strategy
  per player as ``{key: probs}``, and per leaf infoset ``(player, key)`` the
  per-unit average leaf value (``average_leaf_cfv``) and the average
  continuation mixture. ``fixed`` (``{(player, key): probs}``, PSCFR only)
  freezes those infosets, as for the known model in RNR.
  """
  assert use_pscfr or not fixed, "frozen rows are implemented for PSCFR only"
  if use_pscfr:
    pg = to_public_game(game)
    tab = pscfr.solve(pg, iters, fixed=fixed, **scheme)
    sig, tau = pscfr.average_strategy(pg, tab)
    rows = leaf_rows_by_key(pg)
    strat = strategies_from_rows(pg, sig)
    alt = pscfr.average_leaf_cfv(tab)
  else:
    tab = vcfr.solve(game, iters, **scheme)
    sig, tau = vcfr.average_strategy(game, tab)
    rows = vcfr.layout(game).leaf_rows
    strat = vcfr.strategy_dicts_from_rows(game, sig)
    m = tab.leaf_mass_sum
    alt = np.where(m > 0, tab.leaf_cfv_sum / np.where(m > 0, m, 1.0), 0.0)
  values = {k: alt[r] for k, r in rows.items()}
  leaf_tau = {k: tau[r] for k, r in rows.items()}
  return strat, values, leaf_tau


# ── Cross-checking the solvers ───────────────────────────────────────────────


def _row_maps(game: HistoryGame, pg):
  # PSCFR row -> vanilla row, for the node and the leaf tables.
  lay_p, lay_v = pscfr.layout(pg), vcfr.layout(game)
  perm = np.zeros(lay_p.legal.shape[0], int)
  for (s, p), lo in lay_p.rows.items():
    for j, k in enumerate(pg.states[s].keys[p]):
      perm[lo + j] = lay_v.rows[(p, k)]
  lperm = np.zeros(lay_p.t_legal.shape[0], int)
  for (s, p), lo in lay_p.leaf_rows.items():
    for j, k in enumerate(pg.states[s].keys[p]):
      lperm[lo + j] = lay_v.leaf_rows[(p, k)]
  return perm, lperm


def _to_vanilla(tab, perm, lperm):
  inv, linv = np.argsort(perm), np.argsort(lperm)
  return tab._replace(regret=tab.regret[inv], strategy_sum=tab.strategy_sum[inv],
                      leaf_regret=tab.leaf_regret[linv],
                      leaf_strategy_sum=tab.leaf_strategy_sum[linv],
                      leaf_cfv_sum=tab.leaf_cfv_sum[linv], leaf_mass_sum=tab.leaf_mass_sum[linv])


def step_disagreement(game: HistoryGame, iters: int, scheme: dict, pg=None) -> float:
  """Largest relative difference of vanilla CFR's step from PSCFR's along a run.

  Before each of ``iters`` PSCFR steps, its tables are handed to vanilla CFR
  (rows matched by infoset key), and the two next tables are compared:
  regrets and strategy sums, of the nodes and of the leaves.
  """
  pg = to_public_game(game) if pg is None else pg
  perm, lperm = _row_maps(game, pg)
  tab = pscfr.init_tables(pg)
  worst = 0.0
  for _ in range(iters):
    nxt = pscfr.cfr_step(pg, tab, **scheme)
    got = vcfr.cfr_step(game, _to_vanilla(tab, perm, lperm), **scheme)
    want = _to_vanilla(nxt, perm, lperm)
    for f in ("regret", "strategy_sum", "leaf_regret", "leaf_strategy_sum"):
      a, b = getattr(got, f), getattr(want, f)
      if b.size:
        worst = max(worst, np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
    tab = nxt
  return worst


# ── Safety ───────────────────────────────────────────────────────────────────


def opponent_values(sub: HistoryGame, p: int, pi: dict, strat, leaf_tau) -> dict:
  """The opponent's best-response value per unit of reach at each root infoset of ``sub``.

  The opponent best-responds to ``p``'s strategy in ``sub``, the roots
  weighted by ``pi`` as in the gadget (without the TERMINATE option). Against
  the parent solve's strategy these are counterfactual best-response values
  (CBVs), the alternative values for which Resolve is safe.
  """
  o = 1 - p
  g = to_public_game(resolve_gadget(sub, p, pi, {}, with_gadget=False))
  sig, tau = rows_from_strategies(g, strat, leaf_tau)
  _, val = pscfr.best_response(g, sig, tau, o)
  root = g.states[0].children[0][0]
  st = g.states[root]
  # The root public state's mass per opponent infoset: chance times p's reach.
  mass = np.zeros(st.n_inf[o])
  z = sum(pi[r] for r in sub.roots if pi[r] > 0)
  for r in sub.roots:
    if pi[r] > 0:
      mass[st.keys[o].index(sub.keys[r][o])] += pi[r] / z
  return {k: val[root][j] / mass[j] for j, k in enumerate(st.keys[o])}


def gadget_safety(sub: HistoryGame, p: int, pi: dict, alt: dict, strat, leaf_tau) -> float:
  """The largest gain of the opponent over its alternative values.

  ``opponent_values`` against ``p``'s re-solved strategy, minus ``alt``, at
  each root infoset. At most 0, up to the solve's accuracy, when the re-solve
  is safe.
  """
  v = opponent_values(sub, p, pi, strat, leaf_tau)
  return max(v[k] - alt[k] for k in v)
