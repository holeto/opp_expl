"""Public-state CFR on a ``PublicGame``.

The same algorithm as ``oppex.cfr.solver``, with every poker shortcut
replaced by the explicit per-state data of ``public_game``:

* reaches go down through the index maps, the acting player's multiplied by
  their own strategy;
* terminals contract the opponent's reach against ``U``, leaves against ``V``
  and the opponent's continuation strategy;
* values go up by summing into the maps. The acting player then weights their
  actions by sigma; the other player's probabilities are already inside their
  reach at the terminals, so their values are plain sums.

Values are counterfactual from the terminals upward, so regrets need no later
multiply, as explained in ``solver``'s docstring. The update rule is
``solver._accumulate`` and the averaging ``solver._accumulate_average``,
applied once per traversal to every infoset of the game packed into one flat
table, so every CFR variant means exactly what it means in the poker solver.

Numpy, not JAX: states differ in shape, and the games this serves are small.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..solver import Discount, _accumulate, _accumulate_average, regret_matching_plus
from .public_game import DECISION, LEAF, TERMINAL, PublicGame


class Tables(NamedTuple):
  """Every infoset of the game in flat tables; ``Layout`` gives the rows."""

  regret: np.ndarray        # (R, A)
  strategy_sum: np.ndarray  # (R, A)
  iters: int
  leaf_regret: np.ndarray   # (RL, T)
  leaf_strategy_sum: np.ndarray
  leaf_cfv_sum: np.ndarray  # (RL,) weighted continuation-weighted leaf cfv
  leaf_mass_sum: np.ndarray  # (RL,) weighted opponent mass, W @ r_opp


class Layout(NamedTuple):
  rows: dict           # (state, player) -> first row, for acting players
  leaf_rows: dict      # (state, player) -> first leaf row
  legal: np.ndarray    # (R, A) bool
  t_legal: np.ndarray  # (RL, T) bool
  row_player: np.ndarray
  leaf_row_player: np.ndarray


def layout(game: PublicGame) -> Layout:
  """Assign table rows to every (state, acting player) and (leaf, player)."""
  if game._layout is not None:
    return game._layout
  A = game.n_actions
  rows, leaf_rows = {}, {}
  legal, t_legal, rp, lrp = [], [], [], []
  r = lr = 0
  t_max = max([st.V.shape[i] for st in game.states if st.kind == LEAF for i in (0, 1)],
              default=1)
  for s, st in enumerate(game.states):
    for i in (0, 1):
      n = st.n_inf[i]
      if st.kind == DECISION and st.acting[i]:
        rows[s, i] = r
        legal.append(st.legal[i])
        rp.append(np.full(n, i))
        r += n
      elif st.kind == LEAF:
        leaf_rows[s, i] = lr
        pad = np.zeros((n, t_max), bool)
        pad[:, : st.t_legal[i].shape[1]] = st.t_legal[i]
        t_legal.append(pad)
        lrp.append(np.full(n, i))
        lr += n
  cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, bool)
  game._layout = Layout(rows, leaf_rows, cat(legal, (0, A)), cat(t_legal, (0, t_max)),
                        np.concatenate(rp) if rp else np.zeros(0, int),
                        np.concatenate(lrp) if lrp else np.zeros(0, int))
  return game._layout


def init_tables(game: PublicGame, dtype=None) -> Tables:
  # In the game's own precision unless told otherwise.
  dtype = game.root_reach[0].dtype if dtype is None else dtype
  lay = layout(game)
  z = lambda shape: np.zeros(shape, dtype)
  return Tables(z(lay.legal.shape), z(lay.legal.shape), 0, z(lay.t_legal.shape),
                z(lay.t_legal.shape), z(lay.t_legal.shape[0]), z(lay.t_legal.shape[0]))


def _rm(regret, legal):
  # Regret matching per row with a per-row legal mask: each row is its own
  # "node" with one hand, which is how the shared function takes it.
  return np.asarray(regret_matching_plus(regret[:, None, :], legal))[:, 0]


def _normalise(ssum, legal):
  # Per-row distribution from a strategy sum, uniform over legal where empty.
  s = np.where(legal, ssum, 0.0)
  tot = s.sum(-1, keepdims=True)
  uniform = legal / np.maximum(legal.sum(-1, keepdims=True), 1)
  return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), uniform).astype(ssum.dtype)


def current_strategy(game, tables):
  lay = layout(game)
  return _rm(tables.regret, lay.legal), _rm(tables.leaf_regret, lay.t_legal)


def average_strategy(game, tables):
  lay = layout(game)
  return (_normalise(tables.strategy_sum, lay.legal),
          _normalise(tables.leaf_strategy_sum, lay.t_legal))


def average_leaf_cfv(tables) -> np.ndarray:
  """Per-unit average leaf value of every leaf infoset, ``(RL,)``; 0 where unreached."""
  m = tables.leaf_mass_sum
  return np.where(m > 0, tables.leaf_cfv_sum / np.where(m > 0, m, 1.0), 0.0)


# ── Traversal ────────────────────────────────────────────────────────────────


class Traversal(NamedTuple):
  reach: list   # per state, [r0, r1]
  cfv: list     # per state, [cfv0, cfv1]
  q: list       # per state, [q0, q1]: (n_i, A) at decisions, (n_i, T) at leaves
  mass: list    # per leaf state, [m0, m1]


def _leaf_q(V, r_opp, tau_opp, player):
  # (n_own, T_own) value of each own continuation: the opponent's reach after
  # their continuation choice, contracted against the leaf tensor.
  w = r_opp[:, None] * tau_opp[:, : V.shape[1 - player]]
  if player == 0:
    return np.einsum("abjk,kb->ja", V, w)
  return -np.einsum("abjk,ja->kb", V, w)


def traverse(game: PublicGame, sigma, tau, root_reach=None) -> Traversal:
  """Reaches down, values up, for the behaviour strategy ``(sigma, tau)`` in table rows."""
  lay = layout(game)
  A, n_s = game.n_actions, len(game.states)
  reach = [None] * n_s
  r0, r1 = game.root_reach if root_reach is None else root_reach
  reach[0] = [np.asarray(r0, sigma.dtype), np.asarray(r1, sigma.dtype)]

  # Down: each child's reach is a gather from its parent's (flat) reach.
  for s, st in enumerate(game.states):
    if st.kind != DECISION:
      continue
    flat = []
    for i in (0, 1):
      r = reach[s][i]
      if st.acting[i]:
        lo = lay.rows[s, i]
        r = (r[:, None] * sigma[lo: lo + st.n_inf[i]]).reshape(-1)
      flat.append(r)
    for c, m0, m1 in st.children:
      reach[c] = [flat[0][m0], flat[1][m1]]

  # Up: terminals and leaves contract, decisions sum into their maps.
  cfv, q, mass = [None] * n_s, [None] * n_s, [None] * n_s
  for s in reversed(range(n_s)):
    st = game.states[s]
    r0, r1 = reach[s]
    if st.kind == TERMINAL:
      cfv[s] = [st.U @ r1, -(st.U.T @ r0)]
    elif st.kind == LEAF:
      t0 = tau[lay.leaf_rows[s, 0]: lay.leaf_rows[s, 0] + st.n_inf[0]]
      t1 = tau[lay.leaf_rows[s, 1]: lay.leaf_rows[s, 1] + st.n_inf[1]]
      q0, q1 = _leaf_q(st.V, r1, t1, 0), _leaf_q(st.V, r0, t0, 1)
      q[s] = [q0, q1]
      cfv[s] = [(q0 * t0[:, : q0.shape[1]]).sum(-1), (q1 * t1[:, : q1.shape[1]]).sum(-1)]
      mass[s] = [st.W @ r1, st.W.T @ r0]
    else:
      out, qs = [], []
      for i in (0, 1):
        n = st.n_inf[i]
        acc = np.zeros(n * A if st.acting[i] else n, sigma.dtype)
        for c, m0, m1 in st.children:
          np.add.at(acc, (m0, m1)[i], cfv[c][i])
        if st.acting[i]:
          lo = lay.rows[s, i]
          qi = acc.reshape(n, A)
          qs.append(qi)
          out.append((qi * sigma[lo: lo + n]).sum(-1))
        else:
          qs.append(None)
          out.append(acc)
      cfv[s], q[s] = out, qs
  return Traversal(reach, cfv, q, mass)


# ── CFR ──────────────────────────────────────────────────────────────────────


def _coef(t, dtype, linear, discount):
  # (weight of this iteration's contribution, DCFR factors or None), as in
  # `solver._cfr_iteration`.
  tf = np.asarray(t, dtype)
  if discount is None:
    return (tf if linear else np.asarray(1.0, dtype)), None
  ta, tb = tf ** discount.alpha, tf ** discount.beta
  return np.asarray(1.0, dtype), (ta / (ta + 1.0), tb / (tb + 1.0),
                                  (tf / (tf + 1.0)) ** discount.gamma)


def cfr_iteration(game, tables: Tables, *, update_player=-1, advance=True, plus=False,
                  linear=False, discount: Discount | None = None) -> Tables:
  """One traversal, writing back ``update_player``'s rows (``-1`` = both)."""
  lay = layout(game)
  sigma, tau = current_strategy(game, tables)
  tv = traverse(game, sigma, tau)
  dtype = tables.regret.dtype
  t = tables.iters + 1
  weight, coef = _coef(t, dtype, linear, discount)

  # Gather every infoset's instantaneous regret and average contribution.
  delta = np.zeros_like(tables.regret)
  contrib = np.zeros_like(tables.regret)
  l_delta = np.zeros_like(tables.leaf_regret)
  l_contrib = np.zeros_like(tables.leaf_regret)
  l_cfv = np.zeros_like(tables.leaf_cfv_sum)
  l_mass = np.zeros_like(tables.leaf_mass_sum)
  for (s, i), lo in lay.rows.items():
    n = game.states[s].n_inf[i]
    sl = slice(lo, lo + n)
    delta[sl] = np.where(lay.legal[sl], tv.q[s][i] - tv.cfv[s][i][:, None], 0.0)
    contrib[sl] = tv.reach[s][i][:, None] * sigma[sl]
  for (s, i), lo in lay.leaf_rows.items():
    n = game.states[s].n_inf[i]
    sl = slice(lo, lo + n)
    qi = tv.q[s][i]
    l_delta[sl, : qi.shape[1]] = qi - tv.cfv[s][i][:, None]
    l_delta[sl] = np.where(lay.t_legal[sl], l_delta[sl], 0.0)
    l_contrib[sl] = tv.reach[s][i][:, None] * tau[sl]
    l_cfv[sl] = tv.cfv[s][i]
    l_mass[sl] = tv.mass[s][i]

  regret, ssum = (np.asarray(x) for x in _accumulate(
    tables.regret[:, None], delta[:, None], tables.strategy_sum[:, None], contrib[:, None],
    weight, plus, discount, coef))
  l_regret, l_ssum = (np.asarray(x) for x in _accumulate(
    tables.leaf_regret[:, None], l_delta[:, None], tables.leaf_strategy_sum[:, None],
    l_contrib[:, None], weight, plus, discount, coef))
  l_cfv = np.asarray(_accumulate_average(tables.leaf_cfv_sum, l_cfv, weight, discount, coef))
  l_mass = np.asarray(_accumulate_average(tables.leaf_mass_sum, l_mass, weight, discount, coef))
  regret, ssum, l_regret, l_ssum = regret[:, 0], ssum[:, 0], l_regret[:, 0], l_ssum[:, 0]

  if update_player != -1:
    # As in the poker solver: rows of the other player keep their old
    # accumulators, not merely un-incremented ones.
    mine, lmine = lay.row_player == update_player, lay.leaf_row_player == update_player
    regret = np.where(mine[:, None], regret, tables.regret)
    ssum = np.where(mine[:, None], ssum, tables.strategy_sum)
    l_regret = np.where(lmine[:, None], l_regret, tables.leaf_regret)
    l_ssum = np.where(lmine[:, None], l_ssum, tables.leaf_strategy_sum)
    l_cfv = np.where(lmine, l_cfv, tables.leaf_cfv_sum)
    l_mass = np.where(lmine, l_mass, tables.leaf_mass_sum)
  return Tables(regret, ssum, t if advance else tables.iters, l_regret, l_ssum, l_cfv, l_mass)


def cfr_step(game, tables: Tables, *, alternating=False, plus=False, linear=False,
             discount: Discount | None = None) -> Tables:
  """One full iteration, with ``solver.cfr_step``'s two schedules."""
  kw = dict(plus=plus, linear=linear, discount=discount)
  if not alternating:
    return cfr_iteration(game, tables, **kw)
  tables = cfr_iteration(game, tables, update_player=0, advance=False, **kw)
  return cfr_iteration(game, tables, update_player=1, **kw)


def solve(game, iters, **scheme) -> Tables:
  tables = init_tables(game)
  for _ in range(iters):
    tables = cfr_step(game, tables, **scheme)
  return tables


# ── Values and best response ─────────────────────────────────────────────────


def root_value(game, sigma, tau) -> tuple[float, float]:
  """(value to player 0, value to player 1), weighted by the root reaches."""
  tv = traverse(game, sigma, tau)
  r0, r1 = tv.reach[0]
  return float(r0 @ tv.cfv[0][0]), float(r1 @ tv.cfv[0][1])


def best_response(game, sigma, tau, br_player) -> tuple[float, list]:
  """Exact best response of ``br_player`` to the other's ``(sigma, tau)``.

  Returns the root value, weighted by the BR player's root reach, and per
  state the BR player's counterfactual values (``None`` where absent). The BR
  player's own probabilities never enter their reach: a pure best response
  exists, so the per-infoset max going up is exact.
  """
  lay = layout(game)
  A, n_s, p, o = game.n_actions, len(game.states), br_player, 1 - br_player
  reach = [None] * n_s
  reach[0] = np.asarray(game.root_reach[o], sigma.dtype)
  for s, st in enumerate(game.states):
    if st.kind != DECISION:
      continue
    r = reach[s]
    if st.acting[o]:
      lo = lay.rows[s, o]
      r = (r[:, None] * sigma[lo: lo + st.n_inf[o]]).reshape(-1)
    for c, m0, m1 in st.children:
      reach[c] = r[(m0, m1)[o]]

  val = [None] * n_s
  for s in reversed(range(n_s)):
    st = game.states[s]
    if st.kind == TERMINAL:
      val[s] = st.U @ reach[s] if p == 0 else -(st.U.T @ reach[s])
    elif st.kind == LEAF:
      lo = lay.leaf_rows[s, o]
      qp = _leaf_q(st.V, reach[s], tau[lo: lo + st.n_inf[o]], p)
      tl = lay.t_legal[lay.leaf_rows[s, p]: lay.leaf_rows[s, p] + st.n_inf[p], : qp.shape[1]]
      val[s] = np.where(tl, qp, -np.inf).max(-1)
    else:
      n = st.n_inf[p]
      acc = np.zeros(n * A if st.acting[p] else n, sigma.dtype)
      for c, m0, m1 in st.children:
        np.add.at(acc, (m0, m1)[p], val[c])
      if st.acting[p]:
        legal = lay.legal[lay.rows[s, p]: lay.rows[s, p] + n]
        acc = np.where(legal, acc.reshape(n, A), -np.inf).max(-1)
      val[s] = acc
  return float(np.asarray(game.root_reach[p], sigma.dtype) @ val[0]), val


def nash_conv(game, sigma, tau) -> float:
  return best_response(game, sigma, tau, 0)[0] + best_response(game, sigma, tau, 1)[0]
