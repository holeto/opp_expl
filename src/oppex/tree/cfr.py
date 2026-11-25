"""Vanilla CFR over histories, with regrets per infoset, on a ``HistoryGame``.

The textbook form, as the reference for the public-state solver: reaches go
down every history, expected values come up, and each infoset's regret is
the sum over its histories of ``π_{-i}(h) · (q_i(h, a) - v_i(h))``, with
``π_{-i}`` the chance and opponent reach. It shares ``solver._accumulate``
and the flat ``Tables`` of ``cfr.general.pscfr``, so the two solvers can be
compared table for table, keyed by infoset.

Everything is vectorised over histories: two passes over depth levels (reach
down, value up) and scatter-adds for the rest. Leaves pay ``τ0ᵀ V τ1`` and keep
continuation regrets per leaf infoset, like the public-state leaves.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..cfr.general.pscfr import Tables, _coef, _normalise, _rm
from ..cfr.solver import Discount, _accumulate, _accumulate_average
from .history_game import DECISION, LEAF, TERMINAL, HistoryGame


class Layout(NamedTuple):
  rows: dict           # (player, key) -> table row, acting infosets
  leaf_rows: dict      # (player, key) -> leaf table row
  legal: np.ndarray    # (R, A)
  t_legal: np.ndarray  # (RL, T)
  row_player: np.ndarray
  leaf_row_player: np.ndarray
  levels: list         # node ids by depth
  parent: np.ndarray
  action: np.ndarray   # (N, 2)
  prob: np.ndarray
  node_row: np.ndarray       # (N, 2) row of the node's infoset if that player acts, else -1
  node_leaf_row: np.ndarray  # (N, 2) leaf row at leaves, else -1
  payoff: np.ndarray
  V: np.ndarray        # (N, T, T) leaf values, zero padded
  roots: np.ndarray
  root_reach: np.ndarray  # (n_roots, 2)
  rep: np.ndarray      # (R,) one node of every acting infoset
  leaf_rep: np.ndarray
  inner: np.ndarray    # (N,) bool, decision and chance nodes


def layout(game: HistoryGame) -> Layout:
  """Rows for every infoset, and the game as flat arrays."""
  cached = getattr(game, "_cfr_layout", None)
  if cached is not None:
    return cached
  N, A = game.n_nodes, game.n_actions
  rows, leaf_rows, legal, t_legal, rp, lrp, rep, lrep = {}, {}, [], [], [], [], [], []
  node_row = np.full((N, 2), -1)
  node_leaf_row = np.full((N, 2), -1)
  t_max = max([len(m) for m in game.t_legal.values()], default=1)
  for n in range(N):
    for p in (0, 1):
      k = game.keys[n][p]
      if game.kind[n] == DECISION and game.acting[n][p]:
        if (p, k) not in rows:
          rows[(p, k)] = len(legal)
          legal.append(np.asarray(game.legal[n][p], bool))
          rp.append(p)
          rep.append(n)
        node_row[n, p] = rows[(p, k)]
      elif game.kind[n] == LEAF:
        if (p, k) not in leaf_rows:
          leaf_rows[(p, k)] = len(t_legal)
          m = np.zeros(t_max, bool)
          t = np.asarray(game.t_legal[(p, k)], bool)
          m[: t.size] = t
          t_legal.append(m)
          lrp.append(p)
          lrep.append(n)
        node_leaf_row[n, p] = leaf_rows[(p, k)]

  depth = np.zeros(N, int)
  for n in range(N):
    if game.parent[n] >= 0:
      depth[n] = depth[game.parent[n]] + 1
  levels = [np.flatnonzero(depth == d) for d in range(depth.max() + 1)]
  V = np.zeros((N, t_max, t_max))
  for n, v in game.leaf_values.items():
    v = np.asarray(v)
    V[n, : v.shape[0], : v.shape[1]] = v
  roots = np.array(game.roots)
  rr = np.array([game.root_reach.get(r, (1.0, 1.0)) for r in game.roots], float)

  lay = Layout(
    rows, leaf_rows,
    np.array(legal, bool).reshape(-1, A), np.array(t_legal, bool).reshape(-1, t_max),
    np.array(rp, int), np.array(lrp, int), levels, np.array(game.parent),
    np.array(game.action), np.array(game.prob), node_row, node_leaf_row,
    np.array(game.payoff), V, roots, rr, np.array(rep, int), np.array(lrep, int),
    np.array([k not in (TERMINAL, LEAF) for k in game.kind]),
  )
  game._cfr_layout = lay
  return lay


def init_tables(game: HistoryGame, dtype=np.float64) -> Tables:
  lay = layout(game)
  z = lambda shape: np.zeros(shape, dtype)
  return Tables(z(lay.legal.shape), z(lay.legal.shape), 0, z(lay.t_legal.shape),
                z(lay.t_legal.shape), z(lay.t_legal.shape[0]), z(lay.t_legal.shape[0]))


def current_strategy(game, tables):
  lay = layout(game)
  return _rm(tables.regret, lay.legal), _rm(tables.leaf_regret, lay.t_legal)


def average_strategy(game, tables):
  lay = layout(game)
  return (_normalise(tables.strategy_sum, lay.legal),
          _normalise(tables.leaf_strategy_sum, lay.t_legal))


class Traversal(NamedTuple):
  reach: np.ndarray   # (N, 3): player 0, player 1, chance
  value: np.ndarray   # (N,) expected value to player 0
  edge: np.ndarray    # (N, 2) the parent's probability of the node's action, per player
  leaf_q: np.ndarray  # (N, 2, T) value of each own continuation to its player, at leaves


def traverse(game: HistoryGame, sigma, tau) -> Traversal:
  """Reaches down and values up over every history, for table-row strategies."""
  lay = layout(game)
  N = game.n_nodes
  par, act = lay.parent, lay.action

  # Each node's own-action probability at its parent, per player.
  edge = np.ones((N, 2))
  child = par >= 0
  for p in (0, 1):
    row = np.where(child, lay.node_row[np.maximum(par, 0), p], -1)
    acts = row >= 0
    edge[acts, p] = sigma[row[acts], act[acts, p]]

  reach = np.zeros((N, 3))
  reach[lay.roots, :2] = lay.root_reach
  reach[lay.roots, 2] = lay.prob[lay.roots]
  for lvl in lay.levels[1:]:
    reach[lvl] = reach[par[lvl]] * np.stack([edge[lvl, 0], edge[lvl, 1], lay.prob[lvl]], 1)

  # Leaf values from both continuation strategies.
  leaf = np.flatnonzero(lay.node_leaf_row[:, 0] >= 0)
  value = lay.payoff.astype(float).copy()
  leaf_q = np.zeros((N, 2, lay.t_legal.shape[1]))
  if leaf.size:
    t0, t1 = tau[lay.node_leaf_row[leaf, 0]], tau[lay.node_leaf_row[leaf, 1]]
    V = lay.V[leaf]
    leaf_q[leaf, 0] = np.einsum("nab,nb->na", V, t1)
    leaf_q[leaf, 1] = -np.einsum("nab,na->nb", V, t0)
    value[leaf] = (leaf_q[leaf, 0] * t0).sum(-1)

  # Up: a node is the probability-weighted sum of its children.
  w = edge[:, 0] * edge[:, 1] * lay.prob
  inner = lay.inner
  for lvl in reversed(lay.levels[1:]):
    acc = np.zeros(N)
    np.add.at(acc, par[lvl], w[lvl] * value[lvl])
    ps = np.unique(par[lvl])
    value[ps] = np.where(inner[ps], acc[ps], value[ps])
  return Traversal(reach, value, edge, leaf_q)


def cfr_iteration(game, tables: Tables, *, update_player=-1, advance=True, plus=False,
                  linear=False, discount: Discount | None = None) -> Tables:
  """One traversal, writing back ``update_player``'s rows (``-1`` = both)."""
  lay = layout(game)
  sigma, tau = current_strategy(game, tables)
  tv = traverse(game, sigma, tau)
  N, A = game.n_nodes, game.n_actions
  par, act = lay.parent, lay.action
  dtype = tables.regret.dtype
  t = tables.iters + 1
  weight, coef = _coef(t, dtype, linear, discount)

  delta = np.zeros_like(tables.regret)
  l_delta = np.zeros_like(tables.leaf_regret)
  l_cfv = np.zeros_like(tables.leaf_cfv_sum)
  l_mass = np.zeros_like(tables.leaf_mass_sum)
  child = np.flatnonzero(par >= 0)
  for p in (0, 1):
    sign = 1.0 if p == 0 else -1.0
    # Counterfactual reach: chance times the opponent.
    cf = tv.reach[:, 2] * tv.reach[:, 1 - p]

    # q_p(n, a): the value of playing a at n, the opponent's and chance's
    # probabilities on each edge included, p's own left out.
    pn = par[child]
    acting = lay.node_row[pn, p] >= 0
    c, pn = child[acting], pn[acting]
    Q = np.zeros((N, A))
    np.add.at(Q, (pn, act[c, p]), lay.prob[c] * tv.edge[c, 1 - p] * sign * tv.value[c])
    nodes = np.flatnonzero(lay.node_row[:, p] >= 0)
    rows = lay.node_row[nodes, p]
    d = cf[nodes, None] * (Q[nodes] - sign * tv.value[nodes, None])
    np.add.at(delta, rows, np.where(lay.legal[rows], d, 0.0))

    # Leaves: continuation regrets, and the cfv and mass for the averages.
    leaves = np.flatnonzero(lay.node_leaf_row[:, p] >= 0)
    lrows = lay.node_leaf_row[leaves, p]
    v_p = sign * tv.value[leaves]
    ld = cf[leaves, None] * (tv.leaf_q[leaves, p] - v_p[:, None])
    np.add.at(l_delta, lrows, np.where(lay.t_legal[lrows], ld, 0.0))
    np.add.at(l_cfv, lrows, cf[leaves] * v_p)
    np.add.at(l_mass, lrows, cf[leaves])

  # The average weights by own reach, which is one number per infoset.
  own = np.array([tv.reach[n, p] for n, p in zip(lay.rep, lay.row_player)]).reshape(-1)
  contrib = own[:, None] * sigma
  l_own = np.array([tv.reach[n, p] for n, p in zip(lay.leaf_rep, lay.leaf_row_player)])
  l_contrib = l_own.reshape(-1)[:, None] * tau

  regret, ssum = (np.asarray(x)[:, 0] for x in _accumulate(
    tables.regret[:, None], delta[:, None], tables.strategy_sum[:, None], contrib[:, None],
    weight, plus, discount, coef))
  l_regret, l_ssum = (np.asarray(x)[:, 0] for x in _accumulate(
    tables.leaf_regret[:, None], l_delta[:, None], tables.leaf_strategy_sum[:, None],
    l_contrib[:, None], weight, plus, discount, coef))
  l_cfv = np.asarray(_accumulate_average(tables.leaf_cfv_sum, l_cfv, weight, discount, coef))
  l_mass = np.asarray(_accumulate_average(tables.leaf_mass_sum, l_mass, weight, discount, coef))

  if update_player != -1:
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


def strategy_dicts_from_rows(game, sigma) -> tuple[dict, dict]:
  """Behaviour strategies keyed by infoset, per player, from table rows."""
  lay = layout(game)
  out = ({}, {})
  for (p, k), r in lay.rows.items():
    out[p][k] = np.asarray(sigma[r], np.float64)
  return out
