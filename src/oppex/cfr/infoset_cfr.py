"""CFR for a single own hand against a fixed opponent: the re-solving subgame.

Against a fixed opponent the subgame rooted at our infoset is a decision
problem for us. The opponent's hand enters only through their counterfactual
reach ``r_o(z, h)``: their blueprint probability of reaching ``z`` holding
``h``, times card compatibility with our hand and the board. Our own strategy
does not change it. So every terminal contracts to one number before solving,

    u(z) = Σ_h r_o(z, h) · payoff(z, h),

and every matrix leaf to one number per own continuation ``t``. What is left
is public-state CFR with a hand axis of size one. The cfv of our node is
sigma-weighted, the cfv of an opponent node is a plain sum (their probabilities
are inside ``r_o`` already), and regrets need no reach weighting because the
values are counterfactual. The update rule is ``solver._accumulate``, shared
with the public-state solver, so the CFR variants mean the same thing here.

Values are unnormalised: dividing by ``Σ_h r_o(root, h)`` turns them into
expected chips given the root belief. ``best_response`` is the exact solution
(backward induction), against which the CFR average is checked.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .solver import DCFR, Discount, _accumulate, regret_matching_plus
from .tree import PublicTree


class InfosetGame(NamedTuple):
  """A subgame with every payoff contracted against the opponent's reach."""

  tree: PublicTree
  player: int              # our seat
  term_value: np.ndarray   # (n_terminals,) chips to us; ignored at matrix leaves
  leaf_value: np.ndarray   # (n_leaves, T) chips to us per own continuation


class _Layout(NamedTuple):
  levels: list             # node ids by depth
  src: np.ndarray          # (n_nodes, A) index into [node values; terminal values; 0]
  legal: np.ndarray        # (n_nodes, A) bool
  mine: np.ndarray         # (n_nodes,) bool
  parent: np.ndarray       # (n_nodes,) parent node, -1 at the root
  parent_atom: np.ndarray  # (n_nodes,)
  leaf_parent: np.ndarray  # (n_leaves,) node the leaf hangs off
  leaf_atom: np.ndarray


def _layout(game: InfosetGame) -> _Layout:
  tree = game.tree
  n, n_term, n_act = len(tree.nodes), len(tree.terminals), tree.n_actions
  src = np.full((n, n_act), n + n_term, np.int64)
  parent = np.full(n, -1, np.int64)
  parent_atom = np.zeros(n, np.int64)
  term_parent = np.zeros((n_term, 2), np.int64)
  levels, frontier = [], [tree.root]
  while frontier:
    levels.append(np.array(frontier))
    nxt = []
    for i in frontier:
      for a, ch in enumerate(tree.nodes[i].child):
        if ch is None:
          continue
        if ch[0] == "D":
          src[i, a], parent[ch[1]], parent_atom[ch[1]] = ch[1], i, a
          nxt.append(ch[1])
        else:
          src[i, a] = n + ch[1]
          term_parent[ch[1]] = i, a
    frontier = nxt
  leaves = np.array(tree.leaves, np.int64)
  return _Layout(
    levels, src, np.array([nd.legal for nd in tree.nodes]),
    np.array([nd.player == game.player for nd in tree.nodes]), parent, parent_atom,
    term_parent[leaves, 0] if leaves.size else np.zeros(0, np.int64),
    term_parent[leaves, 1] if leaves.size else np.zeros(0, np.int64),
  )


def _values(game, lay, weight, leaf_v):
  # Bottom-up values with per-edge weights `weight` (n_nodes, A); returns
  # (node values, child values per edge).
  n = len(game.tree.nodes)
  term = np.asarray(game.term_value, np.float64).copy()
  if len(game.tree.leaves):
    term[list(game.tree.leaves)] = leaf_v
  vals = np.zeros(n + len(term) + 1)
  vals[n: n + len(term)] = term
  for lvl in reversed(lay.levels):
    ch = vals[lay.src[lvl]]
    vals[lvl] = (np.where(lay.legal[lvl], weight[lvl], 0.0) * ch).sum(-1)
  return vals[:n], vals[lay.src]


def _edge_weights(lay, sigma):
  # Our nodes weight their branches by sigma, the opponent's sum them.
  return np.where(lay.mine[:, None], sigma, 1.0)


def _own_reach(lay, sigma):
  # Our reach of each node: the product of our own probabilities above it.
  reach = np.ones(len(lay.mine))
  for lvl in lay.levels[1:]:
    p, a = lay.parent[lvl], lay.parent_atom[lvl]
    reach[lvl] = reach[p] * np.where(lay.mine[p], sigma[p, a], 1.0)
  return reach


def policy_value(game: InfosetGame, sigma, tau) -> float:
  """Root value of our behaviour strategy ``sigma`` (n_nodes, A) and ``tau``."""
  lay = _layout(game)
  leaf_v = (np.asarray(game.leaf_value) * tau).sum(-1)
  return float(_values(game, lay, _edge_weights(lay, sigma), leaf_v)[0][game.tree.root])


def best_response(game: InfosetGame) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """Exact best response: (root value, per-node values, action per node, t per leaf).

  Our nodes take the best legal child and our leaves the best continuation;
  opponent nodes sum. The actions are only meaningful at our own nodes.
  """
  lay = _layout(game)
  tree = game.tree
  n = len(tree.nodes)
  leaf_v = np.asarray(game.leaf_value, np.float64)
  term = np.asarray(game.term_value, np.float64).copy()
  if len(tree.leaves):
    term[list(tree.leaves)] = leaf_v.max(-1)
  vals = np.zeros(n + len(term) + 1)
  vals[n: n + len(term)] = term
  act = np.zeros(n, np.int64)
  for lvl in reversed(lay.levels):
    ch = vals[lay.src[lvl]]
    legal = lay.legal[lvl]
    best = np.where(legal, ch, -np.inf)
    act[lvl] = best.argmax(-1)
    vals[lvl] = np.where(lay.mine[lvl], best.max(-1), np.where(legal, ch, 0.0).sum(-1))
  t_best = leaf_v.argmax(-1) if leaf_v.size else np.zeros(0, np.int64)
  return float(vals[tree.root]), vals[:n], act, t_best


def solve(
  game: InfosetGame, iters: int = 500, *, plus: bool = False, linear: bool = False,
  discount: Discount | None = DCFR,
) -> tuple[np.ndarray, np.ndarray]:
  """CFR for our seat only: the average ``sigma`` (n_nodes, A) and ``tau`` (n_leaves, T).

  The opponent's rows of ``sigma`` are uniform over legal actions; they are
  never played, the opponent's strategy is inside the contracted values.
  """
  lay = _layout(game)
  n, n_act = lay.src.shape
  leaf_v = np.asarray(game.leaf_value, np.float64)
  n_leaves, n_t = leaf_v.shape
  # Accumulators in float32, the precision of the shared JAX update rule.
  regret = np.zeros((n, 1, n_act), np.float32)
  ssum = np.zeros((n, 1, n_act), np.float32)
  leaf_regret = np.zeros((n_leaves, 1, n_t), np.float32)
  leaf_ssum = np.zeros((n_leaves, 1, n_t), np.float32)
  legal_leaf = np.ones((n_leaves, n_t), bool)

  for it in range(1, iters + 1):
    sigma = np.asarray(regret_matching_plus(regret, lay.legal))[:, 0]
    tau = np.asarray(regret_matching_plus(leaf_regret, legal_leaf))[:, 0]
    v, ch = _values(game, lay, _edge_weights(lay, sigma), (leaf_v * tau).sum(-1))
    reach = _own_reach(lay, sigma)
    leaf_reach = reach[lay.leaf_parent] * np.where(
      lay.mine[lay.leaf_parent], sigma[lay.leaf_parent, lay.leaf_atom], 1.0)

    if discount is None:
      weight, coef = (float(it) if linear else 1.0), None
    else:
      ta, tb = it ** discount.alpha, it ** discount.beta
      weight, coef = 1.0, (ta / (ta + 1.0), tb / (tb + 1.0),
                           (it / (it + 1.0)) ** discount.gamma)
    delta = np.where(lay.legal & lay.mine[:, None], ch - v[:, None], 0.0)[:, None, :]
    new_r, new_s = _accumulate(regret, delta, ssum, (reach[:, None] * sigma)[:, None, :],
                               weight, plus, discount, coef)
    # Only our nodes learn; the opponent's rows keep their zeros.
    mine = lay.mine[:, None, None]
    regret = np.where(mine, np.asarray(new_r), regret)
    ssum = np.where(mine, np.asarray(new_s), ssum)
    leaf_delta = (leaf_v - (leaf_v * tau).sum(-1, keepdims=True))[:, None, :]
    leaf_regret, leaf_ssum = (np.asarray(x) for x in _accumulate(
      leaf_regret, leaf_delta, leaf_ssum, (leaf_reach[:, None] * tau)[:, None, :],
      weight, plus, discount, coef))

  return _average(ssum[:, 0], lay.legal), _average(leaf_ssum[:, 0], legal_leaf)


def _average(ssum, legal):
  # Normalised strategy sum, uniform over legal actions where nothing accrued.
  s = np.where(legal, ssum, 0.0)
  tot = s.sum(-1, keepdims=True)
  uniform = legal / np.maximum(legal.sum(-1, keepdims=True), 1)
  return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), uniform)
