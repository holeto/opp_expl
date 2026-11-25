"""The poker public tree written as a ``PublicGame``.

Every state has the same 1326 infosets per player (the hands), maps are the
identity on hands, and the terminal matrices are the poker solver's own:
``const * MASK`` at a fold, ``stake * EV`` at a showdown or checkdown, the leaf
tensor at a matrix leaf with ``W = MASK``. The chance constant ``1 / N_DEALS``
is left out, as in the poker solver, so the two produce identical tables.
"""

from __future__ import annotations

import numpy as np

from ..cards import MASK, N_HANDS
from ..tree import FOLD_T, MATRIX_T, PublicTree
from .public_game import DECISION, LEAF, TERMINAL, PublicGame, PublicState


def from_poker(tree: PublicTree, ev, leaf_values=None, root_reach=None,
               dtype=np.float32) -> PublicGame:
  """States are the decision nodes in tree order, then the terminals.

  ``ref`` of each state is ``("D", node)`` or ``("T", terminal)``.
  """
  H, A = N_HANDS, tree.n_actions
  mask, ev = np.asarray(MASK, dtype), np.asarray(ev, dtype)
  n_nodes = len(tree.nodes)
  hands = list(range(H))
  leaf_id = {j: l for l, j in enumerate(tree.leaves)}
  states = []

  for i, nd in enumerate(tree.nodes):
    acting = (nd.player == 0, nd.player == 1)
    legal = np.broadcast_to(np.asarray(nd.legal), (H, A))
    children = []
    for a, ch in enumerate(nd.child):
      if ch is None:
        continue
      c = ch[1] if ch[0] == "D" else n_nodes + ch[1]
      own, other = np.arange(H) * A + a, np.arange(H)
      children.append((c,) + ((own, other) if nd.player == 0 else (other, own)))
    states.append(PublicState(DECISION, (H, H), acting, (legal, legal), children,
                              keys=(hands, hands), ref=("D", i)))

  for j, t in enumerate(tree.terminals):
    if t.kind == MATRIX_T:
      V = np.asarray(leaf_values[leaf_id[j]], dtype)
      tl = (np.ones((H, V.shape[0]), bool), np.ones((H, V.shape[1]), bool))
      states.append(PublicState(LEAF, (H, H), V=V, W=mask, t_legal=tl,
                                keys=(hands, hands), ref=("T", j)))
    else:
      U = t.const * mask if t.kind == FOLD_T else t.stake * ev
      states.append(PublicState(TERMINAL, (H, H), U=U.astype(dtype),
                                keys=(hands, hands), ref=("T", j)))

  if root_reach is None:
    root_reach = (np.ones(H, dtype), np.ones(H, dtype))
  root_reach = tuple(np.asarray(r, dtype) for r in root_reach)
  # The tree's root need not be node 0 in general; both builders put it there.
  assert tree.root == 0
  game = PublicGame(states, A, root_reach)
  game.check()
  return game


def to_poker_tables(game: PublicGame, gtab, tree: PublicTree):
  """General tables in the poker layout, for comparison.

  Returns ``(regret, strategy_sum, leaf_regret, leaf_strategy_sum,
  leaf_cfv_sum, leaf_mass_sum)`` shaped like ``solver.Tables``; rows of a
  node's non-acting player are zero, as in the poker tables.
  """
  from .pscfr import layout
  lay = layout(game)
  H, A = N_HANDS, game.n_actions
  n_nodes, n_leaves = len(tree.nodes), len(tree.leaves)
  regret = np.zeros((n_nodes, H, A), gtab.regret.dtype)
  ssum = np.zeros_like(regret)
  T = gtab.leaf_regret.shape[1]
  l_reg = np.zeros((2, n_leaves, H, T), gtab.regret.dtype)
  l_ssum, l_cfv, l_mass = np.zeros_like(l_reg), l_reg[..., 0].copy(), l_reg[..., 0].copy()
  leaf_id = {j: l for l, j in enumerate(tree.leaves)}
  for (s, i), lo in lay.rows.items():
    regret[s] = gtab.regret[lo: lo + H]
    ssum[s] = gtab.strategy_sum[lo: lo + H]
  for (s, i), lo in lay.leaf_rows.items():
    l = leaf_id[game.states[s].ref[1]]
    l_reg[i, l] = gtab.leaf_regret[lo: lo + H]
    l_ssum[i, l] = gtab.leaf_strategy_sum[lo: lo + H]
    l_cfv[i, l] = gtab.leaf_cfv_sum[lo: lo + H]
    l_mass[i, l] = gtab.leaf_mass_sum[lo: lo + H]
  return regret, ssum, l_reg, l_ssum, l_cfv, l_mass
