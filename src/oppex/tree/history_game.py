"""A two-player zero-sum game as a flat list of histories.

``LayeredGameTree`` is the extraction format: one layer per BFS wave, sparse
transitions between layers. The solvers, the public-state closure, the
depth-limit cut and the re-solving gadget all want one flat node list they can
rewrite, which is ``HistoryGame``:

* every node has a kind (chance, decision, terminal, leaf), a parent, the
  joint action on its incoming edge and that edge's chance probability;
* every non-terminal node names both players' infosets (``keys``), whoever
  acts. That is what the closure in ``public.py`` walks;
* a decision node may be simultaneous (both players act) or sequential;
* a terminal pays ``payoff`` chips to player 0, the game being zero-sum;
* a leaf is a depth-limit node where both players pick a continuation: it pays
  ``leaf_values[node][t0, t1]`` to player 0, and ``t_legal[(player, key)]``
  says which continuations that infoset has.

A subgame may have several roots. Each root's ``prob`` is its chance reach in
the full game, and ``root_reach[root]`` the two players' own reaches of it,
which is how a subgame starts from ranges.

Children always come after their parent, so node order is topological.
Infoset keys are any hashable: the ``information_set`` bytes for an extracted
game, tagged tuples for nodes a gadget adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np

from .types import NK_CHANCE, NK_TERMINAL, LayeredGameTree

CHANCE, DECISION, TERMINAL, LEAF = 0, 1, 2, 3


@dataclass
class HistoryGame:
  """Flat history-level game; see the module docstring."""

  n_actions: int
  kind: list = field(default_factory=list)          # node kind
  parent: list = field(default_factory=list)        # parent node, -1 at a root
  # Joint action (a0, a1) on the incoming edge, -1 for a player not acting.
  action: list = field(default_factory=list)
  # Chance probability of the incoming edge, 1 on player edges.
  prob: list = field(default_factory=list)
  acting: list = field(default_factory=list)        # (bool, bool)
  keys: list = field(default_factory=list)          # (key0, key1), None at terminals
  legal: list = field(default_factory=list)         # (mask0, mask1), each (A,)
  payoff: list = field(default_factory=list)        # chips to player 0 at terminals
  children: list = field(default_factory=list)
  ref: list = field(default_factory=list)           # where the node came from, for read-back
  leaf_values: dict = field(default_factory=dict)   # node -> (T0, T1) chips to player 0
  t_legal: dict = field(default_factory=dict)       # (player, key) -> (T,) bool
  roots: list = field(default_factory=list)
  root_reach: dict = field(default_factory=dict)    # root -> (r0, r1), 1 if absent

  @property
  def n_nodes(self) -> int:
    return len(self.kind)

  def add(self, kind, parent, action=(-1, -1), prob=1.0, acting=(False, False),
          keys=(None, None), legal=(None, None), payoff=0.0, ref=None) -> int:
    """Append a node and link it under ``parent``; returns its id."""
    i = len(self.kind)
    self.kind.append(kind)
    self.parent.append(parent)
    self.action.append(tuple(int(a) for a in action))
    self.prob.append(float(prob))
    self.acting.append(tuple(bool(x) for x in acting))
    self.keys.append(tuple(keys))
    self.legal.append(legal)
    self.payoff.append(float(payoff))
    self.children.append([])
    self.ref.append(ref)
    if parent < 0:
      self.roots.append(i)
    else:
      self.children[parent].append(i)
    return i

  def chance_reach(self) -> np.ndarray:
    """Product of the chance probabilities from the root to every node."""
    out = np.ones(self.n_nodes)
    for i in range(self.n_nodes):
      p = self.parent[i]
      out[i] = self.prob[i] * (out[p] if p >= 0 else 1.0)
    return out

  def infosets(self, player: int) -> dict[Hashable, list[int]]:
    """Infoset key -> nodes, over the nodes where the key is defined."""
    out: dict = {}
    for i, k in enumerate(self.keys):
      if self.kind[i] != TERMINAL and k[player] is not None:
        out.setdefault(k[player], []).append(i)
    return out

  def check(self) -> None:
    """Topological order, and legality constant across each acting infoset."""
    for i in range(self.n_nodes):
      assert self.parent[i] < i, f"node {i} comes before its parent"
    for p in (0, 1):
      for key, nodes in self.infosets(p).items():
        acting = [n for n in nodes if self.acting[n][p]]
        if acting:
          assert len(acting) == len(nodes), f"player {p} acts at only part of infoset {key!r}"
          masks = {tuple(np.asarray(self.legal[n][p]).tolist()) for n in acting}
          assert len(masks) == 1, f"legal actions differ within infoset {key!r}"


def _key(b: bytes) -> Any:
  # The extractor writes b"" where there is no infoset (chance root, terminals).
  return b if b else None


def from_layered(tree: LayeredGameTree, num_actions: int | None = None) -> HistoryGame:
  """Flatten an extracted tree; ``ref`` of each node is its ``(layer, row)``.

  An artificial chance root keeps the ``None`` keys the extractor gives it;
  ``public.py`` treats such a root as its own public state.
  """
  A = num_actions or tree.layers[0].legal_mask_p0.shape[1]
  if A == 0:
    A = next(L.legal_mask_p0.shape[1] for L in tree.layers if L.legal_mask_p0.shape[1])
  g = HistoryGame(A)
  ids: dict[tuple[int, int], int] = {}

  def add_row(L, r, parent, action, prob):
    lay = tree.layers[L]
    nk = int(lay.node_kind[r])
    ref = (L, r)
    if nk == NK_TERMINAL:
      assert abs(float(lay.reward_p0[r]) + float(lay.reward_p1[r])) < 1e-6, "not zero-sum"
      return g.add(TERMINAL, parent, action, prob, payoff=float(lay.reward_p0[r]), ref=ref)
    keys = (_key(lay.info_set_id_p0[r]), _key(lay.info_set_id_p1[r]))
    if nk == NK_CHANCE:
      return g.add(CHANCE, parent, action, prob, keys=keys, ref=ref)
    cp = int(lay.acting_player[r])
    acting = (True, True) if cp < 0 else (cp == 0, cp == 1)
    legal = (np.asarray(lay.legal_mask_p0[r], bool), np.asarray(lay.legal_mask_p1[r], bool))
    return g.add(DECISION, parent, action, prob, acting, keys, legal, ref=ref)

  ids[(0, 0)] = add_row(0, 0, -1, (-1, -1), 1.0)
  for L, tr in enumerate(tree.transitions):
    for e in range(tr.parent_idx.shape[0]):
      p = ids[(L, int(tr.parent_idx[e]))]
      c = (L + 1, int(tr.child_idx[e]))
      ids[c] = add_row(c[0], c[1], p, (tr.action_p0[e], tr.action_p1[e]), tr.edge_prob[e])
  g.check()
  return g


def strategy_dicts(game: HistoryGame, uniform: bool = True) -> tuple[dict, dict]:
  """Uniform behaviour strategies over legal actions, keyed by infoset, per player."""
  out = ({}, {})
  for i in range(game.n_nodes):
    for p in (0, 1):
      if game.kind[i] == DECISION and game.acting[i][p] and game.keys[i][p] not in out[p]:
        m = np.asarray(game.legal[i][p], float)
        out[p][game.keys[i][p]] = m / m.sum() if uniform else m
  return out


def expected_value(game: HistoryGame, strategies, leaf_tau=None) -> np.ndarray:
  """Value to player 0 of every node under ``strategies`` (one dict per player).

  Leaves pay ``tau0ᵀ V tau1`` with ``leaf_tau[p][key]`` the continuation
  strategy of each infoset, uniform over its legal continuations if absent.
  The game's value is the chance- and root-reach-weighted sum over its roots.
  """
  v = np.zeros(game.n_nodes)

  def leaf_mix(n, p):
    k = game.keys[n][p]
    m = np.asarray(game.t_legal[(p, k)], float)
    t = None if leaf_tau is None else leaf_tau[p].get(k)
    return m / m.sum() if t is None else np.asarray(t, float)[: m.size]

  for n in reversed(range(game.n_nodes)):
    k = game.kind[n]
    if k == TERMINAL:
      v[n] = game.payoff[n]
    elif k == LEAF:
      V = np.asarray(game.leaf_values[n], float)
      t0, t1 = leaf_mix(n, 0), leaf_mix(n, 1)
      v[n] = t0[: V.shape[0]] @ V @ t1[: V.shape[1]]
    else:
      tot = 0.0
      for c in game.children[n]:
        w = game.prob[c]
        for p in (0, 1):
          if game.acting[n][p]:
            w *= strategies[p][game.keys[n][p]][game.action[c][p]]
        tot += w * v[c]
      v[n] = tot
  return v


def reach_probabilities(game: HistoryGame, strategies) -> np.ndarray:
  """``(N, 3)`` reach of every node: player 0's, player 1's and chance's own factor.

  Root nodes start from ``root_reach`` and their chance ``prob``. Infosets
  missing from ``strategies`` are played uniformly.
  """
  r = np.zeros((game.n_nodes, 3))
  for n in range(game.n_nodes):
    par = game.parent[n]
    if par < 0:
      r[n, :2] = game.root_reach.get(n, (1.0, 1.0))
      r[n, 2] = game.prob[n]
      continue
    r[n] = r[par]
    r[n, 2] *= game.prob[n]
    for p in (0, 1):
      if game.acting[par][p]:
        s = strategies[p].get(game.keys[par][p])
        if s is None:
          m = np.asarray(game.legal[par][p], float)
          s = m / m.sum()
        r[n, p] *= s[game.action[n][p]]
  return r
