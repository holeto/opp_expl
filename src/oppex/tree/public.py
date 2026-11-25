"""Public states by infoset closure, and the history game as a ``PublicGame``.

**The closure.** A public state is a set of histories closed under "some
player cannot tell these apart": start from one infoset, take all its
histories, add every infoset (of either player) that one of them lies in, and
repeat until nothing new appears. That is the general algorithm and the only
one used to build public states; it needs nothing but the two players'
infoset keys at every node. It yields the common-knowledge partition, which
can be finer than grouping by public observations: two deals that share no
card with each other never become indistinguishable, so with a deal set of
disjoint pairs the closure separates them. ``partition_by_observation`` is the
domain-specific grouping (same ``env.public_state`` bytes: same betting
sequence and board in poker, same results and draw cards in Goofspiel) that
the closure is validated against.

**The conversion.** ``to_public_game`` writes a ``HistoryGame`` in the format
of ``cfr.general``, so the general public-state CFR can solve it. Its public
tree has one node per closure class, plus one terminal state per class
gathering the terminal histories directly below it. A terminal history has no
infoset of its own, so each player's is the pair (infoset of the parent, own
action), which is exactly what they know there. With perfect recall every
history of a child class has its parent in the same class, so the public
tree is a tree.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..cfr.general.public_game import DECISION as P_DECISION
from ..cfr.general.public_game import LEAF as P_LEAF
from ..cfr.general.public_game import TERMINAL as P_TERMINAL
from ..cfr.general.public_game import PublicGame, PublicState
from .history_game import CHANCE, DECISION, LEAF, TERMINAL, HistoryGame


# ── Closure ──────────────────────────────────────────────────────────────────


def _index(game: HistoryGame) -> dict:
  # (player, key) -> nodes, over non-terminal nodes with a key.
  out: dict = {}
  for n in range(game.n_nodes):
    if game.kind[n] == TERMINAL:
      continue
    for p in (0, 1):
      k = game.keys[n][p]
      if k is not None:
        out.setdefault((p, k), []).append(n)
  return out


def public_state_of(game: HistoryGame, player: int, key, index: dict | None = None) -> list[int]:
  """The histories of the public state containing ``player``'s infoset ``key``, sorted."""
  index = _index(game) if index is None else index
  stack, seen, nodes = [(player, key)], {(player, key)}, set()
  while stack:
    p, k = stack.pop()
    for n in index[(p, k)]:
      nodes.add(n)
      # Every infoset this history lies in joins the public state.
      for q in (0, 1):
        kq = game.keys[n][q]
        if kq is not None and (q, kq) not in seen:
          seen.add((q, kq))
          stack.append((q, kq))
  return sorted(nodes)


def public_partition(game: HistoryGame) -> tuple[list[list[int]], np.ndarray]:
  """Closure classes covering every non-terminal node.

  Returns ``(states, of_node)``: the classes, each a sorted node list, in order
  of their first node, and the class of every node (-1 at terminals). A node
  with no key for either player, such as an artificial chance root, is a class
  of its own.
  """
  index = _index(game)
  of_node = np.full(game.n_nodes, -1)
  states: list[list[int]] = []
  for n in range(game.n_nodes):
    if game.kind[n] == TERMINAL or of_node[n] >= 0:
      continue
    k0, k1 = game.keys[n]
    if k0 is None and k1 is None:
      cls = [n]
    else:
      p = 0 if k0 is not None else 1
      cls = public_state_of(game, p, game.keys[n][p], index)
    of_node[cls] = len(states)
    states.append(cls)
  return states, of_node


def partition_by_observation(layered, env, history_game: HistoryGame) -> list[frozenset]:
  """Non-terminal nodes grouped by the bytes of ``env.public_state``.

  ``history_game`` must come from ``from_layered(layered)``, whose ``ref``
  locates each node's env state. A node without a state (an artificial chance
  root) is skipped.
  """
  import jax

  pub = jax.jit(jax.vmap(lambda s: env.public_state(s, jax.random.key(0))))
  per_layer = {}
  for L, lay in enumerate(layered.layers):
    if lay.state_batch is not None:
      per_layer[L] = np.asarray(pub(lay.state_batch))
  groups: dict = {}
  for n in range(history_game.n_nodes):
    L, r = history_game.ref[n]
    if history_game.kind[n] == TERMINAL or L not in per_layer:
      continue
    groups.setdefault((L, per_layer[L][r].tobytes()), set()).add(n)
  return [frozenset(g) for g in groups.values()]


# ── Conversion to a PublicGame ───────────────────────────────────────────────


def _terminal_key(game, z, p):
  # A terminal history's infoset for player p: (parent infoset, own action).
  par = game.parent[z]
  return (game.keys[par][p], game.action[z][p] if game.acting[par][p] else None)


def to_public_game(game: HistoryGame, partition=None, dtype=np.float64) -> PublicGame:
  """``game`` as a ``PublicGame``; the public root holds the game's roots.

  ``ref`` of each public state is ``("P", class)``, an index into the
  partition, or ``("T", class)`` for the terminals below a class.
  """
  states, of_node = public_partition(game) if partition is None else partition
  A = game.n_actions
  cr = game.chance_reach()
  root_cls = {int(of_node[r]) for r in game.roots}
  assert len(root_cls) == 1, f"the roots lie in {len(root_cls)} public states"

  # Public tree, breadth first: a class's child classes, then its terminals.
  items, pos, kids_of = [], {}, {}
  queue = deque([("P", root_cls.pop())])
  while queue:
    item = queue.popleft()
    pos[item] = len(items)
    items.append(item)
    if item[0] == "T":
      continue
    kids, has_term = [], False
    for n in states[item[1]]:
      for c in game.children[n]:
        if game.kind[c] == TERMINAL:
          has_term = True
        elif ("P", int(of_node[c])) not in kids:
          kids.append(("P", int(of_node[c])))
    if has_term:
      kids.append(("T", item[1]))
    kids_of[item] = kids
    queue.extend(kids)

  def members(item):
    # The histories of an item and each one's infoset key per player.
    if item[0] == "P":
      nodes = states[item[1]]
      return nodes, [[game.keys[n][p] for n in nodes] for p in (0, 1)]
    nodes = [c for n in states[item[1]] for c in game.children[n] if game.kind[c] == TERMINAL]
    return nodes, [[_terminal_key(game, z, p) for z in nodes] for p in (0, 1)]

  info = []
  for item in items:
    nodes, ks = members(item)
    idx = []
    for p in (0, 1):
      d: dict = {}
      for k in ks[p]:
        d.setdefault(k, len(d))
      idx.append(d)
    info.append((nodes, ks, idx))

  out = []
  for s, item in enumerate(items):
    nodes, ks, idx = info[s]
    n_inf = (len(idx[0]), len(idx[1]))
    keys = (list(idx[0]), list(idx[1]))
    j0 = np.array([idx[0][k] for k in ks[0]])
    j1 = np.array([idx[1][k] for k in ks[1]])
    if item[0] == "T":
      U = np.zeros(n_inf, dtype)
      np.add.at(U, (j0, j1), cr[nodes] * np.array([game.payoff[z] for z in nodes]))
      out.append(PublicState(P_TERMINAL, n_inf, U=U, keys=keys, ref=item))
      continue
    kinds = {game.kind[n] for n in nodes}
    if kinds == {LEAF}:
      out.append(_leaf_state(game, nodes, n_inf, j0, j1, keys, cr, dtype, item))
      continue
    assert kinds <= {DECISION, CHANCE} and len(kinds) == 1, f"mixed node kinds {kinds}"
    acting = tuple(bool(game.acting[nodes[0]][p]) for p in (0, 1))
    assert all(game.acting[n] == acting for n in nodes), "who acts must be public"
    legal = []
    for p in (0, 1):
      m = np.zeros((n_inf[p], A), bool)
      if acting[p]:
        for n, j in zip(nodes, (j0, j1)[p]):
          m[j] = game.legal[n][p]
      legal.append(m)

    # Children: each child infoset's parent (infoset, own action), flattened.
    node_j = [dict(zip(nodes, (j0, j1)[p])) for p in (0, 1)]
    children = []
    for it in kids_of[item]:
      c = pos[it]
      c_nodes, c_ks, c_idx = info[c]
      maps = [np.full(len(c_idx[p]), -1) for p in (0, 1)]
      for z, k0, k1 in zip(c_nodes, *c_ks):
        par = game.parent[z]
        for p, k in ((0, k0), (1, k1)):
          j = node_j[p][par]
          v = j * A + game.action[z][p] if acting[p] else j
          jc = c_idx[p][k]
          assert maps[p][jc] in (-1, v), "a child infoset with two parents: not perfect recall"
          maps[p][jc] = v
      children.append((c, maps[0], maps[1]))
    out.append(PublicState(P_DECISION, n_inf, acting, tuple(legal), children, keys=keys,
                           ref=item))

  # Root reaches: the players' own reaches of the root histories, per infoset.
  root = out[0]
  rr = []
  for p in (0, 1):
    r = np.ones(root.n_inf[p], dtype)
    for n in info[0][0]:
      if n in game.root_reach:
        r[info[0][2][p][game.keys[n][p]]] = game.root_reach[n][p]
    rr.append(r)
  pg = PublicGame(out, A, tuple(rr))
  pg.check()
  return pg


def _leaf_state(game, nodes, n_inf, j0, j1, keys, cr, dtype, item):
  # Leaf public state: V and W folded with chance reach, continuation masks
  # padded to the widest infoset of each player.
  t_max = [max(len(game.t_legal[(p, k)]) for k in keys[p]) for p in (0, 1)]
  V = np.zeros((t_max[0], t_max[1]) + n_inf, dtype)
  W = np.zeros(n_inf, dtype)
  for n, a, b in zip(nodes, j0, j1):
    v = np.asarray(game.leaf_values[n], dtype)
    V[: v.shape[0], : v.shape[1], a, b] += cr[n] * v
    W[a, b] += cr[n]
  tl = []
  for p in (0, 1):
    m = np.zeros((n_inf[p], t_max[p]), bool)
    for j, k in enumerate(keys[p]):
      t = np.asarray(game.t_legal[(p, k)], bool)
      m[j, : t.size] = t
    tl.append(m)
  return PublicState(P_LEAF, n_inf, V=V, W=W, t_legal=tuple(tl), keys=keys, ref=item)


def strategies_from_rows(pg: PublicGame, sigma, tau=None) -> tuple[dict, dict]:
  """Behaviour strategies keyed by infoset, per player, from ``pscfr`` table rows."""
  from ..cfr.general.pscfr import layout
  lay = layout(pg)
  out = ({}, {})
  for (s, p), lo in lay.rows.items():
    for j, k in enumerate(pg.states[s].keys[p]):
      out[p][k] = np.asarray(sigma[lo + j], np.float64)
  return out


def leaf_rows_by_key(pg: PublicGame) -> dict:
  """(player, key) -> leaf table row, for reading leaf strategies and values."""
  from ..cfr.general.pscfr import layout
  lay = layout(pg)
  out = {}
  for (s, p), lo in lay.leaf_rows.items():
    for j, k in enumerate(pg.states[s].keys[p]):
      out[(p, k)] = lo + j
  return out


def rows_from_strategies(pg: PublicGame, strategies, leaf_tau=None):
  """``pscfr`` table rows from strategies keyed by infoset; missing ones are uniform.

  ``leaf_tau`` maps ``(player, key)`` to a continuation distribution.
  """
  from ..cfr.general.pscfr import layout
  lay = layout(pg)
  sigma = lay.legal / np.maximum(lay.legal.sum(-1, keepdims=True), 1)
  tau = lay.t_legal / np.maximum(lay.t_legal.sum(-1, keepdims=True), 1)
  for (s, p), lo in lay.rows.items():
    for j, k in enumerate(pg.states[s].keys[p]):
      if k in strategies[p]:
        sigma[lo + j] = strategies[p][k]
  for (s, p), lo in lay.leaf_rows.items():
    for j, k in enumerate(pg.states[s].keys[p]):
      if leaf_tau is not None and (p, k) in leaf_tau:
        t = np.asarray(leaf_tau[(p, k)])
        tau[lo + j] = 0.0
        tau[lo + j, : t.size] = t
  return sigma, tau
