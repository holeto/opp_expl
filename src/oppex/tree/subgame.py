"""Depth-limited subgames, exact normal-form leaves, and the Resolve gadget.

**The cut.** ``cut_subgame`` copies the histories of one public state and
``depth`` plies below it, or down to the nodes a ``stop`` predicate marks
(the start of a betting round, whose length varies). Each root keeps its
chance reach in the full game and the players' own reaches, so the subgame
starts from ranges. Non-terminal histories at the depth limit become leaves.

**Exact leaves.** At a leaf, each player chooses a continuation, and the
continuations offered are *all* reduced pure strategies of that player below
their leaf infoset. With perfect recall the player's infosets below a leaf
infoset ``J`` are reached only through ``J``, so a pure strategy there is a
local object, and by Kuhn's theorem a mixture over them at each ``J`` is any
behaviour strategy below. So the depth-limited game is strategically
equivalent to the full game, and no deviation of the opponent below the limit
is missing. With a single fixed continuation instead
(``fixed_profile_leaves``), the opponent cannot deviate below the limit, and
a depth-limited equilibrium can be exploitable in the full game.

**The gadget** (Burch et al. 2014, "Resolve"). To re-solve a public state for
player ``p`` without trusting the opponent ``o`` to keep playing what the
previous solve assumed, a chance node picks each root history ``h`` with
probability ``π_{-o}(h) / Σ π_{-o}``, and then ``o`` decides, per root infoset
``J``, between FOLLOW (play the subgame from ``h``) and TERMINATE (take the
alternative value ``alt(J)`` per unit of reach, from the previous solve). So
``p`` must keep ``o``'s counterfactual best-response value at every ``J`` down
to ``alt(J)``, which is what makes re-solving safe. ``π_{-o}`` includes ``p``'s
own reach: that scales ``p``'s values below each root infoset by a constant,
to which regret matching and the reach-weighted average are invariant.
"""

from __future__ import annotations

import itertools

import numpy as np

from .history_game import CHANCE, DECISION, LEAF, TERMINAL, HistoryGame, expected_value

FOLLOW, TERMINATE = 0, 1


# ── Cutting ──────────────────────────────────────────────────────────────────


def _copy_node(src: HistoryGame, n: int, dst: HistoryGame, parent: int, prob: float,
               kind=None) -> int:
  # Append a copy of src's node n under dst's `parent`; `ref` is n.
  return dst.add(src.kind[n] if kind is None else kind, parent, src.action[n], prob,
                 src.acting[n], src.keys[n], src.legal[n], src.payoff[n], ref=n)


def copy_below(src: HistoryGame, root: int, dst: HistoryGame, parent: int, prob: float,
               depth: int | None, stop=None) -> int:
  """Copy ``root`` and ``depth`` plies below it under ``parent``; returns the copy.

  Non-terminal nodes at the depth limit, or for which ``stop(node)`` holds
  (``root`` included), become leaves; ``ref`` points back to the source node.
  Leaves of ``src`` are copied with their values.
  """
  kind = src.kind[root]
  at_limit = depth == 0 or (stop is not None and stop(root))
  if at_limit and kind not in (TERMINAL, LEAF):
    return _copy_node(src, root, dst, parent, prob, LEAF)
  new = _copy_node(src, root, dst, parent, prob)
  if kind == LEAF:
    dst.leaf_values[new] = src.leaf_values[root]
    for p in (0, 1):
      dst.t_legal[(p, src.keys[root][p])] = src.t_legal[(p, src.keys[root][p])]
  for c in src.children[root]:
    copy_below(src, c, dst, new, src.prob[c], None if depth is None else depth - 1, stop)
  return new


def cut_subgame(full: HistoryGame, roots, depth: int | None, own_reach=None,
                chance=None, stop=None) -> HistoryGame:
  """The subgame below the histories ``roots`` (one public state), ``depth`` plies deep.

  ``stop(node)`` of ``full`` also makes a node a leaf. ``own_reach[h]`` is
  ``(π0(h), π1(h))`` and ``chance[h]`` the chance reach of each root in
  ``full``, both 1 if omitted. Leaves have no values yet; see
  ``normal_form_leaves``.
  """
  sub = HistoryGame(full.n_actions)
  for h in roots:
    c = 1.0 if chance is None else chance[h]
    new = copy_below(full, h, sub, -1, c, depth, stop)
    if own_reach is not None:
      sub.root_reach[new] = tuple(float(x) for x in own_reach[h])
  sub.check()
  return sub


# ── Leaf values ──────────────────────────────────────────────────────────────


def _first_infosets(full: HistoryGame, nodes, p: int) -> list:
  # Player p's acting infosets met first below `nodes` (the nodes themselves
  # included), in order of discovery.
  out, seen, stack = [], set(), list(nodes)[::-1]
  while stack:
    n = stack.pop()
    if full.kind[n] == TERMINAL:
      continue
    if full.kind[n] == DECISION and full.acting[n][p]:
      k = full.keys[n][p]
      if k not in seen:
        seen.add(k)
        out.append(k)
      continue
    stack.extend(full.children[n][::-1])
  return out


def reduced_pure_strategies(full: HistoryGame, nodes, p: int) -> list[dict]:
  """Player ``p``'s reduced pure strategies below ``nodes``, as ``{key: action}``.

  Reduced: an infoset that the strategy's own earlier choices rule out gets
  no action, so equivalent strategies are listed once.
  """
  index = full.infosets(p)

  def of_set(keys):
    parts = [of_infoset(k) for k in keys]
    return [dict(itertools.chain.from_iterable(d.items() for d in combo))
            for combo in itertools.product(*parts)]

  def of_infoset(k):
    nodes_k = index[k]
    out = []
    for a in np.flatnonzero(full.legal[nodes_k[0]][p]):
      below = [c for n in nodes_k for c in full.children[n] if full.action[c][p] == a]
      for rest in of_set(_first_infosets(full, below, p)):
        out.append({k: int(a), **rest})
    return out

  return of_set(_first_infosets(full, nodes, p))


def _pure_value(full: HistoryGame, n: int, s0: dict, s1: dict) -> float:
  # Value to player 0 of the subtree at n under two pure strategies.
  k = full.kind[n]
  if k == TERMINAL:
    return full.payoff[n]
  tot = 0.0
  for c in full.children[n]:
    ok = True
    for p, s in ((0, s0), (1, s1)):
      if full.acting[n][p] and s[full.keys[n][p]] != full.action[c][p]:
        ok = False
    if ok:
      tot += full.prob[c] * _pure_value(full, c, s0, s1)
  return tot


def normal_form_leaves(full: HistoryGame, sub: HistoryGame) -> dict:
  """Give every leaf of ``sub`` the exact normal-form continuation matrix.

  The continuations of player ``p`` at leaf infoset ``J`` are
  ``reduced_pure_strategies`` of the full game below the histories of ``J``.
  Every leaf ``h`` gets ``V[t0, t1]``, the full game's value below ``h`` under
  that pair, payoffs already collected included. Returns
  ``{(p, key): [strategy dicts]}`` so a continuation can be read back.
  """
  leaves = [n for n in range(sub.n_nodes) if sub.kind[n] == LEAF]
  conts = {}
  for p in (0, 1):
    index = full.infosets(p)
    for k in dict.fromkeys(sub.keys[n][p] for n in leaves):
      # All of the infoset's histories in the full game: it lies in one
      # public state, so the subgame holds them all anyway.
      conts[(p, k)] = reduced_pure_strategies(full, index[k], p)
      sub.t_legal[(p, k)] = np.ones(len(conts[(p, k)]), bool)
  for n in leaves:
    c0, c1 = conts[(0, sub.keys[n][0])], conts[(1, sub.keys[n][1])]
    sub.leaf_values[n] = np.array([[_pure_value(full, sub.ref[n], s0, s1) for s1 in c1]
                                   for s0 in c0])
  return conts


def fixed_profile_leaves(full: HistoryGame, sub: HistoryGame, strategies) -> None:
  """One continuation per player: the full game's value below each leaf under ``strategies``."""
  v = expected_value(full, strategies)
  for n in range(sub.n_nodes):
    if sub.kind[n] == LEAF:
      sub.leaf_values[n] = np.array([[v[sub.ref[n]]]])
      for p in (0, 1):
        sub.t_legal[(p, sub.keys[n][p])] = np.ones(1, bool)


def behaviour_from_tau(full: HistoryGame, conts: dict, tau_by_key: dict, p: int) -> dict:
  """Player ``p``'s behaviour strategy below the leaves, induced by mixing ``tau``.

  Kuhn's construction: at infoset ``I`` below leaf infoset ``J``, action ``a``
  gets the tau mass of the continuations that define ``I`` and play ``a`` there,
  normalised by the mass of those that define ``I`` at all.
  """
  num: dict = {}
  for (q, k), strats in conts.items():
    if q != p:
      continue
    tau = np.asarray(tau_by_key[(p, k)])
    for t, s in enumerate(strats):
      for key, a in s.items():
        num.setdefault(key, np.zeros(full.n_actions))[a] += tau[t]
  out = {}
  for key, v in num.items():
    out[key] = v / v.sum() if v.sum() > 0 else v + 1.0 / v.size
  return out


# ── The Resolve gadget ───────────────────────────────────────────────────────


def resolve_gadget(sub: HistoryGame, resolver: int, pi_minus_o: dict, alt: dict,
                   with_gadget: bool = True) -> HistoryGame:
  """The Resolve gadget around the subgame ``sub`` for player ``resolver``.

  ``pi_minus_o[root]`` is the chance-times-resolver reach of each root of
  ``sub`` in the full game, and ``alt[key]`` the opponent's alternative value
  per unit of reach at each of their root infosets. Roots with zero reach are
  dropped. TERMINATE pays ``alt`` to the opponent.

  ``with_gadget=False`` keeps the chance root but no decision, for measuring
  the opponent's best response to the re-solved strategy at the root.
  """
  o = 1 - resolver
  roots = [r for r in sub.roots if pi_minus_o[r] > 0]
  z = sum(pi_minus_o[r] for r in roots)
  g = HistoryGame(sub.n_actions)
  top = g.add(CHANCE, -1, keys=(None, None), ref=None)
  legal_o = np.zeros(sub.n_actions, bool)
  legal_o[[FOLLOW, TERMINATE]] = True
  sign = 1.0 if o == 0 else -1.0
  for r in roots:
    prob = pi_minus_o[r] / z
    if not with_gadget:
      copy_below(sub, r, g, top, prob, None)
      continue
    keys = [None, None]
    keys[o] = ("gadget", sub.keys[r][o])
    keys[resolver] = ("gadget", sub.keys[r][resolver])
    legal = [None, None]
    legal[o] = legal_o
    acting = (o == 0, o == 1)
    gate = g.add(DECISION, top, prob=prob, acting=acting, keys=tuple(keys), legal=tuple(legal),
                 ref=("gadget", r))
    act = [-1, -1]
    act[o] = TERMINATE
    g.add(TERMINAL, gate, tuple(act), payoff=sign * alt[sub.keys[r][o]], ref=("terminate", r))
    follow = copy_below(sub, r, g, gate, 1.0, None)
    a = [-1, -1]
    a[o] = FOLLOW
    g.action[follow] = tuple(a)
  g.check()
  return g
