"""Restricted Nash responses (RNR) on history games, for depth-limited re-solving.

RNR (Johanson, Zinkevich and Bowling 2007): a chance node at the root picks a
known opponent model with probability ``p`` or a free opponent with
probability ``1 - p``. The exploiter does not see which, so its infosets are
shared by both branches; the opponent's are not, since the model and the free
opponent are different agents. The exploiter's value is
``p·u(σ, model) + (1 - p)·min_o u(σ, o)``.

**Representation** (``rnr_game``). A history game is copied twice under one
chance root, flat over (branch, deal) so the public tree stays a tree. In the
known copy the opponent's keys become ``("known", key)``, and its decision
rows are frozen at the model (``pscfr.fix_rows``). The model's moves cannot be
chance nodes: they share public states with the free opponent's decisions,
and who acts must be public. At a depth-limit leaf the model has one
continuation, its own play below, so a known leaf's values are one column:
each of the exploiter's reduced pure continuations against the model. The
exploiter's leaf keys are shared, so one continuation mixture serves both
branches.

**Re-solving** (``rnr_gadget``). The re-solve of a public state holds the
known copy of *all* its histories. The gadget couples the exploiter's
infosets through the opponent's, so the known terms of every one of them
matter, not only those of the infoset being played. Known roots are weighted
``p·c·π_i·π_fix``; the unknown ones enter the Resolve gadget with weight
``(1 - p)·c·π_i`` (variant B) or that times the free opponent's parent reach
(variant A), with one normalisation over both branches. Under B the gadget
value lower-bounds the RNR objective, so with CBV alternative values a
re-solve cannot fall below its parent (``notes/rnr_resolve_derivation.md``).
Known roots pass through a node where the opponent's only action is FOLLOW,
so both branches enter the subgame at the same depth.
"""

from __future__ import annotations

import numpy as np

from .best_response import best_response_values
from .history_game import CHANCE, DECISION, LEAF, TERMINAL, HistoryGame, expected_value
from .subgame import FOLLOW, TERMINATE, copy_below, resolve_gadget

KNOWN = "known"


def is_known(key) -> bool:
  """Whether ``key`` is an opponent key of the known copy (the model's)."""
  return isinstance(key, tuple) and len(key) == 2 and key[0] == KNOWN


def _tag_known(g: HistoryGame, start: int, o: int) -> None:
  # The opponent's keys of nodes start.. become the known copy's.
  for n in range(start, g.n_nodes):
    k = list(g.keys[n])
    if k[o] is not None:
      k[o] = (KNOWN, k[o])
      g.keys[n] = tuple(k)


def _fixed_rows(g: HistoryGame, o: int, model: dict) -> dict:
  # The model's frozen rows: every known-copy decision of the opponent.
  out = {}
  for n in range(g.n_nodes):
    k = g.keys[n][o]
    if g.kind[n] == DECISION and g.acting[n][o] and is_known(k):
      out[(o, k)] = np.asarray(model[k[1]], float)
  return out


def _value_vs_model(full: HistoryGame, n: int, s: dict, i: int, model: dict) -> float:
  # Value to player 0 below full's node n, player i playing the pure
  # strategy s and the opponent the behaviour strategy model.
  if full.kind[n] == TERMINAL:
    return full.payoff[n]
  o, tot = 1 - i, 0.0
  for c in full.children[n]:
    if full.acting[n][i] and s[full.keys[n][i]] != full.action[c][i]:
      continue
    w = full.prob[c]
    if full.acting[n][o]:
      w *= model[full.keys[n][o]][full.action[c][o]]
    if w:
      tot += w * _value_vs_model(full, c, s, i, model)
  return tot


def rnr_game(game: HistoryGame, exploiter: int, p: float, model: dict, full=None,
             conts=None) -> tuple[HistoryGame, dict]:
  """``game`` as an RNR game for ``exploiter``; returns ``(game, fixed rows)``.

  ``game`` must start with one keyless chance node (the deal). ``model`` maps
  the opponent's keys to the model's behaviour. If ``game`` has leaves (a
  depth-limited cut of ``full`` with ``normal_form_leaves`` giving ``conts``),
  the known copy's leaves get the exploiter's continuations against the
  model. With ``p == 0`` the game is returned unchanged.
  """
  if p == 0:
    return game, {}
  o = 1 - exploiter
  (root,) = game.roots
  assert game.kind[root] == CHANCE and game.keys[root] == (None, None)
  g = HistoryGame(game.n_actions)
  top = g.add(CHANCE, -1, keys=(None, None))
  for known, w in ((True, p), (False, 1.0 - p)):
    if w == 0:
      continue
    for c in game.children[root]:
      start = g.n_nodes
      copy_below(game, c, g, top, w * game.prob[c], None)
      if known:
        _tag_known(g, start, o)

  # Known leaves: one model continuation, the exploiter's pure ones against it.
  for n in range(g.n_nodes):
    if g.kind[n] == LEAF and is_known(g.keys[n][o]):
      fn = game.ref[g.ref[n]]
      col = np.array([_value_vs_model(full, fn, s, exploiter, model)
                      for s in conts[(exploiter, g.keys[n][exploiter])]])
      g.leaf_values[n] = col[:, None] if exploiter == 0 else col[None, :]
      g.t_legal[(o, g.keys[n][o])] = np.ones(1, bool)
  g.check()
  return g, _fixed_rows(g, o, model)


def rnr_gadget(sub: HistoryGame, exploiter: int, w_known: dict, w_unknown: dict, alt: dict,
               model: dict) -> tuple[HistoryGame, dict]:
  """The RNR re-solving game of the subgame ``sub``; returns ``(game, fixed rows)``.

  ``w_known[root]`` and ``w_unknown[root]`` are each root's weight in the two
  branches (``p·c·π_i·π_fix`` and ``(1 - p)·c·π_i``, or the latter times the
  free opponent's parent reach); roots with zero weight are dropped. The
  unknown roots get the Resolve gadget with ``alt`` per opponent infoset.
  Without known weight this is exactly ``resolve_gadget``.
  """
  o = 1 - exploiter
  if not any(w > 0 for w in w_known.values()):
    return resolve_gadget(sub, exploiter, w_unknown, alt), {}
  z = sum(w for w in (*w_known.values(), *w_unknown.values()) if w > 0)
  g = HistoryGame(sub.n_actions)
  top = g.add(CHANCE, -1, keys=(None, None))
  gate_legal = {False: np.zeros(sub.n_actions, bool), True: np.zeros(sub.n_actions, bool)}
  gate_legal[False][[FOLLOW, TERMINATE]] = True
  gate_legal[True][FOLLOW] = True
  acting = (o == 0, o == 1)
  sign = 1.0 if o == 0 else -1.0
  for known, weights in ((False, w_unknown), (True, w_known)):
    for r in sub.roots:
      if weights[r] <= 0:
        continue
      keys, legal = [None, None], [None, None]
      keys[exploiter] = ("gadget", sub.keys[r][exploiter])
      keys[o] = (KNOWN, "gadget", sub.keys[r][o]) if known else ("gadget", sub.keys[r][o])
      legal[o] = gate_legal[known]
      gate = g.add(DECISION, top, prob=weights[r] / z, acting=acting, keys=tuple(keys),
                   legal=tuple(legal), ref=("gadget", known, r))
      if not known:
        act = [-1, -1]
        act[o] = TERMINATE
        g.add(TERMINAL, gate, tuple(act), payoff=sign * alt[sub.keys[r][o]],
              ref=("terminate", r))
      start = g.n_nodes
      follow = copy_below(sub, r, g, gate, 1.0, None)
      a = [-1, -1]
      a[o] = FOLLOW
      g.action[follow] = tuple(a)
      if known:
        _tag_known(g, start, o)
  g.check()
  return g, _fixed_rows(g, o, model)


# ── Evaluation ───────────────────────────────────────────────────────────────


def seat_metrics(layered, full: HistoryGame, ours: dict, model: dict, i: int, p: float,
                 v0_star: float) -> dict:
  """How ``ours`` does in seat ``i`` of the full game, against the model and the worst case.

  Returns the value against the model, the worst-case value (the opponent
  best-responds), the exploitability of ``ours`` alone (the seat's game value
  ``v_i*`` minus the worst case; ``v0_star`` is player 0's), the RNR
  objective ``p·vs_model + (1 - p)·worst``, and the best-response value
  against the model, the most any strategy wins from it in this seat.
  """
  prof = (ours, model) if i == 0 else (model, ours)
  sign = 1.0 if i == 0 else -1.0
  vs_model = sign * expected_value(full, prof)[full.roots[0]]
  bv = best_response_values(layered, *prof)
  worst = bv.v0_p1_br if i == 0 else bv.v1_p0_br
  br = bv.v0_p0_br if i == 0 else bv.v1_p1_br
  return dict(vs_model=vs_model, worst=worst, expl=sign * v0_star - worst,
              F=p * vs_model + (1.0 - p) * worst, br_vs_model=br)
