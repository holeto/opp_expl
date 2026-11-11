"""Slumbot as an opponent model on the env's action space.

``SlumbotModel`` has the interface of ``resolving.OpponentModel``:
``probs(paths, our_cards, board, our_seat)`` gives, after each env history,
Slumbot's probabilities over env atoms for all 1326 of its hands, and
``belief`` the reach of the history. Slumbot itself (``native.py``) speaks
slumbot.com's language, so every env history is translated into it.

**Chips.** The env plays 1/2 blinds with 400 stacks, Slumbot 50/100 with
20 000, so env chips × 50 are Slumbot chips, exactly. Our env bets therefore
become exact ``b<N>`` tokens. Slumbot's bets are sized against its own
abstract pot and need not hit an env bin: each is played in the env as the
legal raise atom whose street total is nearest in chips (an all-in stays an
all-in). That is the one approximation, and ``stats["bet_error"]`` measures it
as a fraction of the env pot.

**Slumbot's own past bets keep their exact chips** in the strings we send it,
not the env-rounded ones; otherwise it would misread its own line. The env
history only records the atom, so the chips are recovered by asking Slumbot
which of its actions map to that atom. When our own raise is no longer legal
in Slumbot's chips (its exact bet was larger than the env's), it is raised to
Slumbot's minimum raise.

**Translation types.** Slumbot translates our off-tree bets at random, with a
type ``u ~ U[0, 1)`` per hand that fixes every draw (``slumbot_api.h``). The
model is the mixture over ``u``, so it is the policy slumbot.com plays over
its hand numbers. A history is walked as a set of *worlds*. A world is a set
of ``u``, the slumbot.com string Slumbot saw for those ``u`` and the per-hand
likelihood ``L_w(h)`` of Slumbot's past actions in it. At each Slumbot
decision a world is split by the translation cells of its string and by the
distinct chip amounts that map to the observed atom. Then

  π(a | path, h) = Σ_w |w| L_w(h) π_w(a | h) / Σ_w |w| L_w(h),

whose product along a history is the exact joint Σ_w |w| L_w(h), so beliefs
and subgame reaches built from these conditionals are exact. With a fixed
``u`` (one live hand number) the model is a single deterministic Slumbot.

**Observed strings.** In the env only atoms are seen, so a history keeps
every chip amount that maps to its atoms. On slumbot.com Slumbot's exact
chips are seen too: ``set_real`` gives the real string after each step of
the real history, and the worlds on that history are cut to the one string.
That is Bayes on the exact tokens, P(u | string, h). Below the real history
(inside a subgame) worlds split by chips as before. The belief is then the
joint Σ_w |w| L_w(h) of the cut worlds, not the product of the atom
conditionals along the history: once the worlds are cut to a token, the
product of atom probabilities no longer telescopes to it.

Walks are memoised per history prefix, so the many paths of one subgame
share the queries of their common prefixes.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

import numpy as np

import sys

from ...cfr.cards import HAND_CARDS, N_HANDS
from ...cfr.subgame import expand_node
from ...cfr.tree import FOLD_T, SHOWDOWN_T
from ...envs import betting
from ...envs.betting import ALL_IN, CALL, FOLD
from . import native
from .api import BIG_BLIND, STACK_SIZE
from .play import RealGame

_VISIBLE = (0, 3, 4, 5)
_KIND_TOKEN = {native.SB_FOLD: "f", native.SB_CHECK: "k", native.SB_CALL: "c"}


class _Step(NamedTuple):
  # The env side of one history prefix: the state and its expansion.
  bs: betting.BettingState
  legal: list
  alias: list
  children: list


class _World(NamedTuple):
  ivs: tuple         # ((lo, hi), ...): the translation types of this world
  actions: str       # the slumbot.com string so far
  game: RealGame     # the hand in Slumbot's chips
  lik: np.ndarray    # (H,) probability of Slumbot's past actions per hand


class _Mapped(NamedTuple):
  # Slumbot's options at one state, mapped to env atoms and to tokens.
  env_probs: np.ndarray  # (H, A)
  tokens: list           # token strings
  token_atom: np.ndarray  # (T,) env atom of each token
  token_probs: np.ndarray  # (H, T)


def _measure(ivs) -> float:
  return float(sum(hi - lo for lo, hi in ivs))


def _intersect(a, b) -> tuple:
  # Both sorted lists of disjoint intervals.
  out, i, j = [], 0, 0
  while i < len(a) and j < len(b):
    lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
    if lo < hi:
      out.append((lo, hi))
    if a[i][1] < b[j][1]:
      i += 1
    else:
      j += 1
  return tuple(out)


def _clone(g: RealGame) -> RealGame:
  g2 = copy.copy(g)
  g2.committed, g2.street_committed = list(g.committed), list(g.street_committed)
  return g2


def _apply(world: _World, token: str, lik=None) -> _World | None:
  # The world after one token, with '/' when the street ends; None if the
  # token is illegal in Slumbot's chips.
  g = _clone(world.game)
  street = g.street
  try:
    g.apply(token[0], int(token[1:]) if token[0] == "b" else None)
  except AssertionError:
    return None
  sep = "/" if not g.done and g.street != street else ""
  return _World(world.ivs, world.actions + token + sep, g,
                world.lik if lik is None else lik)


def our_token(atom: int, total: float, street_max: float, g: RealGame, seat: int,
              scale: int) -> tuple[str, bool]:
  """Our env atom as a slumbot.com token in Slumbot's chips.

  ``total`` is the atom's street total in env chips and ``street_max`` the
  highest street total before it. A raise keeps its env street total × scale,
  moved into Slumbot's legal range; returns the token and whether it moved.
  """
  if atom == FOLD:
    return "f", False
  if total <= street_max + 1e-9:
    return ("c" if g.call_amount(seat) > 0 else "k"), False
  all_in = g.street_committed[seat] + g.stack(seat)
  if atom == ALL_IN:
    return f"b{all_in}", False
  n = int(round(total * scale))
  lo = max(g.street_committed) + g.min_raise()
  return f"b{min(max(n, lo), all_in)}", n < lo


def slumbot_atom(kind: int, bet: int, totals, raise_ok, g: RealGame, seat: int,
                 scale: int) -> tuple[int, float]:
  """The env atom of one Slumbot action, and its street-total gap in env chips.

  ``totals`` ``(A,)`` are the env street totals of the atoms and ``raise_ok``
  marks the legal atoms that raise (struck duplicates excluded). A bet is the
  raise atom whose total is nearest, the lowest on ties; Slumbot's all-in
  stays all-in, which is a call when calling already commits the stack.
  """
  if kind == native.SB_FOLD:
    return FOLD, 0.0
  if kind != native.SB_BET:
    return CALL, 0.0
  if bet >= g.street_committed[seat] + g.stack(seat):
    return (ALL_IN if raise_ok[ALL_IN] else CALL), 0.0
  target = bet / scale
  best, best_d = CALL, np.inf
  for a in range(ALL_IN, len(totals)):
    if raise_ok[a]:
      d = abs(float(totals[a]) - target)
      if d < best_d - 1e-9:
        best, best_d = a, d
  return best, (best_d if best != CALL else 0.0)


class SlumbotModel:
  """Slumbot's env-atom probabilities for every one of its hands at once.

  ``u=None`` marginalises the translation type (slumbot.com over its hand
  numbers), a float fixes it (``native.get().u_from_hand_no(n)`` is hand
  ``n``). ``rules`` are the env's betting rules; their geometry must be
  Slumbot's 200 BB scaled down.
  """

  def __init__(self, rules: betting.BettingRules, u: float | None = None,
               max_cache: int = 50_000) -> None:
    scale = STACK_SIZE / rules.starting_stack
    if (rules.big_blind * scale != BIG_BLIND or rules.small_blind * 2 != rules.big_blind
        or scale != int(scale)):
      raise ValueError("the env's blinds and stacks are not Slumbot's 50/100/20000 scaled")
    self.rules, self.scale, self.u = rules, int(scale), u
    self.n_actions = betting.num_atoms(rules)
    self.lib = native.get()
    self.max_cache = max_cache
    self._steps: dict = {}
    self._worlds: dict = {}
    self._queries: dict = {}
    self._cells: dict = {}
    self.stats = self._new_stats()
    self.new_hand()

  @staticmethod
  def _new_stats() -> dict:
    # failures: (actions, status) of queries Slumbot refused, answered with a
    # check/call instead. bet_error: |env total - Slumbot total| / env pot per
    # mapped Slumbot bet. worlds: number of worlds per probs() row.
    # real_lost: hands where no world reproduced the real string.
    return {"queries": 0, "cells": 0, "failures": [], "bet_error": [], "raise_clamped": 0,
            "desync": 0, "worlds": [], "real_lost": 0}

  def reset(self) -> None:
    """Drops the caches (not the statistics)."""
    self._steps.clear()
    self._worlds.clear()
    self._queries.clear()
    self._cells.clear()

  def new_hand(self) -> None:
    """Forgets the real strings of the previous hand."""
    self._real_path: tuple = ()
    self._real_strings: tuple = ()
    self._real_lost = False

  def set_real(self, path, strings) -> None:
    """The real history so far: env atoms and the slumbot.com string after each.

    Worlds on a prefix of ``path`` keep only the matching string. Ignored for
    the rest of the hand once no world reproduced a real string.
    """
    if len(path) != len(strings):
      raise ValueError(f"{len(path)} atoms but {len(strings)} strings")
    if not self._real_lost:
      self._real_path = tuple(int(a) for a in path)
      self._real_strings = tuple(strings)

  # ── Public interface ──────────────────────────────────────────────────────

  def probs(self, paths, our_cards, board, our_seat) -> np.ndarray:
    """``(len(paths), N_HANDS, A)`` env-atom probabilities after each path.

    Each path must end at a decision of Slumbot's (seat ``1 - our_seat``).
    ``board`` is the visible board; our cards play no part (card removal is
    the caller's business).
    """
    del our_cards
    if sum(len(c) for c in (self._steps, self._worlds, self._queries)) > self.max_cache:
      self.reset()
    out = np.zeros((len(paths), N_HANDS, self.n_actions))
    for k, path in enumerate(paths):
      out[k] = self._conditional(tuple(int(a) for a in path), list(board), int(our_seat))
    return out

  def belief(self, rules, history, our_cards, board, our_seat):
    """``r_o(h)`` at the end of ``history`` and the steps where the model failed.

    Same as ``OpponentModel.belief``, on this model's ``probs``. On the real
    history it is the joint over the worlds of the real string instead.
    """
    from ..resolving import OpponentModel, compatible
    history = tuple(int(a) for a in history)
    if history and self._on_real(history) == len(history):
      st = self._step(history)
      bkey = tuple(int(c) for c in board[: _VISIBLE[int(st.bs.street)]])
      ws = self._walk(history, bkey, int(our_seat))
      if not self._real_lost:
        joint = sum((1.0 if self.u is not None else _measure(w.ivs)) * w.lik for w in ws)
        r = compatible(our_cards, board) * joint
        if r.sum() > 0:
          return r, []
        self._lose_real(history)
    return OpponentModel.belief(self, rules, history, our_cards, board, our_seat)

  def real_terminal_payoffs(self, tree, payoff, sign, board, our_seat) -> np.ndarray:
    """``(n_terminals, H)`` fold and showdown payoffs from Slumbot's exact chips.

    For ``Resolver(terminal_fn=...)``, when the game is scored in real chips.
    The tree's chips are rounded wherever Slumbot's bets are off our bins. A
    terminal's line holds a set of worlds, each with exact chips, so hand
    ``h`` is paid their mixture with weights ``|w| L_w(h)``, the weights its
    reach is built from. ``payoff`` is the tree's table, kept for matrix leaves
    and for hands or lines that no world explains (their reach is 0).
    """
    out = np.array(payoff, np.float64)
    bkey = tuple(int(c) for c in board)
    seat = int(our_seat)
    for j, t in enumerate(tree.terminals):
      if t.kind not in (FOLD_T, SHOWDOWN_T):
        continue
      path = tuple(int(a) for a in t.path)
      ws = self._walk(path, bkey, seat)
      if not ws:
        continue
      folder = int(self._step(path[:-1]).bs.cur_player) if t.kind == FOLD_T else None
      num, den = np.zeros(N_HANDS), np.zeros(N_HANDS)
      for w in ws:
        weight = (1.0 if self.u is not None else _measure(w.ivs)) * w.lik
        den += weight
        if folder is None:
          # At a showdown an uncalled excess goes back.
          num += weight * sign * min(w.game.committed)
        else:
          num += weight * (-1 if folder == seat else 1) * w.game.committed[folder]
      ok = den > 0
      out[j, ok] = num[ok] / den[ok] / self.scale
    return out

  # ── Walking a history ─────────────────────────────────────────────────────

  def _step(self, path: tuple) -> _Step:
    # The env state after `path` and its expansion, memoised per prefix.
    st = self._steps.get(path)
    if st is None:
      if path:
        prev = self._step(path[:-1])
        if prev.alias[path[-1]] < 0:
          raise ValueError(f"atom {path[-1]} is illegal after {list(path[:-1])}")
        bs = prev.children[prev.alias[path[-1]]]
      else:
        bs = betting.initial_state(self.rules)
        bs = betting.BettingState(*(np.asarray(x) for x in bs))
      legal, alias, children, _ = expand_node(self.rules, bs)
      st = self._steps[path] = _Step(bs, legal, list(alias), children)
    return st

  def _on_real(self, path: tuple) -> int:
    # How many leading atoms of `path` are the real history.
    n = 0
    for a, b in zip(path, self._real_path):
      if a != b:
        break
      n += 1
    return n

  def _lose_real(self, path: tuple) -> None:
    # No world reproduces the real string: back to the env-atom worlds.
    self.stats["real_lost"] += 1
    self._real_lost = True
    self._real_path, self._real_strings = (), ()
    print(f"WARNING: no Slumbot world reproduces the real string after {list(path)}; "
          "ignoring the real strings for the rest of the hand", file=sys.stderr, flush=True)

  def _walk(self, path: tuple, board: tuple, our_seat: int) -> list:
    # The worlds after `path`, memoised per prefix. Returns [_World].
    # The worlds depend on the real strings of the path's real prefix, so
    # the deepest of them is part of the key.
    n_real = self._on_real(path)
    real = self._real_strings[n_real - 1] if n_real else ""
    key = (our_seat, board, path, real)
    ws = self._worlds.get(key)
    if ws is not None:
      return ws
    if not path:
      ws = [_World(((0.0, 1.0),), "", RealGame(), np.ones(N_HANDS))]
      self._worlds[key] = ws
      return ws
    prev = self._walk(path[:-1], board, our_seat)
    st = self._step(path[:-1])
    atom = st.alias[path[-1]]
    actor = int(st.bs.cur_player)
    ws = []
    for w in prev:
      if w.game.done or w.game.cur != actor:
        self.stats["desync"] += 1
        continue
      if actor == our_seat:
        nxt = _apply(w, self._our_token(st, atom, w.game, actor))
        if nxt is not None:
          ws.append(nxt)
        continue
      # Slumbot's action: split by translation cell, then by the chip amounts
      # that map to the observed atom.
      for piece, u in self._pieces(w, our_seat):
        m = self._mapped(piece, u, st, board, our_seat)
        for t in np.flatnonzero(m.token_atom == atom):
          nxt = _apply(piece, m.tokens[t], piece.lik * m.token_probs[:, t])
          if nxt is not None:
            ws.append(nxt)
    if n_real == len(path) and not self._real_lost:
      ws = [w for w in ws if w.actions == real]
      if not ws:
        self._lose_real(path)
        return self._walk(path, board, our_seat)
    self._worlds[key] = ws
    return ws

  def _conditional(self, path: tuple, board: list, our_seat: int) -> np.ndarray:
    # (H, A): the world mixture at the end of `path`.
    st = self._step(path)
    if bool(st.bs.done) or int(st.bs.cur_player) == our_seat:
      raise AssertionError(f"{list(path)} does not end at a decision of Slumbot's")
    vis = _VISIBLE[int(st.bs.street)]
    if len(board) < vis:
      raise ValueError(f"{list(path)} needs {vis} board cards, got {len(board)}")
    bkey = tuple(int(c) for c in board[:vis])
    num = np.zeros((N_HANDS, self.n_actions))
    den = np.zeros(N_HANDS)
    prior = np.zeros((N_HANDS, self.n_actions))
    n_worlds = 0
    for w in self._walk(path, bkey, our_seat):
      for piece, u in self._pieces(w, our_seat):
        m = self._mapped(piece, u, st, bkey, our_seat)
        mass = 1.0 if self.u is not None else _measure(piece.ivs)
        num += (mass * piece.lik)[:, None] * m.env_probs
        den += mass * piece.lik
        prior += mass * m.env_probs
        n_worlds += 1
    self.stats["worlds"].append(n_worlds)
    if n_worlds == 0:
      # No world reproduces the history (a desync): check/call, like a refusal.
      out = np.zeros((N_HANDS, self.n_actions))
      out[:, CALL] = 1.0
      return out
    # A hand no world explains has reach 0 already; give it the prior mixture.
    ok = den > 0
    out = prior / prior.sum(-1, keepdims=True)
    out[ok] = num[ok] / den[ok, None]
    return out

  def _pieces(self, w: _World, our_seat: int) -> list:
    # The world split by the translation cells of its string, each with a u
    # inside to query with. Infeasible cells (Slumbot's past actions are off
    # its tree there) have likelihood 0 and are dropped.
    if self.u is not None:
      return [(w, self.u)]
    key = (our_seat, w.actions)
    cells = self._cells.get(key)
    if cells is None:
      cells = self._cells[key] = self.lib.cells(our_seat, w.actions)
      self.stats["cells"] += 1
    out = []
    for c in cells:
      if not c.feasible:
        continue
      ivs = _intersect(w.ivs, c.intervals)
      if _measure(ivs) > 0:
        out.append((w._replace(ivs=ivs), c.u))
    return out

  # ── Translating actions ───────────────────────────────────────────────────

  def _totals(self, st: _Step, seat: int) -> tuple[np.ndarray, np.ndarray]:
    # Street totals of the atoms after st and which of them are legal raises.
    bs = st.bs
    totals = np.zeros(self.n_actions)
    raise_ok = np.zeros(self.n_actions, bool)
    street_max = float(np.max(bs.committed_street))
    for a, child in enumerate(st.children):
      if child is not None:
        totals[a] = float(bs.committed_street[seat] + child.committed[seat] - bs.committed[seat])
        raise_ok[a] = a >= ALL_IN and totals[a] > street_max + 1e-9
    return totals, raise_ok

  def _our_token(self, st: _Step, atom: int, g: RealGame, seat: int) -> str:
    totals, _ = self._totals(st, seat)
    tok, moved = our_token(atom, totals[atom], float(np.max(st.bs.committed_street)), g, seat,
                           self.scale)
    self.stats["raise_clamped"] += moved
    return tok

  def _mapped(self, w: _World, u: float, st: _Step, board: tuple, our_seat: int) -> _Mapped:
    # Slumbot's options for every hand at the end of w.actions, as env atoms
    # and as tokens. A query Slumbot refuses becomes a check/call.
    key = (our_seat, w.actions, board[: _VISIBLE[int(st.bs.street)]], u)
    m = self._queries.get(key)
    if m is not None:
      return m
    vis_board = list(key[2])
    try:
      q = self.lib.query(our_seat, w.actions, vis_board, HAND_CARDS, u)
      self.stats["queries"] += 1
    except native.SlumbotStatusError as e:
      self.stats["failures"].append((w.actions, e.status))
      tok = "c" if w.game.call_amount(w.game.cur) > 0 else "k"
      env = np.zeros((N_HANDS, self.n_actions))
      env[:, CALL] = 1.0
      m = _Mapped(env, [tok], np.array([CALL]), np.ones((N_HANDS, 1)))
      self._queries[key] = m
      return m
    m = self._queries[key] = self._map_query(q, st, w.game)
    return m

  def _map_query(self, q: native.Query, st: _Step, g: RealGame) -> _Mapped:
    # Rows with the same option list (all of them, except after a small bet
    # Slumbot mapped to a call) share one mapping.
    seat = int(st.bs.cur_player)
    tokens: list[str] = []
    token_atom: list[int] = []
    col: dict[str, int] = {}
    row_cols = np.zeros((N_HANDS, native.MAX_OUT), np.int64)
    patterns: dict = {}
    for h in range(N_HANDS):
      n = int(q.n_out[h])
      pat = tuple(zip(q.kind[h, :n].tolist(), q.bet[h, :n].tolist()))
      cols = patterns.get(pat)
      if cols is None:
        cols = []
        for kind, bet in pat:
          tok = _KIND_TOKEN.get(kind) or f"b{bet}"
          if tok not in col:
            col[tok] = len(tokens)
            tokens.append(tok)
            token_atom.append(self._slumbot_atom(st, g, seat, kind, bet))
          cols.append(col[tok])
        patterns[pat] = cols
      row_cols[h, :n] = cols
    token_probs = np.zeros((N_HANDS, len(tokens)))
    rows = np.repeat(np.arange(N_HANDS), native.MAX_OUT)
    valid = (np.arange(native.MAX_OUT)[None, :] < q.n_out[:, None]).ravel()
    np.add.at(token_probs, (rows[valid], row_cols.ravel()[valid]), q.prob.ravel()[valid])

    # Hands that conflict with the board get a check/call; their reach is 0.
    conflict = q.n_out == 0
    token_probs[conflict] = 1.0
    atom_of = np.asarray(token_atom)
    env = np.zeros((N_HANDS, self.n_actions))
    np.add.at(env.T, atom_of, token_probs.T)
    env[conflict] = 0.0
    env[conflict, CALL] = 1.0
    return _Mapped(env, tokens, atom_of, token_probs)

  def _slumbot_atom(self, st: _Step, g: RealGame, seat: int, kind: int, bet: int) -> int:
    totals, raise_ok = self._totals(st, seat)
    atom, gap = slumbot_atom(kind, bet, totals, raise_ok, g, seat, self.scale)
    all_in = g.street_committed[seat] + g.stack(seat)
    if kind == native.SB_BET and bet < all_in and atom >= ALL_IN:
      self.stats["bet_error"].append(gap / float(st.bs.committed.sum()))
    return atom
