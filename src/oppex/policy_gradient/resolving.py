"""Depth-limited re-solving against the fixed blueprint, for the full game.

At each of our decisions that the current subgame does not cover, we build a
new subgame rooted at it (``cfr.subgame``, ``depth`` plies, matrix leaves at
the depth limit and at street closes) and solve it with ``cfr.infoset_cfr``.
Its average strategy is played at every later decision of ours that it
contains, so with ``depth=4`` two of our moves come from one solve.

**Why one hand and no gadget.** The opponent is fixed, so for us the game is a
decision problem. The root does not need a public belief over both ranges, only
our infoset: our hand plus, for every opponent hand ``h``, its counterfactual
reach ``r_o(h)``, which is the blueprint's probability of the opponent's actions
so far times card compatibility (``belief``). Inside the subgame ``r_o`` is
pushed down through the blueprint's per-hand action probabilities
(``subgame_reach``). Every terminal and leaf then contracts to a number, see
``infoset_cfr``.

**The opponent model** is the blueprint, exactly: no exploration noise mixed
in, which would bias every belief and every opponent node to protect against a
case that only arises when the model is wrong. That case is an observed action
to which the model gives probability zero for every hand the opponent can
still hold. Then the belief would vanish, so ``belief`` treats that one action
as if played by a uniform policy, which leaves the belief unchanged, and warns:
the model has just been contradicted. Probabilities of bet atoms the tree
struck as duplicates are added to their survivor, because only the public
state is observed.

**Leaf values** ``U(l, t) = Σ_h r_o(l, h) · V(l, t, h)`` are estimated by
rollouts. Each one draws ``h`` from ``r_o(l, ·)``, the unknown board cards from
the rest of the deck, replays the leaf's history, and plays our seat as the
blueprint biased by ``t`` and the opponent as the blueprint. The uniforms
behind the opponent hand, the runout and every action are shared across
leaves and continuations (common random numbers), so the differences the
solver acts on are less noisy than the values. Leaves the opponent reaches
with negligible mass are not rolled out (value 0), and a cached node below
that mass is never played from: it is re-solved instead.

**Other fixed opponents.** The opponent enters only through ``model``
(``probs``, ``belief``) and ``rollouts`` (``values``), so any fixed policy with
those two parts can replace the blueprint; ``slumbot.model`` and
``slumbot.rollouts`` make Slumbot one.
"""

from __future__ import annotations

import os
import sys
import time
import zlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..cfr import infoset_cfr
from ..cfr.cards import HAND_CARDS, HAND_INDEX, N_HANDS
from ..cfr.equity import sign_vs_range
from ..cfr.leaves import BucketedValues
from ..cfr.subgame import build_subgame, replay_betting
from ..cfr.tree import FOLD_T, MATRIX_T, SHOWDOWN_T
from ..policy import policy_probs, sample_atom
from .blueprint import N_TRANSFORMS, PLURIBUS_FACTOR, Blueprint, bias_multipliers
from .leaf_values import play_out
from .slumbot.play import TreeAgent
from .tree_policy import pad_paths, replay

_VISIBLE = (0, 3, 4, 5)


def compatible(our_cards, board) -> np.ndarray:
  """``(N_HANDS,)`` 1.0 for hands sharing no card with ours or the board."""
  known = list(our_cards) + list(board)
  return (~np.isin(HAND_CARDS, known).any(1)).astype(np.float64)


def _fill_board(our_cards, board) -> np.ndarray:
  # The known board padded to 5 cards with cards nobody can see; the env
  # masks them by street, so they never reach an infoset.
  rest = [c for c in range(52) if c not in set(our_cards) | set(board)]
  return np.asarray(list(board) + rest[: 5 - len(board)], np.int32)


def _fold(probs, alias) -> np.ndarray:
  # Env-atom probabilities (..., A) onto tree atoms: a struck duplicate's
  # mass goes to its survivor.
  out = np.zeros_like(probs)
  for a, c in enumerate(alias):
    if c >= 0:
      out[..., c] += probs[..., a]
  return out


# ── Opponent model ───────────────────────────────────────────────────────────


class OpponentModel:
  """The blueprint's action probabilities for every opponent hand at once."""

  def __init__(self, bp: Blueprint, chunk: int = 16) -> None:
    self.bp, self.chunk = bp, chunk
    env = bp.env
    key = jax.random.PRNGKey(0)
    hands = jnp.asarray(HAND_CARDS)
    infoset_b = jax.vmap(env.information_set, in_axes=(0, None, None))
    legal_b = jax.vmap(env.legal_actions, in_axes=(0, None))

    @jax.jit
    def probs(paths, ours, board, our_seat):
      # paths (k, W) -> (k, H, A) at the opponent's decision after each path.
      k = paths.shape[0]
      opp = 1 - our_seat
      seat_is_ours = (jnp.arange(2) == our_seat)[None, :, None]
      hole = jnp.where(seat_is_ours, ours[None, None, :], hands[:, None, :])
      s0 = env.init_state(key)
      s = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (k * N_HANDS,) + x.shape), s0)
      s = s._replace(hole_cards=jnp.tile(hole, (k, 1, 1)),
                     board=jnp.broadcast_to(board, (k * N_HANDS, 5)))
      s = replay(env, s, jnp.repeat(paths, N_HANDS, 0))
      legal = legal_b(s, opp)
      p = policy_probs(bp.eval_fn(bp.params, infoset_b(s, opp, key)), legal)
      ok = (s.cur_player == opp) & ~s.done
      return p.reshape(k, N_HANDS, -1), ok.reshape(k, N_HANDS).all(-1)

    self._probs = probs

  def probs(self, paths, our_cards, board, our_seat) -> np.ndarray:
    """``(len(paths), N_HANDS, A)`` env-atom probabilities after each path."""
    if not paths:
      return np.zeros((0, N_HANDS, self.bp.env.num_actions))
    padded = pad_paths(paths, 8)
    ours = jnp.asarray(our_cards, jnp.int32)
    board = jnp.asarray(_fill_board(our_cards, board))
    out = []
    for lo in range(0, len(paths), self.chunk):
      rows = np.full((self.chunk, padded.shape[1]), -1, np.int32)
      rows[: min(self.chunk, len(paths) - lo)] = padded[lo: lo + self.chunk]
      p, ok = self._probs(jnp.asarray(rows), ours, board, jnp.int32(our_seat))
      n = min(self.chunk, len(paths) - lo)
      if not np.asarray(ok)[:n].all():
        raise AssertionError("a path does not end at an opponent decision")
      out.append(np.asarray(p, np.float64)[:n])
    return np.concatenate(out)

  def belief(self, rules, history, our_cards, board, our_seat) -> tuple[np.ndarray, list]:
    """``r_o(h)`` at the end of ``history``, and the steps where the model failed.

    ``r_o`` is compatibility times the opponent's probabilities of their
    actions. An action the model rules out for every remaining hand is
    counted as uniform over the legal tree actions instead, which leaves the
    belief as it was; its step index is returned and a warning printed.
    """
    states = replay_betting(rules, history)
    opp_steps = [j for j, (bs, _) in enumerate(states[:-1]) if int(bs.cur_player) != our_seat]
    r = compatible(our_cards, board)
    failed = []
    if opp_steps:
      p = self.probs([tuple(history[:j]) for j in opp_steps], our_cards, board, our_seat)
      for q, j in enumerate(opp_steps):
        alias = states[j][1]
        new = r * _fold(p[q], alias)[:, alias[int(history[j])]]
        if new.sum() <= 0:
          n_legal = sum(c == a for a, c in enumerate(alias))
          new = r / n_legal
          failed.append(j)
          print(f"WARNING: the opponent model gives probability 0 to atom {history[j]} "
                f"after {list(history[:j])} for every hand the opponent can hold; "
                "treating that action as uniform", file=sys.stderr, flush=True)
        r = new
    return r, failed


def subgame_reach(tree, belief, model, our_cards, board, our_seat):
  """``r_o`` at every node ``(n_nodes, H)`` and terminal ``(n_terminals, H)``.

  Also returns the opponent's tree-atom probabilities ``{node: (H, A)}`` at
  each of their nodes, which per-history values reuse.
  """
  opp_nodes = [i for i, n in enumerate(tree.nodes) if n.player != our_seat]
  p = model.probs([tree.nodes[i].path for i in opp_nodes], our_cards, board, our_seat)
  pi = {i: _fold(p[q], tree.nodes[i].alias) for q, i in enumerate(opp_nodes)}
  r_node = np.zeros((len(tree.nodes), N_HANDS))
  r_term = np.zeros((len(tree.terminals), N_HANDS))
  r_node[tree.root] = belief
  # Children always have larger indices than their parent (preorder slots).
  for i, n in enumerate(tree.nodes):
    for a, ch in enumerate(n.child):
      if ch is None:
        continue
      r = r_node[i] * pi[i][:, a] if i in pi else r_node[i]
      (r_node if ch[0] == "D" else r_term)[ch[1]] = r
  return r_node, r_term, pi


# ── Values ───────────────────────────────────────────────────────────────────


def showdown_sign(our_cards, board, ev) -> np.ndarray:
  """``(H,)`` expected showdown sign of our hand against each opponent hand.

  The ``EV`` row preflop, an exact runout enumeration on a known board.
  """
  if len(board) == 0:
    return np.asarray(ev[HAND_INDEX[our_cards[0], our_cards[1]]], np.float64)
  return sign_vs_range(our_cards, board)


def terminal_payoffs(tree, our_seat, sign) -> np.ndarray:
  """``(n_terminals, H)`` chips to us per opponent hand; matrix leaves are 0.

  The single source of terminal payoffs: the solver contracts it against
  ``r_o``, AIVAT reads it at the revealed opponent hand.
  """
  to_us = 1.0 if our_seat == 0 else -1.0
  out = np.zeros((len(tree.terminals), N_HANDS))
  for j, t in enumerate(tree.terminals):
    if t.kind == FOLD_T:
      out[j] = to_us * t.const
    elif t.kind == SHOWDOWN_T:
      c = np.asarray(t.betting.committed)
      if c[0] != c[1]:
        raise AssertionError(f"unequal commitments at showdown: {c}")
      out[j] = t.stake * sign
  return out


def make_deal(env, chunk: int, n_games: int, row_keys: bool = False):
  """The deal of ``chunk`` leaves × ``n_games`` rollouts, to be jitted by callers.

  ``deal(key, paths, q, ours, board, n_known, our_seat)`` draws the opponent's
  hand from each leaf's ``q``, the unknown board cards from the rest of the
  deck, and replays each leaf's path. Returns the states, the opponent hands
  ``(B,)`` and the per-game key of each rollout slot ``(n_games,)``. Slot ``i``
  of every leaf shares its uniforms with slot ``i`` of every other leaf
  (common random numbers). With ``row_keys``, ``key`` is ``(chunk, 2)``, one
  key per row, and the returned keys are per game ``(B,)``: rows with equal
  keys share their uniforms, others are independent.
  """
  hands = jnp.asarray(HAND_CARDS)
  n_b = chunk * n_games
  leaf = jnp.repeat(jnp.arange(chunk), n_games)
  game = jnp.tile(jnp.arange(n_games), chunk)

  def deal(key, paths, q, ours, board, n_known, our_seat):
    if row_keys:
      keys = jax.vmap(lambda k: jax.random.split(k, n_games))(key).reshape(n_b, -1)
      slot = jnp.arange(n_b)
    else:
      keys = jax.random.split(key, n_games)
      slot = game
    u_h = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 0)))(keys)[slot]
    # Hands outside the leaf's range (q = 0, e.g. blocked by our cards) are
    # never drawn.
    h = sample_atom(q[leaf], u_h)
    opp_cards = hands[h]

    # Unknown board cards: a random order of the deck, used cards last.
    pos = jnp.arange(5)
    known = (pos < n_known)[:, None] * jax.nn.one_hot(board, 52)
    used = (jax.nn.one_hot(ours, 52).sum(0) + known.sum(0))[None, :] \
      + jax.nn.one_hot(opp_cards, 52).sum(1)
    keys_card = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 1), (52,)))(keys)
    order = jnp.argsort(keys_card[slot] + 2.0 * (used > 0), -1)
    drawn = jnp.take_along_axis(order, jnp.clip(pos - n_known, 0, 4)[None, :].repeat(n_b, 0), -1)
    full = jnp.where(pos < n_known, board[None, :], drawn).astype(jnp.int32)

    ours_b = jnp.broadcast_to(ours, (n_b, 2))
    hole = jnp.where(our_seat == 0, jnp.stack([ours_b, opp_cards], 1),
                     jnp.stack([opp_cards, ours_b], 1))
    s0 = env.init_state(jax.random.PRNGKey(0))
    s = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_b,) + x.shape), s0)
    s = s._replace(hole_cards=hole, board=full)
    return replay(env, s, paths[leaf]), h, keys

  return deal


class LeafRollouts:
  """Jitted rollouts valuing ``chunk`` leaves × ``n_games`` per call."""

  def __init__(self, bp: Blueprint, n_games: int = 128, chunk: int = 32,
               factor: float = PLURIBUS_FACTOR, row_keys: bool = False) -> None:
    self.bp, self.n_games, self.chunk, self.factor = bp, n_games, chunk, factor
    self.row_keys = row_keys
    env = bp.env
    mult = bias_multipliers(env.num_actions, factor)
    deal = make_deal(env, chunk, n_games, row_keys)
    game = jnp.arange(chunk * n_games) if row_keys else jnp.tile(jnp.arange(n_games), chunk)

    @jax.jit
    def run(key, paths, q, ours, board, n_known, our_seat, t):
      # Mean chips to us per leaf and the truncated fraction.
      s, _, keys = deal(key, paths, q, ours, board, n_known, our_seat)
      game_keys = jax.vmap(lambda k: jax.random.fold_in(k, 2))(keys)[game]
      t0, t1 = jnp.where(our_seat == 0, t, 0), jnp.where(our_seat == 1, t, 0)
      s, pay = play_out(bp, s, game_keys, t0, t1, mult)
      pay = pay * env.starting_stack * jnp.where(our_seat == 0, 1.0, -1.0)
      return pay.reshape(chunk, n_games).mean(1), (~s.done).mean()

    self._run = run

  def resized(self, n_games: int, chunk: int) -> LeafRollouts:
    """The same opponent's rollouts with ``n_games`` per leaf, ``chunk`` leaves per call.

    For AIVAT's boards valued by rollouts, which also pass a board longer
    than the leaf's street. A subclass opts in by overriding it; otherwise it
    raises ``NotImplementedError``, since a copy of the base class would roll
    out against the blueprint instead of the subclass's opponent.
    """
    if type(self) is not LeafRollouts:
      raise NotImplementedError(f"{type(self).__name__} does not define resized")
    return LeafRollouts(self.bp, n_games, chunk, self.factor)

  def _chunk(self, key, paths, q, ours, full, n_known, our_seat, t, leaf_paths, board):
    # One call: (mean chips to us per leaf of the chunk, truncated fraction).
    # leaf_paths and board (the visible board) are for subclasses.
    mean, tr = self._run(key, paths, q, ours, full, n_known, our_seat, t)
    return np.asarray(mean, np.float64), float(tr)

  def values(self, tree, r_term, our_cards, board, our_seat, seed=0, min_mass=0.0,
             t_mask=None):
    """``(n_leaves, T)`` leaf values ``Z_l · E[chips | t]`` and the truncated fraction.

    ``t_mask`` ``(n_leaves, T)`` limits the continuations rolled out; a chunk
    plays a ``t`` if any of its leaves needs it, and unrequested cells are 0.
    """
    leaves = list(tree.leaves)
    z = r_term[leaves].sum(-1)
    out = np.zeros((len(leaves), N_TRANSFORMS))
    live = [i for i in range(len(leaves)) if z[i] > min_mass]
    if not live:
      return out, 0.0
    leaf_paths = [tree.terminals[leaves[i]].path for i in live]
    padded = pad_paths(leaf_paths, 8)
    q = r_term[[leaves[i] for i in live]] / z[live, None]
    ours = jnp.asarray(our_cards, jnp.int32)
    full = jnp.asarray(_fill_board(our_cards, board))
    key = jax.random.PRNGKey(seed)
    trunc = []
    for lo in range(0, len(live), self.chunk):
      n = min(self.chunk, len(live) - lo)
      # A fixed chunk shape keeps it to one compile; the tail repeats a leaf.
      idx = np.r_[np.arange(lo, lo + n), np.full(self.chunk - n, lo)]
      for t in range(N_TRANSFORMS):
        if t_mask is not None and not t_mask[[live[i] for i in idx[:n]], t].any():
          continue
        mean, tr = self._chunk(key, jnp.asarray(padded[idx]), jnp.asarray(q[idx], jnp.float32),
                               ours, full, jnp.int32(len(board)), jnp.int32(our_seat),
                               jnp.int32(t), [leaf_paths[i] for i in idx], list(board))
        out[[live[i] for i in idx[:n]], t] = mean[:n]
        trunc.append(tr)
    return out * z[:, None], float(np.mean(trunc))

  def values_per_hand(self, tree, r_term, our_cards, board, our_seat, seed=0, min_mass=0.0,
                      known=None):
    """Leaf values from ``n_games`` rollouts per (leaf, opponent hand).

    Returns ``(n_leaves, T)`` values ``Σ_h r(l, h)·V(l, t, h)`` as ``values``
    gives them, the truncated fraction, and the per-history table
    ``V (n_leaves, T, N_HANDS)`` (float32) in chips to us, NaN where ``r(l, h) = 0``
    or the leaf is below ``min_mass``. Sampling every hand instead of drawing
    ``h ~ r`` removes the between-hand noise from the contraction, and the
    table does not depend on the belief. The rows of one hand share their
    uniforms (common random numbers across leaves and ``t``), while hands get
    independent runouts. Sharing them across hands as well would leave the
    contraction with the board noise of ``n_games`` runouts, correlated over
    leaves. Needs ``row_keys``. ``known`` maps leaf positions to ``(T, H)``
    rows from earlier solves, which are reused instead of rolled out.
    """
    known = known or {}
    if not self.row_keys:
      raise ValueError("values_per_hand needs LeafRollouts(..., row_keys=True)")
    leaves = list(tree.leaves)
    z = r_term[leaves].sum(-1)
    # float32, the precision rows are cached at, so that a cached row and a
    # fresh one give the solve identical values.
    table = np.full((len(leaves), N_TRANSFORMS, N_HANDS), np.nan, np.float32)
    for i, row in known.items():
      table[i] = row
    pairs = [(i, int(h)) for i in range(len(leaves)) if z[i] > min_mass and i not in known
             for h in np.flatnonzero(r_term[leaves[i]] > 0)]
    if not pairs:
      out = np.einsum("lth,lh->lt", np.nan_to_num(table), r_term[leaves])
      return out, 0.0, table
    pair_leaf = np.array([i for i, _ in pairs])
    pair_hand = np.array([h for _, h in pairs])
    leaf_paths = [tree.terminals[leaves[i]].path for i in range(len(leaves))]
    padded = pad_paths(leaf_paths, 8)
    ours = jnp.asarray(our_cards, jnp.int32)
    full = jnp.asarray(_fill_board(our_cards, board))
    hand_keys = jax.vmap(jax.random.fold_in, (None, 0))(
      jax.random.PRNGKey(seed), jnp.arange(N_HANDS))
    trunc = []
    for lo in range(0, len(pairs), self.chunk):
      n = min(self.chunk, len(pairs) - lo)
      idx = np.r_[np.arange(lo, lo + n), np.full(self.chunk - n, lo)]
      # The opponent's hand is fixed per row by a one-hot reach.
      q = np.zeros((self.chunk, N_HANDS), np.float32)
      q[np.arange(self.chunk), pair_hand[idx]] = 1.0
      keys = hand_keys[pair_hand[idx]]
      for t in range(N_TRANSFORMS):
        mean, tr = self._chunk(keys, jnp.asarray(padded[pair_leaf[idx]]), jnp.asarray(q),
                               ours, full, jnp.int32(len(board)), jnp.int32(our_seat),
                               jnp.int32(t), [leaf_paths[i] for i in pair_leaf[idx]],
                               list(board))
        table[pair_leaf[idx[:n]], t, pair_hand[idx[:n]]] = mean[:n]
        trunc.append(tr)
    r = r_term[leaves]
    out = np.einsum("lth,lh->lt", np.nan_to_num(table), r)
    return out, float(np.mean(trunc)), table


def bucketed_leaf_values(tree, r_term, our_cards, our_seat, vb, bucket, pscfr_paths,
                         strict: bool = True):
  """Leaf values from a fixed-opponent PSCFR cache (validation only).

  ``vb`` is that run's ``(L, T0, T1, K, K)`` with the opponent's axis of size
  one, ``pscfr_paths`` its leaf paths. Every leaf of ``tree`` must be one of
  them, which holds preflop with no depth limit; with ``strict=False`` the
  others are NaN.
  """
  where = {p: i for i, p in enumerate(pscfr_paths)}
  b = int(bucket[HAND_INDEX[our_cards[0], our_cards[1]]])
  out = np.zeros((len(tree.leaves), vb.shape[1] * vb.shape[2]))
  for k, j in enumerate(tree.leaves):
    path = tree.terminals[j].path
    if path not in where and not strict:
      out[k] = np.nan
      continue
    v = np.asarray(vb[where[path]], np.float64)
    # Chips to us against each opponent hand, per own continuation.
    per_h = v[:, 0, b, :][:, bucket] if our_seat == 0 else -v[0, :, :, b][:, bucket]
    out[k] = per_h @ r_term[j]
  return out


# ── Per-history leaf cache ───────────────────────────────────────────────────


class LeafTableCache:
  """Per-hand leaf rows ``V(l, ·, h)`` kept across solves and, preflop, on disk.

  A row ``(T, N_HANDS)`` depends on the opponent, the rollout settings, our
  seat and hand, the board and the leaf's betting path, but not on the
  belief, so any solve reaching the same leaf can reuse it. Rows are grouped
  by (seat, our hand, board). Groups with a board live in memory for the
  current hand only: exact repeats across hands are rare postflop. Preflop
  groups are also written to ``directory``, one ``.npz`` per group, replaced
  atomically, so concurrent writers can lose rows but never corrupt a file.
  ``directory`` must be specific to the opponent and the rollout settings;
  ``Resolver`` names it so.
  """

  def __init__(self, directory: str | None = None) -> None:
    self.directory = directory
    if directory:
      os.makedirs(directory, exist_ok=True)
    self.groups: dict[tuple, dict[tuple, np.ndarray]] = {}

  @staticmethod
  def key(seat, cards, board) -> tuple:
    return int(seat), tuple(int(c) for c in cards), tuple(int(c) for c in board)

  def _file(self, key) -> str | None:
    seat, cards, board = key
    if not self.directory or board:
      return None
    return os.path.join(self.directory, f"seat{seat}_{cards[0]}-{cards[1]}.npz")

  def group(self, key) -> dict:
    """The rows of a group by leaf path, loaded from disk the first time."""
    if key not in self.groups:
      rows, path = {}, self._file(key)
      if path and os.path.exists(path):
        with np.load(path) as f:
          for p, row in zip(f["paths"], f["rows"]):
            rows[tuple(int(a) for a in str(p).split("-") if a)] = row
      self.groups[key] = rows
    return self.groups[key]

  def save(self, key) -> None:
    path = self._file(key)
    if path is None:
      return
    rows = self.groups[key]
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, paths=np.array(["-".join(map(str, p)) for p in rows]),
             rows=np.stack(list(rows.values())).astype(np.float32))
    os.replace(tmp, path)

  def new_hand(self) -> None:
    # Postflop groups are not reused across hands.
    self.groups = {k: v for k, v in self.groups.items() if not k[2]}


# ── Solving one infoset ──────────────────────────────────────────────────────


class Solution(NamedTuple):
  game: infoset_cfr.InfosetGame
  sigma: np.ndarray          # (n_nodes, A) CFR average, our nodes only meaningful
  tau: np.ndarray            # (n_leaves, T)
  z: float                   # opponent mass at the root, Σ_h r_o(root, h)
  node_of: dict              # history -> node id, our nodes above min_mass
  stats: dict
  board: tuple               # visible board at the root
  pi: dict                   # opponent node -> (H, A) tree-atom probabilities
  payoff: np.ndarray         # (n_terminals, H) chips to us, see terminal_payoffs
  index: dict                # history -> node id, every node
  leaf_mass: np.ndarray      # (n_leaves,) Σ_h r_o(l, h)
  # (n_leaves, T, H) per-history leaf values with per-hand rollouts, else None.
  leaf_table: np.ndarray | None = None


class Resolver:
  """Re-solving player for one seat of one hand at a time.

  ``probs(history, board, ...)`` is all a player loop needs; it decides
  between reusing the cached subgame and re-solving.
  """

  def __init__(self, bp: Blueprint, ev, depth: int | None = 4, leaf_games: int = 128,
               iters: int = 500, factor: float = PLURIBUS_FACTOR,
               min_mass: float = 1e-4, chunk_leaves: int = 32, seed: int = 0,
               leaf_fn=None, model=None, rollouts=None, games_per_hand: int | None = None,
               hand_chunk: int | None = None, leaf_cache_dir: str | None = None,
               terminal_fn=None) -> None:
    self.bp, self.ev, self.depth, self.iters = bp, ev, depth, iters
    self.rules = bp.env.rules
    self.min_mass, self.seed = min_mass, seed
    # The opponent: the blueprint unless a model (probs, belief) and matching
    # rollouts (values) are given, e.g. slumbot.model.SlumbotModel and
    # slumbot.rollouts.SlumbotRollouts.
    self.model = model if model is not None else OpponentModel(bp)
    self.rollouts = (rollouts if rollouts is not None
                     else LeafRollouts(bp, leaf_games, chunk_leaves, factor))
    # leaf_fn(tree, r_term, our_cards, board, seat) overrides the rollouts.
    self.leaf_fn = leaf_fn
    # terminal_fn(tree, payoff, sign, board, seat) replaces the fold and
    # showdown payoffs, e.g. by the chips of the real game.
    self.terminal_fn = terminal_fn
    # With games_per_hand, every (leaf, opponent hand) gets its own rollouts
    # and the solve keeps the per-history table (blueprint opponent only).
    self.hand_rollouts = None
    if games_per_hand:
      if type(self.rollouts) is not LeafRollouts:
        raise ValueError("per-hand rollouts need the blueprint's LeafRollouts")
      # About 8k games per call: smaller calls leave the GPU idle.
      chunk = hand_chunk or max(32, 8192 // games_per_hand)
      self.hand_rollouts = LeafRollouts(bp, games_per_hand, chunk, factor, row_keys=True)
      # Rows are cached per opponent and rollout settings, so the directory
      # names both.
      if leaf_cache_dir:
        run = os.path.basename(os.path.dirname(os.path.abspath(bp.checkpoint)))
        leaf_cache_dir = os.path.join(
          leaf_cache_dir, f"{run}_{bp.which}_step{bp.step}_f{factor:g}_g{games_per_hand}")
      self.leaf_cache = LeafTableCache(leaf_cache_dir)
    self.n_solves = 0
    self.new_hand(0, (0, 1))

  def new_hand(self, seat: int, hole) -> None:
    self.seat, self.cards = int(seat), tuple(int(c) for c in hole)
    self.solution: Solution | None = None
    if self.hand_rollouts is not None:
      self.leaf_cache.new_hand()
    # Every solve played from this hand, for evaluation after it.
    self.solves: list[Solution] = []
    self.log: list[dict] = []

  def solve(self, history, board, seed: int | None = None,
            depth: int | None | str = "default") -> Solution:
    """The subgame rooted after ``history``, solved for our seat.

    The root is normally our decision; validation and evaluation also root it
    at an opponent's decision, which the formulation allows. ``seed`` fixes
    the rollouts of a solve made outside play, so that play's own seeds do
    not shift; ``depth`` overrides the resolver's.
    """
    t0 = time.time()
    history, board = tuple(int(a) for a in history), [int(c) for c in board]
    root = replay_betting(self.rules, history)[-1][0]
    tree = build_subgame(self.rules, root, history,
                         self.depth if depth == "default" else depth)
    t1 = time.time()
    belief, failed = self.model.belief(self.rules, history, self.cards, board, self.seat)
    r_node, r_term, pi = subgame_reach(tree, belief, self.model, self.cards, board, self.seat)
    t2 = time.time()
    z = belief.sum()
    if z <= 0:
      raise AssertionError(f"the opponent cannot have reached {history}")
    sign = showdown_sign(self.cards, board, self.ev)
    payoff = terminal_payoffs(tree, self.seat, sign)
    if self.terminal_fn is not None:
      payoff = self.terminal_fn(tree, payoff, sign, board, self.seat)
    term = (payoff * r_term).sum(-1)
    t3 = time.time()
    trunc, table, n_cached = 0.0, None, 0
    if self.leaf_fn is not None:
      leaf = self.leaf_fn(tree, r_term, self.cards, board, self.seat)
    elif self.hand_rollouts is not None:
      leaf, trunc, table, n_cached = self._per_hand_leaves(tree, r_term, board, z)
    else:
      leaf, trunc = self.rollouts.values(
        tree, r_term, self.cards, board, self.seat,
        seed=self.seed + self.n_solves if seed is None else seed,
        min_mass=self.min_mass * z)
    t4 = time.time()
    game = infoset_cfr.InfosetGame(tree, self.seat, term, leaf)
    sigma, tau = infoset_cfr.solve(game, self.iters)
    br, _, _, _ = infoset_cfr.best_response(game)
    value = infoset_cfr.policy_value(game, sigma, tau)
    t5 = time.time()
    self.n_solves += seed is None
    mass = r_node.sum(-1) / z
    node_of = {n.path: i for i, n in enumerate(tree.nodes)
               if n.player == self.seat and mass[i] >= self.min_mass}
    stats = dict(
      history=list(history), street=len(board), nodes=len(tree.nodes),
      leaves=len(tree.leaves), value=value / z, br=br / z, gap=(br - value) / z,
      model_failures=failed, cached_leaves=n_cached,
      truncated=trunc, t_build=t1 - t0, t_reach=t2 - t1, t_terminals=t3 - t2,
      t_leaves=t4 - t3, t_cfr=t5 - t4, t_total=t5 - t0,
    )
    index = {n.path: i for i, n in enumerate(tree.nodes)}
    return Solution(game, sigma, tau, float(z), node_of, stats, tuple(board), pi, payoff,
                    index, r_term[list(tree.leaves)].sum(-1), table)

  def _per_hand_leaves(self, tree, r_term, board, z):
    # Leaf values from per-hand rollouts, reusing cached rows. The seed
    # depends only on (seat, our hand, board), so a row is the same whichever
    # solve computes it: cached and fresh rows are interchangeable, and
    # evaluation solves that fill the cache do not change play. Returns
    # (values, truncated fraction, table, number of cached leaves).
    key = LeafTableCache.key(self.seat, self.cards, board)
    rows = self.leaf_cache.group(key)
    leaves = list(tree.leaves)
    known = {}
    for i, j in enumerate(leaves):
      row = rows.get(tree.terminals[j].path)
      # A row is usable if it covers every hand that can reach the leaf.
      if row is not None and not np.isnan(row[:, r_term[j] > 0]).any():
        known[i] = row
    seed = (self.seed * 1_000_003 + zlib.crc32(repr(key).encode())) % 2**31
    leaf, trunc, table = self.hand_rollouts.values_per_hand(
      tree, r_term, self.cards, board, self.seat, seed=seed,
      min_mass=self.min_mass * z, known=known)
    new = [i for i in range(len(leaves))
           if i not in known and not np.isnan(table[i]).all()]
    for i in new:
      rows[tree.terminals[leaves[i]].path] = table[i]
    if new:
      self.leaf_cache.save(key)
    return leaf, trunc, table, len(known)

  def probs(self, history, board, n_actions: int) -> tuple[np.ndarray, bool]:
    """Our env-atom probabilities after ``history``, and whether we re-solved."""
    history = tuple(int(a) for a in history)
    sol = self.solution
    resolved = sol is None or history not in sol.node_of
    if resolved:
      if int(replay_betting(self.rules, history)[-1][0].cur_player) != self.seat:
        raise AssertionError(f"history {history} does not end at our decision")
      sol = self.solution = self.solve(history, board)
      self.solves.append(sol)
      self.log.append(sol.stats)
    p = np.asarray(sol.sigma[sol.node_of[history]], np.float64)[:n_actions]
    return p / p.sum(), resolved


# ── Players ──────────────────────────────────────────────────────────────────


def _view(s, seat):
  # The parts of a HunlState a player may see: history and visible board.
  step = int(s.step)
  history = [int(a) for a in np.asarray(s.action_history)[:step]]
  board = [int(c) for c in np.asarray(s.board)[: _VISIBLE[int(s.street)]]]
  return history, board


class ResolvingAgent(TreeAgent):
  """``TreeAgent`` for ``slumbot/play.py`` whose decisions come from a ``Resolver``."""

  def __init__(self, run_dir, checkpoint, which, ev, **resolver_kw) -> None:
    super().__init__(run_dir, checkpoint, which)
    self.resolver = Resolver(self, ev, **self._resolver_kw(resolver_kw))
    self.n_decisions = self.n_resolves = 0

  def _resolver_kw(self, kw: dict) -> dict:
    # The Resolver's arguments; subclasses add an opponent model and rollouts
    # here, once the blueprint is loaded.
    return kw

  def new_state(self, hole, seat):
    self.resolver.new_hand(seat, hole)
    self.n_decisions = self.n_resolves = 0
    return super().new_state(hole, seat)

  def policy(self, s, seat):
    legal = self.legal(s, seat)
    history, board = _view(s, seat)
    probs, resolved = self.resolver.probs(history, board, len(legal))
    if (probs[~legal] > 1e-6).any():
      raise AssertionError(f"the re-solved strategy plays an illegal atom after {history}")
    self.n_decisions += 1
    self.n_resolves += int(resolved)
    return probs, legal

  def hand_info(self) -> dict:
    return {"resolving": {"decisions": self.n_decisions, "resolves": self.n_resolves,
                          "solves": self.resolver.log}}


def play_vs_blueprint(bp: Blueprint, resolver: Resolver, n_hands: int, seed: int = 0,
                      log=None, progress: bool = True, aivat=None,
                      opponent=None) -> list[dict]:
  """Hands against the blueprint in the env, each also played by the blueprint.

  Seats alternate. The control replays the same deal with the blueprint in our
  seat and the same uniform per env step, so ``payoff - control`` is a paired
  difference. ``aivat`` (a ``resolving_aivat.ResolvingAivat``) records every
  decision of the re-solved replay and adds its AIVAT terms under ``"aivat"``. ``opponent`` (e.g.
  ``slumbot.opponent.SlumbotOpponent``) replaces the blueprint in the other
  seat; it gets the same hand type in both replays of a hand. Returns one
  record per hand, also written as JSON lines to ``log``.
  """
  import json

  env = bp.env
  apply = jax.jit(lambda s, a: env.apply_action(s, jnp.full((2,), a, jnp.int32)))
  records = []
  for i in range(n_hands):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), i)
    seat = i % 2
    u = np.random.default_rng([seed, i]).random(env.max_length + 1)
    u_type = float(np.random.default_rng([seed, i, 1]).random())
    rec = {"hand": i, "seat": seat}
    for mode in ("resolve", "control"):
      s = env.init_state(key)
      hole = [int(c) for c in np.asarray(s.hole_cards)[seat]]
      resolver.new_hand(seat, hole)
      if opponent is not None:
        opponent.new_hand(seat, np.asarray(s.hole_cards)[1 - seat], np.asarray(s.board), u_type)
      record = aivat is not None and mode == "resolve"
      if record:
        aivat.new_hand(np.asarray(s.hole_cards)[1 - seat])
      payoff, n_dec, n_res, steps = 0.0, 0, 0, []
      t0 = time.time()
      while not bool(s.done) and int(s.step) < env.max_length:
        cur = int(s.cur_player)
        step = {"actor": cur, "history": None, "board": None, "solve": None,
                "committed": np.asarray(s.committed, np.float64).tolist()}
        step["history"], step["board"] = _view(s, seat)
        if opponent is not None and cur != seat:
          atom, probs = opponent.act(step["history"], u[int(s.step)])
        else:
          probs, legal = bp.policy(s, cur)
          if mode == "resolve" and cur == seat:
            probs, resolved = resolver.probs(step["history"], step["board"], len(legal))
            n_dec, n_res = n_dec + 1, n_res + int(resolved)
            step["solve"] = len(resolver.solves) - 1
          # Inverse CDF; side="right" never lands on an atom of probability 0,
          # and a rounded-up draw goes to the last atom of positive one.
          cdf = np.cumsum(probs)
          atom = int(min(np.searchsorted(cdf, u[int(s.step)] * cdf[-1], side="right"),
                         np.flatnonzero(np.asarray(probs) > 0)[-1]))
          # So an illegal atom means the policy put mass on it, which would
          # also make AIVAT's sampling distribution wrong.
          if not legal[atom]:
            raise AssertionError(f"policy gives probability {probs[atom]} to illegal "
                                 f"atom {atom} after {step['history']}")
        if opponent is not None:
          opponent.observe(step["history"], atom, cur)
        step.update(atom=atom, probs=np.asarray(probs, np.float64).tolist())
        steps.append(step)
        if record:
          aivat.step(cur, step["history"], step["board"], step["committed"], probs, atom)
        s, rewards, _, _ = apply(s, atom)
        payoff += float(np.asarray(rewards)[seat]) * env.starting_stack
      if mode == "control":
        rec["control"] = payoff
        continue
      rec.update(payoff=payoff, decisions=n_dec, resolves=n_res,
                 seconds=time.time() - t0, solves=resolver.log,
                 truncated=not bool(s.done), history=_view(s, seat)[0])
      if record and steps and bool(s.done):
        rec["aivat"] = aivat.finish(payoff, ended_by_fold=steps[-1]["atom"] == 0,
                                    final_committed=np.asarray(s.committed, np.float64).tolist())
        rec["aivat_seconds"] = rec["aivat"].pop("seconds")
    records.append(rec)
    if log is not None:
      log.write(json.dumps(rec) + "\n")
      log.flush()
    if progress:
      tot = np.array([[r["payoff"], r["control"]] for r in records]) / env.big_blind * 1e3
      line = (f"hand {i:4d} seat {seat} {rec['payoff']:+8.0f} (control {rec['control']:+8.0f}) "
              f"{rec['resolves']}/{rec['decisions']} re-solved, {rec['seconds']:5.1f}s | "
              f"mean {tot[:, 0].mean():+8.0f} mbb, diff {(tot[:, 0] - tot[:, 1]).mean():+8.0f}")
      if aivat is not None:
        av = [r["payoff"] + sum(r["aivat"][t] for t in ("deal", "board", "runout", "ours",
                                                       "opponent"))
              for r in records if "aivat" in r]
        line += (f", AIVAT {np.mean(av) / env.big_blind * 1e3:+8.0f} "
                 f"({rec.get('aivat_seconds', 0.0):.0f}s)")
      print(line, flush=True)
  return records


def report_play(recs, mbb) -> None:
  """Prints a match's payoffs (±95%), the paired difference and solve statistics."""
  pay = np.array([r["payoff"] for r in recs]) * mbb
  ctl = np.array([r["control"] for r in recs]) * mbb
  solves = [s for r in recs for s in r["solves"]]
  se = lambda x: 1.96 * x.std() / np.sqrt(len(x))
  print(f"  re-solving {pay.mean():+.0f} ± {se(pay):.0f}, control {ctl.mean():+.0f} ± "
        f"{se(ctl):.0f}, paired difference {(pay - ctl).mean():+.0f} ± {se(pay - ctl):.0f} "
        "mbb/hand")
  dec = sum(r["decisions"] for r in recs)
  print(f"  {dec} decisions, {len(solves)} re-solves ({1 - len(solves) / max(dec, 1):.0%} "
        f"reused), {np.mean([s['t_total'] for s in solves]):.1f}s per re-solve "
        f"(leaves {np.mean([s['t_leaves'] for s in solves]):.1f}s), "
        f"{np.mean([s['leaves'] for s in solves]):.0f} leaves, CFR-BR gap "
        f"{np.mean([s['gap'] for s in solves]):.3f} chips; truncated hands "
        f"{np.mean([r['truncated'] for r in recs]):.1%}")
  for st, name in enumerate(("preflop", "flop", "turn", "river")):
    n = sum(s["street"] == (0, 3, 4, 5)[st] for s in solves)
    print(f"  {name:>7}: {n} re-solves")
