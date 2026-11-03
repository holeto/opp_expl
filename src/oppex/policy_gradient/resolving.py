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

**The opponent model** is the blueprint, mixed with ``eps`` of uniform over its
legal actions. Against the blueprint itself ``eps`` only blurs the belief a
little; against Slumbot it keeps an action the blueprint would never take from
zeroing the belief. Probabilities of bet atoms the tree struck as duplicates
are added to their survivor, because only the public state is observed.

**Leaf values** ``U(l, t) = Σ_h r_o(l, h) · V(l, t, h)`` are estimated by
rollouts. Each one draws ``h`` from ``r_o(l, ·)``, the unknown board cards from
the rest of the deck, replays the leaf's history, and plays our seat as the
blueprint biased by ``t`` and the opponent as the blueprint. The uniforms
behind the opponent hand, the runout and every action are shared across
leaves and continuations (common random numbers), so the differences the
solver acts on are less noisy than the values. Leaves the opponent reaches
with negligible mass are not rolled out (value 0), and a cached node below
that mass is never played from: it is re-solved instead.
"""

from __future__ import annotations

import time
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
from ..policy import policy_probs
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

  def __init__(self, bp: Blueprint, eps: float = 1e-3, chunk: int = 16) -> None:
    self.bp, self.eps, self.chunk = bp, float(eps), chunk
    env = bp.env
    key = jax.random.PRNGKey(0)
    hands = jnp.asarray(HAND_CARDS)
    infoset_b = jax.vmap(env.information_set, in_axes=(0, None, None))
    legal_b = jax.vmap(env.legal_actions, in_axes=(0, None))
    eps = self.eps

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
      p = (1.0 - eps) * p + eps * legal / legal.sum(-1, keepdims=True)
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

  def belief(self, rules, history, our_cards, board, our_seat) -> np.ndarray:
    """``r_o(h)`` at the end of ``history``: compatibility times opponent probabilities."""
    states = replay_betting(rules, history)
    opp_steps = [j for j, (bs, _) in enumerate(states[:-1]) if int(bs.cur_player) != our_seat]
    r = compatible(our_cards, board)
    if opp_steps:
      p = self.probs([tuple(history[:j]) for j in opp_steps], our_cards, board, our_seat)
      for q, j in enumerate(opp_steps):
        alias = states[j][1]
        r = r * _fold(p[q], alias)[:, alias[int(history[j])]]
    return r


def subgame_reach(tree, belief, model: OpponentModel, our_cards, board, our_seat):
  """``r_o`` at every node ``(n_nodes, H)`` and terminal ``(n_terminals, H)``."""
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
  return r_node, r_term


# ── Values ───────────────────────────────────────────────────────────────────


def terminal_values(tree, r_term, our_seat, our_cards, board, ev) -> np.ndarray:
  """``(n_terminals,)`` contracted chips to us; matrix leaves are left at 0.

  Showdowns use the expected sign of our hand against each opponent hand:
  the ``EV`` row preflop, an exact runout enumeration on a known board.
  """
  if len(board) == 0:
    sign = np.asarray(ev[HAND_INDEX[our_cards[0], our_cards[1]]], np.float64)
  else:
    sign = sign_vs_range(our_cards, board)
  to_us = 1.0 if our_seat == 0 else -1.0
  out = np.zeros(len(tree.terminals))
  for j, t in enumerate(tree.terminals):
    if t.kind == FOLD_T:
      out[j] = to_us * t.const * r_term[j].sum()
    elif t.kind == SHOWDOWN_T:
      c = np.asarray(t.betting.committed)
      if c[0] != c[1]:
        raise AssertionError(f"unequal commitments at showdown: {c}")
      out[j] = t.stake * (r_term[j] * sign).sum()
  return out


class LeafRollouts:
  """Jitted rollouts valuing ``chunk`` leaves × ``n_games`` per call."""

  def __init__(self, bp: Blueprint, n_games: int = 128, chunk: int = 32,
               factor: float = PLURIBUS_FACTOR) -> None:
    self.bp, self.n_games, self.chunk = bp, n_games, chunk
    env = bp.env
    mult = bias_multipliers(env.num_actions, factor)
    hands = jnp.asarray(HAND_CARDS)
    n_b = chunk * n_games
    leaf = jnp.repeat(jnp.arange(chunk), n_games)
    game = jnp.tile(jnp.arange(n_games), chunk)

    @jax.jit
    def run(key, paths, q, ours, board, n_known, our_seat, t):
      # Mean chips to us per leaf and the truncated fraction. Game i of every
      # leaf shares its uniforms with game i of every other leaf and t.
      keys = jax.random.split(key, n_games)
      u_h = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 0)))(keys)[game]
      cdf = jnp.cumsum(q, -1)[leaf]
      h = jnp.minimum((cdf < u_h[:, None] * cdf[:, -1:]).sum(-1), N_HANDS - 1)
      opp_cards = hands[h]

      # Unknown board cards: a random order of the deck, used cards last.
      pos = jnp.arange(5)
      known = (pos < n_known)[:, None] * jax.nn.one_hot(board, 52)
      used = (jax.nn.one_hot(ours, 52).sum(0) + known.sum(0))[None, :] \
        + jax.nn.one_hot(opp_cards, 52).sum(1)
      keys_card = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 1), (52,)))(keys)
      order = jnp.argsort(keys_card[game] + 2.0 * (used > 0), -1)
      drawn = jnp.take_along_axis(order, jnp.clip(pos - n_known, 0, 4)[None, :].repeat(n_b, 0), -1)
      full = jnp.where(pos < n_known, board[None, :], drawn).astype(jnp.int32)

      ours_b = jnp.broadcast_to(ours, (n_b, 2))
      hole = jnp.where(our_seat == 0, jnp.stack([ours_b, opp_cards], 1),
                       jnp.stack([opp_cards, ours_b], 1))
      s0 = env.init_state(jax.random.PRNGKey(0))
      s = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_b,) + x.shape), s0)
      s = s._replace(hole_cards=hole, board=full)
      s = replay(env, s, paths[leaf])
      game_keys = jax.vmap(lambda k: jax.random.fold_in(k, 2))(keys)[game]
      t0, t1 = jnp.where(our_seat == 0, t, 0), jnp.where(our_seat == 1, t, 0)
      s, pay = play_out(bp, s, game_keys, t0, t1, mult)
      pay = pay * env.starting_stack * jnp.where(our_seat == 0, 1.0, -1.0)
      return pay.reshape(chunk, n_games).mean(1), (~s.done).mean()

    self._run = run

  def values(self, tree, r_term, our_cards, board, our_seat, seed=0, min_mass=0.0):
    """``(n_leaves, T)`` leaf values ``Z_l · E[chips | t]`` and the truncated fraction."""
    leaves = list(tree.leaves)
    z = r_term[leaves].sum(-1)
    out = np.zeros((len(leaves), N_TRANSFORMS))
    live = [i for i in range(len(leaves)) if z[i] > min_mass]
    if not live:
      return out, 0.0
    padded = pad_paths([tree.terminals[leaves[i]].path for i in live], 8)
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
        mean, tr = self._run(key, jnp.asarray(padded[idx]), jnp.asarray(q[idx], jnp.float32),
                             ours, full, jnp.int32(len(board)), jnp.int32(our_seat),
                             jnp.int32(t))
        out[[live[i] for i in idx[:n]], t] = np.asarray(mean, np.float64)[:n]
        trunc.append(float(tr))
    return out * z[:, None], float(np.mean(trunc))


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


# ── Solving one infoset ──────────────────────────────────────────────────────


class Solution(NamedTuple):
  game: infoset_cfr.InfosetGame
  sigma: np.ndarray          # (n_nodes, A) CFR average, our nodes only meaningful
  tau: np.ndarray            # (n_leaves, T)
  z: float                   # opponent mass at the root, Σ_h r_o(root, h)
  node_of: dict              # history -> node id, our nodes above min_mass
  stats: dict


class Resolver:
  """Re-solving player for one seat of one hand at a time.

  ``probs(history, board, ...)`` is all a player loop needs; it decides
  between reusing the cached subgame and re-solving.
  """

  def __init__(self, bp: Blueprint, ev, depth: int | None = 4, leaf_games: int = 128,
               iters: int = 500, eps: float = 1e-3, factor: float = PLURIBUS_FACTOR,
               min_mass: float = 1e-4, chunk_leaves: int = 32, seed: int = 0,
               leaf_fn=None) -> None:
    self.bp, self.ev, self.depth, self.iters = bp, ev, depth, iters
    self.rules = bp.env.rules
    self.min_mass, self.seed = min_mass, seed
    self.model = OpponentModel(bp, eps)
    self.rollouts = LeafRollouts(bp, leaf_games, chunk_leaves, factor)
    # leaf_fn(tree, r_term, our_cards, board, seat) overrides the rollouts.
    self.leaf_fn = leaf_fn
    self.n_solves = 0
    self.new_hand(0, (0, 1))

  def new_hand(self, seat: int, hole) -> None:
    self.seat, self.cards = int(seat), tuple(int(c) for c in hole)
    self.solution: Solution | None = None
    self.log: list[dict] = []

  def solve(self, history, board) -> Solution:
    """The subgame rooted after ``history``, solved for our seat.

    The root is normally our decision; the validation also roots it at the
    opponent's first decision, which the formulation allows.
    """
    t0 = time.time()
    history, board = tuple(int(a) for a in history), [int(c) for c in board]
    root = replay_betting(self.rules, history)[-1][0]
    tree = build_subgame(self.rules, root, history, self.depth)
    t1 = time.time()
    belief = self.model.belief(self.rules, history, self.cards, board, self.seat)
    r_node, r_term = subgame_reach(tree, belief, self.model, self.cards, board, self.seat)
    t2 = time.time()
    z = belief.sum()
    if z <= 0:
      raise AssertionError(f"the opponent cannot have reached {history}")
    term = terminal_values(tree, r_term, self.seat, self.cards, board, self.ev)
    t3 = time.time()
    trunc = 0.0
    if self.leaf_fn is not None:
      leaf = self.leaf_fn(tree, r_term, self.cards, board, self.seat)
    else:
      leaf, trunc = self.rollouts.values(tree, r_term, self.cards, board, self.seat,
                                         seed=self.seed + self.n_solves,
                                         min_mass=self.min_mass * z)
    t4 = time.time()
    game = infoset_cfr.InfosetGame(tree, self.seat, term, leaf)
    sigma, tau = infoset_cfr.solve(game, self.iters)
    br, _, _, _ = infoset_cfr.best_response(game)
    value = infoset_cfr.policy_value(game, sigma, tau)
    t5 = time.time()
    self.n_solves += 1
    mass = r_node.sum(-1) / z
    node_of = {n.path: i for i, n in enumerate(tree.nodes)
               if n.player == self.seat and mass[i] >= self.min_mass}
    stats = dict(
      history=list(history), street=len(board), nodes=len(tree.nodes),
      leaves=len(tree.leaves), value=value / z, br=br / z, gap=(br - value) / z,
      truncated=trunc, t_build=t1 - t0, t_reach=t2 - t1, t_terminals=t3 - t2,
      t_leaves=t4 - t3, t_cfr=t5 - t4, t_total=t5 - t0,
    )
    return Solution(game, sigma, tau, float(z), node_of, stats)

  def probs(self, history, board, n_actions: int) -> tuple[np.ndarray, bool]:
    """Our env-atom probabilities after ``history``, and whether we re-solved."""
    history = tuple(int(a) for a in history)
    sol = self.solution
    resolved = sol is None or history not in sol.node_of
    if resolved:
      if int(replay_betting(self.rules, history)[-1][0].cur_player) != self.seat:
        raise AssertionError(f"history {history} does not end at our decision")
      sol = self.solution = self.solve(history, board)
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
    self.resolver = Resolver(self, ev, **resolver_kw)
    self.n_decisions = self.n_resolves = 0

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
                      log=None, progress: bool = True) -> list[dict]:
  """Hands against the blueprint in the env, each also played by the blueprint.

  Seats alternate. The control replays the same deal with the blueprint in our
  seat and the same uniform per env step, so ``payoff - control`` is a paired
  difference. Returns one record per hand, also written as JSON lines to
  ``log``.
  """
  import json

  env = bp.env
  apply = jax.jit(lambda s, a: env.apply_action(s, jnp.full((2,), a, jnp.int32)))
  records = []
  for i in range(n_hands):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), i)
    seat = i % 2
    u = np.random.default_rng([seed, i]).random(env.max_length + 1)
    rec = {"hand": i, "seat": seat}
    for mode in ("resolve", "control"):
      s = env.init_state(key)
      hole = [int(c) for c in np.asarray(s.hole_cards)[seat]]
      resolver.new_hand(seat, hole)
      payoff, n_dec, n_res = 0.0, 0, 0
      t0 = time.time()
      while not bool(s.done) and int(s.step) < env.max_length:
        cur = int(s.cur_player)
        probs, legal = bp.policy(s, cur)
        if mode == "resolve" and cur == seat:
          history, board = _view(s, seat)
          probs, resolved = resolver.probs(history, board, len(legal))
          n_dec, n_res = n_dec + 1, n_res + int(resolved)
        cdf = np.cumsum(probs)
        atom = int(min(np.searchsorted(cdf, u[int(s.step)] * cdf[-1], side="right"),
                       len(probs) - 1))
        if not legal[atom]:
          atom = int(np.argmax(probs))
        s, rewards, _, _ = apply(s, atom)
        payoff += float(np.asarray(rewards)[seat]) * env.starting_stack
      if mode == "resolve":
        rec.update(payoff=payoff, decisions=n_dec, resolves=n_res,
                   seconds=time.time() - t0, solves=resolver.log,
                   truncated=not bool(s.done), history=_view(s, seat)[0])
      else:
        rec["control"] = payoff
    records.append(rec)
    if log is not None:
      log.write(json.dumps(rec) + "\n")
      log.flush()
    if progress:
      tot = np.array([[r["payoff"], r["control"]] for r in records]) / env.big_blind * 1e3
      print(f"hand {i:4d} seat {seat} {rec['payoff']:+8.0f} (control {rec['control']:+8.0f}) "
            f"{rec['resolves']}/{rec['decisions']} re-solved, {rec['seconds']:5.1f}s | "
            f"mean {tot[:, 0].mean():+8.0f} mbb, diff {(tot[:, 0] - tot[:, 1]).mean():+8.0f}",
            flush=True)
  return records
