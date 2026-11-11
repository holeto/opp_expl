"""Leaf rollouts against Slumbot, its moves taken from the C++ library.

``SlumbotRollouts`` has the interface of ``resolving.LeafRollouts``
(``values``), so a ``Resolver`` given it and a ``SlumbotModel`` re-solves
against Slumbot. The deal is ``LeafRollouts``'s: the opponent's hand ``h``
from the leaf's reach, the unknown board cards from the rest of the deck, the
same common random numbers.

**Slumbot's translation type is drawn from its posterior.** At a leaf, Slumbot
has already acted, and its actions say something about ``u``. Given the
sampled ``h``, a world ``w`` of the leaf history (``model.py``) is drawn with
probability ∝ ``|w| L_w(h)``, then ``u`` uniformly inside it. The posterior is
flat inside a world, and our bets after the leaf add translations that split
the world further; a uniform ``u`` inside it resolves them with the right
probabilities. The game then continues from the world's exact slumbot.com
string, so Slumbot sees the chips it bet itself.

**The loop runs on the host.** Each env step evaluates our seat (the
blueprint biased by ``t``) for every game in one jitted call, asks Slumbot for
the games where it is to act in one ``sb_act_batch`` call, and applies the
atoms in another jitted call. Our atoms become tokens and Slumbot's actions
become atoms through the same functions the model uses (``our_token``,
``slumbot_atom``), so rollouts and beliefs read the env the same way.

**What a game pays.** Slumbot's bets are rounded to our bins in the env, so
the env's chips are not the chips of the hand. By default a game pays the env's
payoff, which is right when the env is the game being played
(``resolve_vs_slumbot.py``). With ``real_chips`` it pays what the hand pays on
slumbot.com, from Slumbot's chip tracker (divided by the scale): the fold or
showdown there, when Slumbot's string ends. A game stops at that point even if
the env hand would go on, which it does when Slumbot's whole stack is in but
the rounded env stacks are not (*all-in drift*). If the env hand ends first,
the showdown is scored on the chips of the string so far. The blueprint in our
seat still sees the env's rounded state.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ...cfr.cards import HAND_CARDS
from ...envs import betting
from ...envs.betting import CALL
from ...envs.hunl_holdem import hand_rank7
from ...policy import policy_probs, sample_atom
from ..blueprint import PLURIBUS_FACTOR, bias_multipliers, transform_policy
from ..resolving import LeafRollouts, make_deal
from ..tree_policy import where_b
from . import native
from .model import SlumbotModel, _clone, _measure, our_token, slumbot_atom

_KIND_TOKEN = {native.SB_FOLD: "f", native.SB_CHECK: "k", native.SB_CALL: "c"}


def _point(ivs, frac: float) -> float:
  # The point at fraction `frac` of the measure of the intervals `ivs`.
  target, acc = frac * _measure(ivs), 0.0
  for lo, hi in ivs:
    if acc + (hi - lo) > target:
      return lo + (target - acc)
    acc += hi - lo
  lo, hi = ivs[-1]
  return np.nextafter(hi, lo)


def _sample_first_axis(wts: np.ndarray, u: np.ndarray) -> np.ndarray:
  # policy.sample_atom along axis 0 in float64: for each column of the
  # unnormalised (K, G) weights, the row drawn by its uniform. A row of
  # weight 0 is never drawn, even when u is exactly 0.
  cum = np.cumsum(wts, 0)
  k = (cum <= u * cum[-1]).sum(0)
  last = len(wts) - 1 - np.argmax(wts[::-1] > 0, 0)
  return np.minimum(k, last)


def _real_payoff(game, folder: int | None, sign: int, seat: int) -> float:
  # The finished hand's chips to `seat`, in the tracker's chips. folder: who
  # folded, or None for a showdown, whose sign is +1 if `seat` wins. A fold
  # pays the folder's chips; at a showdown an uncalled excess goes back.
  if folder is not None:
    return float(-game.committed[seat] if folder == seat else game.committed[1 - seat])
  return float(sign * min(game.committed))


def _advance(game, actions: str, token: str):
  # Slumbot's string and chip tracker after one token, '/' when the street
  # ends. Returns None if the token is illegal in Slumbot's chips.
  g = _clone(game)
  street = g.street
  try:
    g.apply(token[0], int(token[1:]) if token[0] == "b" else None)
  except AssertionError:
    return None
  return g, actions + token + ("/" if not g.done and g.street != street else "")


class SlumbotRollouts(LeafRollouts):
  """Rollouts valuing ``chunk`` leaves × ``n_games`` per call against Slumbot."""

  def __init__(self, bp, model: SlumbotModel, n_games: int = 128, chunk: int = 32,
               factor: float = PLURIBUS_FACTOR, real_chips: bool = False) -> None:
    self.bp, self.model, self.n_games, self.chunk = bp, model, n_games, chunk
    self.factor, self.real_chips = factor, real_chips
    self.lib = model.lib
    env = bp.env
    rules = env.rules
    mult = bias_multipliers(env.num_actions, factor)
    deal = make_deal(env, chunk, n_games)
    game = jnp.tile(jnp.arange(n_games), chunk)
    obs_key = jax.random.PRNGKey(0)
    L = env.max_length
    n_act = env.num_actions
    infoset_b = jax.vmap(env.information_set, in_axes=(0, 0, None))
    legal_b = jax.vmap(env.legal_actions)
    apply_b = jax.vmap(env.apply_action)

    @jax.jit
    def start(key, paths, q, ours, board, n_known, our_seat):
      # The deal, plus per-game uniforms for the actions (as in play_out), the
      # world and the translation type inside it.
      s, h, keys = deal(key, paths, q, ours, board, n_known, our_seat)
      fold = lambda i: jax.vmap(lambda k: jax.random.fold_in(k, i))(keys)[game]
      uni = lambda i: jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, i)))(keys)[game]
      return s, h, fold(2), uni(3), uni(4)

    @jax.jit
    def observe(s, game_keys, t):
      # For the player to act in each game: our biased-blueprint atom (used
      # only where that is us), env legality, every atom's street total and
      # the step's uniform.
      cur = s.cur_player
      live = ~s.done & (s.step < L)
      legal = legal_b(s, cur)
      probs = policy_probs(bp.eval_fn(bp.params, infoset_b(s, cur, obs_key)), legal)
      probs = transform_policy(probs, legal, t, mult)
      u = jax.vmap(lambda k, i: jax.random.uniform(jax.random.fold_in(k, i)))(game_keys, s.step)
      atom = sample_atom(probs, u)
      bs = jax.vmap(env._to_betting)(s)
      added = jax.vmap(lambda b: jax.vmap(lambda a: betting.chips_added(rules, b, a))(
        jnp.arange(n_act)))(bs)
      own = jnp.take_along_axis(bs.committed_street, cur[:, None], -1)
      totals = own + added
      return cur, live, atom, legal, totals, bs.committed_street.max(-1), u

    @jax.jit
    def apply(s, atoms, live):
      new_s, rewards, _, _ = apply_b(s, jnp.stack([atoms, atoms], -1).astype(jnp.int32))
      return where_b(live, new_s, s), jnp.where(live, rewards[:, 0], 0.0)

    @jax.jit
    def showdown_sign(ours, opp_cards, boards):
      # +1 if our hand wins each game's showdown, -1 if it loses, 0 on a tie.
      rank = jax.vmap(lambda hole, b: hand_rank7(jnp.concatenate([hole, b])))
      ours_b = jnp.broadcast_to(ours, opp_cards.shape)
      return jnp.sign(rank(ours_b, boards) - rank(opp_cards, boards))

    self._start, self._observe, self._apply = start, observe, apply
    self._showdown_sign = showdown_sign
    # When a list, each chunk appends (leaf paths, h, Slumbot's first atom per
    # game or -1), for validation.
    self.record = None
    # When a list, each chunk appends (env pay, real pay, scored from real
    # chips, env and real chips agree at a shared end), for validation.
    self.record_pay = None
    # refused: (string, status) of the first refusals, for diagnosis.
    # string_ended: games stopped because Slumbot's string ended while the env
    # hand went on; env_ended: env hands that ended before Slumbot's string.
    self.stats = {"games": 0, "slumbot_actions": 0, "refusals": 0, "refused": [], "desync": 0,
                  "dead_leaves": 0, "string_ended": 0, "env_ended": 0,
                  "raise_clamped": 0, "seconds_slumbot": 0.0, "seconds_total": 0.0}

  def resized(self, n_games: int, chunk: int) -> SlumbotRollouts:
    """These rollouts with ``n_games`` per leaf and ``chunk`` leaves per call.

    They take a board longer than the leaf's street, as AIVAT's boards valued
    by rollouts need: the deal fixes the extra cards, Slumbot's moves read
    each game's full board, and the model cuts the board to the street of
    every query, so the leaf's worlds do not depend on the extra cards. The
    model is shared, so its caches serve both.
    """
    return SlumbotRollouts(self.bp, self.model, n_games, chunk, self.factor, self.real_chips)

  def _worlds(self, leaf_paths, board, seat, h, u_world, u_type):
    # Per game: Slumbot's string, chip tracker and translation type, drawn
    # from the posterior over the leaf's worlds given the game's hand, and
    # whether the game is dead: its leaf lies behind an atom Slumbot never
    # plays, so no world reaches it (AIVAT asks for such leaves, with weight
    # 0). Dead games are not played and are worth 0.
    n = self.n_games
    strings, games = [None] * len(h), [None] * len(h)
    u = np.zeros(len(h))
    dead = np.zeros(len(h), bool)
    bkey = tuple(int(c) for c in board)
    for c, path in enumerate(leaf_paths):
      ws = self.model._walk(tuple(int(a) for a in path), bkey, seat)
      rows = np.arange(c * n, (c + 1) * n)
      if not ws:
        dead[rows] = True
        self.stats["dead_leaves"] += 1
        continue
      fixed = self.model.u is not None
      mass = np.array([1.0 if fixed else _measure(w.ivs) for w in ws])
      wts = mass[:, None] * np.stack([w.lik[h[rows]] for w in ws])
      tot = wts.sum(0)
      # A hand no world explains has reach 0, so it is never dealt; guard anyway.
      wts[:, tot <= 0] = mass[:, None]
      k = _sample_first_axis(wts, u_world[rows])
      for i, g in enumerate(rows):
        w = ws[k[i]]
        strings[g], games[g] = w.actions, w.game
        u[g] = self.model.u if fixed else _point(w.ivs, u_type[g])
    return strings, games, u, dead

  def _chunk(self, key, paths, q, ours, full, n_known, our_seat, t, leaf_paths, board):
    t0 = time.time()
    seat, opp_seat = int(our_seat), 1 - int(our_seat)
    scale = self.model.scale
    s, h, game_keys, u_world, u_type = self._start(key, paths, q, ours, full, n_known, our_seat)
    h, u_world, u_type = np.asarray(h), np.asarray(u_world), np.asarray(u_type)
    boards = np.asarray(s.board)
    opp_cards = HAND_CARDS[h]
    strings, games, u, dead = self._worlds(leaf_paths, board, seat, h, u_world, u_type)
    payoff = np.zeros(len(h))
    # With real_chips: games whose real payoff is fixed, the payoffs, and games
    # whose tracker is unusable after an illegal token, which keep the env's.
    scored, broken = np.zeros(len(h), bool), np.zeros(len(h), bool)
    real_pay = np.zeros(len(h))
    sign = None
    if self.real_chips:
      sign = np.asarray(self._showdown_sign(ours, jnp.asarray(opp_cards), jnp.asarray(boards)))
      # A world can already be over at the leaf: drift inside the subgame.
      for g in np.flatnonzero(~dead):
        if games[g].done:
          real_pay[g], scored[g] = _real_payoff(games[g], None, int(sign[g]), seat), True
          self.stats["string_ended"] += 1
    raise_atoms = np.arange(self.bp.env.num_actions) >= 2
    first = np.full(len(h), -1)
    t_slumbot = 0.0

    while True:
      cur, live, our_atom, legal, totals, smax, u_step = map(
        np.asarray, self._observe(s, game_keys, t))
      live = live & ~dead & ~scored
      if not live.any():
        break
      atoms = our_atom.astype(np.int32)
      tokens = {}
      theirs = np.flatnonzero(live & (cur == opp_seat))
      if theirs.size:
        t1 = time.time()
        kind, bet, status = self.lib.act_batch(
          seat, [strings[g] for g in theirs], boards[theirs], opp_cards[theirs], u[theirs],
          u_step[theirs])
        t_slumbot += time.time() - t1
        self.stats["slumbot_actions"] += int(theirs.size)
        for j, g in enumerate(theirs):
          if status[j] != native.SB_OK:
            # Slumbot cannot answer (a desync): check/call, as the model does.
            self.stats["refusals"] += 1
            if len(self.stats["refused"]) < 20:
              self.stats["refused"].append((strings[g], int(status[j])))
            atoms[g] = CALL
            tokens[g] = "c" if games[g].call_amount(opp_seat) > 0 else "k"
            continue
          raise_ok = legal[g] & raise_atoms & (totals[g] > smax[g] + 1e-9)
          atoms[g], _ = slumbot_atom(int(kind[j]), int(bet[j]), totals[g], raise_ok, games[g],
                                     opp_seat, scale)
          tokens[g] = _KIND_TOKEN.get(int(kind[j])) or f"b{int(bet[j])}"
      for g in np.flatnonzero(live & (cur == seat)):
        a = int(atoms[g])
        tokens[g], moved = our_token(a, float(totals[g, a]), float(smax[g]), games[g], seat,
                                     scale)
        self.stats["raise_clamped"] += moved
      if theirs.size:
        new = theirs[first[theirs] < 0]
        first[new] = atoms[new]
      s, rew = self._apply(s, jnp.asarray(atoms), jnp.asarray(live))
      payoff += np.asarray(rew, np.float64)
      env_over = np.asarray(s.done)
      for g, tok in tokens.items():
        nxt = _advance(games[g], strings[g], tok)
        if nxt is None:
          self.stats["desync"] += 1
          broken[g] = True
          continue
        games[g], strings[g] = nxt
        if self.real_chips and games[g].done:
          folder = int(cur[g]) if tok == "f" else None
          real_pay[g], scored[g] = _real_payoff(games[g], folder, int(sign[g]), seat), True
          self.stats["string_ended"] += int(not env_over[g])

    env = self.bp.env
    pay = payoff * env.starting_stack * (1.0 if seat == 0 else -1.0)
    env_done = np.asarray(s.done)
    if self.real_chips:
      # The env hand ended first: no fold, since a fold ends the string too.
      for g in np.flatnonzero(env_done & ~scored & ~dead & ~broken):
        real_pay[g], scored[g] = _real_payoff(games[g], None, int(sign[g]), seat), True
        self.stats["env_ended"] += 1
      if self.record_pay is not None:
        agree = np.array([not dead[g] and games[g].done and bool(env_done[g])
                          and list(games[g].committed)
                          == [int(c) * scale for c in np.asarray(s.committed)[g]]
                          for g in range(len(h))])
        self.record_pay.append((pay.copy(), real_pay / scale, scored.copy(), agree))
      use = scored & ~broken
      pay = np.where(use, real_pay / scale, pay)
    self.stats["games"] += len(h)
    self.stats["seconds_slumbot"] += t_slumbot
    self.stats["seconds_total"] += time.time() - t0
    trunc = float((~env_done & ~dead & ~scored).mean())
    if self.record is not None:
      self.record.append((list(leaf_paths), h, first))
    return pay.reshape(self.chunk, self.n_games).mean(1), trunc
