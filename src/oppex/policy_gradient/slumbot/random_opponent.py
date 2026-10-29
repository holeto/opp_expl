"""A local uniformly random opponent behind Slumbot's client interface.

``RandomOpponent`` has the ``new_hand`` / ``act`` methods of ``SlumbotClient``
and answers with responses in Slumbot's format, so ``play.play_hand`` plays it
through the same translation path and ``analyze.py`` reads its logs unchanged.
In the logs and the analysis it still appears as ``slumbot``.

Each decision is uniform over the legal atoms of the training abstraction,
applied to the real game: fold (only when facing a bet), check/call, every bet
bin as a pot fraction, and all-in (not when calling already covers the stack,
as with ``drop_duplicate_allin``). Bets are sized and clipped by
``play.real_incr``, exactly like our own. Seats alternate every hand, as they
do against Slumbot.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...envs.betting import ALL_IN, CALL, FOLD, NUM_POINT_ATOMS
from ...envs.hunl_holdem import hand_rank7
from .play import _RANKS, _SUITS, RealGame, real_incr

_VISIBLE = (0, 3, 4, 5)
_rank7 = jax.jit(hand_rank7)


class RandomOpponent:
  def __init__(self, bins: np.ndarray, seed: int = 0) -> None:
    self.bins = np.asarray(bins)
    self.rng = np.random.default_rng(seed)
    self.n_requests = 0
    self.n_hands = 0

  def new_hand(self) -> dict[str, Any]:
    self.n_requests += 1
    deck = self.rng.permutation(52)
    self.cards = [_RANKS[c // 4] + _SUITS[c % 4] for c in deck[:9]]
    self.deck = deck[:9]
    # client_pos 1 (we are the button) on even hands, as Slumbot alternates.
    self.client_pos = 1 - self.n_hands % 2
    self.n_hands += 1
    self.client_seat = 1 - self.client_pos
    self.game = RealGame()
    self.action = ""
    return self._bot_moves()

  def act(self, incr: str) -> dict[str, Any]:
    self.n_requests += 1
    self._apply(incr)
    return self._bot_moves()

  def _legal_atoms(self, seat: int) -> list[int]:
    # The legal atoms of ``betting.legal_atoms`` on the real game.
    g = self.game
    call, stack = g.call_amount(seat), g.stack(seat)
    atoms = [CALL]
    if call > 0:
      atoms.append(FOLD)
    if stack > call:
      atoms.append(ALL_IN)
      if stack >= call + g.min_raise() and g.stack(1 - seat) > 0:
        atoms += [NUM_POINT_ATOMS + i for i in range(len(self.bins))]
    return atoms

  def _apply(self, incr: str) -> None:
    # Appends the token, plus a '/' when it closes a street without ending
    # the hand.
    street = self.game.street
    self.game.apply(incr[0], int(incr[1:]) if incr[0] == "b" else None)
    self.action += incr
    if not self.game.done and self.game.street != street:
      self.action += "/"

  def _bot_moves(self) -> dict[str, Any]:
    bot = 1 - self.client_seat
    while not self.game.done and self.game.cur == bot:
      atom = int(self.rng.choice(self._legal_atoms(bot)))
      self._apply(real_incr(self.game, bot, atom, self.bins)[0])
    return self._response()

  def _response(self) -> dict[str, Any]:
    g = self.game
    r = {
      "action": self.action, "client_pos": self.client_pos,
      "hole_cards": self._hole(self.client_seat),
      "board": self.cards[4: 4 + _VISIBLE[g.street]],
    }
    if g.done:
      r["bot_hole_cards"] = self._hole(1 - self.client_seat)
      r["winnings"] = self._winnings()
      if not self.action.endswith("f"):
        r["board"] = self.cards[4:9]
    return r

  def _hole(self, seat: int) -> list[str]:
    # Seat 0 holds cards 0-1 and seat 1 cards 2-3, as in ``init_state``.
    return self.cards[2 * seat: 2 * seat + 2]

  def _winnings(self) -> int:
    # Our net chips. Equal stacks mean no side pots: the loser forfeits
    # exactly what they put in.
    g, me = self.game, self.client_seat
    if self.action.endswith("f"):
      folder = g.cur
      return g.committed[1 - me] if folder != me else -g.committed[me]
    board = self.deck[4:9]
    mine = int(_rank7(jnp.asarray(np.r_[self.deck[2 * me: 2 * me + 2], board])))
    opp = 1 - me
    theirs = int(_rank7(jnp.asarray(np.r_[self.deck[2 * opp: 2 * opp + 2], board])))
    return int(np.sign(mine - theirs)) * g.committed[opp]

