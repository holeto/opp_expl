"""Slumbot playing one seat of env hands, for matches in the env.

``SlumbotOpponent`` is the ``opponent`` of ``resolving.play_vs_blueprint``.
Each hand gets a translation type ``u`` (one slumbot.com hand number); Slumbot
then plays its exact moves from the C++ library on its own slumbot.com
string, in which its bets keep their exact chips. Its moves enter the env
through the same conversions the model and the rollouts use, so the env hand
is the one ``SlumbotModel`` reasons about.

**The logged distribution is the model's marginal.** A move is pure given
Slumbot's hand and ``u``, but ``u`` is hidden; given the history, the move is
distributed as ``SlumbotModel.probs`` at Slumbot's hand (the mixture over the
posterior of ``u``). That is the distribution AIVAT's opponent terms need to
be zero-mean.
"""

from __future__ import annotations

import numpy as np

from ...cfr.cards import HAND_INDEX
from ...envs.betting import CALL
from . import native
from .model import SlumbotModel, our_token, slumbot_atom
from .play import RealGame
from .rollouts import _KIND_TOKEN, _advance

_VISIBLE = (0, 3, 4, 5)


class SlumbotOpponent:
  """Slumbot in the env seat ``1 - our_seat`` of one hand at a time."""

  def __init__(self, model: SlumbotModel) -> None:
    self.model, self.lib = model, model.lib
    self.stats = {"hands": 0, "moves": 0, "refused": [], "desync": 0}

  def new_hand(self, our_seat: int, opp_cards, full_board, u: float) -> None:
    self.seat, self.opp = int(our_seat), 1 - int(our_seat)
    self.cards = np.asarray(opp_cards, np.int32)
    self.h = int(HAND_INDEX[self.cards[0], self.cards[1]])
    self.board = np.asarray(full_board, np.int32)
    self.u = float(u)
    self.actions, self.game = "", RealGame()
    # The token of the move act() just returned, for observe().
    self._token = None
    self.stats["hands"] += 1

  def act(self, history, r: float) -> tuple[int, np.ndarray]:
    """Slumbot's env atom after ``history`` and the marginal it is drawn from."""
    history = tuple(int(a) for a in history)
    st = self.model._step(history)
    vis = list(self.board[: _VISIBLE[int(st.bs.street)]])
    probs = self.model.probs([history], [], vis, self.seat)[0, self.h]
    kind, bet, status = self.lib.act_batch(self.seat, [self.actions], self.board[None],
                                           self.cards[None], [self.u], [r])
    self.stats["moves"] += 1
    if status[0] != native.SB_OK:
      # Slumbot cannot answer (a desync): check/call, as the model does.
      self.stats["refused"].append((self.actions, int(status[0])))
      return CALL, probs
    totals, raise_ok = self.model._totals(st, self.opp)
    atom, _ = slumbot_atom(int(kind[0]), int(bet[0]), totals, raise_ok, self.game, self.opp,
                           self.model.scale)
    self._token = _KIND_TOKEN.get(int(kind[0])) or f"b{int(bet[0])}"
    return atom, probs

  def observe(self, history, atom: int, actor: int) -> None:
    """Moves Slumbot's string past the move ``atom`` played after ``history``."""
    if actor == self.opp:
      tok = self._token or ("c" if self.game.call_amount(self.opp) > 0 else "k")
      self._token = None
    else:
      st = self.model._step(tuple(int(a) for a in history))
      a = st.alias[int(atom)]
      totals, _ = self.model._totals(st, self.seat)
      tok, _ = our_token(a, float(totals[a]), float(np.max(st.bs.committed_street)),
                         self.game, self.seat, self.model.scale)
    nxt = _advance(self.game, self.actions, tok)
    if nxt is None:
      self.stats["desync"] += 1
      return
    self.game, self.actions = nxt
