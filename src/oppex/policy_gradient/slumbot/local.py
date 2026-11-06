"""Slumbot from the C++ library behind ``SlumbotClient``'s interface.

``LocalSlumbot`` answers ``new_hand`` / ``act`` the way slumbot.com does, so
``play.play_hand`` runs the exact slumbot.com code path against it without
the network: a dress rehearsal for the real server. Each hand draws a hidden
translation type ``u``, as a slumbot.com hand number would set it, and every
Slumbot move comes from ``sb_act_batch`` on the exact string, i.e. the live
agent's move. Deal, seats and payoffs are ``RandomOpponent``'s. Needs
``SLUMBOT_DIR``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import native
from .random_opponent import RandomOpponent

_KIND_TOKEN = {native.SB_FOLD: "f", native.SB_CHECK: "k", native.SB_CALL: "c"}


class LocalSlumbot(RandomOpponent):
  def __init__(self, seed: int = 0) -> None:
    super().__init__(np.zeros(0), seed)
    self.lib = native.get()
    # us: the hidden translation type of each hand, for checks after play.
    self.stats = {"moves": 0, "refused": [], "us": []}

  def new_hand(self) -> dict[str, Any]:
    self.u = float(self.rng.random())
    self.stats["us"].append(self.u)
    return super().new_hand()

  def _bot_moves(self) -> dict[str, Any]:
    bot = 1 - self.client_seat
    while not self.game.done and self.game.cur == bot:
      kind, bet, status = self.lib.act_batch(
        self.client_seat, [self.action], self.deck[None, 4:9],
        self.deck[None, 2 * bot: 2 * bot + 2], [self.u], [self.rng.random()])
      self.stats["moves"] += 1
      if status[0] != native.SB_OK:
        # The live agent would have no move either: check/call.
        self.stats["refused"].append((self.action, int(status[0])))
        tok = "c" if self.game.call_amount(bot) > 0 else "k"
      else:
        tok = _KIND_TOKEN.get(int(kind[0])) or f"b{int(bet[0])}"
      self._apply(tok)
    return self._response()
