"""The re-solver on slumbot.com, with Slumbot's exact model.

``SlumbotResolvingAgent`` is a ``ResolvingAgent`` for ``play.play_hand``
whose opponent model is ``SlumbotModel`` and whose leaves are rolled out
against Slumbot (``SlumbotRollouts``), as in the env match
(``resolve_vs_slumbot.py``). Only the crossing between the games changes:

  * Our atoms are sent as ``our_token``'s chips (the env street total × 50,
    moved into Slumbot's legal range), not as a fraction of the real pot.
    The model's worlds assume exactly these tokens; with any other amount no
    world would reproduce the real string.
  * Slumbot's bets enter the tree as the nearest raise atom by street total
    (``slumbot_atom``), the mapping the model and the rollouts use, instead
    of the pseudo-harmonic translation.
  * The real string after every step goes to ``SlumbotModel.set_real``, so
    the model conditions on Slumbot's exact chips.

The translation type ``u`` is set by slumbot.com's hand number, which the
server does not send, so it is marginalised per hand. **Possible extension,
not implemented:** if the hand numbers of one session are consecutive, ``u``
moves by a known shift from one hand to the next, and every finished hand
(Slumbot's cards and moves) narrows it. Carried across hands, the posterior
over ``u`` would soon pin it down and make Slumbot deterministic given its
hand. Whether other players' hands interleave on the server's counter is
untested.

Each hand's record carries ``steps`` in ``play_vs_blueprint``'s format.
Slumbot's steps hold the model's marginal at its revealed hand, the
distribution AIVAT's opponent terms need.
"""

from __future__ import annotations

import numpy as np

from ...cfr.cards import HAND_INDEX
from ...envs.betting import CALL, FOLD
from ..blueprint import PLURIBUS_FACTOR
from ..resolving import ResolvingAgent, _view
from . import native
from .model import SlumbotModel, our_token, slumbot_atom
from .play import RealGame, card_index, real_incr
from .rollouts import SlumbotRollouts, _advance

_VISIBLE = (0, 3, 4, 5)
_KIND = {"fold": native.SB_FOLD, "check": native.SB_CHECK, "call": native.SB_CALL}


class SlumbotResolvingAgent(ResolvingAgent):
  """``ResolvingAgent`` against Slumbot itself, for ``play.play_hand``."""

  def __init__(self, run_dir, checkpoint, which, ev, **resolver_kw) -> None:
    super().__init__(run_dir, checkpoint, which, ev, **resolver_kw)
    # desync: a real token illegal in our chip tracker. string_mismatch:
    # responses whose action string differs from ours (first few kept).
    # zero_prob: Slumbot moves the model gave probability 0 at its hand.
    self.stats = {"hands": 0, "desync": 0, "string_mismatch": [], "slumbot_steps": 0,
                  "zero_prob": 0, "lost_hands": 0, "bet_gap": []}

  def _resolver_kw(self, kw: dict) -> dict:
    kw = dict(kw)
    self.model = SlumbotModel(self.env.rules)
    self.rollouts = SlumbotRollouts(self, self.model, n_games=kw.pop("leaf_games", 128),
                                    chunk=kw.pop("chunk_leaves", 32),
                                    factor=kw.get("factor", PLURIBUS_FACTOR))
    return {**kw, "model": self.model, "rollouts": self.rollouts}

  def new_state(self, hole, seat):
    s = super().new_state(hole, seat)
    self.our_seat = int(seat)
    self.game, self.actions = RealGame(), ""
    self.strings: list[str] = []
    self.steps: list[dict] = []
    # Once lost (tree abandoned or a desync), the real string is not passed
    # on for the rest of the hand.
    self.lost = False
    self._probs = None
    self.model.new_hand()
    self.stats["hands"] += 1
    return s

  def policy(self, s, seat):
    probs, legal = super().policy(s, seat)
    self._probs = [float(p) for p in probs]
    return probs, legal

  # ── Crossing between the games ───────────────────────────────────────────

  def slumbot_atom(self, rec, before, tree, actor, rng) -> int:
    kind = _KIND.get(rec["kind"], native.SB_BET)
    if tree is None:
      return FOLD if kind == native.SB_FOLD else CALL
    history, _ = _view(tree, self.our_seat)
    st = self.model._step(tuple(history))
    totals, raise_ok = self.model._totals(st, actor)
    bet = int(rec["token"][1:]) if kind == native.SB_BET else 0
    atom, gap = slumbot_atom(kind, bet, totals, raise_ok, before, actor, self.model.scale)
    rec["translation"] = {"atom": int(atom), "rule": "nearest", "gap": gap}
    if kind == native.SB_BET:
      self.stats["bet_gap"].append(gap / float(st.bs.committed.sum()))
    # The board is filled in when the hand ends: the tree's board is only set
    # at our decisions.
    self.steps.append({"actor": int(actor), "history": history, "street": int(tree.street),
                       "committed": np.asarray(tree.committed, np.float64).tolist(),
                       "solve": None, "atom": int(atom)})
    return atom

  def our_incr(self, game, seat, atom, tree):
    if tree is None:
      return real_incr(game, seat, atom, self.bins)
    history, board = _view(tree, seat)
    st = self.model._step(tuple(history))
    a = st.alias[atom]
    totals, _ = self.model._totals(st, seat)
    tok, _ = our_token(a, float(totals[a]), float(np.max(st.bs.committed_street)), game, seat,
                       self.model.scale)
    self.steps.append({"actor": int(seat), "history": history, "street": int(tree.street),
                       "board": board,
                       "committed": np.asarray(tree.committed, np.float64).tolist(),
                       "solve": len(self.resolver.solves) - 1, "atom": int(atom),
                       "probs": self._probs})
    return tok, atom

  def observe_token(self, token, tree, tree_lost) -> None:
    nxt = _advance(self.game, self.actions, token)
    if nxt is None:
      self.stats["desync"] += 1
      self._lose()
      return
    self.game, self.actions = nxt
    if tree_lost:
      self._lose()
    if self.lost:
      return
    self.strings.append(self.actions)
    history, _ = _view(tree, self.our_seat)
    self.model.set_real(history, self.strings)

  def observe_response(self, r) -> None:
    if r.get("action", "") != self.actions and len(self.stats["string_mismatch"]) < 20:
      self.stats["string_mismatch"].append((r.get("action", ""), self.actions))
    if "winnings" in r:
      self._finish(r)

  def _lose(self) -> None:
    if not self.lost:
      self.lost = True
      self.stats["lost_hands"] += 1
      self.model.new_hand()

  def _finish(self, r: dict) -> None:
    # Fills in the boards of the steps and, with Slumbot's cards revealed, the
    # model's marginal at its hand for each of its steps.
    board = [card_index(c) for c in r.get("board", [])]
    bot = r.get("bot_hole_cards")
    h = None
    if bot and not self.lost:
      h = int(HAND_INDEX[card_index(bot[0]), card_index(bot[1])])
    for st in self.steps:
      st.setdefault("board", board[: _VISIBLE[st["street"]]])
      if st["actor"] == self.our_seat:
        continue
      st["probs"] = None
      if h is not None:
        p = self.model.probs([st["history"]], [], st["board"], self.our_seat)[0, h]
        st["probs"] = [float(x) for x in p]
        self.stats["slumbot_steps"] += 1
        self.stats["zero_prob"] += int(p[st["atom"]] <= 0)

  def hand_info(self) -> dict:
    info = super().hand_info()
    info["resolving"].update(steps=self.steps, real_action=self.actions, lost=self.lost)
    return info

  def summary(self) -> str:
    """One line of the agent's, the model's and the rollouts' health counters."""
    s, m, ro = self.stats, self.model.stats, self.rollouts.stats
    gap = np.asarray(s["bet_gap"]) if s["bet_gap"] else np.zeros(1)
    return (f"{s['hands']} hands, {s['lost_hands']} lost, {s['desync']} desyncs, "
            f"{len(s['string_mismatch'])} string mismatches {s['string_mismatch'][:3]}; "
            f"Slumbot steps {s['slumbot_steps']}, zero-probability {s['zero_prob']}; "
            f"bets exact {np.mean(gap == 0):.0%} (max gap {gap.max():.2f} pot); "
            f"model refusals {len(m['failures'])}, desyncs {m['desync']}, real string lost "
            f"{m['real_lost']}; rollouts {ro['refusals']} refusals, {ro['desync']} desyncs")
