"""Local best response (Lisý & Bowling 2017) against Slumbot and against a chump.

LBR plays the hand itself. At each of our decisions it takes the opponent's
range ``π`` (Bayes on their actions so far, from the opponent model) and plays
the action of highest expected chips under a one-step heuristic, Figure 1 of
the paper. Relative to folding,

  U(call) = wp · pot − (1 − wp) · asked,
  U(a)    = fp · pot + (1 − fp) · (wp' · (pot + a) − (1 − wp') · (asked + a)),

with ``pot`` the chips in the pot, ``asked`` the chips to call, ``a`` the bet's
raise beyond the call, ``fp`` the opponent's fold probability after the bet,
and ``wp``, ``wp'`` our checkdown win probability against ``π`` and against
``π`` without the folds, ties counting half. The heuristic: after our action
the opponent folds or calls, and the hand is checked down. We play the argmax
if its ``U`` is positive and fold otherwise; facing no bet we check instead,
which has the same value (the env has no fold there). On the first
``call_streets`` streets we check or call: the paper's "Rounds" column 3-4 is
``call_streets=2``.

**Bets** are env atoms, so the crossing into the real game is the re-solver's:
``fc`` has none, ``fcpa`` the pot bin and all-in, ``bins`` every bin and
all-in. A pot bet raises by the pot after calling (``betting.py``). A bin
clipped up to the minimum raise or onto the all-in is the paper's "min-bet or
all in is evaluated instead", and a struck duplicate is evaluated once. The
chips in ``U`` are the real game's.

**Against Slumbot** (``SlumbotLBRAgent``) the model is ``SlumbotModel``: the
exact range over the translation type ``u`` and the real strings, as in the
re-solver, so LBR and the re-solver differ only in the decision rule and the
bets. An atom is sent as ``our_token``'s chips, the env street total × 50. That
is the real pot bet when the env and real pots agree; ``pot_mismatch`` counts
the decisions where they do not.

**Against always-call** (``LBRAgent`` with ``AlwaysCallModel`` and
``AlwaysCallOpponent``) the LBR code is checked without Slumbot. The paper's
Table 2 gives 34.0 ± 0.5 BB/h for fcpa on all streets, 49.0 ± 0.4 for fcpa on
streets 3-4, and 0 for fc.

**Against the ACPC chumps** (``RealLBR`` with ``ChumpOpponent``, from
``scripts/policy_gradient/lbr_chumps.py``) LBR plays on the real game alone:
the chumps raise to any chip total, which the env tree cannot follow. The
decision rule and the bet sizing are the code above.
"""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from ...cfr.cards import HAND_INDEX, N_HANDS
from ...cfr.subgame import expand_node, replay_betting
from ...envs.betting import ALL_IN, CALL, FOLD, NUM_POINT_ATOMS
from ..resolving import _view, compatible, showdown_sign
from .agent import SlumbotResolvingAgent
from .api import BIG_BLIND, STACK_SIZE
from .model import our_token
from .play import RealGame, TreeAgent, atom_name, card_index, real_incr, tokenize
from .random_opponent import RandomOpponent
from .rollouts import _advance

BET_SETS = ("fc", "fcpa", "bins")


# ── The decision rule ────────────────────────────────────────────────────────


def lbr_values(pi, sign, pot: float, asked: float, bets: dict) -> tuple[dict, dict]:
  """``U`` of check/call and of each bet, relative to folding (Figure 1).

  Args:
    pi: ``(H,)`` the opponent's range, any scale.
    sign: ``(H,)`` our expected showdown sign against each hand.
    pot, asked: chips in the pot, and chips we must add to call.
    bets: ``{atom: (a, fold)}``: the bet's raise beyond the call in chips, and
      the opponent's ``(H,)`` fold probability after it.

  Returns ``(U, info)``, both keyed by atom, ``CALL`` first and then ``bets``
  in order. ``info[atom] = (fp, wp)``: the fold probability, and the win
  probability against the hands that do not fold.
  """
  pi = np.asarray(pi, np.float64) / np.sum(pi)
  win = (1.0 + np.asarray(sign, np.float64)) / 2.0
  wp = float(pi @ win)
  values = {CALL: wp * pot - (1.0 - wp) * asked}
  info = {CALL: (0.0, wp)}
  for atom, (a, fold) in bets.items():
    fp = float(pi @ fold)
    rest = pi * (1.0 - fold)
    wp2 = float(rest @ win / rest.sum()) if rest.sum() > 0 else 0.0
    values[atom] = fp * pot + (1.0 - fp) * (wp2 * (pot + a) - (1.0 - wp2) * (asked + a))
    info[atom] = (fp, wp2)
  return values, info


def lbr_choice(values: dict, asked: float) -> int:
  """The argmax of ``values`` if positive, else fold (check when nothing is asked).

  Ties go to the first atom in ``values``, so to check/call before any bet.
  """
  best = max(values, key=values.get)
  if values[best] > 0:
    return best
  return FOLD if asked > 0 else CALL


# ── LBR's policy ─────────────────────────────────────────────────────────────


class _LBR:
  # LBR's policy, mixed into an agent with model, ev, env, bins and legal.
  # Subclasses give the real chips (_real_chips).

  def _init_lbr(self, bets: str) -> None:
    if bets not in BET_SETS:
      raise ValueError(f"unknown bet set {bets!r}, expected one of {BET_SETS}")
    bins = [float(b) for b in self.bins]
    if bets == "fc":
      atoms = []
    elif bets == "fcpa":
      if 1.0 not in bins:
        raise ValueError(f"fcpa needs a pot-sized bin; the bins are {bins}")
      atoms = [NUM_POINT_ATOMS + bins.index(1.0), ALL_IN]
    else:
      atoms = [NUM_POINT_ATOMS + i for i in range(len(bins))] + [ALL_IN]
    # Smallest bets first, so that ties go to the smaller bet.
    self.lbr_bets, self.lbr_atoms = bets, atoms
    self._lbr_scale = STACK_SIZE / float(self.env.starting_stack)
    self.lbr_stats = {"decisions": 0, "pot_mismatch": 0, "no_range": 0,
                      "choices": Counter()}
    self._lbr_rec = None

  def _real_chips(self, s, seat, atoms) -> tuple[int, int, dict]:
    # Returns (our committed, their committed, {atom: our committed after
    # it}) in the real game's chips.
    raise NotImplementedError

  def lbr_policy(self, s, seat) -> tuple[np.ndarray, np.ndarray]:
    """One-hot env-atom probabilities of LBR's action, and the legal mask."""
    self.lbr_inputs = None
    legal = np.asarray(self.legal(s, seat))
    probs = np.zeros(len(legal))
    history, board = _view(s, seat)
    cards = [int(c) for c in np.asarray(s.hole_cards)[seat]]
    rules = self.env.rules

    # The candidate bets: the tree's survivors of our atoms, each once.
    _, alias, _, _ = expand_node(rules, replay_betting(rules, history)[-1][0])
    cands = list(dict.fromkeys(alias[a] for a in self.lbr_atoms if alias[a] >= ALL_IN))

    r, _ = self.model.belief(rules, history, cards, board, seat)
    if r.sum() <= 0:
      # The model cannot explain the history; check/call, as the re-solver's
      # fallbacks do.
      self.lbr_stats["no_range"] += 1
      probs[CALL] = 1.0
      self._lbr_rec = {"choice": "call", "no_range": True}
      return probs, legal

    # The opponent's fold probabilities right after each candidate bet.
    fold = (self.model.probs([tuple(history) + (a,) for a in cands], cards, board, seat)
            [:, :, FOLD] if cands else np.zeros((0, N_HANDS)))
    c_us, c_op, after = self._real_chips(s, seat, cands)
    # A bet the crossing turned into a call is no bet.
    bets = {a: (after[a] - c_op, fold[k]) for k, a in enumerate(cands) if after[a] > c_op}
    pot, asked = c_us + c_op, c_op - c_us
    sign = showdown_sign(cards, board, self.ev)
    values, info = lbr_values(r, sign, pot, asked, bets)
    atom = lbr_choice(values, asked)
    if not legal[atom]:
      raise AssertionError(f"LBR chose the illegal atom {atom} after {history}")
    # The inputs of the decision, for validation (validate_lbr.py); not logged.
    self.lbr_inputs = {"range": r, "sign": sign, "bets": bets, "pot": pot, "asked": asked,
                       "values": values, "atom": atom}

    tree_pot = float(np.asarray(s.committed).sum()) * self._lbr_scale
    self.lbr_stats["decisions"] += 1
    self.lbr_stats["pot_mismatch"] += int(abs(tree_pot - pot) > 1e-6)
    self.lbr_stats["choices"][atom_name(atom, self.bins)] += 1
    self._lbr_rec = {
      "choice": atom_name(atom, self.bins), "pot": pot, "asked": asked, "tree_pot": tree_pot,
      "U": {atom_name(a, self.bins): round(v, 3) for a, v in values.items()},
      "fp_wp": {atom_name(a, self.bins): [round(x, 5) for x in v] for a, v in info.items()},
    }
    probs[atom] = 1.0
    return probs, legal

  def lbr_summary(self) -> str:
    st = self.lbr_stats
    choices = ", ".join(f"{k} {v}" for k, v in st["choices"].most_common())
    return (f"LBR {self.lbr_bets}, check/call on the first {self.call_streets} streets: "
            f"{st['decisions']} decisions ({choices}); env pot != real pot at "
            f"{st['pot_mismatch']}; no range {st['no_range']}")


def _committed_after(g: RealGame, seat: int, tok: str) -> int:
  # Our committed chips after the token: a bet raises our street total to N.
  if tok[0] == "b":
    return g.committed[seat] + int(tok[1:]) - g.street_committed[seat]
  return g.committed[1 - seat]


# ── Against Slumbot ──────────────────────────────────────────────────────────


class SlumbotLBRAgent(_LBR, SlumbotResolvingAgent):
  """LBR in ``play.play_hand`` against Slumbot (slumbot.com or ``LocalSlumbot``).

  The re-solving agent with its decision replaced: the model, the crossing
  between the games, the real strings and the per-step records are its own.
  The ``Resolver`` it builds is never called.
  """

  def __init__(self, run_dir, checkpoint, which, ev, bets: str = "fcpa",
               call_streets: int = 0, seed: int = 0) -> None:
    super().__init__(run_dir, checkpoint, which, ev, call_streets=call_streets, seed=seed)
    # The re-solver hands ev only to its Resolver; LBR's showdown signs need it.
    self.ev = ev
    self.lbr_inputs = None
    self._init_lbr(bets)

  def new_state(self, hole, seat):
    # (step index, range, sign) of this hand's LBR decisions, scored at
    # Slumbot's hand when it is revealed (_finish).
    self._calib = []
    return super().new_state(hole, seat)

  def policy(self, s, seat):
    if int(s.street) < self.call_streets:
      return super().policy(s, seat)
    self._called = False
    if self.lost:
      # Without the real string the chips are not known; check/call.
      probs, legal = np.zeros(self.env.num_actions), np.asarray(self.legal(s, seat))
      probs[CALL] = 1.0
      self._lbr_rec = {"choice": "call", "lost": True}
      self.lbr_inputs = None
    else:
      probs, legal = self.lbr_policy(s, seat)
    self._probs = [float(p) for p in probs]
    return probs, legal

  def our_incr(self, game, seat, atom, tree):
    out = super().our_incr(game, seat, atom, tree)
    if tree is not None:
      # The step just recorded has no solve; it carries LBR's values instead.
      self.steps[-1]["solve"] = None
      if not self._called:
        self.steps[-1]["lbr"] = self._lbr_rec
        if self.lbr_inputs is not None:
          self._calib.append((len(self.steps) - 1, self.lbr_inputs["range"],
                              self.lbr_inputs["sign"]))
    return out

  def _finish(self, r: dict) -> None:
    # With Slumbot's cards revealed, each LBR decision's range is scored at
    # its hand, for the calibration tests of validate_lbr.py --logs. If the
    # range is right, log π(h*) − Σ π log π and win(h*) − wp have mean 0.
    super()._finish(r)
    bot = r.get("bot_hole_cards")
    if not bot or self.lost:
      return
    h = int(HAND_INDEX[card_index(bot[0]), card_index(bot[1])])
    for i, rng, sign in self._calib:
      pi = rng / rng.sum()
      nz = pi > 0
      win = (1.0 + sign) / 2.0
      self.steps[i]["lbr"]["calib"] = {
        "pi_h": float(pi[h]), "neg_entropy": float(pi[nz] @ np.log(pi[nz])),
        "win_h": float(win[h]), "wp": float(pi @ win),
        "var_win": float(pi @ win ** 2 - (pi @ win) ** 2),
      }

  def _real_chips(self, s, seat, atoms):
    # The atoms' tokens as our_incr sends them (our_token on the env totals).
    history, _ = _view(s, seat)
    st = self.model._step(tuple(history))
    totals, _ = self.model._totals(st, seat)
    street_max = float(np.max(st.bs.committed_street))
    g = self.game
    after = {}
    for a in atoms:
      tok, _ = our_token(a, float(totals[a]), street_max, g, seat, self.model.scale)
      after[a] = _committed_after(g, seat, tok)
    return g.committed[seat], g.committed[1 - seat], after

  def summary(self) -> str:
    return f"{super().summary()}\n{self.lbr_summary()}"


# ── Against always-call ──────────────────────────────────────────────────────


class AlwaysCallModel:
  """The always-call chump as an opponent model: every hand checks or calls."""

  def __init__(self, n_actions: int) -> None:
    self.n_actions = n_actions

  def probs(self, paths, our_cards, board, our_seat) -> np.ndarray:
    out = np.zeros((len(paths), N_HANDS, self.n_actions))
    out[..., CALL] = 1.0
    return out

  def belief(self, rules, history, our_cards, board, our_seat):
    return compatible(our_cards, board), []


class AlwaysCallOpponent(RandomOpponent):
  """The always-call chump behind Slumbot's client interface.

  It checks or calls every decision, whatever its cards. Deals and seats are
  ``RandomOpponent``'s, so ``duplicate`` works the same way.
  """

  def __init__(self, seed: int = 0, duplicate: bool = False) -> None:
    super().__init__(np.zeros(0), seed, duplicate)

  def _bot_moves(self):
    bot = 1 - self.client_seat
    while not self.game.done and self.game.cur == bot:
      self._apply("c" if self.game.call_amount(bot) > 0 else "k")
    return self._response()


class LBRAgent(_LBR, TreeAgent):
  """LBR in ``play.play_hand`` against an opponent that never bets.

  ``model`` must be exact on the env's atoms, e.g. ``AlwaysCallModel``; it
  may be set after construction, once ``env`` is loaded. Bets
  are sized on the real pot (``play.real_incr``). An opponent bet would go
  through ``TreeAgent``'s pseudo-harmonic translation, which the model knows
  nothing about, hence the restriction.
  """

  def __init__(self, run_dir, checkpoint, which, ev, model=None, bets: str = "fcpa",
               call_streets: int = 0) -> None:
    super().__init__(run_dir, checkpoint, which)
    self.ev, self.model, self.call_streets = ev, model, int(call_streets)
    self._init_lbr(bets)

  def new_state(self, hole, seat):
    self.game, self.actions, self.records = RealGame(), "", []
    return super().new_state(hole, seat)

  def observe_token(self, token, tree, tree_lost) -> None:
    nxt = _advance(self.game, self.actions, token)
    if nxt is None:
      raise AssertionError(f"illegal token {token!r} after {self.actions!r}")
    self.game, self.actions = nxt

  def policy(self, s, seat):
    if int(s.street) < self.call_streets:
      legal = np.asarray(self.legal(s, seat))
      probs = np.zeros(len(legal))
      probs[CALL] = 1.0
      return probs, legal
    probs, legal = self.lbr_policy(s, seat)
    self.records.append(self._lbr_rec)
    return probs, legal

  def _real_chips(self, s, seat, atoms):
    g = self.game
    after = {a: _committed_after(g, seat, real_incr(g, seat, a, self.bins)[0]) for a in atoms}
    return g.committed[seat], g.committed[1 - seat], after

  def hand_info(self) -> dict:
    return {"lbr": self.records}

  def summary(self) -> str:
    return self.lbr_summary()


# ── Against ACPC chumps, in the real game ────────────────────────────────────


# The chumps of the paper's Table 2 as (fold, call, raise) probabilities of
# the ACPC example player. ``uniform`` has none: every legal action is equally
# likely, each raise size counting as one action, so it raises almost always.
CHUMPS = {
  "always_call": (0.0, 1.0, 0.0),
  "call_raise": (0.0, 0.5, 0.5),
  "random": (1 / 3, 1 / 3, 1 / 3),
  "uniform": None,
}
# The chips between consecutive raise sizes of a chump.
RAISE_STEPS = {"chip": 1, "bb": BIG_BLIND}
# The smallest raise of a chump: the legal minimum raise, or one step.
RAISE_FLOORS = ("min", "step")
# The pot fractions of each bet set; ``bins`` are the checkpoint's.
BET_FRACTIONS = {"fc": (), "fcpa": (1.0,), "bins": (0.5, 1.0, 1.5, 2.0)}


def raise_sizes(g: RealGame, seat: int, step: int = 1, floor: str = "min") -> np.ndarray | None:
  """``seat``'s raise-to street totals a chump draws from, or None if it cannot raise.

  The raise beyond the call is a whole number of ``step`` chips, from the
  floor up to the all-in, the last step clipped onto the all-in. The ``min``
  floor is the ACPC and Slumbot minimum raise: the last increment, at least a
  big blind, and an all-in below it is still a raise. The ``step`` floor
  allows raises below the minimum. With ``step=1`` and ``floor="min"`` these
  are the ACPC example player's sizes.
  """
  call, stack = g.call_amount(seat), g.stack(seat)
  if stack <= call or g.stack(1 - seat) <= 0:
    return None
  if floor not in RAISE_FLOORS:
    raise ValueError(f"unknown raise floor {floor!r}, expected one of {RAISE_FLOORS}")
  room = stack - call
  lo = g.min_raise() if floor == "min" else step
  k_lo, k_hi = -(-lo // step), -(-room // step)
  # A floor above the stack leaves the all-in alone.
  ks = np.arange(k_lo, max(k_lo, k_hi) + 1)
  return max(g.street_committed) + np.minimum(ks * step, room)


class ChumpOpponent(RandomOpponent):
  """An ACPC example-player chump behind Slumbot's client interface.

  At each decision it draws an action type with probabilities ``probs`` =
  (fold, call, raise), renormalised over the valid types: folding is valid
  only facing a bet, raising only while both players have chips behind. A
  raise is to a uniformly random size of ``raise_sizes``. With ``probs=None``
  every legal action is equally likely: fold, call and each raise size. Its
  play ignores its cards, so its range is every hand compatible with the
  public cards and ours, whatever it did. Deals and seats are
  ``RandomOpponent``'s, so ``duplicate`` works the same way.
  """

  def __init__(self, probs, seed: int = 0, duplicate: bool = False, raise_step: int = 1,
               raise_floor: str = "min") -> None:
    super().__init__(np.zeros(0), seed, duplicate)
    self.probs = None if probs is None else np.asarray(probs, np.float64)
    self.raise_step, self.raise_floor = int(raise_step), raise_floor

  def raise_sizes(self, g: RealGame, seat: int) -> np.ndarray | None:
    return raise_sizes(g, seat, self.raise_step, self.raise_floor)

  def type_probs(self, g: RealGame, seat: int) -> np.ndarray:
    """(fold, call, raise) probabilities of ``seat``'s next action in ``g``."""
    sizes = self.raise_sizes(g, seat)
    n = 0 if sizes is None else len(sizes)
    valid = np.array([g.call_amount(seat) > 0, True, n > 0])
    weights = np.array([1.0, 1.0, n]) if self.probs is None else self.probs
    p = weights * valid
    return p / p.sum()

  def _bot_moves(self):
    bot = 1 - self.client_seat
    while not self.game.done and self.game.cur == bot:
      g = self.game
      kind = int(self.rng.choice(3, p=self.type_probs(g, bot)))
      if kind == 0:
        tok = "f"
      elif kind == 1:
        tok = "c" if g.call_amount(bot) > 0 else "k"
      else:
        sizes = self.raise_sizes(g, bot)
        # The index draws as ``integers(lo, hi + 1)`` did, so the default
        # chump replays the runs from before ``raise_sizes``.
        tok = f"b{int(sizes[self.rng.integers(0, len(sizes))])}"
      self._apply(tok)
    return self._response()


class RealLBR:
  """LBR deciding on the real game alone, against a ``ChumpOpponent``.

  The env tree cannot follow a chump's raises of any chip size: translated
  onto atoms it drifts from the real game, and ``play.play_hand`` then
  check/calls. Against a chump nothing needs the tree. The range is every
  compatible hand, and the fold probability after our bet is the chump's own
  rule on the real game after it, so the model is exact. Candidates and chips
  are ``_LBR``'s: pot fractions of the real pot by ``play.real_incr``, plus
  the all-in, each distinct total once, and the decision is ``lbr_values`` and
  ``lbr_choice``.
  """

  def __init__(self, ev, bets: str = "fcpa", call_streets: int = 0) -> None:
    if bets not in BET_FRACTIONS:
      raise ValueError(f"unknown bet set {bets!r}, expected one of {BET_SETS}")
    self.ev, self.bets, self.call_streets = ev, bets, int(call_streets)
    self.bins = np.asarray(BET_FRACTIONS[bets], np.float64)
    # Smallest bets first, so that ties go to the smaller bet.
    self.atoms = [NUM_POINT_ATOMS + i for i in range(len(self.bins))] + (
      [ALL_IN] if bets != "fc" else [])
    self.stats = {"decisions": 0, "choices": Counter()}

  def decide(self, g: RealGame, seat: int, cards, board, chump: ChumpOpponent) -> tuple[str, dict | None]:
    """Our token in ``g``, and the decision's record (None when only called)."""
    call = g.call_amount(seat)
    check_call = "c" if call > 0 else "k"
    if g.street < self.call_streets:
      return check_call, None

    # Each candidate bet's token, raise beyond the call, and fold probability.
    c_us, c_op = g.committed[seat], g.committed[1 - seat]
    bets, seen = {}, set()
    for atom in self.atoms:
      tok, _ = real_incr(g, seat, atom, self.bins)
      if tok[0] != "b" or tok in seen:
        continue
      seen.add(tok)
      after = copy.deepcopy(g)
      after.apply("b", int(tok[1:]))
      fold = chump.type_probs(after, 1 - seat)[0]
      bets[tok] = (_committed_after(g, seat, tok) - c_op, np.full(N_HANDS, fold))

    r = compatible(cards, board)
    sign = showdown_sign(cards, board, self.ev)
    values, info = lbr_values(r, sign, c_us + c_op, c_op - c_us, bets)
    choice = lbr_choice(values, c_op - c_us)
    tok = {FOLD: "f", CALL: check_call}.get(choice, choice)
    name = {"f": "fold", "c": "call", "k": "call"}.get(tok, "allin" if
            _committed_after(g, seat, tok) == STACK_SIZE else "bet")
    self.stats["decisions"] += 1
    self.stats["choices"][name] += 1
    key = lambda a: "call" if a == CALL else a
    return tok, {"choice": name, "token": tok, "pot": c_us + c_op, "asked": c_op - c_us,
                 "U": {key(a): round(v, 3) for a, v in values.items()},
                 "fp_wp": {key(a): [round(x, 5) for x in v] for a, v in info.items()}}

  def summary(self) -> str:
    choices = ", ".join(f"{k} {v}" for k, v in self.stats["choices"].most_common())
    return (f"LBR {self.bets} on the real game, check/call on the first "
            f"{self.call_streets} streets: {self.stats['decisions']} decisions ({choices})")


def play_chump_hand(client: ChumpOpponent, lbr: RealLBR) -> dict:
  """One hand of ``lbr`` against the chump, through the client interface."""
  r = client.new_hand()
  seat = 1 - r["client_pos"]
  hole = [card_index(c) for c in r["hole_cards"]]
  g, history, records = RealGame(), "", []
  while True:
    action = r.get("action", "")
    assert action.startswith(history), (history, action)
    for kind, to in tokenize(action[len(history):]):
      if not g.done:
        g.apply(kind, to)
    history = action
    if "winnings" in r:
      return {"client_pos": r["client_pos"], "seat": seat, "hole_cards": r["hole_cards"],
              "board": r.get("board", []), "bot_hole_cards": r.get("bot_hole_cards"),
              "action": action, "winnings": r["winnings"], "lbr": records}
    assert not g.done and g.cur == seat, (action, g.__dict__)
    board = [card_index(c) for c in r.get("board", [])]
    tok, rec = lbr.decide(g, seat, hole, board, client)
    if rec is not None:
      records.append({"street": g.street, **rec})
    g.apply(tok[0], int(tok[1:]) if tok[0] == "b" else None)
    history += tok
    r = client.act(tok)
