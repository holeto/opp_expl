"""Heads-Up No-Limit Texas Hold'em for 2 sequential-move players.

The action space is **fully discrete**: a player's action each step is a single
categorical ``atom`` — an index into ``[FOLD, CALL, ALL_IN, BET_0 … BET_{K-1}]``.
The leading three are point actions; the **last K atoms are bet atoms**, one per
entry of ``bet_bins``.

Bet atoms differ from one another only in size, and sizes are **pot-relative**.
Bet atom ``i`` carries the fraction ``bet_bins[i]`` from
``linspace(max_bet_fraction / K, max_bet_fraction, K)``, and the chips the player
adds are

    round(call_amount + bet_bins[i] * (pot + call_amount))

i.e. call first, then raise by that fraction of the pot as it stands after the
call — so ``bet_bins[i] == 1`` is a pot-sized raise. The result is clipped to the
legal raise range ``[call + min_raise, stack]``, which is what makes a bet atom
collapse to an all-in when the stack is short. Sizing against the pot rather
than the call keeps the bins distinct on an unraised street, where a
call-relative fraction would collapse every bet atom onto the minimum raise.
Clipping and rounding live entirely in the env; ``bet_history`` records the
chips actually applied.

Action array contract for ``apply_action``:
  * ``actions`` of shape ``(num_players,)`` — one atom index per player. Only
    the entry for ``current_player`` is read; the rest are ignored.

Heads-up conventions:
  * Player 0 is the button / small blind: acts first pre-flop, last post-flop.
  * Player 1 is the big blind: acts last pre-flop, first post-flop.
  * Blinds, big blind, starting stack, max bet fraction and bet granularity are constructor arguments (chips).

Cards: card index ``c`` in ``0..51`` decodes to ``rank = c // 4`` (0=2 … 12=A)
and ``suit = c % 4``. Showdown uses a full 7-card best-five evaluation.

Max trajectory length is *derived from the starting stack* — see ``max_length``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import betting
from .base import Env, Info, PRNGKey
from .betting import ALL_IN, CALL, FOLD, NUM_POINT_ATOMS as _NUM_POINT_ATOMS

__all__ = ["HunlHoldem", "HunlState", "hand_rank7", "FOLD", "CALL", "ALL_IN"]

# Hand-category indices for the 7-card evaluator (higher = stronger).
_HIGH_CARD, _PAIR, _TWO_PAIR, _TRIPS, _STRAIGHT = 0, 1, 2, 3, 4
_FLUSH, _FULL_HOUSE, _QUADS, _STRAIGHT_FLUSH = 5, 6, 7, 8
_KICKER_BASE = 13 ** 5  # tiebreak slots: 5 ranks in base 13


# ── 7-card hand evaluation ───────────────────────────────────────────────────


def _straight_high(present: jax.Array) -> jax.Array:
  """Highest straight in a 13-bool rank mask (idx 0=2 … 12=A).

  Returns the straight's high-card rank index (A-high=12 … 6-high=4, wheel
  A-2-3-4-5 = 3) or -1 if no straight. Pure / shape-static.
  """
  windows = jnp.stack(
    [present[h - 4 : h + 1].all() for h in range(4, 13)]  # 6-high … A-high
  )
  highs = jnp.where(windows, jnp.arange(4, 13), -1)
  reg_high = jnp.max(highs)
  wheel = present[12] & present[0] & present[1] & present[2] & present[3]
  return jnp.where(reg_high >= 0, reg_high, jnp.where(wheel, 3, -1))


def _enc5(ranks5: jax.Array) -> jax.Array:
  """Encode 5 rank indices (most significant first) as a base-13 integer."""
  out = jnp.int32(0)
  for i in range(5):
    out = out * 13 + ranks5[i]
  return out


def hand_rank7(cards: jax.Array) -> jax.Array:
  """Strength score of the best 5-card hand out of 7 cards (higher = better).

  ``cards`` is an int array of 7 card indices in ``0..51``. The returned int32
  score orders any two 7-card hands exactly by poker hand ranking.
  """
  ranks = cards // 4
  suits = cards % 4
  rank_counts = jnp.bincount(ranks, length=13)
  suit_counts = jnp.bincount(suits, length=4)
  rank_present = rank_counts > 0

  # Flush: ranks present within the most common suit.
  flush_suit = jnp.argmax(suit_counts)
  is_flush = jnp.max(suit_counts) >= 5
  in_flush = (suits == flush_suit)[:, None] & jax.nn.one_hot(ranks, 13, dtype=bool)
  flush_present = in_flush.any(axis=0)

  straight_high = _straight_high(rank_present)
  sf_high = _straight_high(flush_present)
  is_sf = is_flush & (sf_high >= 0)

  # Multiplicity pattern (counts sorted high→low) gives the category.
  counts_sorted = jnp.sort(rank_counts)[::-1]
  c0, c1 = counts_sorted[0], counts_sorted[1]

  # Best-5 multiset for the "counting" categories: order the 7 cards by
  # (count of their rank desc, rank desc) and take the top 5 ranks *with
  # repetition* (e.g. a pair yields [A,A,K,J,9]). Using cards rather than
  # distinct ranks avoids encoding spurious 6th/7th ranks that would break ties.
  card_key = rank_counts[ranks] * 13 + ranks
  best5 = ranks[jnp.argsort(card_key)[::-1]][:5]
  kick = _enc5(best5)

  # Top 5 ranks within the flush suit (for a plain flush).
  forder = jnp.argsort(jnp.where(flush_present, jnp.arange(13), -1))[::-1]
  flush_kick = _enc5(forder[:5])

  quad = c0 == 4
  full_house = (c0 == 3) & (c1 >= 2)
  trips = (c0 == 3) & (c1 == 1)
  two_pair = (c0 == 2) & (c1 == 2)
  pair = (c0 == 2) & (c1 == 1)

  category = jnp.where(
    is_sf, _STRAIGHT_FLUSH,
    jnp.where(quad, _QUADS,
    jnp.where(full_house, _FULL_HOUSE,
    jnp.where(is_flush, _FLUSH,
    jnp.where(straight_high >= 0, _STRAIGHT,
    jnp.where(trips, _TRIPS,
    jnp.where(two_pair, _TWO_PAIR,
    jnp.where(pair, _PAIR, _HIGH_CARD))))))))

  tiebreak = jnp.where(
    category == _STRAIGHT_FLUSH, sf_high,
    jnp.where(category == _FLUSH, flush_kick,
    jnp.where(category == _STRAIGHT, straight_high, kick)))

  return (category * _KICKER_BASE + tiebreak).astype(jnp.int32)


# ── State ────────────────────────────────────────────────────────────────────


class HunlState(NamedTuple):
  hole_cards: jax.Array       # (2, 2) int32  — 2 private cards per player
  board: jax.Array            # (5,)  int32   — full community cards (always dealt)
  committed: jax.Array        # (2,)  float32 — total chips each player put in this hand
  committed_street: jax.Array # (2,)  float32 — chips each put in on the current street
  street: jax.Array           # ()    int32   — 0 preflop, 1 flop, 2 turn, 3 river
  cur_player: jax.Array       # ()    int32   — 0 or 1
  acted: jax.Array            # ()    int32   — voluntary actions taken this street
  last_raise_size: jax.Array  # ()    float32 — size of the last raise (min-raise base)
  action_history: jax.Array   # (L,)  int32   — atom played per step (-1 = none yet)
  bet_history: jax.Array      # (L,)  float32 — chips added per step
  step: jax.Array             # ()    int32
  done: jax.Array             # ()    bool


class HunlHoldem(Env):
  """Heads-Up No-Limit Texas Hold'em.

  Args:
    starting_stack: chips each player starts with.
    small_blind / big_blind: blind sizes in chips.
    max_bet_fraction: the maximal considered bet/raise fraction of the pot, not
        including all in.
    num_bet_bins: K bins spanning the betting range, as
        ``linspace(max_bet_fraction / K, max_bet_fraction, K)``. For example
        ``max_bet_fraction=2.0`` with ``num_bet_bins=4`` gives fractions
        ``[0.5, 1.0, 1.5, 2.0]`` — half-pot, pot, 1.5x pot, 2x pot.
    reward_type: 'difference' — net chip profit scaled to [-1, 1] by the stack
        (default); 'binary' — ±1 win/loss, 0 tie.
    max_length: optional override of the derived episode-length bound (see the
        ``max_length`` property). Set it from a measured length distribution of
        the policies actually being trained.
  """

  def __init__(
    self,
    starting_stack: int = 400,
    small_blind: int = 1,
    big_blind: int = 2,
    max_bet_fraction: float = 2.0,
    num_bet_bins: int = 4,
    reward_type: str = "difference",
    max_length: int | None = None,
    drop_duplicate_allin: bool = False,
  ) -> None:
    if reward_type not in ("difference", "binary"):
      raise ValueError(f"reward_type must be 'difference' or 'binary', got '{reward_type}'")
    if big_blind > starting_stack:
      raise ValueError("starting_stack must be at least one big blind")
    self.starting_stack = float(starting_stack)
    self.small_blind = float(small_blind)
    self.big_blind = float(big_blind)
    self.num_bet_bins = int(num_bet_bins)
    self.max_bet_fraction = max_bet_fraction
    self.reward_type = reward_type
    self._max_length = int(max_length) if max_length is not None else None
    self._drop_duplicate_allin = bool(drop_duplicate_allin)
    self.rules = betting.make_rules(
      starting_stack=self.starting_stack,
      small_blind=self.small_blind,
      big_blind=self.big_blind,
      max_bet_fraction=self.max_bet_fraction,
      num_bet_bins=self.num_bet_bins,
    )
    self.bet_bins = self.rules.bet_bins

  # ── Static properties ────────────────────────────────────────────────────

  @property
  def num_players(self) -> int:
    return 2

  @property
  def num_actions(self) -> int:
    return _NUM_POINT_ATOMS + self.num_bet_bins  # fold, call, all-in, + K bet atoms

  @property
  def max_length(self) -> int:
    """Practical (not worst-case) episode-length bound; the rare tail is truncated.

    The theoretical maximum is a minimum-raise war committing both stacks one
    big blind at a time — roughly ``stack // big_blind`` steps (~200 for a
    400-chip / 2-BB game), which only a deliberately adversarial all-minimum
    policy ever approaches. Empirically, realistic and even raise-heavy play
    finishes far sooner (99.9th percentile ≈ 80 steps at 200 BB deep), so we
    size the rollout to comfortably cover that and accept that the ~0.1% of
    pathological hands exceeding it are truncated (no terminal reward). The
    per-street action budget grows only logarithmically with stack depth, since
    deeper stacks permit a few more re-raises before someone is all-in.
    """
    if self._max_length is not None:
      return self._max_length
    stack_bb = self.starting_stack / self.big_blind
    per_street = 6 + 3 * math.ceil(math.log2(stack_bb + 1))
    return 4 * per_street + 4

  @property
  def max_reward(self) -> float:
    return 1.0  # rewards are already scaled to [-1, 1] (see _terminal_rewards)

  # ── Bet utils ──────────────────────────────────────────────────────

  # ── State lifecycle ──────────────────────────────────────────────────────

  def init_state(self, key: PRNGKey) -> HunlState:
    deck = jax.random.permutation(key, 52)
    hole_cards = deck[:4].reshape(2, 2).astype(jnp.int32)  # p0: 0,1  p1: 2,3
    board = deck[4:9].astype(jnp.int32)
    L = self.max_length
    bs = betting.initial_state(self.rules)
    return self._from_betting(
      bs,
      hole_cards=hole_cards,
      board=board,
      action_history=jnp.full(L, -1, dtype=jnp.int32),
      bet_history=jnp.zeros(L, dtype=jnp.float32),
      step=jnp.int32(0),
    )

  # ── Internal helpers ──────────────────────────────────────────────────────
  # HunlState stays flat and carries the betting fields inline; these adapters
  # pack/unpack the card-free view that `betting` operates on. Nesting a
  # BettingState inside HunlState would be tidier but would rewrite the field
  # layout that `_betting_features` and all six observation methods depend on,
  # for no functional gain — the rules live in one place either way. Pure pytree
  # plumbing, free under jit.

  def _to_betting(self, state: HunlState) -> betting.BettingState:
    return betting.BettingState(
      committed=state.committed,
      committed_street=state.committed_street,
      street=state.street,
      cur_player=state.cur_player,
      acted=state.acted,
      last_raise_size=state.last_raise_size,
      done=state.done,
    )

  def _from_betting(self, bs: betting.BettingState, **carried) -> HunlState:
    return HunlState(
      committed=bs.committed,
      committed_street=bs.committed_street,
      street=bs.street,
      cur_player=bs.cur_player,
      acted=bs.acted,
      last_raise_size=bs.last_raise_size,
      done=bs.done,
      **carried,
    )

  def _stacks(self, state: HunlState) -> jax.Array:
    return betting.stacks(self.rules, self._to_betting(state))

  def _call_amount(self, state: HunlState) -> jax.Array:
    """Chips the current player must add to match the street's high bet."""
    return betting.call_amount(self.rules, self._to_betting(state))

  # ── Step ──────────────────────────────────────────────────────────────────

  def apply_action(
    self,
    state: HunlState,
    actions: jax.Array,  # (P,) atom index per player; only current_player's is read
  ) -> tuple[HunlState, jax.Array, jax.Array, Info]:
    cp = state.cur_player
    atom = actions[cp].astype(jnp.int32)
    out = betting.step(self.rules, self._to_betting(state), atom)

    new_state = self._from_betting(
      out.state,
      hole_cards=state.hole_cards,
      board=state.board,
      action_history=state.action_history.at[state.step].set(atom),
      bet_history=state.bet_history.at[state.step].set(out.added),
      step=state.step + 1,
    )

    rewards = jnp.where(
      out.done, self._terminal_rewards(state, out.state.committed, cp, out.is_fold),
      jnp.zeros(2, jnp.float32),
    )
    return new_state, rewards, out.done, {}

  def _terminal_rewards(
    self, state: HunlState, committed: jax.Array, folder: jax.Array, is_fold: jax.Array
  ) -> jax.Array:
    """Per-player payoff (zero-sum). Showdown uses the full board.

    For ``reward_type='difference'`` the net chip profit is scaled by the
    starting stack into [-1, 1]: the most a player can win is the opponent's
    whole stack, so ``profit / starting_stack`` is bounded by ±1.
    """
    fold_r = betting.fold_payoffs(committed, folder)

    r0 = hand_rank7(jnp.concatenate([state.hole_cards[0], state.board]))
    r1 = hand_rank7(jnp.concatenate([state.hole_cards[1], state.board]))
    show_r = betting.showdown_payoffs(committed, jnp.sign(r0 - r1))

    raw = jnp.where(is_fold, fold_r, show_r)
    if self.reward_type == "binary":
      sign = jnp.sign(raw[0])
      return jnp.stack([sign, -sign])
    return raw / self.starting_stack  # scale net chips to [-1, 1]


  # ── Action legality & turn order ──────────────────────────────────────────

  def legal_actions(self, state: HunlState, player_id: jax.Array | int) -> jax.Array:
    """Active player: [fold?, call, all-in?, bet…?]. Inactive: only FOLD slot.

    - fold legal only when facing a bet (can't fold a free check).
    - call always legal (a check when nothing is owed).
    - all-in legal whenever the player has chips.
    - a bet atom is legal only when a full minimum raise fits and the opponent
      still has chips to face it (no raising an already all-in opponent).
    """
    is_active = jnp.int32(player_id) == state.cur_player
    active_mask = betting.legal_atoms(
      self.rules,
      self._to_betting(state),
      drop_duplicate_allin=self._drop_duplicate_allin,
    )
    inactive_mask = jnp.zeros(self.num_actions, dtype=bool).at[0].set(True)
    return jnp.where(is_active, active_mask, inactive_mask)

  def current_player(self, state: HunlState) -> jax.Array:
    return state.cur_player

  # ── Observation helpers ────────────────────────────────────────────────────

  _STREET_BOARD = (0, 3, 4, 5)  # cards revealed per street

  def _cards_one_hot(self, cards: jax.Array) -> jax.Array:
    """Flattened 52-dim one-hot per card; cards < 0 (hidden) become all-zero."""
    oh = jax.nn.one_hot(jnp.clip(cards, 0, 51), 52, dtype=jnp.float32)
    return (oh * (cards >= 0)[:, None]).reshape(-1)

  def _visible_board(self, state: HunlState) -> jax.Array:
    n = jnp.asarray(self._STREET_BOARD)[state.street]
    return jnp.where(jnp.arange(5) < n, state.board, -1)

  def _betting_features(self, state: HunlState) -> jax.Array:
    S = self.starting_stack
    return jnp.concatenate([
      state.committed / S,                                  # 2
      self._stacks(state) / S,                              # 2
      jnp.array([state.committed.sum() / (2.0 * S)]),       # 1 pot fraction
      jax.nn.one_hot(state.street, 4, dtype=jnp.float32),   # 4
      jnp.array([self._call_amount(state) / S]),            # 1
      jax.nn.one_hot(state.cur_player, 2, dtype=jnp.float32),  # 2
      ((self._stacks(state)) <= 0.0).astype(jnp.float32),   # 2 all-in flags
      jnp.array([state.last_raise_size / S]),               # 1
    ])

  # ── Observations ────────────────────────────────────────────────────────────

  def player_observation(
    self, state: HunlState, player_id: jax.Array, key: PRNGKey
  ) -> jax.Array:
    """Own hole cards (104) + visible board (260) + betting features (15) + seat (2).

    The seat one-hot encodes *whose perspective* this observation is — without
    it, two players observing the same state produce inputs that differ only in
    hole cards, so a value net cannot tell which side of the pot it is valuing
    (button vs. big blind have very different values at the same node).
    """
    own = self._cards_one_hot(state.hole_cards[player_id])
    seat = jax.nn.one_hot(player_id, 2, dtype=jnp.float32)
    return jnp.concatenate(
      [own, self._cards_one_hot(self._visible_board(state)),
       self._betting_features(state), seat]
    )

  def public_observation(self, state: HunlState, key: PRNGKey) -> jax.Array:
    """Visible board (260) + betting features (15)."""
    return jnp.concatenate(
      [self._cards_one_hot(self._visible_board(state)), self._betting_features(state)]
    )

  def state_observation(self, state: HunlState, key: PRNGKey) -> jax.Array:
    """Both players' hole cards (208) + full board (260) + betting features (15)."""
    return jnp.concatenate([
      self._cards_one_hot(state.hole_cards[0]),
      self._cards_one_hot(state.hole_cards[1]),
      self._cards_one_hot(state.board),
      self._betting_features(state),
    ])

  # ── Perfect-recall representations ──────────────────────────────────────────
  # Append the full betting history (atoms + sizes) to the snapshot so the
  # representation uniquely encodes the player's information set.

  def _history_features(self, state: HunlState) -> jax.Array:
    return jnp.concatenate([
      state.action_history.astype(jnp.float32) / float(self.num_actions),
      state.bet_history / self.starting_stack,
    ])

  def information_set(
    self, state: HunlState, player_id: jax.Array | int, key: PRNGKey
  ) -> jax.Array:
    return jnp.concatenate(
      [self.player_observation(state, jnp.int32(player_id), key), self._history_features(state)]
    )

  def public_state(self, state: HunlState, key: PRNGKey) -> jax.Array:
    return jnp.concatenate(
      [self.public_observation(state, key), self._history_features(state)]
    )

  def state_representation(self, state: HunlState, key: PRNGKey) -> jax.Array:
    return jnp.concatenate(
      [self.state_observation(state, key), self._history_features(state)]
    )
