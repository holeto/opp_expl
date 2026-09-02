"""Card-free betting rules for heads-up no-limit hold'em.

This module is the single source of truth for *how betting works* — legality,
sizing, when a round closes, when the hand ends. None of it reads hole cards or
the board, which is exactly why it can be shared between two very different
consumers:

  * ``HunlHoldem`` (``hunl_holdem.py``) wraps it with cards, history buffers and
    normalised rewards, and steps it one sampled hand at a time.
  * The public-tree builder in ``oppex.cfr`` enumerates it directly. Public-state
    CFR walks a tree of *publicly observable* actions, and preflop the public
    state **is** the betting state — there is nothing else to know. Because bet
    sizes here are pot-relative (a function of public state alone), every child
    is well-defined without reference to anyone's holding.

Keeping one implementation matters more than it looks: a divergence between the
env's rules and the solver's rules does not crash, it silently produces a wrong
equilibrium.

``BettingState`` is card-free and history-free, which gives the tree builder a
free duplicate-action detector: two sibling edges that land on the same
``BettingState`` *are* the same action. See ``drop_duplicate_allin``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── Atom layout: point actions first, then K bet atoms ───────────────────────
FOLD = 0
CALL = 1
ALL_IN = 2
NUM_POINT_ATOMS = 3


# ── Config & state ───────────────────────────────────────────────────────────


class BettingRules(NamedTuple):
  """Static game geometry. ``bet_bins`` may be empty (K = 0, push-fold)."""

  starting_stack: float
  small_blind: float
  big_blind: float
  bet_bins: jax.Array  # (K,) float32 — pot fractions, K may be 0


class BettingState(NamedTuple):
  """Everything about a hand that is public knowledge. No cards, no history."""

  committed: jax.Array        # (2,) float32 — total chips in from each player
  committed_street: jax.Array # (2,) float32 — chips in on the current street
  street: jax.Array           # ()  int32   — 0 preflop, 1 flop, 2 turn, 3 river
  cur_player: jax.Array       # ()  int32
  acted: jax.Array            # ()  int32   — voluntary actions this street
  last_raise_size: jax.Array  # ()  float32 — min-raise base
  done: jax.Array             # ()  bool


class BettingStep(NamedTuple):
  """Outcome of one atom. The flags are what the tree builder classifies on."""

  state: BettingState
  added: jax.Array      # () float32 — chips this atom actually put in
  is_fold: jax.Array    # () bool
  round_over: jax.Array # () bool — betting round closed
  advance: jax.Array    # () bool — round closed and the hand continues (street++)
  showdown: jax.Array   # () bool
  done: jax.Array       # () bool — is_fold | showdown


def make_rules(
  starting_stack: float,
  small_blind: float,
  big_blind: float,
  max_bet_fraction: float,
  num_bet_bins: int,
) -> BettingRules:
  """Build rules, tolerating ``num_bet_bins == 0`` (the push-fold abstraction)."""
  if num_bet_bins == 0:
    bins = jnp.zeros((0,), dtype=jnp.float32)
  else:
    bins = jnp.linspace(
      max_bet_fraction / num_bet_bins, max_bet_fraction, num_bet_bins, dtype=jnp.float32
    )
  return BettingRules(
    starting_stack=float(starting_stack),
    small_blind=float(small_blind),
    big_blind=float(big_blind),
    bet_bins=bins,
  )


def num_atoms(rules: BettingRules) -> int:
  return NUM_POINT_ATOMS + rules.bet_bins.shape[0]


def initial_state(rules: BettingRules) -> BettingState:
  """Blinds posted, SB (player 0, the button) to act first pre-flop."""
  blinds = jnp.array([rules.small_blind, rules.big_blind], dtype=jnp.float32)
  return BettingState(
    committed=blinds,
    committed_street=blinds,
    street=jnp.int32(0),
    cur_player=jnp.int32(0),
    acted=jnp.int32(0),
    last_raise_size=jnp.float32(rules.big_blind),  # min raise base = one BB
    done=jnp.bool_(False),
  )


# ── Derived quantities ───────────────────────────────────────────────────────


def stacks(rules: BettingRules, s: BettingState) -> jax.Array:
  return rules.starting_stack - s.committed


def call_amount(rules: BettingRules, s: BettingState) -> jax.Array:
  """Chips the current player must add to match the street's high bet."""
  return jnp.max(s.committed_street) - s.committed_street[s.cur_player]


def bet_fraction(rules: BettingRules, atom: jax.Array) -> jax.Array:
  """Pot fraction for a bet atom; 0.0 for point actions (masked out downstream).

  The ``K == 0`` branch is a *static* Python test, not a ``jnp.where``: gathering
  from a shape-``(0,)`` array raises at trace time, and JAX's out-of-bounds index
  clamping does not rescue a zero-sized operand. ``bet_bins.shape[0]`` is known
  at trace time even under ``jit``, so this branch is free.
  """
  if rules.bet_bins.shape[0] == 0:
    return jnp.float32(0.0)
  return rules.bet_bins[jnp.maximum(atom - NUM_POINT_ATOMS, 0)]


def chips_added(rules: BettingRules, s: BettingState, atom: jax.Array) -> jax.Array:
  """Chips the current player puts in for ``atom``, clipped to what is legal.

  Pot-relative raise: call first, then add ``fraction`` of the pot as it stands
  *after* that call, so ``fraction == 1`` is a pot-sized raise. Measuring against
  the pot rather than the call keeps the bins distinct on an unraised street,
  where ``call_amt == 0`` would collapse every bet atom onto the minimum raise.
  """
  cp = s.cur_player
  stack_cp = rules.starting_stack - s.committed[cp]
  call_amt = call_amount(rules, s)
  raw_bet = call_amt + bet_fraction(rules, atom) * (s.committed.sum() + call_amt)

  # Smallest legal raise: match the call, then add at least one min-raise unit,
  # capped by the stack (so it collapses to an all-in when the stack is short).
  min_raise_add = jnp.minimum(call_amt + s.last_raise_size, stack_cp)
  bet_add = jnp.clip(jnp.round(raw_bet), min_raise_add, stack_cp)

  return jnp.where(
    atom == FOLD, 0.0,
    jnp.where(atom == CALL, jnp.minimum(call_amt, stack_cp),
    jnp.where(atom == ALL_IN, stack_cp, bet_add)))


# ── Legality ─────────────────────────────────────────────────────────────────


def legal_atoms(
  rules: BettingRules, s: BettingState, *, drop_duplicate_allin: bool = False
) -> jax.Array:
  """Mask over atoms for the player to act: ``[fold?, call, all-in?, bet…?]``.

  - fold legal only when facing a bet (can't fold a free check).
  - call always legal (a check when nothing is owed).
  - all-in legal whenever the player has chips.
  - a bet atom is legal only when a full minimum raise fits and the opponent
    still has chips to face it (no raising an already all-in opponent).

  ``drop_duplicate_allin`` additionally forbids ALL_IN when calling would already
  commit the whole stack (``stack <= call``). In that situation CALL and ALL_IN
  produce bit-identical successors and payoffs, so they are the same action wearing
  two labels. Left in, regret mass splits across the two identical columns and the
  shove frequency you read off a chart is ``σ[CALL] + σ[ALL_IN]`` — half the true
  value, with nothing to indicate anything went wrong. Defaults to ``False`` so the
  env's action space is unchanged; the CFR tree builder passes ``True``.
  """
  cp = s.cur_player
  stack_cp = rules.starting_stack - s.committed[cp]
  opp_stack = rules.starting_stack - s.committed[1 - cp]
  call_amt = call_amount(rules, s)

  can_fold = call_amt > 0.0
  can_all_in = stack_cp > 0.0
  if drop_duplicate_allin:
    can_all_in = can_all_in & (stack_cp > call_amt)
  can_raise = (stack_cp >= call_amt + s.last_raise_size) & (opp_stack > 0.0)

  bet_slots = jnp.full(rules.bet_bins.shape[0], can_raise)
  return jnp.concatenate([jnp.array([can_fold, True, can_all_in]), bet_slots])


# ── Transition ───────────────────────────────────────────────────────────────


def step(rules: BettingRules, s: BettingState, atom: jax.Array) -> BettingStep:
  """Apply one atom by the current player and advance the betting state."""
  cp = s.cur_player
  opp = 1 - cp
  added = chips_added(rules, s, atom)
  is_fold = atom == FOLD
  old_max = jnp.max(s.committed_street)

  new_committed = s.committed.at[cp].add(added)
  new_committed_street = s.committed_street.at[cp].add(added)
  new_all_in = (rules.starting_stack - new_committed) <= 0.0

  # A raise/bet pushes the street bet above the previous high → opponent owes.
  raised = new_committed_street[cp] > old_max
  raise_increment = new_committed_street[cp] - old_max
  new_last_raise = jnp.where(
    raised, jnp.maximum(raise_increment, rules.big_blind), s.last_raise_size
  )

  acted_new = s.acted + jnp.where(is_fold, 0, 1)
  matched = new_committed_street[0] == new_committed_street[1]
  either_all_in = new_all_in[0] | new_all_in[1]
  # Betting closes when both have matched (and each has acted), or a player is
  # all-in with nothing left to contest — including a call all-in for less.
  called_all_in_short = (~raised) & (~is_fold) & new_all_in[cp] & (~matched)
  round_over = (
    is_fold
    | ((matched | (either_all_in & ~raised)) & (acted_new >= 2))
    | called_all_in_short
  )

  # Hand ends on a fold, at showdown after the river, or once all-in is settled.
  showdown = round_over & (~is_fold) & (either_all_in | (s.street == 3))
  done = is_fold | showdown

  # Advance to the next street when the round closes without ending the hand.
  advance = round_over & (~done)
  next_committed_street = jnp.where(
    advance, jnp.zeros(2, jnp.float32), new_committed_street
  )
  # Post-flop the big blind (player 1) acts first; within a street the turn passes.
  next_player = jnp.where(advance, jnp.int32(1), jnp.where(round_over, cp, opp))

  return BettingStep(
    state=BettingState(
      committed=new_committed,
      committed_street=next_committed_street,
      street=s.street + jnp.where(advance, 1, 0),
      cur_player=next_player,
      acted=jnp.where(advance, jnp.int32(0), acted_new),
      last_raise_size=jnp.where(advance, jnp.float32(rules.big_blind), new_last_raise),
      done=done,
    ),
    added=added,
    is_fold=is_fold,
    round_over=round_over,
    advance=advance,
    showdown=showdown,
    done=done,
  )


# ── Terminal payoffs (chips, not normalised) ─────────────────────────────────
# NOTE: both formulas assume no side pot — a pot split as `committed[winner_side]`
# is only chip-conserving while `committed[0] == committed[1]`. That always holds
# at a showdown with equal starting stacks (a short all-in call closes the round
# via `called_all_in_short` without reaching a contested showdown), but the
# assumption becomes wrong the moment unequal stacks are introduced.


def fold_payoffs(committed: jax.Array, folder: jax.Array) -> jax.Array:
  """(2,) chips. The folder forfeits; the opponent wins the folder's contribution."""
  return jnp.where(
    (1 - folder) == 0,
    jnp.stack([committed[1], -committed[1]]),
    jnp.stack([-committed[0], committed[0]]),
  )


def showdown_payoffs(committed: jax.Array, sign: jax.Array) -> jax.Array:
  """(2,) chips, where ``sign`` is +1 if player 0 wins, -1 if player 1, 0 on a tie.

  The caller supplies ``sign``: the env compares ``hand_rank7`` on a concrete
  board, while CFR supplies the sign of an expectation over all runouts.
  """
  return jnp.where(
    sign > 0, jnp.stack([committed[1], -committed[1]]),
    jnp.where(sign < 0, jnp.stack([-committed[0], committed[0]]),
    jnp.zeros(2, jnp.float32)))
