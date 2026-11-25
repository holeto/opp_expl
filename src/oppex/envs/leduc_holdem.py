"""Leduc Hold'em for 2 sequential-move players.

Rules:
  - Deck: J, J, Q, Q, K, K (ranks 0=J, 1=Q, 2=K, two copies each).
  - Each player antes 1 chip and receives 1 private card.
  - Round 1: player 0 acts first; bet/raise size 2; at most 2 raises.
  - Round 2: a public card is revealed; player 0 acts first; bet/raise size 4;
    at most 2 raises.
  - Actions: 0=fold, 1=check/call, 2=bet/raise. Fold is illegal facing no bet.
  - Showdown: a pair (private rank == public rank) beats a high card; the higher
    rank breaks ties; equal ranks split.
  - Reward: net chips (``'difference'``) or ±1/0 (``'binary'``), at the end.

**Ranks, not cards.** The state holds ranks only. Suits never affect payoffs,
so this is OpenSpiel's 6-card Leduc with suit isomorphism: two card-level
infosets that differ only in suits are merged, which loses nothing (same
value, and every equilibrium of either game maps to one of the other).

Chance: the two private cards and the public card are dealt at once in
``init_state`` and the public card is revealed by the observations when round
1 closes, the convention of ``oppex.envs.base``. So ``apply_action`` is
deterministic, and an extracted tree has one chance root over the 24 rank
deals (``oppex.tree.extract``) instead of a chance node between the rounds.

Sequential convention: the inactive player's legal mask has only action 0, and
``apply_action`` reads only ``actions[cur_player]``.

Observation sizes (float32):
  player_observation : 39   own_card(3) + common(36)
  public_observation : 36   common(36)
  state_observation  : 45   card0(3) + card1(3) + pending_public(3) + common(36)
  information_set    : 41   player_onehot(2) + player_observation(39)

common(36) = public_card_onehot(4) + pot_norm(2) + round_onehot(2)
             + call_norm(1) + raises_onehot(3) + action_history(8×3=24)

The snapshot observations already hold the whole action history, so they have
perfect recall. ``information_set`` adds the seat, as in Goofspiel, so that
the two players' infoset keys never coincide.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Env, Info, PRNGKey

FOLD = 0
CALL = 1
RAISE = 2

_MAX_RAISES = 2
# 4 actions per round (check, bet, raise, call) × 2 rounds.
_MAX_STEPS = 8
# Most chips one player can put in: ante 1 + round 1 4 + round 2 8.
_MAX_POT = 13.0
# Largest amount to call: the round-2 bet size.
_MAX_CALL = 4.0


class LeducState(NamedTuple):
  private_cards: jax.Array  # (2,)          int32   — rank per player
  pending_public: jax.Array  # ()           int32   — public rank, hidden in round 1
  public_card: jax.Array  # ()              int32   — -1 in round 1, then the rank
  pot: jax.Array  # (2,)                    float32 — chips put in by each player
  round_idx: jax.Array  # ()                int32   — 0 or 1
  raises: jax.Array  # ()                   int32   — raises so far this round
  cur_player: jax.Array  # ()               int32   — 0 or 1
  call_amount: jax.Array  # ()              float32 — chips cur_player owes to call
  round_actions: jax.Array  # ()            int32   — actions so far this round
  action_history: jax.Array  # (_MAX_STEPS,) int32  — action per step, -1 if not taken
  step: jax.Array  # ()                     int32   — steps taken
  done: jax.Array  # ()                     bool


class LeducHoldem(Env):
  """Leduc Hold'em for 2 sequential-move players.

  Args:
    reward_type: 'difference' — net chips (default);
                 'binary'     — +1 winner, 0 draw, -1 loser.
  """

  def __init__(self, reward_type: str = "difference") -> None:
    if reward_type not in ("difference", "binary"):
      raise ValueError(
        f"reward_type must be 'difference' or 'binary', got '{reward_type}'"
      )
    self.reward_type = reward_type

  # ── Static properties ────────────────────────────────────────────────────

  @property
  def num_players(self) -> int:
    return 2

  @property
  def max_length(self) -> int:
    return _MAX_STEPS

  @property
  def num_actions(self) -> int:
    return 3

  @property
  def max_reward(self) -> float:
    if self.reward_type == "binary":
      return 1.0
    return float(_MAX_POT)

  # ── State lifecycle ──────────────────────────────────────────────────────

  def init_state(self, key: PRNGKey) -> LeducState:
    # Deal 3 of the 6 cards; card index // 2 is the rank.
    deck = jax.random.permutation(key, 6)
    return self.deal(deck[:2] // 2, deck[2] // 2)

  def deal(self, private_ranks, public_rank) -> LeducState:
    """The initial state for given ranks, for enumerating deals."""
    return LeducState(
      private_cards=jnp.asarray(private_ranks, dtype=jnp.int32),
      pending_public=jnp.asarray(public_rank, dtype=jnp.int32),
      public_card=jnp.int32(-1),
      pot=jnp.ones(2, dtype=jnp.float32),
      round_idx=jnp.int32(0),
      raises=jnp.int32(0),
      cur_player=jnp.int32(0),
      call_amount=jnp.float32(0.0),
      round_actions=jnp.int32(0),
      action_history=jnp.full(_MAX_STEPS, -1, dtype=jnp.int32),
      step=jnp.int32(0),
      done=jnp.bool_(False),
    )

  def apply_action(
    self,
    state: LeducState,
    actions: jax.Array,  # (2,) int32 — only actions[cur_player] is read
  ) -> tuple[LeducState, jax.Array, jax.Array, Info]:
    cp = state.cur_player
    action = actions[cp]
    bet_size = jnp.where(state.round_idx == 0, 2.0, 4.0)

    # A call matches the bet; a raise matches it and adds one bet size.
    added = jnp.where(
      action == CALL,
      state.call_amount,
      jnp.where(action == RAISE, state.call_amount + bet_size, 0.0),
    )
    new_pot = state.pot.at[cp].add(added)
    new_call = jnp.where(action == RAISE, bet_size, 0.0)
    new_raises = state.raises + jnp.where(action == RAISE, 1, 0)

    # A round closes on a fold, a call of a bet, or a second check.
    is_fold = action == FOLD
    is_call = (action == CALL) & (state.call_amount > 0.0)
    is_second_check = (
      (action == CALL) & (state.call_amount == 0.0) & (state.round_actions >= 1)
    )
    round_over = is_fold | is_call | is_second_check
    done = is_fold | (round_over & (state.round_idx == 1))

    # Closing round 1 without a fold reveals the public card.
    move_to_r2 = round_over & ~is_fold & (state.round_idx == 0)
    new_public_card = jnp.where(move_to_r2, state.pending_public, state.public_card)
    new_round_idx = state.round_idx + jnp.where(move_to_r2, jnp.int32(1), jnp.int32(0))

    # Player 0 opens every round; within a round the players alternate.
    next_player = jnp.where(round_over, jnp.int32(0), jnp.int32(1 - cp))
    next_call = jnp.where(round_over, jnp.float32(0.0), new_call)
    next_raises = jnp.where(round_over, jnp.int32(0), new_raises)
    next_round_actions = jnp.where(round_over, jnp.int32(0), state.round_actions + 1)
    new_history = state.action_history.at[state.step].set(action)

    # ── Terminal rewards ─────────────────────────────────────────────────
    # A folder loses what they put in.
    fold_winner = jnp.int32(1 - cp)
    fold_r = jnp.stack(
      [
        jnp.where(fold_winner == 0, new_pot[1], -new_pot[0]),
        jnp.where(fold_winner == 1, new_pot[0], -new_pot[1]),
      ]
    )

    # Showdown strength: a pair scores 3 + rank, a high card its rank.
    p0_pair = state.private_cards[0] == new_public_card
    p1_pair = state.private_cards[1] == new_public_card
    p0_str = jnp.where(p0_pair, 3 + state.private_cards[0], state.private_cards[0])
    p1_str = jnp.where(p1_pair, 3 + state.private_cards[1], state.private_cards[1])
    p0_wins = p0_str > p1_str
    p1_wins = p1_str > p0_str
    show_r = jnp.stack(
      [
        jnp.where(p0_wins, new_pot[1], jnp.where(p1_wins, -new_pot[0], 0.0)),
        jnp.where(p1_wins, new_pot[0], jnp.where(p0_wins, -new_pot[1], 0.0)),
      ]
    )
    raw_rewards = jnp.where(is_fold, fold_r, show_r)

    if self.reward_type == "binary":
      sign = jnp.sign(raw_rewards[0])
      terminal_rewards = jnp.stack([sign, -sign])
    else:
      terminal_rewards = raw_rewards
    rewards = jnp.where(done, terminal_rewards, jnp.zeros(2, dtype=jnp.float32))

    new_state = LeducState(
      private_cards=state.private_cards,
      pending_public=state.pending_public,
      public_card=new_public_card,
      pot=new_pot,
      round_idx=new_round_idx,
      raises=next_raises,
      cur_player=next_player,
      call_amount=next_call,
      round_actions=next_round_actions,
      action_history=new_history,
      step=state.step + 1,
      done=done,
    )
    return new_state, rewards, done, {}

  # ── Observation helpers ──────────────────────────────────────────────────

  def _public_card_bits(self, state: LeducState) -> jax.Array:
    """4-dim one-hot: index 0 = no card yet, 1/2/3 = J/Q/K."""
    idx = jnp.where(state.public_card >= 0, state.public_card + 1, jnp.int32(0))
    return jax.nn.one_hot(idx, 4, dtype=jnp.float32)

  def _common_bits(self, state: LeducState) -> jax.Array:
    """36-dim vector of public information."""
    return jnp.concatenate(
      [
        self._public_card_bits(state),  # 4
        state.pot / _MAX_POT,  # 2
        jax.nn.one_hot(state.round_idx, 2, dtype=jnp.float32),  # 2
        jnp.array([state.call_amount / _MAX_CALL]),  # 1
        jax.nn.one_hot(state.raises, _MAX_RAISES + 1, dtype=jnp.float32),  # 3
        jax.nn.one_hot(state.action_history, 3, dtype=jnp.float32).flatten(),  # 24
      ]
    )

  # ── Observations ─────────────────────────────────────────────────────────

  def player_observation(
    self, state: LeducState, player_id: jax.Array, key: PRNGKey
  ) -> jax.Array:
    """39-dim: own_card_onehot(3) + common(36)."""
    own_card = jax.nn.one_hot(state.private_cards[player_id], 3, dtype=jnp.float32)
    return jnp.concatenate([own_card, self._common_bits(state)])

  def public_observation(self, state: LeducState, key: PRNGKey) -> jax.Array:
    """36-dim: common public information only."""
    return self._common_bits(state)

  def state_observation(self, state: LeducState, key: PRNGKey) -> jax.Array:
    """45-dim: both private cards + pending public card + common(36)."""
    card0 = jax.nn.one_hot(state.private_cards[0], 3, dtype=jnp.float32)
    card1 = jax.nn.one_hot(state.private_cards[1], 3, dtype=jnp.float32)
    pending = jax.nn.one_hot(state.pending_public, 3, dtype=jnp.float32)
    return jnp.concatenate([card0, card1, pending, self._common_bits(state)])

  # ── Action legality & turn order ─────────────────────────────────────────

  def legal_actions(self, state: LeducState, player_id: jax.Array | int) -> jax.Array:
    """[fold, call, raise] for the active player; only action 0 for the other."""
    is_active = jnp.int32(player_id) == state.cur_player
    can_raise = state.raises < _MAX_RAISES
    can_fold = state.call_amount > 0.0
    active_mask = jnp.array([can_fold, True, can_raise])
    inactive_mask = jnp.array([True, False, False])
    return jnp.where(is_active, active_mask, inactive_mask)

  def current_player(self, state: LeducState) -> jax.Array:
    return state.cur_player

  # ── Perfect-recall representations ───────────────────────────────────────

  def information_set(
    self, state: LeducState, player_id: jax.Array | int, key: PRNGKey
  ) -> jax.Array:
    """41-dim: player_onehot(2) + player_observation(39)."""
    player_id = jnp.int32(player_id)
    return jnp.concatenate(
      [jax.nn.one_hot(player_id, 2), self.player_observation(state, player_id, key)]
    )

  def public_state(self, state: LeducState, key: PRNGKey) -> jax.Array:
    return self.public_observation(state, key)

  def state_representation(self, state: LeducState, key: PRNGKey) -> jax.Array:
    return self.state_observation(state, key)
