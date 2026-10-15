"""Imperfect-information Goofspiel for 2 simultaneous-move players.

Rules:
  - Both players hold cards 1..n_cards.
  - Each turn a prize card is revealed; both players secretly choose a hand card.
  - Higher card wins the prize; ties discard it.
  - Only the turn result (win / loss / draw) is revealed — not what was played.
  - Reward is given only at the end of the game.

Chance: with ``prize_order='random'`` (the stochastic variant) the entire prize
order is drawn as one permutation in ``init_state`` and revealed one card per
turn by the observation methods — the same "resolve all chance up front"
convention as ``HunlHoldem``, so ``apply_action`` is deterministic and takes no
key. Sampling the order turn by turn instead would give the same distribution
over trajectories (each turn's prize is uniform over the remaining cards
either way); only the interface differs.

Observation sizes (all binary float32 vectors):
  player_observation  : 5 * n_cards  (hand | prize_shown | p0_won | p1_won | draw)
  public_observation  : 4 * n_cards  (prize_shown | p0_won | p1_won | draw)
  state_observation   : n_cards**2 + 5 * n_cards
                        (prize_deck_onehot | hand0 | hand1 | p0_won | p1_won | draw)

Perfect-recall sizes (ordered per-turn history, all binary float32):
  information_set    : 2*n_cards**2 + 3*n_cards + 2
                       (prize_per_turn | own_card_per_turn | results | player_onehot)
  public_state       : 2*n_cards**2 + 3*n_cards
                       (prize_per_turn | draw_card_per_turn | results)
  state_representation: 3*n_cards**2 + 3*n_cards
                       (prize_per_turn | p0_card_per_turn | p1_card_per_turn | results)

  On draw turns card0 == card1, so public_state uses a single shared card one-hot.
  information_set omits the opponent card entirely — on draws it equals own_card
  (derivable from own_card + draw result bit), on wins/losses it is unknown.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Env, Info, PRNGKey


class GoofspielState(NamedTuple):
  prize_deck: jax.Array  # (n_cards,)   int32   — full prize order, all turns
  turn: jax.Array  # ()           int32   — current step, 0-indexed
  hands: jax.Array  # (2, n_cards) bool    — remaining cards per player
  scores: jax.Array  # (2,)         float32 — accumulated prize values
  turn_results: jax.Array  # (n_cards,)   int32   — -1 not played | 0 p0 | 1 p1 | 2 draw
  action_history: (
    jax.Array
  )  # (2, n_cards) int32   — 0-indexed card played at each turn, -1 if not yet


class Goofspiel(Env):
  """Imperfect-information Goofspiel for 2 simultaneous-move players.

  Args:
    n_cards:      cards per player / number of turns (default: 13)
    prize_order:  'random' | 'ascending' | 'descending' (default: 'random')
    reward_type:  'difference' | 'binary' (default: 'difference')
                  'difference' — reward = signed score gap  (e.g. +4 / -4)
                  'binary'     — reward = +1 winner, 0 draw, -1 loser
  """

  def __init__(
    self,
    n_cards: int = 13,
    prize_order: str = "random",
    reward_type: str = "difference",
  ) -> None:
    if prize_order not in ("random", "ascending", "descending"):
      raise ValueError(
        f"prize_order must be 'random', 'ascending', or 'descending', got '{prize_order}'"
      )
    if reward_type not in ("difference", "binary"):
      raise ValueError(
        f"reward_type must be 'difference' or 'binary', got '{reward_type}'"
      )
    self.n_cards = n_cards
    self.prize_order = prize_order
    self.reward_type = reward_type

  # ── Static properties ────────────────────────────────────────────────────

  @property
  def num_players(self) -> int:
    return 2

  @property
  def max_length(self) -> int:
    return self.n_cards

  @property
  def num_actions(self) -> int:
    return self.n_cards

  @property
  def max_reward(self) -> float:
    if self.reward_type == "binary":
      return 1.0
    return float(self.n_cards * (self.n_cards + 1) // 2)

  # ── State lifecycle ──────────────────────────────────────────────────────

  def init_state(self, key: PRNGKey) -> GoofspielState:
    if self.prize_order == "random":
      prize_deck = jax.random.permutation(key, self.n_cards) + 1
    elif self.prize_order == "ascending":
      prize_deck = jnp.arange(1, self.n_cards + 1)
    else:  # descending
      prize_deck = jnp.arange(self.n_cards, 0, -1)

    return GoofspielState(
      prize_deck=prize_deck.astype(jnp.int32),
      turn=jnp.zeros((), dtype=jnp.int32),
      hands=jnp.ones((2, self.n_cards), dtype=jnp.bool_),
      scores=jnp.zeros(2, dtype=jnp.float32),
      turn_results=jnp.full(self.n_cards, -1, dtype=jnp.int32),
      action_history=jnp.full((2, self.n_cards), -1, dtype=jnp.int32),
    )

  def apply_action(
    self,
    state: GoofspielState,
    actions: jax.Array,  # (2,) int32 — 0-indexed card chosen by each player
  ) -> tuple[GoofspielState, jax.Array, jax.Array, Info]:
    prize = state.prize_deck[state.turn]

    # Card values are 1-indexed; actions are 0-indexed card positions
    card0 = actions[0] + 1
    card1 = actions[1] + 1

    p0_wins = card0 > card1
    p1_wins = card1 > card0

    new_scores = state.scores + jnp.array(
      [
        jnp.where(p0_wins, prize, 0),
        jnp.where(p1_wins, prize, 0),
      ],
      dtype=jnp.float32,
    )

    new_hands = jnp.stack(
      [
        state.hands[0].at[actions[0]].set(False),
        state.hands[1].at[actions[1]].set(False),
      ]
    )

    result = jnp.where(p0_wins, 0, jnp.where(p1_wins, 1, 2))
    new_turn_results = state.turn_results.at[state.turn].set(result)

    new_turn = state.turn + 1
    done = (new_turn >= self.n_cards).astype(jnp.bool_)

    if self.reward_type == "difference":
      terminal_r = jnp.array(
        [
          new_scores[0] - new_scores[1],
          new_scores[1] - new_scores[0],
        ]
      )
    else:  # binary
      winner = jnp.sign(new_scores[0] - new_scores[1])
      terminal_r = jnp.array([winner, -winner])

    # Reward is zero every step except the terminal one
    rewards = jnp.where(done, terminal_r, jnp.zeros(2, dtype=jnp.float32))

    new_action_history = state.action_history.at[:, state.turn].set(actions)

    new_state = GoofspielState(
      prize_deck=state.prize_deck,
      turn=new_turn,
      hands=new_hands,
      scores=new_scores,
      turn_results=new_turn_results,
      action_history=new_action_history,
    )
    return new_state, rewards, done, {}

  # ── Observation helpers ──────────────────────────────────────────────────

  def _prize_shown_bits(self, state: GoofspielState) -> jax.Array:
    """Binary vector of length n_cards: bit v=1 if prize card (v+1) has been revealed.

    Includes the current turn's prize (players see it before bidding),
    so turn t shows t+1 cards total.
    """
    turn_idx = jnp.arange(self.n_cards, dtype=jnp.int32)
    revealed = (turn_idx <= state.turn).astype(jnp.float32)
    # Scatter: position prize_deck[t]-1 gets 1.0 if turn t is revealed.
    # prize_deck is a permutation so no duplicate indices.
    return (
      jnp.zeros(self.n_cards, dtype=jnp.float32).at[state.prize_deck - 1].set(revealed)
    )

  def _turn_result_bits(self, state: GoofspielState) -> jax.Array:
    """Three binary vectors of length n_cards concatenated: [p0_won | p1_won | draw].

    Position t is 1 in the respective vector if that turn has been resolved with
    that outcome. Unplayed turns are 0 in all three.
    """
    p0_won = (state.turn_results == 0).astype(jnp.float32)
    p1_won = (state.turn_results == 1).astype(jnp.float32)
    draw = (state.turn_results == 2).astype(jnp.float32)
    return jnp.concatenate([p0_won, p1_won, draw])

  # ── Observations ─────────────────────────────────────────────────────────

  def player_observation(
    self, state: GoofspielState, player_id: jax.Array, key: PRNGKey
  ) -> jax.Array:
    """Binary vector of length 5*n_cards.

    Layout: own_hand (n_cards) | prize_shown (n_cards) | turn_results (3*n_cards)

    player_id is a JAX scalar — safe to vmap:
      jax.vmap(lambda p: env.player_observation(state, p))(jnp.arange(2))
    """
    hand = state.hands[player_id].astype(jnp.float32)
    return jnp.concatenate(
      [hand, self._prize_shown_bits(state), self._turn_result_bits(state)]
    )

  def public_observation(self, state: GoofspielState, key: PRNGKey) -> jax.Array:
    """Binary vector of length 4*n_cards.

    Layout: prize_shown (n_cards) | turn_results (3*n_cards)
    """
    return jnp.concatenate(
      [self._prize_shown_bits(state), self._turn_result_bits(state)]
    )

  def state_observation(self, state: GoofspielState, key: PRNGKey) -> jax.Array:
    """Binary vector of length n_cards**2 + 5*n_cards that uniquely identifies the state.

    Layout: prize_deck_onehot (n_cards**2) | hand0 (n_cards) | hand1 (n_cards)
            | turn_results (3*n_cards)

    prize_deck is encoded as a flat permutation matrix (one-hot per position)
    so the full future prize order is visible — this is the privileged information
    unavailable to players in the random-order variant.
    """
    prize_onehot = jax.nn.one_hot(state.prize_deck - 1, self.n_cards).flatten()
    hand0 = state.hands[0].astype(jnp.float32)
    hand1 = state.hands[1].astype(jnp.float32)
    return jnp.concatenate([prize_onehot, hand0, hand1, self._turn_result_bits(state)])

  # ── Action legality ───────────────────────────────────────────────────────

  def legal_actions(
    self, state: GoofspielState, player_id: jax.Array | int
  ) -> jax.Array:
    """Boolean mask of length n_cards: True for cards still in the player's hand."""
    return state.hands[player_id]

  # ── Perfect-recall representations ───────────────────────────────────────

  def _prize_per_turn(self, state: GoofspielState) -> jax.Array:
    """One-hot prize encoding for each turn, shape (n_cards, n_cards).

    Row t is the one-hot of the prize revealed at turn t, zeroed for turns
    that have not yet been reached (prize is shown at the start of each turn,
    so turn t is visible once state.turn >= t).
    """
    turn_idx = jnp.arange(self.n_cards, dtype=jnp.int32)
    prize_shown = (turn_idx <= state.turn).astype(jnp.float32)  # (n_cards,)
    return jax.nn.one_hot(state.prize_deck - 1, self.n_cards) * prize_shown[:, None]

  def information_set(
    self, state: GoofspielState, player_id: jax.Array | int, key: PRNGKey
  ) -> jax.Array:
    """Ordered action history from player_id's perspective.

    Shape: (2*n_cards**2 + 3*n_cards + 2,).
    Layout: prize_per_turn (n_cards**2) | own_card_per_turn (n_cards**2)
            | results (3*n_cards) | player_onehot (2)

    The opponent card is excluded: on draw turns it equals own_card (derivable
    from own_card + draw result bit); on win/loss turns it is unknown.
    """
    played = (jnp.arange(self.n_cards) < state.turn).astype(jnp.float32)
    own_enc = (
      jax.nn.one_hot(state.action_history[player_id], self.n_cards) * played[:, None]
    )
    return jnp.concatenate(
      [
        jax.nn.one_hot(player_id, 2),
        self._prize_per_turn(state).flatten(),
        own_enc.flatten(),
        self._turn_result_bits(state),
      ]
    )

  def public_state(self, state: GoofspielState, key: PRNGKey) -> jax.Array:
    """Ordered history of all public events.

    Shape: (2*n_cards**2 + 3*n_cards,).
    Layout: prize_per_turn (n_cards**2) | draw_card_per_turn (n_cards**2)
            | results (3*n_cards)

    On draw turns card0 == card1, so a single shared card one-hot suffices.
    Win/loss turns contribute only to the result bits; the cards played remain hidden.
    """
    draw_mask = (state.turn_results == 2).astype(jnp.float32)
    draw_card_enc = (
      jax.nn.one_hot(state.action_history[0], self.n_cards) * draw_mask[:, None]
    )
    return jnp.concatenate(
      [
        self._prize_per_turn(state).flatten(),
        draw_card_enc.flatten(),
        self._turn_result_bits(state),
      ]
    )

  def state_representation(self, state: GoofspielState, key: PRNGKey) -> jax.Array:
    """Full ground-truth game trajectory (privileged — not available to agents).

    Shape: (3*n_cards**2 + 3*n_cards,).
    Layout: prize_per_turn (n_cards**2) | p0_card_per_turn (n_cards**2)
            | p1_card_per_turn (n_cards**2) | results (3*n_cards)
    """
    played = (jnp.arange(self.n_cards) < state.turn).astype(jnp.float32)
    p0_enc = jax.nn.one_hot(state.action_history[0], self.n_cards) * played[:, None]
    p1_enc = jax.nn.one_hot(state.action_history[1], self.n_cards) * played[:, None]
    return jnp.concatenate(
      [
        self._prize_per_turn(state).flatten(),
        p0_enc.flatten(),
        p1_enc.flatten(),
        self._turn_result_bits(state),
      ]
    )
