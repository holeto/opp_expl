"""Matrix-valued leaves: a simultaneous choice of continuation at a depth limit.

At a ``MATRIX`` terminal of the public tree, each player picks one of ``T``
continuation strategies, such as the blueprint and its fold/call/raise-biased
versions in ``oppex.policy_gradient.blueprint``. The hand then ends with
``values[l, t0, t1, a, b]`` chips to player 0, where ``a`` and ``b`` are the
two players' hands.

**Why one simultaneous node and not two sequential ones.** Public-state CFR keys
a node on public information and keeps one ``(n_hands,)`` vector per player.
If player 0 chose first at one node and player 1 at its children, the choice
``t0`` would be public: player 1 would get a separate infoset per ``t0`` and
best-respond to it, which is a different game. Both choices are private, so
each player's infoset at a leaf is ``(leaf, own hand)``, and neither sees the
other's ``t``. The leaf is terminal right after the choice, so the
counterfactual value of each own continuation is a single contraction with the
opponent's ``reach x tau``. ``leaf_action_cfv`` is that contraction.

**Card removal, as at every terminal, enters exactly once.** ``values`` is
stored already multiplied by ``MASK``, like ``EV``, so nothing here masks again.
Unlike ``EV`` it is *not* antisymmetric: player 1's payoff for ``(b, a)`` is
``-values[l, t0, t1, a, b]``, so player 1's contraction runs over the first
hand axis with an explicit sign flip.

**A continuation axis of size one** means that player has a single
continuation, the case of a fixed opponent who is assumed to keep playing the
blueprint. The tensor is then ``(n_leaves, T, 1, ...)`` or
``(n_leaves, 1, T, ...)``: the payoff does not depend on that player's ``t``,
so their ``tau`` drops out of the contraction, and their own per-continuation
values are all equal. Only the free player's column of the matrix is sampled.

**Bucketed values.** ``BucketedValues`` stores the same tensor as bucket-pair
means, ``V[l, t0, t1, a, b] = vb[l, t0, t1, B(a), B(b)] * MASK[a, b]``, which is
how ``oppex.policy_gradient.leaf_values`` estimates it anyway. The full
preflop tree of the blueprint has over a thousand leaves, and the dense form is
7 MB per leaf and continuation pair, so it does not fit. Here the mask is *not*
folded in. The contraction applies it once, by inclusion-exclusion over the
two own cards (the identity behind ``cards.masked_sum``), at bucket level.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cards import HAND_CARDS, INCIDENCE


class Leaves(NamedTuple):
  """Leaf payoffs and the continuation strategies played at them."""

  values: jax.Array   # (n_leaves, T, T, n_hands, n_hands), chips to player 0, mask-folded
  tau: jax.Array      # (2, n_leaves, n_hands, T), per player, hand and continuation


class BucketedValues(NamedTuple):
  """Leaf values as bucket-pair means; see the module docstring."""

  vb: jax.Array       # (n_leaves, T0, T1, K, K), chips to player 0, NOT masked
  bucket: jax.Array   # (n_hands,) int32, bucket of every hand

  @property
  def shape(self) -> tuple[int, ...]:
    # The dense shape it stands for, so shape checks treat both forms alike.
    n = self.bucket.shape[0]
    return tuple(self.vb.shape[:3]) + (n, n)


def leaf_action_cfv(values, r_opp, tau_opp, player):
  """``(n_leaves, n_hands, T)`` counterfactual value of each own continuation.

  ``r_opp`` is the opponent's ``(n_leaves, n_hands)`` reach on arriving at each
  leaf, and ``tau_opp`` their ``(n_leaves, n_hands, T)`` continuation strategy.
  Their product is the opponent's reach *after* choosing, which is what the
  payoff is contracted against. ``T`` is the solver's table width; a size-one
  own axis of ``values`` is broadcast to it.
  """
  w = r_opp[..., None] * tau_opp
  own_t, opp_t = (values.shape[1], values.shape[2])[:: 1 if player == 0 else -1]
  # One opponent continuation: their choice does not change the payoff.
  if opp_t == 1:
    w = w.sum(-1, keepdims=True)
  if isinstance(values, BucketedValues):
    q = _bucketed_cfv(values, w, player)
  elif player == 0:
    q = jnp.einsum("labhk,lkb->lha", values, w)
  else:
    q = -jnp.einsum("labhk,lha->lkb", values, w)
  return jnp.broadcast_to(q, q.shape[:2] + (tau_opp.shape[-1],)) if own_t == 1 else q


def _bucketed_cfv(values: BucketedValues, w, player):
  # The dense contraction for V = vb[B(a), B(b)] * MASK[a, b], never building
  # anything per hand pair. For own hand h with cards d1, d2 and bucket c,
  # card removal by inclusion-exclusion splits the opponent mass into
  #   all hands - hands holding d1 - hands holding d2 + h itself,
  # and each term only needs the opponent's reach summed per bucket (and per
  # card), so the contraction with vb happens at bucket level.
  # Returns (n_leaves, n_hands, own T).
  vb, bucket = values.vb, values.bucket
  if player == 1:
    # Player 1's payoff, axes reordered to (leaf, own t, opp t, own c, opp c).
    vb = -jnp.swapaxes(jnp.swapaxes(vb, 1, 2), 3, 4)
  n_hands, n_cards = INCIDENCE.shape
  n_b = vb.shape[-1]
  onehot = jax.nn.one_hot(bucket, n_b, dtype=w.dtype)               # (H, K)

  # Opponent reach per bucket, and per (card, bucket).
  total = jnp.einsum("hk,lhb->lbk", onehot, w)
  card_bucket = (INCIDENCE[:, :, None] * onehot[:, None, :]).reshape(n_hands, -1)
  per_card = jnp.einsum("hm,lhb->lbm", card_bucket, w)
  per_card = per_card.reshape(per_card.shape[:2] + (n_cards, n_b))

  # Each term contracted with vb for every own bucket c, then read at B(h).
  v_all = jnp.einsum("labck,lbk->lac", vb, total)[:, :, bucket]
  v_card = jnp.einsum("labck,lbdk->lacd", vb, per_card)
  v_card = (v_card[:, :, bucket, HAND_CARDS[:, 0]]
            + v_card[:, :, bucket, HAND_CARDS[:, 1]])
  diag = jnp.diagonal(vb, axis1=3, axis2=4)[..., bucket]            # (L, T, T', H)
  v_self = jnp.einsum("labh,lhb->lah", diag, w)
  return jnp.swapaxes(v_all - v_card + v_self, 1, 2)


def uniform_tau(n_leaves: int, n_hands: int, n_transforms: int, dtype=jnp.float32):
  return jnp.full((2, n_leaves, n_hands, n_transforms), 1.0 / n_transforms, dtype)
