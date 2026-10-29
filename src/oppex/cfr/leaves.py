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
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Leaves(NamedTuple):
  """Leaf payoffs and the continuation strategies played at them."""

  values: jax.Array   # (n_leaves, T, T, n_hands, n_hands), chips to player 0, mask-folded
  tau: jax.Array      # (2, n_leaves, n_hands, T), per player, hand and continuation


def leaf_action_cfv(values, r_opp, tau_opp, player):
  """``(n_leaves, n_hands, T)`` counterfactual value of each own continuation.

  ``r_opp`` is the opponent's ``(n_leaves, n_hands)`` reach on arriving at each
  leaf, and ``tau_opp`` their ``(n_leaves, n_hands, T)`` continuation strategy.
  Their product is the opponent's reach *after* choosing, which is what the
  payoff is contracted against.
  """
  w = r_opp[..., None] * tau_opp
  if player == 0:
    return jnp.einsum("labhk,lkb->lha", values, w)
  return -jnp.einsum("labhk,lha->lkb", values, w)


def uniform_tau(n_leaves: int, n_hands: int, n_transforms: int, dtype=jnp.float32):
  return jnp.full((2, n_leaves, n_hands, n_transforms), 1.0 / n_transforms, dtype)
