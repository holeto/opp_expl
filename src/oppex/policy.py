"""Masked softmax policy transforms.

Networks emit *raw* logits over the full atom set; the distribution is formed at
the point of use, with illegal actions masked out. Ported from RLG's
``policy.py`` / ``utils.py`` with the α-entmax path removed — every policy here
is a plain softmax, so there is no ``alpha`` argument anywhere downstream.

Convention: illegal actions get probability 0 and log-probability **0.0** (not
``-inf``), so that ``Σ p·log p`` and KL sums stay finite without extra masking.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def policy_probs(logits: jax.Array, legal_actions: jax.Array) -> jax.Array:
  """π = softmax(logits) over legal actions; illegal actions get 0."""
  return jax.nn.softmax(logits, where=legal_actions)


def policy_log_probs(logits: jax.Array, legal_actions: jax.Array) -> jax.Array:
  """log π over legal actions; illegal actions get 0.0 (see module docstring)."""
  log_probs = jax.nn.log_softmax(logits, where=legal_actions)
  return jnp.where(legal_actions, log_probs, 0.0)


def policy_entropy_loss(probs: jax.Array, log_probs: jax.Array) -> jax.Array:
  """Negative Shannon entropy for a single sample. Minimising it maximises H."""
  return (probs * log_probs).sum(-1)


def policy_kl(log_p: jax.Array, log_q: jax.Array) -> jax.Array:
  """KL(p‖q) from log-probabilities (masked entries are 0 in both → contribute 0)."""
  return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)
