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


def sample_atom(probs: jax.Array, u: jax.Array) -> jax.Array:
  """Inverse-CDF draw from unnormalised ``(..., A)`` weights with uniforms ``(...)``.

  Picks the first atom whose cumulative weight *exceeds* ``u · total``, so an
  atom of weight 0 is never drawn: ``cdf < u · total`` would pick atom 0 when
  ``u`` is exactly 0, which a float32 uniform is with probability 2^-23. If
  rounding puts ``u · total`` at the total, the draw falls back to the last
  atom of positive weight rather than the last atom.
  """
  cdf = jnp.cumsum(probs, -1)
  atom = (cdf <= u[..., None] * cdf[..., -1:]).sum(-1)
  n = probs.shape[-1]
  last = n - 1 - jnp.argmax(jnp.flip(probs, -1) > 0, -1)
  return jnp.minimum(atom, last)
