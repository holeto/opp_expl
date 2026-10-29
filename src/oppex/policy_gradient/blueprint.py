"""A trained checkpoint as a playing policy, and its biased continuations.

``Blueprint`` loads a run directory (``config.yaml`` + ``step_*.pkl``) and
exposes the policy on raw ``HunlState``s. Both the Slumbot player and the
leaf-value sampler go through it, so they read the same params the same way.

**Continuation strategies (Pluribus).** At a depth-limit leaf of search, each
player picks one of four ways to play the rest of the game: the blueprint
itself, or the blueprint biased toward folding, calling or raising. The
Pluribus supplement (Brown & Sandholm 2019, "Depth-limited search") defines the
bias: the probability of folding (calling; *every* raise action) is multiplied
by 5 at all decision points and the distribution is renormalised. Here the
raise actions are ALL_IN and every bet atom, and ``factor`` defaults to 5.

The bias is a multiplier table indexed by a transformation id, so a traced id
selects the continuation and one jitted rollout serves all four.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ..config import build_from_config
from ..envs.betting import CALL, FOLD, NUM_POINT_ATOMS
from ..policy import policy_probs
from ..trainers import latest_checkpoint, load_checkpoint

BLUEPRINT, BIAS_FOLD, BIAS_CALL, BIAS_RAISE = range(4)
N_TRANSFORMS = 4
TRANSFORM_NAMES = ("blueprint", "fold", "call", "raise")
PLURIBUS_FACTOR = 5.0


# ── Transformations ──────────────────────────────────────────────────────────


def bias_multipliers(num_actions: int, factor: float = PLURIBUS_FACTOR) -> jax.Array:
  """``(N_TRANSFORMS, num_actions)`` multipliers, row ``t`` for transformation ``t``.

  ALL_IN sits in the point atoms (index 2) and the bet atoms follow it, so the
  raise row scales everything from ALL_IN onward.
  """
  mult = np.ones((N_TRANSFORMS, num_actions), np.float32)
  mult[BIAS_FOLD, FOLD] = factor
  mult[BIAS_CALL, CALL] = factor
  mult[BIAS_RAISE, NUM_POINT_ATOMS - 1:] = factor
  return jnp.asarray(mult)


def transform_policy(
  probs: jax.Array, legal: jax.Array, t: jax.Array, mult: jax.Array
) -> jax.Array:
  """``probs`` biased by transformation ``t`` and renormalised over legal actions.

  A bias toward an illegal action, such as FOLD when checking is free, leaves
  the policy unchanged. Pluribus does not say what it does there; renormalising
  after the multiply makes this the natural reading.
  """
  biased = jnp.where(legal, probs * mult[t], 0.0)
  return biased / biased.sum(-1, keepdims=True)


# ── Checkpoint ───────────────────────────────────────────────────────────────


class Blueprint:
  """A checkpoint's policy plus jitted env functions on single states.

  ``which`` picks the current, Polyak target or magnet params of QMMD.
  """

  def __init__(self, run_dir: str, checkpoint: str | None = None, which: str = "params") -> None:
    with open(os.path.join(run_dir, "config.yaml")) as f:
      self.cfg = yaml.safe_load(f)
    self.env, self.agent, _ = build_from_config(self.cfg)
    env = self.env
    self.checkpoint = checkpoint or latest_checkpoint(run_dir)
    state = load_checkpoint(self.checkpoint)
    self.step = int(state.step)
    self.which = which
    self.params = {
      "params": state.params,
      "target": state.extras["target_params"],
      "magnet": state.extras["magnet_params"],
    }[which]
    self.eval_fn = self.agent.build_eval_fn(env)
    self.bins = np.asarray(env.bet_bins)
    key = jax.random.PRNGKey(0)

    def probs_fn(params, s, seat):
      infoset = env.information_set(s, seat, key)
      legal = env.legal_actions(s, seat)
      return policy_probs(self.eval_fn(params, infoset[None])[0], legal), legal

    # Unjitted, for vmapping inside other traced code.
    self.probs_fn = probs_fn
    self._probs = jax.jit(probs_fn)
    self._apply = jax.jit(lambda s, a: env.apply_action(s, a)[0])
    self._legal = jax.jit(env.legal_actions)

  def legal(self, s, seat: int) -> np.ndarray:
    return np.asarray(self._legal(s, jnp.int32(seat)))

  def apply(self, s, atom: int):
    return self._apply(s, jnp.full((2,), atom, jnp.int32))

  def policy(self, s, seat: int) -> tuple[np.ndarray, np.ndarray]:
    probs, legal = self._probs(self.params, s, jnp.int32(seat))
    return np.asarray(probs, np.float64), np.asarray(legal)
