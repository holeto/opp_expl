"""Algorithm abstract base and the training-state pytree.

An Algorithm is a self-contained training unit: it owns the environment, the
agent, the optimizer and all logic to go from one state to the next in a single
pure ``step``. The Trainer only calls ``step`` in a jitted scan and handles side
effects (logging, checkpointing).
"""

from __future__ import annotations
import abc
import inspect
from typing import Any, Callable, NamedTuple

import jax
import optax


class TrainingState(NamedTuple):
  params: Any         # network parameters
  opt_state: Any      # optax optimizer state
  env_state: Any      # (B, ...) live env states — carried across steps (auto-reset rollouts)
  agent_state: Any    # (B, ...) recurrent carry, None for stateless agents
  ep_len: jax.Array   # (B,) int32 — steps taken in each row's current episode
  rng: jax.Array      # PRNG key, threaded through every step
  step: jax.Array     # int32 scalar — training iterations done
  extras: Any = None  # algorithm-specific (target_params, magnet_params for QMMD)


class Algorithm(abc.ABC):
  """Self-contained training unit."""

  @abc.abstractmethod
  def init(self, key: jax.Array) -> TrainingState:
    """Initialise params, optimizer state, env states, rng and step counter."""

  @abc.abstractmethod
  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    """One full iteration: collect data, build targets, apply all updates.

    Returns the updated state and a dict of scalar metrics.
    """


# ── Learning rate ────────────────────────────────────────────────────────────


def inject_learning_rate(factory: Callable[..., optax.GradientTransformation], learning_rate, **kwargs):
  """``factory(learning_rate, **kwargs)`` with only the learning rate in the optimizer state.

  ``learning_rate`` may be a float or an optax schedule; the value applied at
  each update is stored in the state, where ``applied_learning_rate`` finds it.
  Every other argument is static: injecting Adam's ``b1``/``b2``/``eps`` as
  float32 arrays changes the bias-correction rounding by an ulp, which is
  enough for a run to drift away from the same run on plain ``optax.adam``.
  """
  static = tuple(n for n in inspect.signature(factory).parameters if n != "learning_rate")
  return optax.inject_hyperparams(factory, static_args=static)(learning_rate=learning_rate, **kwargs)


def applied_learning_rate(opt_state: Any) -> jax.Array | None:
  """Learning rate of the last update, if the optimizer injects it (else None).

  ``inject_hyperparams`` stores the evaluated rate (a plain array) next to the
  schedule's own state under the same key; the filter keeps only the array.
  """
  return optax.tree_utils.tree_get(
    opt_state, "learning_rate", filtering=lambda _, v: isinstance(v, jax.Array)
  )
