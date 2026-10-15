"""Standard trainer: jitted scan over ``log_every`` steps, stdout logging, pickled checkpoints.

Checkpoints are named ``step_{step:08d}.pkl`` after the training step they hold,
so ``latest_checkpoint`` and the evaluation scripts can sort and parse them.
(RLG saved ``{counter}.pkl`` while its resume path globbed ``step_*.pkl``, so
resuming never found anything.)
"""

from __future__ import annotations
import glob
import math
import os
import pickle
import time
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from .base import Logger, Trainer
from ..algorithms.base import Algorithm


class StdoutLogger(Logger):
  """Prints metrics to stdout, one line per logged chunk."""

  def __init__(self, keys: list[str] | None = None) -> None:
    self._keys = keys

  def write(self, metrics: dict[str, float], step: int) -> None:
    keys = self._keys or sorted(metrics)
    parts = "  ".join(f"{k}: {_fmt(metrics[k])}" for k in keys if k in metrics)
    print(f"step {step:8d} | {parts}", flush=True)

  def close(self) -> None:
    pass


def _fmt(v: float) -> str:
  """Fixed point, except small magnitudes (a decayed learning rate) in scientific."""
  return f"{v:.3e}" if 0 < abs(v) < 1e-3 else f"{v:.4f}"


def checkpoint_path(checkpoint_dir: str, step: int) -> str:
  return os.path.join(checkpoint_dir, f"step_{step:08d}.pkl")


def list_checkpoints(checkpoint_dir: str) -> list[str]:
  """Checkpoint paths in ``checkpoint_dir``, oldest first."""
  return sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pkl")))


def load_checkpoint(path: str) -> Any:
  with open(path, "rb") as f:
    return pickle.load(f)


def latest_checkpoint(checkpoint_dir: str) -> str:
  paths = list_checkpoints(checkpoint_dir)
  if not paths:
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir!r}")
  return paths[-1]


def load_metrics(checkpoint_dir: str) -> dict[str, np.ndarray]:
  """Per-step metric arrays saved alongside checkpoints (``save_metrics=True``)."""
  return dict(np.load(os.path.join(checkpoint_dir, "metrics.npz")))


class StandardTrainer(Trainer):
  """Calls algorithm.step in jitted chunks, logs, checkpoints.

  Args:
    algorithm:        any Algorithm.
    log_every:        steps per jitted chunk; metrics are averaged per chunk.
    checkpoint_every: save every this many steps (None disables); rounded up
                      to a multiple of log_every.
    checkpoint_dir:   directory for checkpoints and metrics.npz.
    logger:           defaults to StdoutLogger.
    save_metrics:     also write raw per-step metrics to metrics.npz. On resume
                      the file restarts from the resumed step.
  """

  def __init__(
    self,
    algorithm: Algorithm,
    log_every: int = 1,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | None = None,
    logger: Logger | None = None,
    save_metrics: bool = False,
  ) -> None:
    self.algorithm = algorithm
    self.log_every = log_every
    self.checkpoint_dir = checkpoint_dir
    self.logger = logger or StdoutLogger()
    self.save_metrics = save_metrics
    if checkpoint_every is not None:
      rounded = math.ceil(checkpoint_every / log_every) * log_every
      if rounded != checkpoint_every:
        print(f"checkpoint_every rounded {checkpoint_every} → {rounded} (log_every={log_every})")
      checkpoint_every = rounded
    self.checkpoint_every = checkpoint_every

  def train(self, state: Any, n_steps: int) -> Any:
    """Run n_steps iterations (rounded up to a multiple of log_every)."""
    chunk = self.log_every
    rounded = math.ceil(n_steps / chunk) * chunk
    if rounded != n_steps:
      print(f"n_steps rounded {n_steps} → {rounded} (log_every={chunk})")
    n_steps = rounded

    def scan_fn(carry, _):
      return self.algorithm.step(carry)

    run_chunk = jax.jit(lambda s: jax.lax.scan(scan_fn, s, None, length=chunk))

    # Save the starting point, unless it is the checkpoint we just resumed from.
    if self.checkpoint_every and self.checkpoint_dir and not os.path.exists(
      checkpoint_path(self.checkpoint_dir, int(state.step))
    ):
      self._save(state)

    t0 = time.perf_counter()
    steps_done = 0
    accumulated: dict[str, list[np.ndarray]] = {}

    for _ in range(n_steps // chunk):
      state, chunk_metrics = run_chunk(state)
      steps_done += chunk
      step = int(state.step)

      scalars = {k: float(jnp.mean(v)) for k, v in chunk_metrics.items()}
      scalars["steps_per_sec"] = steps_done / (time.perf_counter() - t0)
      self.logger.write(scalars, step)

      if self.save_metrics and self.checkpoint_dir:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        for k, v in chunk_metrics.items():
          accumulated.setdefault(k, []).append(np.asarray(v))
        np.savez(
          os.path.join(self.checkpoint_dir, "metrics.npz"),
          **{k: np.concatenate(vs) for k, vs in accumulated.items()},
        )

      if self.checkpoint_every and step % self.checkpoint_every == 0:
        self._save(state)

    self.logger.close()
    return state

  def _save(self, state: Any) -> None:
    if self.checkpoint_dir is None:
      return
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    path = checkpoint_path(self.checkpoint_dir, int(state.step))
    with open(path, "wb") as f:
      pickle.dump(jax.device_get(state), f)
    print(f"checkpoint saved → {path}", flush=True)
