"""Trainer and Logger abstractions (Layer 6).

The Trainer only calls ``algorithm.step`` in a loop, logs metrics and saves
checkpoints. It has no influence on how training proceeds inside the algorithm.
"""

from __future__ import annotations
import abc
from typing import Any


class Logger(abc.ABC):
  @abc.abstractmethod
  def write(self, metrics: dict[str, float], step: int) -> None: ...

  @abc.abstractmethod
  def close(self) -> None: ...


class Trainer(abc.ABC):
  @abc.abstractmethod
  def train(self, state: Any, n_steps: int) -> Any:
    """Run n_steps training iterations and return the final state."""
