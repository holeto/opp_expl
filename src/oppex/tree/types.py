"""Shared layered-tree datatypes and small strategy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Node / edge kind codes (uint8) ───────────────────────────────────────────

NK_CHANCE = 0
NK_SIMULTANEOUS = 1
NK_PLAYER = 2
NK_TERMINAL = 3

EK_CHANCE = 0
EK_SIMULTANEOUS = 1
EK_PLAYER0 = 2
EK_PLAYER1 = 3

# Strategy maps information-set bytes → probability vector (length num_actions)
Strategy = dict[bytes, np.ndarray]


@dataclass
class LayerTransition:
  """Sparse directed edges from ``layers[t]`` → ``layers[t + 1]``."""

  parent_idx: np.ndarray  # int32 (E,), indices into parent layer
  child_idx: np.ndarray  # int32 (E,), indices into child layer
  edge_prob: np.ndarray  # float64 (E,), chance / init probability (else 1.0)
  edge_kind: np.ndarray  # uint8 (E,), EK_* constants
  action_p0: np.ndarray  # int32 (E,), simultaneous / unused → -1
  action_p1: np.ndarray  # int32 (E,)


@dataclass
class GameTreeLayer:
  """All nodes at one ply / one BFS depth (one row per distinct history)."""

  depth: int
  n_nodes: int
  node_kind: np.ndarray  # uint8 (N,), NK_*
  acting_player: np.ndarray  # int8 (N,), -1 if chance / simultaneous / terminal
  legal_mask_p0: np.ndarray  # bool (N, num_actions)
  legal_mask_p1: np.ndarray  # bool (N, num_actions)
  info_set_id_p0: tuple[bytes, ...]
  info_set_id_p1: tuple[bytes, ...]
  reward_p0: np.ndarray  # float32 (N,), nonzero only for terminals
  reward_p1: np.ndarray  # float32 (N,)

  # Stacked env states for this layer (optional; leading axis N). Terminals included.
  state_batch: Any | None = None


@dataclass
class LayeredGameTree:
  layers: list[GameTreeLayer]
  transitions: list[LayerTransition]

  @property
  def n_layers(self) -> int:
    return len(self.layers)

  @property
  def n_transitions(self) -> int:
    return len(self.transitions)

  @property
  def n_nodes(self) -> int:
    return sum(L.n_nodes for L in self.layers)

  @property
  def root_key(self) -> tuple[int, int]:
    """(layer, node) index of the tree root, always (0, 0).

    Single-root trees: layer 0 is the game root (NK_PLAYER or NK_SIMULTANEOUS).
    Multi-root trees: layer 0 is the artificial NK_CHANCE node whose edges carry
    the initial-state probabilities; layer 1 holds the actual game starts.
    In both cases (0, 0) is the unique entry point and carries the
    deal-probability-weighted root value after a backward pass.
    """
    return (0, 0)


def tree_num_actions(tree: LayeredGameTree) -> int:
  for lay in tree.layers:
    if lay.n_nodes == 0:
      continue
    if lay.legal_mask_p0.shape[1] > 0:
      return int(lay.legal_mask_p0.shape[1])
  raise ValueError("cannot infer num_actions from tree")


def policy_prob(
  strategy: Strategy,
  info: bytes,
  legal_mask: np.ndarray,
  action: int,
  num_actions: int,
) -> float:
  if not bool(legal_mask[action]):
    return 0.0
  n_legal = int(legal_mask.sum())
  raw = strategy.get(info)
  if raw is None:
    return 1.0 / n_legal
  arr = np.asarray(raw, dtype=np.float64)
  if arr.size < num_actions:
    arr = np.pad(arr, (0, num_actions - arr.size))
  s = float(arr[legal_mask].sum())
  if s < 1e-12:
    return 1.0 / n_legal
  return float(arr[action] / s)
