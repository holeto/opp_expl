"""Exact NashConv of a network policy on an extracted tree.

Information-set ids in the tree are the raw bytes of ``env.information_set``,
so the network input for each infoset is recovered directly from its id — no
second pass over the env.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp

from .best_response import best_response_values
from .types import NK_PLAYER, NK_TERMINAL, LayeredGameTree, Strategy


def collect_infosets(
  tree: LayeredGameTree, env: Any, player: int
) -> tuple[list[bytes], np.ndarray, np.ndarray]:
  """Unique infosets where ``player`` acts: (ids, obs (N, d), legal masks (N, A))."""
  probe = env.init_state(jax.random.PRNGKey(0))
  dtype = np.asarray(env.information_set(probe, jnp.int32(player), jax.random.key(0))).dtype

  ids: list[bytes] = []
  masks: list[np.ndarray] = []
  seen: set[bytes] = set()
  for lay in tree.layers:
    iset_ids = lay.info_set_id_p0 if player == 0 else lay.info_set_id_p1
    legal = lay.legal_mask_p0 if player == 0 else lay.legal_mask_p1
    for i in range(lay.n_nodes):
      nk = int(lay.node_kind[i])
      if nk == NK_TERMINAL or (nk == NK_PLAYER and int(lay.acting_player[i]) != player):
        continue
      iset = iset_ids[i]
      if not iset or iset in seen:  # b"" marks the artificial chance root
        continue
      seen.add(iset)
      ids.append(iset)
      masks.append(legal[i])
  obs = np.stack([np.frombuffer(i, dtype=dtype) for i in ids]).astype(np.float32)
  return ids, obs, np.stack(masks).astype(bool)


def strategy_from_logits(ids: list[bytes], logits: np.ndarray, masks: np.ndarray) -> Strategy:
  """Masked softmax per infoset → Strategy dict."""
  z = np.where(masks, logits.astype(np.float64), -np.inf)
  z -= z.max(-1, keepdims=True)
  p = np.where(masks, np.exp(z), 0.0)
  p /= p.sum(-1, keepdims=True)
  return dict(zip(ids, p))


class NashConvEvaluator:
  """Builds the tree and infoset batches once; evaluates any params cheaply.

  ``apply_fn`` is ``agent.build_eval_fn(env)``: (params, infoset_batch) → logits.
  NashConv = v0(BR₀ vs σ₁) + v1(BR₁ vs σ₀), in the env's raw reward units.
  """

  def __init__(self, tree: LayeredGameTree, env: Any, apply_fn: Callable) -> None:
    self.tree = tree
    self.apply_fn = apply_fn
    self.infosets = [collect_infosets(tree, env, p) for p in (0, 1)]

  def strategies(self, params: Any) -> tuple[Strategy, Strategy]:
    out = []
    for ids, obs, masks in self.infosets:
      logits = np.asarray(self.apply_fn(params, jnp.asarray(obs)))
      out.append(strategy_from_logits(ids, logits, masks))
    return out[0], out[1]

  def __call__(self, params: Any) -> dict[str, float]:
    s0, s1 = self.strategies(params)
    bv = best_response_values(self.tree, s0, s1)
    return {
      "br_p0": float(bv.v0_p0_br),
      "br_p1": float(bv.v1_p1_br),
      "nash_conv": float(bv.v0_p0_br + bv.v1_p1_br),
    }
