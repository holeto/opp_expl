"""Best-response computation on a layered imperfect-information game tree."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


from .types import (
  NK_CHANCE,
  NK_PLAYER,
  NK_SIMULTANEOUS,
  NK_TERMINAL,
  GameTreeLayer,
  LayerTransition,
  LayeredGameTree,
  Strategy,
  policy_prob,
  tree_num_actions,
)


@dataclass(frozen=True)
class BestResponseValues:
  """Root expected payoffs under one-sided best response (two passes).

  ``p0_br`` / ``p1_br`` hold the profile where that player best-responds to the
  fixed opponent strategy; the companion fields are the other player's payoff.
  """

  # Player 0 best-responds to σ₁ (σ₀ ignored at 0's infosets).
  v0_p0_br: float
  v1_p0_br: float
  # Player 1 best-responds to σ₀.
  v0_p1_br: float
  v1_p1_br: float


def _forward_reach_br(
  tree: LayeredGameTree,
  br: int,
  strategy_p0: Strategy,
  strategy_p1: Strategy,
  num_actions: int,
) -> dict[tuple[int, int], float]:
  """Counterfactual reach for BR player ``br``: chance + σ_opponent; mass ×1 on BR moves."""
  reach: dict[tuple[int, int], float] = {tree.root_key: 1.0}
  op = 1 - br

  for L, tr in enumerate(tree.transitions):
    lay = tree.layers[L]
    for e in range(tr.parent_idx.shape[0]):
      p = int(tr.parent_idx[e])
      c = int(tr.child_idx[e])
      rpar = reach.get((L, p), 0.0)
      if rpar <= 0.0:
        continue
      base = rpar * float(tr.edge_prob[e])
      nk = int(lay.node_kind[p])

      if nk == NK_CHANCE:
        w = base
      elif nk == NK_SIMULTANEOUS:
        aj = int(tr.action_p1[e]) if br == 0 else int(tr.action_p0[e])
        info_op = lay.info_set_id_p1[p] if op == 1 else lay.info_set_id_p0[p]
        mask_op = lay.legal_mask_p1[p] if op == 1 else lay.legal_mask_p0[p]
        strat_op = strategy_p1 if op == 1 else strategy_p0
        pj = policy_prob(strat_op, info_op, mask_op, aj, num_actions)
        w = base * pj
      elif nk == NK_PLAYER:
        cp = int(lay.acting_player[p])
        if cp == br:
          w = base
        else:
          aj = int(tr.action_p0[e]) if cp == 0 else int(tr.action_p1[e])
          info_j = lay.info_set_id_p0[p] if cp == 0 else lay.info_set_id_p1[p]
          mask_j = lay.legal_mask_p0[p] if cp == 0 else lay.legal_mask_p1[p]
          strat_j = strategy_p0 if cp == 0 else strategy_p1
          pj = policy_prob(strat_j, info_j, mask_j, aj, num_actions)
          w = base * pj
      else:
        w = base

      key = (L + 1, c)
      reach[key] = reach.get(key, 0.0) + w

  return reach


def _find_child_sequential_br(
  edge_idxs: list[int], tr: LayerTransition, br: int, a_star: int
) -> int:
  for e in edge_idxs:
    a = int(tr.action_p0[e]) if br == 0 else int(tr.action_p1[e])
    if a == a_star:
      return int(tr.child_idx[e])
  raise ValueError(f"no child for BR action {a_star}")


def _sim_br_values(
  edge_idxs: list[int],
  tr: LayerTransition,
  br: int,
  a_star: int,
  lay: GameTreeLayer,
  r: int,
  strategy_p0: Strategy,
  strategy_p1: Strategy,
  num_actions: int,
  V0: dict[tuple[int, int], float],
  V1: dict[tuple[int, int], float],
  L: int,
) -> tuple[float, float]:
  v0 = 0.0
  v1 = 0.0
  for e in edge_idxs:
    a_br = int(tr.action_p0[e]) if br == 0 else int(tr.action_p1[e])
    if a_br != a_star:
      continue
    aj = int(tr.action_p1[e]) if br == 0 else int(tr.action_p0[e])
    info_op = lay.info_set_id_p1[r] if br == 0 else lay.info_set_id_p0[r]
    mask_op = lay.legal_mask_p1[r] if br == 0 else lay.legal_mask_p0[r]
    strat_op = strategy_p1 if br == 0 else strategy_p0
    pj = policy_prob(strat_op, info_op, mask_op, aj, num_actions)
    c = int(tr.child_idx[e])
    v0 += pj * V0[(L + 1, c)]
    v1 += pj * V1[(L + 1, c)]
  return v0, v1


def _backward_br_values(
  tree: LayeredGameTree,
  br: int,
  reach: dict[tuple[int, int], float],
  strategy_p0: Strategy,
  strategy_p1: Strategy,
  num_actions: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
  V0: dict[tuple[int, int], float] = {}
  V1: dict[tuple[int, int], float] = {}
  T = len(tree.layers) - 1

  for L, lay in enumerate(tree.layers):
    for r in range(lay.n_nodes):
      if int(lay.node_kind[r]) == NK_TERMINAL:
        V0[(L, r)] = float(lay.reward_p0[r])
        V1[(L, r)] = float(lay.reward_p1[r])

  for L in range(T - 1, -1, -1):
    lay = tree.layers[L]
    tr = tree.transitions[L]
    E = int(tr.parent_idx.shape[0])
    edges_by_parent: dict[int, list[int]] = defaultdict(list)
    for e in range(E):
      edges_by_parent[int(tr.parent_idx[e])].append(e)

    groups: dict[bytes, list[int]] = defaultdict(list)
    for r in range(lay.n_nodes):
      nk = int(lay.node_kind[r])
      if nk == NK_TERMINAL:
        continue
      if nk == NK_SIMULTANEOUS:
        h = lay.info_set_id_p0[r] if br == 0 else lay.info_set_id_p1[r]
        groups[h].append(r)
      elif nk == NK_PLAYER and int(lay.acting_player[r]) == br:
        h = lay.info_set_id_p0[r] if br == 0 else lay.info_set_id_p1[r]
        groups[h].append(r)

    resolved_br: set[int] = set()
    for _h, rows in groups.items():
      if not rows:
        continue

      action_sets: list[set[int]] = []
      for r in rows:
        acts = {
          int(tr.action_p0[e]) if br == 0 else int(tr.action_p1[e])
          for e in edges_by_parent[r]
        }
        action_sets.append(acts)
      common_actions = set.intersection(*action_sets) if action_sets else set()

      q_acc: dict[int, float] = defaultdict(float)
      for r in rows:
        reach_r = reach.get((L, r), 0.0)
        nk = int(lay.node_kind[r])
        erows = edges_by_parent[r]
        for e in erows:
          c = int(tr.child_idx[e])
          a_br = int(tr.action_p0[e]) if br == 0 else int(tr.action_p1[e])
          if a_br not in common_actions:
            continue
          if nk == NK_SIMULTANEOUS:
            aj = int(tr.action_p1[e]) if br == 0 else int(tr.action_p0[e])
            info_op = lay.info_set_id_p1[r] if br == 0 else lay.info_set_id_p0[r]
            mask_op = lay.legal_mask_p1[r] if br == 0 else lay.legal_mask_p0[r]
            strat_op = strategy_p1 if br == 0 else strategy_p0
            pj = policy_prob(strat_op, info_op, mask_op, aj, num_actions)
            q_acc[a_br] += (
              reach_r * pj * (V0[(L + 1, c)] if br == 0 else V1[(L + 1, c)])
            )
          else:
            q_acc[a_br] += reach_r * (V0[(L + 1, c)] if br == 0 else V1[(L + 1, c)])

      if q_acc:
        a_star = max(q_acc.keys(), key=lambda a: q_acc[a])
      elif common_actions:
        a_star = min(common_actions)
      else:
        r_first = rows[0]
        er = edges_by_parent[r_first]
        if not er:
          raise RuntimeError("BR infoset with no outgoing edges")
        a_star = int(tr.action_p0[er[0]]) if br == 0 else int(tr.action_p1[er[0]])

      for r in rows:
        nk = int(lay.node_kind[r])
        erows = edges_by_parent[r]
        if nk == NK_PLAYER:
          c_star = _find_child_sequential_br(erows, tr, br, a_star)
          V0[(L, r)] = V0[(L + 1, c_star)]
          V1[(L, r)] = V1[(L + 1, c_star)]
        else:
          v0s, v1s = _sim_br_values(
            erows,
            tr,
            br,
            a_star,
            lay,
            r,
            strategy_p0,
            strategy_p1,
            num_actions,
            V0,
            V1,
            L,
          )
          V0[(L, r)] = v0s
          V1[(L, r)] = v1s
        resolved_br.add(r)

    for r in range(lay.n_nodes):
      nk = int(lay.node_kind[r])
      if nk == NK_TERMINAL:
        continue
      if r in resolved_br:
        continue
      erows = edges_by_parent[r]
      if nk == NK_CHANCE:
        v0 = sum(
          float(tr.edge_prob[e]) * V0[(L + 1, int(tr.child_idx[e]))] for e in erows
        )
        v1 = sum(
          float(tr.edge_prob[e]) * V1[(L + 1, int(tr.child_idx[e]))] for e in erows
        )
        V0[(L, r)] = v0
        V1[(L, r)] = v1
      elif nk == NK_PLAYER:
        cp = int(lay.acting_player[r])
        assert cp == 1 - br
        v0 = 0.0
        v1 = 0.0
        for e in erows:
          c = int(tr.child_idx[e])
          aj = int(tr.action_p0[e]) if cp == 0 else int(tr.action_p1[e])
          info_j = lay.info_set_id_p0[r] if cp == 0 else lay.info_set_id_p1[r]
          mask_j = lay.legal_mask_p0[r] if cp == 0 else lay.legal_mask_p1[r]
          strat_j = strategy_p0 if cp == 0 else strategy_p1
          pj = policy_prob(strat_j, info_j, mask_j, aj, num_actions)
          v0 += pj * V0[(L + 1, c)]
          v1 += pj * V1[(L + 1, c)]
        V0[(L, r)] = v0
        V1[(L, r)] = v1
      else:
        raise RuntimeError(f"unexpected node at ({L},{r}) kind={nk}")

  return V0, V1


def best_response_values(
  tree: LayeredGameTree,
  strategy_p0: Strategy,
  strategy_p1: Strategy,
  *,
  num_actions: int | None = None,
) -> BestResponseValues:
  """Best-response values at the root for both players (two passes).

  Pass A: player 0 best-responds to fixed ``strategy_p1`` (``strategy_p0`` is
  ignored at 0's information sets).  Pass B: player 1 best-responds to
  ``strategy_p0``.

  Each pass propagates counterfactual reaches top-down (chance + opponent
  policy on opponent moves; unit mass on each BR move).  Bottom-up, for each
  information set of the BR player, accumulate reach-weighted action values,
  take ``argmax``, then assign state values (both players' expected payoffs)
  consistent with that BR action and the fixed opponent strategy.

  Args:
      tree: Layered tree from ``extract_game_tree``.
      strategy_p0: maps player-0 info-set bytes → length-``num_actions`` prob vector.
      strategy_p1: same for player 1.
      num_actions: if omitted, inferred from ``tree``.
  """
  na = num_actions if num_actions is not None else tree_num_actions(tree)

  r0 = _forward_reach_br(tree, 0, strategy_p0, strategy_p1, na)
  V0_a, V1_a = _backward_br_values(tree, 0, r0, strategy_p0, strategy_p1, na)

  r1 = _forward_reach_br(tree, 1, strategy_p0, strategy_p1, na)
  V0_b, V1_b = _backward_br_values(tree, 1, r1, strategy_p0, strategy_p1, na)

  rk = tree.root_key
  return BestResponseValues(
    v0_p0_br=V0_a[rk],
    v1_p0_br=V1_a[rk],
    v0_p1_br=V0_b[rk],
    v1_p1_br=V1_b[rk],
  )
