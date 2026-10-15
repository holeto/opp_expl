"""Exact game-tree extraction and best response for small games (from RLG ``src/tree``).

``extract_game_tree`` expands the full tree in BFS order; each wave becomes one
``GameTreeLayer`` (one row per distinct history, never merged) with sparse
``LayerTransition`` edges to the next. ``best_response_values`` runs two
passes (each player best-responding to the other's fixed strategy) with
counterfactual reach propagation and an argmax per information set, giving
exact NashConv = v0_p0_br + v1_p1_br.

Full enumeration, so small games only: Goofspiel up to ~5 cards
(``prize_order='random'`` adds a chance root over all n! prize orders).
Payoffs are the env's raw rewards (Goofspiel 'difference': score gap).

Distinct from ``oppex.cfr``: that is a public-state solver specialised to HUNL
preflop; this is a generic history tree for sanity-checking learned policies.
"""

from .best_response import BestResponseValues, best_response_values
from .evaluate import NashConvEvaluator, collect_infosets, strategy_from_logits
from .extract import extract_game_tree
from .types import (
  EK_CHANCE, EK_PLAYER0, EK_PLAYER1, EK_SIMULTANEOUS,
  GameTreeLayer, LayerTransition, LayeredGameTree,
  NK_CHANCE, NK_PLAYER, NK_SIMULTANEOUS, NK_TERMINAL,
  Strategy, policy_prob, tree_num_actions,
)

__all__ = [
  "BestResponseValues", "best_response_values", "extract_game_tree",
  "NashConvEvaluator", "collect_infosets", "strategy_from_logits",
  "EK_CHANCE", "EK_PLAYER0", "EK_PLAYER1", "EK_SIMULTANEOUS",
  "GameTreeLayer", "LayerTransition", "LayeredGameTree",
  "NK_CHANCE", "NK_PLAYER", "NK_SIMULTANEOUS", "NK_TERMINAL",
  "Strategy", "policy_prob", "tree_num_actions",
]
