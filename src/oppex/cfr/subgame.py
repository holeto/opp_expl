"""Depth-limited public subgames rooted anywhere in the hand, for re-solving.

``build_subgame`` enumerates the betting tree below an arbitrary
``BettingState``, on any street, down to ``depth_limit`` plies. It yields the
same ``PublicTree`` as ``tree.build_preflop_tree``: with the preflop root, no
depth limit and matrix leaves, the two agree node for node, which
``scripts/resolve_fixed_opponent.py`` checks.

A line ends in one of three ways:

* ``FOLD`` — ``const`` chips to player 0;
* ``SHOWDOWN`` — the hand is over (a called all-in, or the river closed);
  ``stake`` chips change hands at the sign of the showdown;
* ``MATRIX`` — a depth-limit leaf, where continuation strategies take over.
  It is placed where the street closes, because the next card is chance and a
  subgame has no chance nodes, and at ply ``depth_limit``.

Paths are full histories from the deal (``root_path`` plus the local line), so
a node's path names the real hand it stands for.

The builder steps all atoms of a node in one jitted call. Stepping them one by
one eagerly, as the preflop builder does, costs ~27 ms per node, which is too
slow to rebuild a subgame at every decision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..envs import betting
from ..envs.betting import BettingRules, BettingState
from .tree import (
  FOLD_T, MATRIX_T, SHOWDOWN_T, DecisionNode, PublicTree, TerminalNode, _key,
)

_EXPANDERS: dict = {}


def _expander(rules: BettingRules):
  # A jitted bs -> (legal, all children) for these rules, built once.
  key = (float(rules.starting_stack), float(rules.small_blind), float(rules.big_blind),
         tuple(np.asarray(rules.bet_bins).tolist()))
  if key not in _EXPANDERS:
    n_act = betting.num_atoms(rules)

    @jax.jit
    def expand(bs):
      legal = betting.legal_atoms(rules, bs, drop_duplicate_allin=True)
      out = jax.vmap(lambda a: betting.step(rules, bs, a))(jnp.arange(n_act, dtype=jnp.int32))
      return legal, out

    _EXPANDERS[key] = expand
  return _EXPANDERS[key]


def expand_node(rules: BettingRules, bs: BettingState):
  """All children of a decision, with duplicate siblings struck.

  Returns ``(legal, alias, children, flags)``: tree legality per atom,
  ``DecisionNode.alias``, the child ``BettingState`` per atom (``None`` when
  illegal or struck) and the step's ``(done, is_fold, advance)`` per atom.
  """
  legal, out = jax.device_get(_expander(rules)(bs))
  legal = [bool(x) for x in legal]
  alias = [a if ok else -1 for a, ok in enumerate(legal)]
  children: list = [None] * len(legal)
  seen: dict = {}
  for atom in range(len(legal)):
    if not legal[atom]:
      continue
    child = BettingState(*(np.asarray(x[atom]) for x in out.state))
    # Two sibling edges reaching the same public state are the same action.
    k = _key(child)
    if k in seen:
      legal[atom] = False
      alias[atom] = seen[k]
      continue
    seen[k] = atom
    children[atom] = child
  return legal, alias, children, (out.done, out.is_fold, out.advance)


def replay_betting(rules: BettingRules, history) -> list:
  """``[(state, alias)]`` before each atom of ``history``, then the final state.

  The last entry's alias is ``None``. Struck duplicates in the history are
  followed to the state they produce, which is the same public state.
  """
  bs = betting.initial_state(rules)
  bs = BettingState(*(np.asarray(x) for x in bs))
  out = []
  for atom in history:
    _, alias, children, _ = expand_node(rules, bs)
    if alias[int(atom)] < 0:
      raise ValueError(f"atom {atom} is illegal after {len(out)} steps of {list(history)}")
    out.append((bs, tuple(alias)))
    bs = children[alias[int(atom)]]
  out.append((bs, None))
  return out


def build_subgame(
  rules: BettingRules, root: BettingState, root_path: tuple[int, ...] = (),
  depth_limit: int | None = 4, max_nodes: int = 100_000,
) -> PublicTree:
  """Enumerate the subgame below ``root`` (see the module docstring).

  ``root`` must be a live decision, not a finished hand. ``depth_limit=None``
  only stops at street closes.
  """
  if bool(np.asarray(root.done)):
    raise ValueError("the subgame root is a finished hand")
  n_act = betting.num_atoms(rules)
  nodes: list = []
  terminals: list[TerminalNode] = []

  def add_terminal(kind, const, stake, bs, path):
    terminals.append(TerminalNode(kind, float(const), float(stake), bs, path))
    return ("T", len(terminals) - 1)

  def visit(bs: BettingState, depth: int, path: tuple[int, ...]):
    slot = len(nodes)
    if slot >= max_nodes:
      raise RuntimeError(f"subgame exceeded max_nodes={max_nodes}")
    # Reserve the slot, so children get later indices, as in the preflop builder.
    nodes.append(None)
    legal, alias, kids, (done, is_fold, advance) = expand_node(rules, bs)
    children: list = [None] * n_act
    player = int(bs.cur_player)

    for atom, child in enumerate(kids):
      if child is None:
        continue
      committed = child.committed
      line = path + (atom,)
      if done[atom] and is_fold[atom]:
        # The folder forfeits what they put in; chips to player 0.
        const = committed[1] if player == 1 else -committed[0]
        children[atom] = add_terminal(FOLD_T, const, 0.0, child, line)
      elif done[atom]:
        children[atom] = add_terminal(SHOWDOWN_T, 0.0, committed[1], child, line)
      elif advance[atom] or (depth_limit is not None and depth + 1 >= depth_limit):
        children[atom] = add_terminal(MATRIX_T, 0.0, 0.0, child, line)
      else:
        children[atom] = visit(child, depth + 1, line)

    nodes[slot] = DecisionNode(
      player=player, legal=tuple(legal), child=tuple(children), betting=bs,
      path=path, alias=tuple(alias),
    )
    return ("D", slot)

  root = BettingState(*(np.asarray(x) for x in root))
  visit(root, 0, tuple(int(a) for a in root_path))
  return PublicTree(rules, tuple(nodes), tuple(terminals), n_act, 0)
