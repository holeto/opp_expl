"""Explicit sequence-form LP over the preflop public tree.

The reference solver. Unlike ``lp_reference``, which hand-derives a compact LP for
the two-move shove-fold game by substituting one player's constraints away and
recovering the other's from the duals, this builds the textbook formulation
straight off the tree, with every realisation constraint written out as a row.

That costs variables and buys two things. It is readable — the sequence-form
equality

    Σ_{a ∈ A(I)} r(σ_I · a) = r(σ_I)

appears literally as a row of ``A_eq`` rather than as an eliminated variable or a
dual multiplier. And it is *general*: it handles trees where a player acts more
than once, which the compact derivation structurally cannot, since eliminating
``r(σ_I·a)`` needs the parent realisation ``r(σ_I)`` to be a constant.

**Formulation** (Koller-Megiddo-von Stengel). Player 0 maximises ``x^T A y`` where
``E x = e, x >= 0`` and player 1 minimises with ``F y = f, y >= 0``. Dualising the
inner minimisation gives a single LP:

    max_{x, q}  f^T q
    s.t.        F^T q - A^T x <= 0        (one row per player-1 sequence)
                E x = e,  x >= 0
                q free

Player 1's equilibrium realisation plan is then the dual of the inequality block.

**Information sets.** Both players hold private cards, so an information set is a
*(public decision node, hand)* pair and a sequence is a *(public action path,
hand)* pair. Chance never enters the constraints — a realisation plan is
conditional on the player's own choices, and the player cannot influence which
hand they are dealt. The deal probability ``MASK[a,b] / N_DEALS`` therefore lives
entirely in the payoff blocks.

**Payoff assembly.** Every terminal is reached by exactly one public sequence per
player, and contributes one dense ``(n_hands, n_hands)`` block to ``A`` at that
sequence pair:

    fold terminal      const * MASK / N_DEALS
    showdown/checkdown stake * (MASK * EV) / N_DEALS

The empty sequence is kept as an explicit variable pinned to 1 by its own equality
row, rather than substituted out. That keeps every terminal uniformly bilinear:
terminals where one player never acted (an immediate fold) would otherwise be
linear terms needing separate handling.

**Matrix-valued leaves** fit the formulation directly. At leaf ``l`` each
player has one information set per hand, whose ``T`` actions are the
continuations, extending that player's own sequence into the leaf. Neither
player's constraint mentions the other's choice, which is exactly what makes
the choice simultaneous. Every pair ``(t0, t1)`` is a terminal with block

    matrix leaf        values[l, t0, t1] / N_DEALS

already mask-folded, like ``EV``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_array, csr_matrix, hstack, vstack

from .cards import MASK_NP, N_DEALS
from .tree import PreflopTree


class SequenceForm(NamedTuple):
  """Sequence indices and constraint matrices for one built tree.

  A public infoset is a decision node or one player's side of a matrix leaf;
  the latter is keyed ``("leaf", i)`` wherever a node id would go.
  """

  n_seq: tuple[int, int]          # public sequences per player, index 0 = empty
  seq_of: dict                    # (player, node_id, atom) -> public sequence index
  parent_seq: dict                # (player, node_id) -> owner's sequence leading in
  terminals: list                 # (seq0, seq1, const, stake, (leaf, t0, t1) | None)
  n_hands: int
  infosets: tuple[list, list]     # per player: (node_id, parent seq, child seqs)


# ── Tree walk ────────────────────────────────────────────────────────────────


def enumerate_sequences(
  tree: PreflopTree, n_hands: int, n_transforms: int = 0,
) -> SequenceForm:
  """Assign a public sequence index to every (public infoset, legal action) edge.

  ``n_transforms`` is the number of continuations at each matrix leaf.
  """
  if tree.leaves and n_transforms <= 0:
    raise ValueError("the tree has matrix leaves; pass their n_transforms")
  n_pub = [1, 1]                                   # index 0 is the empty sequence
  seq_of: dict = {}
  parent_seq: dict = {}
  terminals: list = []
  infosets: tuple[list, list] = ([], [])
  leaf_id = {j: i for i, j in enumerate(tree.leaves)}

  def new_infoset(p, key, actions, cur):
    # Allocates one sequence per action, extending player p's sequence `cur`.
    parent_seq[(p, key)] = cur
    children = []
    for a in actions:
      seq_of[(p, key, a)] = n_pub[p]
      children.append(n_pub[p])
      n_pub[p] += 1
    infosets[p].append((key, cur, children))
    return children

  def walk(ref, cur):
    kind, i = ref
    if kind == "T":
      t = tree.terminals[i]
      if i not in leaf_id:
        terminals.append((cur[0], cur[1], t.const, t.stake, None))
        return
      key = ("leaf", leaf_id[i])
      s0 = new_infoset(0, key, range(n_transforms), cur[0])
      s1 = new_infoset(1, key, range(n_transforms), cur[1])
      for t0 in range(n_transforms):
        for t1 in range(n_transforms):
          terminals.append((s0[t0], s1[t1], 0.0, 0.0, (leaf_id[i], t0, t1)))
      return
    node = tree.nodes[i]
    p = node.player
    legal = [a for a, c in enumerate(node.child) if c is not None]
    for a, s in zip(legal, new_infoset(p, i, legal, cur[p])):
      nxt = list(cur)
      nxt[p] = s
      walk(node.child[a], nxt)

  walk(("D", tree.root), [0, 0])
  return SequenceForm(tuple(n_pub), seq_of, parent_seq, terminals, n_hands, infosets)


def constraint_matrices(tree: PreflopTree, sf: SequenceForm, player: int):
  """``(E, e)`` for one player: the realisation constraints, one row per infoset.

  Variables are flattened as ``sequence * n_hands + hand``. Two kinds of row:

  * ``r(empty, h) = 1`` — one per hand, pinning the root realisation.
  * ``Σ_a r(σ_I·a, h) - r(σ_I, h) = 0`` — one per (public infoset, hand), which
    is the sequence-form equality verbatim.
  """
  h = sf.n_hands
  n_var = sf.n_seq[player] * h
  # Each public infoset holds n_hands private ones, hence identity blocks.
  ident = csr_matrix(np.eye(h))
  rows, rhs = [], []

  # Each root infoset has realisation 1.
  blocks = [None] * sf.n_seq[player]
  blocks[0] = ident
  rows.append(_row_block(blocks, h))
  rhs.append(np.ones(h))

  for key, parent, children in sf.infosets[player]:
    blocks = [None] * sf.n_seq[player]
    for c in children:
      blocks[c] = ident
    # `enumerate_sequences` hands out strictly increasing indices as it descends,
    # and the sequence leading *into* an infoset is allocated at an ancestor (or
    # is the empty sequence 0), so it is always strictly below every sequence
    # leading out. A collision would mean the infoset is reachable from itself —
    # a cycle rather than a tree — so assert it instead of merging blocks.
    assert blocks[parent] is None, (
      f"infoset {key}: incoming sequence {parent} is also one of its outgoing "
      "sequences, so the tree contains a cycle"
    )
    blocks[parent] = -ident
    rows.append(_row_block(blocks, h))
    rhs.append(np.zeros(h))

  E = vstack(rows, format="csr")
  assert E.shape[1] == n_var
  return E, np.concatenate(rhs)


def _row_block(blocks, h):
  """One block-row, substituting explicit sparse zeros for the empty slots."""
  zero = csr_matrix((h, h))
  return hstack([zero if b is None else b for b in blocks], format="csr")


def payoff_matrix(sf: SequenceForm, ev: np.ndarray, leaf_values=None):
  """``A`` with one dense ``(n_hands, n_hands)`` block per terminal.

  Rows are player 0's sequences, columns player 1's, both flattened by hand. The
  deal probability is folded in here, which is the only place chance appears.
  """
  h = sf.n_hands
  mask = MASK_NP.astype(np.float64)
  m_ev = mask * np.asarray(ev, dtype=np.float64)

  grid = [[None] * sf.n_seq[1] for _ in range(sf.n_seq[0])]
  for s0, s1, const, stake, leaf in sf.terminals:
    if leaf is None:
      block = (const * mask + stake * m_ev) / N_DEALS
    else:
      # Leaf values are mask-folded already, so no second mask here.
      block = np.asarray(leaf_values[leaf], np.float64) / N_DEALS
    grid[s0][s1] = block if grid[s0][s1] is None else grid[s0][s1] + block

  zero = csr_matrix((h, h))
  grid = [[zero if b is None else csr_matrix(b) for b in row] for row in grid]
  return block_array(grid, format="csr")


# ── Solve ────────────────────────────────────────────────────────────────────


def solve(
  tree: PreflopTree, ev: np.ndarray, n_hands: int, *, method: str = "highs",
  leaf_values=None,
):
  """Exact Nash equilibrium of ``tree`` by sequence-form LP.

  Returns ``(value_to_player0_chips, realisation_0, realisation_1, sf)``, where
  each realisation plan is shaped ``(n_public_sequences, n_hands)``. Behavioural
  probabilities come from dividing a sequence by its parent — see
  ``behaviour_strategy``. ``leaf_values`` is required when the tree has matrix
  leaves, shaped ``(n_leaves, T, T, n_hands, n_hands)``.
  """
  if tree.leaves and leaf_values is None:
    raise ValueError(f"the tree has {len(tree.leaves)} matrix leaves but no leaf_values")
  n_t = 0 if leaf_values is None else int(np.shape(leaf_values)[1])
  sf = enumerate_sequences(tree, n_hands, n_t)
  E0, e0 = constraint_matrices(tree, sf, 0)
  E1, e1 = constraint_matrices(tree, sf, 1)
  #Matrix is unoptimized, containing 
  # empty sequence row/collumn also per infoset
  # This just for ease of implementation of the 
  # cases where small blind folds where big blind 
  # did not take an action yet.
  A = payoff_matrix(sf, ev, leaf_values)

  n_x, n_q = A.shape[0], E1.shape[0]
  # max f^T q  s.t.  F^T q - A^T x <= 0,  E x = e,  x >= 0
  a_ub = hstack([-A.T, E1.T], format="csr")
  a_eq = hstack([E0, csr_matrix((E0.shape[0], n_q))], format="csr")
  c = np.concatenate([np.zeros(n_x), -e1])
  bounds = [(0.0, None)] * n_x + [(None, None)] * n_q

  res = linprog(c, A_ub=a_ub, b_ub=np.zeros(a_ub.shape[0]),
                A_eq=a_eq, b_eq=e0, bounds=bounds, method=method)
  if not res.success:
    raise RuntimeError(f"sequence-form LP failed: {res.message}")

  x = np.clip(res.x[:n_x], 0.0, None).reshape(sf.n_seq[0], n_hands)
  # Player 1's realisation plan is the dual of the inequality block.
  y = np.clip(-np.asarray(res.ineqlin.marginals), 0.0, None).reshape(sf.n_seq[1], n_hands)
  value = float(-res.fun)
  _check_realisation(sf, x, 0)
  _check_realisation(sf, y, 1)
  return value, x, y, sf


def behaviour_strategy(tree, sf: SequenceForm, r: np.ndarray, player: int) -> dict:
  """Per-infoset behavioural probabilities ``(n_hands, n_actions)`` from a plan.

  ``sigma(a | I) = r(σ_I · a) / r(σ_I)``, uniform where the infoset is unreached
  (realisation zero), which is the standard convention — the strategy there is
  unconstrained by the equilibrium. Keys are node ids, and ``("leaf", i)`` for
  matrix leaves, whose columns are the continuations.
  """
  out = {}
  for key, parent_s, children in sf.infosets[player]:
    width = tree.n_actions if isinstance(key, int) else len(children)
    actions = [a for (p, k, a) in sf.seq_of if p == player and k == key]
    parent = r[parent_s]                               # (n_hands,)
    sigma = np.zeros((sf.n_hands, width))
    for a in actions:
      sigma[:, a] = r[sf.seq_of[(player, key, a)]]
    denom = np.where(parent > 1e-12, parent, 1.0)[:, None]
    sigma = np.where(parent[:, None] > 1e-12, sigma / denom, 0.0)
    unreached = parent <= 1e-12
    if unreached.any():
      for a in actions:
        sigma[unreached, a] = 1.0 / len(actions)
    out[key] = sigma
  return out


def _check_realisation(sf, r, player, tol=1e-6):
  """Assert the sequence-form equalities actually hold in the returned plan."""
  err = float(np.abs(r[0] - 1.0).max())
  assert err <= tol, f"player {player} empty sequence != 1 (max {err:.3e})"
  for key, parent, children in sf.infosets[player]:
    total = sum(r[c] for c in children)
    err = float(np.abs(total - r[parent]).max())
    assert err <= tol, (
      f"player {player} infoset {key}: actions sum to {err:.3e} away from the "
      "parent realisation"
    )
  assert r.min() >= -tol, f"player {player} realisation has negative entries"
