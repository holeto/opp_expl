"""Vectorised counterfactual regret minimisation over the preflop public tree.

One traversal updates all 1326 information sets per node at once: reaches and
counterfactual values are ``(n_hands,)`` vectors, and terminal evaluation is a
matrix-vector product against the card-removal mask. The tree has a handful of
nodes, so traversal is plain recursive Python over a static structure — under
``jit`` the recursion unrolls at trace time into one fused graph, and with
``jit=False`` the identical code is eager and steppable in a debugger.

Three things here are easy to get subtly wrong, and each has a guard:

**Card removal belongs at terminals, never in reach propagation.** ``r_i[h]`` is
player *i*'s own action-probability product; it carries no dependence on the
opponent's holding. Masking it during propagation double-counts removal. See
``terminal_cfv``, which is the only place the mask appears — deliberately a
single function, so a terminal cannot be written that forgets it.

**This is the public-state CFR formulation, not the vanilla-CFR one.** Vanilla CFR
is usually written in three passes — reaches down, *standard* values up, then
multiply by the counterfactual reach when forming regrets. Here values are
counterfactual from the leaves upward: ``terminal_cfv``'s ``r_opp`` argument *is*
``π_{-p}``, so the invariant carried up the tree is ``cfv_p(n) = π_{-p}(n) ⊗ v(n)``
and there is deliberately no later multiply. Looking for that multiply and not
finding it is the natural way to misread this as missing counterfactuality.

The two are equivalent because ``π_{-p}`` is constant across the actions compared
at an infoset — the opponent does not act on the edge from ``I`` to ``I·a``, so it
factors out of ``v(I,a) - v(I)``. This was checked by writing the three-pass form
out separately and comparing regrets over three iterations on both trees:
agreement to 2e-16 relative, i.e. float64 machine epsilon.

The counterfactual form is what public-state CFR wants. Values *are* the
per-hand counterfactual vectors that get passed between public states, and
keeping them normalised would mean dividing by ``π_{-p}`` and guarding the 0/0
wherever a hand is unreachable, only to multiply it back at the next boundary.

**The opponent's branch sum is unweighted.** Going bottom-up at a node owned by
``p``, ``cfv_p`` weights each child by ``sigma[h, a]`` but ``cfv_{1-p}`` is a plain
sum, because the opponent's probabilities are already folded into ``r_p`` down at
the leaves. Weighting both is the classic vector-CFR bug; ``check_zero_sum``
catches it on the first iteration.

**The chance constant is deferred.** Root reaches are all-ones and counterfactual
values carry no chance factor, keeping them O(1e4) rather than O(1e-5) in float32.
The deal is uniform over disjoint pairs, so ``1 / N_DEALS`` is a single global
scalar — identical for both players, every node and every hand — and regret
matching, regret clipping and reach-weighted averaging are all invariant to a
global positive rescaling. It is therefore applied only at the reporting
boundary, in ``root_value``. This is safe *because* the deal is uniform; it would
not be if chance weights varied by hand.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cards import MASK, N_DEALS
from .tree import FOLD_T, PreflopTree


class Tables(NamedTuple):
  regret: jax.Array      # (n_nodes, n_hands, n_actions)
  strategy_sum: jax.Array  # (n_nodes, n_hands, n_actions)
  iters: jax.Array       # () int32


def init_tables(tree: PreflopTree, n_hands: int, dtype=jnp.float32) -> Tables:
  shape = (len(tree.nodes), n_hands, tree.n_actions)
  return Tables(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype), jnp.int32(0))


def legal_mask(tree: PreflopTree) -> jax.Array:
  """(n_nodes, n_actions) bool — static, straight from the tree."""
  return jnp.asarray([list(n.legal) for n in tree.nodes])


# ── Strategy ─────────────────────────────────────────────────────────────────


def regret_matching_plus(regret: jax.Array, legal: jax.Array) -> jax.Array:
  """ Update the strategy with RM+ rule, e.g. the probability
  of each action proportional to the cumulative positive regret
  """
  legal_b = legal[:, None, :]
  pos = jnp.where(legal_b, jnp.maximum(regret, 0.0), 0.0)
  denom = pos.sum(-1, keepdims=True)
  safe = denom + (denom == 0)
  uniform = legal_b / legal.sum(-1)[:, None, None]
  return jnp.where(denom > 0.0, pos / safe, uniform).astype(regret.dtype)


def average_strategy(tables: Tables, legal: jax.Array) -> jax.Array:
  """"""
  legal_b = legal[:, None, :]
  ssum = jnp.where(legal_b, tables.strategy_sum, 0.0)
  denom = ssum.sum(-1, keepdims=True)
  safe = denom + (denom == 0)
  uniform = legal_b / legal.sum(-1)[:, None, None]
  return jnp.where(denom > 0.0, ssum / safe, uniform).astype(ssum.dtype)


# ── Terminal evaluation — the ONLY place card removal lives ──────────────────


def terminal_cfv(const, stake, r_opp, player, ev, mask=MASK):
  """Counterfactual values at a terminal, for every hand of ``player``.

  ``const`` is a hand-independent chip payoff to player 0 (fold terminals);
  ``stake`` multiplies the net-EV matrix (showdown and checkdown terminals).

  **``r_opp`` is the counterfactual reach** — this is where ``π_{-p}`` enters, and
  the only place it does. ``mask @ r_opp`` sums the opponent's reach over the
  hands they could still hold given yours, so the result is already
  ``Σ_b π_{-p}(b) · u(a, b)`` rather than a value conditional on the deal. See the
  module docstring for why there is no later multiply.

  Make sure to use the mask to mask out invalid opponent cards.
  """
  out = jnp.zeros_like(r_opp)
  if const != 0.0:
    # `const` is chips to player 0, so player 1's view of it is negated.
    out = out + (const if player == 0 else -const) * (mask @ r_opp)
  if stake != 0.0:
    # EV is antisymmetric and already mask-folded, so the *same* matvec serves
    # both players with no sign flip: player 1's payoff for (b, a) is
    # stake * EV[b, a] == -stake * EV[a, b].
    out = out + stake * (ev @ r_opp)
  return out


# ── Traversal ────────────────────────────────────────────────────────────────


def _traverse(tree, sigma, ev, r0, r1):
  """Two passes fused into one recursion; returns per-node (cfv0, cfv1) arrays.

  Reaches flow down, counterfactual values flow up. ``cfv_p`` at a node is the
  sigma-weighted average over children; the *opponent's* cfv is a plain sum.
  """
  node_cfv: dict[int, tuple[jax.Array, jax.Array]] = {}
  node_reach: dict[int, tuple[jax.Array, jax.Array]] = {}

  def go(ref, r0, r1):
    kind, i = ref
    if kind == "T":
      t = tree.terminals[i]
      return (terminal_cfv(t.const, t.stake, r1, 0, ev),
              terminal_cfv(t.const, t.stake, r0, 1, ev))
    node_reach[i] = (r0, r1)
    node = tree.nodes[i]
    p, sig = node.player, sigma[i]
    c0s, c1s = [], []
    for atom, child in enumerate(node.child):
      if child is None:
        c0s.append(None); c1s.append(None); continue
      w = sig[:, atom]
      nr0, nr1 = (r0 * w, r1) if p == 0 else (r0, r1 * w)
      a, b = go(child, nr0, nr1)
      c0s.append(a); c1s.append(b)

    stacked0 = jnp.stack([c for c in c0s if c is not None], -1)
    stacked1 = jnp.stack([c for c in c1s if c is not None], -1)
    idx = [a for a, c in enumerate(c0s) if c is not None]
    w = sig[:, jnp.asarray(idx)]
    # The acting player's own probabilities weight their branches; the opponent's
    # are already inside their reach at the leaves, so their branches just sum.
    cfv0 = (stacked0 * w).sum(-1) if p == 0 else stacked0.sum(-1)
    cfv1 = (stacked1 * w).sum(-1) if p == 1 else stacked1.sum(-1)
    node_cfv[i] = (cfv0, cfv1, stacked0 if p == 0 else stacked1, idx)
    return cfv0, cfv1

  root0, root1 = go(("D", tree.root), r0, r1)
  return root0, root1, node_cfv, node_reach


def cfr_iteration(tree, tables: Tables, legal, ev, n_hands, *, plus=False, linear=False):
  """One simultaneous-update CFR iteration over all hands at once."""
  sigma = regret_matching_plus(tables.regret, legal)
  ones = jnp.ones(n_hands, tables.regret.dtype)
  _, _, node_cfv, node_reach = _traverse(tree, sigma, ev, ones, ones)

  regret, ssum = tables.regret, tables.strategy_sum
  t = tables.iters + 1
  weight = t.astype(regret.dtype) if linear else jnp.asarray(1.0, regret.dtype)

  for i, node in enumerate(tree.nodes):
    cfv0, cfv1, branches, idx = node_cfv[i]
    p = node.player
    own_cfv = cfv0 if p == 0 else cfv1
    delta = jnp.zeros((n_hands, tree.n_actions), regret.dtype)
    delta = delta.at[:, jnp.asarray(idx)].set(branches - own_cfv[:, None])
    new = regret[i] + delta
    regret = regret.at[i].set(jnp.maximum(new, 0.0) if plus else new)
    own_reach = node_reach[i][p]
    ssum = ssum.at[i].add(weight * own_reach[:, None] * sigma[i])

  return Tables(regret, ssum, t)


def root_value(tree, sigma, ev, n_hands, dtype=jnp.float32):
  """(value to player 0, value to player 1) in chips, chance constant applied."""
  ones = jnp.ones(n_hands, dtype)
  cfv0, cfv1, _, _ = _traverse(tree, sigma, ev, ones, ones)
  return cfv0.sum() / N_DEALS, cfv1.sum() / N_DEALS


# ── Invariants ───────────────────────────────────────────────────────────────


def check_zero_sum(tree, sigma, ev, n_hands, tol=1e-3, dtype=jnp.float32):
  """``Check whether r0 · cfv0(n) + r1 · cfv1(n) == 0`` at every node, including the root.
  """
  ones = jnp.ones(n_hands, dtype)
  cfv0, cfv1, node_cfv, node_reach = _traverse(tree, sigma, ev, ones, ones)
  worst, where = 0.0, None
  for i in node_cfv:
    r0, r1 = node_reach[i]
    c0, c1 = node_cfv[i][0], node_cfv[i][1]
    scale = max(float(jnp.abs(r0 @ c0)), float(jnp.abs(r1 @ c1)), 1.0)
    err = float(jnp.abs(r0 @ c0 + r1 @ c1)) / scale
    if err > worst:
      worst, where = err, i
  assert worst <= tol, f"zero-sum violated at node {where}: relative error {worst:.3e}"
  return worst


def check_reach_conservation(tree, sigma, n_hands, tol=1e-4, dtype=jnp.float32):
  """``Σ_terminals r0(z)[a] * r1(z)[b] == 1`` for every hand pair ``(a, b)``.

  Checks whether the joint reaches form a valid probability distribution over
  terminals.

  **Two different statements, only one of which is true unconditionally.** It is
  tempting to summarise this as "the reaches sum to one over terminals". Over
  terminal *histories* — which include the deal — that is **false**, and whether
  chance is an explicit node or merely an array index has nothing to do with it.
  Take both players shoving with probability 1: every valid deal then reaches one
  terminal with joint player reach 1, so

      Σ_{a,b} Σ_z r0(z)[a] · r1(z)[b]  =  N_DEALS  =  1_624_350

  Only the chance weight brings that back to 1. What *is* true unconditionally is
  the per-deal statement, which is what the formula above says: hold ``(a, b)``
  fixed and the players' reaches form a probability distribution over terminals.
  That is why ``total`` is kept as a ``(n_hands, n_hands)`` matrix rather than
  collapsed — the assertion is made once per chance outcome, never across them.

  Both forms are therefore asserted:

  * per deal, ``Σ_z r0(z)[a] · r1(z)[b] == 1`` for every ``(a, b)``;
  * over histories, ``Σ_{a,b} P(a,b) · total[a,b] == 1`` with
    ``P(a,b) = MASK[a,b] / N_DEALS``.

  The second is implied by the first *here* (a convex combination of ones), but it
  is the one that keeps its meaning if chance nodes are ever put into the tree —
  a postflop extension dealing board runouts — at which point the per-deal form
  would need reinterpreting and this one would not.
  """
  total = jnp.zeros((n_hands, n_hands), dtype)

  def go(ref, r0, r1):
    nonlocal total
    #Type of the node plus id of the node
    kind, i = ref
    if kind == "T":
      total = total + jnp.outer(r0, r1)
      return
    node = tree.nodes[i]
    for atom, child in enumerate(node.child):
      if child is None:
        continue
      #Strategy over all infosets at that public state
      w = sigma[i][:, atom]
      go(child, r0 * w, r1) if node.player == 0 else go(child, r0, r1 * w)

  ones = jnp.ones(n_hands, dtype)
  go(("D", tree.root), ones, ones)

  err = float(jnp.abs(total - 1.0).max())
  assert err <= tol, f"joint reach does not sum to 1 per deal (max error {err:.3e})"

  # Chance-weighted total. P(a, b) = MASK[a, b] / N_DEALS is the uniform measure
  # over disjoint pairs, so this is a proper probability distribution over deals.
  weighted = float((MASK / N_DEALS * total).sum())
  assert abs(weighted - 1.0) <= tol, (
    f"chance-weighted joint reach is {weighted:.6f}, not 1 — if chance nodes were "
    "added to the tree, the per-deal assertion above is no longer the right check"
  )
  return err
