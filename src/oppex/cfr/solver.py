"""Vectorised counterfactual regret minimisation over the preflop public tree.

One traversal updates all 1326 information sets per node at once: reaches and
counterfactual values are ``(n_hands,)`` vectors, and terminal evaluation is a
matrix-vector product against the card-removal mask. The tree has a handful of
nodes, so traversal is plain recursive Python over a static structure — under
``jit`` the recursion unrolls at trace time into one fused graph, and with
``jit=False`` the identical code is eager and steppable in a debugger.

``cfr_step`` is the entry point: one call is one full iteration, under either the
simultaneous or the alternating update schedule. ``cfr_iteration`` underneath it
is one *traversal*, which is a half-step when alternating — call it directly only
to update a single player, as in fixed-opponent exploitation.

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

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .cards import MASK, N_DEALS
from .tree import FOLD_T, PreflopTree


class Tables(NamedTuple):
  regret: jax.Array      # (n_nodes, n_hands, n_actions)
  strategy_sum: jax.Array  # (n_nodes, n_hands, n_actions)
  iters: jax.Array       # () int32


class Discount(NamedTuple):
  """Discounted-CFR coefficients (Brown & Sandholm 2019).

  At the end of iteration ``t`` (1-indexed, so ``iters + 1``) the accumulators are
  rescaled: positive cumulative regret by ``t^α / (t^α + 1)``, negative cumulative
  regret by ``t^β / (t^β + 1)``, and the cumulative strategy by ``(t / (t+1))^γ``.

  **The instantaneous regret is added first, then the product is discounted.**
  That ordering is not cosmetic — it is the one that makes ``α = β = γ = 1``
  reproduce Linear CFR exactly. Unrolling ``R^t = (R^{t-1} + r^t) · t/(t+1)``
  gives ``R^T = (Σ_t t · r^t) / (T+1)``, i.e. weights proportional to ``t``.
  Discounting before the add gives weights proportional to ``t + 1`` instead —
  the same asymptotics, a different algorithm.

  The defaults are the paper's recommended setting. ``β = 0`` is not "no
  discounting on negatives": ``t^0 = 1``, so negative regret is *halved* every
  iteration, which is the point — it lets a badly-regretted action come back into
  play far faster than vanilla CFR while still not being floored outright the way
  ``plus`` floors it.
  """
  alpha: float = 1.5   # positive cumulative regret
  beta: float = 0.0    # negative cumulative regret
  gamma: float = 2.0   # cumulative strategy


DCFR = Discount()


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


class _Static:
  """Identity-hashable box, so a pytree-shaped value can be a static ``jit`` arg.

  ``PreflopTree`` is a ``NamedTuple`` holding ``BettingState`` arrays, so it is
  neither hashable (arrays aren't) nor usable as a traced argument (its nodes are
  Python structure that traversal must branch on). Boxing it defers both problems
  to object identity: two calls share a trace iff they were handed the same tree
  object. Trees are built once and reused, so that is exactly the caching wanted —
  rebuilding a structurally identical tree merely re-traces, it never silently
  reuses a stale graph.
  """

  __slots__ = ("value",)

  def __init__(self, value: Any):
    self.value = value

  def __hash__(self) -> int:
    return id(self.value)

  def __eq__(self, other: object) -> bool:
    return isinstance(other, _Static) and other.value is self.value


def _cfr_iteration(
  tree, tables: Tables, legal, ev, n_hands, update_player, advance, plus, linear,
  discount,
) -> Tables:
  """One traversal, writing back the tables of ``update_player`` (``-1`` = both).

  Every table is rebuilt rather than mutated: per-node rows are collected in
  Python lists and stacked once at the end. The functional form is what makes the
  whole iteration a single jittable expression — a chain of ``regret.at[i].set``
  would also be functional, but it threads ``n_nodes`` sequential scatters through
  the graph for rows that are all computed independently.

  ``sigma`` is always built for *both* players — the traversal needs the
  opponent's current strategy to reach the leaves at all. ``update_player`` gates
  only the write-back, and it gates the strategy sum as well as the regrets:
  under alternating updates each player's average must accumulate once per full
  iteration, not once per traversal.
  """
  sigma = regret_matching_plus(tables.regret, legal)
  ones = jnp.ones(n_hands, tables.regret.dtype)
  _, _, node_cfv, node_reach = _traverse(tree, sigma, ev, ones, ones)

  dtype = tables.regret.dtype
  t = tables.iters + 1
  tf = t.astype(dtype)
  if discount is None:
    weight = tf if linear else jnp.asarray(1.0, dtype)
  else:
    # Discounting the accumulator subsumes weighting the contribution, so the
    # contribution goes in unweighted; see Discount for why the add comes first.
    weight = jnp.asarray(1.0, dtype)
    ta, tb = tf ** discount.alpha, tf ** discount.beta
    pos_d, neg_d = ta / (ta + 1.0), tb / (tb + 1.0)
    ssum_d = (tf / (tf + 1.0)) ** discount.gamma

  regret_rows, ssum_rows = [], []
  for i, node in enumerate(tree.nodes):
    p = node.player
    if update_player != -1 and p != update_player:
      regret_rows.append(tables.regret[i])
      ssum_rows.append(tables.strategy_sum[i])
      continue
    cfv0, cfv1, branches, idx = node_cfv[i]
    own_cfv = cfv0 if p == 0 else cfv1
    delta = jnp.zeros((n_hands, tree.n_actions), dtype)
    delta = delta.at[:, jnp.asarray(idx)].set(branches - own_cfv[:, None])
    new = tables.regret[i] + delta
    if plus:
      new = jnp.maximum(new, 0.0)
    elif discount is not None:
      new = jnp.where(new > 0.0, new * pos_d, new * neg_d)
    regret_rows.append(new)

    own_reach = node_reach[i][p]
    ssum = tables.strategy_sum[i] + weight * own_reach[:, None] * sigma[i]
    ssum_rows.append(ssum if discount is None else ssum * ssum_d)

  return Tables(jnp.stack(regret_rows), jnp.stack(ssum_rows), t if advance else tables.iters)


@partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9))
def _cfr_iteration_jit(
  tree_s: _Static, tables, legal, ev, n_hands, update_player, advance, plus, linear,
  discount,
):
  return _cfr_iteration(
    tree_s.value, tables, legal, ev, n_hands, update_player, advance, plus, linear,
    discount,
  )


def _check_scheme(plus, linear, discount):
  """``discount`` supersedes ``plus``/``linear``; combining them double-weights."""
  if discount is None:
    return
  clash = [n for n, v in (("plus", plus), ("linear", linear)) if v]
  if clash:
    raise ValueError(
      f"discount= cannot be combined with {'/'.join(clash)}= — alpha/beta/gamma "
      "already control regret and average weighting, so the two compose into a "
      "scheme that is neither. Use beta<0 for the DCFR analogue of plus, and "
      "gamma=1 for the analogue of linear."
    )


def cfr_iteration(
  tree, tables: Tables, legal, ev, n_hands, *, update_player=-1, advance=True,
  plus=False, linear=False, discount: Discount | None = None, jit=True,
) -> Tables:
  """One CFR traversal over all hands at once, updating ``update_player``.

  ``update_player`` is ``-1`` for the simultaneous update (both players written
  back from the same traversal, the default) or a seat index for one leg of an
  alternating update. ``advance=False`` performs the update but leaves
  ``iters`` alone, so that both legs of an alternating step share one step index
  and therefore one ``linear`` weight; ``cfr_step`` is what pairs them up.

  Everything the traversal branches on is static — the tree shape, ``n_hands``,
  and the four algorithm flags — so the recursion unrolls at trace time into one
  fused graph over ``(regret, strategy_sum, iters)``, ``legal`` and ``ev``. Pass
  ``jit=False`` to run the identical code eagerly, which is what to do when
  stepping through a traversal in a debugger or printing intermediate cfvs.
  """
  if update_player not in (-1, 0, 1):
    raise ValueError(f"update_player must be -1, 0 or 1; got {update_player}")
  _check_scheme(plus, linear, discount)
  args = (tables, legal, ev, int(n_hands), int(update_player), bool(advance),
          bool(plus), bool(linear), discount)
  if not jit:
    return _cfr_iteration(tree, *args)
  return _cfr_iteration_jit(_Static(tree), *args)


def cfr_step(
  tree, tables: Tables, legal, ev, n_hands, *, alternating=False,
  plus=False, linear=False, discount: Discount | None = None, jit=True,
) -> Tables:
  """One full CFR iteration: every player updated exactly once.

  ``alternating=False`` is the simultaneous update — a single traversal whose
  regrets for both players are read off the *same* strategy profile.

  ``alternating=True`` splits the step into two traversals, player 0 then player
  1. The second one is the point of the variant: it recomputes ``sigma`` from
  regrets that already include player 0's update, so player 1 responds to where
  the opponent has just moved rather than to where they were. Both legs share one
  step index, so ``iters`` counts full iterations under either scheme and
  ``linear`` weights mean the same thing in both — otherwise player 0's average
  would carry odd weights and player 1's even ones.

  ``alternating=True, plus=True, linear=True`` is CFR+;
  ``alternating=True, discount=DCFR`` is Discounted CFR. Both legs of an
  alternating step share one ``t``, so each player's accumulators are discounted
  exactly once per full iteration under either update schedule.
  """
  _check_scheme(plus, linear, discount)
  common = dict(plus=plus, linear=linear, discount=discount, jit=jit)
  if not alternating:
    return cfr_iteration(tree, tables, legal, ev, n_hands, update_player=-1, **common)
  tables = cfr_iteration(
    tree, tables, legal, ev, n_hands, update_player=0, advance=False, **common
  )
  return cfr_iteration(
    tree, tables, legal, ev, n_hands, update_player=1, advance=True, **common
  )


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
  """``Σ_terminals r0(z)[a] * r1(z)[b]  * r_c(z) [a,b] == 1`` for every hand pair ``(a, b)``.

  Checks whether the joint reaches form a valid probability distribution over
  terminals.
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
