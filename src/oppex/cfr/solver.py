"""Vectorised counterfactual regret minimisation over the preflop public tree.

One traversal updates all 1326 information sets per node at once, and it is
batched twice over. Across **hands**: reaches and counterfactual values are
``(n_hands,)`` vectors and terminal evaluation is a matmul against the
card-removal mask. Across **nodes**: every node at the same depth does the
identical thing, so each pass is one op per *level* rather than per node, and the
update loop over nodes is gone entirely — ``TreeLayout`` in ``tree.py`` holds the
static gathers that move values between levels.

That second axis is why the earlier recursive traversal was replaced. It unrolled
one set of ops per node at trace time, so compile time grew with the tree: 73 s
for 472 nodes, extrapolating to ~26 min at 10k. Per level instead, the graph is
O(depth) — under ten levels even for the 472-node tree — and compile is flat at
~1.5 s across the whole benchmark grid. The cost is a fixed overhead that small
trees do not earn back; the 2-node push-fold tree runs ~4x slower per iteration
than it did. ``jit=False`` still runs the identical code eagerly.

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
import numpy as np

from .cards import MASK, N_DEALS
from .tree import FOLD_T, PreflopTree, TreeLayout, build_layout


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


def _terminal_kinds(layout: TreeLayout):
  """Split terminals into the two matrices they contract against.

  A fold terminal has a payoff constant and no stake; a showdown or checkdown has
  a stake and no constant. The tree builder guarantees the split is exact, which
  is what lets the batched contraction concatenate instead of accumulate.
  """
  fold, show = np.flatnonzero(layout.const), np.flatnonzero(layout.stake)
  if fold.size + show.size != layout.n_terminals:
    raise AssertionError(
      "every terminal must have exactly one of const/stake non-zero; got "
      f"{fold.size} fold + {show.size} showdown for {layout.n_terminals} terminals"
    )
  perm = np.concatenate([fold, show])
  inv = np.empty(perm.size, np.int32)
  inv[perm] = np.arange(perm.size, dtype=np.int32)
  return fold, show, inv


def terminal_cfv(layout: TreeLayout, ev, r_opp, player, mask=MASK):
  """Counterfactual values at **every terminal at once**, per hand of ``player``.

  ``r_opp`` is ``(n_terminals, n_hands)`` in the layout's discovery order: row
  ``j`` is the *opponent's* counterfactual reach at terminal ``j``. Returns the
  same shape.

  **``r_opp`` is the counterfactual reach** — this is where ``π_{-p}`` enters, and
  the only place it does. Contracting against ``MASK`` sums the opponent's reach
  over the hands they could still hold given yours, so the result is already
  ``Σ_b π_{-p}(b) · u(a, b)`` rather than a value conditional on the deal. See the
  module docstring for why there is no later multiply. This function is the one
  place card removal lives, deliberately, so that no terminal can forget it.

  **Why this batches rather than taking one terminal at a time.** Every fold
  terminal multiplies by the same ``MASK`` and every showdown by the same ``EV``;
  only the reach row differs. One matvec per terminal makes that a stream of
  GEMVs, each re-reading a 7 MB matrix to do 1.8 MFLOP — memory-bound. Stacking
  the rows turns each matrix into a single GEMM, worth 1.8-2.6x per iteration on
  trees from 99 to 708 terminals and dropping per-terminal cost from ~0.16 ms to
  ~0.07 ms. An isolated matmul microbenchmark says 13.8x; do not believe that in
  context, because stacking reaches in and gathering values out is new work that
  scales with terminal count too.

  The compile-time half of that trade-off no longer applies: batching terminals
  used to *add* graph (73.5 s at 472 nodes against 32.7 s per-terminal), but
  since the traversal became level-batched, compile is flat at ~1.5 s and the
  terminal contraction is no longer a meaningful share of it.

  **Both matrices are transposed, and only one of them is free.** Rows here are
  terminals, so the contraction is ``r_opp @ M.T`` rather than ``M @ r_opp``.
  ``MASK`` is symmetric so its transpose is cosmetic, but ``EV`` is *anti*
  symmetric — dropping the ``.T`` there silently flips the sign of every showdown
  payoff, which ``check_zero_sum`` catches but only after the fact.

  The two kinds are gathered apart rather than run as one full-width matmul
  because ``const`` and ``stake`` are structurally disjoint — a fold terminal has
  no stake, a showdown no const — so a full-width pair of matmuls would spend
  half its FLOPs multiplying by zero. They come back concatenated by kind and one
  gather restores discovery order.
  """
  fold, show, inv = _terminal_kinds(layout)
  rows = []
  if fold.size:
    # `const` is chips to player 0, so player 1's view of it is negated.
    sign = 1.0 if player == 0 else -1.0
    rows.append((sign * layout.const[fold])[:, None] * (r_opp[fold] @ mask.T))
  if show.size:
    # EV is antisymmetric and already mask-folded, so the *same* matmul serves
    # both players with no sign flip: player 1's payoff for (b, a) is
    # stake * EV[b, a] == -stake * EV[a, b].
    rows.append(layout.stake[show][:, None] * (r_opp[show] @ ev.T))
  out = rows[0] if len(rows) == 1 else jnp.concatenate(rows, 0)
  return out[inv]


# ── Traversal ────────────────────────────────────────────────────────────────


class Traversal(NamedTuple):
  """Per-node reaches and counterfactual values, all in the tree's node order."""

  cfv0: jax.Array     # (n_nodes, n_hands)
  cfv1: jax.Array
  reach0: jax.Array   # (n_nodes, n_hands)
  reach1: jax.Array
  branch: jax.Array   # (n_nodes, n_hands, n_actions) — acting player's children
  term0: jax.Array    # (n_terminals, n_hands) reach at terminals, layout order
  term1: jax.Array


def _traverse(tree, sigma, ev, r0, r1) -> Traversal:
  """Reaches down, all terminals at once, counterfactual values up — by depth.

  Every node at a given depth does the identical thing, so each pass is one
  batched op per *level* rather than per node. That is the whole point: the
  graph handed to XLA is O(depth) instead of O(nodes), and depth is under ten
  even for a 472-node tree. There is no ``lax.scan`` here because there is
  nothing to gain from one — depth is a handful, and scanning would force every
  level to be padded to the widest, which is pure waste when level sizes run
  1, 5, 20, 76, 162, 148, 53, 7.

  Going up, the acting player's cfv is a sigma-weighted sum over the action axis
  and the opponent's is a plain sum, because the opponent's probabilities are
  already inside their reach. Illegal edges gather the layout's zero row, so
  they contribute nothing to either and need no masking.
  """
  L = build_layout(tree)
  n_hands, A, dtype = r0.shape[0], L.n_actions, r0.dtype
  sig = sigma[L.order]                                    # level-major

  # ── down: reaches ──────────────────────────────────────────────────────────
  reach0, reach1 = [r0[None, :]], [r1[None, :]]
  term0, term1 = [], []
  for d, (lo, hi) in enumerate(L.level):
    k = hi - lo
    w = jnp.swapaxes(sig[lo:hi], 1, 2)                    # (k, A, n_hands)
    p0 = jnp.asarray(L.player[lo:hi] == 0)[:, None, None]
    c0 = jnp.where(p0, reach0[d][:, None, :] * w, reach0[d][:, None, :])
    c1 = jnp.where(p0, reach1[d][:, None, :], reach1[d][:, None, :] * w)
    f0, f1 = c0.reshape(k * A, n_hands), c1.reshape(k * A, n_hands)
    if d + 1 < len(L.level):
      reach0.append(f0[L.down_node[d]])
      reach1.append(f1[L.down_node[d]])
    term0.append(f0[L.down_term[d]])
    term1.append(f1[L.down_term[d]])

  # ── terminals: two GEMMs, opponent reach against own payoff ────────────────
  tr0, tr1 = jnp.concatenate(term0, 0), jnp.concatenate(term1, 0)
  cfv0_t = terminal_cfv(L, ev, tr1, 0)
  cfv1_t = terminal_cfv(L, ev, tr0, 1)

  # ── up: counterfactual values ──────────────────────────────────────────────
  zero = jnp.zeros((1, n_hands), dtype)
  cfv0: list = [None] * len(L.level)
  cfv1: list = [None] * len(L.level)
  branch: list = [None] * len(L.level)
  for d in reversed(range(len(L.level))):
    lo, hi = L.level[d]
    k = hi - lo
    nxt0 = cfv0[d + 1] if d + 1 < len(L.level) else jnp.zeros((0, n_hands), dtype)
    nxt1 = cfv1[d + 1] if d + 1 < len(L.level) else jnp.zeros((0, n_hands), dtype)
    ch0 = jnp.concatenate([nxt0, cfv0_t, zero], 0)[L.up_src[d]].reshape(k, A, n_hands)
    ch1 = jnp.concatenate([nxt1, cfv1_t, zero], 0)[L.up_src[d]].reshape(k, A, n_hands)
    w = jnp.swapaxes(sig[lo:hi], 1, 2)                    # (k, A, n_hands)
    p0 = jnp.asarray(L.player[lo:hi] == 0)[:, None]
    # The acting player's own probabilities weight their branches; the opponent's
    # are already inside their reach at the leaves, so their branches just sum.
    cfv0[d] = jnp.where(p0, (ch0 * w).sum(1), ch0.sum(1))
    cfv1[d] = jnp.where(p0, ch1.sum(1), (ch1 * w).sum(1))
    branch[d] = jnp.swapaxes(jnp.where(p0[:, :, None], ch0, ch1), 1, 2)

  back = L.inv_order      # level-major -> the tree's own node order
  return Traversal(
    cfv0=jnp.concatenate(cfv0, 0)[back],
    cfv1=jnp.concatenate(cfv1, 0)[back],
    reach0=jnp.concatenate(reach0, 0)[back],
    reach1=jnp.concatenate(reach1, 0)[back],
    branch=jnp.concatenate(branch, 0)[back],
    term0=tr0, term1=tr1,
  )


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
  tv = _traverse(tree, sigma, ev, ones, ones)

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

  # Every node updates the same way, so there is no loop: `is_p0` picks the
  # acting player's row out of the traversal and the rest is elementwise.
  is_p0 = jnp.asarray([n.player == 0 for n in tree.nodes])[:, None]
  own_cfv = jnp.where(is_p0, tv.cfv0, tv.cfv1)            # (n_nodes, n_hands)
  own_reach = jnp.where(is_p0, tv.reach0, tv.reach1)

  # Illegal actions carry the layout's zero branch value; masking here keeps
  # their regret at exactly 0 rather than at -cfv.
  delta = jnp.where(legal[:, None, :], tv.branch - own_cfv[:, :, None], 0.0)
  regret = tables.regret + delta
  if plus:
    regret = jnp.maximum(regret, 0.0)
  elif discount is not None:
    regret = jnp.where(regret > 0.0, regret * pos_d, regret * neg_d)

  ssum = tables.strategy_sum + weight * own_reach[:, :, None] * sigma
  if discount is not None:
    ssum = ssum * ssum_d

  if update_player != -1:
    # Selecting *after* the transforms, not before: a node the other player owns
    # must keep its accumulators untouched, not merely un-incremented — a
    # discount applied twice per full iteration would be a silent bug.
    mine = jnp.asarray([n.player == update_player for n in tree.nodes])[:, None, None]
    regret = jnp.where(mine, regret, tables.regret)
    ssum = jnp.where(mine, ssum, tables.strategy_sum)

  return Tables(regret, ssum, t if advance else tables.iters)


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
  tv = _traverse(tree, sigma, ev, ones, ones)
  return tv.cfv0[tree.root].sum() / N_DEALS, tv.cfv1[tree.root].sum() / N_DEALS


# ── Invariants ───────────────────────────────────────────────────────────────


def check_zero_sum(tree, sigma, ev, n_hands, tol=1e-3, dtype=jnp.float32):
  """``Check whether r0 · cfv0(n) + r1 · cfv1(n) == 0`` at every node, including the root.
  """
  ones = jnp.ones(n_hands, dtype)
  tv = _traverse(tree, sigma, ev, ones, ones)
  v0 = (tv.reach0 * tv.cfv0).sum(-1)                      # (n_nodes,)
  v1 = (tv.reach1 * tv.cfv1).sum(-1)
  scale = jnp.maximum(jnp.maximum(jnp.abs(v0), jnp.abs(v1)), 1.0)
  err = jnp.abs(v0 + v1) / scale
  where = int(jnp.argmax(err))
  worst = float(err[where])
  assert worst <= tol, f"zero-sum violated at node {where}: relative error {worst:.3e}"
  return worst


def ev_zero(n_hands, dtype):
  """A zero EV matrix — reach conservation is a property of sigma alone."""
  return jnp.zeros((n_hands, n_hands), dtype)


def check_reach_conservation(tree, sigma, n_hands, tol=1e-4, dtype=jnp.float32):
  """``Σ_terminals r0(z)[a] * r1(z)[b]  * r_c(z) [a,b] == 1`` for every hand pair ``(a, b)``.

  Checks whether the joint reaches form a valid probability distribution over
  terminals.
  """
  ones = jnp.ones(n_hands, dtype)
  tv = _traverse(tree, sigma, ev_zero(n_hands, dtype), ones, ones)
  # Σ_z r0(z) ⊗ r1(z) over terminals is exactly one matmul over the terminal
  # axis — the same batching the traversal uses, for the same reason.
  total = tv.term0.T @ tv.term1

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
