"""AIVAT for the re-solving player, with per-history values from its subgames.

The estimator is the one of ``slumbot/aivat.py`` and ``notes/aivat.md``: a
hand's value is ``u(z) + Σ term(n)`` over corrected nodes, with
``term(n) = Σ_a σ(n, a)·ṽ(n, a) − ṽ(n, a*)``. Each term has zero mean given
the history whenever ``σ`` is the distribution ``a*`` was drawn from and ``ṽ``
is any function of the history up to ``n``.

**What is new is ``ṽ``.** The Q-head gives our *infoset* value, an average over
the opponent's hands. A re-solved subgame holds more: before it is contracted
against the opponent's reach, every terminal pays per opponent hand ``h``. For
the opponent's actual hand ``h*`` the per-history values are

    v(z, h*)  = payoff(z, h*)                             fold, showdown
    v(l, h*)  = Σ_t τ(l, t)·V(l, t, h*)                    matrix leaf
    v(n, h*)  = Σ_a σ(n, a)·v(c_a, h*)                     our node
    v(n, h*)  = Σ_a π(a | n, h*)·v(c_a, h*)                opponent node

and ``Q(n, a, h*) = v(c_a, h*)``. The opponent's hand no longer adds noise to
the value of our actions. Since the opponent's policy ``π`` is known, its
decisions get terms as well.

**Recorded during play, summed after it.** ``ṽ(n, ·)`` must be fixed before
``a*`` is drawn. So ``Q(n, ·, h*)`` is computed when the decision is reached,
from the subgame it lies in, and stored with the distribution ``a*`` was drawn
from; the terms are summed when the hand ends. The evaluator knows ``h*``
during play, since the env deals it, but nothing computed here reaches the
player.

**Leaf values for one opponent hand.** The solve's rollouts draw ``h`` from
``r_o``, so they estimate only the reach-weighted mean over ``h``.
``V(l, t, h*)`` is rolled out again with the opponent's hand fixed to ``h*``,
with seeds of its own. Only leaves that the current decision reaches with path
weight at least ``min_weight`` are rolled out (our ``σ`` times ``π(· | h*)``),
only for continuations with ``τ ≥ min_tau``, and the values are cached for the
later decisions of the same solve. Other leaves keep the solve's
belief-averaged value. This trades accuracy, never bias. A solve with
per-hand rollouts (``Resolver(games_per_hand=...)``) already holds
``V(l, t, h*)`` in its table, the values it was solved with, so only leaves
missing from it are rolled out. Choosing the leaves
from *later* decisions would bias the term: only the played branch has later
decisions, so ``Q(n, a*)`` would depend on ``a*``.

**Coverage.** Our decisions always lie in the subgame we played from. An
opponent decision is corrected in the latest subgame that contains it, e.g.
the opponent's reply inside our depth-4 subgame. Many opponent decisions are in
no subgame: the first action of a street when they act first, or the small
blind's opening when we are the big blind. With ``cover_opponent`` an extra
solve rooted at that decision is made when it is reached, for evaluation only.
It needs only the opponent's action and our reply, so it uses
``eval_depth=2``. ``Resolver.solve`` with an explicit seed leaves the
resolver's state and seeds alone.

**Chance** uses the checkdown terms of ``slumbot/aivat.py`` (deal, board cards,
the runout of an early all-in), with exact equity against ``h*``.

**Boards valued by rollouts** (``board_q``, recorded next to ``board``, not in
``TERMS``). Checkdown values a card at the stake in the pot when it arrives
and ignores the betting after it, while the decision before the card already
used the leaf value ``Σ_t τ·V(l*, t, h*)``, which includes that betting. So
the luck of a card that sets up a later stack-off stays in the estimate. The
leaf ``l*`` where the street closed is the chance node: its rollouts deal the
next card. With ``h*`` fixed, ``card_games`` rollouts per continuation give
the mean over cards ``V(l*, t, h*)`` and, with the dealt card(s) fixed as
well, ``V(l*, t, h*, c*)``; the term is ``Σ_t τ_t·(V(l*, t, h*) − V(l*, t,
h*, c*))``. Both use one seed, fixed before ``c*`` is seen, and the same deck
law as the env (uniform over cards neither hand nor the board holds), so the
term has zero mean. Sharing the seed gives common random numbers. A
transition with no leaf (an uncovered opponent decision) keeps the checkdown
term.
"""

from __future__ import annotations

import time

import numpy as np

from ..cfr.cards import HAND_INDEX, N_HANDS
from .resolving import Resolver, Solution, _fold
from .slumbot.aivat import ExactEquity

TERMS = ("deal", "board", "runout", "ours", "opponent")
_STREET = {0: 0, 3: 1, 4: 2, 5: 3}


def leaf_weights(sol: Solution, seat: int, h: int, nodes) -> np.ndarray:
  """``(n_leaves,)`` largest path weight from a child of any of ``nodes``.

  The weight of a path is the product of our ``σ`` and the opponent's
  ``π(· | h)`` along it; a leaf's per-history value enters ``Q`` at ``nodes``
  with that weight.
  """
  tree = sol.game.tree
  leaf_id = {j: k for k, j in enumerate(tree.leaves)}
  out = np.zeros(len(tree.leaves))

  def walk(ref, w):
    kind, i = ref
    if kind == "T":
      if i in leaf_id:
        out[leaf_id[i]] = max(out[leaf_id[i]], w)
      return
    node = tree.nodes[i]
    p = sol.sigma[i] if node.player == seat else sol.pi[i][h]
    for a, ch in enumerate(node.child):
      if ch is not None and p[a] > 0:
        walk(ch, w * p[a])

  for i in nodes:
    for ch in tree.nodes[i].child:
      if ch is not None:
        walk(ch, 1.0)
  return out


def history_values(sol: Solution, seat: int, h: int, leaf_v) -> tuple[np.ndarray, np.ndarray]:
  """Per-history ``(v (n_nodes,), Q (n_nodes, A))`` for opponent hand ``h``.

  ``leaf_v`` is ``(n_leaves,)``, each leaf's value for ``h`` under our
  continuation strategy. Children have larger indices than their parent, so
  one reversed sweep is bottom-up.
  """
  tree = sol.game.tree
  term = sol.payoff[:, h].copy()
  if len(tree.leaves):
    term[list(tree.leaves)] = leaf_v
  v = np.zeros(len(tree.nodes))
  q = np.zeros((len(tree.nodes), tree.n_actions))
  for i in reversed(range(len(tree.nodes))):
    node = tree.nodes[i]
    for a, ch in enumerate(node.child):
      if ch is not None:
        q[i, a] = v[ch[1]] if ch[0] == "D" else term[ch[1]]
    w = sol.sigma[i] if node.player == seat else sol.pi[i][h]
    v[i] = float(w @ q[i])
  return v, q


class ResolvingAivat:
  """Records one hand of a ``Resolver`` and turns it into AIVAT terms.

  Per hand: ``new_hand`` after ``resolver.new_hand``, ``step`` at every
  decision in order, then ``finish``. Values are env chips to us.
  """

  def __init__(self, resolver: Resolver, cover_opponent: bool = True,
               eval_depth: int = 2, min_weight: float = 1e-2, min_tau: float = 1e-2,
               seed: int = 1_000_000, card_games: int | None = 1024) -> None:
    self.res = resolver
    self.env = resolver.bp.env
    self.cover_opponent, self.eval_depth = cover_opponent, eval_depth
    self.min_weight, self.min_tau = min_weight, min_tau
    self.seed = seed
    self.eq = ExactEquity()
    # Counts evaluation solves and targeted rollouts over all hands, so that
    # every call gets a seed of its own.
    self.n_rollouts = 0
    # Rollouts for boards valued by rollouts: one leaf at a time, more games
    # than the solve's. They must take a board longer than the leaf's street;
    # rollouts that do not say so (no resized) keep the checkdown term.
    self.card_rollouts = None
    if card_games:
      try:
        self.card_rollouts = resolver.rollouts.resized(card_games, 1)
      except (AttributeError, NotImplementedError):
        pass

  def new_hand(self, opp_cards) -> None:
    """Starts a hand against ``opp_cards``, which only the evaluator sees."""
    self.opp_cards = [int(c) for c in opp_cards]
    self.h = int(HAND_INDEX[self.opp_cards[0], self.opp_cards[1]])
    self.terms = dict.fromkeys(TERMS, 0.0)
    self.alt = {"board_q": 0.0}
    self.info = {"opponent_steps": 0, "opponent_covered": 0, "extra_solves": 0,
                 "targeted_leaves": 0, "board_q_fallback": 0}
    self.nodes: list[dict] = []
    self.seconds = 0.0
    self._extra: list[Solution] = []
    self._leaf_v: dict[int, dict[int, float]] = {}
    self._street, self._board = 0, []
    # (solve, leaf position) the last decision's action entered, or None.
    self._leaf = None

    t0 = time.time()
    env = self.env
    eq = self._eq([])
    term = (env.small_blind + env.big_blind) * (0.5 - eq)
    self._add({"kind": "deal", "eq": eq}, term)
    self.seconds += time.time() - t0

  def step(self, actor: int, history, board, committed, probs, atom: int) -> None:
    """Records one decision, its board card if one was just dealt, and its term.

    ``probs`` is the env-atom distribution ``atom`` was drawn from and
    ``committed`` the chips in before the decision. For our decisions call it
    after ``resolver.probs``, so that ``resolver.solution`` is the subgame we
    played from. ``Q`` is computed from the history alone; ``atom`` enters
    only the term.
    """
    t0 = time.time()
    seat = self.res.seat

    # A board dealt since the last decision is a chance node, valued by
    # checkdown at the stake in the pot when it arrives.
    street = _STREET[len(board)]
    if street > self._street:
      before, after = self._eq(self._board), self._eq(board)
      self._add({"kind": "board", "street": street, "stake": sum(committed) / 2,
                 "eq_before": before, "eq_after": after},
                sum(committed) * (before - after))
      self._board_q(board, sum(committed) * (before - after))
      self._street = street
    self._board = [int(c) for c in board]

    hist = tuple(int(a) for a in history)
    self._leaf = None
    if actor == seat:
      sol = self.res.solution
    else:
      self.info["opponent_steps"] += 1
      sol = self._covering(hist, board)
      if sol is None:
        self.seconds += time.time() - t0
        return
      self.info["opponent_covered"] += 1
    i = sol.index[hist]
    node = sol.game.tree.nodes[i]
    q = self._q(sol, i)
    p = _fold(np.asarray(probs, np.float64), node.alias)
    p = p / p.sum()
    a = node.alias[atom]
    if a < 0:
      raise AssertionError(f"sampled atom {atom} is illegal at {node.path}")
    ch = node.child[a]
    leaves = sol.game.tree.leaves
    if ch is not None and ch[0] == "T" and ch[1] in leaves:
      self._leaf = (sol, leaves.index(ch[1]))
    kind = "ours" if actor == seat else "opponent"
    self._add({"kind": kind, "history": list(hist), "p": p.tolist(), "q": q.tolist(),
               "action": int(a)}, float(p @ q - q[a]))
    self.seconds += time.time() - t0

  def finish(self, payoff: float, ended_by_fold: bool, final_committed) -> dict:
    """The hand's summed terms, counts, per-node records and seconds spent.

    ``payoff`` is the observed ``u(z)``.
    """
    t0 = time.time()
    # A called all-in before the river: the rest of the board is one chance
    # node, whose checkdown value is the realised payoff.
    if not ended_by_fold and self._street < 3:
      each = sum(final_committed) / 2
      eq = self._eq(self._board)
      self._add({"kind": "runout", "street": self._street, "stake": each, "eq": eq},
                (2 * eq - 1) * each - payoff)
    self.seconds += time.time() - t0
    return {**self.terms, **self.alt, **self.info, "nodes": self.nodes,
            "seconds": self.seconds}

  def _board_q(self, board, checkdown: float) -> None:
    # The rollout-valued term of the board just dealt, at the leaf the last
    # decision entered; without one, the checkdown term stands in.
    if self._leaf is None or self.card_rollouts is None:
      self.info["board_q_fallback"] += 1
      self.alt["board_q"] += checkdown
      self.nodes.append({"kind": "board_q", "fallback": True, "term": checkdown})
      return
    sol, k = self._leaf
    res, tree = self.res, sol.game.tree
    r = np.zeros((len(tree.terminals), N_HANDS))
    r[tree.leaves[k], self.h] = 1.0
    tau = np.where(sol.tau[k] >= self.min_tau, sol.tau[k], 0.0)
    tau = tau / tau.sum()
    t_mask = np.zeros(sol.tau.shape, bool)
    t_mask[k] = tau > 0
    # One seed for both: common random numbers, chosen before c* matters.
    seed = self.seed + 900_000 + self.n_rollouts
    self.n_rollouts += 1
    # The leaf's own board leaves the card to the rollouts; the new board
    # fixes it.
    v = [float(self.card_rollouts.values(tree, r, res.cards, b, res.seat, seed=seed,
                                         t_mask=t_mask)[0][k] @ tau)
         for b in (list(sol.board), [int(c) for c in board])]
    term = v[0] - v[1]
    self.alt["board_q"] += term
    self.nodes.append({"kind": "board_q", "leaf": k, "v_mean": v[0], "v_card": v[1],
                       "term": term})

  def _eq(self, board) -> float:
    # Exact equity of our hand against the opponent's on this board.
    return self.eq(list(self.res.cards), self.opp_cards, [int(c) for c in board])

  def _add(self, record: dict, term: float) -> None:
    self.terms[record["kind"]] += term
    self.nodes.append({**record, "term": term})

  def _covering(self, hist, board) -> Solution | None:
    # The latest solve containing the opponent decision after hist, else an
    # evaluation-only solve rooted at it. Returns None if it is uncovered and
    # cover_opponent is off.
    for sol in [*reversed(self.res.solves), *reversed(self._extra)]:
      if hist in sol.index:
        return sol
    if not self.cover_opponent:
      return None
    sol = self.res.solve(hist, board, seed=self.seed + self.n_rollouts, depth=self.eval_depth)
    self.n_rollouts += 1
    self._extra.append(sol)
    self.info["extra_solves"] += 1
    return sol

  def _q(self, sol: Solution, i: int) -> np.ndarray:
    # Q(n, ·, h*) at node i of sol. The leaves n reaches with weight at least
    # min_weight are rolled out for h*, unless an earlier decision of the same
    # solve did so already; the rest keep the solve's belief average.
    res, seat, h = self.res, self.res.seat, self.h
    tree = sol.game.tree
    if id(sol) not in self._leaf_v:
      cache = self._leaf_v[id(sol)] = {}
      # A solve with per-hand rollouts already holds V(l, t, h*): the values
      # it was solved with, under its full τ.
      if sol.leaf_table is not None:
        v = sol.leaf_table[:, :, h]
        for l in np.flatnonzero(~np.isnan(v).any(-1)):
          cache[int(l)] = float(v[l] @ sol.tau[l])
    cache = self._leaf_v[id(sol)]
    weight = leaf_weights(sol, seat, h, [i])
    need = [int(l) for l in np.flatnonzero(weight >= self.min_weight) if int(l) not in cache]
    if need:
      r = np.zeros((len(tree.terminals), N_HANDS))
      r[[tree.leaves[l] for l in need], h] = 1.0
      # Continuations the leaf barely plays are dropped, the rest renormalised.
      tau = np.where(sol.tau >= self.min_tau, sol.tau, 0.0)
      tau = tau / tau.sum(-1, keepdims=True)
      V, _ = res.rollouts.values(tree, r, res.cards, list(sol.board), seat,
                                 seed=self.seed + 500_000 + self.n_rollouts,
                                 t_mask=tau > 0)
      self.n_rollouts += 1
      for l in need:
        cache[l] = float((V[l] * tau[l]).sum())
      self.info["targeted_leaves"] += len(need)

    mass = np.where(sol.leaf_mass > 0, sol.leaf_mass, 1.0)
    leaf_v = (np.asarray(sol.game.leaf_value) / mass[:, None] * sol.tau).sum(-1)
    for l, v in cache.items():
      leaf_v[l] = v
    return history_values(sol, seat, h, leaf_v)[1][i]


def report_aivat(recs, mbb):
  """AIVAT estimators, cumulatively corrected, and the mean of every term."""
  recs = [r for r in recs if "aivat" in r]
  se = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
  raw = np.array([r["payoff"] for r in recs])
  print(f"\n  AIVAT over {len(recs)} hands (mbb/hand, ±95%; std in BB/hand)")
  rows = [("raw", ()), ("+ chance", ("deal", "board", "runout")),
          ("+ our decisions", ("deal", "board", "runout", "ours")),
          ("+ opponent decisions", TERMS)]
  alt = all("board_q" in r["aivat"] for r in recs)
  if alt:
    rows.append(("boards by rollouts", ("deal", "board_q", "runout", "ours", "opponent")))
  for label, keys in rows:
    v = (raw + np.array([sum(r["aivat"][k] for k in keys) for r in recs])) * mbb
    by_seat = "  ".join(f"seat {p} {v[[r['seat'] == p for r in recs]].mean():+7.0f}"
                        for p in (0, 1))
    print(f"  {label:<22} {v.mean():+8.0f} ± {se(v):6.0f}   std {v.std(ddof=1) / 1e3:6.2f}"
          f"   {by_seat}")
  print("  mean of each term (must be ~0):")
  for k in TERMS + (("board_q",) if alt else ()):
    v = np.array([r["aivat"][k] for r in recs]) * mbb
    print(f"    {k:<9} {v.mean():+8.0f} ± {se(v):6.0f}")
  steps = sum(r["aivat"]["opponent_steps"] for r in recs)
  cov = sum(r["aivat"]["opponent_covered"] for r in recs)
  extra = sum(r["aivat"]["extra_solves"] for r in recs)
  if alt:
    fb = sum(r["aivat"]["board_q_fallback"] for r in recs)
    n_b = sum(n["kind"] == "board_q" for r in recs for n in r["aivat"]["nodes"])
    print(f"  boards valued by rollouts {n_b - fb}/{n_b} (the rest by checkdown)")
  print(f"  opponent decisions corrected {cov}/{steps} ({extra} by evaluation-only solves); "
        f"{np.mean([r['aivat_seconds'] for r in recs]):.1f}s of AIVAT per hand")
