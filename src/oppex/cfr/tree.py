"""The preflop public tree.

Public-state CFR walks *publicly observable* actions only. Preflop that is easy to
pin down: the board is neither public nor dealt, so the public state is exactly
the betting state, and because bet sizes are pot-relative (a function of public
state alone) every child is well-defined without reference to anyone's holding.

The tree is built eagerly in plain Python from ``oppex.envs.betting``. Nothing
here is jitted — ``betting.step`` on concrete arrays returns concrete arrays, so
``bool(out.done)`` just works. That is the whole reason the builder consumes
``BettingState`` rather than the env: ``HunlState`` carries an action history that
would distinguish CALL from ALL_IN even where the resulting public state is
identical, giving a tree with duplicate public states.

Terminals come in three kinds:

* ``FOLD`` — payoff is a hand-independent constant in chips.
* ``SHOWDOWN`` — an all-in settled preflop; payoff is ``stake * EV[a, b]``.
* ``CHECKDOWN`` — where preflop betting closed without an all-in. Under a
  preflop-only tree there is nowhere left to go, so it is valued as if both
  players check to showdown.

That last one makes ``allow_limp=True`` a **different game** from real HUNL: it
hands the small blind a free showdown that postflop play would punish, so the
solver limps far more than any real strategy would. It is a fine debug target but
**cannot be compared to published push-fold charts** — only ``allow_limp=False``,
which removes the limp entirely, reproduces the game those charts describe.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..envs import betting
from ..envs.betting import ALL_IN, CALL, FOLD, BettingRules, BettingState

FOLD_T, SHOWDOWN_T, CHECKDOWN_T = 0, 1, 2
_KIND_NAME = {FOLD_T: "fold", SHOWDOWN_T: "showdown", CHECKDOWN_T: "checkdown"}
_ATOM_NAME = {FOLD: "FOLD", CALL: "CALL", ALL_IN: "ALL_IN"}


class TerminalNode(NamedTuple):
  kind: int
  const: float           # chips to player 0, hand-independent (fold terminals)
  stake: float           # chips at risk, multiplied by EV (showdown / checkdown)
  betting: BettingState


class DecisionNode(NamedTuple):
  player: int
  legal: tuple[bool, ...]
  child: tuple[tuple[str, int] | None, ...]   # ("D", i) | ("T", j) | None if illegal
  betting: BettingState


class PreflopTree(NamedTuple):
  rules: BettingRules
  nodes: tuple[DecisionNode, ...]
  terminals: tuple[TerminalNode, ...]
  n_actions: int
  root: int


def _key(bs: BettingState):
  """Hashable identity of a public state, for the sibling-collision check."""
  return (
    tuple(np.asarray(bs.committed).tolist()),
    tuple(np.asarray(bs.committed_street).tolist()),
    int(bs.street), int(bs.cur_player), int(bs.acted),
    float(bs.last_raise_size), bool(bs.done),
  )


def build_preflop_tree(
  rules: BettingRules, *, allow_limp: bool = True, max_depth: int = 8,
  allow_bets: bool = False, max_nodes: int = 100_000,
) -> PreflopTree:
  """Enumerate the preflop public tree.

  ``allow_bets`` opts in to pot-fraction bet atoms (``num_bet_bins > 0``). It is
  off by default because of the leaf valuation, not the enumeration: every line
  that closes preflop without an all-in becomes a ``CHECKDOWN`` terminal, valued
  as though both players check down from there. With bet atoms most lines end
  that way, and each one hands both players full showdown equity in a pot they
  bought cheaply — so the solver sees raise-and-see-a-flop as far better than it
  is. **A tree built with ``allow_bets=True`` is a scaling and speed benchmark,
  not a preflop solution**, and the same caveat that rules out chart comparison
  for ``allow_limp=True`` applies here with more force.

  ``max_nodes`` is a tripwire, not a bound anyone should hit: the tree unrolls
  into the jit graph at trace time, so node count is compile time.
  """
  if rules.bet_bins.shape[0] != 0 and not allow_bets:
    raise ValueError(
      f"got {rules.bet_bins.shape[0]} bet atoms but allow_bets=False. Pot-fraction "
      "bet atoms need a leaf valuation better than checkdown equity; pass "
      "allow_bets=True to build the tree anyway for scaling work."
    )
  n_actions = betting.num_atoms(rules)
  nodes: list[DecisionNode] = []
  terminals: list[TerminalNode] = []
  dropped: list[tuple[int, int, int]] = []

  def add_terminal(kind, const, stake, bs) -> tuple[str, int]:
    terminals.append(TerminalNode(kind, float(const), float(stake), bs))
    return ("T", len(terminals) - 1)

  def visit(bs: BettingState, depth: int) -> tuple[str, int]:
    if depth > max_depth:
      raise RuntimeError(f"preflop tree deeper than max_depth={max_depth}")
    slot = len(nodes)
    if slot >= max_nodes:
      raise RuntimeError(f"preflop tree exceeded max_nodes={max_nodes}")
    nodes.append(None)  # reserve, so children get later indices
    legal = np.asarray(
      betting.legal_atoms(rules, bs, drop_duplicate_allin=True)
    ).tolist()
    children: list[tuple[str, int] | None] = [None] * n_actions
    seen: dict = {}

    for atom in range(n_actions):
      if not legal[atom]:
        continue
      out = betting.step(rules, bs, jnp.int32(atom))
      done, is_fold, advance = bool(out.done), bool(out.is_fold), bool(out.advance)

      if allow_limp is False and atom == CALL and not done:
        legal[atom] = False        # the limp is what makes this not a shove-fold game
        continue

      # Two sibling edges reaching the same public state ARE the same action;
      # BettingState carries no history, so this catches duplicates for free.
      # Bet atoms collide routinely once `chips_added` clips them — every bin
      # above the stack lands on the all-in state, and adjacent bins can round
      # together in a small pot. Keeping both would split regret mass across
      # identical columns, so the later atom is struck from `legal` instead.
      # Atom order means the survivor is the more canonical label: ALL_IN beats
      # a bet bin that clipped to it, and a smaller bin beats a larger one.
      k = _key(out.state)
      if k in seen:
        legal[atom] = False
        dropped.append((slot, seen[k], atom))
        continue
      seen[k] = atom

      committed = out.state.committed
      if done and is_fold:
        const = betting.fold_payoffs(committed, bs.cur_player)[0]
        children[atom] = add_terminal(FOLD_T, const, 0.0, out.state)
      elif done:
        stake = betting.showdown_payoffs(committed, jnp.int32(1))[0]
        children[atom] = add_terminal(SHOWDOWN_T, 0.0, stake, out.state)
      elif advance:
        # Preflop betting closed without an all-in: the preflop-only cut.
        stake = betting.showdown_payoffs(committed, jnp.int32(1))[0]
        children[atom] = add_terminal(CHECKDOWN_T, 0.0, stake, out.state)
      else:
        children[atom] = visit(out.state, depth + 1)

    nodes[slot] = DecisionNode(
      player=int(bs.cur_player), legal=tuple(legal),
      child=tuple(children), betting=bs,
    )
    return ("D", slot)

  kind, root = visit(betting.initial_state(rules), 0)
  assert kind == "D", "the root must be a decision node"
  if rules.bet_bins.shape[0] == 0 and dropped:
    raise AssertionError(
      "duplicate sibling actions in a push-fold tree, where clipping cannot "
      f"create them — this is a betting-rules bug, not a bin collision: {dropped}"
    )
  return PreflopTree(rules, tuple(nodes), tuple(terminals), n_actions, root)


def format_tree(tree: PreflopTree) -> str:
  """Readable dump of the tree, for eyeballing the constants."""
  lines = []

  def walk(ref, depth):
    kind, i = ref
    pad = "  " * depth
    if kind == "T":
      t = tree.terminals[i]
      detail = (f"const={t.const:+g}" if t.kind == FOLD_T else f"stake={t.stake:g}")
      lines.append(f"{pad}T{i} {_KIND_NAME[t.kind]:9s} {detail}")
      return
    n = tree.nodes[i]
    lines.append(
      f"{pad}n{i} P{n.player} committed={np.asarray(n.betting.committed).tolist()}"
    )
    for atom, ch in enumerate(n.child):
      if ch is None:
        continue
      lines.append(f"{pad}  {_ATOM_NAME.get(atom, f'BET_{atom - 3}')}")
      walk(ch, depth + 2)

  walk(("D", tree.root), 0)
  return "\n".join(lines)
