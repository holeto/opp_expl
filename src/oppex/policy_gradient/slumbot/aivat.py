"""AIVAT values for ``play_slumbot.py`` logs; the reasoning is in ``notes/aivat.md``.

  uv run python scripts/policy_gradient/aivat_slumbot.py checkpoints/qmmd_hunl_560k/slumbot/*.jsonl

A hand's AIVAT value is ``u(z) + Σ term(h)`` over the corrected nodes, where
``term(h) = Σ_a σ(h, a)·ṽ(h, a) − ṽ(h, a*)`` has zero mean given ``h`` for any
``ṽ`` that is a function of the history so far. Corrected nodes:

  * chance: ``ṽ`` is the checkdown value against the opponent's revealed hand,
    so a board card contributes ``2C·(eq_before − eq_after)``, the deal
    ``150·(½ − eq_preflop)``, a called all-in before the river
    ``(2·eq − 1)·C − u(z)``;
  * our decisions: ``ṽ(h, a) = Q(I, a)`` from the checkpoint's Q-head, on the
    translated tree state the agent actually saw;
  * the opponent's decisions, only when its ``σ`` is known (the random
    opponent: uniform over its legal atoms). ``ṽ`` of each alternative is exact
    at terminals, the checkdown value when the action closes the street, and
    ``Σ_a σ(I', a)·Q(I', a)`` at our next decision otherwise.

Each kind of term is reported separately, so its mean can be checked against
zero: a clearly non-zero mean means a bug, never a property of AIVAT.

Equities are exact enumerations over the deck minus both hands and the board.
Preflop that is 1.7M boards (~1.3 s), so preflop equities are cached on disk
under ``$OPPEX_CACHE``, keyed by the suit-canonical matchup.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ...envs.betting import ALL_IN, FOLD, NUM_POINT_ATOMS
from ...envs.hunl_holdem import hand_rank7
from ...policy import policy_probs
from .analyze import mbb_stats
from .api import SMALL_BLIND, BIG_BLIND, STACK_SIZE
from .play import TreeAgent, atom_name, card_index

_VISIBLE = (0, 3, 4, 5)
TERMS = ("deal", "board", "runout", "ours", "opponent")


# ── Exact equity ─────────────────────────────────────────────────────────────


def _make_equity_fn(n_board: int):
  # Equity of H against O over every completion of an n_board-card board.
  # The deck minus the known cards always has 48 - n_board cards, so the
  # completions are fixed index combinations into it: one compile per n_board.
  n_deck = 48 - n_board
  idx = jnp.asarray(np.array(list(itertools.combinations(range(n_deck), 5 - n_board)),
                             np.int32).reshape(-1, 5 - n_board))

  @jax.jit
  def fn(deck, hero, villain, board):
    boards = jnp.concatenate([jnp.broadcast_to(board, (idx.shape[0], n_board)), deck[idx]], 1)
    rh = jax.vmap(lambda b: hand_rank7(jnp.concatenate([hero, b])))(boards)
    rv = jax.vmap(lambda b: hand_rank7(jnp.concatenate([villain, b])))(boards)
    return jnp.mean(jnp.where(rh > rv, 1.0, jnp.where(rh == rv, 0.5, 0.0)))

  return fn


def _canonical(hero, villain) -> str:
  # Equity is invariant under suit permutations and card order within a hand,
  # so the cache key is the smallest relabelling over all 24 permutations.
  best = None
  for perm in itertools.permutations(range(4)):
    relabel = lambda c: (c // 4) * 4 + perm[c % 4]
    key = (tuple(sorted(map(relabel, hero))), tuple(sorted(map(relabel, villain))))
    best = key if best is None or key < best else best
  return f"{best[0][0]},{best[0][1]}|{best[1][0]},{best[1][1]}"


class ExactEquity:
  """Exact equity of ``hero`` against ``villain`` on a board of 0, 3, 4 or 5 cards."""

  def __init__(self, cache_dir: str | None = None) -> None:
    base = Path(cache_dir or os.environ.get("OPPEX_CACHE", Path.home() / ".cache" / "oppex"))
    base.mkdir(parents=True, exist_ok=True)
    self.path = base / "aivat_preflop_equity.json"
    self.preflop = json.loads(self.path.read_text()) if self.path.exists() else {}
    self._fns: dict[int, object] = {}
    self._memo: dict = {}

  def __call__(self, hero, villain, board) -> float:
    hero, villain, board = list(hero), list(villain), list(board)
    if not board:
      key = _canonical(hero, villain)
      if key not in self.preflop:
        self.preflop[key] = self._compute(hero, villain, board)
        self.path.write_text(json.dumps(self.preflop))
      return self.preflop[key]
    memo_key = (tuple(hero), tuple(villain), tuple(board))
    if memo_key not in self._memo:
      self._memo[memo_key] = self._compute(hero, villain, board)
    return self._memo[memo_key]

  def _compute(self, hero, villain, board) -> float:
    if len(board) == 5:
      rh = int(hand_rank7(jnp.asarray(hero + board)))
      rv = int(hand_rank7(jnp.asarray(villain + board)))
      return 1.0 if rh > rv else 0.5 if rh == rv else 0.0
    n = len(board)
    if n not in self._fns:
      self._fns[n] = _make_equity_fn(n)
    used = set(hero) | set(villain) | set(board)
    deck = jnp.asarray([c for c in range(52) if c not in used], jnp.int32)
    return float(self._fns[n](deck, jnp.asarray(hero), jnp.asarray(villain),
                              jnp.asarray(board, jnp.int32).reshape(n)))


# ── Per-hand AIVAT ───────────────────────────────────────────────────────────


class Aivat:
  """Replays logged hands on the agent's tree state and returns their terms."""

  def __init__(self, agent: TreeAgent, equity: ExactEquity) -> None:
    self.agent = agent
    self.eq = equity
    env = agent.env
    self.names = [atom_name(a, agent.bins) for a in range(env.num_actions)]
    # One unit of normalised reward is a whole starting stack.
    self.q_scale = float(STACK_SIZE)
    self.chip_scale = STACK_SIZE / env.starting_stack
    network, zero_state = agent.agent.network, agent.agent.network._zero_state()
    key = jax.random.PRNGKey(0)

    @jax.jit
    def probs_q(params, s, seat):
      (logits, q), _ = network.apply({"params": params}, env.information_set(s, seat, key),
                                     zero_state)
      return policy_probs(logits, env.legal_actions(s, seat)), q

    self._probs_q = probs_q

  def _policy_q(self, s, seat: int) -> tuple[np.ndarray, np.ndarray]:
    probs, q = self._probs_q(self.agent.params, s, jnp.int32(seat))
    return np.asarray(probs, np.float64), np.asarray(q, np.float64) * self.q_scale

  def hand(self, hand: dict) -> dict:
    """``{'raw': u(z), <term>: chips for each of TERMS}`` for one logged hand."""
    seat = hand["seat"]
    hero = [card_index(c) for c in hand["hole_cards"]]
    villain = [card_index(c) for c in hand["bot_hole_cards"]]
    board = [card_index(c) for c in hand["board"]]
    known_opponent = hand.get("opponent") == "random"
    u = hand["winnings"]
    terms = dict.fromkeys(TERMS, 0.0)
    eq = lambda street: self.eq(hero, villain, board[: _VISIBLE[street]])

    # The deal: E[eq] = ½ over all deals.
    terms["deal"] = (SMALL_BLIND + BIG_BLIND) * (0.5 - eq(0))

    tree = self.agent.with_board(self.agent.new_state(hero, seat), board)
    in_sync, street = True, 0
    for e in hand["events"]:
      # Board cards dealt since the previous event; commitments are equal there.
      if e["street"] > street:
        assert e["call"] == 0 and e["pot"] % 2 == 0, e
        terms["board"] += e["pot"] * (eq(street) - eq(e["street"]))
        street = e["street"]

      actor = seat if e["who"] == "mmd" else 1 - seat
      if in_sync and (bool(tree.done) or int(tree.cur_player) != actor):
        in_sync = False
      if not in_sync:
        continue
      if e["who"] == "mmd":
        if "probs" in e:
          probs, q = self._policy_q(tree, seat)
          assert np.abs(probs - e["probs"]).max() < 2e-3, (probs, e["probs"])
          terms["ours"] += probs @ q - q[self.names.index(e["atom"])]
        tree = self.agent.apply(tree, self.names.index(e["played"]))
      else:
        atom = self.names.index(e["tree_atom"])
        if known_opponent:
          # The opponent term assumes the tree equals the real game.
          assert abs(float(tree.committed.sum()) * self.chip_scale - e["pot"]) < 1e-6, e
          terms["opponent"] += self._opponent_term(tree, actor, atom, hero, villain, board)
        tree = self.agent.apply(tree, atom)

    # Runout of an all-in called before the river: one compound chance node.
    last = hand["events"][-1]
    all_in = last["add"] >= last["stack"] or any(e["kind"] == "allin" for e in hand["events"])
    if last["kind"] == "call" and last["street"] < 3 and all_in:
      each = (last["pot"] + last["add"]) / 2
      terms["runout"] = (2 * eq(last["street"]) - 1) * each - u
    return {"raw": float(u), **terms}

  def _opponent_term(self, tree, actor, atom, hero, villain, board) -> float:
    # Σ_a σ(a)·ṽ(child_a) − ṽ(child_atom) with σ uniform over the legal atoms.
    legal = np.flatnonzero(self.agent.legal(tree, actor))
    assert atom in legal, (atom, legal)
    values = {a: self._child_value(tree, actor, a, hero, villain, board) for a in legal}
    return float(np.mean(list(values.values())) - values[atom])

  def _child_value(self, tree, actor, atom, hero, villain, board) -> float:
    # Our value after the opponent plays atom. A bet that reaches the stack is
    # logged as ALL_IN, so it gets the ALL_IN child.
    seat = 1 - actor
    child = self.agent.apply(tree, atom)
    stack = self.agent.env.starting_stack
    if atom >= NUM_POINT_ATOMS and float(child.committed[actor]) >= stack:
      child = self.agent.apply(tree, ALL_IN)
    committed = np.asarray(child.committed, np.float64) * self.chip_scale
    if bool(child.done) and atom == FOLD:
      return committed[actor]
    street = int(tree.street)
    if bool(child.done) or int(child.street) > street:
      # Showdown, a called all-in, or a closed street: the checkdown value,
      # i.e. the expectation over the cards still to come.
      eq = self.eq(hero, villain, board[: _VISIBLE[street]])
      return (2 * eq - 1) * committed[seat]
    probs, q = self._policy_q(child, seat)
    return float(probs @ q)


# ── Report ───────────────────────────────────────────────────────────────────


def report(hands: list[dict], rows: list[dict]) -> None:
  known = any(h.get("opponent") == "random" for h in hands)
  opponent = "random opponent" if known else "Slumbot"
  print(f"# AIVAT, {len(hands)} hands against {opponent}, {hands[0]['checkpoint']}\n")

  stacks = [("raw", ())]
  stacks.append(("+ chance (deal, board, runout)", ("deal", "board", "runout")))
  stacks.append(("+ our decisions", ("deal", "board", "runout", "ours")))
  if known:
    stacks.append(("+ opponent decisions", TERMS))
  print("## Win rate (mbb/hand, ±95%)")
  print("| estimator | all | MMD button | MMD big blind | std (BB/hand) |")
  print("|---|---|---|---|---|")
  for label, keys in stacks:
    vals = np.array([r["raw"] + sum(r[k] for k in keys) for r in rows])
    cells = []
    for sel in (slice(None), [h["seat"] == 0 for h in hands], [h["seat"] == 1 for h in hands]):
      m, ci = mbb_stats(vals[sel] if isinstance(sel, slice) else vals[np.array(sel)])
      cells.append(f"{m:+.0f} ± {ci:.0f}")
    print(f"| {label} | " + " | ".join(cells) + f" | {vals.std(ddof=1) / BIG_BLIND:.1f} |")

  print("\n## Mean of each correction (must be ~0; mbb/hand, ±95%)")
  for k in TERMS:
    vals = [r[k] for r in rows]
    if any(vals):
      m, ci = mbb_stats(vals)
      print(f"- {k}: {m:+.0f} ± {ci:.0f}")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("logs", nargs="+")
  ap.add_argument("--out", help="write per-hand terms as JSONL")
  args = ap.parse_args()
  hands = []
  for path in args.logs:
    with open(path) as f:
      hands += [json.loads(line) for line in f if line.strip()]
  ckpt, which = hands[0]["checkpoint"], hands[0]["params"]
  assert all(h["checkpoint"] == ckpt and h["params"] == which for h in hands)
  agent = TreeAgent(os.path.dirname(ckpt), ckpt, which)
  aivat = Aivat(agent, ExactEquity())
  rows = []
  for i, h in enumerate(hands):
    rows.append(aivat.hand(h))
    if (i + 1) % 100 == 0:
      print(f"... {i + 1}/{len(hands)} hands", flush=True)
  if args.out:
    with open(args.out, "w") as f:
      for h, r in zip(hands, rows):
        f.write(json.dumps({"hand": h["hand"], "seat": h["seat"], **r}) + "\n")
  report(hands, rows)


if __name__ == "__main__":
  main()
