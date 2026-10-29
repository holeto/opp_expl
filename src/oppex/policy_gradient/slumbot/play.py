"""Play a trained QMMD HUNL checkpoint against Slumbot over its HTTP API.

  uv run python scripts/policy_gradient/play_slumbot.py checkpoints/qmmd_hunl_560k --hands 10
  uv run python scripts/policy_gradient/play_slumbot.py checkpoints/qmmd_hunl_560k --opponent random --hands 1000
  uv run python scripts/policy_gradient/analyze_slumbot.py checkpoints/qmmd_hunl_560k/slumbot/<log>.jsonl

The scripts are thin wrappers around ``main`` here and in ``analyze.py``.

Every hand is appended to a JSONL log (default ``<run_dir>/slumbot/``) with
every decision of ours (probabilities, chosen atom, chips sent) and every
Slumbot action (its pot fraction and how it was translated).

**Scale.** Slumbot plays 50/100 with 20 000-chip stacks; the training env is
1/2 with 400. Both are 200 BB, and every quantity crossing between the two
games is a pot fraction, so no chip rescaling is ever needed.

**Two games in parallel.** ``RealGame`` tracks the real hand in Slumbot's
chips; a ``HunlState`` (the *tree* state) tracks the hand as the agent
understands it, i.e. inside its abstraction. The agent only ever sees the tree
state, so its infoset is always one it could have met in training.

  * Slumbot's actions enter the tree after **action translation**. A bet or
    raise of pot fraction ``x`` (raise beyond the call over the pot after the
    call — ``betting.py``'s convention) between two of our sizes ``A ≤ x ≤ B``
    becomes ``A`` with the pseudo-harmonic probability
    ``(B − x)(1 + A) / ((B − A)(1 + x))`` (Ganzfried & Sandholm 2013), otherwise
    ``B``. Our sizes are ``bet_bins`` below the real all-in fraction, plus the
    all-in itself as the top size. A bet below the smallest bin maps to it
    deterministically (never down to a check: we would then face a bet the agent
    believes is not there). Fold, check, call and all-in map to themselves.
  * Our actions are taken in the tree and replayed in the real game as the same
    *pot fraction* of the real pot. Real and tree pots differ after any
    translated Slumbot bet, which is exactly why fractions, not chips, cross
    over. A bet that hits the real stack is recorded in the tree as ALL_IN.

If the two games ever disagree on whose turn it is or on the street (e.g. we
call a Slumbot bet of 90% of the stack that the tree read as an all-in, so the
tree hand is over while the real one goes on), the tree is abandoned for the
rest of the hand and we check/call down. Such hands are flagged ``tree_lost``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...envs.betting import ALL_IN, CALL, FOLD, NUM_POINT_ATOMS
from ..blueprint import Blueprint
from .api import BIG_BLIND, SMALL_BLIND, STACK_SIZE, SlumbotClient, SlumbotError

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"  # any fixed bijection works: the deal is suit-symmetric


def card_index(card: str) -> int:
  """'Ac' → rank * 4 + suit, the env's card encoding."""
  return _RANKS.index(card[0]) * 4 + _SUITS.index(card[1])


def tokenize(action: str):
  """Yield ('k'|'c'|'f', None) or ('b', N); street separators are skipped.

  Streets are advanced by ``RealGame`` itself, so '/' carries no information.
  """
  i = 0
  while i < len(action):
    c = action[i]
    i += 1
    if c == "/":
      continue
    if c == "b":
      j = i
      while i < len(action) and action[i].isdigit():
        i += 1
      yield "b", int(action[j:i])
    else:
      yield c, None


# ── Real game (Slumbot chips) ────────────────────────────────────────────────


class RealGame:
  """The real hand in Slumbot's chips, seats indexed as in the env (0 = SB/button).

  Mirrors Slumbot's rules: a bet ``bN`` raises the street total to N, the
  minimum raise increment is the last one (at least one BB).
  """

  def __init__(self) -> None:
    self.committed = [SMALL_BLIND, BIG_BLIND]
    self.street_committed = [SMALL_BLIND, BIG_BLIND]
    self.street = 0
    self.cur = 0
    self.acted = 0
    self.last_raise = BIG_BLIND - SMALL_BLIND
    self.done = False

  def stack(self, p: int) -> int:
    return STACK_SIZE - self.committed[p]

  def pot(self) -> int:
    return sum(self.committed)

  def call_amount(self, p: int) -> int:
    return max(self.street_committed) - self.street_committed[p]

  def min_raise(self) -> int:
    return max(self.last_raise, BIG_BLIND)

  def apply(self, kind: str, to: int | None) -> dict:
    """Apply one token by ``self.cur``; returns what it did (chips, pot fraction)."""
    p = self.cur
    pot, call, stack = self.pot(), self.call_amount(p), self.stack(p)
    rec = {"street": self.street, "pot": pot, "call": call, "stack": stack}
    if kind == "f":
      self.done = True
      return {**rec, "kind": "fold", "add": 0}
    if kind == "k":
      assert call == 0, "check facing a bet"
      add = 0
    elif kind == "c":
      add = min(call, stack)
    else:
      add = to - self.street_committed[p]
      raise_inc = to - max(self.street_committed)
      self.last_raise = raise_inc
      rec["x"] = raise_inc / (pot + call)                  # raise over the pot after calling
      rec["allin_x"] = (stack - call) / (pot + call)
    self.committed[p] += add
    self.street_committed[p] += add
    self.acted += 1
    all_in = min(self.stack(0), self.stack(1)) <= 0
    matched = self.street_committed[0] == self.street_committed[1]
    raised = kind == "b"
    round_over = (matched and self.acted >= 2) or (all_in and not raised)
    if round_over:
      if self.street == 3 or all_in:
        self.done = True
      else:
        self.street += 1
        self.street_committed = [0, 0]
        self.acted = 0
        self.last_raise = 0
        self.cur = 1  # big blind first post-flop
    else:
      self.cur = 1 - p
    kind_name = {"k": "check", "c": "call", "b": "allin" if self.stack(p) == 0 else "bet"}[kind]
    return {**rec, "kind": kind_name, "add": add}


# ── Action translation ───────────────────────────────────────────────────────


def pseudo_harmonic(x: float, a: float, b: float) -> float:
  """P(map x to the smaller size a) for a ≤ x ≤ b (Ganzfried & Sandholm 2013)."""
  return (b - x) * (1 + a) / ((b - a) * (1 + x))


def translate_bet(x: float, allin_x: float, bins: np.ndarray, rng: np.random.Generator) -> dict:
  """Map a real bet of pot fraction ``x`` onto a bet atom or ALL_IN."""
  sizes = [(float(f), NUM_POINT_ATOMS + i) for i, f in enumerate(bins) if f < allin_x - 1e-9]
  sizes.append((allin_x, ALL_IN))
  if x >= allin_x - 1e-9:
    return {"atom": ALL_IN, "rule": "allin"}
  if x <= sizes[0][0] + 1e-9:
    return {"atom": sizes[0][1], "rule": "below_min" if x < sizes[0][0] - 1e-9 else "exact"}
  for (a, atom_a), (b, atom_b) in zip(sizes, sizes[1:]):
    if a <= x <= b:
      if abs(x - b) < 1e-9:
        return {"atom": atom_b, "rule": "exact"}
      p_a = pseudo_harmonic(x, a, b)
      atom = atom_a if rng.random() < p_a else atom_b
      return {"atom": atom, "rule": "harmonic", "A": a, "B": b, "p_A": p_a}
  raise AssertionError("unreachable")


# ── Agent side ───────────────────────────────────────────────────────────────


class TreeAgent(Blueprint):
  """The blueprint plus the tree state of a Slumbot hand."""

  def __init__(self, run_dir: str, checkpoint: str | None, which: str) -> None:
    super().__init__(run_dir, checkpoint, which)
    env = self.env
    if env.starting_stack / env.big_blind != STACK_SIZE / BIG_BLIND:
      raise ValueError("checkpoint was trained at a different stack depth than Slumbot's 200 BB")

  def new_state(self, hole: list[int], seat: int):
    s = self.env.init_state(jax.random.PRNGKey(0))
    other = [c for c in range(52) if c not in hole][:2]  # never observed by our infoset
    cards = [hole, other] if seat == 0 else [other, hole]
    return s._replace(hole_cards=jnp.asarray(cards, jnp.int32))

  def with_board(self, s, board: list[int]):
    if not board:
      return s
    return s._replace(board=s.board.at[: len(board)].set(jnp.asarray(board, jnp.int32)))


def atom_name(atom: int, bins: np.ndarray) -> str:
  if atom >= NUM_POINT_ATOMS:
    return f"bet{bins[atom - NUM_POINT_ATOMS]:g}"
  return {FOLD: "fold", CALL: "call", ALL_IN: "allin"}[atom]


def real_incr(game: RealGame, seat: int, atom: int, bins: np.ndarray) -> tuple[str, int]:
  """Our tree atom → (Slumbot incr string, atom actually played).

  A bet atom is the same pot fraction of the *real* pot, clipped to Slumbot's
  legal raise range; one that reaches the stack is returned as ALL_IN.
  """
  call, stack = game.call_amount(seat), game.stack(seat)
  can_raise = stack > call and game.stack(1 - seat) > 0
  if atom == FOLD:
    return ("f", FOLD) if call > 0 else ("k", CALL)
  if atom == CALL or not can_raise:
    return ("c" if call > 0 else "k"), CALL
  top = max(game.street_committed)
  max_inc = stack - call
  if atom == ALL_IN:
    return f"b{top + max_inc}", ALL_IN
  frac = float(bins[atom - NUM_POINT_ATOMS])
  inc = int(round(frac * (game.pot() + call)))
  inc = min(max(inc, game.min_raise()), max_inc)
  return f"b{top + inc}", (ALL_IN if inc == max_inc else atom)


# ── One hand ─────────────────────────────────────────────────────────────────


def play_hand(client: SlumbotClient, agent: TreeAgent, rng: np.random.Generator) -> dict:
  r = client.new_hand()
  seat = 1 - r["client_pos"]  # client_pos 0 = big blind = env seat 1
  hole = [card_index(c) for c in r["hole_cards"]]
  game = RealGame()
  tree = agent.new_state(hole, seat)
  tree_lost = False
  history = ""
  events = []

  while True:
    # Slumbot's actions since our last one (our own token is already applied).
    action = r.get("action", "")
    assert action.startswith(history), (history, action)
    for kind, to in tokenize(action[len(history):]):
      if game.done:
        break
      actor = game.cur
      assert actor != seat, f"unexpected own action in {action!r}"
      rec = game.apply(kind, to)
      rec.update(who="slumbot", token=kind + (str(to) if to is not None else ""))
      if rec["kind"] == "fold":
        atom = FOLD
      elif rec["kind"] in ("check", "call"):
        atom = CALL
      elif rec["kind"] == "allin":
        atom = ALL_IN
      else:
        tr = translate_bet(rec["x"], rec["allin_x"], agent.bins, rng)
        rec["translation"] = tr
        atom = tr["atom"]
      if not tree_lost:
        if bool(tree.done) or int(tree.cur_player) != actor:
          tree_lost = True
        else:
          legal = agent.legal(tree, actor)
          if not legal[atom]:  # e.g. a raise the tree thinks is impossible
            atom = ALL_IN if legal[ALL_IN] else CALL
            rec["tree_adjusted"] = True
          tree = agent.apply(tree, atom)
      rec["tree_atom"] = atom_name(atom, agent.bins)
      events.append(rec)
    history = action

    if "winnings" in r:
      return {
        "client_pos": r["client_pos"], "seat": seat, "hole_cards": r["hole_cards"],
        "board": r.get("board", []), "bot_hole_cards": r.get("bot_hole_cards"),
        "action": action, "winnings": r["winnings"], "tree_lost": tree_lost,
        "events": events,
      }

    # Our turn.
    assert not game.done and game.cur == seat, (action, game.__dict__)
    board = [card_index(c) for c in r.get("board", [])]
    if not tree_lost and (bool(tree.done) or int(tree.cur_player) != seat
                          or int(tree.street) != game.street):
      tree_lost = True
    rec = {"who": "mmd", "street": game.street, "pot": game.pot(),
           "call": game.call_amount(seat), "stack": game.stack(seat)}
    if tree_lost:
      atom = CALL
      rec["fallback"] = True
    else:
      tree = agent.with_board(tree, board)
      probs, legal = agent.policy(tree, seat)
      atom = int(rng.choice(len(probs), p=probs / probs.sum()))
      rec.update(probs=[round(float(p), 4) for p in probs], legal=legal.tolist(),
                 tree_pot=float(tree.committed.sum()))
    incr, played = real_incr(game, seat, atom, agent.bins)
    rec.update(atom=atom_name(atom, agent.bins), played=atom_name(played, agent.bins),
               incr=incr)
    kind = incr[0]
    real_rec = game.apply(kind, int(incr[1:]) if kind == "b" else None)
    rec.update(kind=real_rec["kind"], add=real_rec["add"], x=real_rec.get("x"))
    if not tree_lost:
      tree = agent.apply(tree, played)
    events.append(rec)
    history += incr
    r = client.act(incr)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir")
  ap.add_argument("--checkpoint", help="checkpoint file (default: latest in run_dir)")
  ap.add_argument("--params", choices=("params", "target", "magnet"), default="params")
  ap.add_argument("--opponent", choices=("slumbot", "random"), default="slumbot",
                  help="random: a local uniformly random opponent (random_opponent.py)")
  ap.add_argument("--hands", type=int, default=10)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--min-interval", type=float, default=1.0,
                  help="minimum seconds between API requests (do not go below 0.5)")
  ap.add_argument("--out", help="JSONL log (default: <run_dir>/<opponent>/<params>_<time>.jsonl)")
  args = ap.parse_args()
  if args.min_interval < 0.5:
    ap.error("--min-interval below 0.5 s would overload the Slumbot server")

  agent = TreeAgent(args.run_dir, args.checkpoint, args.params)
  out = args.out or os.path.join(
    args.run_dir, args.opponent,
    f"{args.params}_{datetime.datetime.now():%Y%m%d_%H%M%S}.jsonl",
  )
  os.makedirs(os.path.dirname(out), exist_ok=True)
  if args.opponent == "random":
    # Imported here because random_opponent imports this module.
    from .random_opponent import RandomOpponent
    client = RandomOpponent(agent.bins, seed=args.seed + 1)
  else:
    client = SlumbotClient(min_interval=args.min_interval)
    username = os.environ.get("SLUMBOT_USERNAME")
    password = os.environ.get("SLUMBOT_PASSWORD")
    if username and password:
      client.login(username, password)
  rng = np.random.default_rng(args.seed)
  print(f"{agent.checkpoint} (step {agent.step}, {args.params}) → {out}", flush=True)

  total, t0 = 0, time.monotonic()
  with open(out, "a") as f:
    for h in range(args.hands):
      try:
        hand = play_hand(client, agent, rng)
      except SlumbotError as e:
        print(f"hand {h}: {e}; stopping", flush=True)
        break
      hand.update(hand=h, checkpoint=agent.checkpoint, step=agent.step, params=args.params,
                  opponent=args.opponent)
      f.write(json.dumps(hand) + "\n")
      f.flush()
      total += hand["winnings"]
      pos = "BB" if hand["client_pos"] == 0 else "SB"
      print(f"hand {h:4d} {pos} {''.join(hand['hole_cards'])} {hand['action']:<40} "
            f"{hand['winnings']:+7d}  total {total:+8d}  "
            f"({1000 * total / BIG_BLIND / (h + 1):+.0f} mbb/hand, "
            f"{time.monotonic() - t0:.0f}s, {client.n_requests} requests)", flush=True)


if __name__ == "__main__":
  main()
