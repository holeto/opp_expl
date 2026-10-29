"""Summarise a ``play_slumbot.py`` JSONL log: win rate and how the agent plays.

  uv run python scripts/policy_gradient/analyze_slumbot.py checkpoints/qmmd_hunl_560k/slumbot/*.jsonl

Several logs are pooled. Sections:

  * win rate in mbb/hand with a 95% normal interval, overall and per seat, plus
    an **all-in adjusted** rate: a hand whose all-in was called before the river
    scores its equity share of the pot instead of the dealt runout. That removes
    only runout luck, nothing else (no AIVAT), so expect wide intervals.
  * where hands end (street, who folded, showdown) and what each ending earned;
  * preflop frequencies by seat and post-flop action frequencies per street,
    split into "not facing a bet" (check/bet) and "facing a bet" (fold/call/raise);
  * hand strength behind each of our actions: equity against a random hand and
    against the hand Slumbot actually held (Slumbot reveals it after every hand,
    folds included), Monte Carlo over the unseen cards;
  * bet sizes as pot fractions (raise beyond the call over the pot after the
    call) for both players, how Slumbot's were translated, policy entropy, and
    the hands where the tree state had to be abandoned.
"""

from __future__ import annotations

import argparse
import collections
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from ...envs.hunl_holdem import hand_rank7
from .api import BIG_BLIND
from .play import card_index

STREETS = ("preflop", "flop", "turn", "river")
_VISIBLE = (0, 3, 4, 5)


# ── Equity ───────────────────────────────────────────────────────────────────


@jax.jit
def _equity(key, hero, villain, has_villain, board, n_board):
  """MC equity of ``hero`` (ties count half) over 4096 samples of the unseen cards.

  ``villain`` is used only when ``has_villain``; otherwise it is dealt at
  random. ``board`` positions ``>= n_board`` are dealt at random.
  """
  known = jnp.zeros(52, bool).at[hero].set(True)
  known = known.at[board].set(known[board] | (jnp.arange(5) < n_board))
  known = known.at[villain].set(known[villain] | has_villain)

  def one(k):
    order = jnp.argsort(jnp.where(known, jnp.inf, jax.random.uniform(k, (52,))))
    v = jnp.where(has_villain, villain, order[:2])
    b = jnp.where(jnp.arange(5) < n_board, board, order[2:7])
    rh = hand_rank7(jnp.concatenate([hero, b]))
    rv = hand_rank7(jnp.concatenate([v, b]))
    return jnp.where(rh > rv, 1.0, jnp.where(rh == rv, 0.5, 0.0))

  return jax.vmap(one)(jax.random.split(key, 4096)).mean()


class Equity:
  def __init__(self, seed: int = 0) -> None:
    self.key = jax.random.PRNGKey(seed)
    self.cache: dict = {}

  def __call__(self, hero, board, villain=None) -> float:
    k = (tuple(hero), tuple(board), tuple(villain) if villain else None)
    if k not in self.cache:
      self.key, sub = jax.random.split(self.key)
      b = list(board) + [0] * (5 - len(board))
      self.cache[k] = float(_equity(
        sub, jnp.asarray(hero), jnp.asarray(villain or [0, 0]), villain is not None,
        jnp.asarray(b), len(board),
      ))
    return self.cache[k]


# ── Helpers ──────────────────────────────────────────────────────────────────


def mbb_stats(values) -> tuple[float, float]:
  """(mean, 95% half-width) in mbb/hand from per-hand chips."""
  v = np.asarray(values, float) * 1000 / BIG_BLIND
  if len(v) < 2:
    return float(v.mean()) if len(v) else math.nan, math.nan
  return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(len(v)))


def classify(rec: dict) -> str:
  if rec["call"] > 0:
    return {"fold": "fold", "call": "call", "check": "call"}.get(rec["kind"], "raise")
  return "check" if rec["kind"] in ("check", "call") else "bet"


def pct(counter: collections.Counter, keys) -> str:
  n = sum(counter[k] for k in keys)
  if n == 0:
    return "–"
  return ", ".join(f"{k} {100 * counter[k] / n:.0f}%" for k in keys) + f" (n={n})"


def ending(hand: dict) -> tuple[str, str]:
  """(street the hand ended on, how: 'mmd folds' | 'slumbot folds' | 'showdown')."""
  last = hand["events"][-1]
  if last["kind"] == "fold":
    return STREETS[last["street"]], f"{last['who']} folds"
  return STREETS[last["street"]], "showdown"


def allin_adjusted(hand: dict, eq: Equity) -> float:
  """Equity-share winnings when an all-in was called before the river."""
  ev = hand["events"]
  last = ev[-1]
  all_in = last["add"] >= last["stack"] or any(e["kind"] == "allin" for e in ev)
  if last["kind"] != "call" or last["street"] == 3 or not all_in or not hand.get("bot_hole_cards"):
    return hand["winnings"]
  hero = [card_index(c) for c in hand["hole_cards"]]
  bot = [card_index(c) for c in hand["bot_hole_cards"]]
  board = [card_index(c) for c in hand["board"]][: _VISIBLE[last["street"]]]
  each = (last["pot"] + last["add"]) / 2  # both players' totals are equal after the call
  return (2 * eq(hero, board, bot) - 1) * each


# ── Report ───────────────────────────────────────────────────────────────────


def report(hands: list[dict]) -> None:
  eq = Equity()
  n = len(hands)
  print(f"# {n} hands, checkpoint {hands[0]['checkpoint']} ({hands[0]['params']})\n")

  # Win rate.
  print("## Win rate (mbb/hand, ±95%)")
  for label, sel in (("all", hands),
                     ("MMD small blind/button", [h for h in hands if h["seat"] == 0]),
                     ("MMD big blind", [h for h in hands if h["seat"] == 1])):
    m, ci = mbb_stats([h["winnings"] for h in sel])
    ma, cia = mbb_stats([allin_adjusted(h, eq) for h in sel])
    tot = sum(h["winnings"] for h in sel)
    print(f"- {label}: {len(sel)} hands, {tot:+d} chips = {m:+.0f} ± {ci:.0f}; "
          f"all-in adjusted {ma:+.0f} ± {cia:.0f}")
  sd = [h for h in hands if ending(h)[1] == "showdown"]
  print(f"- showdown winnings {sum(h['winnings'] for h in sd):+d} chips over {len(sd)} hands, "
        f"non-showdown {sum(h['winnings'] for h in hands if h not in sd):+d} over {n - len(sd)}")

  # Endings.
  print("\n## Where hands end (count, chips for MMD)")
  ends = collections.defaultdict(list)
  for h in hands:
    ends[ending(h)].append(h["winnings"])
  for st in STREETS:
    for how in ("mmd folds", "slumbot folds", "showdown"):
      if (st, how) in ends:
        w = ends[(st, how)]
        print(f"- {st:7s} {how:13s}: {len(w):3d} hands, {sum(w):+7d} chips")

  # Preflop.
  print("\n## Preflop")
  for who in ("mmd", "slumbot"):
    first_sb, bb_vs_open, bb_vs_limp, sb_vs_3bet = (collections.Counter() for _ in range(4))
    open_sizes = []
    for h in hands:
      pre = [e for e in h["events"] if e["street"] == 0]
      if not pre:
        continue
      sb_is_who = (h["seat"] == 0) == (who == "mmd")
      first = pre[0]
      if sb_is_who and first["who"] == who:
        c = classify(first)
        first_sb[{"call": "limp"}.get(c, c)] += 1
        if c == "raise":
          open_sizes.append(round(first["x"], 2))
        if len(pre) > 2 and pre[1]["kind"] in ("bet", "allin") and pre[2]["who"] == who:
          sb_vs_3bet[classify(pre[2])] += 1
      elif not sb_is_who and len(pre) > 1 and pre[1]["who"] == who:
        c = classify(pre[1])
        (bb_vs_open if pre[0]["kind"] in ("bet", "allin") else bb_vs_limp)[c] += 1
    print(f"- {who} as SB, first action: {pct(first_sb, ('fold', 'limp', 'raise'))}; "
          f"open sizes {sorted(collections.Counter(open_sizes).items())}")
    print(f"- {who} as SB facing a 3-bet: {pct(sb_vs_3bet, ('fold', 'call', 'raise'))}")
    print(f"- {who} as BB facing a raise: {pct(bb_vs_open, ('fold', 'call', 'raise'))}")
    print(f"- {who} as BB facing a limp: {pct(bb_vs_limp, ('check', 'bet'))}")

  # Post-flop frequencies.
  print("\n## Post-flop action frequencies")
  for who in ("mmd", "slumbot"):
    for st in (1, 2, 3):
      c = collections.Counter(classify(e) for h in hands for e in h["events"]
                              if e["who"] == who and e["street"] == st)
      print(f"- {who:7s} {STREETS[st]:5s}: no bet faced: {pct(c, ('check', 'bet'))}; "
            f"facing a bet: {pct(c, ('fold', 'call', 'raise'))}")

  # Hand strength behind actions.
  print("\n## MMD hand strength by action (mean equity vs random hand / vs Slumbot's actual hand)")
  strength = collections.defaultdict(list)
  for h in hands:
    hero = [card_index(c) for c in h["hole_cards"]]
    bot = [card_index(c) for c in h["bot_hole_cards"]] if h.get("bot_hole_cards") else None
    full = [card_index(c) for c in h["board"]]
    for e in h["events"]:
      if e["who"] != "mmd" or e.get("fallback"):
        continue
      board = full[: _VISIBLE[e["street"]]]
      strength[(e["street"], classify(e))].append(
        (eq(hero, board), eq(hero, board, bot) if bot else math.nan)
      )
  for st in range(4):
    row = []
    for a in ("fold", "check", "call", "bet", "raise"):
      v = strength.get((st, a))
      if v:
        v = np.asarray(v)
        row.append(f"{a} {v[:, 0].mean():.2f}/{np.nanmean(v[:, 1]):.2f} (n={len(v)})")
    print(f"- {STREETS[st]:7s}: " + "; ".join(row))

  # Sizes and translation.
  print("\n## Bet sizes (pot fraction of the raise over the pot after calling)")
  for who in ("mmd", "slumbot"):
    for st in range(4):
      xs = [e["x"] for h in hands for e in h["events"]
            if e["who"] == who and e["street"] == st and e["kind"] in ("bet", "allin")
            and e.get("x") is not None]
      if xs:
        hist = collections.Counter(round(x, 2) for x in xs)
        print(f"- {who:7s} {STREETS[st]:7s}: {sorted(hist.items())}")
  tr = collections.Counter()
  for h in hands:
    for e in h["events"]:
      if "translation" in e:
        tr[e["translation"]["rule"]] += 1
      tr["tree_adjusted"] += bool(e.get("tree_adjusted"))
      tr["fallback decisions"] += bool(e.get("fallback"))
  tr["tree_lost hands"] = sum(h["tree_lost"] for h in hands)
  print(f"- translation of Slumbot bets: {dict(tr)}")

  print("\n## MMD policy entropy (nats, mean per street)")
  for st in range(4):
    ent = [-sum(p * math.log(p) for p in e["probs"] if p > 0) for h in hands
           for e in h["events"] if e["who"] == "mmd" and e["street"] == st and "probs" in e]
    if ent:
      print(f"- {STREETS[st]}: {np.mean(ent):.2f} (n={len(ent)})")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("logs", nargs="+")
  args = ap.parse_args()
  hands = []
  for path in args.logs:
    with open(path) as f:
      hands += [json.loads(line) for line in f if line.strip()]
  report(hands)


if __name__ == "__main__":
  main()
