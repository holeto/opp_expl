"""Validate ``SlumbotModel`` (needs the Slumbot data: set ``SLUMBOT_DIR``).

  mc     Games against Slumbot with a uniform translation type per game, played
         by the fixed-type model. The marginal model must give every action
         Slumbot took a positive probability, and its mixed predictions must be
         calibrated (predicted vs observed frequency per probability bin).
  calib  Pools the calibration of several ``mc --save`` runs.
  live   Our slumbot.com logs: in every hand where Slumbot's cards were shown,
         some translation type must reproduce all of its actions exactly.
  rates  Depth-limited subgames at our decisions of ``mc``-style games, every
         opponent node queried as the re-solver would: refusals, bet mapping
         error, worlds per node and time per subgame.

  uv run python scripts/policy_gradient/validate_slumbot_model.py mc --games 2000
  uv run python scripts/policy_gradient/validate_slumbot_model.py live LOG.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from oppex.cfr.cards import HAND_INDEX
from oppex.cfr.subgame import build_subgame
from oppex.envs.betting import FOLD, make_rules
from oppex.policy_gradient.slumbot import native
from oppex.policy_gradient.slumbot.model import SlumbotModel, _intersect, _measure
from oppex.policy_gradient.slumbot.play import card_index

_VISIBLE = (0, 3, 4, 5)


def _rules(args):
  return make_rules(args.stack, args.small_blind, 2 * args.small_blind, args.max_bet_fraction,
                    args.bins)


def _our_atom(rng, st):
  # A random legal atom for our seat: fold 10% when facing a bet, call 45%,
  # else a uniformly chosen raise, so every bin (on and off Slumbot's tree)
  # shows up.
  legal = np.flatnonzero(st.legal)
  raises = [a for a in legal if a >= 2]
  r = rng.random()
  if FOLD in legal and r < 0.10:
    return FOLD
  if r < 0.55 or not raises:
    return 1
  return int(rng.choice(raises))


def play_game(rng, rules, on_slumbot=None, on_ours=None, on_start=None):
  # One game against the fixed-type model; on_start(model, our cards) is
  # called first. on_slumbot(path, board, seat, h, atom) and on_ours(path, board,
  # seat, st) are called at each decision.
  cards = rng.permutation(52)[:9]
  ours, theirs, board = cards[:2].tolist(), cards[2:4].tolist(), cards[4:].tolist()
  seat = int(rng.integers(2))
  h = int(HAND_INDEX[theirs[0], theirs[1]])
  fixed = SlumbotModel(rules, u=float(rng.random()))
  if on_start is not None:
    on_start(fixed, ours)
  path: list[int] = []
  while True:
    st = fixed._step(tuple(path))
    if bool(st.bs.done):
      return fixed
    vis = board[: _VISIBLE[int(st.bs.street)]]
    if int(st.bs.cur_player) == seat:
      if on_ours is not None:
        on_ours(tuple(path), vis, seat, st)
      atom = _our_atom(rng, st)
    else:
      p = fixed.probs([tuple(path)], ours, vis, seat)[0, h]
      atom = int(np.argmax(p))
      if p[atom] < 1 - 1e-9:
        raise AssertionError(f"fixed-type Slumbot is not pure after {path}: {p}")
      if on_slumbot is not None:
        on_slumbot(tuple(path), vis, seat, h, atom)
    path.append(atom)


# ── mc ───────────────────────────────────────────────────────────────────────


def _played_bet_error(model, path, board, seat, h):
  # |env street total - Slumbot's street total| / env pot of the bet the
  # fixed-type model just played for hand h, or None if it did not bet.
  st = model._step(path)
  bkey = tuple(int(c) for c in board)
  w = model._walk(path, bkey, seat)[0]
  m = model._mapped(w, model.u, st, bkey, seat)
  t = int(np.argmax(m.token_probs[h]))
  if m.tokens[t][0] != "b":
    return None
  me = int(st.bs.cur_player)
  child = st.children[int(m.token_atom[t])]
  total = float(st.bs.committed_street[me] + child.committed[me] - st.bs.committed[me])
  return abs(total - int(m.tokens[t][1:]) / model.scale) / float(st.bs.committed.sum())


def _calibration(preds, obs):
  # Predicted vs observed frequency over every (decision, atom) whose
  # prediction is strictly mixed, in ten probability bins.
  mixed = preds.max(-1) < 1 - 1e-9
  p = preds[mixed].ravel()
  y = (np.arange(preds.shape[1])[None, :] == obs[mixed, None]).ravel()
  sel = (p > 1e-9) & (p < 1 - 1e-9)
  p, y = p[sel], y[sel]
  print("calibration (bin: n, mean predicted, observed, z):")
  edges = np.linspace(0, 1, 11)
  for lo, hi in zip(edges, edges[1:]):
    b = (p >= lo) & (p < hi)
    if b.sum() == 0:
      continue
    z = (y[b].sum() - p[b].sum()) / np.sqrt(max((p[b] * (1 - p[b])).sum(), 1e-12))
    print(f"  [{lo:.1f}, {hi:.1f}): {b.sum():6d} {p[b].mean():.4f} {y[b].mean():.4f} {z:+.2f}")
  p_obs = preds[np.arange(len(obs)), obs]
  print(f"mean log-likelihood of observed actions (mixed rows): "
        f"{np.log(np.maximum(p_obs[mixed], 1e-300)).mean():.4f}")


def run_mc(args):
  rules = _rules(args)
  rng = np.random.default_rng(args.seed)
  marg = SlumbotModel(rules)
  preds, obs, t_marg, worlds, rows, errs = [], [], [], [], [], []
  state = {}

  def on_slumbot(path, board, seat, h, atom):
    t = time.time()
    p = marg.probs([path], [], board, seat)[0, h]
    t_marg.append(time.time() - t)
    preds.append(p)
    obs.append(atom)
    worlds.append(marg.stats["worlds"][-1])
    rows.append((list(path), list(board), seat, h))
    e = _played_bet_error(state["fixed"], path, board, seat, h)
    if e is not None:
      errs.append(e)

  failures = 0
  for _ in range(args.games):
    fixed = play_game(rng, rules, on_slumbot, on_start=lambda m, _: state.update(fixed=m))
    failures += len(fixed.stats["failures"])
  preds, obs = np.array(preds), np.array(obs)
  p_obs = preds[np.arange(len(obs)), obs]
  mixed = preds.max(-1) < 1 - 1e-9
  print(f"games {args.games} Slumbot decisions {len(obs)} (mixed predictions {mixed.sum()})")
  print(f"zero-probability observed actions: {(p_obs <= 0).sum()}")
  print(f"fixed-type refusals {failures}, marginal refusals {len(marg.stats['failures'])}, "
        f"desync {marg.stats['desync']}")
  print(f"marginal model: {np.mean(t_marg) * 1e3:.1f} ms per decision, worlds per row "
        f"{np.bincount(marg.stats['worlds']).tolist()}")
  if errs:
    e = np.array(errs)
    print(f"bets Slumbot played: {e.size} ({e.size / len(obs):.2f} of its decisions); "
          f"mapping error (fraction of env pot): exact {np.mean(e < 1e-9):.3f}, "
          f"median {np.median(e):.4f}, p90 {np.quantile(e, 0.9):.4f}, max {e.max():.4f}")
  if mixed.any():
    _calibration(preds, obs)
    print("observed actions predicted below 0.15 (path, board, seat, hand, worlds, p):")
    for i in np.flatnonzero(mixed & (p_obs < 0.15))[:10]:
      nz = {a: round(float(x), 4) for a, x in enumerate(preds[i]) if x > 0}
      print(f"  {rows[i]} worlds {worlds[i]} obs {obs[i]} p {nz}")
  if args.save:
    np.savez(args.save, preds=preds, obs=obs, worlds=np.array(worlds),
             bet_error=np.array(errs))
  return 0 if (p_obs <= 0).sum() == 0 else 1


def run_calib(args):
  # Pools the predictions saved by several mc runs (--save).
  data = [np.load(f) for f in args.logs]
  preds = np.concatenate([d["preds"] for d in data])
  obs = np.concatenate([d["obs"] for d in data])
  print(f"pooled {len(data)} runs, {len(obs)} decisions, "
        f"{(preds.max(-1) < 1 - 1e-9).sum()} mixed")
  _calibration(preds, obs)
  e = np.concatenate([d["bet_error"] for d in data])
  if e.size:
    print(f"bets played {e.size}: exact {np.mean(e < 1e-9):.3f}, median {np.median(e):.4f}, "
          f"p90 {np.quantile(e, 0.9):.4f}")
  return 0


# ── live ─────────────────────────────────────────────────────────────────────


def _tokens(action: str):
  # [(prefix before the token, token, street)] of a slumbot.com string.
  out, i, street = [], 0, 0
  while i < len(action):
    c = action[i]
    if c == "/":
      street += 1
      i += 1
      continue
    j = i + 1
    if c == "b":
      while j < len(action) and action[j].isdigit():
        j += 1
    out.append((action[:i], action[i:j], street))
    i = j
  return out


def _whose(prefix: str) -> int:
  # ACPC position to act after `prefix`: 1 (SB) preflop first, 0 (BB) first
  # postflop, alternating within a street.
  street_actions = prefix.split("/")[-1]
  n = len([t for t in _tokens(street_actions)])
  first = 1 if prefix.count("/") == 0 else 0
  return first if n % 2 == 0 else 1 - first


def run_live(args):
  lib = native.get()
  n_hands = n_shown = n_explained = 0
  bad = []
  for path in args.logs:
    for line in open(path):
      rec = json.loads(line)
      if "action" not in rec or rec.get("opponent", "slumbot") != "slumbot":
        continue
      n_hands += 1
      if not rec.get("bot_hole_cards"):
        continue
      n_shown += 1
      pos = 1 - rec["client_pos"]
      cards = [card_index(c) for c in rec["bot_hole_cards"]]
      board = [card_index(c) for c in rec.get("board", [])]
      ok_set = ((0.0, 1.0),)
      first_bad = None
      for prefix, tok, street in _tokens(rec["action"]):
        if _whose(prefix) != pos:
          continue
        kept = []
        for c in lib.cells(pos, prefix):
          ivs = _intersect(ok_set, c.intervals)
          if not c.feasible or _measure(ivs) <= 0:
            continue
          try:
            q = lib.query(pos, prefix, board[: _VISIBLE[street]], np.array([cards]), c.u)
          except native.SlumbotStatusError as e:
            first_bad = first_bad or (prefix, tok, f"status {e.status}")
            continue
          n = int(q.n_out[0])
          choice = int(np.argmax(q.prob[0, :n]))
          kind, bet = int(q.kind[0, choice]), int(q.bet[0, choice])
          mine = {0: "f", 1: "k", 2: "c"}.get(kind) or f"b{bet}"
          if mine == tok:
            kept.extend(ivs)
          elif first_bad is None:
            first_bad = (prefix, tok, mine)
        ok_set = tuple(sorted(kept))
        if not ok_set:
          break
      if ok_set:
        n_explained += 1
      else:
        bad.append((rec["action"], rec["bot_hole_cards"], rec.get("board"), first_bad))
  print(f"hands {n_hands}, Slumbot's cards shown {n_shown}, explained {n_explained}")
  for b in bad[:20]:
    print("  UNEXPLAINED", b)
  return 0 if not bad else 1


# ── rates ────────────────────────────────────────────────────────────────────


def run_rates(args):
  rules = _rules(args)
  rng = np.random.default_rng(args.seed)
  marg = SlumbotModel(rules)
  rows = []

  def on_ours(path, board, seat, st):
    if len(rows) >= args.roots or rng.random() > args.root_prob:
      return
    tree = build_subgame(rules, st.bs, path, args.depth)
    opp = [n.path for n in tree.nodes if n.player != seat]
    if not opp:
      return
    f0 = len(marg.stats["failures"])
    t = time.time()
    marg.probs(opp, [], board, seat)
    rows.append((len(tree.nodes), len(opp), time.time() - t,
                 len(marg.stats["failures"]) - f0, int(st.bs.street)))

  g = 0
  while len(rows) < args.roots:
    play_game(rng, rules, on_ours=on_ours)
    g += 1
  r = np.array(rows, np.float64)
  err = np.array(marg.stats["bet_error"])
  print(f"subgames {len(rows)} from {g} games; nodes mean {r[:, 0].mean():.0f}, "
        f"opponent nodes mean {r[:, 1].mean():.0f}")
  for s in range(4):
    sel = r[:, 4] == s
    if sel.any():
      print(f"  street {s}: {sel.sum()} subgames, {r[sel, 2].mean():.2f} s each "
            f"(max {r[sel, 2].max():.2f})")
  print(f"queries {marg.stats['queries']}, cells {marg.stats['cells']}, "
        f"refusals {len(marg.stats['failures'])}, raises clamped {marg.stats['raise_clamped']}, "
        f"desync {marg.stats['desync']}")
  if err.size:
    print(f"bet mapping error (fraction of env pot) over {err.size} mapped bets: "
          f"exact {np.mean(err < 1e-9):.3f}, median {np.median(err):.4f}, "
          f"p90 {np.quantile(err, 0.9):.4f}, max {err.max():.4f}")
  # 0 worlds: a node behind an atom Slumbot never plays there (reach 0).
  print(f"worlds per node (count of nodes with 0, 1, ... worlds): "
        f"{np.bincount(marg.stats['worlds']).tolist()}")
  for a, s in marg.stats["failures"][:10]:
    print(f"  refusal {s}: {a}")
  return 0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("mc", "calib", "live", "rates"))
  ap.add_argument("logs", nargs="*",
                  help="live: slumbot.com JSONL logs (play_slumbot.py); calib: mc --save files")
  ap.add_argument("--save", help="mc: save predictions to this .npz")
  ap.add_argument("--games", type=int, default=1000)
  ap.add_argument("--roots", type=int, default=100)
  ap.add_argument("--root-prob", type=float, default=0.5)
  ap.add_argument("--depth", type=int, default=4)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--stack", type=float, default=400)
  ap.add_argument("--small-blind", type=float, default=1)
  ap.add_argument("--max-bet-fraction", type=float, default=2.0)
  ap.add_argument("--bins", type=int, default=4)
  args = ap.parse_args()
  raise SystemExit({"mc": run_mc, "calib": run_calib, "live": run_live,
                    "rates": run_rates}[args.mode](args))


if __name__ == "__main__":
  main()
