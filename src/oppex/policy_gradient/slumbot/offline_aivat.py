"""AIVAT for the re-solver on slumbot.com, computed after play.

During play Slumbot's hand is hidden, and AIVAT's per-history values need it.
So ``SlumbotResolvingAgent`` saves the subgames it played at the end of each
hand, once the server has revealed the hand (``save_hand_solves``, one ``.npz``
per hand next to the JSONL log, ``play_slumbot.py --save-solves``). This module
replays the saved hands through ``ResolvingAivat``. The Q-values come from the
solves that were actually played; no played subgame is solved again. Still
computed here, because play never needed them: the leaf rollouts for Slumbot's
hand, the boards valued by rollouts, and the evaluation-only solves of Slumbot
decisions outside every played subgame.

**Only the past of a decision may enter its value.** ``SlumbotModel`` filters
the worlds of the real history by the exact strings of ``set_real``. A leaf path
that follows the real line would be filtered by later Slumbot moves if the whole
hand's strings were set, which leaks ``a*`` into ``ṽ`` and biases the term. So a
hand is replayed step by step, and each string is set only after its step.

**Chips.** Payoffs and stakes come from the real string, divided by the scale:
slumbot.com scores real chips. The sampling distributions are the logged ones,
our re-solved ``σ`` and the model's marginal at Slumbot's hand; the replay
recomputes the latter as a check (``prob_gap``).

  uv run python scripts/policy_gradient/aivat_resolve_slumbot.py RUN_DIR LOG.jsonl --out A.jsonl
  uv run python scripts/policy_gradient/aivat_resolve_slumbot.py RUN_DIR --report A.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from ...cfr import equity, infoset_cfr
from ...cfr.cards import HAND_INDEX, N_HANDS
from ...cfr.subgame import build_subgame, replay_betting
from ..resolving import Solution
from ..resolving_aivat import ResolvingAivat, report_aivat
from .play import RealGame, card_index, tokenize
from .rollouts import _advance


# ── Saved solves ─────────────────────────────────────────────────────────────


def save_hand_solves(path: str, solves: list, h: int, depth: int | None) -> None:
  """Writes the played ``solves`` of one hand, restricted to Slumbot's hand ``h``."""
  arrays = {"n": np.array(len(solves)), "h": np.array(h),
            "depth": np.array(-1 if depth is None else depth)}
  for i, sol in enumerate(solves):
    nodes = sorted(sol.pi)
    arrays.update({
      f"s{i}_history": np.array(sol.stats["history"], np.int32),
      f"s{i}_board": np.array(sol.board, np.int32),
      f"s{i}_sigma": np.asarray(sol.sigma),
      f"s{i}_tau": np.asarray(sol.tau),
      f"s{i}_leaf_value": np.asarray(sol.game.leaf_value),
      f"s{i}_leaf_mass": np.asarray(sol.leaf_mass),
      f"s{i}_z": np.array(sol.z),
      f"s{i}_payoff": np.asarray(sol.payoff)[:, h],
      f"s{i}_pi_nodes": np.array(nodes, np.int32),
      f"s{i}_pi": np.stack([np.asarray(sol.pi[n])[h] for n in nodes]) if nodes
                  else np.zeros((0, 0)),
    })
  # Replaced atomically, so a killed job never leaves half a file.
  tmp = path + ".tmp"
  with open(tmp, "wb") as f:
    np.savez_compressed(f, **arrays)
  os.replace(tmp, path)


def load_hand_solves(path: str, rules, seat: int) -> list[Solution]:
  """The saved solves as ``Solution``s, valid only at the saved Slumbot hand.

  The tree is rebuilt from the root history. ``payoff`` and ``pi`` hold only
  that hand's entries, and the fields AIVAT does not read are empty.
  """
  out = []
  with np.load(path) as f:
    h, depth = int(f["h"]), int(f["depth"])
    for i in range(int(f["n"])):
      history = tuple(int(a) for a in f[f"s{i}_history"])
      root = replay_betting(rules, history)[-1][0]
      tree = build_subgame(rules, root, history, None if depth < 0 else depth)
      sigma = f[f"s{i}_sigma"]
      if len(tree.nodes) != len(sigma):
        raise ValueError(f"{path}: the rebuilt tree of solve {i} has {len(tree.nodes)} nodes, "
                         f"{len(sigma)} saved")
      payoff = np.zeros((len(tree.terminals), N_HANDS))
      payoff[:, h] = f[f"s{i}_payoff"]
      pi = {}
      for node, row in zip(f[f"s{i}_pi_nodes"], f[f"s{i}_pi"]):
        pi[int(node)] = np.zeros((N_HANDS, len(row)))
        pi[int(node)][h] = row
      game = infoset_cfr.InfosetGame(tree, seat, np.zeros(len(tree.terminals)),
                                     f[f"s{i}_leaf_value"])
      index = {n.path: j for j, n in enumerate(tree.nodes)}
      out.append(Solution(game, sigma, f[f"s{i}_tau"], float(f[f"s{i}_z"]), {}, {},
                          tuple(int(c) for c in f[f"s{i}_board"]), pi, payoff, index,
                          f[f"s{i}_leaf_mass"]))
  return out


def solves_dir(log: str) -> str:
  """Where the solves of the JSONL log ``log`` are saved."""
  return os.path.splitext(log)[0] + ".solves"


# ── Replay ───────────────────────────────────────────────────────────────────


def evaluate_hand(agent, aivat: ResolvingAivat, rec: dict, solves: list[Solution]) -> dict:
  """One hand's record of AIVAT terms, from its log record and saved solves."""
  res, model, scale = agent.resolver, agent.model, agent.model.scale
  seat = int(rec["seat"])
  ours = [card_index(c) for c in rec["hole_cards"]]
  bot = [card_index(c) for c in rec["bot_hole_cards"]]
  h = int(HAND_INDEX[bot[0], bot[1]])
  steps = rec["resolving"]["steps"]
  tokens = [k if to is None else f"b{to}" for k, to in tokenize(rec["resolving"]["real_action"])]
  if len(tokens) != len(steps):
    raise ValueError(f"hand {rec['hand']}: {len(tokens)} tokens, {len(steps)} steps")

  res.new_hand(seat, ours)
  model.new_hand()
  aivat.new_hand(bot)
  game, actions, strings, prob_gap = RealGame(), "", [], 0.0
  for step, token in zip(steps, tokens):
    history, board = step["history"], step["board"]
    actor, atom = int(step["actor"]), int(step["atom"])
    probs = np.asarray(step["probs"], np.float64)
    if actor == seat:
      # The solve this decision was played from, as resolver.probs left it.
      res.solution = solves[step["solve"]]
      if not res.solves or res.solves[-1] is not res.solution:
        res.solves.append(res.solution)
    else:
      # Must match the logged marginal: same model, same strings so far.
      p = model.probs([history], [], board, seat)[0, h]
      prob_gap = max(prob_gap, float(np.abs(p - probs).max()))
    aivat.step(actor, history, board, [c / scale for c in game.committed], probs, atom)

    # The token is revealed only now; see the module docstring.
    nxt = _advance(game, actions, token)
    if nxt is None:
      raise ValueError(f"hand {rec['hand']}: token {token} is illegal after {actions}")
    game, actions = nxt
    strings.append(actions)
    model.set_real([*history, atom], strings)

  payoff = rec["winnings"] / scale
  a = aivat.finish(payoff, ended_by_fold=tokens[-1] == "f",
                   final_committed=[c / scale for c in game.committed])
  seconds = a.pop("seconds")
  return {"hand": rec["hand"], "seat": seat, "payoff": payoff, "aivat": a, "aivat_seconds": seconds,
          "prob_gap": prob_gap}


# ── CLI ──────────────────────────────────────────────────────────────────────


def _report(args) -> None:
  recs = [json.loads(line) for path in args.report for line in open(path)]
  with open(os.path.join(args.run_dir, "config.yaml")) as f:
    mbb = 1e3 / float(yaml.safe_load(f)["env"]["big_blind"])
  print(f"{len(recs)} hands from {len(args.report)} logs")
  report_aivat(recs, mbb)
  print(f"  largest gap between recomputed and logged Slumbot probabilities: "
        f"{max(r['prob_gap'] for r in recs):.2e}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("run_dir")
  ap.add_argument("logs", nargs="*", help="JSONL logs of play_slumbot.py --save-solves")
  ap.add_argument("--checkpoint")
  ap.add_argument("--params", default="params")
  ap.add_argument("--depth", type=int, default=4, help="as in play")
  ap.add_argument("--leaf-games", type=int, default=128)
  ap.add_argument("--cfr-iters", type=int, default=500)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--task", type=int, default=0, help="this task handles hand % tasks == task")
  ap.add_argument("--tasks", type=int, default=1)
  ap.add_argument("--out", help="JSONL of the hands' AIVAT terms")
  ap.add_argument("--report", nargs="+", metavar="LOG", help="only pool and report these")
  args = ap.parse_args()
  if args.report:
    _report(args)
    return
  if not args.out or not args.logs:
    ap.error("give the play logs and --out, or --report")

  # Imported here because the agent imports this module to save its solves.
  from .agent import SlumbotResolvingAgent
  agent = SlumbotResolvingAgent(args.run_dir, args.checkpoint, args.params,
                                equity.preflop_ev_matrix(n_boards=200_000, progress=False),
                                depth=args.depth, leaf_games=args.leaf_games,
                                iters=args.cfr_iters, seed=args.seed)
  aivat = ResolvingAivat(agent.resolver)
  rules = agent.env.rules
  done, skipped = 0, 0
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with open(args.out, "w") as out:
    for log in args.logs:
      for line in open(log):
        rec = json.loads(line)
        if rec["hand"] % args.tasks != args.task:
          continue
        path = os.path.join(solves_dir(log), f"hand_{rec['hand']:05d}.npz")
        # Hands where Slumbot folds before we act have no solves and no file. They
        # must stay in: dropping them selects on Slumbot's action.
        played = any(s["solve"] is not None for s in rec["resolving"]["steps"])
        # A lost hand has no consistent tree to evaluate, and none occurred so far.
        if rec["tree_lost"] or rec["resolving"]["lost"] or not rec.get("bot_hole_cards") \
            or (played and not os.path.exists(path)):
          skipped += 1
          print(f"hand {rec['hand']}: skipped", flush=True)
          continue
        solves = load_hand_solves(path, rules, int(rec["seat"])) if played else []
        r = evaluate_hand(agent, aivat, rec, solves)
        out.write(json.dumps(r) + "\n")
        out.flush()
        done += 1
        print(f"hand {rec['hand']:5d} seat {r['seat']} {r['payoff']:+8.1f} chips, "
              f"AIVAT {r['aivat_seconds']:.0f}s, prob gap {r['prob_gap']:.1e}", flush=True)
  print(f"{done} hands evaluated, {skipped} skipped; model desyncs {agent.model.stats['desync']}, "
        f"refusals {len(agent.model.stats['failures'])}")
