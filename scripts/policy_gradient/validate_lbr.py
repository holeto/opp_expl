"""Validate local best response (``slumbot/lbr.py``) against the QMMD blueprint, locally.

LBR plays the blueprint in the env with ``OpponentModel`` (the blueprint itself)
as its opponent model: the same ``_LBR`` decision code that plays Slumbot, with
the env as the game. The env deals the opponent's hand, so at every LBR
decision the validator knows the true hand ``h*``; LBR never sees it.

  exact   At the first ``--exact`` decisions, everything LBR used is recomputed
          without ``OpponentModel``, ``showdown_sign`` or ``lbr_values``:
            - the range: compatibility × the opponent's probability of each of
              its actions, by stepping the env for all 1326 hands on the true
              board (the model replays padded paths on a filler board and folds
              struck duplicates through the tree's alias; here an atom counts
              with every atom that leads to the same chips);
            - the opponent's fold probability after each candidate bet;
            - wp, by enumerating every runout with ``hand_rank7`` (postflop;
              preflop LBR uses the Monte Carlo EV matrix, taken as is);
            - the chips, from the env, and U from the paper's formula.
  calib   The posterior must be calibrated against ``h*``. Each test sums terms
          that have mean 0 given the public history if the posterior is right.
          Terms are summed per hand, since one hand's terms share ``h*``, and
          the per-hand sums are z-tested across hands:
            - log-likelihood: log π(h*) − Σ_h π(h) log π(h);
            - win probability: s(h*) − Σ_h π(h) s(h), with ``s`` the expected
              showdown sign that LBR's wp is made of;
            - fold probability: 1[the opponent folded] − fp, after each bet LBR made.
          Binned tables follow (rough: they ignore the dependence within a hand).
  --logs  The calibration tests on ``SlumbotLBRAgent`` play logs, which score each
          decision's range at Slumbot's hand once it is shown.

  uv run python scripts/policy_gradient/validate_lbr.py checkpoints/qmmd_hunl_560k --hands 1000
  uv run python scripts/policy_gradient/validate_lbr.py --logs LOG.jsonl ...
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from oppex.cfr import equity
from oppex.cfr.cards import HAND_CARDS, HAND_INDEX, N_HANDS
from oppex.cfr.subgame import expand_node, replay_betting
from oppex.envs.betting import CALL, FOLD
from oppex.envs.hunl_holdem import hand_rank7
from oppex.policy_gradient.blueprint import Blueprint, sort_hole
from oppex.policy_gradient.resolving import OpponentModel, _view
from oppex.policy_gradient.slumbot.lbr import _LBR

_VISIBLE = (0, 3, 4, 5)


class EnvLBRAgent(_LBR, Blueprint):
  """LBR against the blueprint in the env.

  The env is the game, so the real chips are the env's (× 50, Slumbot's
  scale), and the model is exact. ``Blueprint.policy`` stays the network's
  policy, which plays the opponent; LBR decides through ``decide``.
  """

  def __init__(self, run_dir, ev, bets: str, call_streets: int) -> None:
    Blueprint.__init__(self, run_dir)
    self.ev, self.call_streets = ev, int(call_streets)
    self.model = OpponentModel(self)
    self._init_lbr(bets)

  def decide(self, s, seat) -> tuple[int, bool]:
    # Our atom, and whether LBR decided it (else a forced check/call).
    if int(s.street) < self.call_streets:
      return CALL, False
    probs, _ = self.lbr_policy(s, seat)
    return int(np.argmax(probs)), True

  def _real_chips(self, s, seat, atoms):
    history, _ = _view(s, seat)
    bs = replay_betting(self.env.rules, history)[-1][0]
    _, _, children, _ = expand_node(self.env.rules, bs)
    c = np.asarray(s.committed, np.float64)
    if not np.allclose(c, np.asarray(bs.committed)):
      raise AssertionError(f"env and tree chips differ after {history}")
    k = self._lbr_scale
    return c[seat] * k, c[1 - seat] * k, {a: float(children[a].committed[seat]) * k
                                          for a in atoms}


# ── Play ─────────────────────────────────────────────────────────────────────


def same_chips(bp: Blueprint, s, atom: int) -> list[int]:
  """The legal atoms leading to the same committed chips and end flag as
  ``atom``: one public action, whichever label the actor used."""
  legal = np.flatnonzero(bp.legal(s, int(s.cur_player)))
  want = bp.apply(s, atom)
  return [int(a) for a in legal
          if np.array_equal(np.asarray((got := bp.apply(s, int(a))).committed),
                            np.asarray(want.committed)) and bool(got.done) == bool(want.done)]


def _new_played() -> dict:
  # Agreement of LBR's model with the opponent's real policy on the played
  # path. vec: max |Δp| over atoms at each opponent decision; act: Δ of the
  # taken public action's probability; fold: Δ of the fold probability after
  # each bet LBR made; range: |log r(h*) − log Π p_real| at LBR decisions.
  return {"vec": [], "act": [], "fold": [], "range": [], "worst": (0.0, None)}


def play(agent: EnvLBRAgent, n_hands: int, seed: int,
         sorted_opponent: bool = True) -> tuple[list[list[dict]], dict]:
  """LBR against the blueprint: per hand the record of each LBR decision, and
  the played-state check (``_new_played``).

  Seats alternate. A record holds LBR's inputs, the state, the history and the
  true hand; after a bet it also holds whether the opponent folded to it.
  """
  env = agent.env
  hands, played = [], _new_played()
  t0 = time.time()
  for i in range(n_hands):
    s = env.init_state(jax.random.fold_in(jax.random.PRNGKey(seed), i))
    seat, opp = i % 2, 1 - i % 2
    if sorted_opponent:
      s = sort_hole(s, opp)
    hole = np.asarray(s.hole_cards)
    cards = [int(c) for c in hole[seat]]
    h_true = int(HAND_INDEX[hole[opp, 0], hole[opp, 1]])
    rng = np.random.default_rng([seed, i])
    s0, recs, pending = s, [], None
    # log of the real probability of the opponent's public actions so far.
    log_real = 0.0
    while not bool(s.done):
      history, board = _view(s, seat)
      if int(s.cur_player) == seat:
        atom, decided = agent.decide(s, seat)
        if decided:
          rec = dict(agent.lbr_inputs, state=s, s0=s0, seat=seat, h=h_true,
                     history=history, board=board, cards=cards)
          recs.append(rec)
          played["range"].append(abs(np.log(rec["range"][h_true]) - log_real))
          # The opponent's next action answers this bet.
          pending = rec if atom in rec["bets"] else None
      else:
        # The opponent's real distribution, which it samples from, against
        # the row of LBR's model at the true hand.
        p, _ = Blueprint.policy(agent, s, opp)
        q = agent.model.probs([tuple(history)], cards, board, seat)[0, h_true]
        d = float(np.abs(p - q).max())
        played["vec"].append(d)
        if d > played["worst"][0]:
          played["worst"] = (d, history)
        atom = int(rng.choice(len(p), p=p / p.sum()))
        same = same_chips(agent, s, atom)
        played["act"].append(abs(float(p[same].sum() - q[same].sum())))
        log_real += np.log(p[same].sum())
        if pending is not None:
          played["fold"].append(abs(float(pending["bets"][pending["atom"]][1][h_true] - p[FOLD])))
          pending["folded"] = atom == FOLD
          pending = None
      s = agent.apply(s, atom)
    hands.append(recs)
    if (i + 1) % 100 == 0:
      n = sum(len(h) for h in hands)
      print(f"  {i + 1} hands, {n} LBR decisions, {time.time() - t0:.0f} s", flush=True)
  return hands, played


def check_played(played: dict, tol: float = 1e-5) -> None:
  print("\n== played-state check: LBR's model at the true hand vs the opponent's real policy")
  for key, what in (("vec", "max |Δp| over atoms, per opponent decision"),
                    ("act", "|Δ| of the taken action's probability"),
                    ("fold", "|Δ| of fp after each bet LBR made"),
                    ("range", "|log r(h*) − log Π p_real| per LBR decision")):
    x = np.asarray(played[key])
    if len(x):
      print(f"  {what}: {len(x)} cases, max {x.max():.1e}, mean {x.mean():.1e}, "
            f"above {tol:g}: {int((x > tol).sum())}")
  d, hist = played["worst"]
  print(f"  worst vector difference {d:.1e} after {hist}")


# ── Independent recomputation ────────────────────────────────────────────────


class Independent:
  """The range, fold probabilities and wp of one decision, from the env and
  ``hand_rank7`` directly."""

  def __init__(self, bp: Blueprint) -> None:
    env = bp.env
    self.bp, self.env = bp, env
    probs_h = jax.jit(jax.vmap(bp.probs_fn, in_axes=(None, 0, None)))
    self._probs = lambda s, seat: probs_h(bp.params, s, seat)[0]
    self._apply = jax.jit(jax.vmap(
      lambda s, a: env.apply_action(s, jnp.full((2,), a, jnp.int32))[0], in_axes=(0, None)))
    self._rank = jax.jit(jax.vmap(hand_rank7))

  def _states(self, s0, opp):
    # The hand's initial state once per opponent hand, on the true board.
    st = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (N_HANDS,) + x.shape), s0)
    hole = jnp.broadcast_to(jnp.asarray(s0.hole_cards), (N_HANDS, 2, 2))
    return st._replace(hole_cards=hole.at[:, opp].set(jnp.asarray(HAND_CARDS, jnp.int32)))

  def range_and_folds(self, rec) -> tuple[np.ndarray, dict]:
    """The unnormalised range at the decision, and ``{bet: (H,) fold prob}``."""
    seat, opp = rec["seat"], 1 - rec["seat"]
    states, s_ref = self._states(rec["s0"], opp), rec["s0"]
    lik = np.ones(N_HANDS)
    for atom in rec["history"]:
      if int(s_ref.cur_player) == opp:
        p = np.asarray(self._probs(states, jnp.int32(opp)), np.float64)
        lik *= p[:, same_chips(self.bp, s_ref, atom)].sum(-1)
      states = self._apply(states, jnp.int32(atom))
      s_ref = self.bp.apply(s_ref, atom)
    known = set(rec["cards"]) | set(rec["board"])
    compat = np.array([not ({int(a), int(b)} & known) for a, b in HAND_CARDS], np.float64)
    folds = {}
    for a in rec["bets"]:
      after = self._apply(states, jnp.int32(a))
      folds[a] = np.asarray(self._probs(after, jnp.int32(opp)), np.float64)[:, FOLD]
    return compat * lik, folds

  def _rank_rows(self, cards7: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    # hand_rank7 of every row, in fixed-size chunks (one compilation, bounded
    # memory); the last chunk is padded with copies of the first row.
    out = []
    for lo in range(0, len(cards7), chunk):
      part = cards7[lo: lo + chunk]
      pad = np.concatenate([part, np.repeat(part[:1], chunk - len(part), 0)])
      out.append(np.asarray(self._rank(jnp.asarray(pad, jnp.int32)))[: len(part)])
    return np.concatenate(out)

  def win(self, rec) -> np.ndarray | None:
    """``(H,)`` P(win) + P(tie) / 2 against each hand over all runouts, or
    None preflop."""
    board, cards = rec["board"], rec["cards"]
    if not board:
      return None
    deck = [c for c in range(52) if c not in set(cards) | set(board)]
    runs = list(itertools.combinations(deck, 5 - len(board)))
    boards = np.array([list(board) + list(r) for r in runs], np.int32).reshape(len(runs), 5)
    n_r = len(runs)
    ours = np.asarray(self._rank(jnp.asarray(
      np.concatenate([np.tile(cards, (n_r, 1)), boards], 1))))
    # Every (runout, hand) pair in one call; pairs sharing a card are masked.
    pairs = np.concatenate([np.repeat(boards, N_HANDS, 0), np.tile(HAND_CARDS, (n_r, 1))], 1)
    theirs = self._rank_rows(pairs).reshape(n_r, N_HANDS)
    valid = ~(boards[:, :, None, None] == HAND_CARDS[None, None, :, :]).any((1, 3))
    known = set(cards) | set(board)
    compat = np.array([not ({int(a), int(b)} & known) for a, b in HAND_CARDS])
    score = (ours[:, None] > theirs) + 0.5 * (ours[:, None] == theirs)
    n = valid.sum(0)
    out = np.where(compat & (n > 0), (score * valid).sum(0) / np.maximum(n, 1), 0.0)
    return out


def check_exact(ind: Independent, recs: list[dict], ev) -> None:
  print(f"\n== exact: {len(recs)} decisions recomputed independently")
  worst = {"range": 0.0, "fold": 0.0, "wp": 0.0, "U": 0.0}
  for k, rec in enumerate(recs):
    r, folds = ind.range_and_folds(rec)
    pi_m = rec["range"] / rec["range"].sum()
    pi_i = r / r.sum()
    d_range = float(np.abs(pi_m - pi_i).sum())
    d_fold = max([float(np.abs(rec["bets"][a][1] - folds[a]).max()) for a in folds] + [0.0])

    win = ind.win(rec)
    if win is None:
      win = (1.0 + np.asarray(ev[HAND_INDEX[rec["cards"][0], rec["cards"][1]]])) / 2.0
    d_wp = float(np.abs(pi_i @ win - pi_m @ ((1.0 + rec["sign"]) / 2.0)))

    # U from Figure 1 with the independent inputs.
    pot, asked = rec["pot"], rec["asked"]
    wp = pi_i @ win
    values = {CALL: wp * pot - (1 - wp) * asked}
    for a, (raise_by, _) in rec["bets"].items():
      fp = float(pi_i @ folds[a])
      rest = pi_i * (1 - folds[a])
      wp2 = rest @ win / rest.sum() if rest.sum() > 0 else 0.0
      values[a] = fp * pot + (1 - fp) * (wp2 * (pot + raise_by) - (1 - wp2) * (asked + raise_by))
    d_u = max(abs(values[a] - rec["values"][a]) for a in values) / pot
    best = max(values, key=values.get)
    agree = (best if values[best] > 0 else (FOLD if asked > 0 else CALL)) == rec["atom"]
    for key, d in (("range", d_range), ("fold", d_fold), ("wp", d_wp), ("U", d_u)):
      worst[key] = max(worst[key], d)
    print(f"  {k:3d} street {_VISIBLE.index(len(rec['board']))} history {rec['history']}: "
          f"|Δπ|₁ {d_range:.1e}  max|Δfold| {d_fold:.1e}  |Δwp| {d_wp:.1e}  "
          f"max|ΔU|/pot {d_u:.1e}  same action {agree}")
  print("  worst: " + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))


# ── Calibration ──────────────────────────────────────────────────────────────


def _ztest(per_hand: list[float], name: str) -> None:
  x = np.asarray(per_hand)
  if len(x) < 2:
    print(f"  {name}: too few hands")
    return
  se = x.std(ddof=1) / np.sqrt(len(x))
  if np.abs(x).max() < 1e-6:
    # A pure opponent (Slumbot) leaves a uniform posterior on the consistent
    # hands, so every log-likelihood term is 0 and only π(h*) > 0 is tested.
    print(f"  {name:16} {len(x):5d} hands  every term 0 (uniform posteriors): not informative")
    return
  z = x.mean() / se if se > 0 else float("nan")
  print(f"  {name:16} {len(x):5d} hands  mean per hand {x.mean():+.4f} ± {1.96 * se:.4f}  "
        f"z {z:+.2f}")


def _bins(pred, obs, var, name: str) -> None:
  edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
  pred, obs, var = map(np.asarray, (pred, obs, var))
  print(f"  {name}: bin, n, predicted, observed, z (decisions treated as independent)")
  for lo, hi in zip(edges, edges[1:]):
    m = (pred >= lo) & (pred < hi)
    if m.sum() == 0:
      continue
    z = (obs[m].sum() - pred[m].sum()) / max(np.sqrt(var[m].sum()), 1e-12)
    print(f"    [{lo:.1f}, {min(hi, 1):.1f})  {int(m.sum()):6d}  {pred[m].mean():.4f}  "
          f"{obs[m].mean():.4f}  {z:+.2f}")


def check_calibration(hands: list[list[dict]]) -> None:
  n_dec = sum(len(h) for h in hands)
  print(f"\n== calib: {len(hands)} hands, {n_dec} LBR decisions")
  ll, rank_t, wp_t, fp_t, ranks = [], [], [], [], []
  wp_pred, wp_obs, wp_var, fp_pred, fp_obs, fp_var = [], [], [], [], [], []
  zero = 0
  rng = np.random.default_rng(0)
  for recs in hands:
    s_ll = s_rank = s_wp = s_fp = 0.0
    n_fp = 0
    for rec in recs:
      pi = rec["range"] / rec["range"].sum()
      h = rec["h"]
      if pi[h] <= 0:
        zero += 1
        continue
      nz = pi > 0
      s_ll += np.log(pi[h]) - float(pi[nz] @ np.log(pi[nz]))
      # Randomised rank of π(h*) within π: uniform on [0, 1) if h* ~ π, and
      # bounded, so unlike the log-likelihood it has no heavy tail.
      u = float(pi[pi < pi[h]].sum() + rng.random() * pi[pi == pi[h]].sum())
      s_rank += u - 0.5
      ranks.append(u)
      win = (1.0 + rec["sign"]) / 2.0
      wp = float(pi @ win)
      s_wp += win[h] - wp
      wp_pred.append(wp)
      wp_obs.append(win[h])
      wp_var.append(float(pi @ win ** 2) - wp ** 2)
      if "folded" in rec:
        fp = float(pi @ rec["bets"][rec["atom"]][1])
        s_fp += float(rec["folded"]) - fp
        n_fp += 1
        fp_pred.append(fp)
        fp_obs.append(float(rec["folded"]))
        fp_var.append(fp * (1 - fp))
    if recs:
      ll.append(s_ll)
      rank_t.append(s_rank)
      wp_t.append(s_wp)
    if n_fp:
      fp_t.append(s_fp)
  print(f"  true hand with posterior probability 0: {zero} of {n_dec} decisions")
  _ztest(ll, "log-likelihood")
  _ztest(rank_t, "rank of π(h*)")
  counts = np.histogram(ranks, bins=10, range=(0, 1))[0]
  print(f"  rank deciles (each expected {len(ranks) / 10:.0f}): {counts.tolist()}")
  _ztest(wp_t, "win probability")
  _ztest(fp_t, "fold probability")
  _bins(wp_pred, wp_obs, wp_var, "win probability")
  _bins(fp_pred, fp_obs, fp_var, "fold probability after our bets")


def check_logs(paths: list[str]) -> None:
  """The calibration tests on play logs of ``SlumbotLBRAgent`` (LocalSlumbot or
  slumbot.com), from the range scores it logs once Slumbot's cards are shown."""
  ll, wp_t, fp_t = [], [], []
  wp_pred, wp_obs, wp_var, fp_pred, fp_obs, fp_var = [], [], [], [], [], []
  zero = n_dec = n_hands = 0
  for path in paths:
    for line in open(path):
      rec = json.loads(line)
      n_hands += 1
      steps, seat = rec["resolving"]["steps"], int(rec["seat"])
      s_ll = s_wp = s_fp = 0.0
      n_cal = n_fp = 0
      for i, st in enumerate(steps):
        lbr = st.get("lbr") if st["actor"] == seat else None
        if not lbr or "calib" not in lbr:
          continue
        c = lbr["calib"]
        n_dec += 1
        if c["pi_h"] <= 0:
          zero += 1
          continue
        s_ll += np.log(c["pi_h"]) - c["neg_entropy"]
        s_wp += c["win_h"] - c["wp"]
        n_cal += 1
        wp_pred.append(c["wp"])
        wp_obs.append(c["win_h"])
        wp_var.append(c["var_win"])
        choice = lbr["choice"]
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if choice in lbr["fp_wp"] and choice != "call" and nxt and nxt["actor"] != seat:
          fp = lbr["fp_wp"][choice][0]
          folded = float(nxt["atom"] == FOLD)
          s_fp += folded - fp
          n_fp += 1
          fp_pred.append(fp)
          fp_obs.append(folded)
          fp_var.append(fp * (1 - fp))
      if n_cal:
        ll.append(s_ll)
        wp_t.append(s_wp)
      if n_fp:
        fp_t.append(s_fp)
  print(f"\n== logs: {len(paths)} files, {n_hands} hands, {n_dec} scored LBR decisions")
  print(f"  true hand with posterior probability 0: {zero}")
  _ztest(ll, "log-likelihood")
  _ztest(wp_t, "win probability")
  _ztest(fp_t, "fold probability")
  _bins(wp_pred, wp_obs, wp_var, "win probability")
  _bins(fp_pred, fp_obs, fp_var, "fold probability after our bets")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir", nargs="?")
  ap.add_argument("--logs", nargs="+", metavar="LOG",
                  help="instead of playing, run the calibration tests on SlumbotLBRAgent logs")
  ap.add_argument("--hands", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--bets", choices=("fc", "fcpa", "bins"), default="bins",
                  help="bins (default): every size, so that raises and folds are exercised")
  ap.add_argument("--call-streets", type=int, default=0)
  ap.add_argument("--exact", type=int, default=30, help="decisions recomputed independently")
  ap.add_argument("--as-dealt", action="store_true",
                  help="leave the opponent's hole cards in deal order; with the 560k "
                       "checkpoint, which is not card-order invariant, OpponentModel then "
                       "models the wrong order in half the deals (blueprint.sort_hole)")
  args = ap.parse_args()
  if args.logs:
    check_logs(args.logs)
    return
  if not args.run_dir:
    ap.error("give a run directory, or --logs")

  ev = equity.preflop_ev_matrix(n_boards=200_000, progress=False)
  agent = EnvLBRAgent(args.run_dir, ev, args.bets, args.call_streets)
  print(f"{agent.checkpoint}: LBR {args.bets}, call streets {args.call_streets}, "
        f"{args.hands} hands, seed {args.seed}", flush=True)
  hands, played = play(agent, args.hands, args.seed, not args.as_dealt)
  print(agent.lbr_summary())

  # Spread the exact checks over the streets: decisions in hand order,
  # round-robin by street.
  by_street = [[r for h in hands for r in h if len(r["board"]) == n] for n in _VISIBLE]
  picked = [r for group in itertools.zip_longest(*by_street) for r in group if r is not None]
  check_exact(Independent(agent), picked[: args.exact], ev)
  check_played(played)
  check_calibration(hands)


if __name__ == "__main__":
  main()
