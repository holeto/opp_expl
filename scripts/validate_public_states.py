"""The infoset closure must reproduce the public states of the domain.

  uv run python scripts/validate_public_states.py

For every game the closure (``oppex.tree.public``, the general algorithm)
is compared with the grouping of histories by ``env.public_state`` bytes:
same results and draw cards in Goofspiel, same betting sequence and board in
HUNL. Games:

* Goofspiel 3 and 4 with descending prizes, and Goofspiel 3 with a random
  (publicly revealed) prize order, which adds a chance root. Here the closure
  is the **common-knowledge** partition and is strictly finer than the
  observations. After results (P1 won, P1 won) in Goofspiel 3, the history
  where player 0 played 1 then 2 forces player 1's 2 then 3, so both
  infosets are singletons, and nobody confuses it with the mirrored history
  (2, 1 against 3, 2) that shows the same results. So for Goofspiel the
  check is that every closure class lies inside one observation class, and
  that both partitions give the same per-infoset counterfactual values and
  reaches under a random profile: both are valid public trees for the same
  infoset game. (Comparing CFR solutions instead is ill-posed: Goofspiel has
  exact ties, and a 1e-16 difference in summation order flips regret matching
  between uniform and pure within two iterations);
* push-fold HUNL at 10 BB on one fixed board, dealt every ordered pair of
  three disjoint hands, so every two deals are linked through a shared hand
  and the closure must equal the betting-sequence partition;
* a negative control, the same HUNL dealt only two pairs with no hand in
  common. Nothing links the two deals, so the closure must be strictly finer,
  splitting every betting sequence into one class per deal.

Each game is also converted with ``to_public_game``, and the general PSCFR's
root value of a random profile must equal the history-level value.
"""

from __future__ import annotations

import itertools
import time

import jax

# The shared CFR update rule is JAX, which truncates to float32 without x64.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from oppex.cfr.general import pscfr  # noqa: E402
from oppex.envs import build_env  # noqa: E402
from oppex.tree import extract_game_tree  # noqa: E402
from oppex.tree.history_game import expected_value, from_layered, strategy_dicts  # noqa: E402
from oppex.tree.public import (  # noqa: E402
  partition_by_observation, public_partition, to_public_game,
)


def hunl_deals(env, hands, board, pairs):
  # Initial states for the given ordered hand pairs, uniform over them.
  base = env.init_state(jax.random.PRNGKey(0))
  out = []
  for a, b in pairs:
    st = base._replace(hole_cards=jnp.array([hands[a], hands[b]], jnp.int32),
                       board=jnp.array(board, jnp.int32))
    out.append((st, 1.0 / len(pairs)))
  return out


def as_partition(game, groups):
  # Observation groups in the `public_partition` format; nodes outside every
  # group (the chance root) become classes of their own.
  of_node = np.full(game.n_nodes, -1)
  states = [sorted(g) for g in groups]
  for c, g in enumerate(states):
    of_node[g] = c
  for n in range(game.n_nodes):
    if of_node[n] < 0 and game.kind[n] != 2:
      of_node[n] = len(states)
      states.append([n])
  return states, of_node


def random_profile(game, seed=0):
  # A strictly positive random behaviour strategy over legal actions.
  rng = np.random.default_rng(seed)
  out = strategy_dicts(game)
  for p in (0, 1):
    for k, v in out[p].items():
      w = (rng.random(v.size) + 0.1) * (v > 0)
      out[p][k] = w / w.sum()
  return out


def same_values(game, part_a, part_b):
  # PSCFR traversals of two public trees of one game under a random profile:
  # the largest difference in any infoset's action values, cfv or reach.
  strat = random_profile(game)
  out = []
  for part in (part_a, part_b):
    pg = to_public_game(game, part)
    lay = pscfr.layout(pg)
    sig = np.zeros(lay.legal.shape)
    for (s, p), lo in lay.rows.items():
      for j, k in enumerate(pg.states[s].keys[p]):
        sig[lo + j] = strat[p][k]
    tv = pscfr.traverse(pg, sig, np.zeros(lay.t_legal.shape))
    out.append({(p, k): np.concatenate([tv.q[s][p][j], [tv.cfv[s][p][j], tv.reach[s][p][j]]])
                for (s, p) in lay.rows for j, k in enumerate(pg.states[s].keys[p])})
  assert out[0].keys() == out[1].keys()
  return max(np.abs(out[0][k] - out[1][k]).max() for k in out[0])


def check(name, env, initial_states=None, expect="equal"):
  # Compares the two partitions and the uniform-profile value; returns a row.
  t0 = time.time()
  layered = extract_game_tree(env, initial_states)
  game = from_layered(layered)
  states, _ = public_partition(game)
  obs = set(partition_by_observation(layered, env, game))
  # Classes holding a node without an env state (the chance root) are skipped.
  stateless = {n for n in range(game.n_nodes)
               if layered.layers[game.ref[n][0]].state_batch is None}
  clo = {frozenset(c) for c in states if not set(c) & stateless}

  if expect == "equal":
    ok = clo == obs
  elif expect == "refines":
    ok = all(any(c <= o for o in obs) for c in clo)
  else:
    # Strictly finer: every closure class inside one observation class, and
    # at least one observation class split.
    inside = all(any(c <= o for o in obs) for c in clo)
    ok = inside and len(clo) > len(obs)

  pg = to_public_game(game)
  strat = random_profile(game)
  lay = pscfr.layout(pg)
  sig = np.zeros(lay.legal.shape)
  for (s, p), lo in lay.rows.items():
    for j, k in enumerate(pg.states[s].keys[p]):
      sig[lo + j] = strat[p][k]
  v_ps, _ = pscfr.root_value(pg, sig, np.zeros(lay.t_legal.shape))
  v = expected_value(game, strat)
  v_h = sum(game.prob[r] * v[r] for r in game.roots)
  ok_v = abs(v_ps - v_h) < 1e-9 * max(1.0, abs(v_h))
  d_sol = same_values(game, public_partition(game), as_partition(game, obs))
  ok_v = ok_v and d_sol < 1e-9
  print(f"  {name:34s} {game.n_nodes:>7d} nodes  closure {len(clo):>5d}  "
        f"observation {len(obs):>5d}  ({expect}: {'ok' if ok else 'FAIL'})  "
        f"value {v_ps:+.6f} vs history {v_h:+.6f}  per-infoset values {d_sol:.0e}  "
        f"{time.time() - t0:.1f}s", flush=True)
  return ok and ok_v


def main():
  results = []
  print("Goofspiel")
  for n, order in ((3, "descending"), (4, "descending"), (3, "random")):
    env = build_env({"name": "goofspiel", "n_cards": n, "prize_order": order})
    results.append(check(f"Goofspiel {n} {order}", env, expect="refines"))

  print("HUNL push-fold, 10 BB, fixed board")
  env = build_env({"name": "hunl", "starting_stack": 20, "small_blind": 1, "big_blind": 2,
                   "num_bet_bins": 0, "drop_duplicate_allin": True})
  # Card c is rank c // 4 and suit c % 4; three disjoint hands and a board.
  hands = [(48, 49), (44, 41), (0, 5)]
  board = (10, 15, 20, 26, 31)
  linked = list(itertools.permutations(range(3), 2))
  results.append(check("6 deals over 3 hands (linked)", env,
                       hunl_deals(env, hands, board, linked)))
  disjoint = [(48, 49), (44, 41), (0, 5), (12, 17)]
  results.append(check("2 deals, no shared hand (control)", env,
                       hunl_deals(env, disjoint, board, [(0, 1), (2, 3)]), expect="finer"))
  assert all(results), "the closure does not reproduce the domain's public states"
  print("All checks passed.")


if __name__ == "__main__":
  main()
