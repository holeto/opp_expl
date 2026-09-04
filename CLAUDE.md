# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhD research code on opponent exploitation in imperfect-information games. Pure-JAX
environments plus a public-state CFR solver built on them. Early stage: the env layer
(`oppex.envs`) and a preflop push-fold CFR solver (`oppex.cfr`) exist; RL training does
not yet.

## Commands

Dependencies and the virtualenv are managed by `uv` (Python 3.12.1 pinned in `.python-version`):

```bash
uv sync                    # create/refresh .venv from uv.lock, install project editable
uv add <pkg>               # add a dependency (updates pyproject.toml + uv.lock)
uv run python -c "..."     # run anything inside the project env
```

The distribution is `opponent-exploitation` but the importable package is **`oppex`**
(`src/oppex/`), decoupled via `[tool.uv.build-backend] module-name` in `pyproject.toml`.
Always use absolute imports — `from oppex.envs import build_env`.

No test runner, linter, or CLI entry point is wired up yet. JAX resolves to the CPU build
here; a CUDA jaxlib is not installed.

## Architecture

### Layering

Docstrings refer to numbered layers ("Layer 3 — Environments" in
[base.py](src/oppex/envs/base.py)); the lower layers are not written yet. Treat that
numbering as the intended structure, not as existing code. New layers go in as sibling
subpackages of `oppex.envs`.

An earlier design paired the env with a hybrid discrete–continuous policy (a Gaussian
bet-sizing head trained by an MMD loss in `losses/mmd_cont.py`). **That approach was
dropped** in favour of explicit discrete pot-fraction bet bins. If you find references to
`mmd_cont`, mixture components, or a continuous bet-size log-density, they are stale —
delete them rather than building on them.

### The JAX purity contract

Everything under [src/oppex/envs/](src/oppex/envs/) must be jit/vmap/scan-safe. This is the
single constraint that shapes most of the code:

- State is a `NamedTuple` (a pytree) threaded in and out of pure functions. No Python-side
  mutation, no callbacks, no host branching on traced values.
- All shapes are static. Buffers such as `action_history` / `bet_history` are preallocated
  to `env.max_length`, so `max_length` must be a Python int known at construction.
- Control flow is `jnp.where` over eagerly computed branches — every branch is always
  evaluated. Look at `apply_action` and `_terminal_rewards` for the idiom.
- `player_id` is passed as a JAX scalar so per-player logic can be vmapped across seats.

### `Env` interface ([base.py](src/oppex/envs/base.py))

Two orthogonal axes of observation, six abstract methods:

|            | snapshot                | perfect-recall history   |
| ---------- | ----------------------- | ------------------------ |
| private    | `player_observation`    | `information_set`        |
| public     | `public_observation`    | `public_state`           |
| privileged | `state_observation`     | `state_representation`   |

The history variants append encoded action/bet histories to the corresponding snapshot,
so they uniquely identify an information set / full trajectory. The privileged row is
ground truth (both players' hole cards) — for value functions with hidden information and
for debugging, never as agent input.

Every observation method takes a `PRNGKey` even when unused; it is reserved for wrappers
that inject observation noise. `apply_action` deliberately takes **no** key: all chance is
resolved once in `init_state` and revealed progressively by the observation methods, so
transitions are deterministic given the state.

### Sequential vs simultaneous games

`current_player(state)` returns `-1` for simultaneous games (the base-class default) and a
player index for sequential ones. Sequential envs must return a `legal_actions` mask with
exactly one `True` entry for non-acting players. That single legal action forces a
deterministic policy distribution, which zeroes the policy gradient for that step and lets
the training loop stay branch-free.

### HUNL Hold'em ([hunl_holdem.py](src/oppex/envs/hunl_holdem.py))

Heads-up no-limit Texas Hold'em, 2 players, zero-sum.

- **Actions** are fully discrete: a plain `(num_players,)` array of atom indices, of which
  only `current_player`'s entry is read. Atoms are `[FOLD, CALL, ALL_IN, BET_0 … BET_{K-1}]`
  — the *last* K entries are bet atoms, one per `bet_bins` entry.
  `num_actions = 3 + num_bet_bins`, so `len(bet_bins)` must equal `num_bet_bins` exactly;
  a shorter array makes the top atoms silently alias under JAX's index clamping.
- **Bet sizing is pot-relative**: bet atom `i` adds
  `round(call + bet_bins[i] * (pot + call))` chips — call first, then raise by that
  fraction of the post-call pot, so `1.0` is a pot-sized raise. `bet_bins =
  linspace(max_bet_fraction / K, max_bet_fraction, K)`. The result is clipped to
  `[call + min_raise, stack]`, which is what collapses a bet atom into an all-in when the
  stack is short. Sizing against the pot rather than the call is deliberate: a
  call-relative fraction collapses every bet atom onto the min raise on an unraised street,
  where `call == 0`.
- **Seat conventions**: player 0 is button/small blind (first pre-flop, last post-flop),
  player 1 is big blind. Post-flop the turn resets to player 1.
- **Cards**: index `c` in `0..51`, `rank = c // 4` (0=2 … 12=A), `suit = c % 4`. The whole
  board is dealt at `init_state` and masked per street by `_visible_board`; showdown uses a
  branch-free 7-card best-five evaluator (`hand_rank7`), which returns an int32 score
  ordering any two hands exactly.
- **`max_length`** is a *practical* bound (`4 * (6 + 3*ceil(log2(stack_bb+1))) + 4`), not
  the worst case. Pathological min-raise wars are truncated without terminal reward. It
  also fixes the history-buffer size, so changing it changes observation dimensionality.
- **Rewards** are zero-sum and already normalised to `[-1, 1]` (hence `max_reward = 1.0`):
  `'difference'` divides net chips by the starting stack, `'binary'` is ±1/0.

### Betting rules ([betting.py](src/oppex/envs/betting.py))

The card-free betting rules — legality, pot-relative sizing, when a round closes,
when the hand ends — live here, **not** in `HunlHoldem`, because two consumers need
them: the env (which wraps them with cards, history buffers and normalised rewards)
and the CFR public-tree builder (which enumerates them directly). A divergence
between the two would not crash; it would silently produce a wrong equilibrium.

`BettingState` is card-free *and history-free*. That is load-bearing: it means two
sibling edges reaching the same `BettingState` are provably the same action, which
the tree builder uses as an automatic duplicate-action detector.

`HunlState` stays flat and packs/unpacks via `_to_betting` / `_from_betting` — pure
pytree plumbing, free under jit. Nesting would rewrite the field layout that
`_betting_features` and all six observation methods depend on, for no gain.

Two rules quirks worth knowing:
- `num_bet_bins=0` (the push-fold abstraction) needs `bet_fraction` to branch
  *statically* on `bet_bins.shape[0]`. Gathering from a shape-`(0,)` array raises at
  trace time; JAX's index clamping does not rescue a zero-sized operand.
- Facing an all-in that covers you, CALL and ALL_IN produce bit-identical successors.
  `drop_duplicate_allin` suppresses the duplicate (default off in the env, on in the
  tree). Left in, regret mass splits across two identical columns and shove frequency
  reads as `σ[CALL] + σ[ALL_IN]`.

## Public-state CFR ([src/oppex/cfr/](src/oppex/cfr/))

Preflop solver. Walks a tree of publicly observable actions, keeping regrets for
every private holding at once rather than sampling deals.

**Why `init_state`'s all-at-once deal is irrelevant here.** It is not biased — the
chance measure factorizes, so marginalizing the board out of a board-independent
fold payoff is exact — it is simply the wrong interface. PSCFR never enumerates
deals: the hole-card deal becomes the initial uniform *range* over 1326 combos, and
the board is folded into all-in equity at showdown terminals only. `init_state`
remains the sampling/RL path and needs no change.

**Card removal is the real trap, and *where* it lives is the whole game.** It
belongs at terminals (the `MASK` matvec) and in the root chance constant — **never
in reach propagation**, where `r_i[h]` is player *i*'s own action-probability
product and carries no dependence on the opponent's holding. Two specific hazards,
both guarded:

- A fold terminal's payoff is a constant, which makes `c * r_opp.sum()` look
  obviously right. It must be `c * (MASK @ r_opp)`. Measured: against a uniform
  range the two differ by a flat 0.9238 (= 1225/1326) that regret matching absorbs
  entirely; against a sharpened range the ratio spreads over 0.877–0.980,
  hand-dependently. Hence the single `terminal_cfv` funnel.
- Reach conservation holds for the **joint** reach, not per player. At a node owned
  by player 1 every branch carries the same `r0`, so summing `r0` over terminals
  counts it once per opponent continuation.

Going bottom-up at a node owned by `p`, `cfv_p` is sigma-weighted but `cfv_{1-p}` is
a **plain sum** — the opponent's probabilities are already inside `r_p` at the
leaves. `check_zero_sum` catches getting this wrong.

The chance constant `1/N_DEALS` is deferred to the reporting boundary, keeping cfvs
O(1e4) rather than O(1e-5) in float32. Safe only because the deal is uniform, so the
constant is global and regret matching is scale-invariant.

`equity.py` costs ~4.5–5.4 ms/board, dominated by memory traffic in the pairwise
accumulation, not hand evaluation (21%). Every obvious optimisation measured
*slower* and is listed in its docstring — including storing only the upper triangle
(1.8x slower: gathering 878k index pairs costs more than the halved traffic saves,
despite `EV` being antisymmetric). Exhaustive enumeration is ~3-4h, so MC is the
default; runs checkpoint every 20k boards and resume bit-identically.

`allow_limp=True` values a limped pot as a checkdown, which hands the SB a free
showdown real postflop play would punish. It is a valid debug target but **cannot be
compared to published charts** — only `allow_limp=False` reproduces that game.

Validation ladder, in `scripts/solve_pushfold.py`: invariants → the cards-removed
game (`EV = 0`, equilibrium computable by hand) → the exact LP in `lp_reference.py`,
which shares the equity matrix with CFR so a disagreement is a solver bug rather
than sampling noise → published charts last.

**Know what the LP does and does not cover.** It solves a *matrix game* — one move
per player — which is exactly the `allow_limp=False` tree. It is not a sequence-form
LP, so it cannot be pointed at `allow_limp=True`, where the SB acts twice on the
`CALL → BB shove → CALL/FOLD` line and the value stops being bilinear in per-hand
action probabilities. `allow_limp=False` *deletes* the limp action (`legal[CALL] =
False` at the root) rather than assigning it a payoff. So multi-level traversal is
currently checked only by the structural invariants, never against an exact
reference; a sequence-form LP (cheapest on a small synthetic deck) would close that.

### Env construction

[envs/\_\_init\_\_.py](src/oppex/envs/__init__.py) holds a name→class `_REGISTRY` and
`build_env(cfg)`, which pops `name` and forwards the rest as kwargs. Register new
environments there.

## Style

Two-space indentation, `from __future__ import annotations`, lowercase-with-underscores,
leading underscore for module-private helpers. Sections are separated by `# ── Name ───`
banner comments; keep that convention when adding code. Docstrings carry the design
rationale (why a bound was chosen, why a feature is in the observation) — this is research
code where that reasoning is the valuable part.
