# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhD research code on opponent exploitation in imperfect-information games. Pure-JAX
environments and (planned) RL/game-theoretic training built on top of them. Very early
stage: only the environment layer exists so far, and there are no commits yet.

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
