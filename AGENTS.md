# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhD research code on opponent exploitation in imperfect-information games. Pure-JAX
environments plus a public-state CFR solver built on them. Early stage: the env layer
(`oppex.envs`: HUNL, Goofspiel), a preflop push-fold CFR solver (`oppex.cfr`) with
matrix-valued depth-limit leaves, and a policy-gradient baseline
(`oppex.policy_gradient`: QMMD, ported from the sibling RLG repo, its Slumbot player and
the blueprint continuations) with exact exploitability for small games (`oppex.tree`)
exist.

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

```bash
uv run python scripts/train.py configs/qmmd_goofspiel.yaml [--resume]   # QMMD from a YAML config
uv run python scripts/eval_exploitability.py checkpoints/qmmd_goofspiel4 # exact NashConv per checkpoint
```

No test runner or linter is wired up yet. JAX resolves to the CPU build here; a CUDA
jaxlib is not installed.

## Architecture

### Layering

Docstrings refer to numbered layers ("Layer 3 — Environments" in
[base.py](src/oppex/envs/base.py), "Layer 5" for `oppex.policy_gradient.algorithms`,
"Layer 6" for `oppex.trainers`); the lowest layers are not written yet. Treat that
numbering as the intended structure, not as existing code. New layers go in as sibling
subpackages of `oppex.envs` (`networks`, `agents`, `trainers`, `tree` follow this). The
policy-gradient-specific `algorithms` and `losses` live under `oppex.policy_gradient`.

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
checked against `seqform.py`, the textbook sequence-form LP built straight off the
tree, which covers `allow_limp=True` too. Use `method="highs-ipm"` for it: the 10 BB
limp LP takes ~80 s by interior point and over 17 min by simplex.

### Matrix-valued leaves ([leaves.py](src/oppex/cfr/leaves.py))

`build_preflop_tree(..., matrix_leaves=True)` turns each checkdown cut into a `MATRIX`
depth-limit leaf. There both players **simultaneously** pick one of `T` continuation
strategies, and the leaf pays `leaf_values[l, t0, t1, a, b]` chips to player 0. The
tensor is mask-folded like `EV` and is *not* antisymmetric. It must not be split into
two sequential decision nodes: in a public tree that makes `t0` public. Leaf regrets
live in `Tables.leaf_regret` `(2, n_leaves, H, T)` and share `_accumulate` with the node
tables. `seqform.solve(..., leaf_values=)` gives each player one infoset per (leaf,
hand). The leaf values come from `oppex.policy_gradient.leaf_values` (the QMMD blueprint
plus Pluribus' fold/call/raise ×5 biases, bucketed by equity). They are verified by
`scripts/solve_matrix_leaves.py`: degenerate one-continuation leaf == checkdown →
invariants → sequence-form LP.

### Fixed opponent ([solver.py](src/oppex/cfr/solver.py), [exploiter.py](src/oppex/policy_gradient/exploiter.py))

`cfr_step(..., fixed_player=p)` updates only `1 - p`, so it converges to a best
response rather than to an equilibrium. The fixed strategy lives in the ordinary tables:
`fix_strategy` writes `sigma` into `p`'s regret **and** strategy-sum rows. Regret
matching returns a normalised non-negative vector unchanged, so the traversal needs
no new branch and `average_strategy` returns the full profile. Those rows are never
written back, and `scripts/solve_fixed_opponent.py` asserts they are bit-identical
at the end. Its convergence check is the exact `exploit.best_response` against the
same frozen profile. The gap must go to 0.

At matrix leaves the fixed player has **one continuation**. This is encoded by shape: a
leaf tensor with a size-1 continuation axis means the payoff ignores that player's
`t`, so their `tau` drops out (`leaf_action_cfv`). `check_leaves` accepts `T0 == 1`
or `T1 == 1`.

The real target is the blueprint's own preflop abstraction (`env.rules`, 4 bins,
`allow_limp=True`): 2192 nodes, 1095 leaves. Dense leaves would be 28 MB per leaf even
with one continuation, so `leaves.BucketedValues` stores bucket-pair means
**unmasked** and applies card removal in the contraction, by inclusion–exclusion
over the two own cards at bucket level (`all − holding d1 − holding d2 + self`).
That way nothing is built per hand pair: 0.06 s per iteration for both players
against 0.5 s via a masked `(H, L, T, K)` tensor. It is checked against the dense
contraction. `allow_limp=False` deletes every non-closing CALL, so a tree with
bet atoms has no leaves at all. `blueprint_tree_strategy` reads the network off the tree and
folds struck duplicate atoms into their survivor via `DecisionNode.alias`.
`ExploiterPolicy` plays the solution: the CFR average preflop, a continuation drawn
from `tau` at the leaf, and the biased blueprint postflop. It runs in a batched env
match against the blueprint (with a paired blueprint-vs-blueprint control) and in
the Slumbot loop (`play_slumbot.py --exploiter`). Slumbot's strategy is unknown, so
against Slumbot it plays the exploit of MMD.

### Re-solving ([subgame.py](src/oppex/cfr/subgame.py), [infoset_cfr.py](src/oppex/cfr/infoset_cfr.py), [resolving.py](src/oppex/policy_gradient/resolving.py))

Full-game play against the fixed blueprint. At each of our decisions not covered by
the current subgame, `build_subgame` enumerates `depth` plies (default 4, two of our
moves) below the current state. Matrix leaves go at the depth limit and at every street
close, so **a subgame has no chance nodes**: the next card is sampled inside the leaf
rollouts, and the next street starts a new solve. Most lines close a street within two
plies, so most decisions are re-solved and only a minority are reused.

**No gadget, and one own hand.** Against a fixed opponent the game is a decision
problem for us. The root is our infoset: our hand plus `r_o(h)`, the blueprint's
probability of the opponent's history for each of their hands, times card
compatibility. Every terminal and leaf contracts against `r_o` to a scalar, and
`infoset_cfr` is PSCFR with a hand axis of size 1. It shares `solver._accumulate`.
`best_response` is the exact solution, used for the convergence gap. The opponent model
is ε-mixed (`eps=1e-3`) so that an off-model Slumbot action cannot zero the belief.
Duplicate bet atoms are folded through `alias` everywhere, because only the public state
is observed.

Leaf values are rollouts: `h ~ r_o(l,·)`, the unknown cards drawn from the deck, and
our seat plays `blueprint ∘ t`. Common random numbers are shared across leaves and `t`.
Leaves below `min_mass` of the root's opponent mass are valued 0, and a cached node below
it is re-solved rather than played. Showdowns use `EV[ours]` preflop and
`equity.sign_vs_range` (exact runout enumeration) on a board.

`build_subgame` with no depth limit from the preflop root reproduces
`build_preflop_tree` node for node, 30-50x faster (one jitted call per node).
`scripts/resolve_fixed_opponent.py` checks that the single-hand best response equals
PSCFR's per-hand BR (`exploit.best_response(per_hand=True)`) on the cached bucketed
leaves, that re-rooting with the history belief reproduces node values, and that bucket
averages of the rollout values match the uniform-deal leaf sample. Do not compare one
hand's rollout value with the bucketed cache: a bucket mean is far from its best hands.
`--play N` runs the env match with a paired blueprint control. For Slumbot use
`play_slumbot.py --resolve`.

### Env construction

[envs/\_\_init\_\_.py](src/oppex/envs/__init__.py) holds a name→class `_REGISTRY` and
`build_env(cfg)`, which pops `name` and forwards the rest as kwargs. Register new
environments there.

## RL training — QMMD ([src/oppex/policy_gradient/algorithms/](src/oppex/policy_gradient/algorithms/))

Ported from RLG `algorithms/mmd_q.py` (`QMMD`): Magnetic Mirror Descent with a policy head
and a Q-head (no V-head), Q-targets by Retrace(λ) over the Polyak *target* network's
Q-values. On-policy, so the IS ratio is 1 and Retrace reduces to Expected SARSA(λ).
Policy loss is PPO-clipped on a centred all-actions advantage (`loss_type=mmd`) or NeuRD
(`loss_type=rnad`); `+ magnet_coef·KL(π‖magnet) + old_policy_coef·KL(π‖μ) − ent_coef·H`.
The α-entmax path and the LaProp/Muon optimizers from RLG were deliberately not ported:
every policy here is a masked softmax and `oppex.policy` has no `alpha`.

**Moving magnet.** `magnet_update_type="incremental"` (the default) Polyak-averages the
magnet toward the current params at `magnet_update_rate` every step instead of hard-copying
every `magnet_interval` steps (`"periodic"`). For HUNL, `notes/mmd_hunl_tuning.md` puts both
target and magnet rates around 5e-5.

**Learning-rate schedule.** The notes also ask for a cosine-decayed learning rate.
`optimizer.lr` takes a number or a schedule spec
(`{type: cosine_decay, init_value, decay_steps, alpha}`; also `warmup_cosine_decay`).
`decay_steps: null` is filled with the run length. Optax counts *optimizer updates*, not
training steps, so the fill is `n_steps × n_epochs`. The coefficient `schedules` are
evaluated on the training step instead. Only the learning rate is injected into the optimizer state
(`inject_learning_rate`), and the applied rate is logged as `lr`. Injecting Adam's `b1`/`b2`/`eps`
too shifts rounding by an ulp, enough for a run to drift from plain-Adam reruns. Checkpoints
saved before this change have a different `opt_state` and cannot be resumed.

**Rollouts auto-reset; they do not pad** ([rollout.py](src/oppex/policy_gradient/algorithms/rollout.py)).
Each of B rows is a fixed window of T steps over a continuing stream of games: when a game
ends, a fresh `init_state` is dealt in place (so chance is still drawn once per game). The
env state at the window's end is carried in `TrainingState` and resumes next iteration.
This *differs from RLG*, whose collector — despite its docstring — ran every row from a fresh
deal for `env.max_length` steps and masked everything after the terminal. For HUNL that is
~97% padding (hands average ~4 steps against `max_length` 124). Consequences to keep in mind:

- A row usually ends mid-game, so the last Retrace target bootstraps from
  `E_μ[Q_target(s_T)]` (`retrace(..., bootstrap_v=)`). RLG hard-coded 0 there, which is only
  right when every row ends on a terminal. Episode boundaries *inside* a row need nothing:
  `discount = 0` at a terminal cuts both bootstrap and trace.
- There is no validity mask — every step is real data; the loss is a plain mean over B·T
  (summed over players).
- A game reaching `env.max_length` is force-reset and recorded as a **zero-reward
  terminal** (`truncated`). Never fires in Goofspiel; watch `truncated_frac` for HUNL.
- With `rollout_len = n_cards` on Goofspiel each row is exactly one game — RLG's layout.

`PrivilegedPolicyQAgent` (with a `split_twin_head` network) gives the Q-critic
`state_representation` (both hands) while the actor still sees only its infoset; it is the
only case where the rollout stores `state_reps`.

Checkpoints are `step_{step:08d}.pkl` pickles of the whole `TrainingState` (RLG saved
`{counter}.pkl` but resumed from `step_*.pkl`, so its `--resume` never found anything).
Seed with `jax.random.PRNGKey`, not a typed `jax.random.key`, so state pickles plainly.
Pickles name classes by module path. `load_checkpoint` rewrites the pre-move
`oppex.algorithms` / `oppex.losses` paths, so checkpoints from before the move still load.

**Slumbot and the blueprint.** `oppex.policy_gradient.blueprint.Blueprint` loads a run
directory as a playing policy, and `slumbot/` plays it against Slumbot over HTTP
(`scripts/policy_gradient/play_slumbot.py`, `analyze_slumbot.py` are thin CLIs).

**Validation.** `oppex.tree` (RLG `src/tree`) enumerates the full history tree of small
games and computes exact best responses; `scripts/eval_exploitability.py` reports NashConv
for the current, target and magnet params of each checkpoint, in raw reward units. The
port was checked against RLG directly: Goofspiel observations/transitions are identical,
and `mmd_q_loss` values and gradients match RLG's for both loss types and KL directions.

## Style

Two-space indentation, `from __future__ import annotations`, lowercase-with-underscores,
leading underscore for module-private helpers. Sections are separated by `# ── Name ───`
banner comments; keep that convention when adding code. Docstrings carry the design
rationale (why a bound was chosen, why a feature is in the observation) — this is research
code where that reasoning is the valuable part.
