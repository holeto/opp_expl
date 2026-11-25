# Experiment results

One folder per experiment, named `<run date>_<name>`. Each folder holds:

- **`config.yaml`**: what was run, with these keys:
  - `question`, `checkpoint`, `opponent`, and `agent` (the settings that are not CLI flags).
  - `code`: the code version.
  - `jobs`: SLURM job IDs, dates, resources and final state.
  - `variants`: one entry per variant, with:
    - `command`: the exact CLI call per seed.
    - `seeds`, `hands_per_seed`, `total_hands`.
    - `resolved_args`: every argument with its argparse default filled in.
    - `evaluation`: the AIVAT command, where one was run.
  - `pooling`: how the per-seed logs were combined.
  - `raw_data`: where the per-hand logs and solves are.
- **`summary.md`**: the result tables and run checks. `notes/br_against_slumbot.md` has the discussion; each summary names its section there.
- **`pool.py`** (where one exists): the script that produced the summary from the raw logs. Run it from the project root unless its config says otherwise.

Commands name the CLI only. The scripts are in `scripts/policy_gradient/` and run as `uv run python scripts/policy_gradient/<script> ...`.

## Code versions

Runs used the working tree synced to the cluster, and no commit hash was logged. So `code` gives:

- `after_commit`: the last commit before the run.
- `committed_in`: the first commit after it, which contains the run's code.

Both are inferred from dates. From now on, a run should log `git rev-parse HEAD` and `git status --short` with its output.

## Experiments: DLBR (re-solver) and LBR against Slumbot

| Folder | What | Headline (mbb/hand, ±95%) |
|---|---|---|
| `2026-11-09_resolve_slumbotcom_1000` | re-solver vs slumbot.com, 1000 hands | +1929 ± 1373 (AIVAT) |
| `2026-11-10_resolve_local_1000` | re-solver vs LocalSlumbot, 1000 hands | +2678 ± 1227 (AIVAT) |
| `2026-11-11_base_vs_call2_1000` | base vs call2, 1000 hands each | base +3435 ± 1442, call2 +1922 ± 1192 (AIVAT) |
| `2026-11-11_checkdown_1000` | cd_base vs cd_call2, 1000 hands each | cd_base +2887 ± 1460, cd_call2 +2779 ± 1239 (AIVAT) |
| `2026-11-12_checkdown_10k` | cd_base vs cd_call2, 10 000 hands each | cd_base +2337 ± 504, cd_call2 +2864 ± 488 (AIVAT) |
| `2026-11-13_cd_call2_100k` | cd_call2, 100 000 hands | +2874 ± 169 (raw) |
| `2026-11-13_base_100k` | base, 100 000 hands (before the `_point` fix) | +2605 ± 211 (raw) |
| `2026-11-19_base_fix_100k` | base, 100 000 hands (with the fix) | +2694 ± 210 (raw) |
| `2026-11-19_lbr_100k` | LBR fc/fcpa/bins vs LocalSlumbot, 100 000 hands each | fcpa 3-4 +1892 ± 157, bins 3-4 +2258 ± 161 (raw) |
| `2026-11-19_lbr_chumps_rounds34` | LBR vs the paper's chumps, rounds 3-4 (local) | always call fcpa 48.9 ± 1.0 BB/hand (paper 49.0) |
| `2026-11-19_lbr_chumps_rounds14` | LBR vs the raising chumps, rounds 1-4 | see summary; the raising chumps don't match the paper |
| `2026-11-24_lbr_fcpa14_100k` | LBR fcpa 1-4 vs LocalSlumbot, 100 000 hands | +1683 ± 211 (raw; Milec 1388 ± 150) |
| `2026-11-24_lbr_chumps_bb` | LBR vs chumps with raises in BB, and a chump uniform over every legal action | uniform (chips) matches the paper's random chump in all 4 cells, e.g. fcpa 3-4 80.2 ± 1.8 BB/hand (paper 80.7) |

The 100k runs and the LocalSlumbot rows of `2026-11-19_lbr_100k` use seeds 2000–2399. They play the same deals, so they compare deal by deal.

## Seeds

`seeds` and `hands_per_seed` mean one run per seed, each playing that many hands. A seed does not fix one deal for all its hands:

- **Deals:** each hand's deck, and LocalSlumbot's hidden translation type `u`, come from their own generator, `np.random.default_rng([seed, deal_index, stream])` (`RandomOpponent.deal_rng`). So a deal depends only on (seed, hand number), not on how earlier hands went. Different agents on the same seeds play the same deals, which is why runs on seeds 2000–2399 compare deal by deal.
- **`--duplicate`:** hands `2k` and `2k + 1` share deal `k` (`deal_index = hand // 2`), with seats and hole cards swapped. So 250 hands per seed are 125 distinct deals, each played from both seats.
- **Other randomness:** our agent's (rollouts, sampling its strategy) and the opponent's moves come from generators seeded once per run (the agent from the seed, the opponent from seed + 1). They advance from hand to hand and are reproducible for a given seed.
- **slumbot.com** (`2026-11-09_resolve_slumbotcom_1000`): the server deals the cards, so the seed fixes only our agent's randomness.

## Not covered yet

- Earlier experiments in other notes files (`slumbot_eval_qmmd_560k.md`, `Slumbot_handicapped.md`), e.g. the blueprint against Slumbot.
- The blueprint-opponent re-solving runs (`checkpoints/qmmd_hunl_560k/resolving/`).
- Analyses that are not runs: the model calibration and the translation recount.
