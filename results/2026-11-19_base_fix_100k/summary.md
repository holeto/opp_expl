# base, 100 000 hands, with the `_point` fix

Details: `notes/br_against_slumbot.md`, section "2026-11-23: base 100k rerun with the `_point` fix".

| Raw winnings (mbb/hand, ±95%), same 100 000 deals | Estimate | Std (BB/hand) |
|---|---|---|
| base_fix | **+2694 ± 210** (duplicate ± 194, 50 000 pairs) | 33.81 |
| As SB / as BB | +3208 ± 277 / +2180 ± 315 | |
| base, old run (`2026-11-13_base_100k`) | +2605 ± 211 | 33.98 |
| cd_call2 (`2026-11-13_cd_call2_100k`) | +2874 ± 169 | 27.34 |
| **base_fix − base, paired** | **+89 ± 247** | sd 39.8, corr +0.31 |
| **base_fix − cd_call2, paired** | **−180 ± 239** | sd 38.5, corr +0.22 |

## Checks

- Rollout refusals: 0 (245 before the fix).
- 0 desyncs, string mismatches, zero-probability moves, model refusals or lost strings in 318 842 Slumbot steps; bets exact 91.0%.
- Lost hands: 4 (old run 9), all the minimum-bet case; together −309 BB (−3 mbb/hand), included.
- Node failure on n05 stopped 5 tasks; rerun as job 12078378. The 719 hands already played were replayed identically.
