# base, 100 000 hands (play only), before the `_point` fix

Details: `notes/br_against_slumbot.md`, section "2026-11-12: checkdown leaves", subsection "100 000 hands of base". Superseded by `2026-11-19_base_fix_100k`.

| Raw winnings (mbb/hand, ±95%), same 100 000 deals | Estimate | Std (BB/hand) |
|---|---|---|
| base | **+2605 ± 211** (duplicate ± 198) | 33.98 |
| cd_call2 | +2874 ± 169 | 27.34 |
| **cd_call2 − base, paired** | **+269 ± 241** (z ≈ 2.2) | sd 38.9, corr +0.21 |

## Checks

- Lost hands: 9 (8 minimum-bet cases, 1 unexplained: seed 2114, hand 159), together −854 BB (−9 mbb/hand), included.
- Rollout refusals: 245 in 23 tasks, caused by float32 rounding in `_point` (fixed 11-16, rerun as base_fix).
- 0 desyncs, mismatches, zero-probability moves or model refusals; 318 736 Slumbot steps; bets exact 91.1%.
- 16.5 h wall time, tasks 1 h 21 to 2 h 29.
