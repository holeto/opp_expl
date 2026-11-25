# LBR against chumps with other raise distributions (job 12088968)

Details: `notes/br_against_slumbot.md`, section "2026-11-24: the paper's random chump is uniform over every legal action".

**Result: the paper's "random" chump is uniform over every legal action, with raises in whole chips.** All four cells match its Table 2. Raises in whole big blinds don't explain the other mismatches, and 1/2 call 1/2 raise still doesn't match.

BB/hand, duplicate ±95%, 20 000 hands per cell (one cell 19 000). Std: per hand over all hands / duplicate (pair averages × √2), BB. z against the paper combines both 95% intervals.

## Uniform over every legal action (paper: random)

| Cell | Raises in chips | Std | Raises in BB | Std | Paper | z (chips) |
|---|---|---|---|---|---|---|
| fc 1-4 | **15.80 ± 1.28** | 131 / 92 | 15.65 ± 1.28 | 130 / 92 | 15.7 ± 0.4 | +0.2 |
| fcpa 1-4 | **38.89 ± 1.63** (19 000 hands) | 146 / 115 | 38.53 ± 1.59 | 147 / 115 | 39.1 ± 0.6 | −0.2 |
| fc 3-4 | **1.68 ± 0.46** | 183 / 33 | 1.82 ± 0.49 | 182 / 35 | 2.2 ± 0.7 | −1.2 |
| fcpa 3-4 | **80.18 ± 1.80** | 164 / 130 | 73.86 ± 1.79 | 166 / 129 | 80.7 ± 0.7 | −0.5 |

- In chips every cell is within noise. In big blinds fcpa 3-4 is 7 BB/hand low (z ≈ 7), so the raises are in chips.

## Action type first, raises in BB

| Cell | call_raise, BB | Std | Earlier, chips | Paper |
|---|---|---|---|---|
| fc 1-4 | 16.53 ± 1.07 | 102 / 77 | 17.9 ± 1.1 | 7.1 ± 0.5 |
| fcpa 1-4 | 22.40 ± 1.25 | 151 / 90 | 23.9 ± 1.3 | 23.1 ± 0.5 |
| fc 3-4 | 7.86 ± 1.01 | 138 / 73 | 7.2 ± 1.4 | 16.2 ± 0.3 |
| fcpa 3-4 | 21.54 ± 0.92 | 167 / 67 | 22.6 ± 0.9 | 24.4 ± 0.6 |

| Cell | random (1/3 per type), BB | Std | Earlier, chips | Paper |
|---|---|---|---|---|
| fc 1-4 | 14.79 ± 1.04 | 93 / 75 | 13.5 ± 1.0 | 15.7 ± 0.4 |
| fcpa 1-4 | 21.88 ± 1.43 | 121 / 103 | 20.6 ± 1.4 | 39.1 ± 0.6 |
| fc 3-4 | 6.92 ± 1.14 | 126 / 82 | 5.5 ± 1.6 | 2.2 ± 0.7 |
| fcpa 3-4 | 49.15 ± 1.62 | 139 / 117 | 51.1 ± 1.6 | 80.7 ± 0.7 |

- The raise step moves these cells by 2 BB/hand or less. "Earlier, chips" is `2026-11-19_lbr_chumps_rounds34` and `_rounds14`.
- **1/2 call 1/2 raise remains the only mismatch.** Its fc cells still look swapped against the paper's (ours 16.5 / 7.9, the paper's 7.1 / 16.2), and fcpa 3-4 is about 3 BB/hand low.

## Checks

- 319 of 320 tasks completed. **Task 188 (uniform, chips, fcpa 1-4, seed 7188) was cancelled**: a node fault requeued it at start, the admins held it, and we cancelled it on 2026-11-25. That cell has 19 000 hands.
- With the default flags the new code replays 200 hands of the earlier random fc 3-4 run exactly. In a smoke test every BB-step chump raise was a whole number of big blinds or the all-in, and none was below the minimum raise.
