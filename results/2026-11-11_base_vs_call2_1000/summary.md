# base against call2, 1000 hands per variant

Details: `notes/br_against_slumbot.md`, section "2026-11-11: calling the first two streets (call2) against full re-solving (base)".

| Estimate (`board_q` AIVAT, mbb/hand, ±95%) | base | call2 |
|---|---|---|
| Hands | 999 | 1000 |
| Unpaired | +3435 ± 1442 (std 23.25) | +1922 ± 1192 (std 19.23) |
| Duplicate pairs | +3382 ± 1411 (std 22.74, corr −0.04) | +1922 ± 1181 (std 19.05, corr −0.01) |

- **call2 − base, paired by deal: −1433 ± 1904** (raw −1737 ± 2393). Not significant; a gain above +470 is excluded at 95%.
- base lost one hand (seed 102, hand 48: Slumbot's minimum bet mapped to all-in), excluded from AIVAT.
- Speed: call2 9.4 s/hand, base 21.7 s/hand.
