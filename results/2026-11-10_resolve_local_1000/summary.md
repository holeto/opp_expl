# Re-solver against LocalSlumbot, 1000 hands

Details: `notes/br_against_slumbot.md`, section "2026-11-10: 1000 hands against LocalSlumbot".

**Result: +2678 ± 1227 mbb/hand** (AIVAT, boards by rollouts, ±95%).

| Estimate (mbb/hand, ±95%) | Local | Std | slumbot.com | Std |
|---|---|---|---|---|
| Raw | +3794 ± 1667 | 26.89 | +2572 ± 2113 | 34.09 |
| + chance | +2785 ± 1339 | 21.60 | +2454 ± 1799 | 29.02 |
| + our decisions | +2812 ± 1345 | 21.70 | +2514 ± 1798 | 29.01 |
| + opponent decisions | +2711 ± 1364 | 22.01 | +2312 ± 1742 | 28.10 |
| **Boards by rollouts** | **+2678 ± 1227** | **19.79** | **+1929 ± 1373** | **22.16** |

- Local − server: +749, 95% interval [−1092, +2590] (independent runs). Consistent, but only gaps ≥ 2600 would be detected at 80% power.
- Model calibration at Slumbot's real decisions (both runs): see the notes section "Calibration of the model".
