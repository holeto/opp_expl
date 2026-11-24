# Re-solver against slumbot.com, 1000 hands

Details and discussion: `notes/br_against_slumbot.md`, section "2026-11-09: 1000 hands on slumbot.com".

**Result: +1929 ± 1373 mbb/hand** (AIVAT with boards valued by rollouts, ±95%).

| Estimate (mbb/hand, ±95%) | Mean | Std (BB/hand) |
|---|---|---|
| Raw | +2572 ± 2113 | 34.09 |
| + chance | +2454 ± 1799 | 29.02 |
| + our decisions | +2514 ± 1798 | 29.01 |
| + opponent decisions | +2312 ± 1742 | 28.10 |
| **Boards by rollouts** | **+1929 ± 1373** | **22.16** |

## Checks

- 0 lost hands, 0 desyncs, 0 zero-probability Slumbot moves.
- 91% of Slumbot's bets map exactly onto our bins (largest gap 0.46 pot).
- 9 string mismatches, all called all-ins where only the server's trailing `/` differs.
- AIVAT: 1000/1000 hands, 3184/3184 Slumbot decisions, 1143/1143 boards; all term means within noise of 0.
