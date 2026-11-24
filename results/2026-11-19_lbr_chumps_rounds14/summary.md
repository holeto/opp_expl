# LBR against the raising chumps, all four rounds

Details: `notes/br_against_slumbot.md`, section "2026-11-19: LBR against the chumps of the paper's Table 2".

| BB/hand, duplicate ±95%, 20 000 hands per cell | fc 1-4 | fcpa 1-4 |
|---|---|---|
| 1/2 call 1/2 raise, ours | **17.9 ± 1.1** | **23.9 ± 1.3** |
| 1/2 call 1/2 raise, paper | 7.1 ± 0.5 | 23.1 ± 0.5 |
| random, ours | **13.5 ± 1.0** | **20.6 ± 1.4** |
| random, paper | 15.7 ± 0.4 | 39.1 ± 0.6 |

- Only 1/2 call 1/2 raise fcpa is within noise. Our 1/2 call 1/2 raise fc values look swapped against the paper's 1-4 / 3-4 rows (ours 17.9 / 7.2, paper 7.1 / 16.2). Unresolved.
- The always-call fcpa 1-4 cell (34.7 ± 1.0, paper 34.0 ± 0.5) is in `2026-11-19_lbr_100k`.
