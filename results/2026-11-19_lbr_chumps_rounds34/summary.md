# LBR against the chumps of the paper's Table 2, turn and river (run locally)

Details: `notes/br_against_slumbot.md`, section "2026-11-19: LBR against the chumps of the paper's Table 2". The rounds 1-4 cells are in `2026-11-19_lbr_chumps_rounds14`.

| BB/hand, duplicate ±95% | Ours | Hands | Paper |
|---|---|---|---|
| always call, fcpa 3-4 (`lbr_chumps.py`) | **48.9 ± 1.0** | 20 000 | 49.0 ± 0.4 |
| always call, fcpa 3-4 (`play_slumbot.py --lbr`) | **49.3 ± 1.0** | 20 000 | 49.0 ± 0.4 |
| always call, fc 3-4 | **0.0** (every hand checks down) | 1 000 | 0 |
| always call, fc 1-4 | **0.0** | 200 | 0 |
| 1/2 call 1/2 raise, fcpa 3-4 | **22.6 ± 0.9** | 20 000 | 24.4 ± 0.6 |
| 1/2 call 1/2 raise, fc 3-4 | **7.2 ± 1.4** | 10 000 | 16.2 ± 0.3 |
| random, fcpa 3-4 | **51.1 ± 1.6** | 20 000 | 80.7 ± 0.7 |
| random, fc 3-4 | **5.5 ± 1.6** | 10 000 | 2.2 ± 0.7 |

- Always call matches in both code paths. The raising chumps do not; the likely cause is the chumps' raise-size distribution, which the paper does not give (open question to Lisý).
- **Update 2026-11-24:** the paper's random chump is uniform over every legal action (each raise size one action), not 1/3 per action type; that chump matches all four cells. Raises in big blinds change nothing. See `2026-11-24_lbr_chumps_bb`.
