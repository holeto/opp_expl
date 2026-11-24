# Local best response against LocalSlumbot, 100 000 hands per variant

Details: `notes/br_against_slumbot.md`, section "2026-11-19: local best response (LBR) against LocalSlumbot".

| 100 000 hands (mbb/hand, ±95%) | Ours | Paper (Slumbot 2016) |
|---|---|---|
| fcpa, turn and river | **+1892 ± 157** (dup ± 147; SB +2342, BB +1443) | 4020 ± 115 |
| fc, turn and river | **−222 ± 51** (dup ± 39) | 496 ± 55 |
| fc, all streets | **−149 ± 50** (dup ± 38) | 522 ± 50 |
| bins (0.5/1/1.5/2 + all-in), turn and river | **+2258 ± 161** (dup ± 151; SB +2946, BB +1570) | — |
| fcpa, all streets, vs always-call (20 000 hands) | **+34.7 ± 1.0 BB/hand** | 34.0 ± 0.5 |
| cd_call2 (DLBR, `2026-11-13_cd_call2_100k`) | +2874 ± 169 | — |

| Paired by deal (same 100 000 deals) | Difference | sd (BB), corr |
|---|---|---|
| LBR fcpa − cd_call2 | **−982 ± 155** | 25.06, +0.55 |
| LBR bins − cd_call2 | **−616 ± 144** | 23.21, +0.62 |

## Checks

- All 410 tasks completed, 0 lost hands.
- Env pot ≠ real pot: 2 of 167k fcpa decisions; 733 of 158k bins decisions (0.46%).
- Range calibration (`validate_lbr.py --logs`, pooled): true hand never had posterior 0; win-probability z +0.28 fcpa, +0.66 fc_c2, +0.46 fc_c0, +0.77 bins; fold-probability z +0.91 fcpa, +0.33 bins; every decile |z| < 1.5.
