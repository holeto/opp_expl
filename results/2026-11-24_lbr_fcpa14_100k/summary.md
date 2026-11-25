# LBR fcpa 1-4 against LocalSlumbot, 100 000 hands

Details: `notes/br_against_slumbot.md`, section "2026-11-24: Slumbot version confirmed, and Milec's LBR reference".

| mbb/hand, ±95% | Estimate | Std (BB/hand) |
|---|---|---|
| **LBR fcpa 1-4, 100 000 hands** | **+1683 ± 211** (duplicate ± 200, 50 000 pairs, corr −0.11) | 34.05 |
| As SB / as BB | +1960 ± 248 / +1406 ± 342 | |
| Seeds 2000–2199 only (50 000 hands) | +1751 ± 300 | |
| Milec et al., LBR (FCPA) against the same Slumbot, arXiv:2112.12594 Table 2 | 1388 ± 150 | |
| Their CDBR (depth-limited best response, FCPA) | 1774 ± 137 | |

| Paired by deal (same 100 000 deals) | Difference | sd (BB), corr |
|---|---|---|
| fcpa 1-4 − LBR fcpa 3-4 (+1892 ± 157) | −209 ± 238 | 38.38, +0.19 |
| fcpa 1-4 − LBR bins 3-4 (+2258 ± 161) | −575 ± 239 | 38.53, +0.20 |
| fcpa 1-4 − DLBR cd_call2 (+2874 ± 169) | **−1191 ± 243** | 39.28, +0.20 |
| fcpa 1-4 − DLBR base_fix (+2694 ± 210) | **−1011 ± 249** | 40.15, +0.30 |

## Against Milec et al.

- Source: `notes/Continual_depth_limited_responses.pdf`, Table 2 on page 8. Its ± are 95% confidence intervals (stated in the caption).
- **LBR:** ours is 295 higher. Combined interval ±259, z ≈ 2.2, just significant.
  - Their interval is narrow for 50 000 hands: it implies a per-hand std of about 17 BB, against our 34 BB.
  - The paper gives no hand count, duplicate scheme or game rules, so the cause is open: rules, LBR details, or how the interval was computed.
- **DLBR against their CDBR:** our cd_call2 (+2874 ± 169) is about 1100 above CDBR, and base_fix (+2694 ± 210) about 900.
- **Calling the first two rounds:** they say it no longer helps against this Slumbot. That fits our fcpa 3-4 − fcpa 1-4 = +209 ± 238 (not significant).

## Checks

- All 100 tasks completed, 9–11 min each.
- 0 desyncs, string mismatches, zero-probability moves or model refusals in 279 973 Slumbot steps. Bets exact 99.99%.
- Env pot ≠ real pot: 4 of 212k decisions.
- **Lost hands: 4**, all the known minimum-bet case: Slumbot's turn `b100` after `b300b900b2700c/b2700b13500c`. Together −74 BB (−0.7 mbb/hand), included.
- Range calibration, pooled (`validate_lbr.py --logs`, 212k decisions):
  - The true hand never had posterior 0.
  - Per-hand martingale z: log-likelihood +0.14, win probability −1.21, fold probability +0.03.
  - Fold-probability bins all |z| < 0.8.
  - Win-probability bin [0.1, 0.3) has z −2.94 (predicted 0.214, observed 0.210). That z treats decisions as independent, which decisions in the same hand are not, so it overstates the significance. The per-hand test is fine.
