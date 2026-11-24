# Checkdown leaves, 1000 hands per variant (same deals as base/call2)

Details: `notes/br_against_slumbot.md`, section "2026-11-12: checkdown leaves".

| AIVAT (mbb/hand, ±95%) | Blueprint-rollout leaves | Checkdown leaves |
|---|---|---|
| Re-solve every street | base +3435 ± 1442 | cd_base +2887 ± 1460 |
| Call preflop and flop | call2 +1922 ± 1192 | cd_call2 +2779 ± 1239 |

| Paired by deal | Difference |
|---|---|
| cd_call2 − cd_base | −107 ± 1642 |
| cd_call2 − call2 | +858 ± 1256 (corr +0.47) |
| cd_base − base | −676 ± 2045 |
| Interaction (cd_call2 − cd_base) − (call2 − base) | +1413 ± 2282 |

- The base and call2 columns come from `2026-11-11_base_vs_call2_1000`.
