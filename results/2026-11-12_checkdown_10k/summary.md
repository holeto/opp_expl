# Checkdown leaves, 10 000 hands per variant

Details: `notes/br_against_slumbot.md`, section "2026-11-12: checkdown leaves", subsection "10 000 hands per variant".

| Variant | AIVAT (mbb/hand, ±95%) | Std (BB/hand) | Duplicate pairs |
|---|---|---|---|
| cd_base | +2337 ± 504 | 25.7 | ± 488 (std 24.9, corr −0.06) |
| cd_call2 | +2864 ± 488 | 24.9 | ± 466 (std 23.8, corr −0.09) |

- **cd_call2 − cd_base, paired by deal: +526 ± 656** (z ≈ 1.6), raw +772 ± 765.
- With the 1000-hand run (11 000 deals): +469 ± 615.

## Checks

- All 20 000 hands clean: no lost hands, desyncs, mismatches, zero-probability moves or refusals.
- All AIVAT term means ≈ 0.
- Run time: cd_base about 2 h per 500 hands, cd_call2 3 h 20 to 3 h 50; about 450 core-hours.
