# Depth-limited RNR with safe re-solving on Leduc, against a perturbed Nash

Details: `notes/safe_resolving.md`, section "2026-11-26: depth-limited RNR with safe re-solving on Leduc". Derivation: `notes/rnr_resolve_derivation.md`.

**Result: depth-limited RNR reproduces the full-game RNR trade-off.** The model is the reference LP Nash with half of its round-2 folds facing a bet turned into calls. With variant B and CBV alternative values, F is within 1.7e-4 of the full-game RNR at every p. At p = 0.1 we win 0.1166 per hand from the model (the Nash wins 0.098, a best response 0.560) for an exploitability of 6.5e-4. At p = 0 the compound is our earlier safe re-solve bit for bit; at p = 1 it matches the best response to 1.8e-8.

## Trade-off (mean over both seats; root solve 1000 iterations, re-solves 3000, full-game RNR 3000)

| p | Method | Value vs model | Exploitability | F | F − F_parent | F_full − F |
|---|---|---|---|---|---|---|
| 0 | full-game RNR | +0.09844 | 2.06e-5 | −0.00002 | | |
| 0 | depth-limited B+cbv | +0.08947 | 1.55e-4 | −0.00015 | −1.9e-5 | +1.3e-4 |
| 0.1 | full-game RNR | +0.11651 | 4.62e-4 | +0.01124 | | |
| 0.1 | depth-limited B+cbv | +0.11656 | 6.49e-4 | +0.01107 | −6.1e-5 | +1.7e-4 |
| 0.5 | full-game RNR | +0.37656 | 0.1291 | +0.12372 | | |
| 0.5 | depth-limited B+cbv | +0.37600 | 0.1289 | +0.12355 | −1.5e-5 | +1.7e-4 |
| 1 | full-game RNR | +0.55991 | 1.948 | +0.55991 | | |
| 1 | depth-limited B+cbv | +0.55991 | 1.948 | +0.55991 | +1.4e-8 | +1.5e-8 |

- Exploitability is that of our strategy alone (seat game value minus worst case), averaged over seats. F = p·(value vs model) + (1 − p)·(worst case).
- The model: a best response wins +0.44891 (seat 0) and +0.67091 (seat 1), mean +0.55991; the reference Nash wins +0.09412.
- The model has full support, so all 15 round-2 states are re-solved at every p, p = 1 included.
- Per-seat rows are in `run.log`.

## Variants at p = 0.1 and 0.5 (mean over seats; worst seat in brackets)

| p | Variant | Exploitability | F − F_parent | F_full − F | Max exceedance over alt |
|---|---|---|---|---|---|
| 0.1 | B+cbv | 6.49e-4 | −6.1e-5 (−7.0e-5) | +1.7e-4 | +7.3e-4 |
| 0.1 | B+avg | 5.90e-4 | −1.3e-5 (−3.4e-5) | +1.2e-4 | +9.8e-3 |
| 0.1 | A+cbv | 6.26e-4 | −3.6e-5 (−6.5e-5) | +1.4e-4 | +1.9e-3 |
| 0.5 | B+cbv | 0.1289 | −1.5e-5 (−3.9e-5) | +1.7e-4 | +1.2e-3 |
| 0.5 | B+avg | 0.1286 | +6.0e-5 (+1.9e-7) | +9.1e-5 | +0.95 |
| 0.5 | A+cbv | 0.1317 | −1.3e-3 (−2.3e-3) | +1.5e-3 | +0.19 |

- Variant A loses most at p = 0.5: −2.3e-3 in seat 1, about 60× the worst B+cbv drop at the same p (−3.9e-5), and its exploitability rises to 0.1317 against 0.1289. At p = 0.1 the three variants are within solve accuracy of each other.
- B+avg has large exceedances (up to 0.95), but they are measured against averaged cfvs, which lie below the CBVs; F does not suffer at this budget.

## Runtime

- One local run, 15 min 3 s (2026-11-26 13:00:04 to 13:15:07), 12th Gen Intel i5-12450H, WSL2, numpy and JAX on CPU.
- Per p (both seats): full-game RNR 27–30 s; depth-limited walk 83–89 s per variant.

## Checks

- All assertions passed (exit 0):
  1. frozen model rows bit-identical through every solve;
  2. RNR games under random strategies worth p·u(model) + (1 − p)·u(free) (error 2.2e-16);
  3. p = 0: compound equals `resolve_walk` bit for bit (the p = 0 results do not depend on the model), NashConv of our two seats 3.1e-4;
  4. p = 1: value minus best-response value against the model, 1.4e-8 / 1.8e-8 (compound) and 9.0e-10 / 1.6e-9 (full RNR);
  5. B+cbv: F ≥ F_parent − 1e-3 in each seat (smallest margin −7.0e-5).
- Deterministic; no seeds.
