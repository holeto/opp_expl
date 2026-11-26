# Depth-limited RNR with safe re-solving on Leduc, against an always-call chump

Details: `notes/safe_resolving.md`, section "2026-11-26: depth-limited RNR with safe re-solving on Leduc". Derivation: `notes/rnr_resolve_derivation.md`.

**Result: depth-limited RNR reproduces the full-game RNR trade-off.** With variant B (Resolve weights) and CBV alternative values, the RNR objective F is within 2.5e-4 of the full-game RNR at every p, and value against the chump and exploitability match to about 1e-4. At p = 0.1 we win 1.079 chips per hand from the chump (the Nash wins 0.645, a best response 1.467) for an exploitability of only 0.010. At p = 0 the compound is our earlier safe re-solve bit for bit; at p = 1 it wins exactly the best response's 1.46667.

## Trade-off (mean over both seats; root solve 1000 iterations, re-solves 3000, full-game RNR 3000)

| p | Method | Value vs chump | Exploitability | F | F − F_parent | F_full − F |
|---|---|---|---|---|---|---|
| 0 | full-game RNR | +0.64503 | 2.06e-5 | −0.00002 | | |
| 0 | depth-limited B+cbv | +0.62109 | 1.55e-4 | −0.00015 | −1.9e-5 | +1.3e-4 |
| 0.1 | full-game RNR | +1.07932 | 1.010e-2 | +0.09884 | | |
| 0.1 | depth-limited B+cbv | +1.07942 | 1.039e-2 | +0.09859 | −1.7e-5 | +2.5e-4 |
| 0.5 | full-game RNR | +1.34987 | 0.1140 | +0.61794 | | |
| 0.5 | depth-limited B+cbv | +1.34986 | 0.1140 | +0.61791 | −9.8e-6 | +3.1e-5 |
| 1 | full-game RNR | +1.46667 | 2.294 | +1.46667 | | |
| 1 | depth-limited B+cbv | +1.46667 | 2.331 | +1.46667 | +3.5e-9 | +7.7e-10 |

- Exploitability is that of our strategy alone: the seat's game value minus our worst-case value (exact best response), averaged over the two seats. F = p·(value vs chump) + (1 − p)·(worst-case value).
- The chump: a best response wins +1.46667 in either seat; the reference LP Nash wins +0.64735.
- At p = 0 the two methods find different equilibria (exploitability 2e-5 and 1.5e-4); the Nash is not unique, and the value against the chump differs by 0.024.
- At p = 1 the worst case differs (2.29 vs 2.33): our strategy off the chump's path is arbitrary. The chump never raises, so 9 of 15 round-2 states have no weight and keep the parent's strategy.
- Per-seat rows are in `run.log`.

## Variants at p = 0.1 and 0.5 (mean over seats; worst seat in brackets)

| p | Variant | Exploitability | F − F_parent | F_full − F | Max exceedance over alt |
|---|---|---|---|---|---|
| 0.1 | B+cbv | 1.039e-2 | −1.7e-5 (−5.6e-5) | +2.5e-4 | +1.3e-3 |
| 0.1 | B+avg | 1.026e-2 | +9.0e-5 (+4.5e-5) | +1.4e-4 | +5.9e-3 |
| 0.1 | A+cbv | 1.128e-2 | −6.9e-4 (−1.3e-3) | +9.2e-4 | +0.23 |
| 0.5 | B+cbv | 0.1140 | −9.8e-6 (−1.9e-5) | +3.1e-5 | +2.2e-3 |
| 0.5 | B+avg | 0.1140 | −3.0e-5 (−8.1e-5) | +5.2e-5 | +0.36 |
| 0.5 | A+cbv | 0.1142 | −9.0e-5 (−1.6e-4) | +1.1e-4 | +7.9e-3 |

- Max exceedance: the free opponent's best-response value per unit at a gadget root infoset minus its alternative value, maximised over states and seats. RNR may exceed alt to gain against the model, so this is not a violation.
- Variant A loses most: −1.3e-3 in seat 0 at p = 0.1, about 25× the worst B+cbv drop, with the largest exceedance (0.23). The derivation predicts this: A prices exploitation at the parent's free reach, which is no bound.
- B+cbv changes F by at most −5.6e-5 against the parent. The guarantee is F ≥ F_parent − Z·ε for a re-solve solved to ε, and ε was not measured; the parent (1000 iterations) is already within about 2.5e-4 of the full-game RNR, so little improvement is possible.

## Runtime

- One local run, 14 min 18 s (2026-11-26 12:45:46 to 13:00:04), 12th Gen Intel i5-12450H, WSL2, numpy and JAX on CPU.
- Per p (both seats): full-game RNR 27–30 s; depth-limited walk 85–88 s per variant, 36 s at p = 1 (6 of 15 re-solves).
- A depth-limited walk costs about 3× the full-game RNR here: 30 re-solves of 3000 iterations against one full solve of 3000 iterations. Leduc is small enough to solve whole; the depth limit only pays in larger games.

## Checks

- All assertions passed (exit 0):
  1. frozen model rows bit-identical through every solve;
  2. the full and the depth-limited RNR games under random strategies are worth p·u(model) + (1 − p)·u(free) in the original game (error 8.3e-17);
  3. p = 0: compound equals `resolve_walk` bit for bit, NashConv of our two seats 3.1e-4;
  4. p = 1: value minus best-response value against the chump, 9.3e-10 (compound) and 1.6e-10 (full RNR);
  5. B+cbv: F ≥ F_parent − 1e-3 in each seat (smallest margin −5.6e-5).
- Deterministic; no seeds.
