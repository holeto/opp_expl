# Safe re-solving of Leduc round 2, against an independent Nash

Details: `notes/safe_resolving.md`, section "2026-11-25: safe re-solving of Leduc round 2, against an independent Nash".

**Result: the compound strategy converges to an equilibrium.** Its NashConv falls from 0.88 at 10 DCFR iterations per solve to 1.7e-4 at 3000. Round 1 converges onto the reference Nash (TV 5e-5). Round 2 stays about 1% from it, because round-2 equilibria are not unique given the ranges. Against the reference the compound loses 1.2e-6 chips. Re-solving without a gadget gives NashConv 0.10.

## Reference and leaf checks

| Check | Result |
|---|---|
| Reference in our env | 936 card rows → 288 infosets (suit asymmetry 3.7e-7); value −0.0856064 (file −0.0856064); NashConv 1.9e-8 |
| Depth-limited root game | 361 histories, 120 leaves (5×9 each), 90 leaf infosets |
| 1. Leaf values under the reference (Kuhn mixtures) vs the full game below | largest error 3.6e-15 |
| 2. Reference round 1 in the depth-limited game | value −0.0856064, NashConv 1.9e-8 |
| 3. Depth-limited root solve, 3000 iterations | value −0.0856062; NashConv 3.0e-5 (depth-limited and extended full-game) |
| 4. Vanilla CFR's step vs PSCFR's (DCFR, CFR+, vanilla; root game and an 8-history gadget) | largest relative error 7.1e-16 |

## Re-solving sweep (all 15 round-2 public states re-solved at every budget)

| Iterations per solve | Solver | Root-solve NC | Compound NC | Max safety | TV round 1 | TV round 2 | Loss vs reference |
|---|---|---|---|---|---|---|---|
| 10 | PSCFR | 0.795 | 0.875 | +2.42 | 0.226 | 0.214 | 0.223 |
| 10 | vanilla | 0.795 | 0.875 | +2.42 | 0.226 | 0.214 | 0.223 |
| 30 | PSCFR | 0.106 | 0.0987 | +0.441 | 0.0561 | 0.0804 | 0.0255 |
| 30 | vanilla | 0.106 | 0.0987 | +0.441 | 0.0561 | 0.0804 | 0.0255 |
| 100 | PSCFR | 1.77e-2 | 1.66e-2 | +0.136 | 1.23e-2 | 2.57e-2 | 1.26e-3 |
| 100 | vanilla | 1.77e-2 | 1.64e-2 | +0.143 | 1.23e-2 | 2.52e-2 | 1.29e-3 |
| 300 | PSCFR | 3.22e-3 | 3.79e-3 | +6.93e-2 | 1.35e-3 | 1.23e-2 | 9.39e-5 |
| 300 | vanilla | 3.53e-3 | 3.54e-3 | +6.31e-2 | 1.40e-3 | 1.21e-2 | 1.03e-4 |
| 1000 | PSCFR | 2.71e-4 | 5.65e-4 | +5.88e-3 | 4.87e-4 | 9.18e-3 | 2.90e-5 |
| 1000 | vanilla | 2.38e-4 | 5.62e-4 | +4.66e-3 | 4.38e-4 | 8.74e-3 | 1.04e-5 |
| 3000 | PSCFR | 2.99e-5 | 1.67e-4 | +6.66e-4 | 5.10e-5 | 1.11e-2 | 1.23e-6 |
| 3000 | vanilla | 3.36e-5 | 1.27e-4 | +5.98e-4 | 2.82e-5 | 1.29e-2 | 2.16e-7 |

- Root-solve NC: the root solve's round 1 plus the behaviour its leaf mixtures induce in round 2. Max safety: the opponent's best-response value per unit at a gadget root infoset minus its alternative value, maximised over all 30 gadgets.
- TV: per-infoset total variation from the reference, averaged with the reference's joint reach. Loss vs reference: chips lost to the reference, summed over both seats (at most the NashConv).
- PSCFR and vanilla print the same numbers up to 30 iterations (their compound strategies differed by 5e-13 at 10 in an exploratory run), then drift apart through roundoff amplification. They take the same step (check 4).

## Control: re-solving without a gadget

The reference's round 1, with each round-2 public state solved with both players free from the reference's ranges (10000 iterations): full-game NashConv **0.101**, round-2 TV from the reference 1.8e-2.

## Diagnostic: re-solve budget and alternative values (root solve fixed at 1000 iterations, NC 2.71e-4)

| Re-solve iterations | Alternative values | Compound NC | Max safety |
|---|---|---|---|
| 1000 | averaged cfv | 5.65e-4 | +5.88e-3 |
| 1000 | CBV | 6.06e-4 | +3.14e-3 |
| 3000 | averaged cfv | 3.03e-4 | +5.88e-3 |
| 3000 | CBV | 3.10e-4 | +1.47e-3 |
| 10000 | averaged cfv | 2.07e-4 | +5.88e-3 |
| 10000 | CBV | 2.40e-4 | +2.99e-4 |

At an equal budget the compound trails the root solve (main sweep, from 300 iterations on). With 10× the root's iterations it beats the root with either kind of alternative value. The CBV safety gap shrinks with the budget. The averaged-cfv gap is fixed by the root solve.

## Time per iteration (alternating DCFR, numpy on CPU)

Measured after the run with a separate timing script (5 warm-up steps, then 200 steps on the root game and the largest gadget, 100 on each of the 30 gadgets of a 300-iteration root solve):

| Game | Histories | Public states | PSCFR ms | Vanilla ms | Ratio |
|---|---|---|---|---|---|
| Depth-limited root game | 361 | 26 | 4.30 | 3.04 | 1.42 |
| Largest gadget | 137 | 14 | 1.99 | 2.21 | 0.90 |
| All 30 gadgets together | | | 56.1 | 76.4 | 0.73 |

The public-state closure takes 0.05 s for all 30 gadgets. The sweep's wall times agree: at 3000 iterations a full walk (solves, best responses, safety) took 140 s with PSCFR and 187 s with vanilla CFR. Unlike Goofspiel, where vanilla CFR is 2.7-4.9× faster, Leduc's round-2 public states hold many infosets each (one per own rank and history), so PSCFR's per-state vectorisation pays off.

## Checks

- One local run, completed, 36 min 57 s. Deterministic: every number equals the exploratory runs of the same code earlier that day.
- The first diagnostic row took 688 s of wall time against about 40 s for its CBV twin and in the exploratory run; the machine was presumably busy or asleep. The numbers are unaffected.
- All assertions passed: reference NashConv < 1e-6 and value within 1e-6, leaf error < 1e-12, reference NashConv in the depth-limited game < 1e-6, degenerate solve within 1e-3, solver step agreement < 1e-10, final compound NashConv < 1e-2, control NashConv > 1e-2.
