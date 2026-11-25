# Safe depth-limited re-solving on Goofspiel 3 (descending)

Details: `notes/safe_resolving.md`, section "2026-11-25: nested safe re-solving on Goofspiel 3".

**Result: the compound strategy converges to the pure equilibrium.** NashConv falls from 1.2e-2 at 10 DCFR iterations per solve to 5.0e-10 at 3000, and its probability off the highest card falls with it. The general PSCFR and vanilla CFR give bit-identical compound strategies at every budget.

## Leaf checks (depth-1 cut at the root)

| Check | Result |
|---|---|
| Leaf matrices against the rules | largest error 0 |
| Depth-1 root solve, 2000 iterations | value +2.8e-10; depth-limited NashConv 1.3e-9; full-game NashConv of root strategy + the behaviour its leaf mixtures induce 1.3e-9 |
| Player 1's best response to player 0's root mix (1/3, 0, 2/3), player 0 High below | exact leaves +0.3333 (= full game); single highest-card leaf 0.0000 |

The single-continuation leaf accepts an exploitable root mix as an equilibrium. The exact leaves do not.

## Re-solving walk, reach threshold 0.05

| DCFR iterations per solve | NashConv (compound) | Largest probability off the highest card | Max safety (per unit) | PSCFR vs vanilla |
|---|---|---|---|---|
| 10 | 1.16e-2 | 4.70e-3 | +2.60e-3 | identical |
| 30 | 4.73e-4 | 1.91e-4 | +1.06e-4 | identical |
| 100 | 1.32e-5 | 5.35e-6 | +2.96e-6 | identical |
| 300 | 4.95e-7 | 2.00e-7 | +1.11e-7 | identical |
| 1000 | 1.34e-8 | 5.42e-9 | +3.00e-9 | identical |
| 3000 | 4.97e-10 | 2.01e-10 | +1.11e-10 | identical |

- Re-solved at every budget: the root, then one public state per turn (both played 3, a single history), reach 0.99-1.00.
- Safety: the opponent's best-response value per unit at each gadget root infoset, minus its alternative value. Positive values shrink with the budget, as the parent's averaged values approach a best response.

## Gadget stress test: root solve cut short, 1000 iterations below

| Root iterations | NashConv of the root solve's full strategy | NashConv (compound) | Multi-history states re-solved | Max safety, turn 1 | Max safety, turn 2 |
|---|---|---|---|---|---|
| 1 | 2.667 | 2.000 | 4 | +2.00 | +1.4e-8 |
| 2 | 0.695 | 0.620 | 4 | +2.00 | +3.0e-9 |
| 3 | 0.248 | 0.233 | 2 | +0.61 | +1.5e-14 |
| 5 | 0.0632 | 0.0632 | 0 | +9.1e-3 | 0 |

- Re-solving never made the strategy more exploitable than the root solve it refines.
- Gadgets below a converged parent (turn 2) hold the opponent to their alternative values.
- Gadgets below a few-iteration root (turn 1) do not: those alternative values are the opponent's value under their own unconverged continuation, not a best response, so no strategy meets them.

## Checks

- One local run, completed, 5 min 47 s. Deterministic; a second run gave the same numbers.
- All assertions passed: NashConv below 1e-2 at the largest budget, non-increasing past 300 iterations, compound no more exploitable than the root solve, turn-2 safety below 1e-3.
