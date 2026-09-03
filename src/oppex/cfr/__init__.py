"""Public-state CFR for heads-up no-limit hold'em.

Layer above ``oppex.envs``: walks a tree of publicly observable actions, keeping
regrets and strategies at each node for *every* private holding at once, rather
than sampling one deal at a time. The betting rules come from
``oppex.envs.betting`` so the solver and the environment cannot drift apart.
"""

from .cards import HAND_CARDS, HAND_CLASS, MASK, N_DEALS, N_HANDS, N_OPP, hand_label

__all__ = [
  "HAND_CARDS", "HAND_CLASS", "MASK", "N_DEALS", "N_HANDS", "N_OPP", "hand_label",
]
