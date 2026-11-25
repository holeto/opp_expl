"""Public-state CFR for any two-player zero-sum game with perfect recall.

The poker solver in ``oppex.cfr.solver`` relies on poker structure: one
shared 1326-hand axis, card removal through ``MASK`` and a betting tree. This
package drops all of it. A game is a ``PublicGame``, a tree of public states,
each with its own infoset axis per player, and terminals that pay through a
per-state matrix. ``from_poker`` writes the poker tree in this form, and
``oppex.tree.public`` writes any extracted history tree in it.
"""

from .public_game import DECISION, LEAF, TERMINAL, PublicGame, PublicState

__all__ = ["DECISION", "LEAF", "TERMINAL", "PublicGame", "PublicState"]
