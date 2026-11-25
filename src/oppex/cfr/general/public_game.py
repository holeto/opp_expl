"""A game as a tree of public states, with an infoset axis per player.

Public-state CFR keeps, at every public state ``s``, one reach and one cfv
vector per player, indexed by that player's infosets in ``s``. Poker has the
same 1326 hands everywhere; in general the sets differ from state to state
(in Goofspiel an infoset is the player's own card sequence, so how many there
are depends on the public results). Everything the solver needs is therefore
spelled out per state:

* **Transitions** are index maps. A child infoset has exactly one parent pair
  (infoset, own action) by perfect recall, so ``map_i[j']`` is the parent's
  flat index ``j * A + a`` if player ``i`` acts in the parent, and ``j`` if
  they do not. Reaches go down by gathering through it, values go up by
  summing into it. Several child infosets may share a parent, when private
  chance happens in between.
* **Terminals** pay ``U[j0, j1]`` chips to player 0: the sum, over the
  histories in which the players hold infosets ``j0`` and ``j1``, of chance
  reach times payoff. It is the general ``MASK``/``EV``, and as there, joint
  consistency of the two infosets (card removal) lives only here.
* **Leaves** are depth-limit states where both players pick one of their
  continuations at once: ``V[t0, t1, j0, j1]`` chips to player 0, folded with
  chance reach like ``U``, plus ``W[j0, j1]``, the chance-weighted incidence
  alone. ``W`` turns an opponent reach into the mass a cfv is measured against.
  ``t_legal`` masks the continuations an infoset does not have.

Reaches exclude chance; it enters only through ``U``, ``V`` and ``W``, which is
why a public chance state needs no special handling: nobody acts, every map is
the identity, and values just sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DECISION, TERMINAL, LEAF = 0, 1, 2


@dataclass
class PublicState:
  """One public state; see the module docstring for the field conventions.

  ``legal[i]`` is ``(n_i, A)`` and only read when player ``i`` acts.
  ``children`` holds ``(child id, map0, map1)``. ``keys[i]`` names player
  ``i``'s infosets (bytes for an extracted tree, the hand index in poker), so
  strategies can be read back.
  """

  kind: int
  n_inf: tuple[int, int]
  acting: tuple[bool, bool] = (False, False)
  legal: tuple[np.ndarray, np.ndarray] | None = None
  children: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
  U: np.ndarray | None = None
  V: np.ndarray | None = None
  W: np.ndarray | None = None
  t_legal: tuple[np.ndarray, np.ndarray] | None = None
  keys: tuple[list, list] | None = None
  ref: Any = None


@dataclass
class PublicGame:
  """Public states in topological order, the root first.

  ``root_reach`` is ``(r0, r1)`` over the root's infosets: all-ones for a game
  from its start, the ranges for a subgame.
  """

  states: list[PublicState]
  n_actions: int
  root_reach: tuple[np.ndarray, np.ndarray]
  _layout: Any = field(default=None, repr=False)

  def check(self) -> None:
    """Shapes, topological order, and one parent per child state."""
    parent = {}
    for s, st in enumerate(self.states):
      n0, n1 = st.n_inf
      if st.kind == TERMINAL:
        assert st.U is not None and st.U.shape == (n0, n1), s
      elif st.kind == LEAF:
        assert st.V.ndim == 4 and st.V.shape[2:] == (n0, n1), (s, st.V.shape)
        assert st.W.shape == (n0, n1)
        assert st.t_legal[0].shape == (n0, st.V.shape[0])
        assert st.t_legal[1].shape == (n1, st.V.shape[1])
        assert st.t_legal[0].any(-1).all() and st.t_legal[1].any(-1).all()
      for c, m0, m1 in st.children:
        assert c > s, f"state {c} is a child of {s} but comes before it"
        assert c not in parent, f"state {c} has two parents"
        parent[c] = s
        for i, m in enumerate((m0, m1)):
          width = st.n_inf[i] * (self.n_actions if st.acting[i] else 1)
          assert m.shape == (self.states[c].n_inf[i],) and (m < width).all() and (m >= 0).all()
          if st.acting[i]:
            # A child infoset must come from a legal action.
            assert st.legal[i].reshape(-1)[m].all(), f"state {c} reached by an illegal action"
    assert self.root_reach[0].shape == (self.states[0].n_inf[0],)
    assert self.root_reach[1].shape == (self.states[0].n_inf[1],)
