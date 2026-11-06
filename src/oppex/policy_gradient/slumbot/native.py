"""ctypes binding of the C++ Slumbot API (``csrc/slumbot_api``).

The library answers in slumbot.com's own terms: action strings such as
``"b200c/kb300"``, chips at 50/100 blinds with 20 000 stacks, and ACPC
positions (0 = big blind, 1 = small blind). Converting from the env is
``model.py``'s job. See ``csrc/slumbot_api/slumbot_api.h`` for the contract.

The library and the strategy data are found through two environment
variables, since the data (~155 GB) lives outside the repository:

  * ``SLUMBOT_DIR``: the ``slumbot_query`` checkout (parameter files; the data
    defaults to ``$SLUMBOT_DIR/data``, or ``$SLUMBOT_DATA``);
  * ``SLUMBOT_API_LIB``: the library, by default
    ``csrc/slumbot_api/build/libslumbot_api.so`` in this repository.

The C++ agent keeps its configuration in globals and file handles it seeks per
lookup, so there is one instance per process (``get``) and it must not be
called from two threads at once.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

SB_OK, SB_ERR_ARGS, SB_ERR_ACTION, SB_ERR_NOT_TO_ACT = 0, -1, -2, -3
SB_ERR_OFF_TREE, SB_ERR_CAPACITY, SB_ERR_INTERNAL = -4, -5, -6
SB_FOLD, SB_CHECK, SB_CALL, SB_BET = range(4)
STATUS_NAMES = {SB_ERR_ARGS: "bad arguments", SB_ERR_ACTION: "illegal action string",
                SB_ERR_NOT_TO_ACT: "Slumbot is not to act",
                SB_ERR_OFF_TREE: "a past Slumbot action is off its tree",
                SB_ERR_CAPACITY: "output buffers too small", SB_ERR_INTERNAL: "internal error"}

# CFR iterations of the shipped SB / BB strategies.
P1_ITERATIONS, P2_ITERATIONS = 16384, 22528
# Successors of an abstract node, generously: the trees have at most 4.
MAX_OUT = 8

_REPO = Path(__file__).resolve().parents[4]
_DEFAULT_LIB = _REPO / "csrc" / "slumbot_api" / "build" / "libslumbot_api.so"

_INT_P = ctypes.POINTER(ctypes.c_int)
_DBL_P = ctypes.POINTER(ctypes.c_double)


class SlumbotStatusError(RuntimeError):
  """A non-zero status from the library, with the call's arguments."""

  def __init__(self, status: int, what: str) -> None:
    super().__init__(f"{what}: {STATUS_NAMES.get(status, status)} ({status})")
    self.status = status


class Query(NamedTuple):
  """Slumbot's actions for ``n`` hands at one state.

  Row ``h`` lists the successors of Slumbot's abstract node for hand ``h``:
  ``kind`` (``SB_*``), ``bet`` (the street total of a bet, else 0) and
  ``prob``; entries past ``n_out[h]`` are padding. ``n_out[h] == 0`` marks a
  hand that conflicts with the board.
  """

  n_out: np.ndarray  # (n,) int32
  kind: np.ndarray   # (n, MAX_OUT) int32
  bet: np.ndarray    # (n, MAX_OUT) int32
  prob: np.ndarray   # (n, MAX_OUT) float64


class Cell(NamedTuple):
  """A set of translation types ``u`` on which Slumbot's path is constant."""

  feasible: bool
  mass: float
  u: float           # a point inside, to query with
  n_draws: int
  intervals: tuple   # ((lo, hi), ...), disjoint and sorted


class SlumbotNative:
  """The loaded agent. Use ``get()`` rather than constructing it."""

  def __init__(self, slumbot_dir: str, lib_path: str | None = None) -> None:
    lib = ctypes.CDLL(str(lib_path or os.environ.get("SLUMBOT_API_LIB", _DEFAULT_LIB)))
    lib.sb_create.restype = ctypes.c_void_p
    lib.sb_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_int]
    lib.sb_destroy.argtypes = [ctypes.c_void_p]
    lib.sb_u_from_hand_no.restype = ctypes.c_double
    lib.sb_u_from_hand_no.argtypes = [ctypes.c_int]
    lib.sb_query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, _INT_P,
                             ctypes.c_int, _INT_P, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                             _INT_P, _INT_P, _INT_P, _DBL_P]
    lib.sb_act_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, _INT_P, ctypes.c_char_p, _INT_P,
                                 _INT_P, _DBL_P, _DBL_P, _INT_P, _INT_P, _INT_P]
    lib.sb_cells.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_int, _INT_P, _INT_P, _DBL_P, _DBL_P, _INT_P, _INT_P,
                             _INT_P, _DBL_P, _DBL_P]
    self._lib = lib
    err = ctypes.create_string_buffer(1024)
    self._h = lib.sb_create(str(slumbot_dir).encode(), P1_ITERATIONS, P2_ITERATIONS, err, 1024)
    if not self._h:
      raise RuntimeError(f"sb_create failed: {err.value.decode(errors='replace')}")

  def u_from_hand_no(self, hand_no: int) -> float:
    """The translation type of live hand number ``hand_no``."""
    return float(self._lib.sb_u_from_hand_no(int(hand_no)))

  def query(self, position: int, actions: str, board, hands: np.ndarray, u: float) -> Query:
    """Slumbot's actions at ``actions`` for each row of ``hands`` ``(n, 2)``.

    Raises ``SlumbotStatusError`` for a state the library rejects.
    """
    hands = np.ascontiguousarray(hands, np.int32).reshape(-1, 2)
    n = hands.shape[0]
    bd = np.zeros(5, np.int32)
    bd[: len(board)] = board
    out = Query(np.zeros(n, np.int32), np.zeros((n, MAX_OUT), np.int32),
                np.zeros((n, MAX_OUT), np.int32), np.zeros((n, MAX_OUT), np.float64))
    st = self._lib.sb_query(
      self._h, int(position), actions.encode(), _ptr(bd, _INT_P), len(board),
      _ptr(hands, _INT_P), n, float(u), MAX_OUT, _ptr(out.n_out, _INT_P),
      _ptr(out.kind, _INT_P), _ptr(out.bet, _INT_P), _ptr(out.prob, _DBL_P))
    if st != SB_OK:
      raise SlumbotStatusError(st, f"sb_query(position {position}, {actions!r}, u {u})")
    return out

  def act_batch(self, position, actions: list[str], boards, hands, u, r):
    """Slumbot's action in each of ``n`` independent games (for rollouts).

    Game ``g`` is ``actions[g]`` with ``position[g]``, the full board
    ``boards[g]`` ``(5,)``, hand ``hands[g]`` ``(2,)``, translation type
    ``u[g]`` and the uniform ``r[g]`` the action is drawn with. Returns
    ``(kind, bet, status)``, each ``(n,)``; a game with ``status != 0`` has no
    action.
    """
    n = len(actions)
    pos = np.ascontiguousarray(np.broadcast_to(position, (n,)), np.int32)
    buf = b"".join(a.encode() + b"\0" for a in actions)
    boards = np.ascontiguousarray(boards, np.int32).reshape(n, 5)
    hands = np.ascontiguousarray(hands, np.int32).reshape(n, 2)
    u = np.ascontiguousarray(u, np.float64)
    r = np.ascontiguousarray(r, np.float64)
    kind, bet, status = (np.zeros(n, np.int32) for _ in range(3))
    st = self._lib.sb_act_batch(self._h, n, _ptr(pos, _INT_P), buf, _ptr(boards, _INT_P),
                                _ptr(hands, _INT_P), _ptr(u, _DBL_P), _ptr(r, _DBL_P),
                                _ptr(kind, _INT_P), _ptr(bet, _INT_P), _ptr(status, _INT_P))
    if st != SB_OK:
      raise SlumbotStatusError(st, f"sb_act_batch({n} games)")
    return kind, bet, status

  def cells(self, position: int, actions: str) -> list[Cell]:
    """The partition of ``u`` in ``[0, 1)`` by the translations along ``actions``."""
    max_cells, max_iv = 64, 1024
    while True:
      nc, ni = ctypes.c_int(), ctypes.c_int()
      feas, draws = np.zeros(max_cells, np.int32), np.zeros(max_cells, np.int32)
      mass, cu = np.zeros(max_cells), np.zeros(max_cells)
      icell = np.zeros(max_iv, np.int32)
      lo, hi = np.zeros(max_iv), np.zeros(max_iv)
      st = self._lib.sb_cells(
        self._h, int(position), actions.encode(), max_cells, max_iv, ctypes.byref(nc),
        _ptr(feas, _INT_P), _ptr(mass, _DBL_P), _ptr(cu, _DBL_P), _ptr(draws, _INT_P),
        ctypes.byref(ni), _ptr(icell, _INT_P), _ptr(lo, _DBL_P), _ptr(hi, _DBL_P))
      if st == SB_ERR_CAPACITY:
        max_cells, max_iv = 4 * max_cells, 4 * max_iv
        continue
      if st != SB_OK:
        raise SlumbotStatusError(st, f"sb_cells(position {position}, {actions!r})")
      break
    ivs: list[list] = [[] for _ in range(nc.value)]
    for i in range(ni.value):
      ivs[icell[i]].append((float(lo[i]), float(hi[i])))
    return [Cell(bool(feas[c]), float(mass[c]), float(cu[c]), int(draws[c]), tuple(ivs[c]))
            for c in range(nc.value)]


def _ptr(a: np.ndarray, kind):
  return a.ctypes.data_as(kind)


_INSTANCE: SlumbotNative | None = None


def get() -> SlumbotNative:
  """The process-wide agent, loaded from ``$SLUMBOT_DIR`` on first use."""
  global _INSTANCE
  if _INSTANCE is None:
    d = os.environ.get("SLUMBOT_DIR")
    if not d:
      raise RuntimeError("set SLUMBOT_DIR to the slumbot_query checkout")
    _INSTANCE = SlumbotNative(d)
  return _INSTANCE
