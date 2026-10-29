"""Throttled client for the Slumbot HTTP API (adapted from ``sample_api.py``).

Three POST endpoints with JSON bodies: ``/slumbot/api/login``,
``/slumbot/api/new_hand`` and ``/slumbot/api/act``. A response carries the
whole ``action`` string of the hand so far, ``client_pos`` (0 = we are the big
blind, 1 = the small blind / button), our ``hole_cards``, the visible
``board``, and ``winnings`` (our chips) once the hand is over. The token may
change between responses, so the latest one is always kept.

Action strings: ``k`` check, ``c`` call, ``f`` fold, ``bN`` bet/raise *to* N
chips on the current street, ``/`` street separator. Blinds 50/100, stacks
20 000 that reset every hand.

Differences from the sample: errors raise ``SlumbotError`` instead of
``sys.exit``, and every request goes through ``_throttle`` so that consecutive
requests are at least ``min_interval`` seconds apart (the Slumbot author okayed
~0.5 s in slumbot2019 issue #49; 1 s by default for headroom).
"""

from __future__ import annotations

import time
from typing import Any

import requests

HOST = "slumbot.com"
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
REQUEST_TIMEOUT = 30


class SlumbotError(RuntimeError):
  pass


class SlumbotClient:
  def __init__(self, min_interval: float = 1.0, host: str = HOST) -> None:
    self.min_interval = float(min_interval)
    self.base = f"https://{host}/slumbot/api"
    self.token: str | None = None
    self.n_requests = 0
    self._last_request: float | None = None

  def _throttle(self) -> None:
    if self._last_request is not None:
      wait = self.min_interval - (time.monotonic() - self._last_request)
      if wait > 0:
        time.sleep(wait)
    self._last_request = time.monotonic()

  def _post(self, endpoint: str, data: dict) -> dict[str, Any]:
    self._throttle()
    self.n_requests += 1
    response = requests.post(f"{self.base}/{endpoint}", json=data, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
      raise SlumbotError(f"{endpoint}: HTTP {response.status_code}: {response.text[:500]}")
    try:
      r = response.json()
    except ValueError as e:
      raise SlumbotError(f"{endpoint}: response is not JSON") from e
    if "error_msg" in r:
      raise SlumbotError(f"{endpoint}: {r['error_msg']}")
    if r.get("token"):
      self.token = r["token"]
    return r

  def login(self, username: str, password: str) -> None:
    """Registered users only; anonymous play needs no login."""
    self._post("login", {"username": username, "password": password})
    if not self.token:
      raise SlumbotError("login: no token in response")

  def new_hand(self) -> dict[str, Any]:
    return self._post("new_hand", {"token": self.token} if self.token else {})

  def act(self, incr: str) -> dict[str, Any]:
    return self._post("act", {"token": self.token, "incr": incr})
