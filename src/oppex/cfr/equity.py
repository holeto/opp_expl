"""Preflop all-in equity: the only place the board enters public-state CFR.

Preflop the board is neither public nor dealt, so it is not part of the tree. It
appears exclusively at showdown terminals, folded into an expectation over every
runout:

    EV[a, b] = E_B[ sign(rank7(a ∪ B) - rank7(b ∪ B)) ],  B ⊆ deck \\ (a ∪ b), |B| = 5

Stored as **net EV** (``win% - lose%``, ties contributing 0) rather than equity,
because net EV *is* the payoff per unit stake: a showdown for ``S`` chips pays
``S * EV[a, b]`` to player 0 directly. Two properties follow and both are load-
bearing. It is antisymmetric (``EV.T == -EV``), which is a free exactness check
and lets the solver's terminal operator collapse to two matrix-vector products.
And it is stored **already multiplied by the card-removal mask**, so a showdown
terminal cannot silently forget removal.

**Cost is per board, not per pair.** The naive reading — enumerate runouts for
each of the ~1.6M hand pairs — is ~2.8e12 evaluations and hopeless. Instead each
board is dealt once, all 1326 hands are ranked against it in one vmapped call,
and the pairwise comparison is accumulated for the pairs that board is valid for.
That makes exhaustive enumeration of all C(52,5) = 2,598,960 boards merely slow
rather than impossible, and exact beats sampled: no estimator error to reason
about, plus a perfect self-check (every disjoint pair must see exactly
C(48,5) = 1,712,304 boards).

Monte Carlo shares one board sample across all pairs, so a board touching either
hand is invalid *for that pair*. Dividing by a global board count is therefore
biased; the ``C`` accumulator counts valid boards per pair, which is what makes
the estimator unbiased.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..envs.hunl_holdem import hand_rank7
from .cards import HAND_CARDS, MASK, MASK_NP, N_HANDS

N_BOARDS_EXACT = 2_598_960   # C(52, 5)
N_BOARDS_PER_PAIR = 1_712_304  # C(48, 5)

_HAND_CARDS_J = jnp.asarray(HAND_CARDS)


# ── Core accumulation ────────────────────────────────────────────────────────


@jax.jit
def _accumulate(carry, boards):
  """Fold a chunk of boards into the (S, C) accumulators.

  ``lax.scan`` keeps the ``(B, 1326, 1326)`` pairwise intermediate from ever
  being materialised — only one ``(1326, 1326)`` comparison exists at a time.
  """

  def one(carry, board):
    s, c = carry
    ranks = jax.vmap(
      lambda h: hand_rank7(jnp.concatenate([h, board]))
    )(_HAND_CARDS_J)                                              # (1326,) int32
    # A hand is unusable against a board that contains either of its cards. Such
    # hands still produce a finite (meaningless) score, so the mask, not the
    # evaluator, is what keeps them out.
    valid = ~(_HAND_CARDS_J[:, :, None] == board[None, None, :]).any(axis=(1, 2))
    vv = valid[:, None] & valid[None, :]
    sgn = jnp.sign(ranks[:, None] - ranks[None, :]).astype(jnp.int32)
    return (s + jnp.where(vv, sgn, 0), c + vv.astype(jnp.int32)), None

  (s, c), _ = jax.lax.scan(one, carry, boards)
  return s, c


def _mc_boards(rng, n):
  """``n`` uniform 5-card boards. Overlap with a hand is handled by the mask."""
  return np.argsort(rng.random((n, 52)), axis=1)[:, :5].astype(np.int32)


# ── Public entry point ───────────────────────────────────────────────────────


def preflop_ev_matrix(
  *,
  n_boards: int | None = 100_000,
  seed: int = 0,
  exact: bool = False,
  chunk: int = 256,
  cache_dir: Path | None = None,
  progress: bool = True,
  use_cache: bool = True,
  checkpoint_every: int = 20_000,
) -> jax.Array:
  """(1326, 1326) float32 net-EV matrix, antisymmetric and mask-folded.

  ``exact=True`` enumerates every board (~4h; no estimator error); otherwise
  ``n_boards`` are sampled. Results are cached to ``.npy`` beside a JSON sidecar
  recording the hand-index convention — a cache built under a different
  convention is refused rather than silently misinterpreted.

  Long runs checkpoint the raw accumulators every ``checkpoint_every`` boards and
  resume from them, so an interrupted build costs minutes rather than restarting.
  Board generation is deterministic in both modes, so resuming just means
  skipping the boards already folded in.
  """
  path, meta_path = _cache_paths(n_boards, seed, exact, cache_dir)
  if use_cache and path.exists():
    cached = _load_cache(path, meta_path)
    if cached is not None:
      return jnp.asarray(cached)

  total = N_BOARDS_EXACT if exact else int(n_boards)
  ckpt_path = path.with_suffix(".ckpt.npz")
  s, c, done = _load_checkpoint(ckpt_path, progress)
  if done >= total:
    done, s, c = 0, None, None       # a checkpoint for a longer run is not reusable
  if s is None:
    s = jnp.zeros((N_HANDS, N_HANDS), jnp.int32)
    c = jnp.zeros((N_HANDS, N_HANDS), jnp.int32)

  chunks = _board_chunks(exact, total, seed, chunk, skip=done)

  t0, start, since_ckpt = time.time(), done, 0
  for block in chunks:
    s, c = _accumulate((s, c), jnp.asarray(block))
    done += len(block)
    since_ckpt += len(block)
    if since_ckpt >= checkpoint_every and done < total:
      _save_checkpoint(ckpt_path, s, c, done)
      since_ckpt = 0
    if progress and (done % (chunk * 40) < chunk or done >= total):
      el = time.time() - t0
      eta = el / max(done - start, 1) * (total - done)
      print(f"  {done:>9,}/{total:,} boards  {el / 60:6.1f} min elapsed  "
            f"{eta / 60:6.1f} min left", flush=True)

  c_np, s_np = np.asarray(c), np.asarray(s)
  ev = np.zeros((N_HANDS, N_HANDS), np.float32)
  np.divide(s_np, c_np, out=ev, where=c_np > 0)
  ev *= MASK_NP

  _validate(ev, c_np, exact)
  if use_cache:
    _save_cache(path, meta_path, ev, n_boards, seed, exact)
    ckpt_path.unlink(missing_ok=True)
  return jnp.asarray(ev)


def _board_chunks(exact, total, seed, chunk, skip=0):
  """Deterministic board stream, resuming past the first ``skip`` boards."""
  if exact:
    it = itertools.combinations(range(52), 5)
    if skip:
      next(itertools.islice(it, skip, skip), None)
    while True:
      block = list(itertools.islice(it, chunk))
      if not block:
        return
      yield np.asarray(block, dtype=np.int32)
  else:
    rng = np.random.default_rng(seed)
    if skip:                      # advance the stream, cheap next to the equity work
      for i in range(0, skip, chunk):
        _mc_boards(rng, min(chunk, skip - i))
    for i in range(skip, total, chunk):
      yield _mc_boards(rng, min(chunk, total - i))


def _load_checkpoint(ckpt_path, progress):
  if not ckpt_path.exists():
    return None, None, 0
  try:
    z = np.load(ckpt_path)
    done = int(z["done"])
    if progress:
      print(f"  resuming from checkpoint at {done:,} boards", flush=True)
    return jnp.asarray(z["s"]), jnp.asarray(z["c"]), done
  except Exception as e:                              # corrupt/partial write
    print(f"  ignoring unreadable checkpoint {ckpt_path.name}: {e}")
    return None, None, 0


def _save_checkpoint(ckpt_path, s, c, done):
  tmp = ckpt_path.with_suffix(".tmp.npz")
  np.savez(tmp, s=np.asarray(s), c=np.asarray(c), done=done)
  tmp.replace(ckpt_path)          # atomic: never leave a half-written checkpoint


# ── Validation ───────────────────────────────────────────────────────────────


def _validate(ev, counts, exact):
  asym = np.abs(ev + ev.T).max()
  assert asym == 0.0, f"EV must be antisymmetric, max |EV + EV.T| = {asym}"
  assert np.abs(ev[~MASK_NP.astype(bool)]).max(initial=0.0) == 0.0, (
    "overlapping hand pairs must have zero EV"
  )
  if exact:
    disjoint = MASK_NP.astype(bool)
    bad = np.unique(counts[disjoint])
    assert bad.tolist() == [N_BOARDS_PER_PAIR], (
      f"every disjoint pair must see exactly {N_BOARDS_PER_PAIR:,} boards, got {bad}"
    )


# ── Cache ────────────────────────────────────────────────────────────────────


def _convention_hash() -> str:
  """Fingerprint of the hand-index convention this matrix is expressed in."""
  return hashlib.sha256(HAND_CARDS.tobytes()).hexdigest()[:16]


def _cache_paths(n_boards, seed, exact, cache_dir):
  base = Path(cache_dir) if cache_dir else Path(
    os.environ.get("OPPEX_CACHE", Path.home() / ".cache" / "oppex")
  )
  base.mkdir(parents=True, exist_ok=True)
  name = "preflop_ev_exact" if exact else f"preflop_ev_mc_n{int(n_boards)}_s{seed}"
  return base / f"{name}.npy", base / f"{name}.json"


def _load_cache(path, meta_path):
  if not meta_path.exists():
    return None
  meta = json.loads(meta_path.read_text())
  if meta.get("convention") != _convention_hash():
    print(f"  ignoring {path.name}: built under a different hand-index convention")
    return None
  return np.load(path)


def _save_cache(path, meta_path, ev, n_boards, seed, exact):
  np.save(path, ev)
  meta_path.write_text(json.dumps({
    "convention": _convention_hash(),
    "n_boards": N_BOARDS_EXACT if exact else int(n_boards),
    "seed": seed,
    "exact": exact,
    "jax_version": jax.__version__,
  }, indent=2))


# ── Reference path (slow, for spot-checking) ─────────────────────────────────


def exact_pair_ev(hand_a: tuple[int, int], hand_b: tuple[int, int]) -> float:
  """Net EV for one specific matchup by exhaustive enumeration of its runouts.

  Independent of the matrix builder — used to prove that pipeline correct. Note
  the result is *suit-specific*: comparing it to a published class average (which
  averages over suit combinations) looks like a 2-3% discrepancy when nothing is
  wrong.
  """
  used = set(hand_a) | set(hand_b)
  deck = np.array([c for c in range(52) if c not in used], dtype=np.int32)
  boards = np.asarray(list(itertools.combinations(deck, 5)), dtype=np.int32)

  @jax.jit
  def go(boards):
    ra = jax.vmap(lambda b: hand_rank7(jnp.concatenate([jnp.asarray(hand_a), b])))(boards)
    rb = jax.vmap(lambda b: hand_rank7(jnp.concatenate([jnp.asarray(hand_b), b])))(boards)
    return jnp.sign(ra - rb).astype(jnp.float32).mean()

  return float(go(jnp.asarray(boards)))
