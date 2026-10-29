"""Build (env, agent, algorithm) from a parsed YAML config dict.

Shared by ``scripts/train.py`` and the evaluation scripts, which rebuild the
same env and agent from the ``config.yaml`` saved next to the checkpoints.
See ``scripts/train.py`` for the config layout.
"""

from __future__ import annotations

from typing import Any

import optax

from .agents import PolicyQAgent, PrivilegedPolicyQAgent
from .policy_gradient.algorithms import QMMD, inject_learning_rate
from .envs import build_env
from .networks import build_network


_ALGORITHMS = {"qmmd": QMMD}

_AGENTS = {
  "policy_q": PolicyQAgent,
  "privileged_policy_q": PrivilegedPolicyQAgent,
}

# Each entry gives the optax factory and its kwargs other than the learning rate.
_OPTIMIZERS = {
  "adam": lambda c: (optax.adam, {"b1": c.get("b1", 0.9), "b2": c.get("b2", 0.999)}),
  "adamw": lambda c: (optax.adamw, {
    "b1": c.get("b1", 0.9), "b2": c.get("b2", 0.999), "weight_decay": c.get("weight_decay", 1e-4),
  }),
  "sgd": lambda c: (optax.sgd, {"momentum": c.get("momentum", 0.0)}),
}

_SCHEDULES = {
  "cosine_decay": lambda c: optax.cosine_decay_schedule(
    c["init_value"], c["decay_steps"], alpha=c.get("alpha", 0.0)
  ),
  "warmup_cosine_decay": lambda c: optax.warmup_cosine_decay_schedule(
    c.get("init_value", 0.0), c["peak_value"], c["warmup_steps"], c["decay_steps"],
    end_value=c.get("end_value", 0.0),
  ),
  "linear": lambda c: optax.linear_schedule(c["init_value"], c["end_value"], c["transition_steps"]),
  "exponential_decay": lambda c: optax.exponential_decay(
    c["init_value"], c["transition_steps"], c["decay_rate"], end_value=c.get("end_value")
  ),
  "constant": lambda c: optax.constant_schedule(c["value"]),
}

# Schedule lengths that may be left ``null`` in the config and are then filled
# with the run length (see ``_fill_schedule_steps``).
_STEP_KEYS = ("decay_steps", "transition_steps")


# ── Builders ─────────────────────────────────────────────────────────────────


def build_schedule(spec: dict, total_steps: int | None = None):
  """step → value from a ``{type: ..., ...}`` spec; ``null`` lengths become ``total_steps``."""
  spec = _fill_schedule_steps(spec, total_steps)
  if spec["type"] not in _SCHEDULES:
    raise ValueError(f"Unknown schedule '{spec['type']}'. Choose from: {list(_SCHEDULES)}")
  # PyYAML reads ``3e-4`` (no dot) as a string; coerce every numeric-looking field.
  return _SCHEDULES[spec["type"]]({k: _number(v) for k, v in spec.items()})


def build_optimizer(cfg: dict, total_updates: int | None = None) -> optax.GradientTransformation:
  """Optimizer from config; ``lr`` is a number or a schedule spec.

  A schedule is evaluated by optax on its own update counter, i.e. once per
  gradient step: ``n_epochs`` times per training iteration. ``total_updates``
  fills a ``decay_steps: null`` so a cosine decay ends exactly at the last update.

  The learning rate is always injected into the optimizer state
  (``inject_learning_rate``), so ``QMMD`` logs the rate actually applied as the
  ``lr`` metric, constant or scheduled alike.
  """
  kind = cfg.get("type", "adam").lower()
  if kind not in _OPTIMIZERS:
    raise ValueError(f"Unknown optimizer '{kind}'. Choose from: {list(_OPTIMIZERS)}")
  lr = cfg["lr"]
  lr = build_schedule(lr, total_updates) if isinstance(lr, dict) else float(lr)
  factory, kwargs = _OPTIMIZERS[kind](cfg)
  opt = inject_learning_rate(factory, lr, **kwargs)
  if cfg.get("max_grad_norm") is not None:
    opt = optax.chain(optax.clip_by_global_norm(float(cfg["max_grad_norm"])), opt)
  return opt


def build_schedules(cfg: dict, total_steps: int | None = None) -> dict:
  """Coefficient schedules for QMMD; evaluated on the training step."""
  return {key: build_schedule(spec, total_steps) for key, spec in cfg.items()}


def _fill_schedule_steps(spec: dict, total: int | None) -> dict:
  missing = [k for k in _STEP_KEYS if k in spec and spec[k] is None]
  if missing and total is None:
    raise ValueError(f"Schedule {spec} leaves {missing} null but the run length is unknown")
  return {k: total if k in missing else v for k, v in spec.items()}


def _number(v: Any) -> Any:
  if isinstance(v, str):
    try:
      return float(v)
    except ValueError:
      return v
  return v


def _fill_env_dims(cfg: Any, env) -> Any:
  """Recursively replace ``n_actions: null`` with ``env.num_actions``."""
  if not isinstance(cfg, dict):
    return cfg
  return {
    k: env.num_actions if (k == "n_actions" and v is None) else _fill_env_dims(v, env)
    for k, v in cfg.items()
  }


def build_from_config(cfg: dict):
  """(env, agent, algorithm) from a parsed config dict."""
  env = build_env(cfg["env"])

  alg_cfg = dict(cfg["algorithm"])
  alg_type = alg_cfg.pop("type").lower()
  agent_cfg = dict(alg_cfg.pop("agent"))
  if alg_type not in _ALGORITHMS:
    raise ValueError(f"Unknown algorithm '{alg_type}'. Choose from: {list(_ALGORITHMS)}")
  agent_type = agent_cfg.get("type", "policy_q")
  if agent_type not in _AGENTS:
    raise ValueError(f"Unknown agent '{agent_type}'. Choose from: {list(_AGENTS)}")
  agent = _AGENTS[agent_type](build_network(_fill_env_dims(agent_cfg["network"], env)))

  # Run length for ``null`` schedule lengths; absent when only rebuilding for evaluation.
  n_steps = (cfg.get("trainer") or {}).get("n_steps")
  n_updates = n_steps * alg_cfg.get("n_epochs", 1) if n_steps is not None else None
  opt_cfg = alg_cfg.pop("optimizer", None)
  optimizer = build_optimizer(opt_cfg, n_updates) if opt_cfg is not None else None
  schedules = build_schedules(alg_cfg.pop("schedules", None) or {}, n_steps)
  algorithm = _ALGORITHMS[alg_type](
    env=env, agent=agent, optimizer=optimizer, schedules=schedules, **alg_cfg
  )
  return env, agent, algorithm
