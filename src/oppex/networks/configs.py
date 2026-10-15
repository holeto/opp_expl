"""Network config registry.

Each module type registers a builder under a string key; ``build_network``
dispatches on ``type`` and recursively builds nested torso/head dicts. Adding a
module means decorating one builder function:

    @register_torso("my_torso")
    def _(feature_dim: int = 256, my_param: int = 42):
      return MyTorso(feature_dim=feature_dim, my_param=my_param)

YAML usage::

    network = build_network(cfg["network"])
    params = network.init_params(key, dummy_obs)
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any

import flax.linen as nn

from .composite import SplitTwinHead, TwinHead
from .heads import CategoricalHead, QHead
from .modules import Activation, Normalization
from .torsos import MLPTorso


# ── Registries ───────────────────────────────────────────────────────────────

_torso_registry: dict[str, Callable] = {}
_head_registry: dict[str, Callable] = {}
_network_registry: dict[str, Callable] = {}


def _register(registry: dict[str, Callable], name: str) -> Callable:
  def decorator(fn: Callable) -> Callable:
    registry[name] = fn
    return fn

  return decorator


def register_torso(name: str) -> Callable:
  return _register(_torso_registry, name)


def register_head(name: str) -> Callable:
  return _register(_head_registry, name)


def register_network(name: str) -> Callable:
  return _register(_network_registry, name)


# ── Builders ─────────────────────────────────────────────────────────────────


@register_torso("mlp")
def _(feature_dim: int = 256, hidden=(256, 256), activation: str = "relu", norm: str = "none"):
  return MLPTorso(
    feature_dim=feature_dim,
    hidden=tuple(hidden),
    activation=Activation(kind=activation),
    norm=Normalization(kind=norm),
  )


@register_head("categorical")
def _(n_actions: int):
  return CategoricalHead(n_actions=n_actions)


@register_head("q")
def _(n_actions: int, output_scale: float = 0.01):
  return QHead(n_actions=n_actions, output_scale=output_scale)


@register_network("twin_head")
def _(torso: dict, head1: dict, head2: dict):
  return TwinHead(torso=build_torso(torso), head1=build_head(head1), head2=build_head(head2))


@register_network("split_twin_head")
def _(actor_torso: dict, actor_head: dict, critic_torso: dict, critic_head: dict):
  return SplitTwinHead(
    actor_torso=build_torso(actor_torso),
    actor_head=build_head(actor_head),
    critic_torso=build_torso(critic_torso),
    critic_head=build_head(critic_head),
  )


# ── Public API ───────────────────────────────────────────────────────────────


def _build(registry: dict[str, Callable], kind: str, data: dict[str, Any]):
  data = dict(data)
  name = data.pop("type")
  if name not in registry:
    raise ValueError(f"Unknown {kind} '{name}'. Registered: {sorted(registry)}")
  return registry[name](**data)


def build_torso(data: dict[str, Any]):
  return _build(_torso_registry, "torso", data)


def build_head(data: dict[str, Any]):
  return _build(_head_registry, "head", data)


def build_network(data: dict[str, Any]) -> nn.Module:
  """Build a network from a config dict (e.g. from yaml.safe_load)."""
  return _build(_network_registry, "network", data)
