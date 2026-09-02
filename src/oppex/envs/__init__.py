from .base import Env
from .hunl_holdem import HunlHoldem

_REGISTRY: dict[str, type] = {
  "hunl": HunlHoldem
}


def build_env(cfg: dict) -> Env:
  """Instantiate an environment from a config dict.

  The dict must contain a ``name`` key matching a registered environment.
  All remaining keys are passed as keyword arguments to the constructor.

  Example config::

      name: hunl
      big_blind: 4
      small_blind: 2
      starting_stack: 800
      reward_type: binary
  """
  cfg = dict(cfg)
  name = cfg.pop("name")
  if name not in _REGISTRY:
    raise ValueError(f"Unknown environment '{name}'. Available: {list(_REGISTRY)}")
  env = _REGISTRY[name](**cfg)
  return env
