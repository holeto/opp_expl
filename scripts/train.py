"""Train a QMMD agent from a YAML config.

  uv run python scripts/train.py configs/qmmd_goofspiel.yaml
  uv run python scripts/train.py configs/qmmd_goofspiel.yaml --resume

Config layout (see configs/)::

  seed: 0
  env:        {name: goofspiel, ...}            # → oppex.envs.build_env
  algorithm:
    type: qmmd
    agent:    {type: policy_q | privileged_policy_q, network: {...}}
    optimizer: {type: adam, lr: 3e-4, max_grad_norm: 1.0}   # optional
    #   lr may be a schedule: {type: cosine_decay, init_value: 3e-4, decay_steps: null}
    schedules: {magnet_coef: {type: linear, ...}}            # optional
    ...                                                      # QMMD kwargs
  trainer:    {n_steps, log_every, checkpoint_every, checkpoint_dir, save_metrics}

``n_actions: null`` anywhere in the network config is filled with
``env.num_actions``; a ``null`` ``decay_steps`` / ``transition_steps`` in a
schedule is filled with the run length (optimizer updates for ``lr``, training
steps for ``schedules``). The config is copied next to the checkpoints so
evaluation scripts can rebuild the same env and agent (``build_from_config``).
"""

from __future__ import annotations

import argparse
import os

import jax
import yaml

from oppex.config import build_from_config
from oppex.trainers import StandardTrainer, latest_checkpoint, load_checkpoint


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("config")
  ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
  args = ap.parse_args()

  with open(args.config) as f:
    cfg = yaml.safe_load(f)

  env, _, algorithm = build_from_config(cfg)
  print(
    f"env: {env.__class__.__name__}  players={env.num_players}  actions={env.num_actions}"
    f"  max_len={env.max_length}"
  )
  print(
    f"algorithm: QMMD | batch={algorithm.batch_size} × T={algorithm.rollout_len}"
    f" | {algorithm.n_epochs} epochs | magnet={algorithm.magnet_update_type.value}"
    f" | loss={algorithm.loss_type.value}"
  )

  trainer_cfg = dict(cfg["trainer"])
  n_steps = trainer_cfg.pop("n_steps")
  checkpoint_dir = trainer_cfg.get("checkpoint_dir")
  trainer = StandardTrainer(algorithm, **trainer_cfg)

  if args.resume:
    if not checkpoint_dir:
      raise ValueError("Cannot resume: no checkpoint_dir in trainer config")
    path = latest_checkpoint(checkpoint_dir)
    state = load_checkpoint(path)
    n_steps = max(0, n_steps - int(state.step))
    print(f"resumed from {path} (step {int(state.step)}), {n_steps} steps remaining")
  else:
    if checkpoint_dir:
      os.makedirs(checkpoint_dir, exist_ok=True)
      with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    # Raw uint32 key rather than a typed jax.random.key: it pickles as a plain array.
    state = algorithm.init(jax.random.PRNGKey(cfg.get("seed", 0)))
    print(f"params: {sum(x.size for x in jax.tree.leaves(state.params)):,}")

  state = trainer.train(state, n_steps)
  print(f"done. final step = {int(state.step)}")


if __name__ == "__main__":
  main()
