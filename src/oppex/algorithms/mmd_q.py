"""QMMD — Magnetic Mirror Descent with an explicit Q-head and no V-head.

Ported from RLG ``algorithms/mmd_q.py`` (QMMD), with the rollout replaced by the
auto-resetting collector in ``rollout.py``.

Network: policy logits + Q(s, :) (``PolicyQAgent``, or ``PrivilegedPolicyQAgent``
whose Q-head sees ground truth). The state value is derived as
V(s) = E_μ[Q(s, :)].

Q-targets — Retrace(λ) on on-policy data. The bootstrap and the trace use the
rollout policy μ over the Polyak target network's Q-values,
V(s) = E_μ[Q_target(s, :)]. With π = μ the Retrace IS factor min(1, π/μ) is
identically 1, so the trace coefficient is the constant ``gae_lambda``: Expected
SARSA(λ). The target network contributes only its Q-values (for stability),
never its policy.

Policy update — see ``losses/mmd_q.py``: PPO-clipped surrogate on a centred
all-actions advantage (``loss_type=mmd``) or NeuRD on reconstructed regrets
(``loss_type=rnad``), plus
  magnet_coef     · KL(π ‖ magnet)        — the MMD proximal-to-magnet term
  old_policy_coef · KL(π ‖ μ)             — mirror-descent step-size control
  ent_coef        · (−H(π))

Three parameter copies are kept in ``extras``:
  target_params  Polyak average (rate ``target_update_rate``) — Q-targets.
  magnet_params  the MMD magnet, updated either
                 'incremental' — Polyak average at ``magnet_update_rate``, the
                                 *moving magnet*: it trails the current policy
                                 continuously instead of jumping, and is the
                                 default (notes/mmd_hunl_tuning.md: target and
                                 magnet rates of ~5e-5 for HUNL); or
                 'periodic'    — hard copy of params every ``magnet_interval``
                                 steps (the classic reset schedule).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from .advantage import retrace
from .base import Algorithm, TrainingState, applied_learning_rate, inject_learning_rate
from .rollout import Bootstrap, Rollout, RolloutCarry, collect_rollouts
from .types import KLDirection, LossType, MagnetUpdateType, MMD_SCHEDULABLE
from ..agents.base import Agent
from ..envs.base import Env
from ..losses.mmd_q import mmd_q_loss
from ..policy import policy_probs


class QMMD(Algorithm):
  """MMD with a Q-value function trained by Retrace(λ).

  Args:
    env, agent:          environment and PolicyQ-style agent.
    batch_size:          B, rows collected in parallel.
    rollout_len:         T, steps per row per iteration. Defaults to
                         ``env.max_length`` (for Goofspiel: exactly one game).
    n_epochs:            full-batch gradient steps per iteration.
    lr / optimizer:      Adam(lr) unless an optax optimizer is given. A learning-rate
                         schedule belongs in the optimizer (``config.build_optimizer``);
                         if it is built with ``inject_learning_rate``, the rate
                         applied is logged as the ``lr`` metric.
    clip_eps:            PPO ratio clip; also the Q-loss clip.
    vf_coef:             weight of the Q-loss.
    ent_coef:            weight of the entropy bonus.
    gamma, gae_lambda:   discount and Retrace trace decay λ.
    magnet_coef:         weight of KL(π ‖ magnet).
    old_policy_coef:     weight of KL(π ‖ μ).
    target_update_rate:  Polyak τ for target_params.
    magnet_update_type:  'incremental' (Polyak, default) or 'periodic'.
    magnet_update_rate:  Polyak τ for the incremental magnet.
    magnet_interval:     steps between hard resets for the periodic magnet.
    loss_type:           'mmd' (PPO surrogate) or 'rnad' (NeuRD).
    neurd_clip, neurd_threshold: NeuRD advantage clip and logit threshold.
    kl_direction:        'reverse' KL(π‖·) (default) or 'forward' KL(·‖π).
    alternating:         update one player per iteration (loss weight P on the
                         active player, 0 on the other).
    normalize_rewards:   divide rewards by ``env.max_reward``.
    schedules:           {name: step → value} for any key in MMD_SCHEDULABLE.
  """

  def __init__(
    self,
    env: Env,
    agent: Agent,
    batch_size: int = 64,
    rollout_len: int | None = None,
    n_epochs: int = 1,
    lr: float = 3e-4,
    optimizer: optax.GradientTransformation | None = None,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    magnet_coef: float = 0.15,
    old_policy_coef: float = 0.05,
    target_update_rate: float = 0.001,
    magnet_update_type: MagnetUpdateType | str = MagnetUpdateType.INCREMENTAL,
    magnet_update_rate: float = 0.001,
    magnet_interval: int = 2000,
    loss_type: LossType | str = LossType.MMD,
    neurd_clip: float = 5.0,
    neurd_threshold: float = 2.0,
    kl_direction: KLDirection | str = KLDirection.REVERSE,
    alternating: bool = False,
    normalize_rewards: bool = True,
    schedules: dict[str, Callable[[jax.Array], jax.Array]] | None = None,
  ) -> None:
    self.env = env
    self.agent = agent
    self.batch_size = batch_size
    self.rollout_len = int(rollout_len) if rollout_len is not None else env.max_length
    self.n_epochs = n_epochs
    self.optimizer = (
      optimizer if optimizer is not None else inject_learning_rate(optax.adam, lr)
    )
    self.clip_eps = clip_eps
    self.vf_coef = vf_coef
    self.ent_coef = ent_coef
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.magnet_coef = magnet_coef
    self.old_policy_coef = old_policy_coef
    self.target_update_rate = target_update_rate
    self.magnet_update_type = MagnetUpdateType(magnet_update_type)
    self.magnet_update_rate = magnet_update_rate
    self.magnet_interval = magnet_interval
    self.loss_type = LossType(loss_type)
    self.neurd_clip = neurd_clip
    self.neurd_threshold = neurd_threshold
    self.kl_direction = KLDirection(kl_direction)
    self.alternating = alternating
    self.normalize_rewards = normalize_rewards
    schedules = schedules or {}
    unknown = set(schedules) - MMD_SCHEDULABLE
    if unknown:
      raise ValueError(f"Unknown schedule keys: {unknown}. Valid keys: {sorted(MMD_SCHEDULABLE)}")
    self.schedules = schedules

  # ── Init ────────────────────────────────────────────────────────────────────

  def init(self, key: jax.Array) -> TrainingState:
    key, env_key, net_key, obs_key = jax.random.split(key, 4)
    B = self.batch_size
    env_state = jax.vmap(self.env.init_state)(jax.random.split(env_key, B))
    first = jax.tree.map(lambda x: x[0], env_state)
    params = self.agent.init_params(net_key, self.agent.dummy_obs(self.env, first, obs_key))
    agent_state = jax.tree.map(
      lambda x: jnp.broadcast_to(x, (B,) + x.shape), self.agent.init_state(params)
    )
    return TrainingState(
      params=params,
      opt_state=self.optimizer.init(params),
      env_state=env_state,
      agent_state=agent_state,
      ep_len=jnp.zeros(B, jnp.int32),
      rng=key,
      step=jnp.zeros((), jnp.int32),
      extras={"target_params": params, "magnet_params": params},
    )

  # ── Helpers ─────────────────────────────────────────────────────────────────

  def _eval_params(self, params: Any, rollout: Rollout) -> Any:
    """Re-evaluate ``params`` on every stored step → outputs shaped (B, T, P, ...)."""
    obs = self.agent.eval_obs(rollout)
    eval_T = jax.vmap(self.agent.player_evaluate, in_axes=(None, 0, 0))
    eval_BT = jax.vmap(eval_T, in_axes=(None, 0, 0))
    out, _ = eval_BT(params, rollout.agent_states, obs)
    return out

  def _eval_bootstrap(self, params: Any, bootstrap: Bootstrap) -> Any:
    """Evaluate ``params`` at the post-window states → outputs shaped (B, P, ...)."""
    eval_B = jax.vmap(self.agent.player_evaluate, in_axes=(None, 0, 0))
    out, _ = eval_B(params, bootstrap.agent_state, bootstrap.obs)
    return out

  def _q_targets(
    self,
    rollout: Rollout,
    target_q: jax.Array,       # (B, T, P, A) target-net Q
    boot_v: jax.Array,         # (B, P) E_μ[Q_target(s_T)]
  ) -> jax.Array:              # (B, T, P)
    mu = policy_probs(rollout.agent_output.logits, rollout.legal_actions)
    v_target = (mu * target_q).sum(-1)
    q_taken = jnp.take_along_axis(target_q, rollout.actions[..., None], axis=-1).squeeze(-1)
    discount = (1.0 - rollout.dones.astype(jnp.float32)) * self.gamma  # (B, T)

    _retrace = lambda r, q, v, d, b: retrace(r, q, v, d, lambda_=self.gae_lambda, bootstrap_v=b)
    _retrace_P = jax.vmap(_retrace, in_axes=(1, 1, 1, None, 0), out_axes=1)
    _retrace_BP = jax.vmap(_retrace_P, in_axes=(0, 0, 0, 0, 0))
    return _retrace_BP(rollout.rewards, q_taken, v_target, discount, boot_v)

  # ── Step ────────────────────────────────────────────────────────────────────

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    def _get(name: str, default):
      s = self.schedules.get(name)
      return s(state.step) if s is not None else default

    clip_eps = _get("clip_eps", self.clip_eps)
    vf_coef = _get("vf_coef", self.vf_coef)
    ent_coef = _get("ent_coef", self.ent_coef)
    magnet_coef = _get("magnet_coef", self.magnet_coef)
    old_policy_coef = _get("old_policy_coef", self.old_policy_coef)
    target_update_rate = _get("target_update_rate", self.target_update_rate)
    magnet_update_rate = _get("magnet_update_rate", self.magnet_update_rate)
    neurd_clip = _get("neurd_clip", self.neurd_clip)
    neurd_threshold = _get("neurd_threshold", self.neurd_threshold)

    rng, collect_key = jax.random.split(state.rng)
    carry, rollout, bootstrap = collect_rollouts(
      self.env, self.agent, state.params,
      RolloutCarry(state.env_state, state.agent_state, state.ep_len),
      collect_key, self.rollout_len, self.normalize_rewards,
    )

    # ── Retrace targets: target-net Q, rollout policy μ ──────────────────────
    target_params = state.extras["target_params"]
    target_q = lax.stop_gradient(self._eval_params(target_params, rollout).q_values)
    boot_mu = policy_probs(self._eval_bootstrap(state.params, bootstrap).logits, bootstrap.legal_actions)
    boot_q = self._eval_bootstrap(target_params, bootstrap).q_values
    boot_v = lax.stop_gradient((boot_mu * boot_q).sum(-1))  # (B, P)
    q_targets = lax.stop_gradient(self._q_targets(rollout, target_q, boot_v))

    magnet_logits = lax.stop_gradient(
      self._eval_params(state.extras["magnet_params"], rollout).logits
    )

    P = self.env.num_players
    if self.alternating:
      player_weight = jnp.zeros(P).at[state.step % P].set(float(P))
    else:
      player_weight = jnp.ones(P)

    # Sum over players, mean over all B·T steps (every step is valid: no padding).
    reduce = lambda x: (x * player_weight).sum(-1).mean()

    loss_fn = functools.partial(
      mmd_q_loss, kl_direction=self.kl_direction, loss_type=self.loss_type
    )
    _axes = (0,) * 9 + (None,) * 7  # 9 per-sample arrays, 7 scalar coefficients
    loss_BTP = jax.vmap(jax.vmap(jax.vmap(loss_fn, in_axes=_axes), in_axes=_axes), in_axes=_axes)

    def epoch_fn(carry, _):
      params, opt_state = carry

      def total_loss(params):
        out = self._eval_params(params, rollout)
        losses, metrics = loss_BTP(
          out.q_values,
          out.logits,
          rollout.legal_actions,
          rollout.actions,
          rollout.agent_output.logits,
          rollout.agent_output.q_values,
          magnet_logits,
          q_targets,
          target_q,
          clip_eps,
          vf_coef,
          ent_coef,
          magnet_coef,
          old_policy_coef,
          neurd_clip,
          neurd_threshold,
        )
        return reduce(losses), jax.tree.map(reduce, metrics)

      (_, metrics), grads = jax.value_and_grad(total_loss, has_aux=True)(params)
      updates, opt_state = self.optimizer.update(grads, opt_state, params)
      return (optax.apply_updates(params, updates), opt_state), metrics

    (params, opt_state), epoch_metrics = jax.lax.scan(
      epoch_fn, (state.params, state.opt_state), None, length=self.n_epochs
    )

    # ── Slow copies ─────────────────────────────────────────────────────────
    target_params = optax.incremental_update(params, target_params, target_update_rate)
    if self.magnet_update_type == MagnetUpdateType.INCREMENTAL:
      magnet_params = optax.incremental_update(
        params, state.extras["magnet_params"], magnet_update_rate
      )
    else:
      magnet_params = optax.periodic_update(
        params, state.extras["magnet_params"], state.step, self.magnet_interval
      )

    metrics = jax.tree.map(jnp.mean, epoch_metrics)
    metrics.update(self._rollout_metrics(rollout, q_targets))
    lr = applied_learning_rate(opt_state)
    if lr is not None:
      metrics["lr"] = lr

    return TrainingState(
      params=params,
      opt_state=opt_state,
      env_state=carry.env_state,
      agent_state=carry.agent_state,
      ep_len=carry.ep_len,
      rng=rng,
      step=state.step + 1,
      extras={"target_params": target_params, "magnet_params": magnet_params},
    ), metrics

  @staticmethod
  def _rollout_metrics(rollout: Rollout, q_targets: jax.Array) -> dict[str, jax.Array]:
    dones = rollout.dones.astype(jnp.float32)          # (B, T)
    n_done = jnp.maximum(dones.sum(), 1.0)
    return {
      # Games finished per row per window (≈ T / mean game length).
      "episodes_per_row": dones.sum(-1).mean(),
      # Player 0's mean return per finished game (seat advantage under self-play).
      "return_p0": (rollout.rewards[..., 0] * dones).sum() / n_done,
      "truncated_frac": rollout.truncated.astype(jnp.float32).sum() / n_done,
      "q_target_mean": q_targets.mean(),
    }
