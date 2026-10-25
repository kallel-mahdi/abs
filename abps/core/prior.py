"""The GP's prior mean. MPD's is zero (plus the constant the GP fits). ABS's is the advantage prior: the
performance-difference lemma with critics of the center policy π_c, on its own states,

    m(θ) = J(θ_c) + Σ_k w_k · (1/E) Σ_{s ~ π_c} γ^t [Q̄_k(s, π_θ(s)) − Q̄_k(s, π_c(s))],

with Q̄_k the twin average of critic k's target networks. The weights w come from each critic's R² at predicting the
GP window's returns, every policy judged on its own states (the same lemma read the other way round).

Both share one contract: `value(θ)`, `gradient(θ)`, `learn(rollout, stage, window)` after each evaluation, with
`stage` one of "first" (the first center), "center" or "query", and `metrics(window)` once per iteration."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from abps.config import Config
from abps.core import critic
from abps.core.replay import Replay
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Evaluation, Rollout

AGGREGATIONS = {
    "mean": lambda r2: jnp.full_like(r2, 1 / len(r2)),
    "softmax": jax.nn.softmax,
    "max": lambda r2: jax.nn.one_hot(jnp.argmax(r2), len(r2)),
}


def advantages(target: critic.QNetwork, theta, states, weights, center: Evaluation, normalizer: Normalizer, policy):
    """Each critic's estimate of J(θ) − J(θ_c) from the weighted states [critic]."""
    s = normalizer(states)

    def one(twin):
        q = critic.q_values(twin, s, policy(theta, s)).mean(0)
        return jnp.sum(weights * (q - critic.q_values(twin, s, policy(center.theta, s)).mean(0)))

    return nnx.vmap(one)(target)


@partial(nnx.jit, graph=False, static_argnames="policy")
def mean(target: critic.QNetwork, weights, center: Evaluation, theta, normalizer: Normalizer, policy: Policy):
    """m(θ) for every row of θ [n, d]."""
    each = jax.vmap(advantages, in_axes=(None, 0, None, None, None, None, None))
    return center.returns + each(target, theta, center.states, center.weights, center, normalizer, policy) @ weights


@partial(nnx.jit, graph=False, static_argnames="policy")
def mean_gradient(target: critic.QNetwork, weights, center: Evaluation, theta, normalizer: Normalizer, policy: Policy):
    """∇m(θ) at a single θ."""
    return jax.grad(lambda t: mean(target, weights, center, t[None], normalizer, policy)[0])(theta)


@partial(nnx.jit, graph=False, static_argnames="policy")
def r2(target: critic.QNetwork, center: Evaluation, window: Evaluation, normalizer: Normalizer, policy: Policy):
    """Each critic's R² over the window's measured returns, every policy judged on its own states."""
    each = jax.vmap(advantages, in_axes=(None, 0, 0, 0, None, None, None))
    predicted = center.returns + each(target, window.theta, window.states, window.weights, center, normalizer, policy).T
    errors = jnp.maximum((predicted - window.returns) ** 2, 1e-4).sum(-1)
    return 1 - errors / (jnp.sum((window.returns - window.returns.mean()) ** 2) + 1e-6)


class ZeroPrior(nnx.Pytree):
    def value(self, theta: jax.Array, normalizer: Normalizer) -> jax.Array:
        return jnp.zeros(len(theta))

    def gradient(self, theta: jax.Array, normalizer: Normalizer) -> jax.Array:
        return jnp.zeros_like(theta)

    def learn(self, rollout: Rollout, stage: str, window: Evaluation, normalizer: Normalizer) -> dict:
        return {}

    def metrics(self, window: Evaluation, normalizer: Normalizer) -> dict:
        return {}


class AdvantagePrior(nnx.Pytree):
    """The critics, learned from every transition the search has seen, and the center they are critics of."""

    replay: Replay = nnx.data()
    center: Evaluation = nnx.data()

    def __init__(self, config: Config, policy: Policy, key: jax.Array):
        o, a = policy.observation_dim, policy.action_dim
        self.config, self.policy = config, policy
        self.ensemble = critic.Ensemble(o, a, config.critic, key)
        self.replay = Replay.create(config.critic.capacity, o, a)
        self.center = Evaluation(jnp.zeros(policy.dim), jnp.zeros(()), jnp.zeros(()), jnp.zeros((1, o)), jnp.zeros(1))
        self.r2 = jnp.zeros(config.critic.n_critics)

    @property
    def weights(self) -> jax.Array:
        return AGGREGATIONS[self.config.critic.aggregation](self.r2)

    def value(self, theta: jax.Array, normalizer: Normalizer) -> jax.Array:
        return mean(self.ensemble.target, self.weights, self.center, theta, normalizer, self.policy)

    def gradient(self, theta: jax.Array, normalizer: Normalizer) -> jax.Array:
        return mean_gradient(self.ensemble.target, self.weights, self.center, theta, normalizer, self.policy)

    def learn(self, rollout: Rollout, stage: str, window: Evaluation, normalizer: Normalizer) -> dict:
        """Store the transitions. A center becomes the prior's reference; a new one restarts the critics, which then
        train twice as long for its policy. The critics are scored when the weights or the reset read the scores."""
        config = self.config.critic
        self.replay = self.replay.add(*rollout.rows)
        metrics = {}
        if stage != "query":
            self.center = jax.tree.map(lambda x: x[0], rollout.evaluation(rollout.valid.shape[1]))
        if stage == "center":
            metrics["critic/resets"] = critic.reset(self.ensemble, self.r2, config).sum()
        steps = {"first": config.first_steps, "center": 2 * config.steps, "query": config.steps}[stage]
        losses = critic.train(
            self.ensemble,
            self.replay,
            self.center.theta,
            normalizer,
            self.policy,
            config,
            self.config.env.discount,
            steps,
        )
        if config.aggregation != "mean" or config.reset_worst:
            self.r2 = r2(self.ensemble.target, self.center, window, normalizer, self.policy)
        return metrics | {"critic/loss": losses.mean()}

    def metrics(self, window: Evaluation, normalizer: Normalizer) -> dict:
        return {"critic/r2": r2(self.ensemble.target, self.center, window, normalizer, self.policy).max()}
