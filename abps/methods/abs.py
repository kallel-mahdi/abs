"""ABS: local BO whose GP prior mean is the critic advantage prior (core/prior.py), learned from every transition the
search has seen. The center moves one step along Σ'⁻¹ μ', its mean absolute entry capped at the largest lengthscale
so it stays where the GP is informed."""

import jax
import jax.numpy as jnp
from flax import nnx

from abps.config import Config
from abps.core import critic, gp, prior
from abps.core.replay import Replay
from abps.env.policy import Policy
from abps.env.rollout import Rollout
from abps.methods.local_bo import LocalBO, ascent


class ABS(LocalBO):
    replay: Replay = nnx.data()
    center: prior.Center = nnx.data()

    def __init__(self, config: Config, policy: Policy, key: jax.Array):
        super().__init__(config, policy, key)
        o, a = policy.observation_dim, policy.action_dim
        self.critics = critic.ensemble(o, a, config.critic, key)
        self.replay = Replay.create(config.critic.capacity, o, a)
        self.center = prior.Center(jnp.zeros(policy.dim), jnp.zeros(()), jnp.zeros((1, o)), jnp.zeros(1))
        self.r2 = jnp.zeros(config.critic.n_critics)

    @property
    def weights(self) -> jax.Array:
        return prior.AGGREGATIONS[self.config.critic.aggregation](self.r2)

    def prior(self, theta: jax.Array) -> jax.Array:
        return prior.value(self.critics, self.weights, self.center, theta, self.normalizer, self.policy)

    def prior_gradient(self, theta: jax.Array) -> jax.Array:
        return prior.gradient(self.critics, self.weights, self.center, theta, self.normalizer, self.policy)

    def learn(self, rollout: Rollout, *, center: bool, first: bool, final: bool) -> dict:
        """Store the transitions. A new center becomes the prior's reference and resets the critics, which then train
        twice as long for its policy. Score R² when the search needs it, or once at the end for logging."""
        config = self.config.critic
        for transitions, valid in rollout.per_episode():
            self.replay = self.replay.add(transitions, valid)
        resets = jnp.zeros(config.n_critics, bool)
        if center:
            states, weights = rollout.states(rollout.valid.shape[1])
            self.center = prior.Center(rollout.theta[0], rollout.returns[0], states[0], weights[0])
            if not first:
                resets = critic.reset(self.critics, self.r2, config)
        steps = config.first_steps if first else config.steps * (2 if center else 1)
        losses = critic.train(
            self.critics,
            self.replay,
            self.center.theta,
            self.normalizer,
            self.policy,
            config,
            self.config.env.discount,
            steps,
        )
        metrics = {"critic/loss": losses.mean(), "critic/resets": resets.sum()}
        if final or config.aggregation != "mean" or config.reset_worst:
            w = self.window
            self.r2 = prior.r2(
                self.critics, self.center, w.theta, w.returns, w.states, w.weights, self.normalizer, self.policy
            )
            metrics["critic/r2"] = self.r2.max()
        return metrics

    def move(self, posterior: gp.Posterior) -> tuple[jax.Array, dict]:
        mean, cov = gp.gradient(posterior, self.theta, self.prior_gradient(self.theta))
        direction, probability = ascent(mean, cov)
        step = self.config.search.learning_rate * direction
        step *= jnp.minimum(1, self.config.gp.lengthscale_bounds[1] / jnp.mean(jnp.abs(step)))
        return self.theta + step, {"move/ascent_probability": probability}
