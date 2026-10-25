"""ABS: local BO whose GP prior mean is the critic advantage prior (core/prior.py), learned from every transition the
search has seen. The center moves one step along Σ'⁻¹ μ', its mean absolute entry capped at the largest lengthscale
so it stays where the GP is informed."""

import jax
import jax.numpy as jnp

from abps.config import Config
from abps.core import gp
from abps.core.prior import AdvantagePrior
from abps.env.policy import Policy
from abps.methods.local_bo import LocalBO, ascent


class ABS(LocalBO):
    def __init__(self, config: Config, policy: Policy, key: jax.Array):
        super().__init__(config, policy, AdvantagePrior(config, policy, key))

    def move(self, posterior: gp.Posterior) -> tuple[jax.Array, dict]:
        mean, cov = gp.gradient(posterior, self.theta, self.prior.gradient(self.theta, self.normalizer))
        direction, probability = ascent(mean, cov)
        step = self.config.search.learning_rate * direction
        step *= jnp.minimum(1, self.config.gp.lengthscale_bounds[1] / jnp.mean(jnp.abs(step)))
        return self.theta + step, {"move/ascent_probability": probability}
