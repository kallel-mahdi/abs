"""MPD (Nguyen et al., 2022): local BO with a constant prior mean (a GP hyperparameter). The center takes normalized
steps along Σ'⁻¹ μ' while the probability of ascent stays above 0.65, at most 1000 per iteration."""

import jax
import jax.numpy as jnp

from abps.core import gp
from abps.methods.local_bo import LocalBO, ascent

THRESHOLD, MAX_STEPS = 0.65, 1000


@jax.jit
def walk(posterior: gp.Posterior, theta: jax.Array, learning_rate: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    def step(carry):
        x, _, steps = carry
        direction, probability = ascent(*gp.gradient(posterior, x, jnp.zeros_like(x)))
        return x + learning_rate * direction / jnp.linalg.norm(direction), probability, steps + 1

    return jax.lax.while_loop(lambda c: (c[1] > THRESHOLD) & (c[2] < MAX_STEPS), step, (theta, jnp.ones(()), 0))


class MPD(LocalBO):
    def move(self, posterior: gp.Posterior) -> tuple[jax.Array, dict]:
        theta, probability, steps = walk(posterior, self.theta, self.config.search.learning_rate)
        return theta, {"move/ascent_probability": probability, "move/steps": steps}
