"""What the training loop needs from a method: the center θ, the policy's observation normalizer, and one
`iteration`, which plays its evaluations through `evaluate(theta, episodes)`, learns from them, moves θ and returns
the rollouts of the policies it evaluated around the center with its metrics."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from abps.config import Config
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Rollout

Evaluate = Callable[[jax.Array, int], Rollout]


class Searcher(nnx.Pytree):
    normalizer: Normalizer = nnx.data()

    def __init__(self, config: Config, policy: Policy):
        self.config, self.policy = config, policy
        self.theta = jnp.zeros(policy.dim)
        self.normalizer = Normalizer.create(policy.observation_dim)

    def iteration(self, evaluate: Evaluate, key: jax.Array) -> tuple[list[Rollout], dict]:
        raise NotImplementedError
