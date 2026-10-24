"""What the training loop needs from a method: the center θ, the policy's observation normalizer, and one
`iteration`, which plays its evaluations through `evaluate(theta, episodes)`, learns from them and moves θ."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from abps.config import Config
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Rollout

Evaluate = Callable[[jax.Array, int], Rollout]


@dataclass(frozen=True)
class Report:
    queries: list[Rollout]  # the policies evaluated around the center
    metrics: dict


class Searcher(nnx.Pytree):
    normalizer: Normalizer = nnx.data()

    def __init__(self, config: Config, policy: Policy, key: jax.Array):
        self.config, self.policy = config, policy
        self.theta = jnp.zeros(policy.dim)
        self.normalizer = Normalizer.create(policy.observation_dim)

    def iteration(self, evaluate: Evaluate, i: int, key: jax.Array) -> Report:
        raise NotImplementedError
