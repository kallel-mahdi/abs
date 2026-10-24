"""The searched policies: linear maps of normalized observations, parameterized by a flat vector θ."""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct


@dataclass(frozen=True)
class Policy:
    """a = action_scale · tanh(φ(norm(s)) W) with W = θ.reshape(features, actions) and no bias."""

    observation_dim: int
    action_dim: int
    features: Literal["identity", "pendulum"] = "identity"
    action_scale: float = 1.0

    @property
    def feature_dim(self) -> int:
        return 2 if self.features == "pendulum" else self.observation_dim

    @property
    def dim(self) -> int:
        return self.feature_dim * self.action_dim

    def __call__(self, theta: jax.Array, observations: jax.Array) -> jax.Array:
        if self.features == "pendulum":
            # Pendulum-v1 observes (cos, sin, angular velocity); the released map takes atan2(cos, sin), giving a
            # two-parameter policy whose search landscape the acquisition figure draws.
            observations = jnp.stack(
                [jnp.arctan2(observations[..., 0], observations[..., 1]), observations[..., 2]], -1
            )
        return self.action_scale * jnp.tanh(observations @ theta.reshape(self.feature_dim, self.action_dim))


@struct.dataclass
class Normalizer:
    """Running mean and standard deviation of observations (Welford/Chan). With no observations it is the identity,
    which is how tasks without normalization use it."""

    count: jax.Array
    mean: jax.Array
    m2: jax.Array

    @classmethod
    def create(cls, dim: int) -> "Normalizer":
        return cls(jnp.zeros(()), jnp.zeros(dim), jnp.zeros(dim))

    def __call__(self, observations: jax.Array) -> jax.Array:
        std = jnp.clip(jnp.sqrt(self.m2 / jnp.maximum(self.count, 1)), 1e-6, 1e6)
        return (observations - self.mean) / jnp.where(self.count > 0, std, 1)

    @jax.jit
    def update(self, observations: jax.Array, valid: jax.Array) -> "Normalizer":
        """Merge the valid rows of a batch of observations [..., dim] into the running statistics."""
        observations, valid = observations.reshape(-1, self.mean.size), valid.reshape(-1, 1)
        count = valid.sum()
        mean = (valid * observations).sum(0) / jnp.maximum(count, 1)
        m2 = (valid * (observations - mean) ** 2).sum(0)
        total, delta = self.count + count, mean - self.mean
        return Normalizer(
            total,
            self.mean + delta * count / jnp.maximum(total, 1),
            self.m2 + m2 + delta**2 * self.count * count / jnp.maximum(total, 1),
        )
