"""A device-resident ring buffer of every real transition the search has seen."""

import jax
import jax.numpy as jnp
from flax import struct

from abps.env.rollout import Transitions


@struct.dataclass
class Replay:
    data: Transitions
    position: jax.Array
    size: jax.Array

    @classmethod
    def create(cls, capacity: int, obs_dim: int, action_dim: int) -> "Replay":
        data = Transitions(
            jnp.zeros((capacity, obs_dim)),
            jnp.zeros((capacity, action_dim)),
            jnp.zeros(capacity),
            jnp.zeros((capacity, obs_dim)),
            jnp.zeros(capacity, bool),
        )
        return cls(data, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))

    @jax.jit
    def add(self, batch: Transitions, valid: jax.Array) -> "Replay":
        """Write the valid rows of `batch` (at most the capacity) after the newest, overwriting the oldest."""
        capacity = len(self.data.rewards)
        rows = jnp.where(valid, (self.position + jnp.cumsum(valid) - 1) % capacity, capacity)
        data = jax.tree.map(lambda old, new: old.at[rows].set(new, mode="drop"), self.data, batch)
        count = valid.sum(dtype=jnp.int32)
        return Replay(data, (self.position + count) % capacity, jnp.minimum(self.size + count, capacity))

    def sample(self, key: jax.Array, batch_size: int) -> Transitions:
        rows = jax.random.randint(key, (batch_size,), 0, self.size)
        return jax.tree.map(lambda x: x[rows], self.data)
