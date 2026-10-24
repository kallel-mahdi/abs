"""Episodes of a batch of policies in Gymnasium MuJoCo, as fixed-shape arrays [policy, episode, time]."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from abps.config import Env
from abps.env.policy import Normalizer, Policy


@struct.dataclass
class Transitions:
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array  # scaled as in the training objective
    next_observations: jax.Array
    terminated: jax.Array


@struct.dataclass
class Rollout:
    """Every policy's episodes. `valid` marks the steps that happened, the last real one included."""

    theta: jax.Array  # [policy, dim]
    observations: jax.Array  # [policy, episode, time, obs]
    actions: jax.Array
    rewards: jax.Array  # raw environment rewards
    next_observations: jax.Array
    terminated: jax.Array
    valid: jax.Array
    discount: float = struct.field(pytree_node=False)
    reward_scale: float = struct.field(pytree_node=False)

    @property
    def weights(self) -> jax.Array:
        """γ^t on the steps that happened: the discounted state distribution."""
        return self.valid * self.discount ** jnp.arange(self.valid.shape[-1])

    @property
    def episode_returns(self) -> jax.Array:
        """The training objective, discounted and scaled, per episode [policy, episode]."""
        return (self.weights * self.rewards / self.reward_scale).sum(-1)

    @property
    def returns(self) -> jax.Array:
        return self.episode_returns.mean(-1)

    @property
    def noise(self) -> jax.Array:
        """Each policy's spread of returns over its episodes; zero for a single episode."""
        return self.episode_returns.std(-1)

    @property
    def raw_returns(self) -> jax.Array:
        """The environment's undiscounted, unscaled return, averaged over episodes."""
        return (self.valid * self.rewards).sum(-1).mean(-1)

    @property
    def episodes(self) -> int:
        return self.valid.shape[0] * self.valid.shape[1]

    def states(self, episodes: int) -> tuple[jax.Array, jax.Array]:
        """The first `episodes` episodes' states, concatenated in time [policy, episodes · time, obs], and their weights
        γ^t / episodes, so that a weighted sum over them averages over the episodes."""
        p, _, t, o = self.observations.shape
        weights = self.weights[:, :episodes].reshape(p, -1) / episodes
        return self.observations[:, :episodes].reshape(p, episodes * t, o), weights

    def per_episode(self) -> list[tuple[Transitions, jax.Array]]:
        """Every episode's transitions [time, ...] and which of its steps are real: fixed-shape pieces, so whatever
        consumes them compiles once."""
        p, e = self.valid.shape[:2]
        return [
            (
                Transitions(
                    self.observations[i, j],
                    self.actions[i, j],
                    self.rewards[i, j] / self.reward_scale,
                    self.next_observations[i, j],
                    self.terminated[i, j],
                ),
                self.valid[i, j],
            )
            for i in range(p)
            for j in range(e)
        ]


def policy(env: Env) -> Policy:
    """The linear policy class searched on the task."""
    with gym.make(env.name, **env.kwargs) as instance:
        observation_dim, action_dim = instance.observation_space.shape[0], instance.action_space.shape[0]
    return Policy(observation_dim, action_dim, env.features, env.action_scale)


class Envs:
    """A pool of environments reused across calls. A call plays every policy's episodes in lockstep on the CPU;
    finished episodes are no longer stepped."""

    def __init__(self, env: Env, policy: Policy):
        self.env, self.policy, self.pool = env, policy, []
        self.cpu = jax.devices("cpu")[0]
        self.act = jax.jit(jax.vmap(lambda theta, s, normalizer: policy(theta, normalizer(s)), (0, 0, None)))

    def __call__(self, theta: jax.Array, normalizer: Normalizer, *, episodes: int, seed: list[int]) -> Rollout:
        theta = np.asarray(theta)
        n, horizon = len(theta) * episodes, self.env.horizon
        self.pool += [gym.make(self.env.name, **self.env.kwargs) for _ in range(n - len(self.pool))]
        envs, seeds = self.pool[:n], np.random.SeedSequence(seed).generate_state(n)
        s = np.stack([env.reset(seed=int(k))[0] for env, k in zip(envs, seeds, strict=True)])
        parameters, normalizer = jax.device_put((np.repeat(theta, episodes, 0), normalizer), self.cpu)

        record = {
            "observations": np.zeros((n, horizon, s.shape[-1])),
            "actions": np.zeros((n, horizon, self.policy.action_dim)),
            "rewards": np.zeros((n, horizon)),
            "next_observations": np.zeros((n, horizon, s.shape[-1])),
            "terminated": np.zeros((n, horizon), bool),
            "valid": np.zeros((n, horizon), bool),
        }
        active = np.ones(n, bool)
        for t in range(horizon):
            a = np.asarray(self.act(parameters, s, normalizer))
            record["observations"][:, t], record["actions"][:, t], record["valid"][:, t] = s, a, active
            for i in np.flatnonzero(active):
                s[i], record["rewards"][i, t], record["terminated"][i, t], truncated, _ = envs[i].step(a[i])
                active[i] = not (record["terminated"][i, t] or truncated)
            record["next_observations"][:, t] = s
            if not active.any():
                break

        per_policy = {name: jnp.asarray(x.reshape(len(theta), episodes, *x.shape[1:])) for name, x in record.items()}
        return Rollout(jnp.asarray(theta), **per_policy, discount=self.env.discount, reward_scale=self.env.reward_scale)
