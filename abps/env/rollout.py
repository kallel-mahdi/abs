"""Episodes of a batch of policies in Gymnasium MuJoCo, as fixed-shape arrays [policy, episode, time], and the
evaluations the search keeps of them."""

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
    rewards: jax.Array  # raw environment rewards
    next_observations: jax.Array
    terminated: jax.Array


@struct.dataclass
class Evaluation:
    """Policies with their measured objective and the states they visited: what the GP conditions on (a window of
    them), what the advantage prior is centered on (one), and what the critics are validated on."""

    theta: jax.Array  # [n, d]
    returns: jax.Array  # [n]
    noise: jax.Array  # [n] spread of returns over episodes; zero for a single episode
    states: jax.Array  # [n, episodes · time, obs]
    weights: jax.Array  # [n, episodes · time]  γ^t / episodes, so a weighted sum averages over the episodes

    def push(self, other: "Evaluation", n_max: int) -> "Evaluation":
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y])[-n_max:], self, other)

    def __len__(self) -> int:
        return len(self.returns)


@struct.dataclass
class Rollout:
    """Every policy's episodes. `valid` marks the steps that happened, the last real one included."""

    theta: jax.Array  # [policy, dim]
    transitions: Transitions  # [policy, episode, time, ...]
    valid: jax.Array  # [policy, episode, time]
    discount: float = struct.field(pytree_node=False)
    reward_scale: float = struct.field(pytree_node=False)

    @property
    def weights(self) -> jax.Array:
        """γ^t on the steps that happened: the discounted state distribution."""
        return self.valid * self.discount ** jnp.arange(self.valid.shape[-1])

    @property
    def episode_returns(self) -> jax.Array:
        """The training objective, discounted and scaled, per episode [policy, episode]."""
        return (self.weights * self.transitions.rewards / self.reward_scale).sum(-1)

    @property
    def returns(self) -> jax.Array:
        return self.episode_returns.mean(-1)

    @property
    def raw_returns(self) -> jax.Array:
        """The environment's undiscounted, unscaled return, averaged over episodes."""
        return (self.valid * self.transitions.rewards).sum(-1).mean(-1)

    @property
    def episodes(self) -> int:
        return self.valid.shape[0] * self.valid.shape[1]

    @property
    def rows(self) -> tuple[Transitions, jax.Array]:
        """Every step as a row, rewards scaled as in the training objective, and which rows happened."""
        transitions = self.transitions.replace(rewards=self.transitions.rewards / self.reward_scale)
        return jax.tree.map(lambda x: x.reshape(-1, *x.shape[3:]), transitions), self.valid.reshape(-1)

    def evaluation(self, episodes: int) -> Evaluation:
        """Returns over all episodes; the states of the first `episodes`, concatenated in time."""
        p, _, t, o = self.transitions.observations.shape
        states = self.transitions.observations[:, :episodes].reshape(p, episodes * t, o)
        weights = self.weights[:, :episodes].reshape(p, -1) / episodes
        return Evaluation(self.theta, self.returns, self.episode_returns.std(-1), states, weights)


class Envs:
    """A pool of environments reused across calls, and the linear policy class searched on them. A call plays every
    policy's episodes in lockstep on the CPU; finished episodes are no longer stepped."""

    def __init__(self, env: Env):
        self.env, self.pool = env, [gym.make(env.name, **env.kwargs)]
        spaces = self.pool[0].observation_space, self.pool[0].action_space
        self.policy = Policy(spaces[0].shape[0], spaces[1].shape[0], env.features, env.action_scale)
        self.cpu = jax.devices("cpu")[0]
        self.act = jax.jit(jax.vmap(lambda theta, s, normalizer: self.policy(theta, normalizer(s)), (0, 0, None)))

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
        }
        valid, active = np.zeros((n, horizon), bool), np.ones(n, bool)
        for t in range(horizon):
            a = np.asarray(self.act(parameters, s, normalizer))
            record["observations"][:, t], record["actions"][:, t], valid[:, t] = s, a, active
            for i in np.flatnonzero(active):
                s[i], record["rewards"][i, t], record["terminated"][i, t], truncated, _ = envs[i].step(a[i])
                active[i] = not (record["terminated"][i, t] or truncated)
            record["next_observations"][:, t] = s
            if not active.any():
                break

        def per_policy(x):
            return jnp.asarray(x.reshape(len(theta), episodes, *x.shape[1:]))

        return Rollout(
            jnp.asarray(theta),
            Transitions(**{name: per_policy(x) for name, x in record.items()}),
            per_policy(valid),
            discount=self.env.discount,
            reward_scale=self.env.reward_scale,
        )
