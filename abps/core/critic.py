"""Independent critics of the center policy π_θ, each a TD3-style twin of Q-networks with a Polyak target, trained by
TD(0) on the whole replay buffer:  Q(s, a) ← r + γ (1 − terminal) min_twin Q_target(s', π_θ(s'))."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from abps.config import Critic as Config
from abps.core.replay import Replay
from abps.env.policy import Normalizer, Policy

TWIN = 2


class QNetwork(nnx.Module):
    """[s, a] → (Dense → Dropout → LayerNorm → ReLU)* → Dense → tanh; returns are scaled to stay in [−1, 1]."""

    def __init__(self, obs_dim: int, action_dim: int, config: Config, key: jax.Array):
        rngs = nnx.Rngs(key)
        widths = (obs_dim + action_dim, *config.hidden)
        init = nnx.initializers.xavier_uniform()
        self.layers = nnx.List([nnx.Linear(a, b, kernel_init=init, rngs=rngs) for a, b in itertools.pairwise(widths)])
        self.norms = nnx.List([nnx.LayerNorm(width, rngs=rngs) for width in config.hidden])
        self.dropout = nnx.Dropout(config.dropout)
        self.head = nnx.Linear(widths[-1], 1, kernel_init=init, rngs=rngs)

    def __call__(self, s: jax.Array, a: jax.Array, key: jax.Array | None = None) -> jax.Array:
        """Q-values; dropout is active when a key is given."""
        x = jnp.concatenate([s, a], -1)
        keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)
        for layer, norm, k in zip(self.layers, self.norms, keys, strict=True):
            x = nnx.relu(norm(self.dropout(layer(x), deterministic=k is None, rngs=k)))
        return jnp.tanh(self.head(x))[..., 0]


class Critic(nnx.Pytree):
    """One member of the ensemble: a twin, its target, its optimizer and its random streams."""

    def __init__(self, obs_dim: int, action_dim: int, config: Config, key: jax.Array):
        init, streams, reset = jax.random.split(key, 3)
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.online = twin(obs_dim, action_dim, config, init)
        self.target = nnx.clone(self.online)
        self.optimizer = nnx.Optimizer(self.online, optax.adam(config.learning_rate), wrt=nnx.Param)
        self.rngs = nnx.Rngs(streams, reset=reset)


def twin(obs_dim: int, action_dim: int, config: Config, key: jax.Array) -> QNetwork:
    return nnx.vmap(lambda k: QNetwork(obs_dim, action_dim, config, k))(jax.random.split(key, TWIN))


def ensemble(obs_dim: int, action_dim: int, config: Config, key: jax.Array) -> Critic:
    return nnx.vmap(lambda k: Critic(obs_dim, action_dim, config, k))(jax.random.split(key, config.n_critics))


def q_values(network: QNetwork, s: jax.Array, a: jax.Array, key: jax.Array | None = None) -> jax.Array:
    """Both twins' Q-values [twin, ...]."""
    if key is None:
        return nnx.vmap(lambda net: net(s, a))(network)
    return nnx.vmap(lambda net, k: net(s, a, k))(network, jax.random.split(key, TWIN))


@partial(nnx.jit, graph=False, static_argnames=("policy", "config", "discount"))
def train(
    critics: Critic,
    replay: Replay,
    theta,
    normalizer: Normalizer,
    policy: Policy,
    config: Config,
    discount: float,
    steps: jax.Array,
):
    """`steps` TD updates of every critic on its own batches; returns each critic's mean loss. `steps` is a traced
    value, so every schedule (first fit, new center, query) shares one compilation."""

    def train_one(critic: Critic):
        def step(i, carry):
            critic, total = carry
            batch = replay.sample(critic.rngs(), config.batch_size)
            s, s_next = normalizer(batch.observations), normalizer(batch.next_observations)
            next_q = q_values(critic.target, s_next, policy(theta, s_next), critic.rngs()).min(0)
            target = batch.rewards + (1 - batch.terminated) * discount * next_q
            key = critic.rngs()

            def loss(online):
                return jnp.mean((q_values(online, s, batch.actions, key) - target) ** 2)

            value, grads = nnx.value_and_grad(loss)(critic.online)
            critic.optimizer.update(critic.online, grads)
            tau = jnp.where(i % config.target_every == 0, config.polyak, 0.0)
            online, target_params = nnx.state(critic.online, nnx.Param), nnx.state(critic.target, nnx.Param)
            nnx.update(critic.target, optax.incremental_update(online, target_params, tau))
            return critic, total + value

        _, total = nnx.fori_loop(0, steps, step, (critic, jnp.zeros(())))
        return total / steps

    return nnx.vmap(train_one)(critics)


@partial(nnx.jit, graph=False, static_argnames="config")
def reset(critics: Critic, scores: jax.Array, config: Config) -> jax.Array:
    """At a new center: restart every optimizer and, with `config.reset_worst`, re-initialize the worst critic if its
    validation R² is negative. Returns which critics were re-initialized."""
    worst = jax.nn.one_hot(jnp.argmin(scores), config.n_critics, dtype=bool) & (scores.min() < 0) & config.reset_worst

    def fresh(critic: Critic):
        return nnx.state(twin(critic.obs_dim, critic.action_dim, config, critic.rngs.reset()), nnx.Param)

    new = nnx.vmap(fresh)(critics)
    for network in (critics.online, critics.target):
        old = nnx.state(network, nnx.Param)
        nnx.update(
            network, jax.tree.map(lambda o, n: jnp.where(worst.reshape(-1, *[1] * (o.ndim - 1)), n, o), old, new)
        )
    restarted = nnx.vmap(lambda c: nnx.Optimizer(c.online, optax.adam(config.learning_rate), wrt=nnx.Param))(critics)
    nnx.update(critics.optimizer, nnx.state(restarted))
    return worst
