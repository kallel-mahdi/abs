"""Independent critics of the center policy π_θ, each a TD3-style twin of Q-networks with a Polyak target, trained by
TD(0) on the whole replay buffer:  Q(s, a) ← r + γ (1 − terminal) min_twin Q_target(s', π_θ(s')).

The ensemble is one stack of Q-networks [critic, twin] with one Adam: Adam is elementwise and every critic steps and
restarts together, so this is the same as one optimizer per critic."""

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


def stack(obs_dim: int, action_dim: int, config: Config, key: jax.Array) -> QNetwork:
    """Q-networks with leading axes [critic, twin]."""
    keys = jax.random.split(key, (config.n_critics, TWIN))
    return nnx.vmap(nnx.vmap(lambda k: QNetwork(obs_dim, action_dim, config, k)))(keys)


def q_values(twin: QNetwork, s: jax.Array, a: jax.Array, key: jax.Array | None = None) -> jax.Array:
    """One critic's Q-values [twin, ...]."""
    if key is None:
        return nnx.vmap(lambda net: net(s, a))(twin)
    return nnx.vmap(lambda net, k: net(s, a, k))(twin, jax.random.split(key, TWIN))


class Ensemble(nnx.Pytree):
    def __init__(self, obs_dim: int, action_dim: int, config: Config, key: jax.Array):
        init, streams, reset = jax.random.split(key, 3)
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.online = stack(obs_dim, action_dim, config, init)
        self.target = nnx.clone(self.online)
        self.optimizer = nnx.Optimizer(self.online, optax.adam(config.learning_rate), wrt=nnx.Param)
        self.rngs = nnx.Rngs(streams, reset=reset)


@partial(nnx.jit, graph=False, static_argnames=("policy", "config", "discount"))
def train(
    ensemble: Ensemble,
    replay: Replay,
    theta: jax.Array,
    normalizer: Normalizer,
    policy: Policy,
    config: Config,
    discount: float,
    steps: jax.Array,
) -> jax.Array:
    """`steps` TD updates, every critic on its own batches; returns each critic's mean loss. `steps` is traced, so
    every budget (first fit, new center, query) shares one compilation."""

    def step(i, carry):
        ensemble, total = carry
        sample, next_dropout, dropout = jax.random.split(ensemble.rngs(), (3, config.n_critics))
        batch = jax.vmap(lambda k: replay.sample(k, config.batch_size))(sample)  # [critic, batch, ...]
        s, s_next = normalizer(batch.observations), normalizer(batch.next_observations)
        next_q = nnx.vmap(lambda twin, x, k: q_values(twin, x, policy(theta, x), k).min(0))(
            ensemble.target, s_next, next_dropout
        )
        y = batch.rewards + (1 - batch.terminated) * discount * next_q

        def summed(online):
            q = nnx.vmap(q_values)(online, s, batch.actions, dropout)  # [critic, twin, batch]
            losses = jnp.mean((q - y[:, None]) ** 2, axis=(1, 2))
            return losses.sum(), losses

        (_, losses), grads = nnx.value_and_grad(summed, has_aux=True)(ensemble.online)
        ensemble.optimizer.update(ensemble.online, grads)
        tau = jnp.where(i % config.target_every == 0, config.polyak, 0.0)
        online, target = nnx.state(ensemble.online, nnx.Param), nnx.state(ensemble.target, nnx.Param)
        nnx.update(ensemble.target, optax.incremental_update(online, target, tau))
        return ensemble, total + losses

    _, total = nnx.fori_loop(0, steps, step, (ensemble, jnp.zeros(config.n_critics)))
    return total / steps


@partial(nnx.jit, graph=False, static_argnames="config")
def reset(ensemble: Ensemble, r2: jax.Array, config: Config) -> jax.Array:
    """At a new center: restart the optimizer and, with `config.reset_worst`, re-initialize the worst critic if its
    validation R² is negative. Returns which critics were re-initialized."""
    worst = jax.nn.one_hot(jnp.argmin(r2), config.n_critics, dtype=bool) & (r2.min() < 0) & config.reset_worst
    if config.reset_worst:
        fresh = nnx.state(stack(ensemble.obs_dim, ensemble.action_dim, config, ensemble.rngs.reset()), nnx.Param)
        for network in (ensemble.online, ensemble.target):
            old = nnx.state(network, nnx.Param)
            nnx.update(
                network, jax.tree.map(lambda o, n: jnp.where(worst[:, None, *[None] * (o.ndim - 2)], n, o), old, fresh)
            )
    restarted = nnx.Optimizer(ensemble.online, optax.adam(config.learning_rate), wrt=nnx.Param)
    nnx.update(ensemble.optimizer, nnx.state(restarted))
    return worst
