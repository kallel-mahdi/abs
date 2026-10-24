"""The replay buffer, TD training and the advantage prior."""

import jax
import jax.numpy as jnp
import numpy as np

from abps.config import Critic as Config
from abps.core import critic, prior
from abps.core.replay import Replay
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Transitions

CONFIG = Config(aggregation="mean", reset_worst=True, n_critics=3, hidden=(16,), batch_size=8, capacity=5)
POLICY = Policy(observation_dim=2, action_dim=1)


def transitions(n: int, start: float = 0.0) -> Transitions:
    values = start + jnp.arange(n, dtype=jnp.float32)
    return Transitions(
        jnp.stack([values, values], 1), values[:, None], values, jnp.stack([values, values], 1), jnp.zeros(n, bool)
    )


def test_replay_keeps_the_newest_valid_rows():
    replay = Replay.create(5, 2, 1).add(transitions(4), jnp.array([True, False, True, True]))
    replay = replay.add(transitions(4, start=10), jnp.ones(4, bool))
    assert int(replay.size) == 5
    np.testing.assert_array_equal(np.sort(replay.data.rewards), [3, 10, 11, 12, 13])


def test_td_training_fits_a_constant_reward(key):
    critics = critic.ensemble(2, 1, CONFIG, key)
    batch = Transitions(
        jax.random.normal(key, (64, 2)), jnp.zeros((64, 1)), jnp.full(64, 0.5), jnp.zeros((64, 2)), jnp.ones(64, bool)
    )
    replay = Replay.create(64, 2, 1).add(batch, jnp.ones(64, bool))
    args = (jnp.zeros(2), Normalizer.create(2), POLICY, CONFIG, 0.99)
    first = critic.train(critics, replay, *args, 5)
    later = critic.train(critics, replay, *args, 300)
    assert (later < first / 2).all()


def test_reset_reinitializes_only_the_worst_negative_critic(key):
    critics = critic.ensemble(2, 1, CONFIG, key)
    before = jax.tree.map(jnp.copy, jax.tree.leaves(critics.online.head.kernel))[0]
    worst = critic.reset(critics, jnp.array([0.5, -0.2, 0.1]), CONFIG)
    after = critics.online.head.kernel[...]
    np.testing.assert_array_equal(worst, [False, True, False])
    kept = jnp.array([0, 2])
    assert (after[kept] == before[kept]).all()
    assert (after[1] != before[1]).any()


def test_prior_gradient_matches_finite_differences(key):
    critics = critic.ensemble(2, 1, CONFIG, key)
    states = jax.random.normal(key, (10, 2))
    center = prior.Center(jnp.array([0.1, -0.2]), jnp.asarray(1.0), states, jnp.full(10, 0.1))
    weights, normalizer = jnp.full(3, 1 / 3), Normalizer.create(2)
    # Resolve the perturbation under JAX's native GPU matmul precision.
    theta, eps = jnp.array([0.3, 0.2]), 5e-3
    gradient = prior.gradient(critics, weights, center, theta, normalizer, POLICY)
    for i in range(2):
        step = jnp.zeros(2).at[i].set(eps)
        values = prior.value(critics, weights, center, jnp.stack([theta + step, theta - step]), normalizer, POLICY)
        np.testing.assert_allclose(gradient[i], (values[0] - values[1]) / (2 * eps), rtol=2e-2, atol=1e-5)


def test_prior_is_the_center_return_at_the_center(key):
    critics = critic.ensemble(2, 1, CONFIG, key)
    center = prior.Center(jnp.array([0.1, -0.2]), jnp.asarray(1.5), jax.random.normal(key, (10, 2)), jnp.ones(10))
    value = prior.value(critics, jnp.full(3, 1 / 3), center, center.theta[None], Normalizer.create(2), POLICY)
    np.testing.assert_allclose(value, [1.5], atol=1e-6)
