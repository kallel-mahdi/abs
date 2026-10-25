"""The replay buffer, TD training and the advantage prior."""

import jax
import jax.numpy as jnp
import numpy as np

from abps.config import Critic as Config
from abps.core import critic, prior
from abps.core.replay import Replay
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Evaluation, Transitions

CONFIG = Config(aggregation="mean", reset_worst=True, n_critics=3, hidden=(16,), batch_size=8, capacity=5)
POLICY = Policy(observation_dim=2, action_dim=1)


def transitions(n: int, start: float = 0.0) -> Transitions:
    values = start + jnp.arange(n, dtype=jnp.float32)
    return Transitions(
        jnp.stack([values, values], 1), values[:, None], values, jnp.stack([values, values], 1), jnp.zeros(n, bool)
    )


def center(key, theta, value: float, weight: float) -> Evaluation:
    states = jax.random.normal(key, (10, 2))
    return Evaluation(jnp.asarray(theta), jnp.asarray(value), jnp.zeros(()), states, jnp.full(10, weight))


def test_replay_keeps_the_newest_valid_rows():
    replay = Replay.create(5, 2, 1).add(transitions(4), jnp.array([True, False, True, True]))
    replay = replay.add(transitions(4, start=10), jnp.ones(4, bool))
    assert int(replay.size) == 5
    np.testing.assert_array_equal(np.sort(replay.data.rewards), [3, 10, 11, 12, 13])


def test_td_training_fits_a_constant_reward(key):
    ensemble = critic.Ensemble(2, 1, CONFIG, key)
    batch = Transitions(
        jax.random.normal(key, (64, 2)), jnp.zeros((64, 1)), jnp.full(64, 0.5), jnp.zeros((64, 2)), jnp.ones(64, bool)
    )
    replay = Replay.create(64, 2, 1).add(batch, jnp.ones(64, bool))
    args = (jnp.zeros(2), Normalizer.create(2), POLICY, CONFIG, 0.99)
    first = critic.train(ensemble, replay, *args, 5)
    later = critic.train(ensemble, replay, *args, 300)
    assert later.shape == (CONFIG.n_critics,)
    assert (later < first / 2).all()


def test_reset_reinitializes_only_the_worst_negative_critic(key):
    ensemble = critic.Ensemble(2, 1, CONFIG, key)
    before = jnp.copy(ensemble.online.head.kernel[...])
    worst = critic.reset(ensemble, jnp.array([0.5, -0.2, 0.1]), CONFIG)
    after = ensemble.online.head.kernel[...]
    np.testing.assert_array_equal(worst, [False, True, False])
    kept = jnp.array([0, 2])
    assert (after[kept] == before[kept]).all()
    assert (after[1] != before[1]).any()


def test_prior_gradient_matches_finite_differences(key):
    target = critic.Ensemble(2, 1, CONFIG, key).target
    reference = center(key, [0.1, -0.2], 1.0, 0.1)
    weights, normalizer = jnp.full(3, 1 / 3), Normalizer.create(2)
    # Small enough to stay between the ReLU kinks; tests run in float32 on the CPU.
    theta, eps = jnp.array([0.3, 0.2]), 1e-3
    gradient = prior.mean_gradient(target, weights, reference, theta, normalizer, POLICY)
    for i in range(2):
        step = jnp.zeros(2).at[i].set(eps)
        values = prior.mean(target, weights, reference, jnp.stack([theta + step, theta - step]), normalizer, POLICY)
        np.testing.assert_allclose(gradient[i], (values[0] - values[1]) / (2 * eps), rtol=2e-2, atol=1e-5)


def test_prior_is_the_center_return_at_the_center(key):
    target = critic.Ensemble(2, 1, CONFIG, key).target
    reference = center(key, [0.1, -0.2], 1.5, 1.0)
    value = prior.mean(target, jnp.full(3, 1 / 3), reference, reference.theta[None], Normalizer.create(2), POLICY)
    np.testing.assert_allclose(value, [1.5], atol=1e-6)
