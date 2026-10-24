"""Policies, the observation normalizer, and rollouts in real Gymnasium tasks."""

import jax.numpy as jnp
import numpy as np

from abps.config import Env
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Envs


def test_normalizer_merges_batches_into_the_valid_rows_statistics():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(30, 3)), rng.normal(2.0, 3.0, size=(20, 3))
    valid_a, valid_b = rng.random(30) > 0.3, np.ones(20, bool)
    normalizer = (
        Normalizer.create(3).update(jnp.asarray(a), jnp.asarray(valid_a)).update(jnp.asarray(b), jnp.asarray(valid_b))
    )
    rows = np.concatenate([a[valid_a], b])
    np.testing.assert_allclose(normalizer(jnp.asarray(rows)), (rows - rows.mean(0)) / rows.std(0), rtol=1e-4)


def test_empty_normalizer_is_the_identity():
    x = jnp.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(Normalizer.create(3)(x), x)


def test_linear_policy_layout():
    policy = Policy(observation_dim=3, action_dim=2, action_scale=3.0)
    theta, s = jnp.arange(6.0) / 10, jnp.array([1.0, -1.0, 0.5])
    np.testing.assert_allclose(policy(theta, s), 3 * np.tanh(s @ np.arange(6.0).reshape(3, 2) / 10), rtol=1e-4)


def test_rollouts_count_the_terminal_step_and_stop_at_truncation():
    """Pendulum-v1 truncates at 200 steps: every step until then counts, nothing after."""
    env = Env(
        name="Pendulum-v1", horizon=250, discount=0.9, reward_scale=10.0, normalize_obs=False, features="pendulum"
    )
    policy = Policy(3, 1, "pendulum", 2.0)
    rollout = Envs(env, policy)(jnp.zeros((2, 2)), Normalizer.create(3), episodes=3, seed=[0])
    assert rollout.valid.shape == (2, 3, 250)
    assert rollout.valid.sum(-1).tolist() == [[200] * 3] * 2
    expected = (rollout.rewards[..., :200] * 0.9 ** np.arange(200) / 10).sum(-1).mean(-1)
    np.testing.assert_allclose(rollout.returns, expected, rtol=1e-4)
    _, weights = rollout.states(3)
    np.testing.assert_allclose(weights.sum(-1), (0.9 ** np.arange(200)).sum(), rtol=1e-4)  # γ^t averaged over episodes
    np.testing.assert_allclose(rollout.raw_returns, rollout.rewards.sum(-1).mean(-1), rtol=1e-4)


def test_rollouts_are_seeded():
    env = Env(name="InvertedPendulum-v5", horizon=30, discount=0.99, reward_scale=1.0, normalize_obs=False)
    envs, theta = Envs(env, Policy(4, 1)), jnp.ones((2, 4)) * 0.1
    first = envs(theta, Normalizer.create(4), episodes=2, seed=[1, 2])
    again = envs(theta, Normalizer.create(4), episodes=2, seed=[1, 2])
    np.testing.assert_array_equal(first.observations, again.observations)
