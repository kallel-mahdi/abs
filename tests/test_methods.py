"""The update rules: the ABS step and its cap, MPD's walk, ARS's step, and the critics' validation R²."""

from collections import Counter
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from abps.core import critic, gp, prior
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Envs, policy
from abps.methods import ABS, ars, mpd


def window(key, d: int, n: int = 8):
    k1, k2 = jax.random.split(key)
    theta = 0.02 * jax.random.normal(k1, (n, d))
    return theta, theta @ jnp.linspace(1.0, 2.0, d) + 0.01 * jax.random.normal(k2, (n,))


def test_abs_steps_along_the_preconditioned_gradient_and_caps_it(key, tiny):
    config = tiny("abs")
    config = replace(config, search=replace(config.search, learning_rate=10.0))  # large, so the cap binds
    searcher = ABS(config, policy(config.env), key)
    theta, returns = window(key, searcher.policy.dim)
    searcher.window = searcher.window.push(
        type(searcher.window)(theta, returns, jnp.zeros(8), jnp.zeros((8, 20, 4)), jnp.zeros((8, 20))), 8
    )
    mean, cov = gp.gradient(searcher.posterior(), searcher.theta, searcher.prior_gradient(searcher.theta))
    moved, _ = searcher.move(searcher.posterior())
    step = moved - searcher.theta
    np.testing.assert_allclose(jnp.mean(jnp.abs(step)), config.gp.lengthscale_bounds[1], rtol=1e-6)
    direction = jnp.linalg.solve(cov, mean)
    np.testing.assert_allclose(step / jnp.linalg.norm(step), direction / jnp.linalg.norm(direction), rtol=1e-5)


def test_mpd_walks_unit_steps_until_ascent_is_uncertain(key):
    theta, returns = window(key, 3)
    params = gp.Params(jnp.full(3, 0.05), jnp.asarray(0.5), jnp.asarray(0.01), jnp.zeros(()))
    posterior = gp.condition(params, theta, returns, jnp.zeros(8))
    x, probability, steps = mpd.walk(posterior, jnp.zeros(3), 0.01)
    assert probability <= mpd.THRESHOLD or steps == mpd.MAX_STEPS
    assert jnp.linalg.norm(x) <= 0.01 * steps + 1e-9


def test_ars_ascent_is_the_released_formula(key):
    directions = jax.random.normal(key, (6, 4))
    returns = jax.random.normal(jax.random.fold_in(key, 1), (12,))
    b = 2
    plus, minus = jnp.split(returns, 2)
    weight = jnp.where(jnp.argsort(jnp.argsort(-jnp.maximum(plus, minus))) < b, 1, 0)  # released ars.py
    released = jnp.sum((directions.T * weight * (plus - minus)).T, 0) / (
        b * jnp.std(returns, where=jnp.concatenate([weight, weight]) > 0)
    )
    np.testing.assert_allclose(ars.ascent(directions, returns, b)[0], released, rtol=4 * np.finfo(directions.dtype).eps)


def test_validation_r2_is_the_released_formula(key, tiny):
    config = tiny("abs").critic
    pol = Policy(2, 1)
    critics = critic.ensemble(2, 1, config, key)
    center = prior.Center(jnp.zeros(2), jnp.asarray(0.3), jax.random.normal(key, (10, 2)), jnp.full(10, 0.1))
    theta, returns = window(key, 2)
    states, weights = jax.random.normal(key, (8, 5, 2)), jnp.full((8, 5), 0.2)
    normalizer = Normalizer.create(2)
    r2 = prior.r2(critics, center, theta, returns, states, weights, normalizer, pol)
    predicted = (
        center.value
        + jnp.stack(
            [
                prior.advantages(critics, t, s, w, center.theta, normalizer, pol)
                for t, s, w in zip(theta, states, weights, strict=True)
            ]
        ).T
    )
    released = 1 - jnp.clip((returns - predicted) ** 2, min=1e-4).sum(-1) / (
        ((returns - returns.mean()) ** 2).sum() + 1e-6
    )
    np.testing.assert_allclose(r2, released, rtol=1e-5)


def test_gp_sees_vectors_of_one_value_per_policy(key, tiny):
    """Guards against the released bug B1: returns, noise and prior mean are all [n], never [n, 1]."""
    from abps.experiment.evaluate import Evaluator

    config = tiny("abs")
    searcher = ABS(config, policy(config.env), key)
    envs = Envs(config.env, searcher.policy)
    for i in range(2):
        searcher.iteration(Evaluator(envs, searcher.normalizer, [0, i, 0]), i, jax.random.key(i))
    w, n, d = searcher.window, config.gp.n_max, searcher.policy.dim
    assert (w.theta.shape, w.returns.shape, w.noise.shape) == ((n, d), (n,), (n,))
    assert searcher.prior(w.theta).shape == (n,)
    assert searcher.posterior().alpha.shape == (n,)
    mean, cov = gp.gradient(searcher.posterior(), searcher.theta, searcher.prior_gradient(searcher.theta))
    assert (mean.shape, cov.shape) == ((d,), (d, d))


@pytest.mark.parametrize(
    ("aggregation", "reset_worst", "queries"),
    [("mean", False, 2), ("softmax", False, 2), ("mean", True, 2), ("mean", False, 0)],
)
def test_iteration_reuses_prior_and_preserves_per_query_scoring_results(
    aggregation, reset_worst, queries, key, tiny, monkeypatch
):
    from abps.experiment.train import Run, iteration

    class PerQueryABS(ABS):
        def learn(self, rollout, *, center, first, final):
            return super().learn(rollout, center=center, first=first, final=True)

    config = tiny("abs")
    config = replace(
        config,
        env=replace(config.env, normalize_obs=True),
        search=replace(config.search, queries=queries),
        gp=replace(config.gp, n_max=4),
        critic=replace(config.critic, aggregation=aggregation, reset_worst=reset_worst),
    )
    pol = policy(config.env)
    actual, reference = ABS(config, pol, key), PerQueryABS(config, pol, key)
    envs = [Envs(config.env, pol), Envs(config.env, pol)]
    calls = Counter()
    original_r2, original_prior = prior.r2, ABS.prior

    def score(*args, **kwargs):
        calls["r2"] += 1
        return original_r2(*args, **kwargs)

    def mean(searcher, theta):
        calls["prior"] += 1
        return original_prior(searcher, theta)

    monkeypatch.setattr(prior, "r2", score)
    monkeypatch.setattr(ABS, "prior", mean)
    for i in range(2):
        calls.clear()
        measured = iteration(actual, envs[0], Run(eval_episodes=1), i, jax.random.fold_in(key, i))
        observations = 2 if i == 0 else queries + 1
        assert calls["r2"] == (observations if aggregation != "mean" or reset_worst else 1)
        assert calls["prior"] == observations

        calls.clear()
        expected = iteration(reference, envs[1], Run(eval_episodes=1), i, jax.random.fold_in(key, i))
        assert calls["r2"] == observations
        for name in measured:
            if not name.startswith("time/"):
                np.testing.assert_array_equal(measured[name], expected[name], err_msg=name)
        for a, b in zip(jax.tree.leaves(nnx.state(actual)), jax.tree.leaves(nnx.state(reference)), strict=True):
            if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
            else:
                np.testing.assert_array_equal(a, b)
