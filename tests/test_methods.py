"""The update rules: the ABS step and its cap, MPD's walk, ARS's step, and the critics' validation R²."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from abps.core import critic, gp, prior
from abps.env.policy import Normalizer, Policy
from abps.env.rollout import Envs, Evaluation
from abps.experiment.evaluate import Evaluator
from abps.methods import ABS, ars, mpd
from abps.methods.local_bo import ascent


def window(key, d: int, n: int = 8):
    k1, k2 = jax.random.split(key)
    theta = 0.02 * jax.random.normal(k1, (n, d))
    return theta, theta @ jnp.linspace(1.0, 2.0, d) + 0.01 * jax.random.normal(k2, (n,))


def test_abs_steps_along_the_preconditioned_gradient_and_caps_it(key, tiny):
    config = tiny("abs")
    config = replace(config, search=replace(config.search, learning_rate=10.0))  # large, so the cap binds
    searcher = ABS(config, Envs(config.env).policy, key)
    theta, returns = window(key, searcher.policy.dim)
    searcher.window = searcher.window.push(
        Evaluation(theta, returns, jnp.zeros(8), jnp.zeros((8, 20, 4)), jnp.zeros((8, 20))), 8
    )
    posterior, _ = searcher.fit(key)
    mean, cov = gp.gradient(posterior, searcher.theta, searcher.prior.gradient(searcher.theta, searcher.normalizer))
    moved, _ = searcher.move(posterior)
    step = moved - searcher.theta
    np.testing.assert_allclose(jnp.mean(jnp.abs(step)), config.gp.lengthscale_bounds[1], rtol=1e-6)
    direction = jnp.linalg.solve(gp.regularize(cov), mean)
    np.testing.assert_allclose(step / jnp.linalg.norm(step), direction / jnp.linalg.norm(direction), rtol=1e-5)


def test_ascent_probability_is_the_released_formula(key):
    mean = jax.random.normal(key, (3,))
    root = jax.random.normal(jax.random.fold_in(key, 1), (3, 3))
    cov = root @ root.T + jnp.eye(3)
    direction, probability = ascent(mean, cov)
    released = jax.scipy.stats.norm.cdf(mean @ direction / jnp.sqrt(direction @ gp.regularize(cov) @ direction))
    np.testing.assert_allclose(probability, released, rtol=1e-5)
    assert ascent(jnp.zeros(3), cov)[1] == 0.5  # a flat window: no evidence either way


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
    pol, normalizer = Policy(2, 1), Normalizer.create(2)
    target = critic.Ensemble(2, 1, config, key).target
    center = Evaluation(
        jnp.zeros(2), jnp.asarray(0.3), jnp.zeros(()), jax.random.normal(key, (10, 2)), jnp.full(10, 0.1)
    )
    theta, returns = window(key, 2)
    validation = Evaluation(theta, returns, jnp.zeros(8), jax.random.normal(key, (8, 5, 2)), jnp.full((8, 5), 0.2))
    r2 = prior.r2(target, center, validation, normalizer, pol)
    predicted = (
        center.returns
        + jnp.stack(
            [
                prior.advantages(target, t, s, w, center, normalizer, pol)
                for t, s, w in zip(theta, validation.states, validation.weights, strict=True)
            ]
        ).T
    )
    released = 1 - jnp.clip((returns - predicted) ** 2, min=1e-4).sum(-1) / (
        ((returns - returns.mean()) ** 2).sum() + 1e-6
    )
    np.testing.assert_allclose(r2, released, rtol=1e-5)


def test_gp_sees_vectors_of_one_value_per_policy(key, tiny):
    """Guards against the released bug B1: returns, noise and prior mean are all [n], never [n, 1]."""
    config = tiny("abs")
    envs = Envs(config.env)
    searcher = ABS(config, envs.policy, key)
    for i in range(2):
        searcher.iteration(Evaluator(envs, searcher.normalizer, [0, i, 0]), jax.random.key(i))
    w, n, d = searcher.window, config.gp.n_max, searcher.policy.dim
    assert (w.theta.shape, w.returns.shape, w.noise.shape) == ((n, d), (n,), (n,))
    assert searcher.prior.value(w.theta, searcher.normalizer).shape == (n,)
    posterior, _ = searcher.fit(key)
    assert posterior.alpha.shape == (n,)
    mean, cov = gp.gradient(posterior, searcher.theta, searcher.prior.gradient(searcher.theta, searcher.normalizer))
    assert (mean.shape, cov.shape) == ((d,), (d, d))
