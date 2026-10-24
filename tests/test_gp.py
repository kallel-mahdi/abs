"""The GP equations against dense references and against the released implementation's formulas."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abps.config import GP
from abps.core import acquisition, gp
from abps.core.optimize import minimize

D, N = 3, 7


@pytest.fixture
def problem(key):
    k1, k2, k3 = jax.random.split(key, 3)
    params = gp.Params(jnp.array([0.3, 0.5, 0.4]), jnp.asarray(1.3), jnp.asarray(0.2), jnp.asarray(0.1))
    theta = jax.random.normal(k1, (N, D)) * 0.05
    returns = jax.random.normal(k2, (N,))
    prior = jax.random.normal(k3, (N,)) * 0.1
    return params, theta, returns, prior


def test_nll_is_the_gaussian_marginal_likelihood(problem):
    params, theta, returns, prior = problem
    mean = prior + params.constant
    expected = -jax.scipy.stats.multivariate_normal.logpdf(returns, mean, gp.covariance(params, theta))
    np.testing.assert_allclose(gp.nll(params, theta, returns - prior), expected, rtol=1e-4)


def test_nll_gradient_matches_the_two_solve_form(problem):
    params, theta, returns, prior = problem
    residual = returns - prior

    def reference(params):
        factor = gp.cholesky(gp.covariance(params, theta))
        r = residual - params.constant
        return 0.5 * r @ gp.solve(factor, r) + jnp.log(jnp.diag(factor)).sum() + N / 2 * jnp.log(2 * jnp.pi)

    actual = jax.grad(gp.nll)(params, theta, residual)
    expected = jax.grad(reference)(params)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_average_mean_fit_is_the_released_broadcast_objective(problem):
    """B1: the released likelihood of an (n, 1) target minus an (n,) mean equals n² times the likelihood with the
    average mean, so both have the same maximizer."""
    params, theta, returns, prior = problem
    y = returns[:, None] - prior[None] - params.constant  # the released (n, n) broadcast
    k = gp.covariance(params, theta)
    released = 0.5 * jnp.sum(y.T @ jnp.linalg.solve(k, y) + jnp.linalg.slogdet(k)[1] + N * jnp.log(2 * jnp.pi))
    np.testing.assert_allclose(released, N**2 * gp.nll(params, theta, returns - prior.mean()), rtol=1e-4)


def test_gradient_posterior_is_the_derivative_of_the_posterior(problem):
    params, theta, returns, prior = problem
    posterior = gp.condition(params, theta, returns, prior)
    x = theta[0] + 0.01

    def mean(x):  # the posterior mean of f, with zero prior gradient at x
        return gp.kernel(params, x[None], theta)[0] @ posterior.alpha

    def cross_covariance(x, z):  # the posterior covariance of f(x) and f(z)
        kxz = gp.kernel(params, x[None], z[None])[0, 0]
        kx, kz = gp.kernel(params, x[None], theta)[0], gp.kernel(params, z[None], theta)[0]
        return kxz - kx @ gp.solve(posterior.factor, kz)

    mu, sigma = gp.gradient(posterior, x, jnp.zeros(D))
    np.testing.assert_allclose(mu, jax.grad(mean)(x), rtol=1e-4)
    np.testing.assert_allclose(sigma, jax.jacfwd(jax.grad(cross_covariance), argnums=1)(x, x), rtol=1e-4, atol=1e-6)


def test_info_is_the_released_acquisition(problem):
    """The Woodbury form equals the released b1 + b2, computed from the joint factorization of X ∪ Z."""
    params, theta, returns, prior = problem
    posterior = gp.condition(params, theta, returns, prior)
    x, prior_gradient = theta[0] + 0.01, jnp.array([0.2, -0.1, 0.3])
    local = acquisition.Local.at(posterior, x, prior_gradient)
    offset = jnp.array([0.02, -0.03, 0.01])

    z = x + offset[None]
    joint = jnp.concatenate([theta, z])
    cross = gp.kernel_gradient(params, x, joint)
    curvature = jnp.diag(params.scale**2 / (params.lengthscales**2 * D))
    sigma_xz = curvature - cross.T @ jnp.linalg.solve(gp.covariance(params, joint), cross)
    mu, _ = gp.gradient(posterior, x, prior_gradient)
    k_xz = gp.kernel(params, theta, z)
    sigma_z = gp.covariance(params, z) - k_xz.T @ jnp.linalg.solve(gp.covariance(params, theta), k_xz)
    c = gp.kernel_gradient(params, x, z).T - gp.kernel_gradient(params, x, theta).T @ jnp.linalg.solve(
        gp.covariance(params, theta), k_xz
    )
    a = jax.scipy.linalg.solve_triangular(jnp.linalg.cholesky(sigma_z), c.T, lower=True).T
    released = mu @ jnp.linalg.solve(sigma_xz, mu) + jnp.trace(a.T @ jnp.linalg.solve(sigma_xz, a))
    np.testing.assert_allclose(acquisition.info(offset, local), released, rtol=1e-4)


def test_minimize_stops_at_the_box():
    def fun(x):
        return jnp.sum((x - jnp.array([2.0, 0.3])) ** 2)

    starts = jnp.array([[0.1, 0.1], [0.9, 0.9]])
    lower, upper = jnp.zeros(2), jnp.ones(2)
    best, _ = minimize(fun, starts, lower, upper, maxiter=200, tolerance=1e-8)
    np.testing.assert_allclose(best, [1.0, 0.3], atol=2e-3)


def test_acquire_reports_full_information_when_optimizing_gain(problem, key):
    params, theta, returns, prior = problem
    local = acquisition.Local.at(gp.condition(params, theta, returns, prior), theta[0], jnp.ones(D))
    config = GP(n_max=N, center_episodes=1, lengthscale_bounds=(0.1, 1.0), constant_mean=True, restarts=2, maxiter=20)
    query, metrics = acquisition.acquire(local, key, 0.02, config)
    offset = query[0] - local.theta
    np.testing.assert_allclose(metrics["info"], acquisition.info(offset, local), rtol=1e-4)
    np.testing.assert_allclose(jax.grad(acquisition.gain)(offset, local), jax.grad(acquisition.info)(offset, local))


def test_minimize_skips_nonfinite_starts():
    def fun(x):
        return jnp.where(x[0] < 0.5, (x[0] - 0.2) ** 2, jnp.nan)

    starts = jnp.array([[0.9], [0.1]])
    best, value = minimize(fun, starts, jnp.zeros(1), jnp.ones(1), maxiter=100, tolerance=1e-6)
    np.testing.assert_allclose(best, [0.2], atol=1e-3)
    assert jnp.isfinite(value)
    _, invalid = minimize(fun, starts[:1], jnp.zeros(1), jnp.ones(1), maxiter=100, tolerance=1e-6)
    assert jnp.isposinf(invalid)


def test_minimize_stops_on_a_float32_plateau():
    calls = []

    def fun(x):
        jax.debug.callback(lambda _: calls.append(1), x, ordered=True)
        return 1e8 + jnp.sum(x**2)

    result = minimize(fun, jnp.array([[0.5]]), jnp.zeros(1), jnp.ones(1), maxiter=200, tolerance=1e-12)
    jax.block_until_ready(result)
    assert len(calls) < 150
    assert jnp.isfinite(result[1])


def test_fit_improves_on_its_starts(problem, key):
    params, theta, returns, prior = problem
    config = GP(n_max=N, center_episodes=1, lengthscale_bounds=(0.1, 1.0), constant_mean=True, restarts=4)
    noise = jnp.full(N, 0.2)
    fitted, value = gp.fit(theta, returns, noise, prior, key, config)
    assert value < gp.nll(params, theta, returns - prior)
    assert fitted.lengthscales.min() >= 0.1
    assert fitted.lengthscales.max() <= 1.0
