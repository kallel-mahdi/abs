"""The GP equations against dense references and against the released implementation's formulas."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abps.config import GP
from abps.core import acquisition, gp
from abps.core.optimize import minimize
from abps.methods.local_bo import ascent

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
    expected = -jax.scipy.stats.multivariate_normal.logpdf(returns, mean, gp.regularize(gp.covariance(params, theta)))
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
    k = gp.regularize(gp.covariance(params, theta))
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


def test_dense_window_is_regularized_before_conditioning(key):
    """Near-duplicate policies with tiny observation noise must still give a valid gradient posterior in FP32."""
    points = jax.random.normal(key, (51, 6)) * 1e-4
    params = gp.Params(jnp.full(6, 0.1), jnp.asarray(0.3), jnp.asarray(1e-6), jnp.zeros(()))
    returns = jnp.sin(points[:, 0])
    prior = jnp.zeros(len(points))
    posterior = gp.condition(params, points, returns, prior)
    mean, cov = gp.gradient(posterior, points[0], jnp.zeros(6))
    assert jnp.isfinite(posterior.factor).all()
    assert jnp.linalg.eigvalsh(cov).min() > 0
    assert jnp.isfinite(acquisition.Local.at(posterior, points[0], jnp.zeros(6)).factor).all()
    assert jnp.isfinite(ascent(mean, cov)[0]).all()
    assert all(jnp.isfinite(x).all() for x in jax.tree.leaves(jax.grad(gp.nll)(params, points, returns)))


def test_nearly_duplicate_windows_remain_positive_definite(key):
    """FP32 kernel and derivative rounding disagree for clustered policies; an epsilon-sized nugget is insufficient."""
    def one(key):
        a, b, c, d, e = jax.random.split(key, 5)
        length = 10 ** jax.random.uniform(a, (), minval=-3.0, maxval=0.0)
        scale = 10 ** jax.random.uniform(b, (), minval=-3.0, maxval=0.0)
        noise = 10 ** jax.random.uniform(c, (), minval=-8.0, maxval=-1.0)
        radius = length * 10 ** jax.random.uniform(d, (), minval=-5.0, maxval=1.0)
        points = radius * jax.random.normal(e, (51, 33))
        params = gp.Params(jnp.full(33, length), scale, noise, jnp.zeros(()))
        posterior = gp.condition(params, points, jnp.zeros(51), jnp.zeros(51))
        _, cov = gp.gradient(posterior, points[0], jnp.zeros(33))
        return jnp.linalg.eigvalsh(cov)[0]

    assert (jax.jit(jax.vmap(one))(jax.random.split(key, 128)) > 0).all()


def test_hopper_gradient_covariance_remains_factorizable():
    """The first failed query from seed 0; inverse-form covariance lost positive definiteness on the GPU."""
    with np.load(Path(__file__).parent / "data" / "hopper_gradient_covariance.npz") as fixture:
        data = {name: jnp.asarray(value) for name, value in fixture.items()}
    params = gp.Params(*(data[name] for name in ("lengthscales", "scale", "noise", "constant")))
    posterior = gp.Posterior(params, data["points"], data["factor"], data["alpha"])
    mean, cov = gp.gradient(posterior, data["theta"], data["prior_gradient"])
    direction, probability = ascent(mean, cov)
    local = acquisition.Local.at(posterior, data["theta"], data["prior_gradient"])
    assert jnp.isfinite(gp.cholesky(cov)).all()
    assert jnp.isfinite(direction).all()
    assert jnp.isfinite(probability)
    assert jnp.isfinite(acquisition.info(jnp.zeros_like(data["theta"]), local))


def test_ascent_confidence_uses_the_regularized_covariance():
    direction, probability = ascent(jnp.ones(1), jnp.array([[-5e-11]]))
    assert jnp.isfinite(direction).all()
    assert probability == 1
    zero, probability = ascent(jnp.zeros(1), jnp.ones((1, 1)))
    np.testing.assert_array_equal(zero, [0])
    assert probability == 0.5


def test_info_matches_direct_gaussian_conditioning(problem):
    """Woodbury equals the direct expected Mahalanobis norm after conditioning, with the same regularization."""
    params, theta, returns, prior = problem
    posterior = gp.condition(params, theta, returns, prior)
    x, prior_gradient = theta[0] + 0.01, jnp.array([0.2, -0.1, 0.3])
    local = acquisition.Local.at(posterior, x, prior_gradient)
    offset = jnp.array([0.02, -0.03, 0.01])

    z = x + offset[None]
    curvature = jnp.diag(params.scale**2 / (params.lengthscales**2 * D))
    k = gp.regularize(gp.covariance(params, theta))
    cross = gp.kernel_gradient(params, x, theta)
    sigma = gp.regularize(curvature - cross.T @ jnp.linalg.solve(k, cross))
    mu, _ = gp.gradient(posterior, x, prior_gradient)
    k_xz = gp.kernel(params, theta, z)
    sigma_z = gp.covariance(params, z) - k_xz.T @ jnp.linalg.solve(k, k_xz)
    c = gp.kernel_gradient(params, x, z).T - cross.T @ jnp.linalg.solve(k, k_xz)
    conditional = sigma_z - c.T @ jnp.linalg.solve(sigma, c)
    sigma_z += gp.regularize(conditional) - conditional
    explained = c @ jnp.linalg.solve(sigma_z, c.T)
    sigma_xz = sigma - explained
    released = mu @ jnp.linalg.solve(sigma_xz, mu) + jnp.trace(jnp.linalg.solve(sigma_xz, explained))
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
