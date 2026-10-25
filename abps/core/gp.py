"""Exact GP regression of returns on policy parameters, its hyperparameter fit, and its gradient posterior (GIBO).

    k(x, z) = σ_f² exp(−½ ‖(x − z) / (ℓ √d)‖²),    y = f(θ) + ε,  ε ~ N(0, σ_n²),    f ~ GP(m, k)

The prior mean m is the critic advantage prior (ABS) or zero, plus a fitted constant (MPD)."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from abps.config import GP
from abps.core.optimize import minimize


@struct.dataclass
class Params:
    lengthscales: jax.Array  # [d]
    scale: jax.Array  # σ_f
    noise: jax.Array  # σ_n
    constant: jax.Array  # added to the prior mean


def kernel(params: Params, x: jax.Array, z: jax.Array) -> jax.Array:
    delta = (x[:, None] - z[None]) / (params.lengthscales * jnp.sqrt(x.shape[-1]))
    return params.scale**2 * jnp.exp(-0.5 * jnp.sum(delta**2, -1))


def kernel_gradient(params: Params, x: jax.Array, z: jax.Array) -> jax.Array:
    """∂k(x, z_j)/∂x for a single x: [len(z), d]."""
    return kernel(params, x[None], z)[0, :, None] * (z - x) / (params.lengthscales**2 * x.size)


def regularize(matrix: jax.Array) -> jax.Array:
    """Regularize before factorization, including the conditioning kernel: posterior covariances subtract nearly
    equal terms, so repairing only their final Cholesky leaves the unstable conditioning solve untouched."""
    jitter = 1e-10 + 1e-5 * jnp.max(jnp.abs(jnp.diag(matrix)))
    return matrix + jitter * jnp.eye(len(matrix), dtype=matrix.dtype)


def cholesky(matrix: jax.Array) -> jax.Array:
    return jnp.linalg.cholesky(regularize(matrix))


def solve(factor: jax.Array, rhs: jax.Array) -> jax.Array:
    return jax.scipy.linalg.cho_solve((factor, True), rhs)


def covariance(params: Params, theta: jax.Array) -> jax.Array:
    return kernel(params, theta, theta) + params.noise**2 * jnp.eye(len(theta))


def nll(params: Params, theta: jax.Array, residual: jax.Array) -> jax.Array:
    """Negative log marginal likelihood of `residual` = returns − prior mean, from one Cholesky factor."""
    factor = cholesky(covariance(params, theta))
    whitened = jax.scipy.linalg.solve_triangular(factor, residual - params.constant, lower=True)
    return 0.5 * jnp.sum(whitened**2) + jnp.log(jnp.diag(factor)).sum() + len(theta) / 2 * jnp.log(2 * jnp.pi)


@partial(jax.jit, static_argnames="config")
def fit(
    theta: jax.Array, returns: jax.Array, noise: jax.Array, prior: jax.Array, key: jax.Array, config: GP
) -> tuple[Params, jax.Array]:
    """Maximum-likelihood hyperparameters from `config.restarts` uniform starts. The output scale and noise are boxed
    within a factor 3 of the residual spread and of the measured return noise (the spread of returns over a policy's
    episodes, where there were several)."""
    if config.fit_mean == "average":
        # B1: the released code subtracted an (n,) mean from (n, 1) returns, so both the likelihood and the scale
        # bound saw every pairing y_i − m_j; the likelihood then reduces to one with the average prior mean.
        spread, prior = jnp.std(returns[:, None] - prior[None]), jnp.full_like(prior, prior.mean())
    else:
        spread = jnp.std(returns - prior)
    measured = noise > 0
    noise = jnp.where(measured.any(), (noise * measured).sum() / measured.sum(), 1e-4)
    constant = 1.0 if config.constant_mean else 0.0
    lo, hi = config.lengthscale_bounds
    d = theta.shape[-1]
    lower = Params(jnp.full(d, lo), spread / 3, noise / 3, jnp.asarray(-constant))
    upper = Params(jnp.full(d, hi), spread * 3, noise * 3, jnp.asarray(constant))
    keys = jax.random.split(key, 4)
    starts = jax.tree.map(
        lambda k, a, b: jax.random.uniform(k, (config.restarts, *a.shape), minval=a, maxval=b),
        Params(*keys),
        lower,
        upper,
    )
    params, value = minimize(
        lambda p: nll(p, theta, returns - prior),
        starts,
        lower,
        upper,
        maxiter=config.maxiter,
        tolerance=config.tolerance,
    )
    return params, value


@struct.dataclass
class Posterior:
    params: Params
    theta: jax.Array  # [n, d] conditioning points
    factor: jax.Array  # Cholesky factor of K + σ_n² I
    alpha: jax.Array  # (K + σ_n² I)⁻¹ (y − m)


def condition(params: Params, theta: jax.Array, returns: jax.Array, prior: jax.Array) -> Posterior:
    factor = cholesky(covariance(params, theta))
    return Posterior(params, theta, factor, solve(factor, returns - prior - params.constant))


def gradient(posterior: Posterior, theta: jax.Array, prior_gradient: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Mean and covariance of ∇f(θ) given the window:
    μ' = ∇m(θ) + ∂K(θ, X)ᵀ α,    Σ' = σ_f² diag(1 / (ℓ² d)) − ∂K(θ, X)ᵀ (K + σ_n² I)⁻¹ ∂K(θ, X)."""
    params = posterior.params
    cross = kernel_gradient(params, theta, posterior.theta)
    whitened = jax.scipy.linalg.solve_triangular(posterior.factor, cross, lower=True)
    curvature = params.scale**2 / (jnp.broadcast_to(params.lengthscales, theta.shape) ** 2 * theta.size)
    cov = jnp.diag(curvature) - whitened.T @ whitened
    return prior_gradient + cross.T @ posterior.alpha, cov
