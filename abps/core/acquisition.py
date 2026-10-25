"""Where to evaluate next: the queries Z near θ that most sharpen the GP's belief about the gradient at θ (MPD's
acquisition). With μ', Σ' the gradient posterior given the window X, C = Cov(∇f(θ), y_Z | X), S = Cov(y_Z | X) and
U = Cᵀ Σ'⁻¹ C, the expected squared Mahalanobis norm of the gradient after observing Z is, by Woodbury,

    info(Z) = μ'ᵀ Σ'⁻¹ μ' + pᵀ (S − U)⁻¹ p + tr((S − U)⁻¹ U),    p = Cᵀ Σ'⁻¹ μ',

which needs only |Z|-sized factorizations per evaluation (the released code refactored a d × d matrix each time)."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from abps.config import GP
from abps.core import gp
from abps.core.optimize import minimize


@struct.dataclass
class Local:
    """What every evaluation of `info` reuses: the gradient posterior at θ and its factorization."""

    posterior: gp.Posterior
    theta: jax.Array
    mean: jax.Array  # μ'
    factor: jax.Array  # Cholesky factor of Σ'
    whitened_cross: jax.Array  # L_K⁻¹ ∂K(θ, X)
    whitened_mean: jax.Array  # L_gradient⁻¹ μ'

    @classmethod
    def at(cls, posterior: gp.Posterior, theta: jax.Array, prior_gradient: jax.Array) -> "Local":
        mean, cov = gp.gradient(posterior, theta, prior_gradient)
        factor = gp.cholesky(cov)
        cross = gp.kernel_gradient(posterior.params, theta, posterior.theta)
        return cls(
            posterior,
            theta,
            mean,
            factor,
            jax.scipy.linalg.solve_triangular(posterior.factor, cross, lower=True),
            jax.scipy.linalg.solve_triangular(factor, mean, lower=True),
        )


def info(offsets: jax.Array, local: Local) -> jax.Array:
    return local.whitened_mean @ local.whitened_mean + gain(offsets, local)


def gain(offsets: jax.Array, local: Local) -> jax.Array:
    """The query-dependent information; exclude the constant baseline during line search."""
    params, posterior = local.posterior.params, local.posterior
    queries = local.theta + offsets.reshape(-1, local.theta.size)
    cross = gp.kernel(params, posterior.theta, queries)
    whitened = jax.scipy.linalg.solve_triangular(posterior.factor, cross, lower=True)
    s = gp.covariance(params, queries) - whitened.T @ whitened
    c = gp.kernel_gradient(params, local.theta, queries).T - local.whitened_cross.T @ whitened
    whitened_c = jax.scipy.linalg.solve_triangular(local.factor, c, lower=True)
    u = whitened_c.T @ whitened_c
    p = whitened_c.T @ local.whitened_mean
    conditional = s - u
    if len(queries) == 1:
        # Independent observation noise remains after conditioning on the gradient.
        conditional = jnp.maximum(conditional, params.noise**2)
    factor = gp.cholesky(conditional)
    return p @ gp.solve(factor, p) + jnp.trace(gp.solve(factor, u))


@partial(jax.jit, static_argnames="config")
def acquire(local: Local, key: jax.Array, sigma: float, config: GP) -> tuple[jax.Array, dict]:
    """The query maximizing `info`, from `config.restarts` starts θ + σ ε clipped to the query box."""
    bound = jnp.full(local.theta.size, config.query_bound)
    starts = jnp.clip(sigma * jax.random.normal(key, (config.restarts, bound.size)), -bound, bound)
    offsets, value = minimize(
        lambda z: -gain(z, local), starts, -bound, bound, maxiter=config.maxiter, tolerance=config.tolerance
    )
    return local.theta + offsets[None], {"info": local.whitened_mean @ local.whitened_mean - value}
