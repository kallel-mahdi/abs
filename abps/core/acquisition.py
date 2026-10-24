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
    whitened_cross: jax.Array  # (K + σ_n² I)⁻¹ ∂K(θ, X)
    whitened_mean: jax.Array  # Σ'⁻¹ μ'

    @classmethod
    def at(cls, posterior: gp.Posterior, theta: jax.Array, prior_gradient: jax.Array) -> "Local":
        mean, cov = gp.gradient(posterior, theta, prior_gradient)
        factor = gp.cholesky(cov)
        cross = gp.kernel_gradient(posterior.params, theta, posterior.theta)
        return cls(posterior, theta, mean, factor, gp.solve(posterior.factor, cross), gp.solve(factor, mean))


def info(offsets: jax.Array, local: Local) -> jax.Array:
    return local.mean @ local.whitened_mean + gain(offsets, local)


def gain(offsets: jax.Array, local: Local) -> jax.Array:
    """The query-dependent information; exclude the constant baseline during line search."""
    params, posterior = local.posterior.params, local.posterior
    queries = local.theta + offsets.reshape(-1, local.theta.size)
    cross = gp.kernel(params, posterior.theta, queries)
    s = gp.covariance(params, queries) - cross.T @ gp.solve(posterior.factor, cross)
    c = gp.kernel_gradient(params, local.theta, queries).T - local.whitened_cross.T @ cross
    u = c.T @ gp.solve(local.factor, c)
    p = c.T @ local.whitened_mean
    factor = gp.cholesky(s - u)
    return p @ gp.solve(factor, p) + jnp.trace(gp.solve(factor, u))


@partial(jax.jit, static_argnames="config")
def acquire(local: Local, key: jax.Array, sigma: float, config: GP) -> tuple[jax.Array, dict]:
    """The query maximizing `info`, from `config.restarts` starts θ + σ ε clipped to the query box."""
    bound = jnp.full(local.theta.size, config.query_bound)
    starts = jnp.clip(sigma * jax.random.normal(key, (config.restarts, bound.size)), -bound, bound)
    offsets, value = minimize(
        lambda z: -gain(z, local), starts, -bound, bound, maxiter=config.maxiter, tolerance=config.tolerance
    )
    return local.theta + offsets[None], {"info": local.mean @ local.whitened_mean - value}
