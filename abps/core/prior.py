"""ABS's advantage prior: the performance-difference lemma with critics of the center policy π_c, on its own states,

    m(θ) = J(θ_c) + Σ_k w_k · (1/E) Σ_{s ~ π_c} γ^t [Q̄_k(s, π_θ(s)) − Q̄_k(s, π_c(s))],

with Q̄_k the twin average of critic k's target networks. The weights w come from each critic's R² at predicting the
GP window's returns, every policy judged on its own states (the same lemma read the other way round)."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx, struct

from abps.core.critic import Critic, q_values
from abps.env.policy import Normalizer, Policy

AGGREGATIONS = {
    "mean": lambda r2: jnp.full_like(r2, 1 / len(r2)),
    "softmax": jax.nn.softmax,
    "max": lambda r2: jax.nn.one_hot(jnp.argmax(r2), len(r2)),
}


@struct.dataclass
class Center:
    theta: jax.Array
    value: jax.Array  # its measured return J(θ_c)
    states: jax.Array  # all its episodes' states, concatenated in time
    weights: jax.Array  # γ^t / episodes


def advantages(critics: Critic, theta, states, weights, center_theta, normalizer: Normalizer, policy: Policy):
    """Each critic's estimate of J(θ) − J(θ_c) from the weighted states [critic]."""
    s = normalizer(states)

    def one(network):
        q = q_values(network, s, policy(theta, s)).mean(0)
        return jnp.sum(weights * (q - q_values(network, s, policy(center_theta, s)).mean(0)))

    return nnx.vmap(one)(critics.target)


@partial(nnx.jit, graph=False, static_argnames="policy")
def value(critics: Critic, weights, center: Center, theta, normalizer: Normalizer, policy: Policy):
    """m(θ) for every row of θ [n, d]."""
    each = jax.vmap(advantages, in_axes=(None, 0, None, None, None, None, None))
    return (
        center.value + each(critics, theta, center.states, center.weights, center.theta, normalizer, policy) @ weights
    )


@partial(nnx.jit, graph=False, static_argnames="policy")
def gradient(critics: Critic, weights, center: Center, theta, normalizer: Normalizer, policy: Policy):
    """∇m(θ) at a single θ."""
    return jax.grad(lambda t: value(critics, weights, center, t[None], normalizer, policy)[0])(theta)


@partial(nnx.jit, graph=False, static_argnames="policy")
def r2(critics: Critic, center: Center, theta, returns, states, weights, normalizer: Normalizer, policy: Policy):
    """Each critic's R² over the policies θ [n, d] with measured `returns`, from their own weighted `states`."""
    each = jax.vmap(advantages, in_axes=(None, 0, 0, 0, None, None, None))
    predicted = center.value + each(critics, theta, states, weights, center.theta, normalizer, policy).T  # [critic, n]
    errors = jnp.maximum((predicted - returns) ** 2, 1e-4).sum(-1)
    return 1 - errors / (jnp.sum((returns - returns.mean()) ** 2) + 1e-6)
