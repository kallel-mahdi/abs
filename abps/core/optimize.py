"""Box-constrained minimization from many starts at once: L-BFGS on x = lower + (upper − lower) · sigmoid(u).
The GP hyperparameter fit and the acquisition both use it."""

import jax
import jax.numpy as jnp
import optax


def minimize(fun, starts, lower, upper, *, maxiter: int, tolerance: float):
    """Minimize `fun` over the box [lower, upper] (pytrees) from every start (pytree with a leading restart axis).
    Returns the best finite point seen and its value, or +inf if every start is invalid."""

    def to_box(u):
        return jax.tree.map(lambda lo, hi, x: lo + (hi - lo) * jax.nn.sigmoid(x), lower, upper, u)

    def from_box(x):
        def logit(lo, hi, x):
            fraction = jnp.where(hi > lo, (x - lo) / jnp.where(hi > lo, hi - lo, 1), 0.5)
            return jax.scipy.special.logit(jnp.clip(fraction, 1e-6, 1 - 1e-6))

        return jax.tree.map(logit, lower, upper, x)

    def objective(u):
        return fun(to_box(u))

    optimizer = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(objective)

    def finite(tree):
        return jax.tree.reduce(lambda valid, x: valid & jnp.all(jnp.isfinite(x)), tree, jnp.asarray(True))

    def step(carry):
        u, state, best_u, best_value, stale = carry
        value, grad = value_and_grad(u, state=state)
        updates, state = optimizer.update(grad, state, u, value=value, grad=grad, value_fn=objective)
        u = optax.apply_updates(u, updates)
        value = optax.tree.get(state, "value")
        improved = finite(u) & jnp.isfinite(value) & (value < best_value)
        best_u = jax.tree.map(lambda new, old: jnp.where(improved, new, old), u, best_u)
        best_value = jnp.where(improved, value, best_value)
        return u, state, best_u, best_value, jnp.where(improved, 0, stale + 1)

    def running(carry):
        u, state, _, best_value, stale = carry
        count, grad = optax.tree.get(state, "count"), optax.tree.get(state, "grad")
        info = optax.tree.get(state, "info")
        healthy = finite((u, grad)) & jnp.isfinite(optax.tree.get(state, "value")) & (info.decrease_error <= 0)
        # A float32 line search can stop making progress before the gradient tolerance is reachable.
        return (
            (count < maxiter)
            & jnp.isfinite(best_value)
            & ((count == 0) | (healthy & (stale < 5) & (optax.tree.norm(grad) >= tolerance)))
        )

    def solve(start):
        u = from_box(start)
        value = objective(u)
        initial = (u, optimizer.init(u), u, jnp.where(jnp.isfinite(value), value, jnp.inf), jnp.asarray(0))
        _, _, best_u, best_value, _ = jax.lax.while_loop(running, step, initial)
        return to_box(best_u), best_value

    solutions, values = jax.vmap(solve)(starts)
    best = jnp.argmin(values)
    return jax.tree.map(lambda x: x[best], solutions), values[best]
