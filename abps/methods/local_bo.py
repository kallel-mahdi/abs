"""Local Bayesian optimization (GIBO, MPD, ABS). A GP models the return near the center θ from its n_max most recent
evaluations; each iteration evaluates the center, queries where the GP most sharpens its belief about ∇J(θ), and
moves θ along the posterior gradient. Subclasses choose the prior mean and the move."""

import jax
import jax.numpy as jnp
from flax import nnx, struct

from abps.config import Config
from abps.core import acquisition, gp
from abps.env.policy import Policy
from abps.env.rollout import Rollout
from abps.methods.base import Evaluate, Report, Searcher


@struct.dataclass
class Window:
    """The evaluations the GP conditions on, with the states a learned prior is validated on."""

    theta: jax.Array  # [n, d]
    returns: jax.Array  # [n]
    noise: jax.Array  # [n]
    states: jax.Array  # [n, episodes · time, obs]
    weights: jax.Array  # [n, episodes · time]  γ^t / episodes

    @classmethod
    def of(cls, rollout: Rollout, episodes: int) -> "Window":
        return cls(rollout.theta, rollout.returns, rollout.noise, *rollout.states(episodes))

    def push(self, other: "Window", n_max: int) -> "Window":
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y])[-n_max:], self, other)

    def __len__(self) -> int:
        return len(self.returns)


class LocalBO(Searcher):
    window: Window = nnx.data()
    params: gp.Params = nnx.data()

    def __init__(self, config: Config, policy: Policy, key: jax.Array):
        super().__init__(config, policy, key)
        self.params = gp.Params.create(policy.dim, config.gp)
        steps = config.search.episodes * config.env.horizon
        d, o = policy.dim, policy.observation_dim
        self.window = Window(
            jnp.zeros((0, d)), jnp.zeros(0), jnp.zeros(0), jnp.zeros((0, steps, o)), jnp.zeros((0, steps))
        )

    # The prior mean and what it learns from each rollout: zero and nothing, unless a subclass learns one.
    def prior(self, theta: jax.Array) -> jax.Array:
        return jnp.zeros(len(theta))

    def prior_gradient(self, theta: jax.Array) -> jax.Array:
        return jnp.zeros_like(theta)

    def learn(self, rollout: Rollout, *, center: bool, first: bool, final: bool) -> dict:
        return {}

    def move(self, posterior: gp.Posterior) -> tuple[jax.Array, dict]:
        raise NotImplementedError

    def iteration(self, evaluate: Evaluate, i: int, key: jax.Array) -> Report:
        """Evaluate the center; seed the window with n_max − 1 random policies around it (first iteration) or acquire
        `queries` policies one at a time; move. Per-query metrics are the last query's."""
        search, center_key, query_key = self.config.search, *jax.random.split(key)
        center = evaluate(self.theta[None], self.config.gp.center_episodes)
        posterior, metrics = self.observe(
            center, center=True, first=i == 0, final=i > 0 and search.queries == 0, key=center_key
        )
        if i == 0:
            noise_key, fit_key = jax.random.split(query_key)
            noise = jax.random.normal(noise_key, (self.config.gp.n_max - 1, self.theta.size))
            queries = [evaluate(self.theta + search.sigma * noise, search.episodes)]
            posterior, observed = self.observe(queries[0], center=False, first=False, final=True, key=fit_key)
            metrics |= observed
        else:
            queries = []
            for j, (acquire_key, fit_key) in enumerate(jax.random.split(query_key, (search.queries, 2))):
                local = acquisition.Local.at(posterior, self.theta, self.prior_gradient(self.theta))
                query, acquired = acquisition.acquire(local, acquire_key, search.sigma, self.config.gp)
                queries.append(evaluate(query, search.episodes))
                posterior, observed = self.observe(
                    queries[-1], center=False, first=False, final=j == search.queries - 1, key=fit_key
                )
                metrics |= observed
                metrics |= {f"acquisition/{name}": value for name, value in acquired.items()}
        self.theta, moved = self.move(posterior)
        return Report(queries, metrics | moved)

    def observe(
        self, rollout: Rollout, *, center: bool, first: bool, final: bool, key: jax.Array
    ) -> tuple[gp.Posterior, dict]:
        """Learn from a rollout, evaluate the prior once, and return the fitted posterior."""
        self.window = self.window.push(Window.of(rollout, self.config.search.episodes), self.config.gp.n_max)
        metrics = self.learn(rollout, center=center, first=first, final=final)
        w = self.window
        mean = self.prior(w.theta)
        if len(w) > 1:
            self.params, nll = gp.fit(w.theta, w.returns, w.noise, mean, key, self.config.gp)
            metrics |= {"gp/nll": nll, "gp/scale": self.params.scale, "gp/noise": self.params.noise}
        return gp.condition(self.params, w.theta, w.returns, mean), metrics

    def posterior(self) -> gp.Posterior:
        w = self.window
        return gp.condition(self.params, w.theta, w.returns, self.prior(w.theta))


def ascent(mean: jax.Array, cov: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The most probable ascent direction Σ'⁻¹ μ' and the probability that it ascends."""
    direction = gp.solve(gp.cholesky(cov), mean)
    return direction, jax.scipy.stats.norm.cdf(mean @ direction / jnp.sqrt(direction @ cov @ direction))
