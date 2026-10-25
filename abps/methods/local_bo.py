"""Local Bayesian optimization (GIBO, MPD, ABS). A GP models the return near the center θ from its n_max most recent
evaluations; each iteration evaluates the center, queries where the GP most sharpens its belief about ∇J(θ), and
moves θ along the posterior gradient. Methods choose the prior mean (core/prior.py) and the move."""

import jax
import jax.numpy as jnp
from flax import nnx

from abps.config import Config
from abps.core import acquisition, gp
from abps.core.prior import AdvantagePrior, ZeroPrior
from abps.env.policy import Policy
from abps.env.rollout import Evaluation, Rollout
from abps.methods.base import Evaluate, Searcher


class LocalBO(Searcher):
    window: Evaluation = nnx.data()

    def __init__(self, config: Config, policy: Policy, prior: AdvantagePrior | ZeroPrior):
        super().__init__(config, policy)
        self.prior = prior
        steps, d, o = config.search.episodes * config.env.horizon, policy.dim, policy.observation_dim
        self.window = Evaluation(
            jnp.zeros((0, d)), jnp.zeros(0), jnp.zeros(0), jnp.zeros((0, steps, o)), jnp.zeros((0, steps))
        )

    def move(self, posterior: gp.Posterior) -> tuple[jax.Array, dict]:
        raise NotImplementedError

    def iteration(self, evaluate: Evaluate, key: jax.Array) -> tuple[list[Rollout], dict]:
        """Evaluate the center; seed the window with n_max − 1 random policies around it (first iteration) or acquire
        `queries` policies one at a time; move. Per-query metrics are the last query's."""
        search, first = self.config.search, len(self.window) == 0
        noise_key, fit_key, query_key = jax.random.split(key, 3)
        metrics = self.observe(
            evaluate(self.theta[None], self.config.gp.center_episodes), "first" if first else "center"
        )
        if first:
            noise = jax.random.normal(noise_key, (self.config.gp.n_max - 1, self.theta.size))
            queries = [evaluate(self.theta + search.sigma * noise, search.episodes)]
            metrics |= self.observe(queries[0], "query")
        else:
            queries = []
            for acquire_key, key in jax.random.split(query_key, (search.queries, 2)):
                posterior, fitted = self.fit(key)
                local = acquisition.Local.at(posterior, self.theta, self.prior.gradient(self.theta, self.normalizer))
                query, acquired = acquisition.acquire(local, acquire_key, search.sigma, self.config.gp)
                queries.append(evaluate(query, search.episodes))
                metrics |= fitted | {f"acquisition/{name}": value for name, value in acquired.items()}
                metrics |= self.observe(queries[-1], "query")
        posterior, fitted = self.fit(fit_key)
        self.theta, moved = self.move(posterior)
        return queries, metrics | fitted | moved | self.prior.metrics(self.window, self.normalizer)

    def observe(self, rollout: Rollout, stage: str) -> dict:
        self.window = self.window.push(rollout.evaluation(self.config.search.episodes), self.config.gp.n_max)
        return self.prior.learn(rollout, stage, self.window, self.normalizer)

    def fit(self, key: jax.Array) -> tuple[gp.Posterior, dict]:
        """The GP's hyperparameters fitted to the window, and its posterior."""
        w = self.window
        mean = self.prior.value(w.theta, self.normalizer)
        params, nll = gp.fit(w.theta, w.returns, w.noise, mean, key, self.config.gp)
        metrics = {"gp/nll": nll, "gp/scale": params.scale, "gp/noise": params.noise}
        return gp.condition(params, w.theta, w.returns, mean), metrics


def ascent(mean: jax.Array, cov: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The most probable ascent direction d = Σ'⁻¹ μ' and the probability that it ascends,
    Φ(μ'ᵀd / √(dᵀΣ'd)) = Φ(√(μ'ᵀ Σ'⁻¹ μ'))."""
    factor = gp.cholesky(cov)
    whitened = jax.scipy.linalg.solve_triangular(factor, mean, lower=True)
    direction = jax.scipy.linalg.solve_triangular(factor.T, whitened, lower=False)
    return direction, jax.scipy.stats.norm.cdf(jnp.linalg.norm(whitened))
