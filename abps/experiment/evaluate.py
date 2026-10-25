"""Every episode a run plays goes through here, seeded from (seed, iteration, call). The searcher's own evaluations
are counted as the episodes it consumes; the measurements below never change the searcher."""

import time

import jax.numpy as jnp

from abps.env.policy import Normalizer
from abps.env.rollout import Envs, Rollout


class Evaluator:
    """A searcher's `evaluate` for one iteration: it plays policies with the iteration's normalizer and counts the
    episodes and the seconds spent in the simulator."""

    def __init__(self, envs: Envs, normalizer: Normalizer, seed: list[int]):
        self.envs, self.normalizer, self.seed = envs, normalizer, seed
        self.calls, self.episodes, self.seconds = 0, 0, 0.0

    def __call__(self, theta, episodes: int) -> Rollout:
        start = time.perf_counter()
        rollout = self.envs(theta, self.normalizer, episodes=episodes, seed=[*self.seed, self.calls])
        self.calls, self.episodes = self.calls + 1, self.episodes + rollout.episodes
        self.seconds += time.perf_counter() - start
        return rollout


def measure(center: Rollout, queries: list[Rollout], evaluator: Evaluator) -> dict:
    """The center and this iteration's best return, raw and discounted/scaled. Log retains these iteration maxima
    and accumulates their running maxima for the paper's learning curve."""
    raw = jnp.concatenate([center.raw_returns, *[q.raw_returns for q in queries]])
    objective = jnp.concatenate([center.returns, *[q.returns for q in queries]])
    return {
        "eval/return": center.raw_returns[0],
        "eval/return_max": raw.max(),
        "eval/objective": center.returns[0],
        "eval/objective_max": objective.max(),
        "episodes": evaluator.episodes,
        "time/simulator": evaluator.seconds,
    }


def test_prior(envs: Envs, searcher, queries: list[Rollout], episodes: int, seed: list[int]) -> dict:
    """R² of the learned prior at this iteration's queries, against fresh evaluations of them (the paper's R²
    figure); only methods with a learned prior."""
    theta = jnp.concatenate([q.theta for q in queries])
    observed = envs(theta, searcher.normalizer, episodes=episodes, seed=seed).returns
    residual = jnp.sum((observed - searcher.prior.value(theta, searcher.normalizer)) ** 2)
    return {"eval/prior_r2": 1 - residual / jnp.sum((observed - observed.mean()) ** 2)}
