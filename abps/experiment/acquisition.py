"""Collect the notebook's four-query Pendulum illustration without moving the center.

    python -m abps.experiment.acquisition

Bootstrap only the critic replay with 100 random policies, then condition the GP on the center and successive
queries. The noise cap and initial GP parameters are notebook illustration settings, separate from training runs.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from abps.config import Config
from abps.core import acquisition, gp
from abps.env.rollout import Envs
from abps.methods import ABS


def collect(output: Path = Path("plots/acquisition_loop.npz"), seed: int = 44, points: int = 200) -> None:
    config = Config.read("abs", "pendulum")
    config = replace(config, gp=replace(config.gp, lengthscale_bounds=(0.00125, 0.025)))
    init, bootstrap, key = jax.random.split(jax.random.key(seed), 3)
    envs = Envs(config.env)
    searcher = ABS(config, envs.policy, init)
    theta = searcher.theta
    random_policies = theta + config.search.sigma * jax.random.normal(bootstrap, (100, theta.size))
    rollout = envs(random_policies, searcher.normalizer, episodes=config.search.episodes, seed=[seed, 0])
    searcher.prior.replay = searcher.prior.replay.add(*rollout.rows)
    center = envs(theta[None], searcher.normalizer, episodes=config.gp.center_episodes, seed=[seed, 1])
    searcher.observe(center, "first")

    # Like the notebook, do not fit hyperparameters to the single center observation.
    params = gp.Params(
        jnp.full(2, sum(config.gp.lengthscale_bounds) / 2), jnp.asarray(0.02), jnp.asarray(0.01), jnp.zeros(())
    )
    grid = np.linspace(-0.05, 0.05, points)
    offsets = jnp.asarray(np.stack(np.meshgrid(grid, grid, indexing="xy"), -1).reshape(-1, 2))
    evaluate_grid = jax.jit(jax.vmap(acquisition.gain, (0, None)))
    data = {
        "grid": grid,
        "theta": np.asarray(theta),
        "seed": seed,
        "config": json.dumps(asdict(config)),
        "stages": config.search.queries,
    }

    for stage, (fit_key, query_key) in enumerate(jax.random.split(key, (config.search.queries, 2))):
        window = searcher.window
        mean = searcher.prior.value(window.theta, searcher.normalizer)
        if stage:
            params, _ = gp.fit(window.theta, window.returns, window.noise, mean, fit_key, config.gp)
        params = params.replace(noise=jnp.minimum(params.noise, params.scale / 2))  # notebook's illustration cap
        posterior = gp.condition(params, window.theta, window.returns, mean)
        prior_gradient = searcher.prior.gradient(theta, searcher.normalizer)
        local = acquisition.Local.at(posterior, theta, prior_gradient)
        query, _ = acquisition.acquire(local, query_key, config.search.sigma, config.gp)
        gain = evaluate_grid(offsets, local).reshape(points, points)
        values = {
            "gain": gain,
            "observed": window.theta,
            "returns": window.returns,
            "prior": mean,
            "query": query[0],
            "prior_gradient": prior_gradient,
            "posterior_gradient": local.mean,
            "lengthscales": params.lengthscales,
            "scale": params.scale,
            "noise": params.noise,
        }
        values = {name: np.asarray(value) for name, value in jax.device_get(values).items()}
        if not all(np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError(f"Nonfinite illustration at query {stage + 1}")
        data.update({f"{name}_{stage}": value for name, value in values.items()})
        rollout = envs(query, searcher.normalizer, episodes=config.search.episodes, seed=[seed, stage + 2])
        searcher.observe(rollout, "query")
        print(f"query {stage + 1}: {values['query']}, objective {float(rollout.returns[0]):.4f}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    for env in envs.pool:
        env.close()
    print(output, flush=True)


if __name__ == "__main__":
    tyro.cli(collect)
