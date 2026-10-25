"""The run: the same loop for every method. Each iteration measures the center, lets the searcher run its iteration,
and updates the observation normalizer with the queries' states; the run is checkpointed, logged and resumable."""

import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax

from abps.config import Config
from abps.env.rollout import Envs
from abps.experiment import evaluate
from abps.methods import METHODS
from abps.methods.base import Searcher
from abps.runs import store
from abps.runs.log import Log

terminated = threading.Event()


@dataclass(frozen=True, kw_only=True)
class Run:
    """How a run is executed and measured; nothing here changes the search."""

    seed: int = 0
    output: Path = Path("runs")
    eval_episodes: int = 5  # fresh episodes of the center per iteration, for the learning curve
    prior_test_episodes: int = 0  # >0: re-evaluate the queries to test the learned prior (the paper's R² figure)
    save_every: int = 10  # iterations between checkpoints
    wandb: Literal["disabled", "offline", "online"] = "disabled"


def train(config: Config, run: Run) -> Path:
    """Run or resume the experiment; returns its directory."""
    path = store.directory(run.output, config, run.seed)
    init, key = jax.random.split(jax.random.key(run.seed))
    envs = Envs(config.env)
    searcher = METHODS[config.method](config, envs.policy, init)
    signal.signal(signal.SIGTERM, lambda *_: terminated.set())
    with store.checkpointer(path, config, run.seed, every=run.save_every) as checkpointer:
        searcher, start = store.restore(checkpointer, searcher)
        with Log(path, config, run.wandb, start=start) as log:
            for i in range(start, config.search.iterations):
                log(i + 1, iteration(searcher, envs, run, i, jax.random.fold_in(key, i)))
                store.save(checkpointer, searcher, i + 1, now=terminated.is_set())
                if terminated.is_set():  # leaving the checkpointer waits for the save to commit
                    break
    return path


def iteration(searcher: Searcher, envs: Envs, run: Run, i: int, key: jax.Array) -> dict:
    start = time.perf_counter()
    center = envs(searcher.theta[None], searcher.normalizer, episodes=run.eval_episodes, seed=[run.seed, i, 1])
    evaluator = evaluate.Evaluator(envs, searcher.normalizer, seed=[run.seed, i, 0])
    queries, metrics = searcher.iteration(evaluator, key)
    metrics |= evaluate.measure(center, queries, evaluator)
    if run.prior_test_episodes:
        metrics |= evaluate.test_prior(envs, searcher, queries, run.prior_test_episodes, seed=[run.seed, i, 2])
    # The normalizer learns from the queries after they are measured, as in the released runs.
    if searcher.config.env.normalize_obs:
        for query in queries:
            searcher.normalizer = searcher.normalizer.update(query.transitions.observations, query.valid)
    return metrics | {"time/iteration": time.perf_counter() - start}
