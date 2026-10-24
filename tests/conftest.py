"""Tests run on the CPU with tiny budgets; `tiny` shrinks a real experiment config to seconds."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import replace

import pytest

from abps.config import Config


def shrink(method: str, task: str = "inverted_pendulum", iterations: int = 2) -> Config:
    config = Config.read(method, task)
    config = replace(config, env=replace(config.env, horizon=20), search=replace(config.search, iterations=iterations))
    if config.gp:
        config = replace(config, gp=replace(config.gp, restarts=2, maxiter=20))
    if config.critic:
        config = replace(config, critic=replace(config.critic, hidden=(16,), steps=10, first_steps=5, capacity=2000))
    return config


@pytest.fixture
def key():
    import jax

    return jax.random.key(0)


@pytest.fixture
def tiny():
    return shrink
