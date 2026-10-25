"""Whole runs on InvertedPendulum: every method trains, a resumed run equals an uninterrupted one, and a run
directory refuses a different experiment."""

import json
from dataclasses import asdict, replace

import pytest

from abps.config import Config, corrected, presets
from abps.experiment import train as training
from abps.experiment.train import Run, train
from abps.runs.log import Log


def metrics(path) -> list[dict]:
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def test_maxima_are_running_bests_and_resume_from_the_retained_log(tmp_path, tiny):
    config = tiny("ars")

    def values(value):
        return {
            "eval/return": 3.0,
            "eval/return_max": value,
            "eval/objective_max": value / 10,
            "episodes": 1,
            "time/iteration": 0.1,
        }

    with Log(tmp_path, config, "disabled", start=0) as log:
        for i, value in enumerate((10.0, 5.0, 20.0), 1):
            log(i, values(value))
    assert [row["eval/return_max"] for row in metrics(tmp_path)] == [10.0, 10.0, 20.0]
    with Log(tmp_path, config, "disabled", start=2) as log:
        log(3, values(7.0))
    rows = metrics(tmp_path)
    assert [row["eval/return_max"] for row in rows] == [10.0, 10.0, 10.0]
    assert [row["eval/objective_max"] for row in rows] == [1.0, 1.0, 1.0]
    assert [row["eval/return_iteration_max"] for row in rows] == [10.0, 5.0, 7.0]
    assert [row["eval/return"] for row in rows] == [3.0, 3.0, 3.0]


def test_abs_presets_use_pointwise_prior_r2_weighting_and_resets():
    configs = presets()
    assert {"abs/hopper", "mpd/hopper", "ars/hopper", "abs/pendulum"} <= set(configs)
    assert all(
        (config.critic.steps, config.critic.first_steps) == (500, 1000)
        for config in configs.values()
        if config.method == "abs"
    )
    assert all(
        (config.gp.fit_mean, config.critic.reset_worst, config.critic.aggregation) == ("pointwise", True, "softmax")
        for config in configs.values()
        if config.method == "abs"
    )
    abs_, mpd = configs["abs/hopper"], configs["mpd/hopper"]
    assert corrected(mpd) == mpd
    assert corrected(abs_) == abs_
    released = replace(
        abs_,
        gp=replace(abs_.gp, fit_mean="average"),
        critic=replace(abs_.critic, reset_worst=False, aggregation="mean"),
    )
    fixed = corrected(released)
    assert (fixed.gp.fit_mean, fixed.critic.reset_worst, fixed.critic.aggregation) == ("pointwise", True, "softmax")
    assert Config.load(json.loads(json.dumps(asdict(abs_)))) == abs_


@pytest.mark.parametrize("method", ["abs", "mpd", "ars"])
def test_every_method_runs(method, tmp_path, tiny):
    path = train(tiny(method), Run(output=tmp_path, eval_episodes=2, save_every=1))
    rows = metrics(path)
    assert [row["iteration"] for row in rows] == [1, 2]
    assert all(row["episodes"] > 0 for row in rows)


def test_resumed_run_equals_uninterrupted_run(tmp_path, monkeypatch, tiny):
    config = tiny("abs", iterations=3)
    whole = metrics(train(config, Run(output=tmp_path / "whole", eval_episodes=2)))

    log = training.Log

    class Interrupted(log):  # stops the run after its second iteration, as SIGTERM would
        def __call__(self, iteration, values):
            super().__call__(iteration, values)
            if iteration == 2:
                training.terminated.set()

    monkeypatch.setattr(training, "Log", Interrupted)
    train(config, Run(output=tmp_path / "split", eval_episodes=2))
    training.terminated.clear()
    monkeypatch.setattr(training, "Log", log)
    split = metrics(train(config, Run(output=tmp_path / "split", eval_episodes=2)))

    def science(row):
        return {name: value for name, value in row.items() if not name.startswith("time/")}

    assert [science(row) for row in split] == [science(row) for row in whole]


def test_run_directory_refuses_another_experiment(tmp_path, tiny):
    config = tiny("ars", iterations=1)
    train(config, Run(output=tmp_path, eval_episodes=1))
    with pytest.raises(ValueError, match="different experiment"):
        train(replace(config, search=replace(config.search, learning_rate=1.0)), Run(output=tmp_path, eval_episodes=1))


def test_nonfinite_metrics_stop_before_replacing_a_valid_checkpoint(tmp_path, tiny, monkeypatch):
    original = training.iteration

    def invalid(searcher, envs, run, i, key):
        values = original(searcher, envs, run, i, key)
        return values | {"eval/return": float("nan")} if i == 1 else values

    monkeypatch.setattr(training, "iteration", invalid)
    config, run = tiny("ars"), Run(output=tmp_path, eval_episodes=1, save_every=1)
    with pytest.raises(ValueError, match="Out of range float"):
        train(config, run)
    path = training.store.directory(tmp_path, config, run.seed)
    assert [row["iteration"] for row in metrics(path)] == [1]
    with training.store.checkpointer(path, config, run.seed, every=1) as checkpointer:
        assert checkpointer.latest.step == 1
