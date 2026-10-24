"""Whole runs on InvertedPendulum: every method trains, a resumed run equals an uninterrupted one, and a run
directory refuses a different experiment."""

import json
from dataclasses import asdict, replace

import pytest

from abps.config import Config, corrected, presets
from abps.experiment import train as training
from abps.experiment.train import Run, train


def metrics(path) -> list[dict]:
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def test_presets_load_and_corrected_changes_only_abs():
    configs = presets()
    assert {"abs/hopper", "mpd/hopper", "ars/hopper", "abs/pendulum"} <= set(configs)
    assert all(
        (config.critic.steps, config.critic.first_steps) == (500, 1000)
        for config in configs.values()
        if config.method == "abs"
    )
    abs_, mpd = configs["abs/hopper"], configs["mpd/hopper"]
    assert corrected(mpd) == mpd
    fixed = corrected(abs_)
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
