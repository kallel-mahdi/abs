"""The paper's training figures from local runs: `python -m figures.curves <output> [<label>=<output> ...]`.

Every run directory `<output>/<method>/<task>/seed_<s>/` becomes rows of one table; curves are seed means with 68%
intervals, against the cumulative episodes the method consumed."""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from abps.runs.log import read
from abps.runs.store import runs

TASKS = {
    "inverted_pendulum": 1000,
    "swimmer": 1000,
    "hopper": 5000,
    "halfcheetah": 5000,
    "walker2d": 14000,
    "ant": 14000,
}  # task → episode budget shown


def table(output, label: str | None = None) -> pd.DataFrame:
    """All runs under `output`, one row per iteration, with the running best returns."""
    frames = []
    for method, task, seed, path in runs(output):
        frame = read(path).reset_index()
        frame["best_return"] = frame["eval/return_max"].cummax()
        frame["best_objective"] = frame["eval/objective_max"].cummax()
        frames.append(frame.assign(method=label or method.upper(), task=task, seed=seed))
    return pd.concat(frames, ignore_index=True)


def panels(data: pd.DataFrame, y: str, tasks: dict, name: str, ylabel: str) -> None:
    tasks = {task: budget for task, budget in tasks.items() if task in set(data.task)}
    figure, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.2), squeeze=False)
    for ax, (task, budget) in zip(axes[0], tasks.items(), strict=True):
        rows = data[(data.task == task) & (data.total_episodes < budget)]
        sns.lineplot(rows, x="total_episodes", y=y, hue="method", errorbar=("ci", 68), linewidth=2, ax=ax)
        ax.set(title=task, xlabel="episodes", ylabel=ylabel)
        ax.legend().set_visible(ax is axes[0, 0])
    figure.tight_layout()
    figure.savefig(f"{name}.pdf", bbox_inches="tight")


def main(arguments: list[str]) -> None:
    """The first output holds the main runs; `label=output` pairs add ablation arms (e.g. corrected ABS)."""
    main_runs = table(arguments[0])
    arms = [table(output, label) for label, output in (argument.split("=") for argument in arguments[1:])]
    panels(main_runs, "best_return", TASKS, "learning_curves", "best return")
    if arms:
        ablation = pd.concat([main_runs[main_runs.method == "ABS"], *arms])
        panels(ablation, "best_objective", TASKS, "ablation", "best discounted return")
    if "critic/r2" in main_runs:
        r2 = main_runs[main_runs.method == "ABS"].melt(
            ["task", "seed", "total_episodes"], ["critic/r2", "eval/prior_r2"], var_name="method", value_name="R²"
        )
        panels(r2.assign(**{"R²": r2["R²"].clip(-1, 1)}), "R²", TASKS, "r2", "R²")


if __name__ == "__main__":
    main(sys.argv[1:])
