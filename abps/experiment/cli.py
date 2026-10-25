"""abps-train <method>/<task> [--seed N] [--corrected] [--experiment.<section>.<field> V] ...

Experiments are the config files `configs/<method>/<task>.yaml`, with reduced critic-training budgets; `--corrected`
enforces the paper's settings (pointwise prior fitting, softmax R² weighting, worst-critic resets) and wins over explicit
overrides of those three fields."""

from dataclasses import dataclass, field
from typing import Annotated

import tyro

from abps.config import corrected, presets
from abps.experiment.train import Run, train


@dataclass(frozen=True, kw_only=True)
class Command:
    experiment: tyro.extras.subcommand_type_from_defaults(presets(), prefix_names=False)
    run: Annotated[Run, tyro.conf.OmitArgPrefixes] = field(default_factory=Run)
    corrected: bool = False


def main(args: list[str] | None = None) -> None:
    command = tyro.cli(Command, args=args, prog="abps-train", config=(tyro.conf.CascadeSubcommandArgs,))
    train(corrected(command.experiment) if command.corrected else command.experiment, command.run)
