"""The movie's thumbnail: panel t = 1 drawn complete, without axes, legend or text, 1200x750 like the site's other cards.

python -m figures.acquisition_movie.card   # plots/acquisition_card.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tyro

from figures.acquisition_movie.draw import load, panel
from figures.acquisition_movie.story import PanelState


def main(data: Path = Path("plots/acquisition_rerun_6101969.npz"), output: Path = Path("plots/acquisition_card.png"),
         stage: int = 1) -> None:
    stages = load(data)
    figure = plt.figure(figsize=(12, 7.5), dpi=100)
    ax = figure.add_axes([0, 0, 1, 1])
    complete = PanelState(heat=1, grow=1, query=True, ring=0, observed=False, update=0, step=0)
    panel(ax, stages[stage], None, complete, prior_only=stage == 0, markers=6)
    # θ, the rollouts and the brightest lobes fit a 16:10 window on the full width.
    ax.set(ylim=(-0.0175, 0.045))
    ax.set_axis_off()
    figure.savefig(output)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    tyro.cli(main)
