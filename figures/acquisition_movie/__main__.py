"""Render the acquisition movie: one ABS iteration in the style of the paper's acquisition figure.

python -m figures.acquisition_movie   # plots/acquisition_movie.gif

The data comes from figures/acquisition_movie/rerun.py (the paper's notebook, seed 6101969). story.py sets what is shown
when, draw.py how a panel looks, and this file the layout around the panels.
"""

import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.lines import Line2D

from figures.acquisition_movie import story
from figures.acquisition_movie.draw import FRAME, HANDLES, INK, QUERY, load, panel


def layout():
    """The figure, its panels, and the artists that change between frames."""
    figure = plt.figure(figsize=(12, 6.27), dpi=100, facecolor="white")
    axes = [figure.add_axes([0.02 + 0.33 * k, 0.3, 0.315, 0.603]) for k in range(story.PANELS)]
    caption = figure.text(0.5, 0.075, "", fontsize=20, color=INK, ha="center", va="center")
    traveller = figure.add_artist(Line2D([], [], marker="X", markersize=10, color=QUERY, linestyle="",
                                         transform=figure.transFigure, zorder=10))
    legend = figure.legend([HANDLES[kind] for kind, _, _ in story.LEGEND], [label for _, label, _ in story.LEGEND],
                           loc="center", bbox_to_anchor=(0.5, 0.958), ncol=len(story.LEGEND), frameon=False,
                           fontsize=14, handlelength=1.5, columnspacing=1.2)
    # A key for the map, which every panel normalizes to its own range.
    key = figure.add_axes([0.66, 0.18, 0.15, 0.02])
    key.imshow(np.linspace(0, 1, 256)[None], cmap="jet", aspect="auto")
    key.set(xticks=[], yticks=[])
    for spine in key.spines.values():
        spine.set_color(FRAME)
    figure.text(0.605, 0.19, "map: how informative a rollout there is about the ascent direction", fontsize=15,
                color=INK, ha="right", va="center")
    figure.text(0.655, 0.19, "low", fontsize=14, color=INK, ha="right", va="center")
    figure.text(0.815, 0.19, "high", fontsize=14, color=INK, ha="left", va="center")
    return figure, axes, caption, traveller, legend


def draw_frame(frame, stages, figure, axes, caption, traveller, legend) -> None:
    for k, ax in enumerate(axes):
        ax.clear()
        state = story.panel_state(k, frame)
        ax.set_axis_off()
        if state is None:
            continue
        ax.set_axis_on()
        panel(ax, stages[k], stages[k + 1] if k == story.PANELS - 1 else None, state, prior_only=k == 0)
        ax.text(0.5, -0.03, f"$t = {k}$", transform=ax.transAxes, ha="center", va="top", fontsize=17, color=INK)

    traveller.set_data([], [])
    if (move := story.travel(frame)) is not None:
        k, t = move
        query = stages[k].query
        at = lambda ax: figure.transFigure.inverted().transform(ax.transData.transform(query))
        traveller.set_data(*((1 - t) * at(axes[k]) + t * at(axes[k + 1]))[:, None])

    caption.set_text(story.caption(frame))
    for handle, label, (_, _, start) in zip(legend.legend_handles, legend.get_texts(), story.LEGEND):
        handle.set_visible(frame >= start)
        label.set_visible(frame >= start)


def main(data: Path = Path("plots/acquisition_rerun_6101969.npz"),
         output: Path = Path("plots/acquisition_movie.gif")) -> None:
    stages = load(data)
    with plt.rc_context({"font.family": "cmss10", "mathtext.fontset": "cm", "axes.unicode_minus": False}):
        artists = layout()
        with tempfile.TemporaryDirectory() as tmp:
            for frame in range(story.FRAMES):
                draw_frame(frame, stages, *artists)
                artists[0].savefig(f"{tmp}/{frame:04d}.png", facecolor="white")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(story.FPS), "-i", f"{tmp}/%04d.png",
                 "-filter_complex",
                 "[0:v]split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                 "-loop", "0", str(output)],
                check=True,
            )
        plt.close(artists[0])
    print(output, f"{story.FRAMES} frames, {story.FRAMES / story.FPS:.1f} s, {output.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    tyro.cli(main)
