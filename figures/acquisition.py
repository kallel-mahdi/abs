"""Render the notebook's acquisition sequence from saved measurements.

python -m abps.experiment.acquisition
python -m figures.acquisition plots/acquisition_loop.npz
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def panel(ax, data, stage: int, arrow_scale: float, *, show_query: bool = True) -> None:
    theta, grid = data["theta"], data["grid"]
    extent = [theta[0] + grid[0], theta[0] + grid[-1], theta[1] + grid[0], theta[1] + grid[-1]]
    ax.imshow(data[f"gain_{stage}"], extent=extent, origin="lower", cmap="jet", interpolation="bilinear")
    observed = data[f"observed_{stage}"][1:]
    ax.scatter(*observed.T, marker="X", color="white", s=32, zorder=4)
    if show_query:
        ax.scatter(*data[f"query_{stage}"], marker="X", color="darkturquoise", s=36, zorder=5)
    ax.scatter(*theta, marker="*", s=150, color="darkorange", zorder=6)
    for name, color in (("prior_gradient", "violet"), ("posterior_gradient", "purple")):
        end = theta + arrow_scale * data[f"{name}_{stage}"]
        ax.annotate(
            "",
            xy=end,
            xytext=theta,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8, "mutation_scale": 13},
            zorder=7,
        )
    ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
    ax.set_xticks([-0.04, 0, 0.04])
    ax.set_yticks([-0.04, 0, 0.04])
    ax.tick_params(labelsize=10, length=3)


def landscape(path: Path) -> None:
    with np.load(path) as receipt:
        data = dict(receipt)
    stages = min(3, int(data["stages"]))
    # Preserve directions and relative lengths within a panel; rescale between panels for legibility.
    arrow_scales = [
        0.018
        / max(1e-12, *(np.linalg.norm(data[f"{name}_{stage}"]) for name in ("prior_gradient", "posterior_gradient")))
        for stage in range(int(data["stages"]))
    ]
    legend = [
        Line2D([], [], marker="*", color="darkorange", linestyle="", markersize=11, label=r"$\theta$"),
        Line2D([], [], marker="X", color="white", linestyle="", markersize=6, label=r"$z$"),
        Patch(color="violet", label=r"$\nabla_\theta m$"),
        Patch(color="purple", label=r"$\mu_\theta$"),
    ]

    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, stages, figsize=(3.2 * stages, 3.4), layout="constrained")
        for stage, ax in enumerate(np.atleast_1d(axes)):
            panel(ax, data, stage, arrow_scales[stage])
            ax.text(
                0.5,
                -0.16,
                f"({chr(97 + stage)}) $t={stage}$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=12,
            )
        np.atleast_1d(axes)[1].legend(
            handles=legend,
            loc="upper center",
            ncol=4,
            fontsize=9,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.8,
            facecolor="white",
            framealpha=0.85,
            borderpad=0.35,
        )
        for suffix in ("png", "pdf"):
            figure.savefig(path.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
        plt.close(figure)

        for stage in range(int(data["stages"])):
            figure, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
            panel(ax, data, stage, arrow_scales[stage])
            figure.savefig(path.with_name(f"{path.stem}_{stage}.png"), dpi=180, bbox_inches="tight")
            plt.close(figure)
    print(path.with_suffix(".pdf"))


if __name__ == "__main__":
    landscape(Path(sys.argv[1]))
