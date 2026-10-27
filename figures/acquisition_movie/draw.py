"""The data of the acquisition movie and how one panel is drawn, shared by the movie and its thumbnail."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Wedge

from figures.acquisition_gif import angle, cone, turn
from figures.acquisition_movie.story import PanelState

INK, FRAME = (35 / 255, 42 / 255, 49 / 255), (208 / 255, 213 / 255, 221 / 255)
THETA, QUERY, CRITIC, POSTERIOR = "darkorange", "darkturquoise", "violet", "purple"
ARROW, CONE = 0.012, 0.02  # display length of both gradients (only their directions matter) and radius of the cone
STEP_LENGTH = 0.012  # the step's direction is ABS's; its length (0.0037 here) is enlarged for display
STAR, CROSS = 280, 60  # marker areas, in points²

HANDLES = {
    "theta": Line2D([], [], marker="*", color=THETA, linestyle="", markersize=15),
    "query": Line2D([], [], marker="X", color=QUERY, linestyle="", markersize=10),
    "observed": Line2D([], [], marker="X", color="white", markeredgecolor="0.55", linestyle="", markersize=10),
    "critic": Line2D([], [], color=CRITIC, lw=3.5),
    "posterior": Line2D([], [], color=POSTERIOR, lw=3.5),
    "cone": Patch(facecolor=(0.85, 0.8, 0.9), edgecolor=POSTERIOR),
}


@dataclass(frozen=True)
class Stage:
    """One stage of the run, in display units: the gradients are rescaled by 0.004 as in the paper's notebook."""

    field: np.ndarray  # acquisition, normalized to [0, 1], rows along the second parameter
    extent: np.ndarray
    theta: np.ndarray
    observed: np.ndarray  # rollouts observed so far, θ excluded
    query: np.ndarray
    critic: np.ndarray  # ∇θm
    posterior: np.ndarray  # μθ
    cone: np.ndarray  # central 68% of gradient directions, in radians relative to μθ
    step: np.ndarray  # where ABS moves θ from this posterior


def load(path: Path) -> list[Stage]:
    with np.load(path) as d:
        def stage(k):
            field = d[f"f_Z_{k}"].T
            return Stage((field - field.min()) / (field.max() - field.min()), d[f"extent_{k}"], d[f"x_t_{k}"].ravel(),
                         d[f"X_{k}"][1:], d[f"x_acq_{k}"].ravel(), d[f"mean_d_{k}"], d[f"corrected_mean_d_{k}"],
                         cone(d[f"grad_mean_{k}"], d[f"grad_cov_{k}"]), d[f"step_{k}"])
        return [stage(k) for k in range(int(d["stages"]))]


def rotate_towards(u, v, t: float):
    """Rotate and rescale u towards v."""
    length = (1 - t) * np.linalg.norm(u) + t * np.linalg.norm(v)
    direction = angle(u) + t * turn(angle(u), angle(v))
    return length * np.array([np.cos(direction), np.sin(direction)])


def panel(ax, stage: Stage, after: Stage | None, state: PanelState, prior_only: bool, markers: float = 1) -> None:
    """Draw a panel. `after` is the next stage, towards which the last panel's posterior updates; with `prior_only`
    (t = 0) the posterior is still the prior and only the critic's gradient is drawn. `markers` scales marker areas."""
    star, cross = markers * STAR, markers * CROSS
    ax.imshow(state.heat * stage.field, extent=stage.extent, cmap="jet", vmin=0, vmax=1, origin="lower",
              interpolation="bicubic")
    ax.set(xlim=stage.extent[:2], ylim=stage.extent[2:], aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color(FRAME)

    critic, posterior, edges = stage.critic, stage.posterior, stage.cone
    if after is not None and state.update:
        critic = rotate_towards(critic, after.critic, state.update)
        posterior = rotate_towards(posterior, after.posterior, state.update)
        edges = (1 - state.update) * edges + state.update * after.cone
    if state.grow > 0.15:
        lo, hi = np.degrees(angle(posterior) + edges)
        ax.add_patch(Wedge(stage.theta, state.grow * CONE, lo, hi, facecolor=(1, 1, 1, 0.35),
                           edgecolor=CRITIC if prior_only else POSTERIOR, lw=1.6, zorder=6))
        for vector, color in ((critic, CRITIC), (posterior, POSTERIOR))[: 1 if prior_only else 2]:
            ax.arrow(*stage.theta, *(state.grow * ARROW * vector / np.linalg.norm(vector)), color=color, width=0.0011,
                     length_includes_head=True, zorder=7)

    observed = np.vstack([stage.observed, stage.query]) if state.observed else stage.observed
    if len(observed):
        ax.scatter(*observed.T, marker="X", color="white", s=cross, zorder=8)
    if state.query:
        ax.scatter(*stage.query, marker="X", color=QUERY, s=cross, zorder=8)
        if state.ring > 0:
            ax.add_patch(Circle(stage.query, 0.002 + 0.008 * state.ring, fill=False, color="white", lw=1.5, zorder=8))

    center = stage.theta
    if state.step and after is not None:
        direction = after.step - stage.theta
        center = stage.theta + state.step * STEP_LENGTH * direction / np.linalg.norm(direction)
        ax.plot(*np.stack([stage.theta, center]).T, color="white", lw=1.8, ls=(0, (3, 2)), zorder=9)
        ax.scatter(*stage.theta, marker="*", s=star, facecolor="none", edgecolor=THETA, lw=1.4, zorder=9)
    ax.scatter(*center, marker="*", color=THETA, s=star, zorder=10)
