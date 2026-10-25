"""Animate one ABS iteration on a synthetic 2-D return landscape: a critic-based prior, the gradient posterior given
earlier returns and two queries chosen by the acquisition, and the step along the most probable ascent direction.

python -m figures.acquisition_gif   # plots/acquisition_loop.gif, for the README and the site preview

The landscape and the critic's 50° error are made up; the GP posterior, the acquisition and the step are ABS's own
(abps.core.gp, abps.core.acquisition). The cone holds the central 68% of gradient directions under the posterior.
Flat colors and the palette of the RIC README animation keep the GIF small and free of banding.
"""

import subprocess
import tempfile
from dataclasses import dataclass, fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

from abps.core import acquisition, gp

jax.config.update("jax_enable_x64", True)


def rgb(*c):
    return tuple(v / 255 for v in c)


FPS = 10
INK, BACKGROUND, FRAME, CONTOUR, MUTED = (rgb(35, 42, 49), rgb(252, 251, 252), rgb(208, 213, 221), rgb(214, 219, 226),
                                     rgb(150, 158, 170))
CRITIC, GP = rgb(227, 119, 194), rgb(110, 60, 150)  # paper: pink ∇m, violet μθ
PRIOR_CONE, POSTERIOR_CONE = np.array(rgb(250, 226, 241)), np.array(rgb(231, 222, 244))
BANDS = [rgb(232, 242, 249), rgb(194, 221, 237), rgb(112, 172, 214), rgb(0, 118, 185)]
LEVELS = [0.25, 0.45, 0.65, 0.85, 1.01]
HALF = 0.22  # the view is θ ± HALF
ARROW, CONE = 0.075, 0.095  # display lengths of the gradient arrows (directions only) and of the cone
STEP = 0.075  # the step size δ, chosen for display


# The synthetic problem: θ at the origin, two bumps of return, and a critic whose gradient is 50° off.
def objective(x):
    bump = lambda center, width: np.exp(-np.sum((x - np.asarray(center)) ** 2, -1) / (2 * width**2))
    return bump((0.45, 1.25), 0.9) + 0.35 * bump((-0.9, 0.6), 0.5)


def rotate(v, degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s], [s, c]]) @ v


def simulate(queries: int = 2, points: int = 111) -> dict:
    """The prior (θ only), then the posterior given two earlier returns and after each acquired query."""
    theta = np.zeros(2)
    true = np.array([objective(theta + 1e-5 * e) - objective(theta - 1e-5 * e) for e in np.eye(2)]) / 2e-5
    critic = rotate(true, -50.0)  # the advantage mean function's slope at θ
    prior = lambda x: objective(theta) + (x - theta) @ critic
    params = gp.Params(jnp.full(2, 0.4), jnp.asarray(0.4), jnp.asarray(0.003), jnp.asarray(0.0))
    grid = np.linspace(-HALF, HALF, points)
    offsets = np.stack(np.meshgrid(grid, grid), -1).reshape(-1, 2)
    gain = jax.jit(jax.vmap(acquisition.gain, (0, None)))
    known = np.array([[0.0, 0.15], [0.0, -0.15]])

    def belief(observed):
        posterior = gp.condition(params, *map(jnp.asarray, (observed, objective(observed), prior(observed))))
        mean, cov = map(np.asarray, gp.gradient(posterior, jnp.asarray(theta), jnp.asarray(critic)))
        return posterior, mean, cov

    _, mean, cov = belief(theta[None])
    observed, stages = np.vstack([theta, known]), [{"mean": mean, "cov": cov}]
    for _ in range(queries + 1):
        posterior, mean, cov = belief(observed)
        local = acquisition.Local.at(posterior, jnp.asarray(theta), jnp.asarray(critic))
        field = np.asarray(gain(jnp.asarray(offsets), local))
        stages.append({"mean": mean, "cov": cov, "gain": field.reshape(points, points) / field.max(),
                       "query": theta + offsets[field.argmax()]})
        observed = np.vstack([observed, stages[-1]["query"]])
    ascent = np.linalg.solve(stages[-1]["cov"], stages[-1]["mean"])
    return {"theta": theta, "critic": critic, "true": true, "known": known, "stages": stages, "grid": grid,
            "move": theta + STEP * ascent / np.linalg.norm(ascent)}


def angle(v) -> float:
    return float(np.arctan2(v[1], v[0]))


def turn(a: float, b: float) -> float:
    """Signed shortest rotation from angle a to angle b."""
    return (b - a + np.pi) % (2 * np.pi) - np.pi


def cone(mean, cov) -> np.ndarray:
    """Central 68% of gradient directions under N(mean, cov), as angles relative to the mean direction."""
    samples = np.random.default_rng(0).multivariate_normal(mean, cov, 4000)
    relative = [turn(angle(mean), angle(s)) for s in samples]
    return np.percentile(relative, [16, 84])


@dataclass(frozen=True)
class State:
    heat: np.ndarray  # acquisition field, normalized to its maximum
    gp: float  # posterior mean angle
    cone: np.ndarray  # cone edges relative to the posterior mean
    critic_len: float = 0.0
    cone_shown: float = 0.0
    tint: float = 0.0  # 0: prior cone, 1: posterior cone
    gp_shown: float = 0.0
    observed: int = 0
    query: int | None = None
    ring: float = 0.0  # radius of the ring closing in on the next query
    step: float = 0.0  # progress of θ towards the new center
    caption: str = ""


def tween(a: State, b: State, t: float) -> State:
    t = t * t * (3 - 2 * t)
    values = {}
    for f in fields(State):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "gp":
            values[f.name] = x + t * turn(x, y)
        elif f.name in ("heat", "cone") or isinstance(y, float):
            values[f.name] = (1 - t) * x + t * y
        else:
            values[f.name] = y
    return State(**values)


def timeline(run) -> list[State]:
    stages = run["stages"]
    blank = np.zeros_like(stages[1]["gain"])
    frames, state = [], State(blank, angle(stages[0]["mean"]), cone(stages[0]["mean"], stages[0]["cov"]))

    def to(duration: float, hold: float = 0.0, **change) -> None:
        nonlocal state
        target = replace(state, **change)
        steps = round(duration * FPS)
        frames.extend(tween(state, target, (i + 1) / steps) for i in range(steps))
        frames.extend([target] * round(hold * FPS))
        state = target

    def believe(k: int, caption: str, **change) -> None:
        stage = stages[k]
        to(1.2, 2.4, gp=angle(stage["mean"]), cone=cone(stage["mean"], stage["cov"]), tint=1.0, gp_shown=1.0,
           caption=caption, **change)

    def query(k: int, ask: str, learn: str) -> None:
        to(0.0, query=k, ring=0.03, caption=ask)
        to(0.6, 0.5, ring=0.0)
        to(0.0, 0.6, observed=k, query=None)
        believe(k + 1, learn, heat=stages[k + 1]["gain"])

    to(0.0, 2.0, caption="Policy $\\theta$ and the returns observed so far.")
    to(0.8, 2.6, critic_len=1.0, cone_shown=1.0,
       caption="A prior from the performance difference lemma:\nthe critic's policy gradient and its uncertainty.")
    believe(1, "The gradient posterior uses those returns:\nit turns away from the critic.")
    to(0.8, 2.0, heat=stages[1]["gain"], caption="Acquisition: the rollout that most raises\nthe probability of ascent.")
    query(1, "Query its maximum and observe the return.", "The posterior turns further\nand its cone narrows.")
    query(2, "Query the new maximum.", "The posterior sharpens.")
    to(0.8, 0.4, heat=blank)
    to(1.6, 2.6, step=1.0, caption="Step along the most probable\nascent direction.")
    return frames


def arrow(ax, start, direction, length, color, width):
    tip = start + length * np.array([np.cos(direction), np.sin(direction)])
    ax.add_patch(FancyArrowPatch(start, tip, arrowstyle="-|>", mutation_scale=5 * width + 4, lw=width, color=color,
                                 shrinkA=0, shrinkB=0, zorder=6))


def draw(ax, s: State, run, contours) -> None:
    theta, critic = run["theta"], angle(run["critic"])
    ax.clear()
    ax.set(xlim=(-HALF, HALF), ylim=(-HALF, HALF), aspect="equal", xticks=[], yticks=[], facecolor=BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(FRAME)
    ax.contour(*contours, levels=12, colors=[CONTOUR], linewidths=1.0, zorder=1)
    # Which way the contours rise: the true gradient at θ, known only because the landscape is synthetic.
    cue = np.array([-0.19, 0.15])
    arrow(ax, cue, angle(run["true"]), 0.045, MUTED, 1.4)
    ax.text(*(cue + [0.012, 0.022]), "higher return", color=MUTED, fontsize=13, ha="left", va="center", zorder=6)
    if s.heat.max() > LEVELS[0]:
        ax.contourf(run["grid"], run["grid"], s.heat, levels=LEVELS, colors=BANDS, antialiased=False, zorder=2)

    if s.cone_shown > 0.05:
        lo, hi = np.degrees(s.gp + s.cone)
        face = (1 - s.tint) * PRIOR_CONE + s.tint * POSTERIOR_CONE
        edge = (1 - s.tint) * np.array(CRITIC) + s.tint * np.array(GP)
        ax.add_patch(Wedge(theta, s.cone_shown * CONE, lo, hi, facecolor=face, edgecolor=edge, lw=1.0, zorder=4))
    if s.critic_len > 0.35:
        arrow(ax, theta, critic, s.critic_len * ARROW, CRITIC, 4.5)
    if s.gp_shown > 0.35:
        arrow(ax, theta, s.gp, s.gp_shown * ARROW, GP, 1.8)

    halo = [pe.withStroke(linewidth=3, foreground=BACKGROUND)]
    # The critic's label past its arrow; the posterior's at the cone edge away from the critic, clear of the step.
    edge = s.gp + (s.cone[1] if turn(s.gp, critic) < 0 else s.cone[0])
    for direction, label, color, shown in (
        (critic, "policy gradient", CRITIC, s.critic_len == 1.0),
        (edge, "posterior", GP, s.gp_shown == 1.0),
    ):
        if shown:
            along = np.array([np.cos(direction), np.sin(direction)])
            ha = "left" if along[0] > 0.2 else "right" if along[0] < -0.2 else "center"
            ax.text(*(theta + 1.08 * CONE * along), label, color=color, fontsize=15, ha=ha, va="center", zorder=8,
                    path_effects=halo)

    queries = [stage["query"] for stage in run["stages"][1:]]
    for point in run["known"]:
        ax.scatter(*point, s=45, color=INK, zorder=7)
    for point in queries[: s.observed]:
        ax.scatter(*point, s=45, color=INK, zorder=7)
    if s.query is not None:
        ax.scatter(*queries[s.query - 1], s=70, color="white", edgecolor=INK, lw=1.8, zorder=7)
        if s.ring > 0.002:
            ax.add_patch(Circle(queries[s.query - 1], s.ring, fill=False, color=INK, lw=1.6, zorder=7))

    center = theta + s.step * (run["move"] - theta)
    if s.step:
        ax.plot(*np.stack([theta, center]).T, color=INK, lw=1.6, ls=(0, (3, 2)), zorder=7)
        ax.scatter(*theta, marker="*", s=320, color=BACKGROUND, edgecolor=INK, lw=1.2, zorder=9)
    ax.scatter(*center, marker="*", s=320, color=INK, zorder=9)


def animate(output: Path = Path("plots/acquisition_loop.gif")) -> None:
    run = simulate()
    for k, stage in enumerate(run["stages"]):
        edges = np.degrees(cone(stage["mean"], stage["cov"])).round()
        print(f"stage {k}: posterior {np.degrees(angle(stage['mean'])):.0f}°, cone {edges}, query {stage.get('query')}")
    print(f"true {np.degrees(angle(run['true'])):.0f}°, critic {np.degrees(angle(run['critic'])):.0f}°, "
          f"step {np.degrees(angle(run['move'] - run['theta'])):.0f}°")
    frames = timeline(run)
    x = np.linspace(-HALF, HALF, 200)
    xx, yy = np.meshgrid(x, x)
    contours = (xx, yy, objective(np.stack([xx, yy], -1)))
    with plt.rc_context({"font.family": "cmss10", "mathtext.fontset": "cm", "axes.unicode_minus": False}):
        figure = plt.figure(figsize=(6, 6.8), dpi=100, facecolor=BACKGROUND)
        ax = figure.add_axes([0.01, 0.125, 0.98, 0.865])
        caption = figure.text(0.5, 0.062, "", fontsize=18, color=INK, ha="center", va="center", linespacing=1.4)
        with tempfile.TemporaryDirectory() as tmp:
            for i, state in enumerate(frames):
                draw(ax, state, run, contours)
                caption.set_text(state.caption)
                figure.savefig(f"{tmp}/{i:04d}.png", facecolor=BACKGROUND)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png", "-filter_complex",
                 "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:diff_mode=rectangle",
                 "-loop", "0", str(output)],
                check=True,
            )
        plt.close(figure)
    print(output, f"{len(frames)} frames, {len(frames) / FPS:.1f} s, {output.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    tyro.cli(animate)
