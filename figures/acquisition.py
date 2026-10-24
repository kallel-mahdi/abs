"""The paper's acquisition figure: on the two-parameter Pendulum policy, the information landscape around the center
with the window, the chosen query, and the prior and posterior gradients.

    python -m figures.acquisition <run directory of abs/pendulum>"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from abps.config import Config
from abps.core import acquisition, gp
from abps.env.rollout import policy
from abps.methods import METHODS
from abps.runs import store


def landscape(path: Path, points: int = 200) -> None:
    record = store.record(path)
    config = Config.load(record["config"])
    searcher = METHODS[config.method](config, policy(config.env), jax.random.key(record["seed"]))
    with store.checkpointer(path, config, record["seed"], every=1) as checkpointer:
        searcher, iteration = store.restore(checkpointer, searcher)

    theta, prior_gradient = searcher.theta, searcher.prior_gradient(searcher.theta)
    radius = config.gp.query_bound  # the box the acquisition searches
    local = acquisition.Local.at(searcher.posterior(), theta, prior_gradient)
    grid = np.linspace(-radius, radius, points)
    offsets = jnp.stack(np.meshgrid(grid, grid, indexing="xy"), -1).reshape(-1, 2)
    info = jax.jit(jax.vmap(acquisition.info, (0, None)))(offsets, local).reshape(points, points)
    query, _ = acquisition.acquire(local, jax.random.key(0), config.search.sigma, config.gp)
    posterior_gradient, _ = gp.gradient(searcher.posterior(), theta, prior_gradient)

    extent = [theta[0] - radius, theta[0] + radius, theta[1] - radius, theta[1] + radius]
    plt.imshow(info, extent=extent, origin="lower", cmap="jet")
    plt.scatter(*searcher.window.theta.T, marker="X", color="white")
    plt.scatter(*theta, marker="*", s=200, color="darkorange", label=r"$\theta$")
    plt.scatter(*query[0], marker="X", color="darkturquoise", label="$z$")
    scale = radius / 2 / max(jnp.linalg.norm(prior_gradient), jnp.linalg.norm(posterior_gradient))
    for vector, color, label in [(prior_gradient, "violet", r"$\nabla m$"), (posterior_gradient, "purple", r"$\mu'$")]:
        plt.arrow(*theta, *(scale * vector), color=color, width=radius / 200, label=label)
    plt.legend()
    plt.savefig(f"acquisition_{iteration}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    landscape(Path(sys.argv[1]))
