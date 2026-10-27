"""Rerun the paper's generate_acq_plots.ipynb (commit 86ecfad, seed 6101969) and save what the movie needs.

Run from a checkout of that commit with its pinned requirements (Python 3.10, jax 0.4.23 with CUDA, flax 0.7.4, gymnasium
0.29) and LD_LIBRARY_PATH unset so that the cuDNN from pip is used:

    PYTHONHASHSEED=1 WANDB_MODE=disabled python rerun.py   # regen/stages.npz -> plots/acquisition_rerun_6101969.npz

Per stage it saves what the notebook plots (f_Z, extent, X, x_acq, x_t, mean_d, corrected_mean_d), the gradient posterior
at θ (grad_mean, grad_cov) and the step ABS takes from it. Stage 0 matches the paper's figure; later stages differ
slightly because the critic retraining is not bit-reproducible.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

os.makedirs("regen", exist_ok=True)
cells = ["".join(c["source"]) for c in json.load(open("generate_acq_plots.ipynb"))["cells"] if c["cell_type"] == "code"]
code = "\n".join(cells)
code = code.replace('os.environ["WANDB_API_KEY"]', 'os.environ["WANDB_UNUSED"]')
code = code.replace("plt.savefig('./plots/plot'", "DUMP.append(dict(f_Z=np.asarray(f_Z), extent=np.array([x_min,x_max,y_min,y_max]), X=np.asarray(X), x_acq=np.asarray(x_acq), x_t=np.asarray(x_t), mean_d=np.asarray(mean_d), corrected_mean_d=np.asarray(corrected_mean_d), **EXTRA(mpd)))\n    SAVE()\n    plt.savefig('./regen/plot'")
code = code.replace("plt.show()", "plt.close()")
DUMP = []


def EXTRA(mpd):
    """The gradient posterior at θ (unscaled) and the step ABS would take from it."""
    mean, cov = mpd.gp.posterior_derivative(mpd.x)
    x_new, ascent_prob = mpd.gradient_step(mpd.x, mpd.gp.X, mpd.gp.y)
    return dict(grad_mean=np.asarray(mean).ravel(), grad_cov=np.asarray(cov), step=np.asarray(x_new).ravel(),
                ascent_prob=np.asarray(ascent_prob))


def SAVE():
    np.savez("regen/stages.npz", stages=len(DUMP), **{f"{k}_{i}": v for i, d in enumerate(DUMP) for k, v in d.items()})


exec(compile(code, "generate_acq_plots", "exec"), {"__name__": "__main__", "DUMP": DUMP, "SAVE": SAVE, "EXTRA": EXTRA})
print("stages", len(DUMP))
