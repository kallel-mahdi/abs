# Augmented Bayesian Policy Search

JAX implementation of Augmented Bayesian Search (ABS), from our ICLR 2024 paper
[Augmented Bayesian Policy Search](https://openreview.net/forum?id=OvlcyABNQT) ([arXiv](https://arxiv.org/abs/2407.04864)).
If you use our code, please cite the paper.

ABS searches deterministic policies by local Bayesian optimization. A Gaussian process models the return around the
current policy, and its prior mean comes from the performance difference lemma: critics of the current policy predict
how the return changes away from it. The gradient posterior then combines the deterministic policy gradient with the
returns actually observed.

![One ABS iteration on Pendulum: each map shows how informative a rollout would be about the ascent direction, ABS runs the most informative one, the posterior over the gradient moves away from the critic's policy gradient, and the policy steps along the most probable ascent direction.](.github/assets/abs-acquisition-paper.gif)

## Installation

```bash
git clone https://github.com/kallel-mahdi/abs.git
cd abs
uv sync                      # CUDA 12 JAX; CPU only: uv sync --no-group cuda
```

## Usage

```bash
abps-train abs/hopper --seed 0    # ABS
abps-train mpd/hopper --seed 0    # MPD: the same local Bayesian optimization with a constant prior mean
abps-train ars/hopper --seed 0    # Augmented Random Search
abps-train --help
```

The experiments are in `configs/<method>/<task>.yaml`, for `inverted_pendulum`, `swimmer`, `hopper`, `halfcheetah`,
`walker2d` and `ant`. Any config field can be overridden on the command line; for example,
`--experiment.critic.steps 5000` restores the paper's critic budget (the default is 500 updates per query). A run
writes checkpoints and `metrics.jsonl` to `runs/<method>/<task>/seed_<s>/` and resumes from its latest checkpoint.

`python -m figures.curves runs` draws the learning curves, the ablation and the critic R² plots.
`python -m figures.acquisition_movie` renders the GIF above from a rerun of the paper's acquisition notebook (see
`figures/acquisition_movie/rerun.py`); `story.py` there sets what it shows and when.

## Citation

```bibtex
@inproceedings{kallel2024abs,
  author    = {Mahdi Kallel and Debabrota Basu and Riad Akrour and Carlo D'Eramo},
  title     = {Augmented Bayesian Policy Search},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=OvlcyABNQT},
}
```

## License

MIT
