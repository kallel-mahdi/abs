# Augmented Bayesian Policy Search

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Code for [Augmented Bayesian Policy Search](https://openreview.net/forum?id=OvlcyABNQT) (ICLR 2024), rewritten in JAX
0.11, Flax NNX, Optax and Orbax.

ABS searches linear policies by local Bayesian optimization. A Gaussian process models the return around the current
policy θ. Each iteration evaluates the queries that most sharpen the GP's belief about the gradient at θ, then moves
θ along the posterior gradient. ABS's prior mean comes from the performance-difference lemma: an ensemble of critics of
the current policy, trained by TD on every transition seen, predicts how the return changes away from θ. The baselines
are MPD (the same local BO with a constant prior mean) and ARS.

## Install

```bash
uv sync                      # CUDA 12 JAX by default; CPU only: uv sync --no-group cuda
uv sync --group figures      # pandas, matplotlib, seaborn for the figures
```

JAX reuses compiled executables across processes in `$XDG_CACHE_HOME/abps/jax` (normally `~/.cache/abps/jax`),
including short compilations used by tests. Override the location with `JAX_COMPILATION_CACHE_DIR`; the standard
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` and `JAX_ENABLE_COMPILATION_CACHE` settings are also respected.

## Run

```bash
abps-train abs/hopper --seed 0                 # 500 critic updates per query
abps-train abs/hopper --seed 0 --corrected     # with the released-code deviations fixed (DEVIATIONS.md)
abps-train mpd/hopper --experiment.gp.n-max 60  # any config field can be overridden
abps-train --help
```

The experiments are `configs/<method>/<task>.yaml` for `abs`, `mpd` and `ars` on `inverted_pendulum`, `swimmer`,
`hopper`, `halfcheetah`, `walker2d` and `ant`, plus `abs/pendulum` for the acquisition figure.

ABS defaults to 500 critic updates after each query, 1,000 after a new center, and 1,000 for the first fit. To restore
the paper's critic budget, pass `--experiment.critic.steps 5000` (20,000 for `abs/pendulum`); the new-center budget
remains twice the query budget. `--corrected` leaves this training budget unchanged.

A run writes `runs/<method>/<task>/seed_<s>/`, which holds:
- `checkpoints/`: Orbax checkpoints, whose root metadata is the run's config. A run resumes from its latest checkpoint, saves on SIGTERM, and refuses a directory that belongs to a different experiment.
- `metrics.jsonl`: one line per iteration (`eval/`, `gp/`, `critic/`, `acquisition/`, `move/`, `time/`), mirrored to W&B with `--wandb online`.

## Figures

```bash
python -m figures.curves runs corrected=runs_corrected   # learning curves, ablation, R² (with --prior-test-episodes 5)
python -m figures.acquisition runs/abs/pendulum/seed_0   # the acquisition landscape
```

## Layout

```
abps/
  config.py        Env · Search · GP · Critic · ARS sections, read from configs/
  core/            the equations, pure JAX
    policy.py        linear policies and the observation normalizer
    gp.py            kernel, likelihood, hyperparameter fit, gradient posterior
    acquisition.py   the information objective and its maximization
    optimize.py      multistart box-constrained L-BFGS
    critic.py        Q-network ensemble and TD training
    prior.py         the advantage prior and its validation R²
    replay.py        the replay buffer
  rollout.py       Gymnasium episodes as fixed-shape arrays
  methods/         base (the contract), local_bo, abs, mpd, ars
  experiment/      train (the loop), evaluate (episodes, measurements)
  runs/            store (directories, Orbax), log (metrics, W&B)
  cli.py
figures/           curves.py, acquisition.py
tests/
```

`uv run pytest` checks the equations against dense references and the released formulas, and checks that a resumed
run equals an uninterrupted one.

## License

MIT.
