# Deviations of the released code (tag `baseline-iclr2024`)

The ICLR 2024 runs used the released code. This rewrite keeps the three deviations that change ABS's behaviour, as the
configs' defaults (`configs/abs/*.yaml`); `abps-train ... --corrected` replaces them by what the paper describes.
The critic-training budget is reduced by default, as documented below. The unconditional fixes change measurements
or edge cases.

## Kept in the configs, switched by `--corrected`

| | Released behaviour | Config field: paper → corrected |
|---|---|---|
| B1 | The GP likelihood subtracted an `(n,)` prior mean from `(n, 1)` returns. The `(n, n)` broadcast makes the objective `n²` times the likelihood with the **average** prior mean, and the output-scale bound used the spread of all `y_i − m_j`. ABS therefore fitted its hyperparameters as if the prior were a constant. MPD's constant mean is unaffected. The posterior gradient was always pointwise. Checked numerically in `tests/test_gp.py`. | `gp.fit_mean: average → pointwise` |
| B2 | `critic_mask.at[argmin].set(1)` discarded its result, so no critic was ever reset. At each new center the Adam states of all critics were reset, which both settings keep. Because `--reset_critic` used `type=bool`, it was on in every run, so the paper's "+R" ablation arms probably ran the same code as the plain arms (inferred; check against the W&B run configs). | `critic.reset_worst: false → true` |
| B4 | The CLI default `--aggregation mean` overrode the YAML's `softmax`, so runs averaged the critics unless "softmax" was passed explicitly. | `critic.aggregation: mean → softmax` |

## Fixed unconditionally

- **B3:** `--reset_critic False` evaluated to True (argparse `type=bool`). There is no such flag any more.
- **B5:** `--seed` was ignored, and seeds came from `np.random.seed(40)`. Runs are now seeded by `--seed`; the env seeds are `(seed, iteration, call)`.
- **B6:** the critics' validation ran twice in a row.
- **B7:** the target networks were indexed twice. This was harmless with two networks, so the REDQ subsampling is gone.
- **B8:** float64 and matmul precision were flipped globally around every critic fit. Now everything runs in JAX's native float32
  (TF32 matmuls on the GPU). On a GP from a Hopper run, float32 and float64 agree:
  - step direction cosine 0.99999998; ascent probability 0.93917 against 0.93918;
  - the same acquired query, and the same fitted likelihood to 1e-5;
  - the same MPD walk.

  The GP fit and the acquisition run 2–3.5× faster.
- **B9:** the reward of the terminating step was dropped from returns, and its state from the prior's weights and from
  the observation normalizer.
- **B10:** `n_transitions` summed discounted weights. The log now counts episodes per iteration as `episodes`.
- **B12:** a truncated episode (Pendulum-v1 at step 200) kept counting its frozen last reward until step 1000. Episodes now end at termination or truncation.
- **Observation noise** was added through a pairwise `allclose` on every kernel call. It is now σ_n² I on the observations only.

## Other changes worth knowing when comparing curves

- **Critic budget.** Every ABS preset uses 500 TD updates after each query and 1,000 after a new center; the first fit
  remains 1,000 updates. Restore the paper's query budget with `--experiment.critic.steps 5000`, or 20,000 for
  `abs/pendulum`; new centers still train for twice the query budget. `--corrected` does not change these budgets.
- **Environments** are Gymnasium **v5** with MuJoCo 3, and the configs restore the v4 tasks where v5 changed them.
  - Ant: `include_cfrc_ext_in_observation: false` (the 27-dimensional observation) and `contact_cost_weight: 0` (v4 had
    no contact cost).
  - Walker2d: `xml_file: walker2d.xml` (v4's left-foot friction of 0.9; v5's model uses 1.9).
  - What v5 still changes:
    - Hopper and Walker2d pay no healthy reward on the step that ends an episode.
    - InvertedPendulum pays 0 instead of 1 on that step.
    - Each of these differs from v4 by one reward per terminated episode.
- Box-constrained L-BFGS is `optax.lbfgs` on a sigmoid reparametrization, not jaxopt's L-BFGS-B, which is no longer
  maintained. Convergence is judged on the gradient in the sigmoid coordinates. Restarts also stop on nonfinite
  states, failed sufficient decrease in Optax's line search, or five steps without improving the best finite objective.
  - On a GP fit from a Hopper run, both reach the same likelihood (−35.2675 against −35.2676).
  - It costs 0.6 s per 32-restart fit against jaxopt's 0.07 s.
- **The acquisition** uses the Woodbury form of the released objective. The value is identical (checked in `tests/test_gp.py`), but each evaluation needs only
  query-sized factorizations, where the released code refactored a d × d matrix every time.
  Its query-independent baseline is excluded during optimization to preserve float32 resolution, then restored in
  the reported `info`. The GP likelihood uses the squared norm of a whitened residual, needing one triangular solve.
- **Learning curve.** The center is evaluated on 5 fresh episodes, as before. Each query's raw return is now read from its training episode, where
  the released code played one more evaluation episode, and the discounted best (`eval/objective_max`) uses the same
  episodes, where the released code played five more per query. The R² test of the prior (`eval/prior_r2`) re-evaluates the
  queries only with `--prior-test-episodes N`. This cuts the evaluation episodes per iteration by about 5×.
- **Episode count.** `episodes` counts what each iteration actually played. On the first iteration that includes the n_max − 1 random policies that seed the window; the released count did not.
- **Removed metric.** `info_gain` (the gain of the winning acquisition restart over its start) is no longer logged.
- **Per-query metrics.** The `gp/`, `critic/` and `acquisition/` metrics are the iteration's last query's.
