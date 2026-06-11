# Changelog

## Unreleased

### Changed (behavior)

- **`CausalFlowDAG.fit(..., restore_best=False)` is now the default.** Training keeps
  the **final converged weights** instead of restoring per-node best-validation
  weights. Rationale:
  - *Least surprise* — `fit()` returns the model you trained, not a silently
    swapped earlier epoch.
  - *Exact classical comparison* — an all-`ls` model trained to convergence is now
    exactly the maximum-likelihood (proportional-odds) estimate, matching
    `statsmodels` `OrderedModel` and R `MASS::polr` to ~1e-3 (see
    `experiments/validate_ls.py`, `tests/test_simulations.py::test_all_ls_flow_is_exact_mle`).
    This was **not achievable before**: best-validation restoration pinned the fit
    off the training optimum.
  - Early stopping is now an explicit, opt-in regularization choice.

  To restore the previous behavior, pass `restore_best=True`.

  **Note for flexible (`ci`/`cs`) models:** their MLE *overfits the observational
  confounding*, so they need `restore_best=True` to recover the causal effect (lower
  validation NLL confirms it generalizes better). `experiments/run_experiment`
  therefore defaults `restore_best` per style — off for all-`ls`, on for flexible.

### Added

- `src/zuko_dag/simulations/magic_mrclean.py` — synthetic stroke cohort (SCM with
  known ground truth); `ls`/`nl` variants; CLI to (re)generate frozen CSVs.
- `data/magic-mrclean/` — frozen public CSVs + `fit_ls.R` classical R reference and
  committed `ref_ls/` outputs. The public, reproducible substitute for the private
  clinical data.
- `experiments/common.py::load_data(source)` — switch between `"magic"` (private) and
  `"magic-mrclean/{ls,nl}"` (synthetic, default).
- `experiments/sim_flow.py` — known-truth recovery storyline; `validate_ls.py`
  rewritten as a spot-on flow-vs-MLE-vs-R comparison.
- `tests/test_simulations.py` — generator, known-truth recovery, the all-`ls`
  spot-on MLE check, and the Python-vs-R regression.
