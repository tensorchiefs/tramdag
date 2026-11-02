# Tests: what they guarantee

This file gives an overview of the tests and how to run them.

## Running the tests

The commands are in [`CONTRIBUTING.md`](../CONTRIBUTING.md).

- **`slow` marker**: the long fits carry `@pytest.mark.slow`. The option
  `-m "not slow"` skips these fits. The marker is not "everything that trains a
  flow". A feature's acceptance number (the `VC` recovery bar, the centering
  bias reduction) trains one flow deliberately in the fast subset. Every run
  therefore measures that number.
- **CI** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the
  fast subset on every pull request and the **full** suite nightly and on
  demand. The split exists because the full suite is long.
- **Determinism**: tests seed `torch` before they construct the flow, so the
  fits are reproducible.

## Testing principles

Four kinds of test, in rough order of how much trust they carry:

1. **Known mathematical identities**: properties that must hold by the math,
   independent of any reference implementation:
   - the monotone transforms invert exactly (`test_univariate_roundtrip`).
   - abduction → push-forward reproduces the data (bijective round-trip,
     `test_abduction_roundtrip`).
   - the joint `log_prob` equals the sum of per-node terms
     (`test_log_prob_finite_and_decomposes`).
   - a `do` intervention changes only descendants
     (`test_counterfactual_only_changes_descendants`).

2. **Equivalence to independent implementations**: the strongest external
   check. An all-`ls` model *is* an ordered logit or a Colr model. It must
   therefore match software that other people wrote in other languages.
   - vs. **`statsmodels`** `OrderedModel` (computed at test time):
     `test_ls_node_equals_proportional_odds`, `test_matches_statsmodels_mle`,
     `test_torch_plateau_scheduler_preserves_exact_mle`.
   - the two optimizers agree on the same optimum: `test_agrees_with_adam_mle`.

3. **Known-truth recovery**: the inline DGPs (see `conftest.py`) *are* the
   ground truth. The tests can therefore measure quantities that no real
   dataset exposes:
   - the true linear-shift coefficients of the all-`ls` chain
     (`test_continuous_only_all_ls_recovers_the_true_shift`,
     `test_matches_statsmodels_mle`).
   - the pointwise effect function `beta(x)` and the bias reduction from
     propensity centering. The bars are in
     [varying-coefficients.md](../docs/varying-coefficients.md#validation).

   The **research** DGPs (the paper replications and their frozen CSVs) are not
   tested here. They live in [`experiments/`](../experiments/). The experiments
   workflow checks them against its committed ground truth.

4. **Numerical-stability & invariant guards**:
   - the ordinal log-likelihood keeps non-zero gradients under float32
     saturation (`test_ordinal_log_prob_gradient_survives_saturation`).
   - cutpoints stay in increasing order and PMFs sum to one
     (`test_ordinal_cutpoints_increasing_and_pmf_sums_to_one`).
   - `save`/`load` round-trips a fitted model (`test_save_load_roundtrip`).
   - a schedule through the hooks does not break the exact-MLE property
     (`test_torch_plateau_scheduler_preserves_exact_mle`).
   - DAG validation catches cycles and orders the nodes correctly
     (`test_cycle_detected`, `test_topological_order`).

## Where the ground truth comes from

The reference values that the tests compare against come from two sources:

- **By construction.** The inline DGPs are logistic-latent flows with fixed
  coefficients, cutpoints and effect functions. The true parameters are
  therefore the numbers that generate the data. The DGPs are numpy-only and
  deliberately independent of the flow.
- **Independent software.** The classical-equivalence tests fit `statsmodels`
  `OrderedModel` at test time on the same design matrix that the flow builds.
  An all-`ls` outcome node *is* an ordered-logit model. This is therefore an
  equality claim against software that other people wrote. It is not a tuned
  similarity.

## Layout

`conftest.py` holds the **inline DGPs** (all-`ls` chain, heterogeneous effect,
confounded-with-misfit). It also holds the helpers that are provably
identical across modules. Specs stay per-module. Each module pins the one
syntax variant that its property needs. A shared spec can couple unrelated
acceptance bars.

Each test file opens with a docstring that says what it covers.
`tests/tools/` holds the state-dict smoke script and its baseline.

Note: the additive-CI and joint-CS *known-truth* acceptance tests carry the
slow marker. The fast lane (`-m "not slow"`) therefore covers those features
with build smokes and finiteness smokes only. The truth bars run in the full
suite.

Note: `test_ordinal_log_prob_matches_pmf` compares the log-space form with
the pmf only where the pmf is accurate. Below about 1e-4 the naive
sigmoid-difference pmf hits float32 quantization (eps about 1.2e-7 near
sigmoid = 1). The log-space form stays accurate there.

Note: `_halve` in `test_input_transform.py` sits at module level. `save()`
accepts only a picklable input transform, and a local function is not
picklable.
