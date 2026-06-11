# CLAUDE.md — working context for tram-dag-zuko

## What this is

A causal normalizing-flow implementation of **TRAM-DAG** (transformation models on a
DAG) built on [zuko](https://zuko.readthedocs.io/stable/). One triangular flow from iid
standard-logistic latents to the observed variables; Jacobian sparsity = the DAG.
Supports the do-operator, Pearl abduction (counterfactuals), analytic interventional
PMFs, and per-node configurable monotone transforms (Bernstein / RQ-spline / affine).

Origin: extracted from the private `tensorchiefs/tram-dag-stroke` paper repo (folder
`zuko_dag/`). The paper analyzed the MAGIC stroke cohort against the MR CLEAN RCT;
that **clinical data is NOT in this repo** and never should be. The synthetic
`data/magic-mrclean/` cohort is the public stand-in (same schema, known ground truth).

## Commands

```bash
uv sync                          # install (uv.lock pinned: zuko, torch, statsmodels, ...)
uv run pytest tests/ -q          # full suite ~7 min; tests/test_flow.py alone ~20 s
cd experiments
uv run python sim_flow.py nl     # headline storyline (all-ls vs flexible vs known truth)
uv run python validate_ls.py     # spot-on flow == statsmodels == R polr check
```

Experiments default to the synthetic data (`magic-mrclean/nl`). The `magic` source
(private clinical data) only works inside the original paper monorepo.

## Architecture (src/zuko_dag/)

- `spec.py` — user-facing DAG spec: `{name: ContinuousNode|OrdinalNode}`, each node
  declares `parents={parent: term}` with term ∈ `ls` (linear shift), `cs` (complex
  shift MLP), `ci` (complex intercept — transform params from parents; multiple ci
  parents feed ONE joint network).
- `transforms.py` — monotone 1-D transforms wrapping zuko (`BernsteinUT`, `SplineUT`,
  `AffineUT`; pre-scaled from train 5%/95% quantiles to [-5,5], expanding-bracket
  bisection inverse) + the ordinal ordered-logit transform
  (`P(Y<=k) = sigmoid(theta_k - shift)`, cutpoints `[t0, t0+cumsum(exp(...))]`).
- `conditioners.py` — ls/cs/ci networks (widths replicate the original tramdag pkg).
- `flow.py` — `CausalFlowDAG`: `fit`, `sample(n, do=, u=)`, `abduct`, `pmf`,
  `log_prob`, `save/load`. NLL decomposes per node → one Adam fits all nodes jointly.
- `simulations/magic_mrclean.py` — the synthetic stroke SCM (numpy-only, independent
  of the flow); `ls`/`nl` variants; CLI regenerates `data/magic-mrclean/`.

## Conventions that matter (easy to get wrong)

- **Latent scale**: continuous `z = h(x) + shift` (shifts ADDED); ordinal
  `P(Y<=k) = sigmoid(theta_k − shift)` (shift SUBTRACTED). Both are the tramdag
  conventions; tests pin them.
- **Parent encoding**: continuous parents enter RAW (no standardization); ordinal
  parents one-hot (all levels). With cutpoints, only shift *differences* between
  one-hot levels are identified — compare `w[k] − w[0]` against classical references.
- **Ordinal log-prob is computed in log-space** (`logsigmoid` + stable `log1mexp`,
  better-conditioned side chosen per element). The naive sigmoid difference saturates
  in float32 → *exactly zero* gradients → a node can freeze at init forever. Do not
  "simplify" it back.
- **Seeding**: weight init happens at construction — call `torch.manual_seed` BEFORE
  `CausalFlowDAG(spec)`, not just in `fit`.
- **`fit(restore_best=False)` is the default** (keeps final converged weights = exact
  MLE; an all-`ls` model then matches statsmodels/R-polr to ~1e-3). `restore_best=True`
  = per-node best-validation restoration (early stopping). Key empirical finding:
  **flexible (ci/cs) models overfit observational confounding at the MLE and need
  `restore_best=True` to recover the causal effect; all-`ls` models don't.**
  `run_experiment` defaults per style. See CHANGELOG.md.

## Ground truth & reference numbers (seed 7 synthetic data)

- `data/magic-mrclean/{ls,nl}/truth.json` — true ATE from the SCM: `ls` +0.132,
  `nl` +0.104; naive confounded contrast +0.26/+0.30.
- `nl` storyline: all-`ls` flow ≈ +0.076 (biased — can't extrapolate the age-fading
  treatment effect to the younger RCT population), flexible flow ≈ +0.10 (recovers).
- Spot-on check (`ls` variant, full-data, restore_best=False): flow = statsmodels =
  R polr at Age 0.0526, NIHSSa 0.1630, T −0.9424; ATE +0.1429 vs +0.1428.
- R reference: `data/magic-mrclean/fit_ls.R` (needs `tram`, `MASS`); its committed
  `ref_ls/` outputs let tests run without R.
- Original clinical-data numbers (context only, not reproducible here): TRAM-DAG
  nihss6 +0.108, md_dag_ls +0.054, MR CLEAN RCT +0.135 [0.057, 0.213].

## Testing policy

- Frozen CSVs in `data/magic-mrclean/` are a contract — **never regenerate silently**;
  a new seed/equations → new folder (sim2-style), regenerate `ref_ls/` with R, update
  truth-dependent tests.
- New causal features should be validated against the simulator's known truth
  (`MagicMrClean.true_ate`, `counterfactual_pair` gives true individual
  counterfactuals via shared latents).

## Roadmap notes

- Generalize `simulations/` registry beyond the stroke DAG (different shapes, hidden
  confounding à la DeCaFlow).
- Package for PyPI when API stabilizes (the `CausalFlowDAG`/`ContinuousNode`/
  `OrdinalNode` surface).
