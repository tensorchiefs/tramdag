# zuko_dag — a causal normalizing flow re-implementation of the TRAM-DAG stroke experiments

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tram-dag-zuko/blob/main/notebooks/demo_tram_dag_colab.ipynb)
**5-minute demo** (`notebooks/demo_tram_dag_colab.py`): fit the paper's bimodal
VACA benchmark live (GPU-ready), then walk all three rungs of Pearl's ladder —
each answer checked against analytic ground truth.

This folder re-implements the TRAM-DAG stroke analysis from the parent repository as a
**single normalizing flow from latent variables to observed data**, built on
[zuko](https://zuko.readthedocs.io/stable/) and in the spirit of *Causal Normalizing
Flows* / DeCaFlow: one triangular transform whose Jacobian sparsity is exactly the
user-supplied DAG. It is mathematically the same model family as TRAM-DAG (so the
numbers are directly comparable) but packaged as one `torch.nn.Module` flow with

- a **do-operator** (interventional sampling and analytic interventional PMFs),
- **Pearl abduction** (observed data → latent variables; counterfactuals are
  composable as `abduct` + `sample(do=..., u=...)`),
- **configurable one-dimensional transforms** per continuous node
  (`bernstein` — TRAM-faithful, `spline` — rational-quadratic, `affine`).

Everything lives in this folder; nothing outside `zuko_dag/` is touched.

## Model

One flow `T: U → X`, base `U ~ iid standard logistic` (the TRAM latent), triangular in
the causal ordering. Per node, with the parents' contributions entering exactly as in
TRAM-DAG:

- **continuous:** `u = h(x; θ(pa_ci)) + Σ_ls β·x_pa + Σ_cs g(x_pa)` with `h` a monotone
  1-D transform fitted on the train 5%/95% quantile range and linearly extrapolated
  outside;
- **ordinal (K levels):** ordered-logit, `P(x ≤ k) = σ(θ_k − shift)` with increasing
  cutpoints (tramdag's parametrization). The latent is interval-identified, so
  abduction samples from the truncated logistic on the observed level's interval.

Edge term types keep TRAM-DAG semantics 1:1 — `ls` linear shift, `cs` complex shift
(64-128-64 MLP), `ci` complex intercept (8-8 MLP; several `ci` parents of one node feed
**jointly** into one intercept network, i.e. tramdag's interacting-ci groups happen
automatically). The joint NLL decomposes per node, so **one Adam optimizer trains the
whole flow at once** and the optimum coincides with TRAM-DAG's per-node training.
Per-node best-validation weights are tracked across `fit` calls and restored at the end.

## API

```python
from zuko_dag import CausalFlowDAG, ContinuousNode, OrdinalNode

spec = {                                            # the nihss6 DAG, transcribed
    "Age":     ContinuousNode(transform="bernstein"),
    "mRS_pre": OrdinalNode(levels=6, parents={"Age": "ci"}),
    "NIHSSa":  ContinuousNode(transform="bernstein",
                              parents={"Age": "ci", "mRS_pre": "ls"}),
    "T":       OrdinalNode(levels=2,
                           parents={"Age": "ci", "mRS_pre": "ls", "NIHSSa": "cs"}),
    "mRS_3m":  OrdinalNode(levels=7,
                           parents={"Age": "ci", "mRS_pre": "ls",
                                    "NIHSSa": "cs", "T": "ls"}),
}
flow = CausalFlowDAG(spec)        # validates acyclicity, topo-sorts, builds the flow
flow.fit(train_df, val_df, epochs=3000, learning_rate=1e-2, batch_size=256, seed=123)
flow.fit(train_df, val_df, epochs=1000, learning_rate=1e-3)   # phase 2, lower lr

# or self-stopping: per-node plateau lr decay + freezing of converged nodes
# (the NLL decomposes per node with independent gradients, so per-node
# schedules are exact; see docs/training-speed.md for benchmarks)
flow.fit(train_df, val_df, epochs=4000, learning_rate=1e-2,
         schedule="plateau", plateau_patience=30, freeze_patience=120)

flow.log_prob(df)                          # joint log-likelihood per row
flow.sample(1000)                          # observational sampling
flow.sample(1000, do={"T": 1})             # interventional sampling (mutilation)
flow.pmf(df, node="mRS_3m", do={"T": 1})   # analytic per-row interventional PMF

u  = flow.abduct(df)                       # Pearl step 1: latents from observations
cf = flow.sample(do={"T": 1}, u=u)         # Pearl steps 2+3: counterfactuals

flow.save("flow.pt"); flow = CausalFlowDAG.load("flow.pt")
```

`ContinuousNode(transform=...)` accepts `"bernstein"` (default, 20 coefficients),
`"spline"` (8-bin monotone RQS) and `"affine"`; extra kwargs are forwarded
(`transform_kwargs={"n_coeffs": 30}`). Abduction is exact for continuous nodes
(`u = h(x) + shift`) and a truncated-logistic draw for ordinal nodes — so
`flow.sample(u=flow.abduct(df))` reproduces `df` exactly (continuous) /
level-exactly (ordinal).

## Reproducing the experiments

From the project root (the `zuko_dag/` folder when inside the paper monorepo):

```bash
uv sync                                       # installs zuko, torch, ...
uv run pytest tests/ -q                       # unit + simulation tests

cd experiments
uv run python sim_flow.py nl                  # storyline on the synthetic cohort (default data)
uv run python all_ls_flow.py                  # all-edges-ls flow  (synthetic data by default)
uv run python nihss6_flow.py                  # flexible nihss6 config
uv run python validate_ls.py                  # all-ls flow vs statsmodels MLE
uv run python counterfactual_demo.py all_ls   # Pearl abduction showcase
uv run python counterfactual_demo.py nihss6
uv run python all_ls_long.py                  # 4x-longer convergence check
```

**Data sources.** The experiments default to the **public synthetic cohort**
`data/magic-mrclean/nl` (see [`data/magic-mrclean/README.md`](data/magic-mrclean/README.md)),
so everything above runs with no clinical data. Pass a source to switch — e.g.
`uv run python all_ls_flow.py magic` for the private MAGIC / MR CLEAN cohort
(`data/obs_data.csv`, filtered to NIHSSa ≥ 6, N = 1275), or `magic-mrclean/ls` for the
linear synthetic variant. Every run uses the same 80/10/10 split (`random_state=42`),
trains in two phases (3000 epochs @ lr 1e-2, 1000 @ 1e-3), and writes to
`results/<name>/`: training/diagnostic plots, per-patient interventional PMFs on the
RCT covariates (`rct_predicted_proba.csv`), and the checkpoint (`flow.pt`).

### The synthetic cohort (`magic-mrclean`)

A fully synthetic, **publishable** dataset shaped like the stroke study, generated from
a hand-specified SCM in the flow's own model family (logistic latents) — so it carries
**known ground truth** the real data cannot: the true ATE, individual counterfactuals,
and a case where an all-`ls` model is provably misspecified. It is a drop-in substitute
for the clinical data (identical schema), which is the migration path for the
standalone package: ship synthetic, keep the clinical data private.

Two variants (`ls`, `nl`) differ only by three mild non-linearities; the trial cohort
(`rct.csv`) is enrolled younger than the observational cohort (`obs.csv`), as real
trials are. Known-truth recovery (seed 7), ATE on P(good) over the trial population:

| nl variant | ATE | vs true **+0.104** |
|---|---|---|
| naive observational contrast | +0.303 | confounded (overstates 2.9×) |
| all-`ls` flow | +0.076 | undershoots (misses age-varying effect) |
| flexible (`ci`/`cs`) flow | +0.101 | **recovers the truth** |

This reproduces the real-data finding in miniature (flexible +0.108 vs all-`ls` +0.054)
but here against a *known* answer. On the `ls` variant (constant effect) the all-`ls`
flow, the R reference and the truth all coincide.

**R cross-check.** [`data/magic-mrclean/fit_ls.R`](data/magic-mrclean/fit_ls.R) refits
the all-`ls` DAG node-by-node in classical R (`MASS::polr` / `tram::Colr` / `glm`); its
committed `ref_ls/` outputs let `tests/test_simulations.py` assert the Python flow ≡ the
R fit (outcome-node coefficients and ATE) — a language-independent equivalence check,
with no clinical data involved.

### The TRAM-DAG paper DGPs

All simulation studies of the paper (Sick & Dürr, *Interpretable Neural Causal Models
with TRAM-DAGs*, CLeaR 2025, [arXiv:2503.16206](https://arxiv.org/abs/2503.16206)) are
implemented as registry families — numpy-only SCMs with known (often analytic) ground
truth, frozen n=5000 CSVs under `data/<name>/`, and replication experiment scripts:

| family | paper | DGP | demonstrates |
|---|---|---|---|
| `triangle` (`linear`,`atan`,`sin`) | §6.1, C.3 | logistic-latent TRAM: `h₂=5x₂+2x₁`, `h₃=0.63x₃−0.2x₁−f(x₂)`, GMM source | LS coefficient recovery (β=2, −0.2, +0.3), CS curve ≡ −f(x₂), non-monotone f |
| `triangle-mixed` (`linear`,`exp`) | §6.2, C.4 | same skeleton, x₃ ordinal (4 levels, θ=(−2, 0.42, 1.02)) | mixed-data L1/L2 + the odds-ratio intervention check (OR ≈ 7.4) |
| `vaca` | §5.1–5.2, C.1 | bimodal Gaussian triangle (Sánchez-Martín 2022) | L1 (bimodal marginal the default CNF misses) + L2 `p(x₃ \| do(x₂))` |
| `carefl` | §5.3, C.2 | 4-var Laplace SCM (Khemakhem 2021) | L3: counterfactual curves vs **analytic** truth |

```bash
cd experiments
uv run python paper_triangle.py atan cs        # Fig. 7/15/16: CS recovery
uv run python paper_triangle.py linear ls      # Fig. 14: coefficient trajectories
uv run python paper_triangle_mixed.py linear ls # Fig. 9/19 + C.4 OR check
uv run python paper_vaca.py                    # Fig. 4/5: L1 + L2 benchmark
uv run python paper_carefl.py                  # Fig. 6: counterfactual curves
```

Sign conventions: continuous nodes match the paper directly (`z = h(x) + shift`); for
ordinal nodes the flow *subtracts* the shift while the paper *adds* it, so fitted
ordinal weights are the paper's with flipped sign (recorded per family in
`truth.json` as `zuko` expectations). `tests/test_paper_dgps.py` pins the generators
(TRAM identities by KS test, frozen-CSV contract, analytic counterfactuals) and the
flow's recovery of each family's ground truth.

## Results

ATE = mean over RCT covariates of `P(mRS_3m ≤ 2 | do(T=1)) − P(mRS_3m ≤ 2 | do(T=0))`,
computed analytically from the outcome node's PMF (no Monte Carlo):

| model | ATE on P(good) | source |
|---|---|---|
| MR CLEAN RCT (ground truth) | **+0.135** [+0.057, +0.213] | Berkhemer et al. 2015 |
| zuko flow, nihss6 config | **+0.063** | `experiments/nihss6_flow.py` |
| zuko flow, all-ls | **+0.057** | `experiments/all_ls_flow.py` |
| classical proportional-odds MLE, same 80% split | +0.055 | `experiments/validate_ls.py` |
| TRAM-DAG `md_dag_ls` (all-ls) | +0.054 | parent repo |
| TRAM-DAG `nihss6` | +0.108 (seed 2: +0.092) | parent repo, paper |
| classical MLE, full data | +0.082 | parent repo |

Two reading notes:

- The **all-ls flow IS the classical MLE** when trained to convergence without early
  stopping (the default, `restore_best=False`): on the synthetic `ls` cohort its
  outcome-node coefficients match `statsmodels` *and* R `polr` to 4 decimals
  (Age 0.0526, NIHSSa 0.1630, T −0.9424; ATE +0.1429 vs +0.1428) — see
  `experiments/validate_ls.py` and `test_simulations.py::test_all_ls_flow_is_exact_mle`.
  This spot-on match is only possible because `fit()` no longer restores
  best-validation weights by default; early stopping would pin the fit off the train
  optimum (see CHANGELOG). The earlier +0.057-vs-+0.055 residual on the clinical data
  was exactly that early-stopping effect.
- **Flexible (`ci`/`cs`) models are different**: their MLE *overfits the observational
  confounding*, so they need `restore_best=True` (early-stopping regularization) to
  recover the causal effect — confirmed on the synthetic `nl` cohort, where the
  flexible MLE undershoots (+0.076) but the early-stopped fit recovers the true ATE
  (+0.10), with a lower validation NLL. `run_experiment` defaults `restore_best` per
  style accordingly (off for all-`ls`, on for flexible).
- The treatment effect is **weakly identified** in this observational cohort
  (see STORYLINE.md in the parent repo): the likelihood around the T-coefficient is
  nearly flat, and the parent repo documents that refits drift to the 80%-split
  optimum of ≈ +0.054 — the TRAM-DAG +0.108 is a near-likelihood-equivalent solution,
  not a sharper optimum. Both flow results sit inside the established acceptance band
  [+0.03, +0.14]; matching the band, not the point estimate, is the meaningful check.

**Training speed** (Apple-silicon CPU, joint single-optimizer training of all nodes):
the default all-ls schedule (3000 epochs @ lr 1e-2 + 1000 @ 1e-3) takes **44 s**
(~90 epochs/s); the 16000-epoch convergence check takes 163 s. Each run writes
`plots/training_speed.png` — NLL vs wall-clock time with the lr schedule overlaid.

The counterfactual demo (capability beyond the original tramdag scripts) abducts the
latents of the 128 held-out test patients, verifies exact factual reconstruction, and
predicts each patient's outcome under the opposite treatment
(`results/<name>/counterfactuals_test.csv`, `plots/counterfactual_mrs3m.png`).

## Layout

```
zuko_dag/
├── pyproject.toml                 # own uv project (zuko, torch, statsmodels, ...)
├── src/zuko_dag/
│   ├── spec.py                    # ContinuousNode / OrdinalNode, DAG validation
│   ├── transforms.py              # Bernstein / RQS / affine wrappers, ordinal
│   │                              #   (ordered-logit) transform + truncated-logistic
│   │                              #   abduction, numerically stable log-prob
│   ├── conditioners.py            # ls / cs / ci modules (tramdag widths)
│   ├── flow.py                    # CausalFlowDAG: fit, sample(do=,u=), abduct, pmf
│   └── simulations/               # synthetic-cohort generators (known ground truth)
│       ├── magic_mrclean.py       #   the stroke SCM, ls/nl variants + CLI
│       ├── triangle.py            #   paper §6 triangles (continuous + ordinal) + CLI
│       ├── vaca.py  carefl.py     #   paper §5 benchmarks (L1/L2 + L3) + CLIs
│       └── __init__.py            #   REGISTRY = {name: generator}
├── data/                          # frozen synthetic CSVs — a contract, see tests
│   ├── magic-mrclean/             #   stroke cohort + R reference (README, fit_ls.R)
│   ├── triangle/{linear,atan,sin}/        obs.csv truth.json
│   ├── triangle-mixed/{linear,exp}/       obs.csv truth.json
│   └── {vaca,carefl}/                     obs.csv truth.json
├── experiments/
│   ├── common.py                  # load_data(source), split, specs, plots, RCT eval
│   ├── sim_flow.py                # storyline: all-ls vs flexible vs known truth
│   ├── all_ls_flow.py             # all-ls experiment   (takes a data source arg)
│   ├── nihss6_flow.py             # flexible nihss6 config
│   ├── validate_ls.py             # flow vs statsmodels OrderedModel
│   ├── counterfactual_demo.py     # abduction → counterfactual showcase
│   └── paper_*.py                 # TRAM-DAG paper figure replications (+ paper_common)
├── tests/
│   ├── test_flow.py               # core unit tests
│   ├── test_simulations.py        # generator + known-truth recovery + R-reference regression
│   └── test_paper_dgps.py         # paper DGPs: generator pinning + flow recovery
└── results/<experiment>/          # plots, CSVs, checkpoints, run logs  (git-ignored)
```

## Implementation notes

- zuko's `BernsteinTransform` uses (nearly line-for-line) the same parametrization as
  tramdag's Bernstein code — softplus-cumsum increasing coefficients, linear
  extrapolation — so it is used directly rather than re-ported.
- Inversion (sampling) solves `h(x) = u − shift` by bisection with an **expanding
  bracket**, so latents arbitrarily far in the logistic tails invert correctly
  (zuko's built-in bisection would clip at its `bound`).
- The ordinal log-likelihood is computed **in log-space**
  (`logsigmoid` + stable `log1mexp`, choosing the better-conditioned of the CDF- and
  survival-side identities per element). The naive `σ(θ_u−s) − σ(θ_l−s)` saturates to
  an exact 0/1 in float32 for |t| ≳ 17, which produces *exactly zero* gradients — a
  badly initialized node (e.g. a linear shift on a raw-scale parent like Age) would
  freeze at its initialization forever.
- Parent encoding follows tramdag: continuous parents enter raw, ordinal parents
  one-hot.
