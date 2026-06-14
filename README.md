# tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/tramdag)](https://pypi.org/project/tramdag/)
[![CI](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Status: beta (0.x), under active development.** The API may change between
> releases until 1.0; pin a version (`tramdag==0.2.*`) for reproducibility.

**TRAM-DAGs** model each variable of a structural causal model with a
(transformation-model) flow: one triangular normalizing flow from iid
standard-logistic latents to the observed variables, whose Jacobian sparsity is
exactly your causal DAG. Fit it **once** on observational data and answer all
three rungs of Pearl's causal hierarchy — observational (L1), interventional
(L2, the do-operator), and counterfactual (L3, Pearl abduction) — while keeping
**interpretable effects**: every linear-shift coefficient is a log-odds ratio,
exactly as in classical proportional-odds models.

> Beate Sick & Oliver Dürr, *Interpretable Neural Causal Models with TRAM-DAGs*,
> CLeaR 2025 ([arXiv:2503.16206](https://arxiv.org/abs/2503.16206)).
> This repo is the reference implementation (PyTorch, built on
> [zuko](https://zuko.readthedocs.io/stable/)); all of the paper's experiments are
> replicated here with pinned tests.

**5-minute showcase**: the Colab badge above fits the paper's bimodal benchmark
live (GPU-ready) and walks L1 → L2 → L3, every answer checked against analytic
ground truth. Didactic walkthrough of the model:
[`notebooks/intro_tram_dag.py`](notebooks/intro_tram_dag.py).

## Install

```bash
pip install tramdag            # PyPI
uv sync                        # or: dev setup from a clone (tests, experiments)
```

## 30 seconds of API

```python
import tramdag as td
from tramdag import CausalFlowDAG, ContinuousNode, OrdinalNode

spec = {                                  # the spec IS the labelled DAG
    "Age":     ContinuousNode(),
    "mRS_pre": OrdinalNode(levels=6, parents={"Age": "ci"}),
    "NIHSSa":  ContinuousNode(parents={"Age": "ci", "mRS_pre": "ls"}),
    "T":       OrdinalNode(levels=2,
                           parents={"Age": "ci", "mRS_pre": "ls", "NIHSSa": "cs"}),
    "mRS_3m":  OrdinalNode(levels=7,
                           parents={"Age": "ci", "mRS_pre": "ls",
                                    "NIHSSa": "cs", "T": "ls"}),
}
flow = CausalFlowDAG(spec)                # validates acyclicity, builds the flow

# self-stopping training: per-node plateau lr decay + freezing of converged
# nodes (exact, since the per-node NLLs have independent gradients);
# see docs/training-speed.md for benchmarks and the classic two-phase recipe
flow.fit(train_df, val_df, epochs=4000, learning_rate=1e-2,
         schedule="plateau", plateau_patience=30, freeze_patience=120)

# all-`ls` model? fit it classically instead: deterministic float64 L-BFGS,
# exact MLE matching statsmodels/R (see notebooks/classical_fit_tram_dag.py)
flow.fit_classical(train_df)               # raises on cs/ci specs

flow.log_prob(df)                          # L1: joint log-likelihood per row
flow.sample(1000)                          # L1: observational sampling
flow.sample(1000, do={"T": 1})             # L2: interventional (graph mutilation)
flow.pmf(df, node="mRS_3m", do={"T": 1})   # L2: analytic interventional PMF

u  = flow.abduct(df)                       # L3 step 1: latents from observations
cf = flow.sample(do={"T": 1}, u=u)         # L3 steps 2+3: counterfactuals

flow.save("flow.pt"); flow = CausalFlowDAG.load("flow.pt")

td.simulations.REGISTRY                    # synthetic DGPs with known ground truth
```

## The model in one table

Per node, the transformation is additive on the latent (log-odds) scale —
`u = h(x; θ) + Σ β·x_pa + Σ g(x_pa)` — and each parent edge declares how it enters:

| edge term | meaning | interpretability |
|---|---|---|
| `ls` | linear shift `β·x_pa` | `exp(β)` is an odds ratio — one number per edge |
| `cs` | complex shift `g(x_pa)` (MLP), still additive | odds-ratio *function*, plot `g` |
| `ci` | complex intercept: the transform's parameters depend on the parents (several `ci` parents feed one joint network) | maximal flexibility, interactions |

Continuous nodes carry a monotone 1-D transform (`bernstein` — TRAM-faithful
default, `spline`, `affine`; `ContinuousNode(transform=..., transform_kwargs=...)`);
ordinal nodes an ordered-logit head `P(x ≤ k) = σ(θ_k − shift)`. Abduction is exact
for continuous nodes and truncated-logistic for ordinal ones, so
`flow.sample(u=flow.abduct(df))` reproduces `df` exactly / level-exactly.

How both fitting paths (stochastic `fit` and classical `fit_classical`) work, with
links to the code and notes on the memory model and future optimizers:
[`docs/fitting.md`](docs/fitting.md).

## Validation (all pinned by tests)

- **Paper replication** — every experiment of the CLeaR paper is a registry
  family (numpy-only SCM + frozen CSVs + replication script):

  | family | paper | demonstrates |
  |---|---|---|
  | `triangle` (`linear`,`atan`,`sin`) | §6.1 | LS coefficient recovery (β = 2, −0.2, +0.3), CS curve ≡ −f(x₂), non-monotone f |
  | `triangle-mixed` (`linear`,`exp`) | §6.2 | mixed data L1/L2 + the C.4 odds-ratio check (OR ≈ 7.4) |
  | `vaca` | §5.1–5.2 | the bimodal L1 case a default CNF misses; L2 `p(x₃ \| do(x₂))` |
  | `carefl` | §5.3 | L3 counterfactual curves vs **analytic** truth |

  ```bash
  cd experiments && uv run python paper_triangle.py atan cs   # etc., see paper_*.py
  ```

  Sign note: ordinal shifts are *subtracted* here but *added* in the paper, so
  fitted ordinal weights are the paper's with flipped sign (`truth.json` records
  both conventions per family).

- **Exact classical equivalence** — an all-`ls` flow trained to convergence *is*
  the proportional-odds MLE: coefficients match `statsmodels` **and** R
  `MASS::polr` to ~4 decimals (`experiments/validate_ls.py`, R reference committed
  under `data/magic-mrclean/*/ref_ls/`).

- **Training speed** — schedules, per-node freezing, LBFGS and device benchmarks:
  [`docs/training-speed.md`](docs/training-speed.md).

What the tests actually guarantee — the principles behind them (known identities,
the datasets and software they compare against, and how the ground truth was
obtained) — is documented in [`tests/README.md`](tests/README.md).

## Case study: individualized treatment effects in stroke

The method's flagship application estimates individualized thrombectomy effects
from the observational MAGIC cohort with external validation against the
MR CLEAN trial:

> Dürr, Herzog, Bühler, Wegener & Sick, *Estimating Individualized Treatment
> Effects in Acute Ischemic Stroke with Causal Transformation Models (TRAM-DAG)*
> ([arXiv:2606.12623](https://arxiv.org/abs/2606.12623)).

The clinical data is private and **never** part of this repo. Its public
stand-in is `data/magic-mrclean/` — a fully synthetic cohort with the same
schema and **known ground truth** (true ATE, true individual counterfactuals),
including an `nl` variant where an all-`ls` model is provably misspecified:

| `nl` variant | ATE | vs true **+0.104** |
|---|---|---|
| naive observational contrast | +0.303 | confounded (overstates 2.9×) |
| all-`ls` flow | +0.076 | undershoots (misses the age-varying effect) |
| flexible (`ci`/`cs`) flow | +0.101 | **recovers the truth** |

Full storyline, clinical-data context, R cross-check and reading notes:
[`docs/stroke-case-study.md`](docs/stroke-case-study.md).

## Layout

```
src/tramdag/            spec.py transforms.py conditioners.py flow.py
                        simulations/   (magic_mrclean, triangle, vaca, carefl + CLIs)
data/                   frozen synthetic CSVs + truth.json — a test contract
experiments/            stroke pipeline, paper replications, training benchmark
notebooks/              intro (didactic) + Colab demo   (jupytext .py — see README there)
tests/                  66 tests: unit, known-truth recovery, R regression
docs/                   training-speed.md, stroke-case-study.md
```

Implementation conventions (latent-scale signs, raw/one-hot parent encoding,
log-space ordinal likelihood, seeding) are documented in
[`CLAUDE.md`](CLAUDE.md) and pinned by tests.

## Citation

If you use `tramdag`, please cite the method paper:

```bibtex
@inproceedings{sick2025tramdag,
  title     = {Interpretable Neural Causal Models with TRAM-DAGs},
  author    = {Sick, Beate and D{\"u}rr, Oliver},
  booktitle = {Proceedings of the 4th Conference on Causal Learning and Reasoning (CLeaR)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {275},
  year      = {2025},
}
```

For the stroke application (and the `magic-mrclean` cohort design) additionally:

```bibtex
@article{duerr2026stroke,
  title  = {Estimating Individualized Treatment Effects in Acute Ischemic Stroke
            with Causal Transformation Models (TRAM-DAG): A Multi-Centre
            Observational Study with External RCT Validation},
  author = {D{\"u}rr, Oliver and Herzog, Lisa and B{\"u}hler, Pascal and
            Wegener, Susanne and Sick, Beate},
  journal = {arXiv preprint arXiv:2606.12623},
  year   = {2026},
}
```
