# tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/tramdag)](https://pypi.org/project/tramdag/)
[![CI](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Status: 1.0 release candidate.** This README documents the
> **unreleased 1.0** API (term classes, `callbacks=`, `flow.shift_curve`,
> `VC(center="col")`). `0.3.0` on PyPI predates all of it. Until 1.0 ships,
> only a git install matches the docs below.

**TRAM-DAGs** model each variable of a structural causal model with a
(transformation-model) flow. The flow is one triangular normalizing flow from
iid standard-logistic latents to the observed variables. The triangular
adjacency structure is exactly your causal DAG. One fit on observational data
answers all three rungs of Pearl's causal hierarchy:

- observational (L1),
- interventional (L2, the do-operator),
- counterfactual (L3, Pearl abduction).

The model keeps **interpretable effects**. Every linear-shift coefficient is a
log-odds ratio, exactly as in classical proportional-odds models.

> Beate Sick & Oliver Dürr, *Interpretable Neural Causal Models with TRAM-DAGs*,
> CLeaR 2025 ([arXiv:2503.16206](https://arxiv.org/abs/2503.16206)).
> This repo is the reference implementation (PyTorch, built on
> [zuko](https://zuko.readthedocs.io/stable/)). Pinned tests replicate all of
> the paper's experiments here.

**Start here**: the [Colab badge above](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb) opens a notebook.
The notebook does three things:

- it fits the paper's bimodal benchmark,
- it walks L1 to L3,
- it writes the same DAG with interpretable terms.

Every answer is a computed check against analytic ground truth. A failed claim
therefore stops the notebook. The notebook runs on CPU in a few minutes.

[`docs/model.md`](docs/model.md) describes the model itself.
[`docs/interpretation.md`](docs/interpretation.md) describes how to read a
fitted model. [`notebooks/README.md`](notebooks/README.md) lists the other
notebooks.

## Install

```bash
uv add tramdag                 # latest release (PyPI); pip install tramdag works too
uv add "tramdag[plots]"        # with matplotlib, for plot_dag / plot_marginals / plot_training
uv add "tramdag @ git+https://github.com/tensorchiefs/tramdag.git@main"   # dev version (track main)
uv sync                        # or: dev setup from a clone (tests, experiments)
```

Pin the dev install to a commit for reproducibility. Use the form
`...tramdag.git@<sha>`.

## 30 seconds of API

```python
from tramdag import CausalFlowDAG, ContinuousNode, OrdinalNode, I, LS, CS

spec = {  # the spec IS the labelled DAG
    "X1": ContinuousNode(),
    "X2": ContinuousNode(I("X1")),
    "T": OrdinalNode(2, LS("X1") + CS("X2")),  # 2 levels, column coded 0..1
    "Y": OrdinalNode(4, I("X1") + CS("X2") + LS("T")),
}
# train_df / val_df: DataFrames with one column per node
flow = CausalFlowDAG(spec)  # validates acyclicity, builds the flow

# fit() is one minibatch Adam loop with Keras-shaped extras: validation_data=
# (or validation_split=0.1) records the per-epoch validation NLL in
# flow.history["val"], verbose= prints progress; schedules and early stopping
# attach through optimizer= and callbacks= (docs/fitting.md)
from tramdag.callbacks import EarlyStopping

flow.fit(
    train_df,
    epochs=4000,
    batch_size=512,
    validation_data=val_df,
    verbose=100,
    # keep the best-validation weights; stop once the best is 200 epochs old
    callbacks=[EarlyStopping(patience=200)],
)

# all-`ls` model? fit it classically instead: deterministic float64 L-BFGS,
# exact MLE matching statsmodels/R (see docs/fitting.md)
flow.fit_classical(train_df)  # raises on cs/ci/vc specs

flow.log_prob(df)  # L1: joint log-likelihood per row
flow.sample(1000)  # L1: observational sampling
flow.sample(1000, do={"T": 1})  # L2: interventional (graph mutilation)
flow.pmf(df, node="Y", do={"T": 1})  # L2: analytic interventional PMF
flow.density(df, node="X2", grid=grid, do={"X1": 0.5})  # ... and density, continuous nodes

u = flow.abduct(df)  # L3 step 1: latents from observations
cf = flow.sample(do={"T": 1}, u=u)  # L3 steps 2+3: counterfactuals

flow.ls_coefficients()  # interpret: per-edge log-odds-ratios (LS terms)
# per-parent partial effects of an additive complex intercept (centered):
# flow.intercept_contributions(df, "Y") on a CI("A", "B", allow_interaction=False)

# heterogeneous treatment effects: a small, penalized effect head beta(x)*T
# (VC term) with a first-class read-out — see docs/varying-coefficients.md
# e.g. CS("X1", "X2") + VC("X1", t="T") ->
# flow.varying_coef(df, "Y")               # beta(x): deterministic, y-free

flow.scores(df, node="Y")  # per-observation scores dl_i/dtheta
flow.effect_modifier_scan(df, "Y", t="T")  # which VC modifiers? (CUSUM
# scan from a cheap all-ls fit) — docs/scores.md

flow.save("flow.pt")
flow = CausalFlowDAG.load("flow.pt")
```

## See the DAG

```python
from tramdag import plot_dag
plot_dag(spec)   # or plot_dag(flow) — layers left to right, one edge style per term
```

Continuous nodes are ellipses. Ordinal nodes are rounded boxes with their
level count. The edge styles are:

- `LS` is thin gray,
- `CS` is thick blue,
- a complex intercept is dashed orange,
- a `VC` treatment edge is red with its modifiers dotted.

`plot_dag` needs the `plots` extra.

## The model in detail: spec → math → networks

Per node, the transformation is additive on the latent (log-odds) scale — one
intercept term `I` plus any number of shifts (notation:
[`docs/notation.md`](docs/notation.md)):

`u = h_ϑ(x) + Σ β·x_pa + Σ g(x_pa) + (β₀ + b_Θ(x_mod))·x_t`

| term | math | what gets built | interpretability |
|------------------|------------------|---------------------------------------------|-------------------------------|
| `I()` / bare `I` / omitted | `h_ϑ(x)` — constant ϑ | `SimpleIntercept`: one free parameter vector, no network | the baseline transform |
| `I("A")` | `h_ϑ(a)(x)` — ϑ bends with the parent | `ComplexIntercept`: NN `[8, 8] → n_params` | the parent reshapes the whole distribution; no single coefficient |
| `I("A","B")` (default `allow_interaction=True`) | `h_ϑ(a,b)(x)` | **one joint** NN over both parents — they interact in ϑ | maximal flexibility |
| `I("A","B", allow_interaction=False)` | `h_ϑ(a)+ϑ(b)(x)` | one NN **per parent**, parameter vectors summed in coefficient space | per-parent partial effects via `flow.intercept_contributions` |
| `LS("A")` | `β·a` | `Linear(width, 1)`, no bias — **one parameter per feature column** (one for a continuous parent, `levels` for a one-hot ordinal, identified only as differences `w[k]-w[0]`) | `exp(β)` is an odds ratio |
| `CS("A")` | `g(a)`, additive | `ComplexShift`: NN `[64, 128, 64] → 1` | plot `g` |
| `CS("A","B")` | `g(a,b)` — joint | one NN over the concatenated features | interaction *in the shift* |
| `CS("A") + CS("B")` | `g₁(a) + g₂(b)` | two NNs, scalars added | GAM-style, each effect plottable |
| `VC("A","B", t="T")` | `(β₀ + b_Θ(a,b))·x_t` | scalar `β₀` + zero-initialised penalized NN `[16] → 1`; the treatment value multiplies, it never enters the net | `β₀` ≈ constant effect, `flow.varying_coef` reads `β(x)` |

A node takes **at most one `I` term with parents**. A term list is therefore
always purely additive on the latent scale. Interactions exist only *inside* a
term. Lists and `+` sums are interchangeable:
`[LS("A"), CS("B")]` ≡ `LS("A") + CS("B")`.

Three knobs on the terms:

- **`transform=` on `I`** picks the class of `h_ϑ` for a continuous node.
  There are three classes:

  - `"bernstein"` is the default. It has 20 coefficients. Its tails
    extrapolate with the boundary slope.
  - `"spline"` is a monotone RQ spline. It has 23 params at `bins=8`. Its
    tail slope is fixed.
  - `"affine"` has 2 params. The latent is then exactly logistic.

  Ordinal nodes have no transform to pick. Their intercept is the cutpoint
  vector, `P(x ≤ k) = σ(ϑ_k − shift)`.
- **`input_transform=` on `CI`/`CS`/`VC`** transforms that term's continuous
  network inputs. The statistics freeze at the first fit. The value is
  `"minmax"`, `"standardize"`, or a callable `fn(x, train)`. `LS` and the `VC`
  treatment stay raw, so their coefficients keep their units.
- **`units=` on `I`/`CS`/`VC`** sizes the term's network. For example,
  `units=[16]` gives one hidden layer of 16 neurons. The defaults match the
  PyTorch reference of this package
  ([buehlpa/TramDag](https://github.com/buehlpa/TramDag), `tram_models.py`).
  The defaults do **not** match the TRAM-DAG paper's own R nets. A replication
  therefore sets `units=` and `activation=` explicitly, as the configs in
  `experiments/paper/` do.

Feature widths: a continuous parent enters raw (1 column). An ordinal parent
enters one-hot (`levels` columns). Abduction is exact for continuous nodes and
truncated-logistic for ordinal ones. `flow.sample(u=flow.abduct(df))`
therefore reproduces `df` exactly / level-exactly.

There are two ways to fit the model: a stochastic optimizer (`fit`) and the
classical fit (`fit_classical`). In an all-`ls` model, each node-conditional
is a classical transformation model. For these models the classical fit is
deterministic and takes seconds. The measurement on the CI runner is ~10 s
against ~200 s for Adam. That measurement predates the 2026-09 epoch cut,
which made the Adam route ~3.5x faster. CI re-measures the numbers on the next
push.

[`docs/fitting.md`](docs/fitting.md) gives more information.

## Validation

Two layers, deliberately separate.

**The framework's own test suite** ([`tests/`](tests/)) measures the library
against three inline data-generating processes. The test suite carries these
processes itself, so it needs no research code to run. Its strongest claim is
an equality, not a similarity. An all-`ls` outcome node *is* an ordered-logit
model. The flow's MLE must therefore match `statsmodels` `OrderedModel` on the
same design matrix, and it does. The varying-coefficient acceptance bars
(effect recovery, propensity centering) are pinned the same way.

[`tests/README.md`](tests/README.md) documents what the tests guarantee and
how each ground truth is obtained.

**The paper replications** ([`experiments/`](experiments/)) are separate.
There is one self-contained script per dataset. Each script keeps its
hyperparameters in a sibling YAML file. Each area commits its expected results
under `ground_truth/`. A dedicated workflow runs the scripts and compares the
results.

| experiment | paper | demonstrates |
|----------------------|-------------|----------------------------------------------------------------------------|
| `triangle.py` (`linear-ls`, `linear-cs`, `atan-cs`, `sin-cs`) | §6.1, C.3 | LS coefficient recovery (β = 2, −0.2, +0.3), CS curve ≡ −f(x₂) for non-monotone f |
| `triangle_mixed.py` (`linear-ls`, `exp-cs`) | §6.2 | mixed data L1/L2 + the C.4 odds-ratio check (OR ≈ 7.4) |
| `vaca.py` | §5.1–5.2 | the bimodal L1 case a default CNF misses; L2 `p(x₃ \| do(x₂))` against analytic means |
| `carefl.py` | §5.3 | L3 counterfactual curves vs **analytic** truth |
| `validate_ls.py` | — | flow ≡ `statsmodels` ≡ R `MASS::polr` to ~4 decimals with `fit_classical` (a converged Adam `fit` gets ~1e-3), on a frozen synthetic cohort with a known true effect |

Sign note: ordinal shifts are *subtracted* here but *added* in the paper.
Fitted ordinal weights therefore carry the flipped sign of the paper values.
Each `truth.json` records both conventions.
[`experiments/paper/PAPER_COVERAGE.md`](experiments/paper/PAPER_COVERAGE.md)
gives a figure-by-figure account of what the replications reproduce and what
they do not reproduce (the competing CNF/NSF baselines).

**Training speed** — [`docs/training-speed.md`](docs/training-speed.md)
covers:

- the schedules,
- the per-node freezing,
- L-BFGS,
- the device benchmarks.

**Paper replication** — [`docs/paper-replication.md`](docs/paper-replication.md)
covers the eight `experiments/paper` variants:

- every hyperparameter with its source in the paper's R code,
- the deviations,
- the numbers (paper / previous protocol / now).

## Testing policy
See the [`tests/README.md`](tests/README.md) file for more details.

## Layout

```
src/tramdag/            spec.py terms.py transforms.py conditioners.py
                        nodes.py flow.py fitting.py readouts.py scores.py
                        callbacks.py plots.py     <- the framework, and nothing else
tests/                  unit tests, identities, acceptance bars, three inline DGPs
experiments/            research code, one directory per area, each self-contained:
                          paper/       the replications + generators + frozen data
                          benchmarks/  training-speed and machine measurements
                          misc/        the classical-MLE validation
                        paper/misc: <name>.py + <name>.yaml, data/,
                        ground_truth/, tests/, results/ (benchmarks reads the
                        other areas' data and pins no ground truth)
notebooks/              five executed examples: the combined Colab
                        introduction and demo, training strategies,
                        additive-vs-joint
                        intercepts, varying coefficients, classical fitting
docs/                   model.md and notation.md (what the model is),
                        interpretation.md (reading a fitted one), fitting.md,
                        training-speed.md, paper-replication.md,
                        varying-coefficients.md, scores.md, code-map.md,
                        zuko-upstream.md,
                        architecture.md + adr/ (the 1.0 design and its
                        refusals)
```

[`CLAUDE.md`](CLAUDE.md) documents these implementation conventions, and tests
pin them:

- the latent-scale signs,
- the raw parent encoding and the one-hot parent encoding,
- the log-space ordinal likelihood,
- the seeding.

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

<!-- For the stroke application (and the `magic-mrclean` cohort design) additionally:

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
``` -->
