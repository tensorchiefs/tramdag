# tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/tramdag)](https://pypi.org/project/tramdag/)
[![CI](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TRAM-DAGs** model each variable of a structural causal model with a
transformation-model flow. The flow is one triangular normalizing flow from
iid standard-logistic latents to the observed variables, and its triangular
structure is exactly your causal DAG. One fit on observational data answers
all three rungs of Pearl's causal hierarchy: observational, interventional
(the do-operator) and counterfactual (abduction). The model keeps
**interpretable effects**: every linear-shift coefficient is a log-odds ratio,
as in a classical proportional-odds model.

> Beate Sick & Oliver Dürr, *Interpretable Neural Causal Models with TRAM-DAGs*,
> CLeaR 2025 ([arXiv:2503.16206](https://arxiv.org/abs/2503.16206)).
> This repository is the reference implementation, in PyTorch on
> [zuko](https://zuko.readthedocs.io/stable/). It replicates the paper's
> experiments under continuous integration.

## Start here

The [demo notebook](notebooks/demo_tram_dag_colab.py) is the introduction. It
fits the paper's bimodal benchmark, walks the three rungs, and checks every
answer against analytic ground truth. It runs on CPU in a few minutes, in
[Colab](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
or locally.

## Install

The package is on PyPI as `tramdag`; `uv add tramdag` or `pip install tramdag`
installs it, and the `plots` extra (`tramdag[plots]`) adds matplotlib for the
figures. The PyPI release lags `main`; for the current state install from git
(`tramdag @ git+https://github.com/tensorchiefs/tramdag.git@main`, pinned to a
commit for reproducibility). A clone with `uv sync` gives the development
setup with tests and experiments.

## Documentation

Guides, executed notebooks and the API reference are published at the
[documentation site](https://tensorchiefs.github.io/tramdag/). Each subject has
one guide, and its worked example is one notebook.

| Subject | Guide | Example |
|---|---|---|
| The model, its notation, what it cannot do | [`docs/model.md`](docs/model.md), [`docs/notation.md`](docs/notation.md) | [demo](notebooks/demo_tram_dag_colab.py) |
| Reading a fitted model: coefficients, curves, interventional distributions | [`docs/interpretation.md`](docs/interpretation.md) | [classical fitting](notebooks/classical_fit_tram_dag.py) |
| Fitting: the likelihood, the Adam loop and its callbacks, the classical fit | [`docs/fitting.md`](docs/fitting.md) | [training strategies](notebooks/training_strategies.py) |
| Training speed measurements | [`docs/training-speed.md`](docs/training-speed.md) | |
| Complex intercepts, joint versus additive | | [additive vs joint intercepts](notebooks/additive_vs_joint_ci.py) |
| Treatment effects that vary with covariates | [`docs/varying-coefficients.md`](docs/varying-coefficients.md) | [varying coefficients](notebooks/varying_coefficients.py) |
| Scores and the effect-modifier scan | [`docs/scores.md`](docs/scores.md) | [varying coefficients](notebooks/varying_coefficients.py) |
| Agreement with statsmodels and R, standard errors | | [classical fitting](notebooks/classical_fit_tram_dag.py) |
| What the tests guarantee | [`tests/README.md`](tests/README.md) | |
| The API, every default and where it lives | [`docs/code-map.md`](docs/code-map.md) | |
| How the package is built inside | [`docs/architecture.md`](docs/architecture.md) | |
| zuko, and what tramdag would upstream | [`docs/zuko-upstream.md`](docs/zuko-upstream.md) | |

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, tests, linting and
releases; [`notebooks/README.md`](notebooks/README.md) how the notebooks are
written and run.

## Experiments

`experiments/` replicates the paper's experiments against its own R code and
checks them against committed ground truth in continuous integration.
[`experiments/README.md`](experiments/README.md) explains the layout, the
YAML variants and the frozen data;
[`docs/paper-replication.md`](docs/paper-replication.md) gives the protocol,
the hyperparameters and the numbers, experiment by experiment.

## Layout

- `src/tramdag/` is the framework, and nothing else.
- `tests/` measures it against three inline data-generating processes; it
  never imports research code.
- `experiments/` holds the paper replications, the frozen data sets and the
  benchmarks, one directory per area, each self-contained.
- `notebooks/` are jupytext `.py` files, executed by the docs workflow.
- `docs/` are the guides; the API pages render from the docstrings.

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
