# tramdag: Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/tramdag)](https://pypi.org/project/tramdag/)
[![CI](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TRAM-DAGs** model each variable of a structural causal model with a
monotone normalizing flow, the statistician's *transformation model*. The
whole model is one triangular normalizing flow from iid standard-logistic
latents to the observed variables. Its triangular structure is exactly your
causal DAG. One fit on observational data answers
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

`uv add tramdag` or `pip install tramdag` installs the package from PyPI. The
`plots` extra (`tramdag[plots]`) adds matplotlib for the figures. A clone with
`uv sync` gives the development setup ([`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Documentation

The [documentation site](https://tensorchiefs.github.io/tramdag/) publishes
the guides, the executed notebooks and the API reference.

- The model: [`docs/model.md`](docs/model.md) and
  [`docs/notation.md`](docs/notation.md)
- Reading a fitted model: [`docs/interpretation.md`](docs/interpretation.md)
- Fitting: [`docs/fitting.md`](docs/fitting.md)
- Treatment effects that vary with covariates:
  [`docs/varying-coefficients.md`](docs/varying-coefficients.md) and
  [`docs/scores.md`](docs/scores.md)
- The design of the package: [`docs/architecture.md`](docs/architecture.md) and
  [`docs/code-map.md`](docs/code-map.md)
- The paper replications: [`docs/paper-replication.md`](docs/paper-replication.md)
  and [`experiments/README.md`](experiments/README.md)
- The tests: [`tests/README.md`](tests/README.md)
- The notebooks: [`notebooks/README.md`](notebooks/README.md)

## Use of AI tools

LLM coding assistants helped to develop this project. The authors reviewed
all content and take responsibility for it.

## Citation

If you use `tramdag`, please cite the method paper:

```bibtex
@inproceedings{sick2025tramdag,
  title     = {Interpretable Neural Causal Models with {TRAM-DAGs}},
  author    = {Sick, Beate and D{\"u}rr, Oliver},
  booktitle = {Proceedings of the 4th Conference on Causal Learning and Reasoning (CLeaR)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {275},
  pages     = {606-630},
  year      = {2025},
  eprint    = {2503.16206},
  url       = {https://arxiv.org/abs/2503.16206},
}
```
