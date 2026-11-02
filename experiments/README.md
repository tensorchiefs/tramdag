# Experiments: the TRAM-DAG paper replications

This directory holds research code. The installed `tramdag` package does not
contain it. It replicates the TRAM-DAG paper
([Sick & Dürr, 2025](https://arxiv.org/abs/2503.16206)) against the paper's
own R code. It holds four scripts, their SCM generators under `simulations/`,
the frozen datasets under `data/`, and the expected results under
`ground_truth/`. [`common.py`](common.py) holds the output layout that
the workflow reads and [`check.py`](check.py) the ground-truth comparison.

## Running one

Experiments run as modules, from this directory:

```bash
cd experiments
uv run python -m triangle atan-cs        # fit + figures + metrics
uv run python -m check triangle-atan-cs  # vs ground truth
uv run python -m check_data              # frozen data regenerates
uv run pytest .                          # the checks on the checker, configs and generators
```

Every run writes to `results/<name>/` (gitignored):

- `metrics.json`, the numbers that CI checks,
- `report.md`, what the run is, the metrics table and the figures with their
  captions (the experiments workflow posts this file as a commit comment),
- `flow.pt`, the fitted model, so `CausalFlowDAG.load` can query a finished
  run again instead of refitting it,
- `plots/*.png`: the package's own DAG, training and marginals figures, and
  the paper's figures.

## The scripts

| script | dataset | paper | variants |
|---|---|---|---|
| [`triangle.py`](triangle.py) | continuous triangle | Sec. 6.1, App. C.3 | `linear-ls`, `linear-cs`, `atan-cs`, `sin-cs` |
| [`triangle_mixed.py`](triangle_mixed.py) | triangle with an ordinal x3 | Sec. 6.2, App. C.4 | `linear-ls`, `exp-cs` |
| [`vaca.py`](vaca.py) | VACA/CNF bimodal benchmark | Sec. 5.1 and 5.2, App. C.1 | `flexible` |
| [`carefl.py`](carefl.py) | CAREFL Laplace SCM | Sec. 5.3, App. C.2 | `flexible` |

Each variant's protocol, its deviations from the paper and the measurements
behind them are in [`docs/paper-replication.md`](../docs/paper-replication.md).
[`PAPER_COVERAGE.md`](PAPER_COVERAGE.md) lists which paper figure each variant
reproduces and what the replications deliberately do not reproduce. To run
variants locally, run one or two at a time with `OMP_NUM_THREADS=2`.

All four scripts have the same shape. They hold imports, function definitions,
a `run(variant)` function with the whole experiment, and a `__main__` block
whose argparse call selects the variant.

## The blueprint: spec and hyperparameters live in YAML, not in code

Each script reads its sibling `<script>.yaml` and nothing else. There are no
defaults in the code and no CLI flags that change a number.
`common.py::load_variant` parses the file and gives the variant's section.
A value that several variants share appears once under a YAML anchor. The
variants merge it with `<<`, which keeps the merge visible in the file.

Every variant carries the whole model and training recipe, deliberately
verbose, so one variant reads top to bottom:

- `spec:` is the full DAG in `tramdag.spec_from_dict` form. Per node it gives
  `kind` (plus `levels` for ordinal) and one `{term, parents, options}` entry
  per term. Every part of the model is a term option here: the transform,
  `n_coeffs`, `range_q`, the network `units` and `activation`,
  `input_transform`.
- `flow_kwargs:` go verbatim to `CausalFlowDAG(spec, **flow_kwargs)`: `seed`
  and `init`.
- `marginal_init:` says whether the fit also starts the simple intercepts at
  their empirical marginals (`flow.fit(..., marginal_init=True)`); the
  reference has no such start ([`docs/paper-replication.md`](../docs/paper-replication.md)).
- `fit_kwargs:` go verbatim to `flow.fit(train, **fit_kwargs)`: `epochs`,
  `batch_size` and `seed`. `learning_rate` is a top-level key and reaches
  `flow.fit(learning_rate=)`; `schedule` and the `plateau_*` keys configure
  the per-node plateau scheduler the script builds.
- `figures:` gives every figure of the report its title and caption;
  placeholders such as `{f}` fill from the variant.
- everything else is data and scoring configuration, for example `n_train`,
  `dgp_seed` and the grids.

To change what a run does, edit the YAML file. To add a variant, add a
section; `argparse` finds it, because its choices come from the file, and so
does the experiments workflow's matrix.

## Ground truth

`ground_truth/<result-dir>.json` holds one entry per checked metric, in one
of two forms:

```json
{"beta12": {"value": 1.9825, "atol": 0.05},
 "cs_curve_max_abs_err": {"max": 0.23}}
```

`{value, atol}` is two-sided, for a quantity that must stay where it is.
`{max}` is an upper bound for an error measure, where a smaller number is a
better fit and must not fail the run. Tolerances are per metric, because torch
results differ slightly across operating systems and CPUs.

`check.py` fails on a metric outside its tolerance, on a metric above its
bound, and on a ground-truth entry that the run no longer produces. It reports
two conditions as notes, not failures. One is a `{max}` bound outside the band
from 1.5x to 4x its measurement. The other is a `{value, atol}` measurement
more than half-way to its tolerance. Below 1.5x a bound fails on another
machine for no reason. Above 4x it cannot catch a regression. A drifting
center keeps passing while it describes an older run. A bound that is meant to
be wide says so in a `"why"` string, which `check.py` prints instead of the
note. The string excuses width only.

Ground-truth regeneration is a deliberate act. Run the experiment and review
the figures. Then write the new values with a commit message that says what
moved and why.

## The frozen data is a contract

`data/` is input that the repository commits, not a cache. `check_data.py`
regenerates every dataset from the seed in its `truth.json` and compares to
1e-9. It does not compare to bit equality, because numpy's transcendental
functions move their last bits between releases.
`experiments/tests/test_generators.py` runs the same comparison.
/carefl-cf` is the one folder with no generator here: it is CAREFL's own
committed data, external frozen input.

## Notes on the reference protocols

These facts explain choices in the experiment code.

- `carefl.py` uses the grid of `carefl_fig5.r`, `seq(-3, 2.9, 0.1)`, for the
  curve files. The reference also uses the training rows as validation rows,
  so the plateau rule watches the training NLL. It holds nothing out, so a
  fresh standardized draw scores the fit.
- `helpers.fit_paper` sets the plateau patience to one less than the
  configured value. Torch reduces the rate at `bad > patience` and the
  reference reduces it at `bad == patience`. It computes the per-epoch
  validation NLL only for the plateau rule. The triangle scripts never
  computed it per epoch.
- `simulations/vaca.py` intervenes at -3, -1 and 0. The paper figures and its
  R code (`vaca_triangle.r`) use these values. The paper text says -3, -2 and
  0. The code follows the figures.
- `simulations/carefl.py` gives the paper observation in raw units. It pushes
  the latents (2, 1.5, 1.4, -1) through the SCM. The paper prints x3 and x4
  divided by the CAREFL sample standard deviations (6.01 and 1.91).
- The `source` strings in the simulation scripts match the frozen
  `data/*/truth.json` files, so they do not follow the cite style.
- `triangle.py` reports the `cs_curve` figure of a different paper figure per
  function: 7 for atan, 17 for the misspecified linear case and 18 for sin.
  `PAPER_COVERAGE.md` has the detail.
- Report tables carry no pipes in cell text. `cml` strips the backslash
  escapes, which desyncs the header from the separator row.
