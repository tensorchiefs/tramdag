# Experiments — the TRAM-DAG paper replications

This directory holds research code. The installed `tramdag` package does not
contain it. There is one directory per **area**, and each area owns everything
it needs:

| area | what it holds |
|---|---|
| [`paper/`](paper/) | the replications of [arXiv:2503.16206](https://arxiv.org/abs/2503.16206), their SCM generators, frozen datasets and expected results |
| [`benchmarks/`](benchmarks/) | training-speed and cross-machine measurements |
| [`misc/`](misc/) | everything else — currently the classical-MLE validation |

The `paper` area and the `misc` area each contain their own:

- `data/`,
- `ground_truth/`,
- `results/` (gitignored),
- `tests/`,
- the helpers that only that area needs.

`benchmarks/` is the exception by nature. It measures training speed *on* the
data of the other areas. It therefore reads `misc/data/` and `paper/data/`, and
it commits no ground truth of its own. Its output is a write-up in `docs/`, not
a pinned number.

Two files are shared. [`common.py`](common.py) holds the output layout that the
workflow reads. [`check.py`](check.py) holds the ground-truth comparison.

## Running one

Experiments run as modules, from this directory:

```bash
cd experiments
uv run python -m paper.triangle atan-cs        # fit + figures + metrics
uv run python -m check paper triangle-atan-cs  # vs ground truth
uv run python -m paper.check_data              # frozen data regenerates
uv run pytest .                                # the area checks (seconds)
```

Every run writes to `<area>/results/<name>/`. The run writes these files:

- `metrics.json`, the numbers that CI checks,
- `report.md`, the table plus the figures (the experiments workflow posts this
  file as a commit comment),
- `flow.pt`, the fitted model, so a finished run can be queried again with
  `CausalFlowDAG.load` instead of refitted,
- `plots/*.png`, for the runs that draw figures.

`validate_ls` is a numbers-only comparison and draws no figures.

## The scripts

| script | dataset | paper | variants |
|---|---|---|---|
| [`paper/triangle.py`](paper/triangle.py) | continuous triangle | Sec. 6.1, App. C.3 | `linear-ls`, `linear-cs`, `atan-cs`, `sin-cs` |
| [`paper/triangle_mixed.py`](paper/triangle_mixed.py) | triangle with an ordinal x3 | Sec. 6.2, App. C.4 | `linear-ls`, `exp-cs` |
| [`paper/vaca.py`](paper/vaca.py) | VACA/CNF bimodal benchmark | Sec. 5.1–5.2, App. C.1 | `flexible` |
| [`paper/carefl.py`](paper/carefl.py) | CAREFL Laplace SCM | Sec. 5.3, App. C.2 | `flexible` |
| [`misc/validate_ls.py`](misc/validate_ls.py) | frozen synthetic cohort | — (framework anchor) | `adam`, `classical` |

Two benchmarks live beside these scripts. They are measured, not checked
against ground truth. A maintainer runs them by hand and reports the numbers in
the docs:

| script | measures | output |
|---|---|---|
| [`benchmarks/bench_training.py`](benchmarks/bench_training.py) | time-to-target for lr schedules, batch size, device, L-BFGS | [`docs/training-speed.md`](../docs/training-speed.md) |
| [`benchmarks/perf_machine.py`](benchmarks/perf_machine.py) | fixed 200-epoch throughput per machine and device | [`docs/perf/`](../docs/perf/) |

`perf_machine.py` deliberately depends on nothing but the installed package.
A user can download it and run it on a machine without a checkout. It therefore
carries its own copy of the bimodal DGP. `benchmarks/tests/` pins that copy to
the maintained generator. A drifted copy can make the collected
`final_val_nll` values incomparable.

Runtime and the CI deviations: the triangle configs run batch 256 / lr 0.004
for 300 epochs. Three variants differ:

- `linear-cs` runs 500 epochs,
- mixed `exp-cs` runs 350 epochs,
- mixed `linear-ls` runs 200 epochs @ lr 0.002.

The paper runs 500 epochs at batch 32 / lr 0.001. The tuning round of
2026-09-01 set the values above, and every ground-truth metric kept its value.
VACA and CAREFL run their references 1:1. VACA runs 10000 full-batch epochs @
lr 0.001 with the plateau rule, and CAREFL runs 7000. CAREFL runs on the
reference's own committed rows (`paper/data/carefl-cf`).
`docs/paper-replication.md` holds the selection grid, the epoch floors and the
rejected alternatives with their numbers.

To run variants locally, run one or two at a time with `OMP_NUM_THREADS=2`.
The step is overhead-bound.

`vaca.py` and `carefl.py` keep `input_transform: minmax` and tanh on their CI
terms. The reference trains in `scale_df` space. Every raw-parent alternative
measurably fails: tanh and sigmoid saturate, and relu wanders or underfits.
`docs/paper-replication.md` holds those measurements. The reference of the
triangle scripts fits raw parents, so those specs leave `input_transform`
unset.

[`paper/PAPER_COVERAGE.md`](paper/PAPER_COVERAGE.md) lists which paper figure
each variant reproduces. It also lists what the replications deliberately do
not reproduce. [`docs/paper-replication.md`](../docs/paper-replication.md)
holds every hyperparameter with its source in the R code, the deviations and
the measured numbers.

All five scripts have the same shape:

- imports,
- function definitions,
- a `run(variant)` function that holds the whole experiment,
- a `__main__` block whose argparse call selects the variant.

## The blueprint: spec and hyperparameters live in YAML, not in code

Each script reads its sibling `<script>.yaml` and **nothing else**. There are
no defaults in the code. There are no CLI flags that change a number.
`common.py::load_variant` parses the file and gives the variant's section.
That function lives here, so the package depends on no config parser and
ships no config helper.

A value that several variants share appears once under a YAML anchor. The
variants merge it with `<<`, which keeps the merge visible in the file.

Since 2026-09 every variant carries the **whole model and training recipe**.
This form is deliberately verbose. It prefers duplication over indirection, so
one variant reads top to bottom. A variant holds these keys:

- `spec:` — the full DAG in `tramdag.spec_from_dict` form. Per node it gives
  `kind` (+ `levels` for ordinal) and one `{term, parents, options}` entry per
  term. Every part of the model is a term option here, not a separate config
  key:

  - the transform,
  - `n_coeffs`,
  - `range_q`,
  - the network `units` and the network `activation`,
  - `input_transform`.
- `flow_kwargs:` — the script passes these verbatim to
  `CausalFlowDAG(spec, **flow_kwargs)`. The keys are `seed` and `init`.
- `fit_kwargs:` — the script passes these verbatim to
  `flow.fit(train, **fit_kwargs)`. The keys are `epochs`, `batch_size` and
  `seed`. The `learning_rate` key and the `schedule` and `plateau_*` keys stay
  top-level. They configure the optimizer and the scheduler that the script
  builds, not `fit` itself.
- everything else is data configuration and scoring configuration, for example
  `n_train`, `dgp_seed` and the grids.

`benchmarks/bench_training.yaml` is workloads-shaped rather than
variants-shaped, because its subject is a recipe grid. It follows the same
rule. These parts live in the YAML file:

- the specs,
- the data descriptors,
- the targets,
- every recipe number.

The script is harness only. The one exemption is `perf_machine.py`. That file
is deliberately a single curl-and-run file with no sibling file of any kind.

To change what a run does, edit the YAML file. To add a variant, add a section.
`argparse` then finds the new section automatically, because its choices come
from the file.

## Ground truth

`<area>/ground_truth/<result-dir>.json` holds one entry per checked metric, in
one of two forms:

```json
{"beta12": {"value": 1.9825, "atol": 0.05},
 "cs_curve_max_abs_err": {"max": 0.23}}
```

`{value, atol}` is two-sided. It fits a quantity that must stay where it is.
`{max}` is an upper bound for an **error measure**. For an error measure a
smaller number is a better fit, not a drift. A smaller number must not fail the
run. Tolerances are per metric, because torch results differ slightly across
operating systems and CPUs.

`check.py` fails on a metric outside its tolerance. It also fails on a metric
above its bound. It also fails on a ground-truth entry that the run no longer
produces.

A `{max}` bound is only informative in a band. Below **1.5x** its measurement
the bound fails on another machine for no reason. Above **4x** the bound cannot
catch a regression. A `{value, atol}` center decays the other way. Such a
center still passes, but it describes an older run.

`check.py` reports two conditions as notes, not as failures: a bound outside
the band, and a measurement more than half-way to its `atol`. They are notes
because a tolerance is a judgement call. A bound that is *meant* to be wide
says so in a `"why"` string. `check.py` prints that string instead of the note.
The string excuses width only. The too-tight note always fires:

```json
"max_abs_diff_flow_vs_statsmodels": {
  "max": 0.25,
  "why": "the maximum is over a coefficient with 7 of 1275 observations: 0.028 here, 0.113 on the CI runner"}
```

Ground-truth regeneration is a deliberate act. Do these steps:

1. Run the experiment.
2. Review the figures.
3. Write the new values with a commit message that says what moved and why.

## The frozen data is a contract

`<area>/data/` is committed input, not a cache. `paper/check_data.py`
regenerates every paper dataset from the seed in its `truth.json`. It compares
to 1e-9, not to bit equality, because numpy's transcendental functions move
their last bits between releases. `paper/tests/` runs the same comparison in
the ordinary test run. A new seed or a changed equation means a **new folder**,
never an edit in place.

`misc/data/magic-mrclean/ls/` is the exception with no generator here. It came
from the stroke simulator. That simulator left the repository with the clinical
storyline. `misc/tests/` pins the schema and the size of this dataset instead.
You can recover the generator from the `pre-experiments-cut` tag.
