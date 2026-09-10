# CLAUDE.md — working context for tramdag

## What this is

This repository is a causal normalizing-flow implementation of **TRAM-DAG**
(transformation models on a DAG). It builds on
[zuko](https://zuko.readthedocs.io/stable/). One triangular flow maps iid
standard-logistic latents to the observed variables. The Jacobian sparsity is
the DAG.

The package supports these features:

- the do-operator,
- Pearl abduction (counterfactuals),
- analytic interventional PMFs,
- per-node configurable monotone transforms (Bernstein, RQ-spline or affine).

**Origin:** the code came out of the private `tensorchiefs/tram-dag-stroke`
paper repository as `zuko_dag`. It became `tramdag` in June 2026, in the
repository `tensorchiefs/tramdag`. The clinical stroke storyline of that paper
lives in its own repository. That storyline is **not** here. No patient data is
here either.

**The split that matters:** `src/tramdag/` holds framework code only. Research
code lives in `experiments/`, outside the installed package. Research code
covers three groups:

- the SCM generators,
- the frozen datasets,
- the paper replications.

`pre-experiments-cut` is the tag before that separation. Use it to recover
deleted research code. The test suite does not import `experiments/`. It
measures against three inline DGPs in `tests/conftest.py`.

## Commands

```bash
uv sync                              # install (uv.lock pinned: zuko, torch, ...)
uv run pytest tests/ -q              # full suite; add -m "not slow" for the fast subset
cd experiments                       # experiments run as modules, per area
uv run python -m paper.triangle atan-cs      # one replication (config in the YAML)
uv run python -m misc.validate_ls classical  # flow == statsmodels == R polr
uv run python -m check paper triangle-atan-cs  # metrics vs committed ground truth
uv run python -m paper.check_data            # frozen data still regenerates
```

Every experiment reads its hyperparameters from its sibling `<script>.yaml`.
The code holds **no defaults**. `experiments/common.py::load_variant` parses
that file.

The blueprint (2026-09, experiments/README.md) gives each variant the FULL
model in three keys:

- `spec:` — the model in `tramdag.spec_from_dict` form,
- `flow_kwargs:` — the arguments for CausalFlowDAG construction,
- `fit_kwargs:` — the arguments for flow.fit, verbatim.

Inside `spec:`, these names are term options and not config keys:

- transform,
- n_coeffs,
- range_q,
- units,
- activation,
- input_transform.

learning_rate and schedule stay top-level, because they build the optimizer.
The blueprint is verbose on purpose, and it prefers duplication over
indirection. bench_training.yaml is workloads-shaped and follows the same rule.
perf_machine.py is exempt, because it is a single curl-and-run file.

`experiments/` splits into `paper/`, `benchmarks/` and `misc/`.
`experiments/check.py` is the shared ground-truth comparison, and
`experiments/tests/` holds its test. `paper` and `misc` each own these
directories, and `paper` also owns `simulations/`:

- `data/`,
- `ground_truth/`,
- `tests/`,
- `results/`.

`benchmarks/` measures speed on the data of the other two areas. It pins no
ground truth. It writes up its numbers in `docs/` instead. The areas share only
two files: `common.py` for the output layout and `check.py` for the
ground-truth comparison. The area tests run in the ordinary `uv run pytest`.
See `experiments/README.md`.

## Architecture (src/tramdag/)

The 1.0-RC refactor makes an effect TWO classes. That refactor covers three
references:

- the branch rc/1.0-architecture,
- docs/adr/001-term-owned-architecture.md,
- the 2026-09 term-classes revision.

The spec class is a `tramdag.Term` subclass in `spec.py`. It is frozen data,
and its annotated attributes are the options. The spec classes are:

- `Intercept`, aliased `I`,
- `LinearShift`, aliased `LS`,
- `ComplexShift`, aliased `CS`,
- `VaryingCoefficient`, aliased `VC`,
- `FnShift`, aliased `Fn`.

`Term.name` carries the symbol, and `term:` is the serialized key. The spec
class also owns the spec-level rules:

- the `__post_init__` arity and option checks,
- `edge_parents`,
- `cells`,
- `classical`.

The module class in `terms.py` declares `data = <that class>` and owns the
runtime:

- build,
- shift_value and theta_value,
- post_init,
- regularizer,
- finalize,
- score_columns,
- side inputs.

Built-ins subclass their conditioners, so checkpoints and the seeded RNG stream
stay bit-stable. `terms.module_for(term)` dispatches on `data`. Subclassing is
therefore the registration. There is no registry and no `register_term`.

Node-kind branches live ONLY in the four kind_* functions of nodes.py.
`fitting.py` and `readouts.py` are mixins that CausalFlowDAG composes. Each
method is defined once, and there is no delegate layer. The public names are:

- `flow.shift_curve`,
- `Fn`,
- `ordinal_bounds`,
- the spec exports `spec_to_dict`, `spec_from_dict`, `validate_and_sort` and
  `node_parents`,
- `effect_modifier_scan(column=)`.

docs/architecture.md carries the module map and the term-contract diagram.

- `spec.py` — the user-facing DAG spec. The spec is
  `{name: ContinuousNode|OrdinalNode}`. Each node declares its transformation
  as the first positional argument. That argument is a list of terms or a `+`
  sum, for example `I("a") + LS("b")`. The term classes are:

  - `I(*parents)` — the intercept. Without parents it is the paper's SI. With
    parents it is the CI. `SI()` and `CI(*parents)` are the arity-checked
    spellings.
  - `LS(parent)` — the linear shift.
  - `CS(*parents)` — the complex shift MLP.
  - `VC(*modifiers, t=, penalty=)` — the varying-coefficient effect head
    `beta(modifiers)·x_t`. It is a small penalized zero-init net. Read it out
    with `flow.varying_coef`. See docs/varying-coefficients.md.
  - `Fn(*parents, fn=)`.

  The paper's symbols are the classes, for example `LS is LinearShift`. There
  are no snake_case aliases. `transform=` on an intercept picks the monotone
  transform class, and **extra keyword arguments pass straight to the transform
  class**, for example `SI(transform="spline", bins=16)`. `units=[...]` on CI,
  CS or VC sizes the network of that term.

  A node takes at most ONE intercept term with parents. A **multi-parent**
  `CI("a","b")` is one *joint* network, which puts the interaction on the
  thetas. That form is the default. `CI("a","b", allow_interaction=False)` is
  the *additive* intercept, which builds one net per parent and sums the
  coefficient vectors. For shifts, the grouping decides: `CS("a","b")` is
  joint, and `CS("a")+CS("b")` is additive.

  If the effect type comes from config or from the CLI, then put the
  constructor itself in the table, for example `{"x2": CS}`. A formula without
  an intercept gets `SI()` prepended during normalization, so `node.terms[0]`
  is always the intercept. Every parent enters through exactly one edge-owning
  term. VC modifiers are exempt, because they can also appear prognostically.
- `transforms.py` — the monotone 1-D transforms that wrap zuko. There are three
  of them:

  - `BernsteinUT`,
  - `SplineUT`,
  - `AffineUT`.

  Each one pre-scales the data from the train `range_q`/1−`range_q` quantiles
  to [-5,5]. `range_q` is an intercept option and its default is 0.05.
  `SI(range_q=0.0)` gives the min-max `scale_df` domain of the reference
  comparisons. The transforms use zuko's own inverse with its closed-form tail.
  This module also holds the ordinal ordered-logit transform
  `P(Y<=k) = sigmoid(theta_k - shift)`, with cutpoints
  `[t0, t0+cumsum(exp(...))]`.
- `conditioners.py` — the LS, CS and intercept networks. The default widths and
  `relu` replicate the PyTorch reference `buehlpa/TramDag`, file
  `tram_models.py`:

  - `ComplexShiftDefaultTabular` 64-128-64,
  - `ComplexInterceptDefaultTabular` 8-8,
  - `n_thetas=20`.

  These defaults are **not** the paper's R nets. Every `experiments/paper/`
  config sets those R nets explicitly instead.
- `callbacks.py` — the shipped `fit` callbacks on the `Callback` base. The base
  has three hooks, `on_fit_begin`, `on_epoch_end` and `on_fit_end`, and it
  resets its state at fit begin. The module ships:

  - `EarlyStopping` — it restores the best-validation weights automatically at
    fit end, and it takes an optional stopping `patience`.
  - `PerNodePlateau` and `per_node_adam` — per-node lr decay and freezing.
    Afterwards `frozen = {node: epoch}`.

  All callbacks read `history["val"]`, which
  `fit(validation_data=|validation_split=)` fills per epoch. `fit` also records
  the rate of the optimizer per epoch in `history["lr"]`. `verbose=` owns the
  progress printing. The callbacks are optional, and `fit` itself stays one
  plain loop.
- `plots.py` — three plot functions:

  - `plot_dag(spec|flow)` — the labelled DAG, layered, with one edge style per
    effect.
  - `plot_marginals`.
  - `plot_training(frozen=)`.

  matplotlib is the optional extra `tramdag[plots]`. The module imports it on
  first call, so the package import never needs it. `plot_dag` is exported at
  top level.
- There is no `utils.py` any more. `machine_info` moved to
  `experiments/benchmarks/perf_machine.py`, next to its only caller, so the
  package holds modelling code only.
- `flow.py` — the `CausalFlowDAG` class. Its main methods are, with the full
  list in docs/code-map.md:

  - `fit`.
  - `fit_classical` — float64 full-batch L-BFGS, the exact MLE for all-`ls`
    specs.
  - `sample(n, do=, u=)`.
  - `abduct`.
  - `pmf`.
  - `density` — the continuous counterpart of `pmf`, on a grid.
  - `log_prob`.
  - `save/load`.
  - `ls_coefficients` — the LS weights only, because it skips the network
    shifts.
  - `varying_coef` — the VC read-out.
  - `scores` and `effect_modifier_scan` — the analytic per-row ∂ℓᵢ/∂θ plus the
    CUSUM modifier scan, in `scores.py`.

  The NLL decomposes per node, so one Adam fits all nodes jointly.

## Conventions that matter (easy to get wrong)

- **Latent scale**: the continuous scale is `z = h(x) + shift`, which ADDS the
  shift. The ordinal scale is `P(Y<=k) = sigmoid(theta_k − shift)`, which
  SUBTRACTS the shift. Both follow the original TRAM-DAG conventions. The tests
  pin them.
- **Parent encoding**: continuous parents enter RAW, with no standardization. A
  term-level `input_transform=` changes that. It takes one of three values:

  - "minmax", like the reference's `scale_df`,
  - "standardize",
  - a callable `fn(x, train)` over frozen train columns.

  `input_transform=` feeds the transformed parent to the *network* of that
  term, which means the CI, CS and VC modifiers. LS and the VC treatment stay
  raw either way. Ordinal parents enter one-hot, with all levels. With
  cutpoints, only the shift *differences* between one-hot levels are
  identified. Compare `w[k] − w[0]` against classical references.
- **Ordinal log-prob stays in log-space**: it uses `logsigmoid` plus a stable
  `log1mexp`, and it picks the better-conditioned side per element. The naive
  sigmoid difference saturates in float32. The gradients then become *exactly
  zero*, and a node can freeze at init forever. Do not "simplify" it back.
- **Seeding**: the weight init happens at construction. Use
  `CausalFlowDAG(spec, seed=...)`, which is the one obvious knob. As an
  alternative, call `torch.manual_seed` BEFORE `CausalFlowDAG(spec)`.
  `fit(seed=...)` seeds the minibatch shuffling only, and not the init.
- **Spline tails are slope-clamped**: zuko's RQS extrapolates with a *fixed*
  slope outside [-5,5], whatever θ is. If the true tail slope differs, then the
  model misweights the ~10% of data beyond the 5%/95% pre-scaling range. This
  is the structural reason why `spline` always trails `bernstein`. The linear
  extrapolation of `bernstein` follows the boundary derivative instead.
  docs/zuko-upstream.md writes this up. The demo notebook measures the affine
  case and links to that document for the spline case.
- **`fit` keeps the final weights and is one minibatch Adam loop**: an all-`ls`
  model then matches statsmodels and R-polr to ~1e-3. The caller owns four
  concerns:

  - validation,
  - lr schedules,
  - early stopping and best-weight restoration,
  - logging.

  The caller reaches them through these `fit` keywords:

  - `validation_data=` or `validation_split=`,
  - `validation_batch_size=`,
  - `verbose=`,
  - `optimizer=`,
  - `callbacks=`.

  `fit` fills `history["val"]` per epoch, and `verbose=N` prints every Nth
  line.

  `callbacks=` takes `Callback` instances with the `on_fit_begin`,
  `on_epoch_end` and `on_fit_end` hooks, or bare `on_epoch_end` callables. The
  epoch hooks get `(flow, epoch, opt)`. Any `True` return stops the fit.
  `on_fit_end` runs before the VC re-centering.

  `tramdag/callbacks.py` ships `EarlyStopping`, which auto-restores the best
  weights and takes an optional patience. It also ships `PerNodePlateau` and
  `per_node_adam`, which read `history["val"]`.

  `flow.calibrate(train_df)` takes the data-dependent state once, and the first
  fit calls it. Each term calibrates itself: its ranges and its
  input-transform statistics, never its weights. `init_marginals(train_df)`
  applies the calibrated start. It is always an explicit call, and nothing runs
  it for you.

  The stroke storyline produced one key empirical finding. **Flexible CI and CS
models overfit observational confounding at the MLE.** They need
best-validation weights to recover the causal effect. **All-`ls` models do
not.** Use `callbacks.EarlyStopping` for that now. See docs/fitting.md.

## Ground truth & reference numbers

Framework tests (inline DGPs, `tests/conftest.py`):

- `ls_chain` — every conditional is an exact linear shift. The outcome node is
  a proportional-odds model. The flow's MLE must therefore equal
  `statsmodels` `OrderedModel` on the same design matrix. True weights: x2←x1 +1.2,
  y←(x1 +0.4, x2 +0.6, t −0.8), cutpoints (−1.5, 0, +1.5).
- `vc_hetero` — known `beta(x) = −1 + 0.8·X2 − 0.6·X3`, with confounded
  assignment. The VC acceptance bar is corr ≥ 0.9, and the measurement is
  ≈ 0.99.
- `confounded` — constant effect τ = −1 with a quadratic prognostic part.
  Propensity centering must cut the bias of `beta_hat` by ≥ 2×, and the
  measurement is 5–10×.

Experiments (`experiments/`, seed 42 unless stated, arXiv:2503.16206). The
paper states only four training numbers: n=40000, 500 epochs, Adam, and
Bernstein order 20. The Adam lr 1e-3 is the default of the R code's
`optimizer_adam()`.
Each config follows the paper's own R code 1:1 where the framework allows.

**The triangle scripts.** One continuous Adam run with a separate validation
draw, 40k / 10k mixed. The coefficients are read after every epoch, through
`fit(callbacks=)`. The protocol is batch 256 at lr 0.004 for 300 epochs, with
these exceptions:

- linear-cs 500 epochs,
- mixed exp-cs 350 epochs,
- mixed linear-ls 200 epochs at lr 0.002.

The paper runs 500 epochs at the Keras defaults, batch 32 and lr 0.001. The
deviation buys CI runtime and keeps every metric. For the grid, the epoch
floors and the 2026-09-01 tuning round, see docs/paper-replication.md.

**The VACA and CAREFL comparisons.** Both take one full-batch step per epoch
on nTrain = 2500.

- VACA: 10000 epochs at lr 0.001, with the reference's `ReduceLROnPlateau`
  (factor 0.1, patience 50, min_lr 1e-7). This is torch's scheduler on the
  summed validation NLL, global as in `update_learning_rate`. Restored 1:1 on
  2026-09-02.
- CAREFL: the reference run 1:1 since 2026-09-03. It trains on CAREFL's own
  committed 2500 rows, frozen under `experiments/paper/data/carefl-cf` with
  xObs and the truth and prediction curves, in sd-standardized units. The
  settings are `val = train`, 7000 epochs at 0.001, the same plateau rule, and
  `range_q: 0`, which is the reference's min-max Bernstein domain. That puts
  the Fig. 6 curves on the paper's own: fig6 x4 max 0.204 against CAREFL's
  0.174. The earlier 3000@0.002-against-7000 trade-off was an artifact of the
  fresh-draw data. Minibatch and raw-parent alternatives measurably fail, as
  docs/paper-replication.md records.

**Seeds.** The triangle scripts run unseeded. The comparison scripts seed R's
RNG with 42, which torch cannot replay. Every seed here is therefore a repo
choice.

**Init follows each reference.** `init: normal` is Keras `random_normal`, the
triangle scripts' `LinearMasked` layers. `init: glorot` is Keras `Dense`,
`make_model`. Under the full-batch protocol the init decides the fit. The VACA
do(x2) errors measure that:

- 0.52/0.33/0.13 with torch's default init,
- 0.098/0.159/0.026 with glorot at the config's seed.

Both were measured on the earlier −3/−2/0 grid. On the shipped −3/−1/0 grid,
glorot scored 0.097/0.088/0.019 under the old 10000-epoch plateau protocol,
and 0.096/0.086/0.018 under the restored reference protocol.

**Known, documented deviations.**

- The triangle scripts use the 5%/95% quantiles for the Bernstein domain,
  which matches the reference. The comparison scripts use min-max, `scale_df`.
  CAREFL matches that through `range_q: 0`. VACA keeps the quantiles, because
  min-max measures worse there: 0.289/0.040/0.067 against 0.096/0.080/0.022.
- Both comparisons scale the *network inputs* min-max, through
  `input_transform: minmax` on the CI terms. Raw parents saturate the tanh
  nets: the `do(x2=-3)` error goes 0.731 → 0.098. The 2026-09-01 relu and
  sigmoid raw-parent attempts fail too.
- The intercept output layer carries no bias.
- Adam eps is 1e-8 against Keras 1e-7, which has no effect.
- No marginal init anywhere in the comparisons. `validate_ls` calls
  `init_marginals` explicitly, and the framework never does.

**Each config takes its architecture from *its own* reference script**, and the
reference uses two different scripts.

The triangle experiments come from `summerof24/triangle_structured_*.R`:

- `hidden_features_I = hidden_features_CS` = `c(2,25,25,2)` continuous,
  `c(2,2,2,2)` mixed,
- **sigmoid**, because the ReLU line is commented out,
- `len_theta = 20`.

CORRECTED 2026-09-02: the c(...) vector reads as the in and out dims around the
hidden stack. The hidden stack is (25,25) continuous and (2,2) mixed. The
earlier literal [2,25,25,2] reading put a 2-sigmoid bottleneck on the input.
Under that reading sin did not reproduce paper Fig. 18 at any protocol, at a
curve err of 1.22. The (25,25) stack lands on the figure at 0.24. It also moves
linear-cs from 0.13 to 0.025, and it removes the atan edge flattening.

The VACA and CAREFL comparisons come from `comparison/utils.R::make_model`:

- one net per node,
- `dense(10, tanh) -> dense(100, tanh) -> dense(len_theta)`,
- `M = 30`.

An earlier revision applied the triangle net to CAREFL. That cost an order of
magnitude on the counterfactual MAE for x4. The measurement used the old
18k-row protocol, the misread bottleneck net and M = 20.

Note also that `n_coeffs` counts *unconstrained* coefficients. zuko ties two
extra control points on. `n_coeffs=20` is therefore order 21, and the
reference's `len_theta=20` is order 19. The free-parameter count is what
matches.

- **Paper DGPs**: the `triangle` true coefficients are β12=+2 and β13=−0.2, and
  `linear` uses +0.3 on x2. A fitted `cs` learns −f(x2)+const. The
  `triangle-mixed` cutpoints are θ=(−2, 0.42, 1.02), from
  `triangle_structured_mixed.R`. The paper's text does not state them.

  **Ordinal sign flip**: the paper ADDS the ordinal shift and the flow
  SUBTRACTS it, so the fitted weights are −0.2 and +0.3. The C.4 odds-ratio
  check gives OR ≈ e² ≈ 7.4.

  For `vaca`, E[x3|do(x2=a)] = −0.25 + 0.25a. do(x2=−3) is off-manifold
  extrapolation, so it takes a looser tolerance.

  `carefl` trains on CAREFL's own committed rows in `data/carefl-cf`, which
  hold four items:

  - X.csv,
  - xObs,
  - the analytic truth curves,
  - CAREFL's own predictions.

  Everything there is in CAREFL's sd-standardized units, and x3/x4 are divided
  by 6.0104/1.9114. The rows are external frozen input, and there is no
  generator here. `carefl` scores the Fig. 6 curves point by point against the
  committed truth. It also scores held-out rows, which are fresh `Carefl4`
  draws scaled by the committed sds. Those rows sit next to the single xObs,
  because one point is a noisy yardstick.
- **`validate_ls`** runs on `experiments/misc/data/magic-mrclean/ls`, with seed
  7, n=1275, the full data and the final weights. There flow = statsmodels = R
  polr at three coefficients:

  - Age 0.0526,
  - NIHSSa 0.1630,
  - T −0.9424.

  The ATE is +0.1428 against +0.1428, and the true ATE is +0.132. The R
  reference is `fit_ls.R`, and it needs `tram` and `MASS`. Its outputs are
  committed under `ref_ls/`, so nothing needs R installed.
- Committed expectations live in
  `experiments/<area>/ground_truth/<name>.json`, and `check.py` enforces them.
  There are two entry forms:

  - `{value, atol}` — a two-sided check.
  - `{max}` — an upper bound for error measures, so a better fit cannot fail.

  A `{max}` bound belongs in a band from 1.5x to 4x its measurement.
  `check.py` notes a bound outside that band, unless the entry carries a
  `"why"` that says why the bound is deliberately wide. Whenever the code moves
  the centers, you must re-pin them. A stale center is how a variant ends up
  passing while it describes a model that no longer runs.

## Testing policy

- Framework tests must not depend on `experiments/`. You must validate a new
  causal feature against the known truth of an inline DGP. If no DGP fits, then
  add one to `conftest.py`. Never validate with "runs without error".
- The frozen CSVs in `experiments/<area>/data/` are a contract. **Never
  regenerate them silently.** A new seed or new equations means a **new
  folder**. `check_data.py` regenerates each CSV from the seed in its
  `truth.json`. It compares at **1e-9** and not at bit equality, because
  numpy's transcendental functions move their last bits between releases. The
  measured move was ~1e-15 after the 2026-08 dependency bump.
- `experiments/misc/data/magic-mrclean/ls` has no generator here, because the
  generator left with the stroke storyline. That directory holds frozen input
  data. If it ever needs regeneration, then recover the generator from
  `pre-experiments-cut`.
- The fit checks for the paper DGPs train on the paper protocol, which is n=40k
  with the epochs of the tuned configs, 200-500. They do not train on the
  frozen n=5k CSVs. β13 multiplies x1, whose two mixture components sit at 0.25
  and 0.73. The sd of x1 is 0.254, against 0.375 for x2 and 2.918 for x3. β13
  is therefore too weakly identified at n=5k.

## Roadmap notes

- Upstream PRs to zuko: docs/zuko-upstream.md ranks five candidates:

  - analytic Bernstein `call_and_ladj`,
  - linear spline tails,
  - a public `_constrain_theta` inverse,
  - a θ-shape docstring fix,
  - a `Logistic` distribution.
- ~~Generalize the generators beyond the stroke DAG~~ — done in June 2026 for
  the DGPs of the TRAM-DAG paper: triangle, triangle-mixed, vaca and carefl.
  Hidden confounding in the manner of DeCaFlow stays open.
- ~~Package for PyPI~~ — the package is on PyPI as `tramdag`, latest 0.3.0,
  June 2026. Since the 1.0-RC the release flow is tag-driven, by the skeleton
  convention. The version IS the git tag, through hatch-vcs. `cz bump` derives
  the tag from the conventional commits, or you tag `vX.Y.Z` by hand. A push of
  the tag runs `.github/workflows/release.yaml`, which does three steps:

  - uv build,
  - upload to PyPI through trusted publishing,
  - a sigstore-signed GitHub release.

  There are two one-time prerequisites. First, register the GitHub repo as a
  trusted publisher on pypi.org/manage/project/tramdag, which needs the PyPI
  project owner Oliver. Second, create the `pypi` environment in the repo
  settings. The CHANGELOG section stays hand-written.
- The `experiments/` tree is the candidate for a companion repository,
  `tensorchiefs/tramdag-simu`. The tree is already self-contained, so the move
  is a directory copy plus a workflow.
