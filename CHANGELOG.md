# Changelog

## 1.0.0-rc (unreleased)

0.4.0 was never released, so this section is the whole change from 0.3.0, the
last released version. It describes the shipped state only. Where an entry
below replaced an earlier attempt inside this same unreleased range, only the
result is recorded.

### Changed (breaking) — a term is a class, and the registry is gone

- `Intercept`, `LinearShift`, `ComplexShift`, `VaryingCoefficient` and
  `FnShift` are the terms: frozen dataclasses with the paper's symbols `I`,
  `LS`, `CS`, `VC`, `Fn` as aliases, not constructor functions that return one
  string-tagged `Term`. `SI()` and `CI()` are the arity-checked intercept
  spellings. There are no snake_case names.
- Options are typed fields with defaults. An option the term does not take
  fails at construction, so a hand-built or serialized term can no longer
  carry a foreign key. `Term.options` (the canonical pairs) and
  `Term.__getattr__` are gone, and `term.options()` gives the non-default
  options.
- A serialized term is keyed `term:`, not `effect:`, and the class variable
  holding the symbol is `Term.name`, not `Term.effect`. The word "effect"
  named two things, the causal quantity the model estimates and the kind of a
  term, and only the first meaning survives.
- `tramdag.terms` keeps the modules only. `register_term`, `get_term` and the
  registry are removed: a module class declares `data = <Term subclass>` and
  `module_for(term)` finds it, so subclassing is the whole registration.
- The spec-level hooks `check_arity`, `edge_parents`, `cells`,
  `term_is_classical` and `option_defaults` moved onto the term classes as
  `__post_init__`, `edge_parents`, `cells`, `classical` and fields.
- `ShiftTerm` slims down. Six paired flags and hooks become one each:

  - `regularizer()` returns `None` instead of pairing with `has_regularizer`,
  - `finalize` decides itself instead of taking a `finalizes` flag,
  - `marginal_start` is a no-op by default instead of pairing with
    `has_marginal_start`,
  - `net_parents` is gone, because nothing read it,
  - the node sets `parents` after `build`, so a custom `build` sets `key` only,
  - the summation order is the `order` class attribute, not an `isinstance`
    check.
- `term(effect, *parents)`, the string-label factory, is removed. It was a
  second and weaker way to build a term, and a generic dispatcher carrying one
  term's parameter is the shape this release removed from `Term` itself. When
  the term type comes from config or the CLI, hold the constructor in the
  table: `{"Age": I, "NIHSSa": CS}`, then `t["Age"]("Age")` (see
  `experiments/paper/helpers.py::shift_term`).
- `Term.slot` is removed. It was derived from the term kind, and its only user
  in the repository was a test assertion.
- The ADR 001 refusal of per-term classes is revised in place.

### Changed (breaking) — the node formula reads like the math

- A node's additive formula is its first positional argument and can be
  written as a `+` sum: `ContinuousNode(I("x1") + CS("x2"))`,
  `OrdinalNode(4, [I, LS("x1")])`. Everything normalizes to the same internal
  term list, and state-dict-identical tests pin the equivalence
  (`tests/test_transformation_syntax.py`).
- The formula argument is named `terms`, not `transformation`, and the
  attribute is `node.terms`. It is still the first positional argument, so
  positional calls are unaffected.
- A `+` sum nested inside a list is rejected. `+` already returns a flat list,
  so `[I("x1") + LS("x2")]` was a list of lists that the normalizer silently
  flattened. The error says so, because the usual cause is expecting `+` to
  combine list entries.
- `SI()` is prepended to a formula that has no intercept, so `node.terms[0]`
  is always the intercept. `ContinuousNode()` and `OrdinalNode(k)` hold
  `[SI()]` rather than `None`, which also makes node specs hashable: a set of
  nodes used to raise `TypeError`.
- `transform=` moves onto the intercept term, `I("x1", transform="spline")`,
  and extra keyword arguments pass straight to the transform class,
  `I("x1", transform="spline", bins=6)`. The `transform_kwargs=` constructor
  keyword and the node-level `ContinuousNode(transform=/transform_kwargs=)`
  are gone. The canonical storage inside the term is unchanged, so serialized
  specs are unaffected.
- `Intercept`, and therefore `I`, `SI` and `CI`, names every option in its
  signature instead of taking them through `**options`. Every call spelling is
  unchanged, including the pass-through: a keyword that is not an option still
  reaches the transform class (`SI(transform="spline", bins=16)`,
  `I(n_coeffs=40)`), and a written-out keyword still wins over the same key
  inside a serialized `transform_kwargs` mapping. `CS`, `VC` and `Fn` are
  untouched.
- `Term.__init_subclass__` refuses a term whose `__init__` default disagrees
  with its field default. Without that check the disagreement would silently
  change what `Term.options()` writes into a checkpoint, and a round-trip test
  cannot see it, because both sides of the round trip carry the same drift.
- `allow_interaction=False` makes a multi-parent intercept additive: one net
  per parent, their coefficient vectors summed, written
  `CI("a", "b", allow_interaction=False)`. A node takes at most one intercept
  term with parents. `CI("a", allow_interaction=False)` now raises, because
  with one parent there is no interaction to disallow. It used to be silently
  coerced to a joint net.
- `VC(*modifiers, t=...)`: the positional arguments are the covariates that
  enter `b_theta` and the treatment `t` is a required keyword, so
  `VC("X2", "X3", t="T")` reads as `(beta0 + b_theta(x2, x3)) * x_t`. 0.3
  wrote `VC("T", "X2", "X3")`.
- `units=` on `I`, `CS` and `VC` sizes that term's network, for example
  `units=[16]` for one hidden layer. The defaults are `[8, 8]` for `I`,
  `[64, 128, 64]` for `CS` and `[16]` for `VC`, serialized per term. All
  conditioner networks build through one `_mlp()` helper.
- A wrong-term option errors at construction instead of silently defaulting,
  `spec_from_dict` rejects stale or misspelled option keys, and `node_terms`
  is gone because `node.terms` is canonical.

### Changed (breaking) — `fit` is one loop, the strategies are yours

`CausalFlowDAG.fit` shrank from 18 keyword arguments to a minibatch Adam loop
with an `optimizer=` hook and one `callbacks=` list, and the shipped recipes
moved into their own `callbacks.py`. Three independent reviews of the file
agreed on the cut: what left was training *strategy*, tuned on this
repository's own DGPs, and not the TRAM-DAG model. Each of them also had a
default that the paper replication or the tests had to switch off.

- **`fit(train_df, *, epochs, learning_rate=1e-2, batch_size=512, seed=None,
  optimizer=None, callbacks=None)`**, plus the validation and progress
  keywords below. One optimizer covers all parameters, which equals per-node
  training because the per-node NLLs have independent gradients.
  `optimizer=` takes any torch optimizer, which is how a
  `torch.optim.lr_scheduler` attaches. `flow.history` holds the per-node train
  NLL per epoch, and the per-node validation NLL under `"val"` when
  validation is configured.
- **`callbacks=`** takes one entry or a list. A `tramdag.callbacks.Callback`
  hooks `on_fit_begin(flow, optimizer)`,
  `on_epoch_end(flow, epoch, optimizer)` and `on_fit_end(flow, optimizer)`,
  and a bare callable is an `on_epoch_end` hook. Every epoch hook runs after
  every epoch, and the fit stops after an epoch in which any returned `True`.
  `on_fit_end` runs before the VC re-centering, so restored weights get
  re-centered.
- **`fit(epochs=)` is required.** There is no default training budget.
  `docs/training-speed.md` measures a fixed budget going wrong in both
  directions on this repository's own workloads: the stroke fit converges
  after ~1500 of 4000 budgeted epochs, and the vaca fit is 0.03 nats short at
  520. A package-level number could only be arbitrary. A generous budget with
  a stopping callback is the alternative.
- **Gone from `fit`, with what replaces each one.** `docs/fitting.md` shows
  them all.

  - `schedule="plateau"`, `plateau_patience`, `plateau_factor`,
    `plateau_min_lr` and `min_delta` are torch's `ReduceLROnPlateau` on the
    optimizer you pass. The paper's VACA and CAREFL replication therefore
    runs the reference's *global* rule, instead of the per-node approximation
    that was a documented deviation.
  - `freeze_patience` and the per-node freeze are back as the opt-in
    `tramdag.callbacks.PerNodePlateau`.
  - `restore_best` is `tramdag.callbacks.EarlyStopping`, or a six-line
    snapshot callback.
  - `epoch_callback` is an entry in `callbacks=` that receives the optimizer.
  - `marginal_init` moved to `calibrate` and then out again. See the
    calibration section.
  - `vc_warm_start` and the hidden classical proxy fit are two lines of user
    code (`docs/varying-coefficients.md`). Measured on `vc_hetero`, `beta0`
    lands 0.15 from the truth with the warm start and 0.16 without it, at a
    recovery correlation of 0.99 either way.
  - `vc_oof_fit` and the hidden five-fold stage-1 fits are replaced by the
    propensity column below.
  - `verbose` and the `tramdag.flow` logger both went. `verbose=` returned in
    Keras shape and the module logger stayed gone.
- **No progress output from the package.** `fit` and `fit_classical` print and
  log nothing of their own. `verbose=N` prints every Nth epoch plus the final
  one, `0` being the default and silent, with no progress bars. A bare
  `on_epoch_end` callable reports whatever else you want, and `fit_classical`
  returns its report dict.
- **`fit_classical`** lost `verbose`, because the report dict is the summary,
  and computes its gradient norm with `torch.nn.utils.get_total_norm`. It
  records that report, minus the coefficients, in
  `flow.history["classical"]`, so a classically fitted checkpoint says how it
  was fitted. `fit_classical(history_size=50)` exposes the L-BFGS memory.
  `chunk=` is gone: L-BFGS runs once with torch's own `tolerance_change` (the
  `tol` argument, now 1e-9) instead of in hand-rolled rounds with a relative
  NLL-flatness test, and `n_iter` is torch's count. Measured on the classical
  anchor, 1e-9 reproduces the old convergence exactly, with coefficients
  within 0.0027 of statsmodels, while 1e-6 would stop on a plateau step.
- **Every query and read-out is `(df, node, *, ...)`.** `varying_coef` and
  `intercept_contributions` took the node first and called the frame `data`,
  and they now match `pmf`, `density`, `scores`, `design_matrix` and
  `effect_modifier_scan`. `do=`, `seed=`, `t=` and `candidates=` are
  keyword-only, as every `fit` knob already was.
- **`sample` returns ordinal columns as int64 level indices**, which they were
  not, matching what `fit` requires on the way in. **`log_prob` runs under
  `no_grad`** like every other query. Differentiate through `node_log_prob`
  instead. **`abduct` keeps the caller's index.**

### Changed (breaking) — no magic: calibration is per term, the start is yours

- **`calibrate` never touches the weights.** The `marginal_init` parameter is
  gone. `flow.init_marginals(train_df)` is the one way to a calibrated start
  and nothing calls it for you, not the first `fit` and not `fit_classical`. A
  recipe tuned with the warm start now says so in its own code, as
  `validate_ls` does. With `range_q=0`, a min-max domain, the marginal start
  is undefined and `init_marginals` raises instead of silently retargeting the
  5% quantiles.
- **Each term calibrates itself.** `calibrate` loops over the node's terms.
  The intercept term maps its own column's `range_q` quantiles onto the
  transform domain, and every term freezes its own `input_transform`
  statistics. The `_InputTransform` modules move from the node's
  `input_transforms` ModuleDict onto the owning term, so state-dict paths
  change: `nodes.X.input_transforms.@I.*` becomes `nodes.X.intercept.*`, and
  the shift keys likewise.
- **One `calibrated` buffer** replaces the three per-module first-fit latches
  (`ut._fitted`, `_net_scaled`, `_marginal_inited`) that `load` had to
  re-close by hand.
- **Degenerate columns fail loudly instead of training into NaN.**
  `calibrate` raises on two column shapes that used to run and produce `nan`
  or silently truncated levels. The first is a continuous node whose 5% and
  95% quantiles coincide, a constant or 95%-constant column with no transform
  domain. The second is an ordinal column that is not an integer level index
  in `0..levels-1`. Four more failures are named now:

  - an unknown `activation=` or `init=` raises with the valid choices,
    instead of a bare `KeyError` or a silent fallback,
  - `fit(batch_size=)` below 1 raises with a plain message, instead of
    reaching `range()` with a cryptic one,
  - a frame that lacks a spec column fails in `_tensorize` with a `KeyError`
    that names the columns, instead of deep inside a tensor op,
  - an ordinal value that is not a level index is refused at every entry
    point, by node name. Inference used to truncate `1.5` to level 1 in
    silence, and to let torch raise an unnamed error for an out-of-range
    value.
- **Misconfiguration fails plainly.** `PerNodePlateau` refuses an optimizer
  whose groups lack the `initial_lr` stamp, which `per_node_adam` builds,
  instead of adopting a possibly-decayed current rate. `VC`'s serialized
  penalty default now matches the constructor's 1.0, where it was a phantom
  `None` that a hand-written spec without `penalty:` tripped over, and
  `penalty: null` is rejected as not a number.

### Changed (breaking) — term-owned architecture (docs/adr/001, docs/architecture.md)

The 0.3 monolith split along its seams, and every per-term behavior moved onto
the term classes in the new `terms.py`. No term string-switch survives outside
`nodes.py`. Public behavior, error messages, sign conventions, seeding and all
ten experiment ground truths are unchanged, verified per step by a seeded
state-dict smoke and by replications. The fit API, the queries and the
read-outs keep their exact signatures as flow methods.

- Four new modules:

  - `nodes.py` holds the node model and the ONLY four continuous/ordinal
    branches: `kind_log_prob`, `kind_sample`, `kind_abduct` and
    `kind_marginal_theta`,
  - `fitting.py` holds `_FitMixin`, which is fit and fit_classical,
  - `readouts.py` holds `_ReadoutsMixin`, the stateless read-outs,
  - `terms.py` holds the term modules LSTerm, CSTerm, VCTerm, FnTerm, SITerm,
    CITerm and AdditiveCITerm, on the `ShiftTerm` and `InterceptTerm` hooks.
- `fitting.py` and `readouts.py` are mixins that `CausalFlowDAG` composes,
  with every method defined once and no delegate layer.
- **A centered VC names its out-of-fold propensity COLUMN.**
  `VC("X", t="T", center="ps")` reads column `ps` of the training frame
  (`df.assign(ps=e_oof)`), which splits, shuffles and minibatches like any
  data. Queries still recompute the propensity live from the treatment node.
  `center=True` refuses with the new spelling named, and no term-specific
  argument crosses `fit` any more.
- The internal legacy views are gone: `_VCGroup`, `node.vc_column` (now
  `VCTerm.regressor`), `node._vc_groups`, `node._shift_groups`, and the
  `ci_parents` and `_intercept_groups` pass-through properties. Read
  `node.intercept.ci_parents` and `.groups`.
- `flow.shift_curve(node, parent, grid)` is a flow METHOD and evaluates
  through the term's own `shift_value`, so `Fn` and custom terms plot too. It
  replaces reaching into `nd.shifts[..]` and `net_input`.
- `_fit_validated` is set `False` by `fit_classical`, so a callback cannot
  read a pre-classical validation entry.
- Additive-CI checkpoints re-key, `nodes.*.intercept_nets.*` becoming
  `nodes.*.intercept.nets.*`. The parameters are bit-identical under the
  rename.

### Changed (breaking) — the paper replications follow the R code 1:1

- **Triangle**: one continuous Adam run with a separate validation draw, and
  the coefficients read after every epoch through `fit(callbacks=)`. Gone: the
  90/10 split, `chunk_epochs` (a fresh Adam per chunk, which is a warm-restart
  schedule the reference never had), the `polish_*` phase and `fit_in_chunks`.
- **VACA and CAREFL**: nTrain 2500, one full-batch step per epoch, 10000 and
  7000 epochs at lr 0.001, with the reference's `ReduceLROnPlateau`
  (factor 0.1, patience 50, min_lr 1e-7) on the summed validation NLL of a
  separate 5000-row draw.
- **The CI protocol trades runtime for the same metrics.** Triangle runs 300
  epochs at batch 256 and lr 0.004, against the paper's 500 at batch 32 and lr
  0.001. linear-cs keeps 500, because its cs curve needs them, and mixed runs
  350 for exp-cs and 200 at lr 0.002 for linear-ls. `validate_ls`
  descends in 800/700/500 epochs at 1e-2/1e-3/1e-4, cut from 4000/2000/1000:
  the same MLE, a named-coefficient gap to statsmodels of 1.6e-5 against
  1.8e-5, and about 3.5x less wall clock. Every ground-truth bound holds under
  30 to 71% less training than the paper's own budget. Every ground-truth file
  is re-pinned from these runs, and the deviations that remain are
  framework-level and listed in each YAML, in `docs/paper-replication.md` and
  in CLAUDE.md.
- **Two attempted deviations are documented negative results.** Raw network
  inputs fail on VACA and CAREFL, where sigmoid saturates like tanh and relu
  underfits and grows fragile Bernstein tails. Minibatching biases the
  interventional decomposition, at a −0.11 common-mode offset of the do-means
  on a perfect observational fit. Both keep tanh with
  `input_transform: minmax`.

### Removed (breaking)

- **All backward compatibility.** Pre-1.0 there is one API and one checkpoint
  format. The lowercase `"ls"`, `"cs"` and `"ci"` aliases are gone, so a
  checkpoint carries `"I"`, `"LS"`, `"CS"` and `"VC"`. The two 0.3-checkpoint
  shims in `spec_from_dict` (the multi-`I` merge and the node-level-basis
  carry) are gone with the redundant node-level `transform` and
  `transform_kwargs` keys that fed them. The 0.3-spec detector is gone too: a
  spec without an `"options"` mapping fails as malformed input. `load`
  requires a complete checkpoint (spec, weights, history, meta) instead of
  tolerating missing blocks, and it no longer defaults a missing `init` field
  to `"torch"`. The 0.3 `Intercept`, `LinShift` and `CShift` aliases and the
  `parents={...}` checkpoint loader are gone. `Intercept` is now the full
  class name of `I`. **Checkpoints and serialized specs from any earlier
  version do not load. Refit and save again.**
- **`tramdag.simulations` is no longer part of the package.** The SCM
  generators are research code and moved to `experiments/paper/simulations/`,
  with the frozen datasets moving from `data/` to `experiments/paper/data/`.
  The wheel contains framework code only, and `import tramdag.simulations`
  fails. The stroke storyline left with them: the `magic_mrclean` generator,
  the `magic-mrclean/nl` cohort, `sim_flow.py`, the `vc_shift` DGP,
  `experiments/stale/` and `docs/stroke-case-study.md` are deleted, and the
  clinical case study lives in its own repository.
  `experiments/misc/data/magic-mrclean/ls` stays as the frozen input of
  `validate_ls`. **Everything deleted is recoverable at the
  `pre-experiments-cut` tag**: `git checkout pre-experiments-cut -- <path>`.
- **`tramdag.utils` is gone.** `machine_info` moved to
  `experiments/benchmarks/perf_machine.py`, next to its only caller.
  `config_section` and its key-set check are deleted outright: a script's YAML
  is the single source of its numbers, and the check bought a second copy of
  every key list in code. The installed package is modelling code only.
  `save` no longer records the machine, and `meta` keeps the version, the time
  and the device.
- **The `"onecycle"` and `"cosine"` schedules.** They had no caller outside
  one parametrized test, and the June 2026 benchmark measured both behind
  plateau on every workload. A schedule is now a torch `lr_scheduler` on the
  optimizer you pass.
- **The `bound` knob on the univariate transforms.** Nothing ever set it, and
  the pre-scaled domain is fixed at `[-5, 5]` (`transforms.BOUND`).
- **The unmaintained notebooks and experiment scripts.** They moved to
  `notebooks/stale/` and `experiments/stale/` and were then deleted. Two demo
  notebook sections went with them, the misspecification excursion and the
  GPU-versus-CPU race, so the demo stays a five-minute walk.
- **`docs/research/`** and the autoresearch measurement guard it documented.
- **The observational ITE study** (the `ITEObservational` DGP, its didactic
  notebook and the train-size experiment) moved to the simulation-study
  companion repository `tensorchiefs/tramdag-simu`. The package keeps only
  DGPs that validate the implementation, and a benchmark comparing tramdag to
  other methods belongs in the method-neutral paper repository.

### Added — `tramdag.plots` (optional extra `tramdag[plots]`)

- `plot_dag(spec | flow)`: the labelled DAG, layered left to right, with one
  edge style per term (LS, CS, CI, VC with modifiers, Fn, and `joint` for a
  multi-parent net). Exported as `tramdag.plot_dag`. `plot_marginals(flow,
  df)` and `plot_training(flow, frozen=)` join it. matplotlib is imported on
  the first call, so the package import never needs it.
- `fit` records the optimizer's learning rate after every epoch in
  `history["lr"]`, a `{node: lr}` mapping with `per_node_adam`'s tagged groups
  and a float otherwise. `plot_training` reads the freezes off it, so no
  tracer callback is needed.
- `PerNodePlateau.frozen` is `{node: epoch}`, the epoch each node left
  training, rather than a bare set, so a training figure can mark the freezes.
  `step(nll, opt, epoch=None)` records it, and `len` and `in` are unchanged.

### Added — `tramdag.callbacks`, and Keras-shaped validation

- **`tramdag.callbacks`** ships the common training recipes as opt-in
  callbacks on a small `Callback` base (`on_fit_begin`, `on_epoch_end`,
  `on_fit_end`, each resetting its state at fit begin so instances are
  reusable). `EarlyStopping` keeps the best-validation weights and restores
  them automatically at fit end, which is the recipe the stroke finding needs,
  since flexible CI and CS models overfit confounding at the MLE. An optional
  `patience=` also stops the fit once the best epoch is that many epochs old,
  and `restore_best=False` keeps the final weights. `PerNodePlateau` with
  `per_node_adam` decays the learning rate of each node and freezes converged
  nodes, off that node's own validation NLL, with `patience` and `freeze`
  defaulting to the benchmark's 15 and 50. All of them read `history["val"]`,
  so a fit that registers them needs `validation_data=` or
  `validation_split=`, and the NLL is computed once and shared. `fit` itself
  stays one plain loop.
- **Validation and progress in `fit`.** `validation_data=` takes a DataFrame
  and `validation_split=` takes a float, the LAST fraction of `train_df`,
  unshuffled like Keras, with only the head training and calibrating. Either
  makes `fit` compute the per-node validation NLL once per epoch into
  `flow.history["val"]`, in `validation_batch_size=` chunks.

### Added — spec, terms and read-outs

- **`I`, `CS` and `VC` take `batch_norm=`**, off by default, which puts a
  `BatchNorm1d` between each hidden layer and its activation. Every knob of
  the term networks is reachable from a serialized spec.
- **`range_q`, the transform-domain quantile, is an intercept option.**
  `SI(range_q=0.0)` on any intercept and any basis calibrates that node's
  domain to the train min and max instead of the default 5% and 95%
  quantiles, which is the original implementation's `scale_df` scaling
  exactly. The option rides the transform-kwargs path, serializes with the
  spec, and validates to `[0, 0.5)`. The CAREFL replication uses it, where it
  measured Fig. 6 max errors of 0.37 down to 0.20 for x4 and 0.20 down to
  0.07 for x3. VACA measured worse with it and keeps the quantile default, so
  the tails matter per problem, which is why the option is per node.
- **`CI/CS/VC(input_transform=)`**: every term's network chooses its own input
  transform, one of `"minmax"`, `"standardize"`, or a callable `fn(x, train)`
  that receives the frozen raw training column and never the batch's, since
  batch statistics would encode the same value differently per call.
  Statistics freeze at `calibrate` and live in per-term buffers, so they
  checkpoint, and a callable must be a picklable module-level function to
  `save`, with a lambda raising instructions. Linear shifts and the VC
  treatment stay raw, so `ls_coefficients` keep their units. Default off. Raw
  parents saturate a tanh net whenever `|x| > 2`, which is 40% of the VACA
  rows, and that put the VACA and CAREFL replications 2 to 20 times off the
  paper until the minmax transform: the `do(x2=-3)` error went 0.731 to 0.098.
- **`CausalFlowDAG(spec, init="glorot")`** is Keras' `Dense` default
  initialization, glorot-uniform weights with zero biases, for every linear
  layer, which is the paper's reference. The default stays torch's
  Kaiming-uniform. Under the reference's full-batch VACA and CAREFL protocol
  with its global plateau rule the init decides the fit. The `do(x2)` errors
  are 0.52 / 0.33 / 0.13 with torch's init and 0.098 / 0.159 / 0.026 with
  glorot at the config's seed. Another init draw gives 0.035 / 0.006 / 0.007,
  and Adam's eps makes no difference. `init="normal"` is Keras'
  `RandomNormal`, sd 0.05 on weights and biases, which is the initializer of
  the paper's triangle scripts. The VACA and CAREFL configs set glorot and the
  triangle configs normal. Stored in the checkpoint.
- **`init_marginals(train_df)`** is the calibrated start as an explicit,
  repeatable step. It resets every unconditional intercept to its column's
  marginal. For Bernstein that is the approximation of `logit(F_hat(y))`, the
  same quantity the ordinal cutpoints carry. For ordinal it is the empirical
  class log-odds. Spline and affine have no calibrated start. It is a pure
  initialization and the converged MLE is unchanged. Unlike `calibrate` it is
  not once-guarded, so a loaded or already-trained flow can be restarted at
  the marginal, and an uncalibrated flow takes its ranges from the same rows
  first.
- **`density(df, node, grid, do=)`** is the analytic conditional density of a
  continuous node on a grid, the continuous counterpart of `pmf`. It is closed
  form from the transform with no sampling, and it is pinned against
  `exp(log_prob)` at the observed value, unit mass, and `do=` being
  equivalent to column substitution.
- **`log_prob(df, nodes=[...])`** sums a subset of the nodes, which is exact
  and is the log-space way to read one node's conditional likelihood per row.
- **`flow.scores(df, node)` and `flow.effect_modifier_scan(df, node, t)`**
  (issue #29) give the per-observation scores ψᵢ = ∂ℓᵢ/∂θ for every `LS`
  weight and every `VC` `beta0`. They are **analytic and exact**: shifts enter
  the latent additively, so ∂ℓᵢ/∂β = (∂ℓᵢ/∂sᵢ)·xᵢ with the latent derivative
  in closed form, pinned by a float64 finite-difference test and by the
  scores-sum-to-≈0-at-the-MLE property. The Zeileis and Hornik fluctuation
  scan is packaged on top: order the treatment coefficient's scores by each
  candidate covariate, then compare `sup|CUSUM|` against the Kolmogorov 5%
  critical value to rank candidates for `VC` modifiers, from a seconds-long
  all-`ls` `fit_classical`. It is a pure read-out that touches no fitting or
  sampling path. End to end, the scan flags the true X2 and X3 modifiers of a
  heterogeneous-effect SCM and not the inert prognostic X1. `column=` scans a
  named level contrast of a multi-level ordinal treatment. Docs:
  `docs/scores.md`.
- **`VC(*modifiers, t=, penalty=)`, the varying-coefficient shift term**
  (issue #28) is a treatment-effect head `beta(x) = beta0 + b_theta(x)`. Its
  small 16-unit **penalized** zero-initialised network only multiplies `x_t`.
  The first-class read-out `flow.varying_coef(df, node)` comes with it, and it
  is closed-form, deterministic and y-free, and equals the abduct difference
  for binary treatments. The objective is the penalized likelihood
  `Σ NLL + penalty·‖w‖²` on the total-NLL scale, with `beta0` unpenalized.
  `penalty → ∞`, or `modifiers=()` exactly, nests `LS(t)`. VC modifiers may
  also appear in prognostic terms, since only the treatment owns its edge.

  Measured motivation: the `CS(t, x…)` reduced form is *expressive but
  unestimated*. It reaches a correlation of only ≈ 0.5 against the true
  effect function even in-class (tramdag-simu#18, PR #21), because nothing in
  the NLL rewards a smooth arm difference. The regularized head reaches
  ≈ 0.99 on the same task class. The validation DGP has a known
  `beta_true = −1 + 0.8·X2 − 0.6·X3` and is the inline `vc_hetero` DGP in
  `tests/conftest.py`. Acceptance in `tests/test_vc_term.py` is a recovery
  bar of corr ≥ 0.9 at n = 5000, measured at 0.986 as the minimum over seeds.
  Docs: `docs/varying-coefficients.md`.
- **Propensity-centered VC: `VC(..., center="col")`** (issue #30) is the
  R-learner orthogonalization `beta(x)·(t − ê(x))` inside the likelihood, as a
  **two-stage frozen** design. Training uses **out-of-fold** ê, one value per
  training row, read from the named column. That is the DML cross-fitting
  requirement. The caller computes them in six lines with `fit_classical`
  (`docs/varying-coefficients.md`). They are frozen as data, with zero
  gradient into the treatment node from the outcome loss, which is tested.
  Inference recomputes ê from the flow's own fitted treatment node and
  re-derives `t − ê(x)` under `do`, never cached, which is tested on fresh
  rows. Binary ordinal treatments only, and `center=False`, the default, is
  bit-identical to the uncentered term, which is tested. Measured on the Dandl
  et al. 2024 reproduction (`tests/test_vc_centered.py`): under strong
  confounding with an under-specified prognostic part, centering cuts β̂ bias
  **5 to 10 times**, from 1.10–1.24 down to 0.11–0.27 over three seeds.
- **`flow.intercept_contributions(df, node)`** (issue #20, Option A) is a
  post-hoc, mean-centered decomposition of an **additive complex intercept**
  (`CI("x1", "x2", allow_interaction=False)`). The per-term networks are
  summed in unconstrained parameter space, so the sum is identified but each
  term's contribution only up to a constant. It returns each term's
  **sum-to-zero** contribution to the transform parameters, GAM-style
  mean-centered over the frame, plus the absorbed `baseline`, for plotting
  per-parent partial effects. It is exact, with
  `baseline + Σ contributions == theta`, and purely interpretive: it reads the
  fitted weights and changes nothing about the model or any frozen number.
  Shift terms remain a separate slot (`ls_coefficients`).
- **`Fn`/`FnShift`** puts a callable or a trainable `nn.Module` into the
  additive shifts, and a whole custom term is a `Term` subclass plus a
  `ShiftTerm` subclass declaring `data =`. **`transforms.ordinal_bounds`** is
  public. **`I(transform=)` accepts a `_ScaledUT` subclass.**
  **`node_parents`, `validate_and_sort`, `spec_to_dict` and `spec_from_dict`
  are exported.**
- **Paper-aligned intercept constructors**: `SI` for the parentless baseline
  and `CI`, which needs at least one parent, matching the paper's notation.
  `I` stays as the fallback and dispatches on its arguments, and both `I` and
  `SI` work as bare names in a term list.
- **Specs serialize YAML-readably and round-trip through JSON.**
  `spec_to_dict` emits nested kwargs tuples (`transform_kwargs`) as mappings,
  and `spec_from_dict` turns the lists JSON gives back into the tuples `Term`
  stores. A spec written by hand in YAML, or saved with `json`, therefore
  compares equal and hashes the same as the one written. This is what the
  experiments'
  blueprint builds on: every experiment variant carries its full DAG as a
  `spec:` block in its sibling YAML (`experiments/README.md`).
- **`src/tramdag/py.typed` ships (PEP 561).** The package is fully annotated,
  and pip users' type checkers now see the inline types.

### Added — the experiments tree, and its gates

- **The experiments are self-contained and configuration-driven.** One script
  per dataset (`triangle`, `triangle_mixed`, `vaca`, `carefl`, `validate_ls`),
  each with the same shape: imports, function definitions, a `run(variant)`
  function, and a `__main__` block whose argparse call selects the variant.
  Each reads **every** hyperparameter from a sibling `<script>.yaml`, and
  `experiments/paper/tests/test_configs.py` checks that every key in a variant
  is read, so a leftover key cannot quietly become a default.
  `experiments/paper/PAPER_COVERAGE.md` maps every figure of arXiv:2503.16206
  to the variant that reproduces it, including the paper's misspecified case
  (Fig. 17, the `triangle linear-cs` variant) and the two competing baselines
  that are deliberately not reimplemented.
- **An experiments workflow** (`.github/workflows/experiments.yaml`) runs all
  replication variants as a matrix on every push and on demand. It compares
  each run's `metrics.json` against the committed
  `experiments/<area>/ground_truth/<name>.json`, which holds a
  `{value, atol}` entry per metric or `{max}` for an error measure. It then
  posts the run's report, a metrics table plus figures, as a commit comment
  through CML. Where an
  analytic or generator truth exists, the table carries `DGP truth` and
  `abs. error` columns next to the fitted value. Every run records
  `fit_seconds` against a `{max, why}` tripwire pinned at three times the
  CI-runner measurement, which is a gross-regression alarm such as a lost
  `no_grad`, and not a benchmark. `experiments/paper/check_data.py` verifies
  that every frozen dataset still regenerates from its stored seed, at 1e-9
  rather than bit equality.

### Added — notebooks and guides

- **`notebooks/classical_fit_tram_dag.py` is back**, and it is the only place
  `fit_classical` is walked through end to end. It reads the stroke cohort
  from `experiments/misc/data/`, simulates the VACA triangle inline as the
  other notebooks do, and uses `flow.design_matrix(..., drop_first=True)`
  rather than a hand-rolled one-hot design. It joins the docs workflow's
  executed list, so it runs on every push to `main` and `dev-*`.
  - **Logistic regression is the zeroth example.** A two-level `OrdinalNode`
    with `LS` terms *is* logistic regression,
    `logit P(Y=1) = -theta_0 + w_0 + sum_p w_p x_p`. It runs on
    `MASS::birthwt` (the new `notebooks/data/`, exported verbatim from MASS,
    with a pasteable `glm` snippet that needs no file because the data ships
    with MASS) and agrees with `statsmodels.Logit` and R `glm` on all four
    coefficients to ~1e-8, on the log-likelihood to 1e-5, and on the fitted
    probabilities to 9e-8. It makes two conventions concrete on the simplest
    possible model: an ordinal node *subtracts* its shift, and an ordinal
    parent's one-hot level-0 column is part of the intercept.
  - **Section 1 opens on a real continuous outcome.** `birthwt.csv` gains
    `bwt`, the birth weight in grams that Section 0's binary `low` is cut
    from, so the same three predictors are fitted continuously and checked
    against R `tram::Colr` at matched degree. The `smoke` log-odds ratio comes
    out at +0.664 from the flow, +0.669 from `Colr` and +0.671 from Section
    0's `glm` on the dichotomized outcome. That is the proportional-odds
    property made concrete: a linear shift moves the whole latent
    distribution, so cutting the outcome at 2500 g discards information about
    `h` but leaves the shift alone.
  - **Section 1 reproduces the continuous fit outside the flow too.** R
    `tram::Colr` is shown with its real output rather than as untested code,
    and `statsmodels` gets a route it was previously said to lack: a
    continuous transformation model is the limit of an ordered logit, so
    cutting the outcome into `K` quantile bins and fitting `OrderedModel`
    converges to the flow's shift coefficients. Neither is an exact-MLE check,
    unlike Sections 0 and 2, because the two libraries implement `h`
    differently: the log-likelihoods differ by ~0.7 to 0.9 nats and in
    **opposite directions** on the two VACA nodes, which proves neither
    function class contains the other. The cause is the basis range. The flow
    pre-scales onto [-5, 5] and leaves 10% of the rows in linear extrapolation
    tails, while `Colr` uses the full support. The section writes its
    1000-row sample to `notebooks/data/vaca.csv` and reads it back, so the R
    snippet runs on the identical rows the flow was fitted on, and it makes
    the per-node-kind sign convention explicit: a continuous node adds its
    shift, so `OrderedModel`'s coefficients need negating, while an ordinal
    node's do not.
  - **Standard errors and confidence intervals, as a notebook prototype.**
    `fit_classical` leaves the model at the MLE in float64, which is what a
    Hessian-based standard error needs, so the notebook computes the observed
    Fisher information by double backward and inverts it. Sections 1 and 2
    both reproduce their classical reference: `Colr`'s standard errors to
    three decimals on `bwt`, and `statsmodels`' to four on the stroke
    outcome. The information matrix is **singular** in both, at 13 of 24
    parameters on `bwt` and one flat direction per one-hot parent on the
    stroke node. The helper therefore uses a pseudo-inverse, and it reports
    how far each contrast leaks into the flat subspace. A contrast with a non-zero leak
    still gets a finite, plausible number that means nothing, which is why
    `flow.conf_int` is not API: one row per parameter would be mostly
    nonsense.
  - **Section 2 says which coefficient is weakly identified, and it is not
    the treatment.** `T` sits 6.6 standard errors from zero. The weak one is
    `mRS_pre` level 5, carried by 7 of 1275 rows, with a standard error six
    times level 1's, which matches what `validate_ls.py`'s docstring already
    recorded. The flow-versus-`statsmodels` coefficient gaps turn out to
    measure something else entirely, the optimizer stopping while the
    likelihood is still flat: the displacement `c' I+ grad` predicts them
    exactly, at +1.4e-03 for `Age` and +7.4e-03 for `T`.
- **One introduction.** `notebooks/intro_tram_dag.py` is folded into
  `notebooks/demo_tram_dag_colab.py`, which keeps its filename so the Colab
  badge URL still resolves. The two carried separate copies of the same
  three-rung walkthrough, which is what let both drift apart. The merged
  notebook also writes the same DAG with `LS` terms and reads the
  coefficients, so the interpretability claim is demonstrated where the badge
  points. Every numeric claim in it is an assertion.
- **`notebooks/training_strategies.py`** runs every shipped fitting recipe on
  one workload and is the executable reference for `fit`, `optimizer=` and
  `callbacks=`. `docs/fitting.md` keeps the decision table and one line per
  recipe and gives up its six worked examples, and `docs/training-speed.md`
  gives up its recipe-to-API table and keeps the measurements.
- **`docs/model.md` and `docs/interpretation.md`.** The first holds the model
  theory that the intro notebook used to carry. The second is the guide that
  was missing: what a fitted coefficient, cutpoint, density or adjacency
  read-out actually means.

### Changed — tag-driven releases, and the docs are gated

- **The version is the git tag.** hatch-vcs derives it
  (`no-local-version`), `pyproject` carries no version field, and pushing a
  `v*` tag runs the release workflow: build with uv, publish to PyPI through
  trusted publishing with no token secret, sigstore-sign, and create the
  GitHub release. `cz bump` computes the next tag from the conventional
  commits. Two one-time steps live outside the repository, the PyPI
  trusted-publisher registration and the `pypi` GitHub environment.
- **The docs site moves from pdoc to MkDocs** (Material, mkdocstrings for the
  numpy docstrings, mkdocs-jupyter for the executed notebooks, mike for one
  version per branch). The 233-line workflow with its stub modules, link
  rewriter and jinja template becomes `mkdocs.yml` plus a 40-line link hook.
  Math renders through MathJax, and the code map's symbols link to the API
  pages (`.github/workflows/docs.yaml`). The PDF is typeset by pandoc and
  XeLaTeX from the README, the guides and the executed notebooks, so its math
  is real math. This also fixed every
  formula publishing as raw LaTeX. `pdoc` renders maths only with `--math`,
  which defaults to false, and the workflow never passed it. Around 110
  `$...$` spans in the notebooks and guides, plus 122 in the classical
  notebook, reached the site as literal source.
- **`mkdocs.yml` sets `strict: true`**, so a docstring that documents a
  parameter the signature does not have fails the build instead of logging a
  warning. CI also regenerates the diagrams in `docs/architecture.md` and
  fails on a diff. The committed copy had drifted to naming `_check_levels`
  after that method became `_check_level_values`.
- **`tools/` joins the max-15 complexity tier.** It matched neither tier
  before, and `gen_diagrams.py` had drifted to 22.
- **No documentation refers to a previous version.** `PerNodePlateau`
  described itself as the `fit(schedule="plateau", freeze_patience=)` recipe
  of an earlier release, in its own docstring and in four more places.
  `docs/training-speed.md` carried three measured rows for the `onecycle` and
  `cosine` schedules that `fit` no longer offers, with a footnote saying they
  cannot be re-measured. Those rows and that framing are gone, and the table
  lists only recipes a reader can run.
- **`notebooks/classical_fit_tram_dag.py` drops a blanket
  `warnings.filterwarnings("ignore")`.** Measured with
  `simplefilter("always")`, the notebook emits no warning, so the filter hid
  nothing and would have hidden a real one later.
- **The docstrings carry one markup, not two.** 97 Sphinx roles
  (``:func:`x```, ``:class:`x```, ``:meth:`x```, ``:mod:`x```, ``:data:`x```)
  and five reST literal blocks were left over from the pdoc era. mkdocstrings
  renders a docstring as Markdown, so a role reached the site as the literal
  text ``:func:`` followed by a code span, and a ``::`` block printed its own
  colons. Roles are now Markdown cross-references and literal blocks are
  fenced. `mkdocs.yml` turns on `scoped_crossrefs` and `relative_crossrefs`,
  so 53 of the references need no dotted path at all: a docstring writes
  ``[`pmf`][]`` and the handler resolves it in scope, which cannot go stale on
  a rename. The 20 that keep a path name something outside the referring
  module's scope. Verified with griffe that all 497 documented objects parse
  to the same numpydoc sections as before. The 19 reST literal blocks in
  `experiments/` are fenced too, so one markup holds across the repository.

### Changed (internal, no API surface)

- **Dependencies move to torch 2.14.0 and statsmodels 0.15.0**, with the
  pre-commit hooks on ruff 0.16.6, uv 0.12.10 and commitizen 4.18.0, and the
  release workflow's actions on checkout 7.0.1, upload-artifact 7.0.1,
  download-artifact 8.0.1 and sigstore 3.5.0. `release.yaml` was the one
  workflow still using bare `@v4` tags, so its actions are now SHA-pinned with
  a version comment like every other workflow.

  statsmodels is the exact-MLE oracle of four test modules and of the
  `validate_ls` ground truth, so the bump was verified rather than assumed.
  Its reference coefficients are unchanged to four decimals: NIHSSa +0.1630
  and the treatment −0.9424. The seeded state-dict baseline is still
  bit-identical under the new torch, and all four paper replications hold
  every committed bound. No ground-truth number was re-pinned.

  ruff 0.16.6 formats the Python blocks inside Markdown, which 0.16.3 left
  alone, so three documentation pages are reformatted.

- **`transforms` uses zuko's own inverse** (`Transform.inv`: bisection inside
  the bound, closed-form linear or identity tail). The 85-line
  expanding-bracket bisection duplicated it, at measurably identical
  residuals, and the spline inverse went from 166 ms to 0.3 ms on 4000 rows.
  `StandardLogistic.icdf` is `torch.logit(u, eps)`.
- **Single-value keyword arguments that no caller ever set became
  constants**: `StandardLogistic.sample(eps=)` and `icdf(eps=)` (`_U_EPS`),
  `BernsteinUT.marginal_init_theta(q=)` (`RANGE_Q`),
  `ordinal_marginal_init_theta(eps=)` (`_CDF_EPS`) and
  `scores.sup_bb_pvalue(terms=)`. `RANGE_Q` is shared with `_set_ranges`,
  which hard-coded the same 0.05: the two only calibrate each other when they
  agree, so any other `q` was a silent miscalibration rather than a setting.
- **The serialized term is `{term, parents, options}`.** `term.options()` is
  already canonical, sorted with defaults dropped, so `spec_to_dict` emits it
  whole and the per-key reader disappears. `spec_from_dict` builds `Term`
  directly, which makes `validate_and_sort` the only guard on the load path,
  and a malformed checkpoint is rejected there
  (`tests/test_transformation_syntax.py`).
- **Every complexity hotspot is dissolved into named stages**: `fit`
  (cognitive complexity 103 to 10), `validate_and_sort` (65 to 1),
  `_Node.__init__`, `to_matrix`, `check.compare`, `bench_training.main` and
  seven more. Every module follows one `# %% <section>` layout. Verified
  behavior-identical: a fixed-seed harness compares state dicts, history and
  samples bit-equal before and after, and every error message moved verbatim.
- **The framework tests carry their own data.** Three inline numpy DGPs in
  `tests/conftest.py` (an all-`ls` chain, a heterogeneous-effect DGP and a
  confounded DGP with a prognostic misfit) replace the generator package the
  suite used to import. The external-software anchor is unchanged in
  substance: an all-`ls` outcome node is an ordered-logit model, so the flow's
  MLE must equal `statsmodels` on the same design matrix. It is now measured
  on inline data at test time, instead of against a committed R reference.
  The R comparison lives on in `experiments/misc/validate_ls.py`,
  and the generator-pinning and frozen-CSV tests moved to the experiments
  workflow.
- **The SCM generators share one layer.** `simulations/_common.py` holds
  `logistic`, `sigmoid`, `resolve_latents`, a shared `clamp`, and the
  `DatasetDraws` mixin (`observational`, `interventional`,
  `counterfactual_pair`). With it the seed offsets behind the frozen CSVs
  (`+1`, `+501`, `+2`) are defined once instead of once per generator, and
  every generator exposes the same three named draws. What is left after the
  stroke storyline moved out is 916 lines for the four paper generators, with
  the frozen-CSV contract unchanged.
- **The stacked ternaries in the `vaca` and `carefl` generators' `simulate`
  became one if/else per variable**, for the readability of experiment code.
  Verified behaviour-neutral: `vaca` regenerates bit-identically and `carefl`
  within 7e-15, the same machine-epsilon drift the untouched `triangle`
  generator shows after the dependency bump.
- **`experiments/paper/helpers.py`**: the per-epoch coefficient read-out is
  `fit(callbacks=)` inside one `fit_paper(generator, spec, config, out,
  record)` call. There is no chunked or snapshotting fit helper any more.

### Fixed

- **`EarlyStopping` blamed its own wiring for a diverged fit.** A NaN
  validation NLL never beats `inf - min_delta`, so nothing was ever
  snapshotted and fit end raised "EarlyStopping has seen no epoch". The
  message now separates the three cases: no epoch ran, every epoch was
  non-finite, or none improved. The second one names the likely cause.

- **`plot_training` drew the validation curve from epoch 1 whatever epoch it
  came from.** `history` accumulates across `fit` calls, so a `fit` without
  validation followed by one with it put the validation curve two epochs
  before the data it was measured on. `fit` now records `history["val_epoch"]`
  and the figure draws each curve on its own epochs. The docstring also
  stopped claiming it shows the last `fit` alone.

- **`fit` trained on nothing and reported an NLL of zero.** The batch loop
  skips any batch under two rows, so a one-row frame ran its epochs, moved no
  weight and recorded `0.0` per node. It now raises and says why. The epoch
  NLL is also averaged over the rows actually stepped on, so a skipped
  trailing row no longer scales every node's number down by `1/n`.

- **`shift_curve` died inside torch on an ordinal parent.** An ordinal parent
  enters a term one-hot over all its levels, so a 1-D grid of level indices is
  not its input, and the mismatch surfaced as
  "mat1 and mat2 shapes cannot be multiplied". It is named now, and points at
  `ls_coefficients`, whose weights are the level contrasts.

- **`check.py` had no test, and the escape hatch it grew hid the thing it was
  meant to expose.** A `"why"` on a `{max}` entry silenced *both* edges of the
  band check. Three `validate_ls` bounds therefore sat 12x, 141x and 566x
  above their measurements, one of them the entry introduced as "the precision
  claim", and a fabricated 560x regression passed with an `ok`. A `"why"` now
  excuses
  width only, and no argument survives a bound below 1.5x its measurement.
  The three bounds are re-derived from a stated floor of 1e-3, the accuracy
  any comparison here claims, rather than inherited from the classical
  variant, whose scale they do not share.

  The other decay mode had no check at all: a `{value, atol}` center keeps
  passing while drifting through its tolerance, which is exactly how two
  centers reached 62% and 75% before anyone noticed. A measurement past half
  its `atol` is now reported the same way.

  `experiments/tests/test_check.py` covers all of it, including that a better
  fit never fails a `{max}`, that a `"why"` cannot silence a too-tight bound,
  and that a truth entry whose metric disappeared is an error. It is the first
  test the file has had, and the band logic had never run on a real input,
  because every committed bound was inside the band.

- **Ground-truth centers now follow the code.** The architecture change moved
  every complex-shift variant, but only two files were re-pinned, so several
  `{value}` centers described a net that no longer runs:
  `triangle-atan-cs`'s interventional mean had consumed 62% of its tolerance
  and `triangle-sin-cs`'s beta13 75%. All centers are re-measured. `{max}`
  bounds are kept inside a band instead of hand-tuned, useful between 1.5x and
  4x the measurement and set to 2.5x outside it. Below 1.5x is not
  hypothetical: one bound at 1.7x passed locally at 0.028 and failed CI at
  0.113, because its maximum is over a coefficient with 7 of 1275
  observations. That comparison is now split into
  `max_abs_diff_named_coefs` (Age, NIHSSa and the treatment contrast, the
  precision claim at ~1e-3) and the all-coefficient maximum, a sanity bound.

- **vaca's ground truth pinned each flow mean *and* bounded its error against
  the analytic truth.** Those are two windows on one number, offset by
  `|center - analytic|`, and they disagreed: a run landing exactly on the
  pinned `do(x2=-3)` mean would have failed its own paired bound, at an error
  of 0.0605 against a bound of 0.0324. Setting the `atol` to the bound does
  not fix it unless the center *is* the analytic value. The three centers are
  gone. The analytic value and the error bound determine the flow mean between
  them, and the run report still prints it.

- **Two of the four paper replications used the wrong reference
  architecture.** The reference implementation has two. The triangle scripts
  use `hidden_features_I = hidden_features_CS = c(2,25,25,2)` with sigmoid,
  while its own CAREFL and VACA comparisons use
  `comparison/utils.R::make_model`, one net per node,
  `dense(10, tanh) -> dense(100, tanh) -> dense(len_theta)`, with `M = 30`.
  `vaca.yaml` and `carefl.yaml` cited the first while replicating the second.
  On the correct net CAREFL improves on every previously committed number, at
  a counterfactual MAE of 0.078 / 0.059 / 0.086 against bounds of
  0.216 / 0.174 / 0.219. VACA's off-manifold `do(x2=-3)` mean lands 0.037 from
  the analytic truth, instead of 0.21.

- **The ordinal counterfactual score had a reference point that was not a
  bound.** `cf_prob_true_level_ceiling` was documented as "the best any model
  could do". It is `E[p_true] = E[sum_i p_i^2]`, which is what a model that
  knew the identifiable law exactly would score. The largest *expected* score
  is `E[max_i p_i]`, from always naming the *modal* level, which is a strictly
  worse distribution estimate. The metric is now
  `cf_prob_true_level_analytic`, with `cf_prob_true_level_mode_bound`
  alongside it, at 0.921 and 0.954 on the mixed DGP. This surfaced when the
  corrected architecture pushed the flow to 0.924, above its own stated
  ceiling.

  Both references are expectations while the metric is one finite draw, so
  neither is a per-run ceiling either: on the `linear` DGP the mode predictor
  scores **0.829** against its own 0.806 expectation, an overshoot larger than
  the 0.003 gap the metric was introduced to explain. Read the two as
  reference points a run sits between, and `cf_pmf_tv_vs_analytic` as the
  metric that cannot be gamed by sharpening a prediction.

- **The `conditioners` provenance claim was wrong.** The module, the README,
  CLAUDE.md and `docs/code-map.md` all said the default architectures come
  from "the original Keras implementation (`tram_models.py` in
  tensorchiefs/tram-dag)". That repository is pure R and has no Python in it.
  The defaults come from the PyTorch reference this package grew out of,
  buehlpa/TramDag, file `tram_models.py`:
  `ComplexShiftDefaultTabular` 64-128-64 ReLU,
  `ComplexInterceptDefaultTabular` 8-8 ReLU, and `n_thetas=20`, which is also
  why `DEFAULT_ACTIVATION` is relu. Corrected in all four places, each of
  which now also says these are *not* the paper's nets, so a replication
  states `units=` and `activation=` itself.

- **The `n_coeffs` documentation was off by two control points.** zuko
  constrains `n` unconstrained coefficients into `n + 2` monotone control
  points, duplicating the end differences for a smooth extrapolation, so
  `n_coeffs=20` is a degree-21 polynomial where the reference's
  `len_theta=20` is degree 19. The configs claimed "order M = 20 from the
  paper", and they now state the mapping and that the free-parameter count is
  what matches.

- **`ls_coefficients()` crashed on a node mixing `LS` and `CS` terms.** It
  read `.weight` off every shift module, but a `CS` shift is a network and a
  `VC` shift is an effect head, so neither has one and any such node raised
  `AttributeError`. That broke the paper's headline complex-shift
  replication, whose documented default was `atan cs`. The method returns the
  linear-shift weights it is named for and skips network shifts, and a node
  with no `LS` term is absent from the result (`tests/test_api_papercuts.py`).

- **`experiments/benchmarks/bench_training.py` reported a recipe that reached
  its target in the first recorded epoch as a miss**, because it tested a time
  of `0.0` for truthiness.

- **A whole-tree `ruff check .` reported 53 errors that CI never saw.** The
  per-file-ignore key for the notebooks did not match the tracked symlinks
  under `docs/notebooks/`, which ruff follows.

## 0.3.0 (2026-06-19)

### Removed (breaking)

- **The legacy `parents={parent: "ls"|"cs"|"ci"}` constructor argument** is gone.
  Use the term-formula notation `terms=[I(...), LS(...), CS(...)]` (see below);
  `tramdag.term(effect, *parents)` helps when the effect is data-driven. Old
  *checkpoints* saved with the dict layout still load.

### Added

- **Term-formula spec notation** — declare a node's transformation as an additive
  list of terms, `terms=[I(...), LS(...), CS(...)]`, replacing the per-edge
  `parents={parent: "ls"|"cs"|"ci"}` dict (now **deprecated**, still accepted with
  a `DeprecationWarning`). Each term names the parent(s) it depends on; **joint
  (multi-parent) terms** express interactions — `CS("x1", "x2")` is one shift
  network over both parents and `I("x1", "x2")` one joint intercept — while
  separate terms stay additive: `CS("x1") + CS("x2")` are two additive shifts, and
  `I("x1") + I("x2")` is an **additive complex intercept** (each parent reshapes the
  transform independently, the per-term coefficient vectors summed in unconstrained
  space). The grouping *is* the joint/additive choice. A `term(effect, *parents)`
  factory helps data-driven specs, and `flow.to_matrix()` renders the paper's
  meta-adjacency view.

- **API papercuts (issue #12):** `CausalFlowDAG(spec, seed=...)` seeds weight
  initialization deterministically (one obvious reproducibility knob — `fit(seed=)`
  only seeds shuffling); `save`/`load` now also carry a provenance `meta` block
  (tramdag version, save time, device, and a machine/environment snapshot) and
  `flow.meta` is repopulated on load, so cached models are self-describing;
  `tramdag.machine_info()` exposes that snapshot (host, OS, CPU/GPU, cores, RAM,
  python/torch/zuko/tramdag versions); a dev-install one-liner
  (`pip install "git+https://github.com/tensorchiefs/tramdag.git@main"`) is
  documented in the README and the Colab demo. (Training `history` already
  round-tripped through `save`/`load`; now covered by a regression test.)

- **`fit(marginal_init=True)`** — opt-in calibrated initialization for *unconditional*
  (`SimpleIntercept`) nodes, replacing zuko's default zero init. Bernstein roots
  start at the linear map of the pre-scaled domain onto the standard-logistic
  5/95 quantiles (the default is ~2.5× too steep); ordinal roots start at the
  empirical class log-odds (default zeros ≈ uniform). A **pure init** — the
  converged MLE is unchanged (the exact-`ls` MLE / R-`polr` equivalence is
  preserved), applied once on the first fit, conditional `ci` intercepts untouched.
  Large time-to-target win where a root's marginal shape dominates the NLL gap
  (vaca-ci ~2.5× faster to target over 6 seeds); small where convergence is
  coefficient-bound. Defaults unchanged (off).

- **`CausalFlowDAG.fit_classical()`** — deterministic, full-batch, **float64**
  L-BFGS for all-`ls` models (each node-conditional is then a classical
  transformation model). Bit-reproducible, reaches the exact MLE, matches
  `statsmodels` ordered-logit / R `polr`/`Colr` to ~1e-3 on well-identified
  coefficients; raises on `cs`/`ci` specs (use `fit()`). Plus `ls_coefficients()`
  to read the per-node shift weights. float64 is a transient compute mode
  (`self.double()/.float()`), so the stored model stays float32; as a side effect
  the data path (`_tensorize`/`sample`/`pmf`) is now dtype-agnostic.
- `notebooks/classical_fit_tram_dag.py` (didactic) and a `--classical` flag for
  `experiments/validate_ls.py`.
- **Next:** standard-error table from the float64 Hessian at the MLE (the float64
  bracket here is the groundwork); needs a reference-level constraint for the
  one-hot ordinal-parent flat directions.

## 0.2.0 (2026-06-12)

First PyPI release: `pip install tramdag`.

### Changed (naming & packaging)

- **Renamed**: Python package `zuko_dag` → **`tramdag`** (conventional alias
  `import tramdag as td`); GitHub repo `tram-dag-zuko` → `tensorchiefs/tramdag`
  (old URLs redirect). The package implements TRAM-DAGs; zuko names the backend.
  No API changes; old checkpoints still load. References to the original
  Keras/TF implementation (tensorchiefs/tram-dag) reworded to avoid
  self-reference.
- **MIT license** added; PyPI metadata (authors, urls, classifiers); runtime
  dependencies trimmed to `torch`, `zuko`, `numpy`, `pandas` (pytest/scipy/
  statsmodels/scikit-learn/matplotlib moved to the `dev` dependency group).
- **README rewritten method-first**: the repo is the reference implementation of
  the CLeaR 2025 paper (arXiv:2503.16206); the stroke analysis is the case study
  (arXiv:2606.12623) with its detail moved to `docs/stroke-case-study.md`.
  Citation BibTeX added for both papers.

### Added

- **`fit(schedule=..., freeze_patience=...)`** — learning-rate schedules and
  per-node early stopping (defaults unchanged). The optimizer now holds one
  param group per node; `schedule="plateau"` decays each node's lr off its own
  validation NLL, and `freeze_patience` drops converged nodes from the loss
  (real FLOP savings — per-node gradients are independent) with early exit when
  all nodes froze. Also `"onecycle"`/`"cosine"`. Benchmarks + recommendation in
  `docs/training-speed.md` (`experiments/bench_training.py`): plateau+freeze
  matches the hand-tuned two-phase recipe's time-to-accuracy with **no budget
  tuning and ~3× less total compute**; full-batch LBFGS solves the classical
  all-`ls` MLE in <2 s (2/3 seeds). Existing defaults intentionally untouched.
- **Colab demo** `notebooks/demo_tram_dag_colab.py` (+ tracked output-stripped
  `.ipynb` for the badge): the paper's bimodal VACA benchmark fitted live
  (cuda/cpu auto-detect), L1 pairs plot, analytic do-checks, per-individual
  counterfactuals vs DGP truth, GPU-vs-CPU race.

- **The TRAM-DAG paper's DGPs** (Sick & Dürr, CLeaR 2025, arXiv:2503.16206) as
  simulation registry families, each a numpy-only SCM with known/analytic ground
  truth + frozen n=5000 CSVs (`data/<name>/`, the test contract) and CLIs:
  - `simulations/triangle.py` — `TriangleContinuous` (§6.1: logistic-latent TRAM
    DGP, h₂=5x₂+2x₁, h₃=0.63x₃−0.2x₁−f(x₂)) and `TriangleMixed` (§6.2: ordinal x₃,
    θ=(−2, 0.42, 1.02)); f variants `linear`/`cubic`/`exp`/`atan`/`sin`; supports
    array-valued `do` (C.4 soft interventions).
  - `simulations/vaca.py` — `VacaTriangle` (App. C.1 bimodal Gaussian L1/L2
    benchmark vs CNF).
  - `simulations/carefl.py` — `Carefl4` (App. C.2 Laplace SCM; **analytic**
    counterfactuals via `abduct_noise`/`true_counterfactual`).
- `experiments/paper_{triangle,triangle_mixed,vaca,carefl}.py` (+ `paper_common.py`)
  — replicate the paper's figures: coefficient trajectories (Fig. 14/15/19), CS-curve
  recovery (Fig. 7), L1/L2 distribution overlays (Fig. 4/5/9/16/20), counterfactual
  curves at the paper's x_obs (Fig. 6), and the C.4 odds-ratio check (OR ≈ 7.4).
- `tests/test_paper_dgps.py` — generator pinning (KS TRAM-identities, frozen-CSV
  contract, analytic ground truth) + flow recovery (coefficients with the ordinal
  sign-flip, CS curve, VACA do-moments, CAREFL counterfactual MAE).

### Changed (behavior)

- **`CausalFlowDAG.fit(..., restore_best=False)` is now the default.** Training keeps
  the **final converged weights** instead of restoring per-node best-validation
  weights. Rationale:
  - *Least surprise* — `fit()` returns the model you trained, not a silently
    swapped earlier epoch.
  - *Exact classical comparison* — an all-`ls` model trained to convergence is now
    exactly the maximum-likelihood (proportional-odds) estimate, matching
    `statsmodels` `OrderedModel` and R `MASS::polr` to ~1e-3 (see
    `experiments/validate_ls.py`, `tests/test_simulations.py::test_all_ls_flow_is_exact_mle`).
    This was **not achievable before**: best-validation restoration pinned the fit
    off the training optimum.
  - Early stopping is now an explicit, opt-in regularization choice.

  To restore the previous behavior, pass `restore_best=True`.

  **Note for flexible (`ci`/`cs`) models:** their MLE *overfits the observational
  confounding*, so they need `restore_best=True` to recover the causal effect (lower
  validation NLL confirms it generalizes better). `experiments/run_experiment`
  therefore defaults `restore_best` per style — off for all-`ls`, on for flexible.

### Added

- `src/tramdag/simulations/magic_mrclean.py` — synthetic stroke cohort (SCM with
  known ground truth); `ls`/`nl` variants; CLI to (re)generate frozen CSVs.
- `data/magic-mrclean/` — frozen public CSVs + `fit_ls.R` classical R reference and
  committed `ref_ls/` outputs. The public, reproducible substitute for the private
  clinical data.
- `experiments/common.py::load_data(source)` — switch between `"magic"` (private) and
  `"magic-mrclean/{ls,nl}"` (synthetic, default).
- `experiments/sim_flow.py` — known-truth recovery storyline; `validate_ls.py`
  rewritten as a spot-on flow-vs-MLE-vs-R comparison.
- `tests/test_simulations.py` — generator, known-truth recovery, the all-`ls`
  spot-on MLE check, and the Python-vs-R regression.
