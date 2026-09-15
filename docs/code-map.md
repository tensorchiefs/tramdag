# Code map — every module of `src/tramdag/` — the public names and the private plumbing

This document has one entry per name, with its role and its place in the
pipeline. Names in parentheses are private machinery. They are useful to know,
and they are not part of the API. The last section lists every training
hyperparameter and where it lives.

## `spec.py` — declare the model

Every term is a `Term` subclass under its pythonic name. The paper's symbol is
the same object, so `LS is LinearShift`.

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`Term`][tramdag.spec.Term] | One additive term of a node's transformation, plain data: a subclass per term with `name` and `module` as class attributes and its options as the keyword arguments of `__init__` assigned to `self`, so `options()` is `dict(vars(term))`. `+` on terms builds plain lists. Carries the spec-level rules `check`, `edge_parents`, `cells`, `classical` and `from_serialized`. The contract is in [architecture.md](architecture.md). |
| [`SI()`][tramdag.spec.SI] | The parentless intercept — the paper's SI. Free transform parameters, the same for every row. Carries the transform choice (`transform=`, default `"bernstein"`); extra keyword arguments pass straight to the transform class. |
| [`CI()`][tramdag.spec.CI] | The parent-conditioned intercept — the paper's CI: the parents reshape the monotone transform. Needs at least one parent. Also carries `units=` and `allow_interaction=` (joint vs. additive multi-parent intercept). |
| [`Intercept`][tramdag.spec.Intercept] / `I` | The intercept term class: without parents the paper's SI, with parents the CI; `SI()`/`CI()` are the two spellings with their arity checked. |
| [`LinearShift`][tramdag.spec.LinearShift] / `LS` | Linear shift `beta * x` — the interpretable log-odds coefficient. Exactly one parent. |
| [`ComplexShift`][tramdag.spec.ComplexShift] / `CS` | Complex shift: an NN `g(x)`, additive on the latent scale. Several parents form one joint network. |
| [`VaryingCoefficient`][tramdag.spec.VaryingCoefficient] / `VC` | Varying-coefficient shift `(beta0 + b_theta(mods)) * x_t` — the penalized treatment-effect head. `center=` adds propensity centering. |
| [`ContinuousNode`][tramdag.spec.ContinuousNode] | Continuous variable: monotone 1-D transform plus shifts. `terms` is the first positional argument. |
| [`OrdinalNode`][tramdag.spec.OrdinalNode] | Ordinal variable with `levels` classes: ordered logit (cutpoints) plus shifts. |
| [`node_parents()`][tramdag.spec.node_parents] | Ordered de-duplicated parent names of a node (the canonical term list is `node.terms`). |
| [`validate_and_sort()`][tramdag.spec.validate_and_sort] | Edge-ownership validation plus Kahn topological sort. The returned order makes the flow triangular. |
| [`spec_to_dict()`][tramdag.spec.spec_to_dict] / [`spec_from_dict()`][tramdag.spec.spec_from_dict] | Checkpoint (de)serialization. A term serializes as `{term, parents, options}` with every option it carries; a hand-written spec may name only some, and the constructor fills the rest. A built-in term's `term` key is its `name`; a custom term writes its `module.ClassName` import path, which `_import_object` resolves. `spec_from_dict` rejects a term without `options`, the node constructors normalize the formula, and `validate_and_sort` checks the DAG. |
| (`_normalize_terms`, `_check_term`, `_term_class`, `_import_object`, `_checked_input_transform`, `_serialized`) | Formula flattening and per-entry validation (a `+` sum nested in a list is rejected), the one-parented-`I` rule plus transform hoisting in one pass. An option another term takes is refused by Python's own argument binding, as a `TypeError`. |
| (`_check_node`, `_kahn_sort`) | The stages behind `validate_and_sort`: parents exist, then each term's `check` (the VC treatment and centering rules), then edge ownership through `edge_parents`; a term's own shape (arity, option values) and a node's own (ordinal levels, the transform) are checked when they are built. Kahn's sort emits ready nodes in sorted batches, so the order is deterministic. |

## `transforms.py` — the monotone map h and the ordinal transform

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`StandardLogistic`][tramdag.transforms.StandardLogistic] | The TRAM base distribution: `log_prob`, `sample` (generator-aware), `icdf`. |
| [`BernsteinUT`][tramdag.transforms.BernsteinUT] | Bernstein-polynomial transform, the default. Linear tail extrapolation follows the boundary derivative. `marginal_init_theta(column)` gives the start `init_marginals` applies. |
| [`SplineUT`][tramdag.transforms.SplineUT] | Monotone rational-quadratic spline. Tails extrapolate with a fixed slope ([zuko-upstream.md](zuko-upstream.md)). |
| [`AffineUT`][tramdag.transforms.AffineUT] | Monotone affine transform: the node-conditional is a logistic GLM. |
| [`make_univariate_transform()`][tramdag.transforms.make_univariate_transform] | Transform registry: name → transform instance. |
| [`ordinal_cutpoints()`][tramdag.transforms.ordinal_cutpoints] | Unconstrained `(n, K-1)` → increasing cutpoints with ±inf ends. |
| [`ordinal_log_prob()`][tramdag.transforms.ordinal_log_prob] | `log P(Y=y)`, computed in log space ([model.md](model.md#ordinal-nodes)). |
| [`ordinal_pmf()`][tramdag.transforms.ordinal_pmf] / [`ordinal_sample()`][tramdag.transforms.ordinal_sample] / [`ordinal_abduct()`][tramdag.transforms.ordinal_abduct] | Class probabilities / latent → level / truncated-logistic latent recovery (Pearl step 1) for ordinal nodes. |
| [`ordinal_marginal_init_theta()`][tramdag.transforms.ordinal_marginal_init_theta] | Cutpoint start that matches the empirical class frequencies (`init_marginals`). |
| [`ordinal_bounds()`][tramdag.transforms.ordinal_bounds] | The shifted cutpoint interval of each observed level. `scores.py` reads it for the latent-scale derivative. |
| (`_ScaledUT`, `_log1mexp`) | Quantile pre-scaling base class, whose inverse is zuko's with its closed-form tail. Stable `log(1-exp(x))`. |

## `flow.py` — the model

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`CausalFlowDAG`][tramdag.flow.CausalFlowDAG] | The flow: one [`Node`][tramdag.nodes.Node] per variable in topological order. Construction seeds the weights. |
| [`calibrate()`][tramdag.flow.CausalFlowDAG.calibrate] | The data-dependent state of every term, taken once from the training rows ([fitting.md](fitting.md)). |
| [`init_marginals()`][tramdag.flow.CausalFlowDAG.init_marginals] | The marginal start of every simple intercept, as an explicit step callable any time; on a trained flow it restarts those intercepts. |
| [`fit()`][tramdag.flow.CausalFlowDAG.fit] | Joint maximum likelihood by one minibatch Adam loop, with `validation_data=`/`validation_split=`, `verbose=`, `optimizer=` and `callbacks=`; `history` holds `train`, `val` and `lr` per epoch. A second call continues training. The mechanics are in [fitting.md](fitting.md). |
| [`fit_classical()`][tramdag.flow.CausalFlowDAG.fit_classical] | Float64 full-batch L-BFGS for all-`ls` specs; refuses flexible specs. |
| [`sample()`][tramdag.flow.CausalFlowDAG.sample] | Observational, interventional (`do=`, graph mutilation) and counterfactual (`u=`) sampling. |
| [`abduct()`][tramdag.flow.CausalFlowDAG.abduct] | Pearl step 1: recover the latents. |
| [`pmf()`][tramdag.flow.CausalFlowDAG.pmf] | Analytic class probabilities of an ordinal node, with `do=` overrides. |
| [`density()`][tramdag.flow.CausalFlowDAG.density] | Analytic conditional density of a continuous node on a grid, with `do=` overrides — the continuous counterpart of `pmf`. |
| [`log_prob()`][tramdag.flow.CausalFlowDAG.log_prob] / [`node_negative_log_prob()`][tramdag.flow.CausalFlowDAG.node_negative_log_prob] (alias `nll`) | Joint per-row log-likelihood, or a `nodes=` subset (exact, and the log-space way to get one node's conditional likelihood per row) / mean per-node NLL diagnostic. |
| [`node_log_prob()`][tramdag.flow.CausalFlowDAG.node_log_prob] | The per-node decomposition everything trains and evaluates through. |
| [`varying_coef()`][tramdag.flow.CausalFlowDAG.varying_coef] | Closed-form read-out `beta(x)` of a fitted VC term. Deterministic, y-free. |
| [`scores()`][tramdag.flow.CausalFlowDAG.scores] / [`effect_modifier_scan()`][tramdag.flow.CausalFlowDAG.effect_modifier_scan] | Analytic per-observation scores and the CUSUM modifier scan (delegate to `scores.py`). |
| [`intercept_contributions()`][tramdag.flow.CausalFlowDAG.intercept_contributions] | Post-hoc GAM-style decomposition of a complex intercept into mean-centered per-term parts. |
| [`ls_coefficients()`][tramdag.flow.CausalFlowDAG.ls_coefficients] | The per-node linear-shift weights — the interpretable coefficients. |
| [`design_matrix()`][tramdag.flow.CausalFlowDAG.design_matrix] | Parent encoding as a DataFrame (`drop_first=` gives the classical statsmodels/`polr` design). |
| [`to_matrix()`][tramdag.flow.CausalFlowDAG.to_matrix] | The labeled meta-adjacency matrix of term tags. |
| [`save()`][tramdag.flow.CausalFlowDAG.save] / [`load()`][tramdag.flow.CausalFlowDAG.load] | Checkpoints with history and provenance (version, time, device). `load` requires a complete checkpoint and fails loudly otherwise. |
| (`_node`, `_features`, `_tensorize`, `_generator`, `_dtype`, `_init_linear`) | Node lookup with one shared error; parent encoding through `Node.encode` (continuous raw, ordinal one-hot); `_tensorize(df, cols=None)` for any column subset, checking every ordinal column against its levels on the way in; seeded-generator and dtype plumbing. |
| (`_check_side_columns`, `_propensity`, `_side_feats`, `_query_side_columns`, `_recenter_vc`) | The generic side-column plumbing (each term names/validates/recomputes its own columns via the `ShiftModule` hooks) plus the binary propensity fit and the post-fit `finalize` loop. |


## `modules.py` — the term modules

There is one module class per term, and the term class holds it as `module`
(`ComplexShift.module is ComplexShiftModule`); the module holds the term's
network and owns the runtime hooks ([architecture.md](architecture.md)).

The default architectures replicate the PyTorch reference
[buehlpa/TramDag](https://github.com/buehlpa/TramDag), file `tram_models.py`,
so a fitted model stays comparable to it. They are not the paper's nets, which
[paper-replication.md](paper-replication.md) lists per experiment.

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`intercept_module()`][tramdag.modules.intercept_module] | The one factory: an `I` term becomes the free theta, one joint net, or one net per parent (`allow_interaction=False`). |
| [`feat_width()`][tramdag.modules.feat_width] | Total encoded width of a parent set: `levels` per ordinal parent, 1 per continuous one. |
| [`ShiftModule`][tramdag.modules.ShiftModule] / [`InterceptModule`][tramdag.modules.InterceptModule] | The behavior hooks a term module owns: `__init__(term, spec)` (builds the net, sets `key` and `parents`), `shift_value`/`theta_value`, `post_init`, `regularizer`, post-fit `finalize`, `score_columns`, the side-input contract. |
| [`LinearShiftModule`][tramdag.modules.LinearShiftModule] | `LS`: `Linear(n, 1, bias=False)`. `.weight` is the interpretable coefficient; no bias because the intercept slot owns the constant. |
| [`ComplexShiftModule`][tramdag.modules.ComplexShiftModule] | `CS`: an NN to one shift value. |
| [`VaryingCoefficientModule`][tramdag.modules.VaryingCoefficientModule] | `VC`: `beta0 + b_theta(mods)` with the L2 hook `l2()`; `beta()` evaluates the effect and `recenter()` re-splits `beta0`/`b_theta` after training ([varying-coefficients.md](varying-coefficients.md)). `regressor` is both the forward regressor and the `beta0` score. |
| [`FnShiftModule`][tramdag.modules.FnShiftModule] | `Fn`: a user-supplied shift function over the parent features. |
| [`SimpleInterceptModule`][tramdag.modules.SimpleInterceptModule] / [`ComplexInterceptModule`][tramdag.modules.ComplexInterceptModule] / [`AdditiveInterceptModule`][tramdag.modules.AdditiveInterceptModule] | The intercept slot: free theta (`I()`), one joint net from the parent features to the transform parameters, or one net per parent summed in coefficient space. The additive one holds its nets in `nets`. |
| (`_nn`) | The one NN builder: a stack of the given `units` with the term's `activation` (optional `batch_norm` before it), then a bias-free output layer. |
| (`_InputTransform`) | One term's frozen network-input transform (minmax / standardize / callable over frozen train columns). |
## `nodes.py` — the node model

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`Node`][tramdag.nodes.Node] | One sub-model per variable: builds its intercept and shift terms through each term's `module`; `theta_shift()` sums the terms' `shift_value`s (plain shifts first, then VC); `net_input()` feeds every term network, `input_transform` applied. |
| [`Node`][tramdag.nodes.Node] `.log_prob` / `.sample` / `.abduct` / `.marginal_theta` / `.encode` | The continuous-vs-ordinal branches and the parent encoding ([architecture.md](architecture.md#node-kinds)). |

## `fitting.py` — `FitMixin`

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`fit()`][tramdag.flow.CausalFlowDAG.fit] / [`fit_classical()`][tramdag.flow.CausalFlowDAG.fit_classical] | Defined here once, methods of the flow via the mixin. |
| (`_split_validation`, `_normalize_callbacks`, `_check_fit_sizes`, `_learning_rates`, `_log_epoch`, `_val_nll`, `_fit_epoch`, `_FnCallback`) | The loop plumbing: Keras-shaped validation split, callback normalization, the validation pass, the rate record, verbose printing. |

## `readouts.py` — `ReadoutsMixin`

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`shift_curve()`][tramdag.flow.CausalFlowDAG.shift_curve] | One fitted shift term on a 1-D grid, through the term's own `shift_value`. |
| the read-out methods | `varying_coef`, `ls_coefficients`, `to_matrix`, `intercept_contributions`, `design_matrix` — defined here once, methods of the flow via the mixin. |

## `scores.py` — effect-modifier detection

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`node_scores()`][tramdag.scores.node_scores] | Per-observation scores of the interpretable shift coefficients ([scores.md](scores.md)). |
| [`effect_modifier_scan()`][tramdag.scores.effect_modifier_scan] | The fluctuation scan over candidate modifiers ([scores.md](scores.md)). |
| [`sup_bb_pvalue()`][tramdag.scores.sup_bb_pvalue] | `P(sup |Brownian bridge| > stat)`, the Kolmogorov series. |
| (`_dl_ds`, `CRIT_5PCT`) | Closed-form latent-scale derivative and the 5 % critical value. The per-term columns come from each term's `score_columns` hook. |

## `callbacks.py` — the shipped `fit` callbacks

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`EarlyStopping`][tramdag.callbacks.EarlyStopping] | Best-validation weights, restored at fit end (`restore_best=False` keeps the final ones); `patience=` also stops the fit. |
| [`PerNodePlateau`][tramdag.callbacks.PerNodePlateau] | Per-node rate decay and freezing; stops the fit once every node froze and records `frozen = {node: epoch}`. `step(nll, opt, epoch)` for a hand-driven loop. |
| [`per_node_adam()`][tramdag.callbacks.per_node_adam] | Adam with one `node`-tagged parameter group per node — the optimizer `PerNodePlateau` needs. |

## `plots.py` — the figures (matplotlib optional: `tramdag[plots]`)

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`plot_dag()`][tramdag.plots.plot_dag] | The labelled DAG of a spec or flow: layered left to right, ellipses for continuous and rounded boxes for ordinal nodes, every edge drawn by the term that owns it (LS / CS / CI / VC + modifiers / Fn, `joint` for a multi-parent net). Exported as `tramdag.plot_dag`. |
| [`plot_marginals()`][tramdag.plots.plot_marginals] | Observed vs sampled marginal per node, one panel each. |
| [`plot_training()`][tramdag.plots.plot_training] | Summed train/val NLL per epoch, with a dashed mark per `frozen=` entry. |
| (`_layout`, `_term_edges`) | Longest-path layers with one barycenter sweep; the edge list with the VC treatment/modifier split. matplotlib is imported on the first call, never at package import. |

## What is *not* in the package

Three groups of files are research code:

- the SCM generators
- the frozen datasets
- the replication scripts

They live in [`experiments/`](../experiments/), outside the installed package;
[`experiments/README.md`](../experiments/README.md) describes them and
[`tests/README.md`](../tests/README.md) what the framework tests measure
instead.

## Where every training hyperparameter lives

Everything that shapes a fit is either a keyword you pass or a documented
default you can read at the call site. Nothing numeric is buried.

| Knob | Where | Default |
|----------------------|-------------------------------|----------------------------------------------------------|
| learning rate, batch size | `fit()` | 1e-2 / 512 (in-repo callers state them explicitly anyway) |
| validation, progress | `fit(validation_data=, validation_split=, verbose=)` | validation off, `verbose=0` |
| schedules, early stopping | `fit(optimizer=, callbacks=)` | `tramdag.callbacks` ships `EarlyStopping`, `PerNodePlateau`; anything else is torch's `lr_scheduler` and a few lines of callback ([fitting.md](fitting.md)) |
| calibrated init | `init_marginals(train_df)` | never implicit; without it zuko's zero start |
| VC stage-1 propensities | the training-frame column `VC(center=)` names | required for a centered VC term ([varying-coefficients.md](varying-coefficients.md)) |
| VC penalty and centering | `VC(penalty=, center=)` | 1.0 / False (`center="col"` names the propensity column) |
| L-BFGS budget | `fit_classical(max_iter=, history_size=)` | 400 / 50; torch's `tolerance_change` is 1e-9 and `tolerance_grad` is off — one full-batch run, no chunks |
| training budget | `fit(epochs=)` | **required** ([fitting.md](fitting.md)) |
| network widths | `units=` on `I`/`CS`/`VC` | (8, 8) / (64, 128, 64) — parity with the PyTorch reference's default classes; VC's (16,) has no counterpart there and comes from the recovery measurement |
| activation | `activation=` on `I`/`CS`/`VC` | `"relu"` (the reference default classes); `"sigmoid"` and `"tanh"` are the paper's |
| batch norm | `batch_norm=` on `I`/`CS`/`VC` | `False` — neither reference uses it. `True` puts a `BatchNorm1d` between each hidden layer and its activation, so the fit needs more than one row per batch and inference needs `eval()` mode (`fit` and `load` leave the flow there) |
| transform class | `I(transform=, **kwargs)` (extra kwargs go to the transform class) | `"bernstein"`, `n_coeffs=20` unconstrained coefficients ([zuko-upstream.md](zuko-upstream.md) on the order they give); spline `bins=8`, zuko's NSF default; the domain is fixed at [-5, 5], `transforms.BOUND` |
| shuffling / weight init | `fit(seed=)` / `CausalFlowDAG(seed=)` | init happens at construction — the constructor seed is the reproducibility knob |
| weight init | `CausalFlowDAG(init=)` | `"torch"` (`nn.Linear` Kaiming-uniform); `"glorot"` = Keras `Dense` default, glorot-uniform weights and zero biases — the paper's reference; decisive under its full-batch protocol |
| network inputs | `CI/CS/VC(input_transform=)` | `None`, raw parents; `"minmax"`, `"standardize"` or a callable `fn(x, train)` ([model.md](model.md)) |
