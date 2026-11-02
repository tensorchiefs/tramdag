# Code map: every module of `src/tramdag/`

This page lists each module with its role and the decisions behind it. The
[API pages](api/tramdag.md) document every signature and default. Names in
parentheses are private machinery and are not part of the API.

## `spec.py`: declare the model

Every term is a `Term` subclass under its pythonic name. The paper's symbol is
the same object, so `LS is LinearShift`. A spec is plain data: it holds no
tensors, so it serializes and compares without torch.

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`Term`][tramdag.spec.Term] | One additive term of a node's transformation. `+` on terms builds plain lists. The contract for a custom term is in [architecture.md](architecture.md). |
| [`SI()`][tramdag.spec.SI] / [`CI()`][tramdag.spec.CI] / [`Intercept`][tramdag.spec.Intercept] | `Intercept` (`I`) is the one intercept class. `SI` and `CI` are its two spellings with the arity checked. |
| [`LinearShift`][tramdag.spec.LinearShift] / [`ComplexShift`][tramdag.spec.ComplexShift] / [`VaryingCoefficient`][tramdag.spec.VaryingCoefficient] | The three shifts: an interpretable coefficient, a network, and the penalized treatment-effect head ([varying-coefficients.md](varying-coefficients.md)). |
| [`ContinuousNode`][tramdag.spec.ContinuousNode] / [`OrdinalNode`][tramdag.spec.OrdinalNode] | The two node kinds. |
| [`validate_and_sort()`][tramdag.spec.validate_and_sort] | Edge-ownership validation plus Kahn topological sort. The order makes the flow triangular. Kahn's sort emits ready nodes in sorted batches, so the order is deterministic. `CausalFlowDAG.outcome` gives the one node without children. |
| [`spec_to_dict()`][tramdag.spec.spec_to_dict] / [`spec_from_dict()`][tramdag.spec.spec_from_dict] | Checkpoint (de)serialization. A custom term writes its import path, so a checkpoint loads without a registry. `strict=True` refuses a term that omits a shaping option (`Term.acting_options`). |

## `transforms.py`: the monotone map h and the ordinal transform

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`StandardLogistic`][tramdag.transforms.StandardLogistic] | The TRAM base distribution. zuko has none ([zuko-upstream.md](zuko-upstream.md)). |
| [`BernsteinUT`][tramdag.transforms.BernsteinUT] / [`SplineUT`][tramdag.transforms.SplineUT] / [`AffineUT`][tramdag.transforms.AffineUT] | The three transforms of a continuous node ([model.md](model.md#the-three-knobs-on-a-term)). |
| [`make_univariate_transform()`][tramdag.transforms.make_univariate_transform] | Name to transform instance. |
| [`ordinal_log_prob()`][tramdag.transforms.ordinal_log_prob] and the other `ordinal_*` functions | The ordinal likelihood and its inverses. The log probability stays in log space ([model.md](model.md#ordinal-nodes)). |

## `flow.py`: the model

[`CausalFlowDAG`][tramdag.flow.CausalFlowDAG] holds one
[`Node`][tramdag.nodes.Node] per variable in topological order. Construction
seeds each node's weights from the constructor seed and the node's name. A node
starts from the same weights in every spec that contains it. Construction ends
with `torch.manual_seed(seed)`, which seeds the unseeded shuffling of `fit`. The
constructor seed is the reproducibility knob. A lone
`Node` needs `torch.manual_seed` before you build it. The class composes
`FitMixin` and `ReadoutsMixin`. The modules below list their methods.

- `sample` covers observational, interventional (`do=`, graph mutilation) and
  counterfactual (`u=`) sampling. `abduct` recovers the latents. `pmf` and
  `density` give the analytic conditionals.
- `save` and `load` write checkpoints with the nodes' histories and
  provenance. `load` requires a complete checkpoint and fails loudly
  otherwise.
- The flow calibrates every node before the first fit and rolls the flags
  back when a node refuses the frame. A term names, validates and recomputes
  its own side columns through the `ShiftModule` hooks, and the flow fits the
  binary propensity.

## `modules.py`: the term modules

There is one module class per term, and the term class holds it as `module`
(`ComplexShift.module is ComplexShiftModule`). The module holds the term's
network and owns the runtime hooks ([architecture.md](architecture.md)).

The default architectures replicate the PyTorch reference
[buehlpa/TramDag](https://github.com/buehlpa/TramDag), file `tram_models.py`,
so a fitted model stays comparable to it. They are not the paper's nets, which
[paper-replication.md](paper-replication.md) lists per experiment.

| Name | Role |
|----------------------------------|------------------------------------------------------------------------------|
| [`ShiftModule`][tramdag.modules.ShiftModule] / [`InterceptModule`][tramdag.modules.InterceptModule] | The behavior hooks a term module owns. |
| [`LinearShiftModule`][tramdag.modules.LinearShiftModule] | A linear layer without bias, because the intercept slot owns the constant. `.weight` is the interpretable coefficient. |
| [`ComplexShiftModule`][tramdag.modules.ComplexShiftModule] / [`VaryingCoefficientModule`][tramdag.modules.VaryingCoefficientModule] | The network shift and the varying-coefficient head. |
| [`intercept_module()`][tramdag.modules.intercept_module] | The one factory for the intercept slot: a free theta, one joint net, or one net per parent summed in coefficient space. |

## `nodes.py`: the node model

[`Node`][tramdag.nodes.Node] is one sub-model per variable, a TRAM regression
on its parents. `tramdag.Node` exports it. It builds its terms through
each term's `module`. The continuous-versus-ordinal branches live in its
methods `log_prob`, `sample`, `abduct`, `marginal_theta` and `encode`
([architecture.md](architecture.md#node-kinds)). The shared helpers (`encode`,
`schema_entry`, `tensorize`, the column checks and the checkpoint IO) serve
both the node and the flow.

## `fitting.py`: `NodeFitMixin` and `FitMixin`

[`Node.fit()`][tramdag.nodes.Node.fit] and
[`Node.fit_classical()`][tramdag.nodes.Node.fit_classical] fit one node,
through `NodeFitMixin`. The flow's [`fit()`][tramdag.flow.CausalFlowDAG.fit] and
[`fit_classical()`][tramdag.flow.CausalFlowDAG.fit_classical] loop over the
nodes, through `FitMixin`, serially or in `n_jobs` forked processes. The
mechanics are in [fitting.md](fitting.md).

## `readouts.py`: `ReadoutsMixin`

The read-out methods (`varying_coef`, `ls_coefficients`, `to_matrix`,
`intercept_contributions`, `design_matrix`, `shift_curve`) live here
once and are methods of the flow through the mixin.

## `scores.py`: effect-modifier detection

[`node_scores()`][tramdag.scores.node_scores] and
[`effect_modifier_scan()`][tramdag.scores.effect_modifier_scan] are the
analytic scores and the fluctuation scan ([scores.md](scores.md)). The
per-term score columns come from each term's `score_columns` hook.

## `callbacks.py`: the shipped `fit` callbacks

[`EarlyStopping`][tramdag.callbacks.EarlyStopping] is the only shipped
callback. Anything else is torch's `lr_scheduler` and a few lines of callback
([fitting.md](fitting.md)).

## `plots.py`: the figures

matplotlib is optional (`tramdag[plots]`) and the first plot
call imports it, never the package import. [`plot_dag()`][tramdag.plots.plot_dag],
[`plot_marginals()`][tramdag.plots.plot_marginals],
[`plot_training()`][tramdag.plots.plot_training] and
[`plot_varying_coef()`][tramdag.plots.plot_varying_coef] each take `ax=` to
draw into a given layout. `plot_dag` draws every edge by the term that owns
it.

## What is *not* in the package

Three groups of files are research code:

- the SCM generators
- the frozen datasets
- the replication scripts

They live in [`experiments/`](../experiments/), outside the installed package;
[`experiments/README.md`](../experiments/README.md) describes them and
[`tests/README.md`](../tests/README.md) what the framework tests measure
instead.

## Decisions behind the training defaults

The signatures hold the defaults. This list keeps the reasons.

- **`fit(epochs=)` has no default.** No default epoch count fits both an `LS`
  and a flexible model ([fitting.md](fitting.md)).
- **Weight initialization is `CausalFlowDAG(init=)`.** Initialization happens
  at construction, so the constructor seed is the reproducibility knob. Each
  node draws from a stream keyed on the seed and the node's name.
  `"torch"` is the `nn.Linear` default. `"glorot"` is the Keras `Dense`
  default, the paper's reference, and it is decisive under the paper's
  full-batch protocol. `zero_shift_networks=True` starts every `CS` shift
  at 0.
- **Network widths and `activation=`** follow the default classes of the
  PyTorch reference, for parity. The `VC` head has no counterpart there, so
  its width comes from the recovery measurement. `"sigmoid"` and `"tanh"` are
  the paper's activations.
- **`batch_norm=` is off**, because neither reference uses it. When on, it
  needs more than one row per batch, and inference needs `eval()` mode. `fit`
  and `load` leave the flow there.
- **`marginal_init=` is off.** Only the fit that calibrates a node applies
  it, so a node that `fit_classical` calibrated ignores it.
- **`fit_classical` is one full-batch run with no chunks.** `converged` needs
  the gradient bound as well as torch's tolerance
  ([fitting.md](fitting.md)).
- **The transform domain** is `transforms.BOUND`, a constant. The order of the
  Bernstein coefficients is in [zuko-upstream.md](zuko-upstream.md).
- **A centered `VC` term needs a propensity column** in the training frame
  ([varying-coefficients.md](varying-coefficients.md)).
