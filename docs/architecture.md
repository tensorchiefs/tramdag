# Architecture

The package has ten modules and one rule. Term-specific behavior lives on the
term's two classes: its `Term` subclass (the spec) and its module in
`modules.py`. Node-kind maths lives in five `Node` methods, `nodes.encode`
and the score derivative `scores._dl_ds`. Everything else is framework code.

## Module map
```mermaid
graph TD
    subgraph data["spec"]
        spec["spec.py<br/>DSL: Term + one subclass per term<br/>(Intercept LinearShift ComplexShift<br/>VaryingCoefficient = I LS CS VC),<br/>each holding its module class;<br/>nodes, normalization, Kahn sort, (de)serialization"]
    end
    subgraph torch["torch modules"]
        modules["modules.py<br/>one nn.Module per term, built from (term, schema):<br/>ShiftModule/InterceptModule hooks,<br/>intercept_module, feat_width, _InputTransform;<br/>LinearShiftModule ComplexShiftModule<br/>VaryingCoefficientModule,<br/>SimpleInterceptModule ComplexInterceptModule<br/>AdditiveInterceptModule"]
        transforms["transforms.py<br/>Bernstein/Spline/Affine,<br/>ordinal_* likelihood,<br/>StandardLogistic"]
        nodes["nodes.py<br/>Node: intercept + shifts,<br/>encode, log_prob, sample,<br/>abduct, marginal_theta,<br/>calibrate, save/load"]
        flow["flow.py<br/>CausalFlowDAG: construct, calibrate (private),<br/>log_prob, sample/abduct/pmf/density,<br/>save/load; composes the mixins"]
    end
    subgraph functions["behavior by concern"]
        fitting["fitting.py<br/>NodeFitMixin: fit (Adam loop, callbacks),<br/>fit_classical (L-BFGS), nll;<br/>FitMixin: the flow's loops over the nodes"]
        readouts["readouts.py<br/>ReadoutsMixin: ls_coefficients,<br/>varying_coef, to_matrix, contributions,<br/>design_matrix, shift_curve"]
        scores["scores.py<br/>node_scores,<br/>effect_modifier_scan"]
    end
    callbacks["callbacks.py<br/>Callback, EarlyStopping"]
    plots["plots.py<br/>plot_dag, plot_marginals,<br/>plot_training, plot_varying_coef<br/>(matplotlib optional)"]

    spec --> modules
    nodes --> spec
    nodes --> transforms
    nodes --> fitting
    nodes --> modules
    flow --> nodes
    flow --> modules
    flow --> spec
    flow --> transforms
    flow --> fitting
    flow --> readouts
    flow --> scores
    fitting --> callbacks
    readouts --> modules
    scores --> transforms
    plots --> spec
```

`modules.py` imports nothing from `spec.py` at run time: it reads the node's parent schema,
`{parent: "continuous" | n_levels}`, and a term's `parents` and options. That
is what lets each term class hold its module class directly
(`LinearShift.module is LinearShiftModule`). `fitting.py` and `readouts.py`
are mixins: `Node` composes `NodeFitMixin`, and `CausalFlowDAG` composes
`FitMixin` and `ReadoutsMixin`. `fitting` imports `flow` and `nodes` under
`TYPE_CHECKING` only. So the import graph is acyclic at module load,
although the package UML draws both directions between `fitting` and `nodes`.

## The term contract
```mermaid
classDiagram
    class Term {
        <<spec.py, plain data>>
        name: class attribute, the wire key
        module: class attribute, the module class
        parents
        __init__(*parents): the base assigns the parents
        options: each subclass's keyword arguments, assigned to self
        __repr__(): the only one, every entry as name=value
        check(name, spec)
        edge_parents
        cells()
        classical
        options() / from_serialized()
    }
    Term <|-- Intercept : I
    Term <|-- LinearShift : LS
    Term <|-- ComplexShift : CS
    Term <|-- VaryingCoefficient : VC
    class TermModule {
        <<modules.py, nn.Module>>
        input_transform / calibrate(train_df)
    }
    class ShiftModule {
        __init__(term, schema): builds the net, sets key and parents
        order
        shift_value(node, feats)
        post_init()
        regularizer() -> Tensor | None
        finalize(node, feats)
        score_columns(node, feats, dlds)
        side_columns() / check_column() / live_side() / extra_columns()
    }
    class InterceptModule {
        __init__(term, schema, n_params): intercept_module picks the class
        groups / ci_parents
        calibrate_intercept(train_df, own, ut)
        theta_value(node, feats, n)
        marginal_start(theta)
    }
    TermModule <|-- ShiftModule
    TermModule <|-- InterceptModule
    ShiftModule <|-- LinearShiftModule
    ShiftModule <|-- ComplexShiftModule
    ShiftModule <|-- VaryingCoefficientModule
    InterceptModule <|-- SimpleInterceptModule
    InterceptModule <|-- ComplexInterceptModule
    InterceptModule <|-- AdditiveInterceptModule
```

Each module builds its layers in a fixed order under fixed attribute names, so
the state-dict paths (`nodes.<n>.shifts.<key>.…`) and the seeded RNG stream are
stable; `tests/test_statedict_stability.py` pins them.

A custom term is two classes:

- a `ShiftModule` subclass with `__init__(term, schema)`, which builds the
  network from the parent schema and sets `key` and `parents`, and `shift_value`;
- a `tramdag.Term` subclass with `name` and `module` as class attributes
  (`__init_subclass__` refuses one without). Its options are the keyword
  arguments of its `__init__`, assigned to `self` after
  `super().__init__(*parents)`; its rules are the checks after that, plus
  `check` and `edge_parents`.

## Node kinds

The two node kinds are continuous and ordinal. The maths that differs by
kind is an if/else in three places:

- the five `Node` methods `log_prob`, `sample`, `abduct`, `marginal_theta`
  and `encode`, plus the transform choice in `Node.__init__`;
- the column encoding `nodes.encode`;
- the score derivative `scores._dl_ds`.

Other code reads the kind only to check, convert or display values.
`flow.py` checks ordinal levels, sets the output dtype and limits `pmf` and
`density` to their kind. `readouts.py` refuses an ordinal parent in
`shift_curve` and keeps a continuous parent raw in `design_matrix`.
`effect_modifier_scan` finds the score column of a binary ordinal
treatment. `spec.py` validates and serializes the kind, and `plots.py`
styles the node.
The term modules read a node's schema entry (`nodes.schema_entry`):
`"continuous"` or a level count.

## Guards that pin all of this

- `tests/tools/statedict_smoke.py`: seeded per-DGP state dicts, bit-compared.
- The inline DGP truths (`tests/conftest.py`) and 45+ regex-pinned refusals.
- `experiments/ground_truth/*.json`: eight CI-checked replications with
  wall-time tripwires. Centers move only with a documented reason.

<!-- AUTOGEN:diagrams (tools/gen_diagrams.py) — do not edit by hand -->
## Generated views

To regenerate this section, run ``uv run python tools/gen_diagrams.py``.

The package UML and the class UML come from pyreverse. The class diagram
carries names and inheritance only. The call graphs come from a profile trace
of one flow construction and one three-epoch ``fit`` on a 3-node SI/LS/CS/VC
spec. The graphs keep tramdag-internal edges only, and ``3x`` means once per
node.

### Package UML (pyreverse)

```mermaid
classDiagram
  class tramdag {
  }
  class callbacks {
  }
  class fitting {
  }
  class flow {
  }
  class modules {
  }
  class nodes {
  }
  class plots {
  }
  class readouts {
  }
  class scores {
  }
  class spec {
  }
  class transforms {
  }
  tramdag --> flow
  tramdag --> nodes
  tramdag --> plots
  tramdag --> spec
  fitting --> callbacks
  flow --> fitting
  flow --> modules
  flow --> nodes
  flow --> readouts
  flow --> spec
  flow --> transforms
  nodes --> fitting
  nodes --> modules
  nodes --> spec
  nodes --> transforms
  plots --> spec
  readouts --> modules
  scores --> transforms
  spec --> modules
  fitting ..> flow
  fitting ..> nodes
  modules ..> nodes
  modules ..> spec
```

### Class UML (pyreverse)

Each built-in term module holds its network and takes its contract from
``ShiftModule``/``InterceptModule``.

```mermaid
classDiagram
  class AdditiveInterceptModule {
  }
  class AffineUT {
  }
  class BernsteinUT {
  }
  class Callback {
  }
  class CausalFlowDAG {
  }
  class ComplexInterceptModule {
  }
  class ComplexShift {
  }
  class ComplexShiftModule {
  }
  class ContinuousNode {
  }
  class EarlyStopping {
  }
  class FitMixin {
  }
  class Intercept {
  }
  class InterceptModule {
  }
  class LinearShift {
  }
  class LinearShiftModule {
  }
  class Node {
  }
  class NodeFitMixin {
  }
  class OrdinalNode {
  }
  class ReadoutsMixin {
  }
  class ShiftModule {
  }
  class SimpleInterceptModule {
  }
  class SplineUT {
  }
  class StandardLogistic {
  }
  class Term {
  }
  class TermModule {
  }
  class VaryingCoefficient {
  }
  class VaryingCoefficientModule {
  }
  class _FnCallback {
  }
  class _InputTransform {
  }
  class _ScaledUT {
  }
  EarlyStopping --|> Callback
  _FnCallback --|> Callback
  CausalFlowDAG --|> FitMixin
  CausalFlowDAG --|> ReadoutsMixin
  AdditiveInterceptModule --|> InterceptModule
  ComplexInterceptModule --|> InterceptModule
  ComplexShiftModule --|> ShiftModule
  InterceptModule --|> TermModule
  LinearShiftModule --|> ShiftModule
  ShiftModule --|> TermModule
  SimpleInterceptModule --|> InterceptModule
  VaryingCoefficientModule --|> ShiftModule
  Node --|> NodeFitMixin
  ComplexShift --|> Term
  Intercept --|> Term
  LinearShift --|> Term
  VaryingCoefficient --|> Term
  AffineUT --|> _ScaledUT
  BernsteinUT --|> _ScaledUT
  SplineUT --|> _ScaledUT
  ComplexShiftModule --o ComplexShift : module
  LinearShiftModule --o LinearShift : module
  VaryingCoefficientModule --o VaryingCoefficient : module
```

### Call graph — flow construction (traced)

```mermaid
flowchart LR
  subgraph flow
    n0["CausalFlowDAG.__init__"]
    n1["CausalFlowDAG._apply_init"]
    n2["CausalFlowDAG._schema"]
  end
  subgraph modules
    n7["ComplexShiftModule.__init__"]
    n11["LinearShiftModule.__init__"]
    n14["SimpleInterceptModule.__init__"]
    n12["VaryingCoefficientModule.__init__"]
    n8["_attach_input_transform"]
    n9["_nn"]
    n10["feat_width"]
    n13["intercept_module"]
  end
  subgraph nodes
    n3["Node.__init__"]
    n5["schema_entry"]
  end
  subgraph spec
    n18["Term.check"]
    n19["Term.edge_parents"]
    n20["VaryingCoefficient.check"]
    n21["VaryingCoefficient.edge_parents"]
    n17["_check_node"]
    n22["_kahn_sort"]
    n6["node_parents"]
    n4["validate_and_sort"]
  end
  subgraph transforms
    n23["BernsteinUT.__init__"]
    n15["BernsteinUT.n_params"]
    n24["_ScaledUT.__init__"]
    n16["make_univariate_transform"]
  end
    n0 --> n1
    n0 -- "3x" --> n2
    n0 -- "3x" --> n3
    n0 --> n4
    n2 -- "3x" --> n5
    n2 -- "3x" --> n6
    n7 --> n8
    n7 --> n9
    n7 --> n10
    n11 --> n10
    n12 --> n8
    n12 --> n9
    n12 --> n10
    n13 -- "3x" --> n14
    n3 --> n7
    n3 --> n11
    n3 --> n12
    n3 -- "3x" --> n13
    n3 -- "3x" --> n5
    n3 -- "3x" --> n6
    n3 -- "2x" --> n15
    n3 -- "2x" --> n16
    n17 -- "5x" --> n18
    n17 -- "5x" --> n19
    n17 --> n20
    n17 --> n21
    n22 -- "3x" --> n6
    n4 -- "3x" --> n17
    n4 --> n22
    n23 -- "2x" --> n24
    n16 -- "2x" --> n23
```

### Call graph — one fit (traced)

```mermaid
flowchart LR
  subgraph callbacks
    n0["EarlyStopping.__init__"]
    n1["EarlyStopping._reset"]
    n2["EarlyStopping.on_epoch_end"]
    n4["EarlyStopping.on_fit_begin"]
    n23["EarlyStopping.on_fit_end"]
    n3["_last_val"]
  end
  subgraph fitting
    n5["FitMixin._check_frames"]
    n8["FitMixin._fit_nodes"]
    n10["FitMixin.fit"]
    n6["NodeFitMixin._check_side_columns"]
    n19["NodeFitMixin._recenter"]
    n9["NodeFitMixin.fit"]
    n24["NodeFitMixin.row_log_prob"]
    n11["NodeFitMixin.side_columns"]
    n12["_check_fit_sizes"]
    n25["_fit_epoch"]
    n26["_learning_rates"]
    n27["_log_epoch"]
    n13["_normalize_callbacks"]
    n14["_per_node"]
    n15["_split_validation"]
  end
  subgraph flow
    n16["CausalFlowDAG._calibrate"]
    n35["CausalFlowDAG._dtype"]
    n7["CausalFlowDAG._tensorize"]
  end
  subgraph modules
    n38["ComplexShiftModule.forward"]
    n37["ComplexShiftModule.shift_value"]
    n40["InterceptModule.calibrate_intercept"]
    n44["LinearShiftModule.forward"]
    n43["LinearShiftModule.shift_value"]
    n20["ShiftModule.finalize"]
    n28["ShiftModule.regularizer"]
    n17["ShiftModule.side_columns"]
    n46["SimpleInterceptModule.forward"]
    n45["SimpleInterceptModule.theta_value"]
    n41["TermModule.calibrate"]
    n47["TermModule.input_transform"]
    n50["VaryingCoefficientModule.beta"]
    n21["VaryingCoefficientModule.finalize"]
    n49["VaryingCoefficientModule.forward"]
    n51["VaryingCoefficientModule.l2"]
    n48["VaryingCoefficientModule.recenter"]
    n53["VaryingCoefficientModule.regressor"]
    n29["VaryingCoefficientModule.regularizer"]
    n52["VaryingCoefficientModule.shift_value"]
    n18["VaryingCoefficientModule.side_columns"]
  end
  subgraph nodes
    n30["Node.calibrate"]
    n22["Node.features"]
    n32["Node.log_prob"]
    n39["Node.net_input"]
    n31["Node.tensorize"]
    n33["Node.theta_shift"]
    n34["check_columns"]
    n58["check_level_values"]
    n54["encode"]
    n36["tensorize"]
  end
  subgraph transforms
    n59["BernsteinUT._build"]
    n55["StandardLogistic.log_prob"]
    n60["_ScaledUT._log_dt_dx"]
    n61["_ScaledUT._scale"]
    n56["_ScaledUT.forward"]
    n42["_ScaledUT.set_range"]
    n64["_log1mexp"]
    n62["ordinal_bounds"]
    n63["ordinal_cutpoints"]
    n57["ordinal_log_prob"]
  end
    n0 --> n1
    n2 -- "9x" --> n3
    n4 -- "3x" --> n1
    n5 -- "3x" --> n6
    n5 -- "2x" --> n7
    n8 -- "3x" --> n9
    n10 --> n5
    n10 -- "2x" --> n8
    n10 -- "3x" --> n11
    n10 -- "3x" --> n12
    n10 -- "4x" --> n13
    n10 -- "9x" --> n14
    n10 --> n15
    n10 --> n16
    n6 -- "6x" --> n11
    n6 -- "4x" --> n17
    n6 -- "2x" --> n18
    n19 -- "2x" --> n20
    n19 --> n21
    n19 -- "3x" --> n22
    n9 -- "9x" --> n2
    n9 -- "3x" --> n4
    n9 -- "3x" --> n23
    n9 -- "3x" --> n6
    n9 -- "3x" --> n19
    n9 -- "9x" --> n24
    n9 -- "3x" --> n12
    n9 -- "9x" --> n25
    n9 -- "9x" --> n26
    n9 -- "9x" --> n27
    n9 -- "3x" --> n13
    n9 -- "3x" --> n15
    n9 -- "2x" --> n28
    n9 --> n29
    n9 -- "3x" --> n30
    n9 -- "6x" --> n31
    n24 -- "27x" --> n11
    n24 -- "27x" --> n22
    n24 -- "27x" --> n32
    n24 -- "27x" --> n33
    n11 -- "24x" --> n17
    n11 -- "12x" --> n18
    n25 -- "18x" --> n24
    n25 -- "6x" --> n29
    n16 -- "3x" --> n30
    n16 --> n34
    n7 -- "2x" --> n35
    n7 -- "2x" --> n36
    n37 -- "9x" --> n38
    n37 -- "9x" --> n39
    n40 -- "3x" --> n41
    n40 -- "2x" --> n42
    n43 -- "9x" --> n44
    n45 -- "27x" --> n46
    n41 -- "6x" --> n47
    n21 --> n48
    n21 --> n39
    n49 -- "9x" --> n50
    n29 -- "7x" --> n51
    n52 -- "9x" --> n49
    n52 -- "9x" --> n53
    n52 -- "9x" --> n39
    n30 -- "3x" --> n40
    n30 -- "3x" --> n41
    n30 -- "3x" --> n31
    n22 -- "30x" --> n54
    n32 -- "18x" --> n55
    n32 -- "18x" --> n56
    n32 -- "9x" --> n57
    n39 -- "19x" --> n47
    n31 -- "9x" --> n36
    n33 -- "9x" --> n37
    n33 -- "9x" --> n43
    n33 -- "27x" --> n45
    n33 -- "9x" --> n52
    n36 -- "11x" --> n34
    n36 -- "8x" --> n58
    n56 -- "18x" --> n59
    n56 -- "18x" --> n60
    n56 -- "18x" --> n61
    n62 -- "9x" --> n63
    n57 -- "9x" --> n64
    n57 -- "9x" --> n62
```
<!-- AUTOGEN:end -->
