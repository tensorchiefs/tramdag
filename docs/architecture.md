# Architecture

The package has ten modules and one rule. Term-specific behavior lives on the
term's two classes: its `Term` subclass (the spec) and its module in
`modules.py`. Node-kind behavior lives in five `Node` methods and two
`nodes` functions. Everything else
is framework code.

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
    subgraph functions["flow behavior by concern"]
        fitting["fitting.py<br/>NodeFitMixin: fit (Adam loop, callbacks),<br/>fit_classical (L-BFGS), nll;<br/>FitMixin: the flow's loops over the nodes"]
        readouts["readouts.py<br/>ReadoutsMixin: ls_coefficients,<br/>varying_coef, to_matrix, contributions,<br/>design_matrix, shift_curve"]
        scores["scores.py<br/>node_scores,<br/>effect_modifier_scan"]
    end
    callbacks["callbacks.py<br/>Callback, EarlyStopping"]
    plots["plots.py<br/>plot_dag, plot_marginals,<br/>plot_training (matplotlib optional)"]

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
    readouts --> spec
    scores --> spec
    scores --> transforms
    plots --> spec
```

`modules.py` imports nothing from `spec.py`: it reads the node's parent schema,
`{parent: "continuous" | n_levels}`, and a term's `parents` and options. That
is what lets each term class hold its module class directly
(`LinearShift.module is LinearShiftModule`). `fitting.py` and `readouts.py`
are mixins: `Node` composes `NodeFitMixin`, and `CausalFlowDAG` composes
`FitMixin` and `ReadoutsMixin`. `fitting` imports `flow` and `nodes` under
`TYPE_CHECKING` only, plus `nodes.load_weights` lazily in
`FitMixin._fit_nodes`. So the import graph is acyclic at module load,
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
        __repr__(): the only one — every entry as name=value
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

The two node kinds are continuous and ordinal. They stay an if/else in ONE
place: the five `Node` methods `log_prob`, `sample`, `abduct`,
`marginal_theta` and `encode` in nodes.py, plus the column encoding
`nodes.encode` and the schema entry `nodes.schema_entry` they share with the
parent features.

## Guards that pin all of this

- `tests/tools/statedict_smoke.py` — seeded per-DGP state dicts, bit-compared.
- The inline DGP truths (`tests/conftest.py`) and 45+ regex-pinned refusals.
- `experiments/ground_truth/*.json` — eight CI-checked replications with
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
  fitting --> nodes
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
    n24["EarlyStopping.on_fit_end"]
    n3["_last_val"]
  end
  subgraph fitting
    n5["FitMixin._check_frames"]
    n8["FitMixin._fit_nodes"]
    n10["FitMixin.fit"]
    n6["NodeFitMixin._check_side_columns"]
    n20["NodeFitMixin._recenter"]
    n9["NodeFitMixin.fit"]
    n25["NodeFitMixin.row_log_prob"]
    n11["NodeFitMixin.side_columns"]
    n12["_check_fit_sizes"]
    n13["_check_optimizer_not_a_class"]
    n26["_fit_epoch"]
    n27["_learning_rates"]
    n28["_log_epoch"]
    n14["_normalize_callbacks"]
    n15["_per_node"]
    n16["_split_validation"]
  end
  subgraph flow
    n17["CausalFlowDAG._calibrate"]
    n36["CausalFlowDAG._dtype"]
    n7["CausalFlowDAG._tensorize"]
  end
  subgraph modules
    n39["ComplexShiftModule.forward"]
    n38["ComplexShiftModule.shift_value"]
    n41["InterceptModule.calibrate_intercept"]
    n45["LinearShiftModule.forward"]
    n44["LinearShiftModule.shift_value"]
    n21["ShiftModule.finalize"]
    n29["ShiftModule.regularizer"]
    n18["ShiftModule.side_columns"]
    n47["SimpleInterceptModule.forward"]
    n46["SimpleInterceptModule.theta_value"]
    n42["TermModule.calibrate"]
    n48["TermModule.input_transform"]
    n51["VaryingCoefficientModule.beta"]
    n22["VaryingCoefficientModule.finalize"]
    n50["VaryingCoefficientModule.forward"]
    n52["VaryingCoefficientModule.l2"]
    n49["VaryingCoefficientModule.recenter"]
    n54["VaryingCoefficientModule.regressor"]
    n30["VaryingCoefficientModule.regularizer"]
    n53["VaryingCoefficientModule.shift_value"]
    n19["VaryingCoefficientModule.side_columns"]
  end
  subgraph nodes
    n31["Node.calibrate"]
    n23["Node.features"]
    n33["Node.log_prob"]
    n40["Node.net_input"]
    n32["Node.tensorize"]
    n34["Node.theta_shift"]
    n35["check_columns"]
    n59["check_level_values"]
    n55["encode"]
    n37["tensorize"]
  end
  subgraph transforms
    n60["BernsteinUT._build"]
    n56["StandardLogistic.log_prob"]
    n61["_ScaledUT._log_dt_dx"]
    n62["_ScaledUT._scale"]
    n57["_ScaledUT.forward"]
    n43["_ScaledUT.set_range"]
    n65["_log1mexp"]
    n63["ordinal_bounds"]
    n64["ordinal_cutpoints"]
    n58["ordinal_log_prob"]
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
    n10 --> n13
    n10 -- "4x" --> n14
    n10 -- "9x" --> n15
    n10 --> n16
    n10 --> n17
    n6 -- "6x" --> n11
    n6 -- "4x" --> n18
    n6 -- "2x" --> n19
    n20 -- "2x" --> n21
    n20 --> n22
    n20 -- "3x" --> n23
    n9 -- "9x" --> n2
    n9 -- "3x" --> n4
    n9 -- "3x" --> n24
    n9 -- "3x" --> n6
    n9 -- "3x" --> n20
    n9 -- "9x" --> n25
    n9 -- "3x" --> n12
    n9 -- "3x" --> n13
    n9 -- "9x" --> n26
    n9 -- "9x" --> n27
    n9 -- "9x" --> n28
    n9 -- "3x" --> n14
    n9 -- "3x" --> n16
    n9 -- "2x" --> n29
    n9 --> n30
    n9 -- "3x" --> n31
    n9 -- "6x" --> n32
    n25 -- "27x" --> n11
    n25 -- "27x" --> n23
    n25 -- "27x" --> n33
    n25 -- "27x" --> n34
    n11 -- "24x" --> n18
    n11 -- "12x" --> n19
    n26 -- "18x" --> n25
    n26 -- "6x" --> n30
    n17 -- "3x" --> n31
    n17 --> n35
    n7 -- "2x" --> n36
    n7 -- "2x" --> n37
    n38 -- "9x" --> n39
    n38 -- "9x" --> n40
    n41 -- "3x" --> n42
    n41 -- "2x" --> n43
    n44 -- "9x" --> n45
    n46 -- "27x" --> n47
    n42 -- "6x" --> n48
    n22 --> n49
    n22 --> n40
    n50 -- "9x" --> n51
    n30 -- "7x" --> n52
    n53 -- "9x" --> n50
    n53 -- "9x" --> n54
    n53 -- "9x" --> n40
    n31 -- "3x" --> n41
    n31 -- "3x" --> n42
    n31 -- "3x" --> n32
    n23 -- "30x" --> n55
    n33 -- "18x" --> n56
    n33 -- "18x" --> n57
    n33 -- "9x" --> n58
    n40 -- "19x" --> n48
    n32 -- "9x" --> n37
    n34 -- "9x" --> n38
    n34 -- "9x" --> n44
    n34 -- "27x" --> n46
    n34 -- "9x" --> n53
    n37 -- "11x" --> n35
    n37 -- "8x" --> n59
    n57 -- "18x" --> n60
    n57 -- "18x" --> n61
    n57 -- "18x" --> n62
    n63 -- "9x" --> n64
    n58 -- "9x" --> n65
    n58 -- "9x" --> n63
```
<!-- AUTOGEN:end -->
