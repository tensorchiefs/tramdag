# Architecture

The package has ten modules and one rule. Term-specific behavior lives on the
term's two classes: its `Term` subclass (the spec) and its module in
`modules.py`. Node-kind behavior lives in five `Node` methods. Everything else
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
    flow --> nodes
    flow --> modules
    flow --> spec
    flow --> transforms
    flow --> fitting
    flow --> readouts
    flow --> scores
    fitting --> callbacks
    fitting --> modules
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
`FitMixin._fit_nodes`. The import graph is acyclic.

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
  fitting --> modules
  fitting --> nodes
  flow --> fitting
  flow --> modules
  flow --> nodes
  flow --> readouts
  flow --> spec
  flow --> transforms
  nodes --> fitting
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
    n21["EarlyStopping.on_fit_end"]
    n3["_last_val"]
  end
  subgraph fitting
    n5["FitMixin._fit_nodes"]
    n7["FitMixin.fit"]
    n8["NodeFitMixin._check_side_columns"]
    n17["NodeFitMixin.finalize"]
    n6["NodeFitMixin.fit"]
    n22["NodeFitMixin.row_log_prob"]
    n9["NodeFitMixin.side_columns"]
    n10["_check_fit_sizes"]
    n11["_check_optimizer_not_a_class"]
    n23["_fit_epoch"]
    n24["_learning_rates"]
    n25["_log_epoch"]
    n26["_normalize_callbacks"]
    n12["_per_node"]
    n13["_split_validation"]
  end
  subgraph flow
    n33["CausalFlowDAG._dtype"]
    n14["CausalFlowDAG._tensorize"]
  end
  subgraph modules
    n36["ComplexShiftModule.forward"]
    n35["ComplexShiftModule.shift_value"]
    n38["InterceptModule.calibrate_intercept"]
    n42["LinearShiftModule.forward"]
    n41["LinearShiftModule.shift_value"]
    n18["ShiftModule.finalize"]
    n27["ShiftModule.regularizer"]
    n15["ShiftModule.side_columns"]
    n44["SimpleInterceptModule.forward"]
    n43["SimpleInterceptModule.theta_value"]
    n39["TermModule.calibrate"]
    n45["TermModule.input_transform"]
    n48["VaryingCoefficientModule.beta"]
    n19["VaryingCoefficientModule.finalize"]
    n47["VaryingCoefficientModule.forward"]
    n49["VaryingCoefficientModule.l2"]
    n46["VaryingCoefficientModule.recenter"]
    n51["VaryingCoefficientModule.regressor"]
    n28["VaryingCoefficientModule.regularizer"]
    n50["VaryingCoefficientModule.shift_value"]
    n16["VaryingCoefficientModule.side_columns"]
  end
  subgraph nodes
    n29["Node.calibrate"]
    n20["Node.features"]
    n31["Node.log_prob"]
    n37["Node.net_input"]
    n30["Node.tensorize"]
    n32["Node.theta_shift"]
    n56["check_columns"]
    n57["check_level_values"]
    n52["encode"]
    n34["tensorize"]
  end
  subgraph transforms
    n58["BernsteinUT._build"]
    n53["StandardLogistic.log_prob"]
    n59["_ScaledUT._log_dt_dx"]
    n60["_ScaledUT._scale"]
    n54["_ScaledUT.forward"]
    n40["_ScaledUT.set_range"]
    n63["_log1mexp"]
    n61["ordinal_bounds"]
    n62["ordinal_cutpoints"]
    n55["ordinal_log_prob"]
  end
    n0 --> n1
    n2 -- "9x" --> n3
    n4 -- "3x" --> n1
    n5 -- "3x" --> n6
    n7 -- "2x" --> n5
    n7 -- "3x" --> n8
    n7 -- "3x" --> n9
    n7 -- "3x" --> n10
    n7 --> n11
    n7 -- "9x" --> n12
    n7 --> n13
    n7 -- "2x" --> n14
    n8 -- "6x" --> n9
    n8 -- "4x" --> n15
    n8 -- "2x" --> n16
    n17 -- "2x" --> n18
    n17 --> n19
    n17 -- "3x" --> n20
    n6 -- "9x" --> n2
    n6 -- "3x" --> n4
    n6 -- "3x" --> n21
    n6 -- "3x" --> n8
    n6 -- "3x" --> n17
    n6 -- "9x" --> n22
    n6 -- "3x" --> n10
    n6 -- "3x" --> n11
    n6 -- "9x" --> n23
    n6 -- "9x" --> n24
    n6 -- "9x" --> n25
    n6 -- "3x" --> n26
    n6 -- "3x" --> n13
    n6 -- "2x" --> n27
    n6 --> n28
    n6 -- "3x" --> n29
    n6 -- "6x" --> n30
    n22 -- "27x" --> n9
    n22 -- "27x" --> n20
    n22 -- "27x" --> n31
    n22 -- "27x" --> n32
    n9 -- "24x" --> n15
    n9 -- "12x" --> n16
    n23 -- "18x" --> n22
    n23 -- "6x" --> n28
    n14 -- "2x" --> n33
    n14 -- "2x" --> n34
    n35 -- "9x" --> n36
    n35 -- "9x" --> n37
    n38 -- "3x" --> n39
    n38 -- "2x" --> n40
    n41 -- "9x" --> n42
    n43 -- "27x" --> n44
    n39 -- "6x" --> n45
    n19 --> n46
    n19 --> n37
    n47 -- "9x" --> n48
    n28 -- "7x" --> n49
    n50 -- "9x" --> n47
    n50 -- "9x" --> n51
    n50 -- "9x" --> n37
    n29 -- "3x" --> n38
    n29 -- "3x" --> n39
    n29 -- "3x" --> n30
    n20 -- "30x" --> n52
    n31 -- "18x" --> n53
    n31 -- "18x" --> n54
    n31 -- "9x" --> n55
    n37 -- "19x" --> n45
    n30 -- "9x" --> n34
    n32 -- "9x" --> n35
    n32 -- "9x" --> n41
    n32 -- "27x" --> n43
    n32 -- "9x" --> n50
    n34 -- "11x" --> n56
    n34 -- "8x" --> n57
    n54 -- "18x" --> n58
    n54 -- "18x" --> n59
    n54 -- "18x" --> n60
    n61 -- "9x" --> n62
    n55 -- "9x" --> n63
    n55 -- "9x" --> n61
```
<!-- AUTOGEN:end -->
