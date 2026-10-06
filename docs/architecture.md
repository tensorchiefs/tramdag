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
`FitMixin` and `ReadoutsMixin`. `fitting` imports `flow` under
`TYPE_CHECKING` only, and `nodes` inside the two flow methods that check
columns. The import graph is acyclic.

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
        score_columns(node, flow, feats, dlds)
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
`marginal_theta` and `encode` in nodes.py.

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
    n6["ComplexShiftModule.__init__"]
    n10["LinearShiftModule.__init__"]
    n13["SimpleInterceptModule.__init__"]
    n11["VaryingCoefficientModule.__init__"]
    n7["_attach_input_transform"]
    n8["_nn"]
    n9["feat_width"]
    n12["intercept_module"]
  end
  subgraph nodes
    n3["Node.__init__"]
  end
  subgraph spec
    n17["Term.check"]
    n18["Term.edge_parents"]
    n19["VaryingCoefficient.check"]
    n20["VaryingCoefficient.edge_parents"]
    n16["_check_node"]
    n21["_kahn_sort"]
    n5["node_parents"]
    n4["validate_and_sort"]
  end
  subgraph transforms
    n22["BernsteinUT.__init__"]
    n14["BernsteinUT.n_params"]
    n23["_ScaledUT.__init__"]
    n15["make_univariate_transform"]
  end
    n0 --> n1
    n0 -- "3x" --> n2
    n0 -- "3x" --> n3
    n0 --> n4
    n2 -- "3x" --> n5
    n6 --> n7
    n6 --> n8
    n6 --> n9
    n10 --> n9
    n11 --> n7
    n11 --> n8
    n11 --> n9
    n12 -- "3x" --> n13
    n3 --> n6
    n3 --> n10
    n3 --> n11
    n3 -- "3x" --> n12
    n3 -- "3x" --> n5
    n3 -- "2x" --> n14
    n3 -- "2x" --> n15
    n16 -- "5x" --> n17
    n16 -- "5x" --> n18
    n16 --> n19
    n16 --> n20
    n21 -- "3x" --> n5
    n4 -- "3x" --> n16
    n4 --> n21
    n22 -- "2x" --> n23
    n15 -- "2x" --> n22
```

### Call graph — one fit (traced)

```mermaid
flowchart LR
  subgraph callbacks
    n0["EarlyStopping.__init__"]
    n1["EarlyStopping._reset"]
    n2["EarlyStopping.on_epoch_end"]
    n4["EarlyStopping.on_fit_begin"]
    n19["EarlyStopping.on_fit_end"]
    n3["_last_val"]
  end
  subgraph fitting
    n5["FitMixin._fit_nodes"]
    n7["FitMixin.fit"]
    n8["NodeFitMixin._check_side_columns"]
    n15["NodeFitMixin.finalize"]
    n6["NodeFitMixin.fit"]
    n20["NodeFitMixin.row_log_prob"]
    n9["NodeFitMixin.side_columns"]
    n21["_check_fit_sizes"]
    n22["_fit_epoch"]
    n23["_learning_rates"]
    n24["_log_epoch"]
    n25["_normalize_callbacks"]
    n10["_per_node"]
    n11["_split_validation"]
  end
  subgraph modules
    n33["ComplexShiftModule.forward"]
    n32["ComplexShiftModule.shift_value"]
    n35["InterceptModule.calibrate_intercept"]
    n39["LinearShiftModule.forward"]
    n38["LinearShiftModule.shift_value"]
    n16["ShiftModule.finalize"]
    n26["ShiftModule.regularizer"]
    n13["ShiftModule.side_columns"]
    n41["SimpleInterceptModule.forward"]
    n40["SimpleInterceptModule.theta_value"]
    n36["TermModule.calibrate"]
    n42["TermModule.input_transform"]
    n45["VaryingCoefficientModule.beta"]
    n17["VaryingCoefficientModule.finalize"]
    n44["VaryingCoefficientModule.forward"]
    n46["VaryingCoefficientModule.l2"]
    n43["VaryingCoefficientModule.recenter"]
    n48["VaryingCoefficientModule.regressor"]
    n27["VaryingCoefficientModule.regularizer"]
    n47["VaryingCoefficientModule.shift_value"]
    n14["VaryingCoefficientModule.side_columns"]
  end
  subgraph nodes
    n28["Node.calibrate"]
    n18["Node.features"]
    n30["Node.log_prob"]
    n34["Node.net_input"]
    n29["Node.tensorize"]
    n31["Node.theta_shift"]
    n12["check_columns"]
    n49["check_level_values"]
    n50["encode"]
  end
  subgraph transforms
    n54["BernsteinUT._build"]
    n51["StandardLogistic.log_prob"]
    n55["_ScaledUT._log_dt_dx"]
    n56["_ScaledUT._scale"]
    n52["_ScaledUT.forward"]
    n37["_ScaledUT.set_range"]
    n59["_log1mexp"]
    n57["ordinal_bounds"]
    n58["ordinal_cutpoints"]
    n53["ordinal_log_prob"]
  end
    n0 --> n1
    n2 -- "9x" --> n3
    n4 -- "3x" --> n1
    n5 -- "3x" --> n6
    n7 -- "2x" --> n5
    n7 -- "3x" --> n8
    n7 -- "3x" --> n9
    n7 -- "9x" --> n10
    n7 --> n11
    n7 --> n12
    n8 -- "6x" --> n9
    n8 -- "4x" --> n13
    n8 -- "2x" --> n14
    n15 -- "2x" --> n16
    n15 --> n17
    n15 -- "3x" --> n18
    n6 -- "9x" --> n2
    n6 -- "3x" --> n4
    n6 -- "3x" --> n19
    n6 -- "3x" --> n8
    n6 -- "3x" --> n15
    n6 -- "9x" --> n20
    n6 -- "3x" --> n21
    n6 -- "9x" --> n22
    n6 -- "9x" --> n23
    n6 -- "9x" --> n24
    n6 -- "3x" --> n25
    n6 -- "3x" --> n11
    n6 -- "2x" --> n26
    n6 --> n27
    n6 -- "3x" --> n28
    n6 -- "6x" --> n29
    n20 -- "27x" --> n9
    n20 -- "27x" --> n18
    n20 -- "27x" --> n30
    n20 -- "27x" --> n31
    n9 -- "24x" --> n13
    n9 -- "12x" --> n14
    n22 -- "18x" --> n20
    n22 -- "6x" --> n27
    n32 -- "9x" --> n33
    n32 -- "9x" --> n34
    n35 -- "3x" --> n36
    n35 -- "2x" --> n37
    n38 -- "9x" --> n39
    n40 -- "27x" --> n41
    n36 -- "6x" --> n42
    n17 --> n43
    n17 --> n34
    n44 -- "9x" --> n45
    n27 -- "7x" --> n46
    n47 -- "9x" --> n44
    n47 -- "9x" --> n48
    n47 -- "9x" --> n34
    n28 -- "3x" --> n35
    n28 -- "3x" --> n36
    n28 -- "3x" --> n12
    n28 --> n49
    n18 -- "30x" --> n50
    n30 -- "18x" --> n51
    n30 -- "18x" --> n52
    n30 -- "9x" --> n53
    n34 -- "19x" --> n42
    n29 -- "6x" --> n12
    n29 -- "4x" --> n49
    n31 -- "9x" --> n32
    n31 -- "9x" --> n38
    n31 -- "27x" --> n40
    n31 -- "9x" --> n47
    n52 -- "18x" --> n54
    n52 -- "18x" --> n55
    n52 -- "18x" --> n56
    n57 -- "9x" --> n58
    n53 -- "9x" --> n59
    n53 -- "9x" --> n57
```
<!-- AUTOGEN:end -->
