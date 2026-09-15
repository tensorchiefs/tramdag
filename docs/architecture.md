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
        modules["modules.py<br/>one nn.Module per term, built from (term, spec):<br/>ShiftModule/InterceptModule hooks,<br/>intercept_module, feat_width, _InputTransform;<br/>LinearShiftModule ComplexShiftModule<br/>VaryingCoefficientModule,<br/>SimpleInterceptModule ComplexInterceptModule<br/>AdditiveInterceptModule"]
        transforms["transforms.py<br/>Bernstein/Spline/Affine,<br/>ordinal_* likelihood,<br/>StandardLogistic"]
        nodes["nodes.py<br/>Node: intercept + shifts,<br/>encode, log_prob, sample,<br/>abduct, marginal_theta"]
        flow["flow.py<br/>CausalFlowDAG: construct, calibrate,<br/>log_prob, sample/abduct/pmf/density,<br/>save/load; composes the mixins"]
    end
    subgraph functions["flow behavior by concern"]
        fitting["fitting.py<br/>FitMixin: fit (Adam loop, callbacks),<br/>fit_classical (L-BFGS)"]
        readouts["readouts.py<br/>ReadoutsMixin: ls_coefficients,<br/>varying_coef, to_matrix, contributions,<br/>design_matrix, shift_curve"]
        scores["scores.py<br/>node_scores,<br/>effect_modifier_scan"]
    end
    callbacks["callbacks.py<br/>Callback, EarlyStopping,<br/>PerNodePlateau, per_node_adam"]
    plots["plots.py<br/>plot_dag, plot_marginals,<br/>plot_training (matplotlib optional)"]

    spec --> modules
    nodes --> spec
    nodes --> transforms
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

`modules.py` imports nothing from `spec.py`: it reads a spec node's `kind` and
`levels` and a term's `parents` and options. That is what lets each term class
hold its module class directly (`LinearShift.module is LinearShiftModule`).
`fitting.py` and `readouts.py` are mixins that `CausalFlowDAG` composes;
`fitting` imports `flow` under `TYPE_CHECKING` only. The graph is acyclic.

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
        __init__(term, spec): builds the net, sets key and parents
        order
        shift_value(node, feats)
        post_init()
        regularizer() -> Tensor | None
        finalize(node, feats)
        score_columns(node, flow, feats, dlds)
        side_columns() / check_column() / live_side() / extra_columns()
    }
    class InterceptModule {
        __init__(term, spec, n_params): intercept_module picks the class
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

- a `ShiftModule` subclass with `__init__(term, spec)`, which builds the
  network and sets `key` and `parents`, and `shift_value`;
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
  tramdag --> plots
  tramdag --> spec
  fitting --> callbacks
  flow --> fitting
  flow --> modules
  flow --> nodes
  flow --> readouts
  flow --> spec
  flow --> transforms
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
  class OrdinalNode {
  }
  class PerNodePlateau {
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
  PerNodePlateau --|> Callback
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
  end
  subgraph modules
    n4["ComplexShiftModule.__init__"]
    n8["LinearShiftModule.__init__"]
    n11["SimpleInterceptModule.__init__"]
    n9["VaryingCoefficientModule.__init__"]
    n5["_attach_input_transform"]
    n6["_nn"]
    n7["feat_width"]
    n10["intercept_module"]
  end
  subgraph nodes
    n2["Node.__init__"]
  end
  subgraph spec
    n16["Term.check"]
    n17["Term.edge_parents"]
    n18["VaryingCoefficient.check"]
    n19["VaryingCoefficient.edge_parents"]
    n15["_check_node"]
    n20["_kahn_sort"]
    n12["node_parents"]
    n3["validate_and_sort"]
  end
  subgraph transforms
    n21["BernsteinUT.__init__"]
    n13["BernsteinUT.n_params"]
    n22["_ScaledUT.__init__"]
    n14["make_univariate_transform"]
  end
    n0 --> n1
    n0 -- "3x" --> n2
    n0 --> n3
    n4 --> n5
    n4 --> n6
    n4 --> n7
    n8 --> n7
    n9 --> n5
    n9 --> n6
    n9 --> n7
    n10 -- "3x" --> n11
    n2 --> n4
    n2 --> n8
    n2 --> n9
    n2 -- "3x" --> n10
    n2 -- "3x" --> n12
    n2 -- "2x" --> n13
    n2 -- "2x" --> n14
    n15 -- "5x" --> n16
    n15 -- "5x" --> n17
    n15 --> n18
    n15 --> n19
    n20 -- "3x" --> n12
    n3 -- "3x" --> n15
    n3 --> n20
    n21 -- "2x" --> n22
    n14 -- "2x" --> n21
```

### Call graph — one fit (traced)

```mermaid
flowchart LR
  subgraph callbacks
    n0["EarlyStopping.__init__"]
    n1["EarlyStopping._reset"]
    n2["EarlyStopping.on_epoch_end"]
    n4["EarlyStopping.on_fit_begin"]
    n6["EarlyStopping.on_fit_end"]
    n3["_last_val"]
  end
  subgraph fitting
    n5["FitMixin.fit"]
    n7["_check_fit_sizes"]
    n8["_fit_epoch"]
    n9["_learning_rates"]
    n10["_log_epoch"]
    n11["_normalize_callbacks"]
    n12["_split_validation"]
  end
  subgraph flow
    n28["CausalFlowDAG._check_columns"]
    n29["CausalFlowDAG._check_level_values"]
    n13["CausalFlowDAG._check_side_columns"]
    n30["CausalFlowDAG._dtype"]
    n23["CausalFlowDAG._features"]
    n14["CausalFlowDAG._mean_nll"]
    n15["CausalFlowDAG._recenter_vc"]
    n27["CausalFlowDAG._side_feats"]
    n16["CausalFlowDAG._tensorize"]
    n31["CausalFlowDAG._theta_shift"]
    n17["CausalFlowDAG.calibrate"]
    n20["CausalFlowDAG.node_log_prob"]
  end
  subgraph modules
    n37["ComplexShiftModule.forward"]
    n36["ComplexShiftModule.shift_value"]
    n33["InterceptModule.calibrate_intercept"]
    n41["LinearShiftModule.forward"]
    n40["LinearShiftModule.shift_value"]
    n25["ShiftModule.finalize"]
    n18["ShiftModule.regularizer"]
    n21["ShiftModule.side_columns"]
    n43["SimpleInterceptModule.forward"]
    n42["SimpleInterceptModule.theta_value"]
    n34["TermModule.calibrate"]
    n44["TermModule.input_transform"]
    n47["VaryingCoefficientModule.beta"]
    n26["VaryingCoefficientModule.finalize"]
    n46["VaryingCoefficientModule.forward"]
    n48["VaryingCoefficientModule.l2"]
    n45["VaryingCoefficientModule.recenter"]
    n50["VaryingCoefficientModule.regressor"]
    n19["VaryingCoefficientModule.regularizer"]
    n49["VaryingCoefficientModule.shift_value"]
    n22["VaryingCoefficientModule.side_columns"]
  end
  subgraph nodes
    n24["Node.encode"]
    n35["Node.log_prob"]
    n38["Node.net_input"]
    n32["Node.theta_shift"]
  end
  subgraph transforms
    n54["BernsteinUT._build"]
    n51["StandardLogistic.log_prob"]
    n55["_ScaledUT._log_dt_dx"]
    n56["_ScaledUT._scale"]
    n52["_ScaledUT.forward"]
    n39["_ScaledUT.set_range"]
    n59["_log1mexp"]
    n57["ordinal_bounds"]
    n58["ordinal_cutpoints"]
    n53["ordinal_log_prob"]
  end
    n0 --> n1
    n2 -- "3x" --> n3
    n4 --> n1
    n5 -- "3x" --> n2
    n5 --> n4
    n5 --> n6
    n5 --> n7
    n5 -- "3x" --> n8
    n5 -- "3x" --> n9
    n5 -- "3x" --> n10
    n5 --> n11
    n5 --> n12
    n5 --> n13
    n5 -- "3x" --> n14
    n5 --> n15
    n5 -- "2x" --> n16
    n5 --> n17
    n5 -- "2x" --> n18
    n5 --> n19
    n8 -- "6x" --> n20
    n8 -- "6x" --> n19
    n13 -- "2x" --> n21
    n13 --> n22
    n23 -- "30x" --> n24
    n14 -- "3x" --> n20
    n15 --> n23
    n15 -- "2x" --> n25
    n15 --> n26
    n27 -- "18x" --> n21
    n27 -- "9x" --> n22
    n16 -- "2x" --> n28
    n16 -- "2x" --> n29
    n16 -- "6x" --> n30
    n31 -- "27x" --> n27
    n31 -- "27x" --> n32
    n17 --> n28
    n17 --> n29
    n17 -- "3x" --> n33
    n17 -- "3x" --> n34
    n20 -- "9x" --> n23
    n20 -- "27x" --> n31
    n20 -- "27x" --> n35
    n36 -- "9x" --> n37
    n36 -- "9x" --> n38
    n33 -- "3x" --> n34
    n33 -- "2x" --> n39
    n40 -- "9x" --> n41
    n42 -- "27x" --> n43
    n34 -- "6x" --> n44
    n26 --> n45
    n26 --> n38
    n46 -- "9x" --> n47
    n19 -- "7x" --> n48
    n49 -- "9x" --> n46
    n49 -- "9x" --> n50
    n49 -- "9x" --> n38
    n35 -- "18x" --> n51
    n35 -- "18x" --> n52
    n35 -- "9x" --> n53
    n38 -- "19x" --> n44
    n32 -- "9x" --> n36
    n32 -- "9x" --> n40
    n32 -- "27x" --> n42
    n32 -- "9x" --> n49
    n52 -- "18x" --> n54
    n52 -- "18x" --> n55
    n52 -- "18x" --> n56
    n57 -- "9x" --> n58
    n53 -- "9x" --> n59
    n53 -- "9x" --> n57
```
<!-- AUTOGEN:end -->
