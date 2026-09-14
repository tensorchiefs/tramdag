# Architecture

The package has eleven modules and one rule. Term-specific behavior lives on the
term's two classes. These are its `Term` subclass (the spec) and its module in
`terms.py`. Node-kind behavior lives in four adjacent functions. Everything
else is framework code.

[ADR 001](adr/001-term-owned-architecture.md) records these decisions and the
alternatives that the project refused.

## Module map
```mermaid
graph TD
    subgraph data["pure data"]
        spec["spec.py<br/>DSL: Term + one subclass per term<br/>(Intercept LinearShift ComplexShift<br/>VaryingCoefficient FnShift = I LS CS VC Fn),<br/>nodes, normalization, Kahn sort, (de)serialization"]
    end
    subgraph torch["torch modules"]
        terms["terms.py<br/>one module per term, data = its Term class;<br/>ShiftTerm/InterceptTerm hooks; module_for;<br/>_InputTransform;<br/>LinearShiftTerm ComplexShiftTerm<br/>VaryingCoefficientTerm FnShiftTerm,<br/>SimpleInterceptTerm ComplexInterceptTerm<br/>AdditiveInterceptTerm"]
        conditioners["conditioners.py<br/>raw nn heads (frozen:<br/>anchors checkpoints + RNG)"]
        transforms["transforms.py<br/>Bernstein/Spline/Affine,<br/>ordinal_* likelihood,<br/>StandardLogistic"]
        nodes["nodes.py<br/>_Node (intercept + shifts),<br/>kind_log_prob/sample/abduct/<br/>marginal_theta"]
        flow["flow.py<br/>CausalFlowDAG: build, calibrate,<br/>log_prob, sample/abduct/pmf/density,<br/>save/load; composes the mixins"]
    end
    subgraph functions["flow behavior by concern"]
        fitting["fitting.py<br/>_FitMixin: fit (Adam loop, callbacks),<br/>fit_classical (L-BFGS)"]
        readouts["readouts.py<br/>_ReadoutsMixin: ls_coefficients,<br/>varying_coef, to_matrix, contributions,<br/>design_matrix, shift_curve"]
        scores["scores.py<br/>node_scores,<br/>effect_modifier_scan"]
    end
    callbacks["callbacks.py<br/>Callback, EarlyStopping,<br/>PerNodePlateau, per_node_adam"]
    plots["plots.py<br/>plot_dag, plot_marginals,<br/>plot_training (matplotlib optional)"]

    spec --> terms
    plots --> spec
    terms --> conditioners
    nodes --> terms
    nodes --> spec
    nodes --> transforms
    flow --> nodes
    flow --> fitting
    flow --> readouts
    flow --> scores
    fitting --> callbacks
```

`spec.py` imports nothing from `terms.py`. Therefore the data layer stays
importable and torch runs no model code. `fitting.py` and `readouts.py` are
mixins that `CausalFlowDAG` composes. They import `flow` under
`TYPE_CHECKING` only. The graph is acyclic.

## The term contract
```mermaid
classDiagram
    class Term {
        <<spec.py, plain data>>
        parents
        name: the class name
        __init__(*parents): the base assigns the parents
        options: each subclass's keyword arguments, assigned to self
        __repr__(): the only one — every entry as name=value
        edge_parents(name, spec)
        cells()
        classical
        options() / from_serialized()
    }
    Term <|-- Intercept : I
    Term <|-- LinearShift : LS
    Term <|-- ComplexShift : CS
    Term <|-- VaryingCoefficient : VC
    Term <|-- FnShift : Fn
    class TermDef {
        <<terms.py, module>>
        data: the Term subclass it builds (stamps itself as its .module)
        input_transform / calibrate(train_df)
    }
    class ShiftTerm {
        key (build) / parents (node) / order
        build(term, spec)
        shift_value(node, feats)
        post_init()
        regularizer() -> Tensor | None
        finalize(node, feats)
        score_columns(node, flow, feats, dlds)
        side_columns() / check_column() / live_side() / extra_columns()
    }
    class InterceptTerm {
        groups / ci_parents
        build(term, spec, n_params)
        calibrate_intercept(train_df, own, ut)
        theta_value(node, feats, n)
        marginal_start(theta)
    }
    TermDef <|-- ShiftTerm
    TermDef <|-- InterceptTerm
    ShiftTerm <|-- LinearShiftTerm
    ShiftTerm <|-- ComplexShiftTerm
    ShiftTerm <|-- VaryingCoefficientTerm
    ShiftTerm <|-- FnShiftTerm
    InterceptTerm <|-- SimpleInterceptTerm
    InterceptTerm <|-- ComplexInterceptTerm
    InterceptTerm <|-- AdditiveInterceptTerm
    LinearShiftTerm --|> LinearShift : nn
    ComplexShiftTerm --|> ComplexShift : nn
    VaryingCoefficientTerm --|> VaryingCoef : nn
    SimpleInterceptTerm --|> SimpleIntercept : nn
    ComplexInterceptTerm --|> ComplexIntercept : nn
```

Built-in terms subclass their conditioners. Therefore the state-dict paths
(`nodes.<n>.shifts.<key>.…`) and the seeded RNG stream are those of 0.4.

A custom term is two classes:

- a `tramdag.Term` subclass. Its options are the keyword arguments of its
  `__init__`, assigned to `self` after `super().__init__(*parents)`; its rules
  are the checks after that, plus `edge_parents` and `cells`.
- a `ShiftTerm` subclass. It declares `data =` that class. It implements
  `build` and `shift_value`. The `build` method must set `key`.

Subclassing is the registration. For a one-off term, the cheap path is `Fn`.

## Node kinds

The two node kinds are continuous and ordinal. They stay an if/else in ONE
place. These four functions hold that if/else, adjacent in nodes.py:

- `kind_log_prob`
- `kind_sample`
- `kind_abduct`
- `kind_marginal_theta`

A third node kind is the trigger for a protocol. Until a third kind arrives,
the if/else stays.

## Guards that pin all of this

- `tests/tools/statedict_smoke.py` — seeded per-DGP state dicts, bit-compared.
- The inline DGP truths (`tests/conftest.py`) and 45+ regex-pinned refusals.
- `experiments/*/ground_truth/*.json` — ten CI-checked replications with
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
  modules --> spec
  nodes --> modules
  nodes --> spec
  nodes --> transforms
  plots --> spec
  readouts --> modules
  readouts --> spec
  scores --> spec
  scores --> transforms
  fitting ..> flow
  modules ..> nodes
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
  class FnShift {
  }
  class FnShiftModule {
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
  FnShiftModule --|> ShiftModule
  InterceptModule --|> TermModule
  LinearShiftModule --|> ShiftModule
  ShiftModule --|> TermModule
  SimpleInterceptModule --|> InterceptModule
  VaryingCoefficientModule --|> ShiftModule
  ComplexShift --|> Term
  FnShift --|> Term
  Intercept --|> Term
  LinearShift --|> Term
  VaryingCoefficient --|> Term
  AffineUT --|> _ScaledUT
  BernsteinUT --|> _ScaledUT
  SplineUT --|> _ScaledUT
  ComplexShift --o ComplexShiftModule : data
  FnShift --o FnShiftModule : data
  Intercept --o InterceptModule : data
  LinearShift --o LinearShiftModule : data
  VaryingCoefficient --o VaryingCoefficientModule : data
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
    n6["ComplexShiftModule.build"]
    n9["InterceptModule.build"]
    n12["LinearShiftModule.__init__"]
    n11["LinearShiftModule.build"]
    n10["SimpleInterceptModule.__init__"]
    n13["VaryingCoefficientModule.__init__"]
    n14["VaryingCoefficientModule.build"]
    n7["_attach_input_transform"]
    n5["_nn"]
    n15["module_for"]
  end
  subgraph nodes
    n2["Node.__init__"]
  end
  subgraph spec
    n20["Term.check"]
    n21["Term.edge_parents"]
    n22["VaryingCoefficient.check"]
    n23["VaryingCoefficient.edge_parents"]
    n19["_check_node"]
    n24["_kahn_sort"]
    n8["feat_width"]
    n16["node_parents"]
    n3["validate_and_sort"]
  end
  subgraph transforms
    n25["BernsteinUT.__init__"]
    n17["BernsteinUT.n_params"]
    n26["_ScaledUT.__init__"]
    n18["make_univariate_transform"]
  end
    n0 --> n1
    n0 -- "3x" --> n2
    n0 --> n3
    n4 --> n5
    n6 --> n4
    n6 --> n7
    n6 --> n8
    n9 -- "3x" --> n10
    n11 --> n12
    n11 --> n8
    n13 --> n5
    n14 --> n13
    n14 --> n7
    n14 --> n8
    n2 --> n6
    n2 -- "3x" --> n9
    n2 --> n11
    n2 --> n14
    n2 -- "6x" --> n15
    n2 -- "3x" --> n16
    n2 -- "2x" --> n17
    n2 -- "2x" --> n18
    n19 -- "5x" --> n20
    n19 -- "5x" --> n21
    n19 --> n22
    n19 --> n23
    n24 -- "3x" --> n16
    n3 -- "3x" --> n19
    n3 --> n24
    n25 -- "2x" --> n26
    n18 -- "2x" --> n25
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
    n13["_val_nll"]
  end
  subgraph flow
    n28["CausalFlowDAG._check_columns"]
    n29["CausalFlowDAG._check_level_values"]
    n14["CausalFlowDAG._check_side_columns"]
    n30["CausalFlowDAG._dtype"]
    n23["CausalFlowDAG._features"]
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
    n5 -- "3x" --> n13
    n5 --> n14
    n5 --> n15
    n5 -- "2x" --> n16
    n5 --> n17
    n5 -- "2x" --> n18
    n5 --> n19
    n8 -- "6x" --> n20
    n8 -- "6x" --> n19
    n13 -- "3x" --> n20
    n14 -- "2x" --> n21
    n14 --> n22
    n23 -- "30x" --> n24
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
