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
  class conditioners {
  }
  class fitting {
  }
  class flow {
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
  class terms {
  }
  class transforms {
  }
  tramdag --> flow
  tramdag --> plots
  tramdag --> spec
  fitting --> callbacks
  flow --> fitting
  flow --> nodes
  flow --> readouts
  flow --> scores
  flow --> spec
  flow --> terms
  flow --> transforms
  nodes --> spec
  nodes --> terms
  nodes --> transforms
  plots --> spec
  readouts --> conditioners
  readouts --> spec
  readouts --> terms
  scores --> spec
  scores --> transforms
  terms --> conditioners
  terms --> spec
  fitting ..> flow
  terms ..> nodes
```

### Class UML (pyreverse)

The built-in terms are conditioners with a term-hook mixin: each concrete
term class inherits its network from ``conditioners`` and its contract
from ``ShiftTerm``/``InterceptTerm``.

```mermaid
classDiagram
  class AdditiveInterceptTerm {
  }
  class AffineUT {
  }
  class BernsteinUT {
  }
  class Callback {
  }
  class CausalFlowDAG {
  }
  class ComplexIntercept {
  }
  class ComplexInterceptTerm {
  }
  class ComplexShift {
  }
  class ComplexShift {
  }
  class ComplexShiftTerm {
  }
  class ContinuousNode {
  }
  class EarlyStopping {
  }
  class FnShift {
  }
  class FnShiftTerm {
  }
  class Intercept {
  }
  class InterceptTerm {
  }
  class LinearShift {
  }
  class LinearShift {
  }
  class LinearShiftTerm {
  }
  class OrdinalNode {
  }
  class PerNodePlateau {
  }
  class ShiftTerm {
  }
  class SimpleIntercept {
  }
  class SimpleInterceptTerm {
  }
  class SplineUT {
  }
  class StandardLogistic {
  }
  class Term {
  }
  class TermDef {
  }
  class VaryingCoef {
  }
  class VaryingCoefficient {
  }
  class VaryingCoefficientTerm {
  }
  class _FitMixin {
  }
  class _FnCallback {
  }
  class _InputTransform {
  }
  class _Node {
  }
  class _ReadoutsMixin {
  }
  class _ScaledUT {
  }
  EarlyStopping --|> Callback
  PerNodePlateau --|> Callback
  _FnCallback --|> Callback
  CausalFlowDAG --|> _FitMixin
  CausalFlowDAG --|> _ReadoutsMixin
  ComplexShift --|> Term
  FnShift --|> Term
  Intercept --|> Term
  LinearShift --|> Term
  VaryingCoefficient --|> Term
  AdditiveInterceptTerm --|> InterceptTerm
  ComplexInterceptTerm --|> ComplexIntercept
  ComplexInterceptTerm --|> InterceptTerm
  ComplexShiftTerm --|> ComplexShift
  ComplexShiftTerm --|> ShiftTerm
  FnShiftTerm --|> ShiftTerm
  InterceptTerm --|> TermDef
  LinearShiftTerm --|> LinearShift
  LinearShiftTerm --|> ShiftTerm
  ShiftTerm --|> TermDef
  SimpleInterceptTerm --|> SimpleIntercept
  SimpleInterceptTerm --|> InterceptTerm
  VaryingCoefficientTerm --|> VaryingCoef
  VaryingCoefficientTerm --|> ShiftTerm
  AffineUT --|> _ScaledUT
  BernsteinUT --|> _ScaledUT
  SplineUT --|> _ScaledUT
  ComplexShift --o ComplexShiftTerm : data
  FnShift --o FnShiftTerm : data
  Intercept --o InterceptTerm : data
  LinearShift --o LinearShiftTerm : data
  VaryingCoefficient --o VaryingCoefficientTerm : data
```

### Call graph — flow construction (traced)

```mermaid
flowchart LR
  subgraph conditioners
    n0["ComplexShift.__init__"]
    n22["LinearShift.__init__"]
    n21["SimpleIntercept.__init__"]
    n2["VaryingCoef.__init__"]
    n1["_nn"]
  end
  subgraph flow
    n3["CausalFlowDAG.__init__"]
    n4["CausalFlowDAG._apply_init"]
  end
  subgraph nodes
    n5["_Node.__init__"]
  end
  subgraph spec
    n16["Term.edge_parents"]
    n17["VaryingCoefficient.edge_parents"]
    n15["_check_node"]
    n18["_kahn_sort"]
    n19["feat_width"]
    n7["node_parents"]
    n6["validate_and_sort"]
  end
  subgraph terms
    n8["ComplexShiftTerm.build"]
    n9["InterceptTerm.build"]
    n10["LinearShiftTerm.build"]
    n11["VaryingCoefficientTerm.build"]
    n20["_attach_input_transform"]
    n12["module_for"]
  end
  subgraph transforms
    n23["BernsteinUT.__init__"]
    n13["BernsteinUT.n_params"]
    n24["_ScaledUT.__init__"]
    n14["make_univariate_transform"]
  end
    n0 --> n1
    n2 --> n1
    n3 --> n4
    n3 -- "3x" --> n5
    n3 --> n6
    n5 -- "3x" --> n7
    n5 --> n8
    n5 -- "3x" --> n9
    n5 --> n10
    n5 --> n11
    n5 -- "6x" --> n12
    n5 -- "2x" --> n13
    n5 -- "2x" --> n14
    n15 -- "5x" --> n16
    n15 --> n17
    n18 -- "3x" --> n7
    n6 -- "3x" --> n15
    n6 --> n18
    n8 --> n0
    n8 --> n19
    n8 --> n20
    n9 -- "3x" --> n21
    n10 --> n22
    n10 --> n19
    n11 --> n2
    n11 --> n19
    n11 --> n20
    n23 -- "2x" --> n24
    n14 -- "2x" --> n23
```

### Call graph — one fit (traced)

```mermaid
flowchart LR
  subgraph callbacks
    n0["EarlyStopping.__init__"]
    n1["EarlyStopping._reset"]
    n2["EarlyStopping.on_epoch_end"]
    n4["EarlyStopping.on_fit_begin"]
    n8["EarlyStopping.on_fit_end"]
    n3["_last_val"]
  end
  subgraph conditioners
    n48["ComplexShift.forward"]
    n50["LinearShift.forward"]
    n51["SimpleIntercept.forward"]
    n6["VaryingCoef.beta"]
    n5["VaryingCoef.forward"]
    n53["VaryingCoef.l2"]
    n52["VaryingCoef.recenter"]
  end
  subgraph fitting
    n7["_FitMixin.fit"]
    n9["_check_fit_sizes"]
    n10["_fit_epoch"]
    n11["_learning_rates"]
    n12["_log_epoch"]
    n13["_normalize_callbacks"]
    n14["_split_validation"]
    n15["_val_nll"]
  end
  subgraph flow
    n32["CausalFlowDAG._check_columns"]
    n33["CausalFlowDAG._check_level_values"]
    n16["CausalFlowDAG._check_side_columns"]
    n28["CausalFlowDAG._dtype"]
    n26["CausalFlowDAG._encode_parent"]
    n25["CausalFlowDAG._features"]
    n27["CausalFlowDAG._np_dtype"]
    n17["CausalFlowDAG._recenter_vc"]
    n31["CausalFlowDAG._side_feats"]
    n18["CausalFlowDAG._tensorize"]
    n34["CausalFlowDAG._theta_shift"]
    n19["CausalFlowDAG.calibrate"]
    n22["CausalFlowDAG.node_log_prob"]
  end
  subgraph nodes
    n39["_Node.net_input"]
    n35["_Node.theta_shift"]
    n38["kind_log_prob"]
  end
  subgraph terms
    n41["ComplexShiftTerm.shift_value"]
    n36["InterceptTerm.calibrate_intercept"]
    n42["LinearShiftTerm.shift_value"]
    n29["ShiftTerm.finalize"]
    n20["ShiftTerm.regularizer"]
    n23["ShiftTerm.side_columns"]
    n43["SimpleInterceptTerm.theta_value"]
    n37["TermDef.calibrate"]
    n40["TermDef.input_transform"]
    n30["VaryingCoefficientTerm.finalize"]
    n54["VaryingCoefficientTerm.regressor"]
    n21["VaryingCoefficientTerm.regularizer"]
    n44["VaryingCoefficientTerm.shift_value"]
    n24["VaryingCoefficientTerm.side_columns"]
  end
  subgraph transforms
    n55["BernsteinUT._build"]
    n45["StandardLogistic.log_prob"]
    n56["_ScaledUT._log_dt_dx"]
    n57["_ScaledUT._scale"]
    n46["_ScaledUT.forward"]
    n49["_ScaledUT.set_range"]
    n60["_log1mexp"]
    n58["ordinal_bounds"]
    n59["ordinal_cutpoints"]
    n47["ordinal_log_prob"]
  end
    n0 --> n1
    n2 -- "3x" --> n3
    n4 --> n1
    n5 -- "9x" --> n6
    n7 -- "3x" --> n2
    n7 --> n4
    n7 --> n8
    n7 --> n9
    n7 -- "3x" --> n10
    n7 -- "3x" --> n11
    n7 -- "3x" --> n12
    n7 --> n13
    n7 --> n14
    n7 -- "3x" --> n15
    n7 --> n16
    n7 --> n17
    n7 -- "2x" --> n18
    n7 --> n19
    n7 -- "2x" --> n20
    n7 --> n21
    n10 -- "6x" --> n22
    n10 -- "6x" --> n21
    n15 -- "3x" --> n22
    n16 -- "2x" --> n23
    n16 --> n24
    n25 -- "30x" --> n26
    n27 -- "2x" --> n28
    n17 --> n25
    n17 -- "2x" --> n29
    n17 --> n30
    n31 -- "18x" --> n23
    n31 -- "9x" --> n24
    n18 -- "2x" --> n32
    n18 -- "2x" --> n33
    n18 -- "2x" --> n27
    n34 -- "27x" --> n31
    n34 -- "27x" --> n35
    n19 --> n32
    n19 --> n33
    n19 -- "3x" --> n36
    n19 -- "3x" --> n37
    n22 -- "9x" --> n25
    n22 -- "27x" --> n34
    n22 -- "27x" --> n38
    n39 -- "19x" --> n40
    n35 -- "9x" --> n41
    n35 -- "9x" --> n42
    n35 -- "27x" --> n43
    n35 -- "9x" --> n44
    n38 -- "18x" --> n45
    n38 -- "18x" --> n46
    n38 -- "9x" --> n47
    n41 -- "9x" --> n48
    n41 -- "9x" --> n39
    n36 -- "3x" --> n37
    n36 -- "2x" --> n49
    n42 -- "9x" --> n50
    n43 -- "27x" --> n51
    n37 -- "6x" --> n40
    n30 --> n52
    n30 --> n39
    n21 -- "7x" --> n53
    n44 -- "9x" --> n5
    n44 -- "9x" --> n39
    n44 -- "9x" --> n54
    n46 -- "18x" --> n55
    n46 -- "18x" --> n56
    n46 -- "18x" --> n57
    n58 -- "9x" --> n59
    n47 -- "9x" --> n60
    n47 -- "9x" --> n58
```
<!-- AUTOGEN:end -->
