# Architecture

The package has ten modules and one rule. Term-specific behavior lives on the
term's two classes: its `Term` subclass (the spec) and its module in
`modules.py`. Node-kind maths lives in five `Node` methods, `nodes.encode` and
the score derivative in `scores.py` ([scores.md](scores.md)). Everything else is
framework code.

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
        fitting["fitting.py<br/>NodeFitMixin: fit (Adam loop, callbacks),<br/>fit_classical (L-BFGS), nll;<br/>FitMixin: fit, fit_jointly,<br/>fit_classical for the flow"]
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

`modules.py` imports nothing from `spec.py` at run time: it reads the node's
parent schema, `{parent: "continuous" | n_levels}`, and a term's `parents` and
options. That is what lets each term class hold its module class directly
(`LinearShift.module is LinearShiftModule`). `fitting.py` and `readouts.py` are
mixins: `Node` composes `NodeFitMixin`, and `CausalFlowDAG` composes `FitMixin`
and `ReadoutsMixin`. `fitting` imports `flow` and `nodes` under `TYPE_CHECKING`
only, so the import graph is acyclic at module load.

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
        parameter_groups()
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

Each module builds its layers in a fixed order under fixed attribute names.
So the state-dict paths (`nodes.<n>.shifts.<key>.…`) and the seeded RNG stream
are stable. `tests/test_statedict_stability.py` pins them.

A custom term is two classes:

- a `ShiftModule` subclass with `__init__(term, schema)`, which builds the
  network from the parent schema and sets `key`, `parents` and
  `shift_value`;
- a `tramdag.Term` subclass with `name` and `module` as class attributes
  (`__init_subclass__` refuses one without). Its options are the keyword
  arguments of its `__init__`, assigned to `self` after
  `super().__init__(*parents)`; its rules are the checks after that, plus
  `check` and `edge_parents`.

## Node kinds

The two node kinds are continuous and ordinal. The maths that differs by
kind is an if/else in three places.

- The five `Node` methods `log_prob`, `sample`, `abduct`, `marginal_theta` and
  `encode`, plus the transform choice in `Node.__init__`.
- The column encoding `nodes.encode`.
- The score derivative in `scores.py` ([scores.md](scores.md)).

Other code reads a node's schema entry (`nodes.schema_entry`):
`"continuous"` or a level count.

## Guards that pin all of this

- `tests/tools/statedict_smoke.py`: seeded per-DGP state dicts, bit-compared.
- The inline DGP truths (`tests/conftest.py`) and the regex-pinned refusals.
- `experiments/ground_truth/*.json`: the CI-checked replications with
  wall-time tripwires. Centers move only with a documented reason.
