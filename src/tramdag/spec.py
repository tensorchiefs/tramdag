"""User-facing DAG specification.

A model is one dict ``{node_name: NodeSpec}``. Each node declares its
transformation ``h`` as an **additive formula of terms** — the first
positional argument, written as a list or as a ``+`` sum:

```python
"X3": ContinuousNode([I("X1"), CS("X2")])       # h = h_theta(x1) + g(x2)
"X3": ContinuousNode(I("X1") + CS("X2"))        # the same, formula style
```

Each term is a [`Term`][] subclass, called with the parent(s) it
depends on: [`Intercept`][], [`LinearShift`][], [`ComplexShift`][],
[`VaryingCoefficient`][] and [`FnShift`][]. The paper's symbols
``I``, ``LS``, ``CS``, ``VC``, ``Fn`` are the same objects (``SI``/``CI`` build
the two intercept arities) and the notation of the docs, so use whichever
reads better:

- ``I``, the ``Intercept`` term: the parent(s) reshape the monotone
  transform, meaning its Bernstein coefficients or ordinal cutpoints. ``I``
  dispatches on its arguments. Without parents it is the paper's simple
  intercept [`SI`][], always present and optional to write. With parents it
  is the complex intercept [`CI`][]. ``transform="spline"`` picks the class of the
  monotone transform for a continuous node, and extra keyword arguments go
  straight to that class (``SI(transform="spline", bins=16)``).
- ``LS``, the ``LinearShift`` term: ``beta * x``, one interpretable weight
  and one parent.
- ``CS``, the ``ComplexShift`` term: an additive NN ``g(x)`` on the latent
  scale.
- ``VC``, the ``VaryingCoefficient`` term: ``beta(modifiers) * x_on`` with
  ``beta(x) = beta0 + b_theta(x)``, where ``b_theta`` is a small
  **penalized** network. It is a treatment-effect head with its own
  bias-variance budget.

A formula holds **exactly one intercept term, first** — written, or added as
``SI()`` when the formula only lists shifts — so ``node.terms[0]`` is always
the intercept. The intercept slot sums in coefficient space; the shift slot
sums on the latent scale. "Joint vs additive" is argument grouping: a
multi-parent term such as
``CS("a","b")`` is one **joint** network over both parents (an interaction),
whereas ``CS("a") + CS("b")`` are two **additive** terms. For intercepts the
grouping is said explicitly: ``I("a", "b", allow_interaction=False)`` is the
additive intercept. Several ``I`` terms with parents on one node are an
error — the flag is the only way to say it, so a term list is always purely
additive on the latent scale.

What ``h`` looks like per transformation, for a continuous ``x3``:

| `terms=` | `u_3 = h(x_3 given pa)` |
|---------------------------------|--------------------------------------------|
| `None` / `[I()]` | `h_theta(x3)` |
| `[LS("X1")]` | `h_theta(x3) + beta*x1` |
| `[I("X1")]` | `h_theta(x1)(x3)` |
| `[CS("X1")]` | `h_theta(x3) + g_1(x1)` |
| `[LS("X1"), CS("X2")]` | `h_theta(x3) + beta*x1 + g_2(x2)` |
| `[CS("X1", "X2")]` | `h_theta(x3) + g_12(x1, x2)` |
| `[CS("X1"), CS("X2")]` | `h_theta(x3) + g_1(x1) + g_2(x2)` |
| `[I("X1", "X2")]` | `h_theta(x1,x2)(x3)`, joint |
| `[I("X1","X2", allow_interaction=False)]` | `h_theta(x1)+theta(x2)(x3)`, additive |
| `[CS("X1"), VC("X2", t="T")]` | `h_theta(x3) + g_1(x1) + beta(x2)*t` |

Each parent enters through exactly one *edge-owning* term (I/LS/CS parents,
and a VC term's treatment ``t``). VC **modifiers** are exempt:
``CS("x2")`` + ``VC("x2", t="T")`` is the intended pattern — ``x2`` acts
prognostically through the shift *and* modifies the treatment effect.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import importlib
import sys

from .modules import (
    ComplexShiftModule,
    FnShiftModule,
    LinearShiftModule,
    VaryingCoefficientModule,
    intercept_module,
)

# %% global variables ------------------------------------------------------------------
# why bernstein and not spline: the `transform` parameter of Intercept
DEFAULT_TRANSFORM = "bernstein"
INPUT_TRANSFORMS = ("minmax", "standardize")


# %% private functions -----------------------------------------------------------------
def _import_object(path: str):
    """Give the object that a dotted import path names, ``my.pkg.module.Name``."""
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _term_class(name: str) -> type[Term]:
    """Give the term class that a serialized ``term:`` entry names.

    A bare name is an attribute of this module: the paper's symbols and the
    class names. A dotted path ``my.pkg.module.ClassName`` is imported, which
    is how a custom term travels through a checkpoint.

    Raises
    ------
    ValueError
        If the name resolves to nothing, or to something that is not a
        [`Term`][] subclass.
    """
    try:
        if "." in name:
            cls = _import_object(name)
        else:
            cls = getattr(sys.modules[__name__], name)
    except (ImportError, AttributeError) as err:
        raise ValueError(
            f"unknown term {name!r}. A custom term serializes as its import "
            "path, module.ClassName, and that module must be importable here."
        ) from err
    if not (isinstance(cls, type) and issubclass(cls, Term)):
        # a domain error (a wrong serialized entry), not a Python type error
        raise ValueError(  # noqa: TRY004
            f"unknown term {name!r}: it is not a tramdag.Term subclass"
        )
    return cls


def _serialized(term: Term) -> dict:
    """Give one term's wire entry: its name, its parents, its other options.

    ``parents`` is the one entry the wire keeps out of ``options``: it is a
    positional argument of every term constructor, and a hand-written spec
    names it that way. A term of this module is written by its name; any
    other class by its import path, ``module.ClassName``.
    """
    options = term.options()
    parents = options.pop("parents")
    cls = type(term)
    is_builtin = getattr(sys.modules[__name__], term.name, None) is cls
    return {
        "term": term.name if is_builtin else f"{cls.__module__}.{cls.__name__}",
        "parents": list(parents),
        "options": {
            k: list(v) if isinstance(v, tuple) else v for k, v in options.items()
        },
    }


def _checked_input_transform(value):
    """Give ``input_transform`` back, or refuse a value no term can apply.

    Raises
    ------
    ValueError
        If the value is neither a known name nor a callable ``fn(x, train)``.
    """
    if value is not None and not (callable(value) or value in INPUT_TRANSFORMS):
        raise ValueError(
            "input_transform must be 'minmax', 'standardize' or a callable "
            f"fn(x, train), got {value!r}"
        )
    return value


def _check_term(value) -> Term:
    """Take one entry of a formula to a [`Term`][].

    Raises
    ------
    TypeError
        If the entry is not a term.
    """
    if isinstance(value, Term):
        return value
    raise TypeError(
        "a transformation is built from terms (I/LS/CS/VC) — got "
        f"{type(value).__name__}. A '+' sum is already a flat list, so do "
        "not nest one inside another list: write either a list or a sum."
    )


def _normalize_terms(value):
    """Flatten a node's formula into its canonical term list.

    Accepted: ``None`` (a source node), one term, a ``+`` sum, or a list of
    terms. A ``+`` sum is already flat, so
    a list of lists is a mistake rather than a shape to flatten.

    The canonical form starts with the intercept: a formula written
    without one gets ``I()`` prepended, so ``terms[0]`` is always the
    intercept term. Exactly one intercept is allowed, and it must come
    first when written; a source node (``None``) gives ``[I()]``.
    """
    if value is None:
        return [Intercept()]
    written = value if isinstance(value, (list, tuple)) else [value]
    items = [_check_term(e) for e in written]
    intercept_at = [i for i, t in enumerate(items) if isinstance(t, Intercept)]
    if len(intercept_at) > 1:
        parented = [t for t in items if isinstance(t, Intercept) and t.parents]
        if len(parented) > 1:
            raise ValueError(
                "a formula takes exactly one intercept term. For an additive "
                "intercept write CI("
                + ", ".join(repr(p) for t in parented for p in t.parents)
                + ", allow_interaction=False), not several intercept terms."
            )
        raise ValueError(
            "a formula takes exactly one intercept term, and CI(...) already "
            "contains the baseline — drop the extra I/SI"
        )
    if not intercept_at:
        return [Intercept(), *items]
    if intercept_at[0] != 0:
        raise ValueError(
            "the intercept term comes first: write "
            "I(...) + <shifts>, not the other way around"
        )
    return items


def _check_node(name: str, node: NodeSpec, spec: dict[str, NodeSpec]) -> None:
    """Validate one node against the spec: parents exist, each owns one edge.

    A term validates its own shape when it is built (arity, option
    values); what needs the spec — parents exist, the ``VC`` treatment and
    centering rules — runs here, through the term's ``check``.

    Raises
    ------
    ValueError
        If a parent is unknown, a term's spec-level rules fail, or a parent
        enters through more than one edge-owning term.
    """
    seen: set[str] = set()
    for term in node.terms:
        for p in term.parents:
            if p not in spec:
                raise ValueError(f"node {name!r}: unknown parent {p!r}")
        term.check(name, spec)
        for p in term.edge_parents:
            if p in seen:
                raise ValueError(
                    f"node {name!r}: parent {p!r} appears in more than one "
                    "term. Each parent must enter through exactly one "
                    "edge-owning term. Only VC modifiers may repeat."
                )
            seen.add(p)


def _kahn_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Topologically sort the nodes with Kahn's algorithm.

    Dependencies are ``pa(x_i)``, the union of all term parents. Ready
    nodes are emitted in sorted batches, so the order is deterministic.

    Raises
    ------
    ValueError
        If the graph has a cycle.
    """
    remaining = {name: set(node_parents(node)) for name, node in spec.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError(f"graph has a cycle among {sorted(remaining)}")
        for n in ready:
            order.append(n)
            del remaining[n]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


# %% public functions ------------------------------------------------------------------
def SI(**options) -> Intercept:
    """Build the simple-intercept baseline, the paper's SI: ``I()`` without parents.

    Parameters
    ----------
    **options
        As for [`Intercept`][]: ``transform`` and its keyword arguments.

    Returns
    -------
    Intercept
        The parentless intercept term.
    """
    return Intercept(**options)


def CI(*parents: str, **options) -> Intercept:
    """Build the complex intercept, the paper's CI: ``I(*parents)`` with parents.

    Parameters
    ----------
    *parents : str
        Parent names, at least one.
    **options
        As for [`Intercept`][].

    Returns
    -------
    Intercept
        The parent-conditioned intercept term.

    Raises
    ------
    ValueError
        If no parent is given.
    """
    if not parents:
        raise ValueError(
            "CI() needs at least one parent; the parentless baseline is SI()"
        )
    return Intercept(*parents, **options)


def node_parents(node: NodeSpec) -> list[str]:
    """Give the parent names that the terms of a node reference.

    Parameters
    ----------
    node : NodeSpec
        The node specification.

    Returns
    -------
    list[str]
        The parent names, ordered by first appearance, without
        duplicates.
    """
    seen: dict[str, None] = {}
    for term in node.terms:
        for p in term.parents:
            seen.setdefault(p, None)
    return list(seen)


def validate_and_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Validate the spec and return a topological ordering of the nodes.

    Edge ownership: every parent must enter through exactly one
    edge-owning term. Edge-owning are all parents of I/LS/CS terms and
    the ``t`` of a VC term. VC *modifiers* are exempt — they can repeat
    across terms, because a modifier typically also acts prognostically
    through a CS or LS term.

    Parameters
    ----------
    spec : dict[str, NodeSpec]
        The DAG specification.

    Returns
    -------
    list[str]
        The node names in topological order.

    Raises
    ------
    ValueError
        If a parent is unknown, a parent enters through more than one
        edge-owning term, a VC treatment is unsupported, or the graph has a
        cycle.
    """
    for name, node in spec.items():
        _check_node(name, node, spec)
    return _kahn_sort(spec)


def spec_to_dict(spec: dict[str, NodeSpec]) -> dict:
    """Give the serialized representation of a spec, for checkpoints.

    A term serializes as its term name, its parents and **every** option it
    carries, so the result describes the model in full and does not depend on
    what the defaults happen to be today. A hand-written spec may still name
    only the options it cares about: [`spec_from_dict`][] passes them to the
    term's constructor, which fills in the rest.

    The result is JSON- and YAML-safe (tuple options such as ``units`` become
    lists), so a spec round-trips through ``json``/YAML as well as through
    ``torch.save`` — except when a term carries a *callable*
    (``input_transform``, ``fn``), which serializes only through pickle
    (``torch.save``) and only as a module-level function.

    A custom term is written as its import path, ``module.ClassName``, and
    [`spec_from_dict`][] imports it from there. The class must be defined at
    module level; one defined in ``__main__`` (a script, a notebook cell)
    loads in the same process only.

    Parameters
    ----------
    spec : dict[str, NodeSpec]
        The DAG specification.

    Returns
    -------
    dict
        The serialized spec. ``spec_from_dict`` inverts it.
    """
    out = {}
    for name, node in spec.items():
        d = {
            "kind": node.kind,
            "terms": [_serialized(t) for t in node.terms],
        }
        if node.kind == "ordinal":
            d["levels"] = node.levels
        out[name] = d
    return out


def spec_from_dict(d: dict) -> dict[str, NodeSpec]:
    """Rebuild a spec from its serialized form.

    Each term is rebuilt through its class, so a wrong arity fails as the
    term's own ``ValueError`` and a misspelled option key as Python's own
    ``TypeError``, naming the keyword. An option the entry does not mention
    takes its constructor default.

    Parameters
    ----------
    d : dict
        The serialized spec, as produced by [`spec_to_dict`][].

    Returns
    -------
    dict[str, NodeSpec]
        The node specification, keyed by node name.

    Raises
    ------
    ValueError
        If the entry names an unknown term, or a term's own checks refuse it.
    TypeError
        If a term does not take an option the entry names.
    """
    spec: dict[str, NodeSpec] = {}
    for name, entry in d.items():
        terms = []
        for t in entry["terms"]:
            cls = _term_class(t["term"])
            terms.append(cls.from_serialized(tuple(t["parents"]), t["options"]))
        if entry["kind"] == "continuous":
            spec[name] = ContinuousNode(terms)
        elif entry["kind"] == "ordinal":
            spec[name] = OrdinalNode(int(entry["levels"]), terms)
        else:
            raise ValueError(f"node {name!r}: unknown kind {entry['kind']!r}")
    return spec


# %% public classes --------------------------------------------------------------------
class Term:
    """One additive term of a node's transformation; each kind is a subclass.

    Terms add: ``I("a") + CS("b")`` is the same transformation as
    ``[I("a"), CS("b")]``. A term is plain data — comparable, hashable,
    serializable by [`spec_to_dict`][] — and knows its own spec-level rules
    (``check``, ``edge_parents``, ``cells``, ``classical``). ``module`` is the
    class in [`modules`][tramdag.modules] that trains it, constructed as
    ``module(term, spec)``; the intercept slot adds ``n_params``.

    Subclass to add a term: set ``name`` (what the ``term`` key serializes)
    and ``module`` (a [`ShiftModule`][tramdag.modules.ShiftModule] subclass)
    as class attributes, and assign the options — the keyword arguments of
    ``__init__``, with their defaults — to ``self``:

    ```python
    class Scaled(Term):
        name = "Scaled"
        module = ScaledModule

        def __init__(self, *parents, scale=1.0):
            super().__init__(*parents)
            self.scale = scale
            if len(self.parents) != 1:
                raise ValueError("Scaled() takes exactly one parent.")
    ```

    A term is exactly its ``__dict__``: the parents the base assigns and the
    options each subclass assigns from its own signature. That is what makes
    [`options`][], equality and serialization one line each.

    Attributes
    ----------
    parents : tuple[str, ...]
        Ordered parent names the term depends on. Empty only for the bare simple
        intercept ``I()``. For a [`VaryingCoefficient`][] term,
        ``parents[0]`` is the treatment ``t`` and the rest are the effect modifiers;
        every other built-in term's parents all own their edges.
    """

    def __init_subclass__(cls, **kwargs):
        """Refuse a term class without ``name`` (the wire key) or ``module``."""
        super().__init_subclass__(**kwargs)
        for attr in ("name", "module"):
            if attr not in cls.__dict__:
                raise TypeError(f"{cls.__name__}: a Term subclass sets `{attr} = ...`.")

    def __init__(self, *parents: str):
        self.parents = tuple(parents)

    def options(self) -> dict:
        """Give everything the term carries, by name — the parents included."""
        return dict(vars(self))

    @classmethod
    def from_serialized(cls, parents: tuple[str, ...], options: dict) -> Term:
        """Rebuild a term from its serialized parents and options."""
        return cls(*parents, **options)

    @property
    def classical(self) -> bool:
        """Say whether the exact classical fit (``fit_classical``) handles this term."""
        return False

    def check(self, name: str, spec: dict[str, NodeSpec]) -> None:
        """Check the term against the spec it sits in; nothing to check here.

        A built-in term's own checks need only its arguments and run in
        ``__init__``. ``VC`` overrides this: its treatment and centering rules
        need the other nodes.
        """

    @property
    def edge_parents(self) -> tuple[str, ...]:
        """The parents that own an edge; every parent, for a plain term."""
        return self.parents

    def cells(self) -> list[tuple[str, str, bool]]:
        """Give the term's adjacency cells as ``(parent, tag, joint)`` triples.

        The tag is the term's name. ``joint`` says whether the term is one
        network over several parents; it comes from the term, not from its
        parent count, because a ``VC`` has a treatment plus modifiers and is
        not one network over them.
        """
        joint = len(self.parents) > 1
        return [(p, self.name, joint) for p in self.parents]

    def __eq__(self, other):
        """Compare the class and everything the term carries."""
        return type(self) is type(other) and vars(self) == vars(other)

    def __hash__(self):
        """Hash the class and the parents.

        Deliberately coarser than ``__eq__``: two terms that differ only in
        their options share a bucket, which costs nothing at a spec's size and
        frees the option values from having to be hashable themselves.
        """
        return hash((type(self), self.parents))

    def __repr__(self):
        """Name everything the term carries, the parents included.

        The one ``__repr__`` of the term classes: each entry of
        [`options`][] reads as ``name=value``, so a term with a
        keyword-only parent (a ``VC`` treatment) needs no spelling of its own.
        """
        args = [f"{k}={v!r}" for k, v in self.options().items()]
        return f"{self.name}({', '.join(args)})"

    def __add__(self, other: Term) -> list[Term]:
        """Start a plain term list."""
        if isinstance(other, Term):
            return [self, other]
        return NotImplemented

    def __radd__(self, other: list[Term]) -> list[Term]:
        """Extend a term list from the right.

        ``I("a") + CS("b") + LS("c")`` evaluates left to right: the first
        ``+`` gives a list, so the second is ``list + Term`` and lands here.
        """
        if isinstance(other, list):
            return [*other, self]
        return NotImplemented


class Intercept(Term):
    """The intercept term ``I``: the parents reshape the monotone transform.

    Without parents it is the paper's simple intercept **SI** — one free
    parameter vector, the same for every row. With parents it is the
    complex intercept **CI**: the transform parameters become a function
    of them. [`SI`][] and [`CI`][] are the two spellings with their
    arity checked.

    Parameters
    ----------
    *parents : str
        Parent names. Several parents form one **joint** network (an
        interaction) unless ``allow_interaction=False``.
    transform : str | None, optional
        A continuous node's monotone transform: ``"bernstein"``,
        ``"spline"`` or ``"affine"``. ``None``, the default, means the node
        picks ``"bernstein"``. The name stays
        ``None`` here rather than becoming the literal, because ``None`` is
        also how an ordinal node tells that no transform was asked for, and
        an ordinal intercept is the cutpoint vector, which has none to pick.
        Bernstein is the choice because zuko's spline extrapolates outside
        ``[-B, B]`` with a *fixed* slope, independent of the fitted
        parameters, so the ~10% of data beyond the 5%/95% pre-scaling range
        is misweighted whenever the true tail slope differs; Bernstein
        extrapolates linearly along its own boundary derivative.
    transform_kwargs : Mapping | None, optional
        The transform's keyword arguments as one mapping. This is the
        serialized form, which is how a spec YAML and a checkpoint carry
        them; write them out instead when calling by hand.
    allow_interaction : bool, optional
        ``False`` makes a multi-parent term **additive**: one network per
        parent, their parameter vectors summed in coefficient space. A node
        takes at most one intercept term with parents — write an additive
        intercept with this flag, not with several intercept terms. Default
        ``True``: one joint network is what the reference implementations
        do.

    Other Parameters
    ----------------
    units : list[int] | tuple[int, ...], optional
        Hidden layers of the term's network, for example ``units=[16]``. By
        default ``(8, 8)``, the two hidden layers of the PyTorch reference's
        ``ComplexInterceptDefaultTabular`` — see
        [`modules`][tramdag.modules] for the provenance and for why a paper
        replication sets this explicitly.
    activation : str, optional
        Activation of the network's hidden layers, by default ``"relu"`` —
        one of the keys of ``modules.ACTIVATIONS``.
    batch_norm : bool, optional
        Batch-normalize the network's hidden layers, by default False.
    input_transform : str | callable | None, optional
        ``"minmax"``, ``"standardize"`` or a callable ``fn(x, train)``
        applied per continuous parent column (``train`` is that column's
        raw training data, frozen at ``calibrate``). Parents only.
        ``None``, the default, applies no transform.
    **kwargs
        Any keyword that is not an option above goes straight to the
        transform class, for example ``I(transform="spline", bins=16)`` or
        ``I(n_coeffs=40)``. A keyword written out here wins over the same
        key inside ``transform_kwargs``.

    Raises
    ------
    ValueError
        If a parentless term carries ``input_transform``, or
        ``allow_interaction=False`` comes with fewer than two parents (an
        interaction to disallow needs two).
    """

    name = "I"
    module = staticmethod(intercept_module)  # a function, not a class: no binding

    def __init__(
        self,
        *parents: str,
        transform: str | None = None,
        transform_kwargs: dict | None = None,
        allow_interaction: bool = True,
        units: tuple[int, ...] | list[int] = (8, 8),
        activation: str = "relu",
        batch_norm: bool = False,
        input_transform: object = None,
        **transform_options,
    ):
        # Python's argument binding IS the pass-through boundary: what binds
        # above is an option of the term, whatever is left over is a keyword of
        # the transform class. A written-out keyword wins over the same key
        # inside a serialized transform_kwargs mapping, hence the merge order.
        super().__init__(*parents)
        self.transform = transform
        self.transform_kwargs = {**(transform_kwargs or {}), **transform_options}
        self.allow_interaction = allow_interaction
        self.units = tuple(units)
        self.activation = activation
        self.batch_norm = batch_norm
        self.input_transform = _checked_input_transform(input_transform)
        if not self.parents and self.input_transform is not None:
            raise ValueError(
                "a simple intercept has no network inputs — input_transform= "
                "belongs on CI/CS/VC terms"
            )
        if not self.allow_interaction and len(self.parents) < 2:
            raise ValueError(
                "allow_interaction=False makes a MULTI-parent intercept additive; "
                "with one parent there is no interaction to disallow — drop the "
                "argument"
            )

    @property
    def classical(self) -> bool:
        """Say yes only for a parentless ``I()`` — the simple baseline."""
        return not self.parents

    def cells(self) -> list[tuple[str, str, bool]]:
        """Tag an intercept edge ``CI``: a cell exists only when it has parents."""
        return [(p, "CI", len(self.parents) > 1) for p in self.parents]


class LinearShift(Term):
    """The linear shift ``LS``: ``beta * x``, one interpretable raw-unit coefficient.

    Parameters
    ----------
    *parents : str
        Exactly one parent name. The weight stays the interpretable
        raw-unit coefficient, so an LS takes no ``input_transform``.

    Raises
    ------
    ValueError
        If the parent count is not one.
    """

    name = "LS"
    module = LinearShiftModule

    def __init__(self, *parents: str):
        super().__init__(*parents)
        if len(self.parents) != 1:
            raise ValueError("LS() takes exactly one parent.")

    @property
    def classical(self) -> bool:
        """Say yes — an LS is a classical transformation-model coefficient."""
        return True


class ComplexShift(Term):
    """The complex shift ``CS``: an additive network ``g(x)`` on the latent scale.

    Parameters
    ----------
    *parents : str
        At least one parent name. Several parents feed one joint network;
        ``CS("a") + CS("b")`` are two additive terms instead.

    Other Parameters
    ----------------
    units : list[int] | tuple[int, ...], optional
        Hidden layers, for example ``units=[16]``. By default
        ``(64, 128, 64)``, the three hidden layers of the PyTorch reference's
        ``ComplexShiftDefaultTabular`` — see
        [`modules`][tramdag.modules] for the provenance.
    activation : str, optional
        Activation of the hidden layers, by default ``"relu"``.
    batch_norm : bool, optional
        Batch-normalize the hidden layers, by default False.
    input_transform : str | callable | None, optional
        As for [`Intercept`][]. ``None``, the default, applies no
        transform.

    Raises
    ------
    ValueError
        If no parent is given.
    """

    name = "CS"
    module = ComplexShiftModule

    def __init__(
        self,
        *parents: str,
        units: tuple[int, ...] | list[int] = (64, 128, 64),
        activation: str = "relu",
        batch_norm: bool = False,
        input_transform: object = None,
    ):
        super().__init__(*parents)
        self.units = tuple(units)
        self.activation = activation
        self.batch_norm = batch_norm
        self.input_transform = _checked_input_transform(input_transform)
        if not self.parents:
            raise ValueError("CS() needs at least one parent.")


class VaryingCoefficient(Term):
    """The varying-coefficient shift ``VC``: ``beta(modifiers) * x_t``.

    The treatment-effect term: ``VC("X2", "X3", t="T")`` is
    ``(beta0 + b_theta(x2, x3)) * x_t``, with ``b_theta`` a small network
    whose weights carry the L2 ``penalty``. The fitting objective is the
    penalized NLL ``sum_i nll_i + penalty * ||b_theta weights||^2`` on the
    total-likelihood scale — a fixed Gaussian prior whose shrinkage
    vanishes as n grows. ``beta0`` is not penalized.

    The output of ``b_theta`` is zero-initialized and, after the fit,
    mean-centered over the training data, so ``beta0`` is the
    interpretable main effect on the log-odds scale — the classical
    ``Colr``/``LS`` reading when ``beta`` is constant. ``penalty -> inf``,
    or exactly zero modifiers, reduces the term to ``LS(t)``, so VC-vs-LS
    is a nested question. Read the fitted effect out with
    [`varying_coef`][tramdag.flow.CausalFlowDAG.varying_coef].

    Unlike other terms, VC *modifiers* can also appear in the node's
    prognostic terms (``CS``/``LS``/``I``). Only ``t`` owns its edge.

    Parameters
    ----------
    *modifiers : str
        The effect modifiers — the covariates that enter ``b_theta``.
        Empty means a constant effect.
    t : str
        The treatment (required keyword). Must be a continuous node or a
        binary (2-level) ordinal node. The term is linear in ``x_t``.

    Other Parameters
    ----------------
    penalty : float, optional
        L2 weight on the ``b_theta`` weights, by default 1.0. Must be
        >= 0. 1.0 is the value at which ``tests/test_vc_term.py`` recovers
        the known ``beta(x)`` of the ``vc_hetero`` DGP at corr ~ 0.99. The
        penalty is on the total-NLL scale, so its effective strength moves
        with ``n``: raise it for small ``n`` or many modifiers.
    center : str | None, optional
        Propensity centering, by default ``None``: no centering,
        bit-identical to the uncentered term. A string names the
        **training-frame column** holding the out-of-fold propensities
        ``P(t = 1 | pa_t)`` per row — compute them with any cross-fitted
        classifier OUTSIDE the flow and merge them as a column (in-sample
        values reintroduce the own-observation bias). The regressor becomes
        ``beta(x) * (x_t - e_hat(pa_t))`` — the Robinson/R-learner
        orthogonalization inside the likelihood; every query after the fit
        recomputes the propensity live from the flow's own treatment node.
        Requires a binary ordinal ``t``. ``docs/varying-coefficients.md``
        measures a 5-10x bias reduction from turning it on.
    units : list[int] | tuple[int, ...], optional
        Hidden layers of ``b_theta``, by default ``(16,)`` — the size is
        justified at
        [`VaryingCoefficientModule`][tramdag.modules.VaryingCoefficientModule].
    activation : str, optional
        Activation of ``b_theta``'s hidden layers, by default ``"relu"``.
    batch_norm : bool, optional
        Batch-normalize ``b_theta``'s hidden layers, by default False.
    input_transform : str | callable | None, optional
        As for [`Intercept`][], over the modifiers. ``None``, the default,
        applies no transform.

    Raises
    ------
    ValueError
        If ``t`` is also a modifier or if ``penalty`` is negative.

    Notes
    -----
    With ``center="col"``, training reads **out-of-fold** ``e_hat`` for
    every row from that column of the training frame — the DML
    cross-fitting requirement; in-sample centering can be *worse* than
    none. The values are frozen as data, so no gradient reaches the ``t``
    node from this node's loss. Inference (``log_prob``/``sample``/
    ``abduct``/``pmf``) recomputes ``e_hat`` from the flow's own fitted
    ``t`` node — the full-data fit, the standard DML train/predict split —
    and always re-derives ``x_t - e_hat`` under ``do``, never from a
    cache. With centering, ``beta0`` is the effect at the treatment margin
    (the observed propensities). The LS-nesting reading applies to the
    uncentered term only.
    """

    name = "VC"
    module = VaryingCoefficientModule

    def __init__(
        self,
        *modifiers: str,
        t: str,
        penalty: float = 1.0,
        center: str | None = None,
        units: tuple[int, ...] | list[int] = (16,),
        activation: str = "relu",
        batch_norm: bool = False,
        input_transform: object = None,
    ):
        if penalty < 0:
            raise ValueError(f"VC(): penalty must be >= 0, got {penalty}")
        # the treatment leads the parents: it is the one that owns an edge
        super().__init__(t, *modifiers)
        self.penalty = float(penalty)
        self.center = center
        self.units = tuple(units)
        self.activation = activation
        self.batch_norm = batch_norm
        self.input_transform = _checked_input_transform(input_transform)
        if t in modifiers:
            raise ValueError(
                f"VC(): {t!r} cannot be both the treatment (t) and a modifier"
            )

    @classmethod
    def from_serialized(cls, parents: tuple[str, ...], options: dict) -> Term:
        """Rebuild from the serialized parents: the treatment comes first."""
        return cls(*parents[1:], t=parents[0], **options)

    def check(self, name: str, spec: dict[str, NodeSpec]) -> None:
        """Check the treatment and the centering against the spec.

        Raises
        ------
        ValueError
            If the centering column is malformed or collides with a node,
            the treatment is a multi-level ordinal, or centering meets a
            continuous or an itself-centered treatment.
        """
        t = self.parents[0]
        t_node = spec[t]
        if self.center is not None and not isinstance(self.center, str):
            raise ValueError(
                f"node {name!r}: VC(center=) names the propensity COLUMN of "
                "the training frame (out-of-fold P(t=1|pa_t) per row), or is "
                f"None — got {self.center!r}. Cross-fit the propensities "
                "outside and merge them as a column."
            )
        if self.center and self.center in spec:
            raise ValueError(
                f"node {name!r}: the propensity column {self.center!r} "
                "collides with a node name"
            )
        if t_node.kind == "ordinal" and t_node.levels != 2:
            raise ValueError(
                f"node {name!r}: VC treatment {t!r} is ordinal with "
                f"{t_node.levels} levels. Only a 2-level (binary) ordinal "
                "treatment is supported. Multi-level is a follow-up."
            )
        if self.center and t_node.kind != "ordinal":
            raise ValueError(
                f"node {name!r}: VC(center=...) needs a binary ordinal "
                f"treatment, and {t!r} is continuous. E[T|x] centering is a "
                "follow-up."
            )
        if self.center and any(
            isinstance(term, VaryingCoefficient) and term.center
            for term in t_node.terms
        ):
            raise ValueError(
                f"node {name!r}: treatment {t!r} carries a centered VC term "
                "itself; chained centering is not supported"
            )

    @property
    def edge_parents(self) -> tuple[str, ...]:
        """Only the treatment owns an edge; the modifiers may repeat elsewhere."""
        return self.parents[:1]

    def cells(self) -> list[tuple[str, str, bool]]:
        """Tag the treatment cell ``VC`` and the modifiers ``VCm``; never joint."""
        t, mods = self.parents[0], self.parents[1:]
        return [(t, "VC", False)] + [(p, "VCm", False) for p in mods]


class FnShift(Term):
    """The function shift ``Fn``: ``fn(features)`` joins the additive shifts.

    The cheapest custom term: ``fn`` takes the term's concatenated parent
    features ``(n, k)`` (continuous raw, ordinal one-hot — through
    ``input_transform`` when given) and returns the shift contribution,
    shape ``(n,)`` or ``(n, 1)``. A plain function is a fixed offset; an
    ``nn.Module`` registers as a submodule and trains with the flow.

    Checkpoints pickle ``fn``, so it must be a module-level function or an
    importable ``nn.Module`` — ``save()`` refuses a lambda. For a whole new
    term (own options, penalty, side inputs) subclass [`Term`][] and
    [`ShiftModule`][tramdag.modules.ShiftModule] instead.

    Parameters
    ----------
    *parents : str
        Parent node names feeding ``fn``.

    Other Parameters
    ----------------
    fn : callable | torch.nn.Module
        The shift function. Required.
    input_transform : str | callable | None, optional
        As for [`ComplexShift`][]. ``None``, the default, applies no
        transform.

    Raises
    ------
    ValueError
        If no parent is given or ``fn`` is not callable.
    """

    name = "Fn"
    module = FnShiftModule

    def __init__(self, *parents: str, fn, input_transform: object = None):
        super().__init__(*parents)
        self.fn = fn
        self.input_transform = _checked_input_transform(input_transform)
        if not self.parents:
            raise ValueError("Fn() needs at least one parent.")
        if not callable(self.fn):
            # a domain error (a wrong option value), not a Python type error
            raise ValueError(  # noqa: TRY004
                f"Fn(fn=) must be callable, got {type(self.fn).__name__}"
            )


class ContinuousNode:
    """Continuous variable, modelled by a monotone 1-D transform + shifts.

    Parameters
    ----------
    terms : Term | list[Term] | None, optional
        The additive formula for ``h``: a list of terms, a ``+`` sum or a
        single term. ``None`` (default) is a source node. The
        class of the monotone transform is chosen on the intercept term,
        ``I(..., transform="spline")``; the default is ``"bernstein"``.
    """

    kind = "continuous"

    def __init__(self, terms=None):
        self.terms = _normalize_terms(terms)
        # the arguments go straight to the transform class; if they are
        # wrong, that class says so — this layer does not second-guess it
        intercept = self.terms[0]
        self.transform = intercept.transform or DEFAULT_TRANSFORM
        self.transform_kwargs = dict(intercept.transform_kwargs)

    def __repr__(self):
        """Show the terms and the transform."""
        return f"ContinuousNode({self.terms!r}, transform={self.transform!r})"

    def __eq__(self, other):
        """Compare the terms; the transform is derived from them."""
        # transform/transform_kwargs are derived from the terms, so equal
        # term lists already imply an equal transform
        return isinstance(other, ContinuousNode) and self.terms == other.terms

    def __hash__(self):
        """Hash what ``__eq__`` compares, so nodes work in sets and as keys."""
        return hash((self.kind, tuple(self.terms)))


class OrdinalNode:
    """Ordinal variable with ``levels`` ordered classes, stored 0 to levels-1.

    An ordered logit models it: increasing cutpoints plus the shift terms.

    Parameters
    ----------
    levels : int
        Number of ordered classes.
    terms : Term | list[Term] | None, optional
        The additive formula, as for [`ContinuousNode`][], by default
        ``None``.
    """

    kind = "ordinal"

    def __init__(self, levels: int, terms=None):
        self.levels = int(levels)
        if self.levels < 2:
            raise ValueError(f"OrdinalNode(levels=) must be >= 2, got {self.levels}")
        self.terms = _normalize_terms(terms)
        intercept = self.terms[0]
        if intercept.transform or intercept.transform_kwargs:
            raise ValueError(
                "I(transform=...) is for continuous nodes. An ordinal node's "
                "intercept is the cutpoint vector, it has no transform to choose."
            )

    def __repr__(self):
        """Show the levels and the terms."""
        return f"OrdinalNode({self.levels}, {self.terms!r})"

    def __eq__(self, other):
        """Compare levels and terms."""
        return (
            isinstance(other, OrdinalNode)
            and self.levels == other.levels
            and self.terms == other.terms
        )

    def __hash__(self):
        """Hash what ``__eq__`` compares, so nodes work in sets and as keys."""
        return hash((self.kind, self.levels, tuple(self.terms)))


# kept after the classes: the union is evaluated at definition time
NodeSpec = ContinuousNode | OrdinalNode


# %% alias -----------------------------------------------------------------------------
# The paper's symbols are the notation of the docs and the spelling nearly
# every caller uses; they are the classes above, unchanged.
I = Intercept  # noqa: E741 - ambiguous only out of context
LS = LinearShift
CS = ComplexShift
VC = VaryingCoefficient
Fn = FnShift
