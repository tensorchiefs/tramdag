"""User-facing DAG specification.

A model is one dict ``{node_name: NodeSpec}``. Each node declares its
transformation as an **additive formula of terms** (the native form), e.g.::

    "X3": ContinuousNode(terms=[I("X1"), CS("X2")])     # h = baseline + I(x1) + CS(x2)

Term constructors name the parent(s) a term depends on:

- :func:`I`  — *intercept* term: the parent(s) reshape the monotone transform
  (its Bernstein coefficients / ordinal cutpoints). ``I()`` with no parent is the
  implicit simple-intercept baseline (always present, optional to write).
- :func:`LS` — *linear shift*: ``beta * x`` (one interpretable weight), one parent.
- :func:`CS` — *complex shift*: an additive MLP ``g(x)`` on the latent scale.

The intercept slot sums in coefficient space; the shift slot sums on the latent
scale. "Joint vs additive" is just argument grouping — a multi-parent term such
as ``CS("a","b")`` is one **joint** network over both parents (an interaction),
whereas ``CS("a") + CS("b")`` are two **additive** terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# legacy dict term labels -> term effect (still accepted by ``term()`` and by the
# checkpoint loader, so old saved models keep loading)
_LEGACY = {"ls": "LS", "cs": "CS", "ci": "I"}

EFFECTS = ("I", "LS", "CS")


@dataclass(frozen=True)
class Term:
    """One additive term of a node's transformation.

    ``effect`` ∈ {"I", "LS", "CS"}; ``slot`` is "intercept" for ``I`` and "shift"
    for ``LS``/``CS``. ``parents`` is the (ordered) tuple of parent names the term
    depends on — empty only for the bare simple-intercept ``I()``.
    """
    effect: str
    slot: str
    parents: tuple[str, ...]


def I(*parents: str) -> Term:  # noqa: E743 - single-letter name is the intended notation
    """Intercept term — the parent(s) reshape the transform. ``I()`` = SI base."""
    return Term("I", "intercept", tuple(parents))


def LS(*parents: str) -> Term:
    """Linear shift ``beta * x`` — exactly one parent."""
    if len(parents) != 1:
        raise ValueError("LS() takes exactly one parent.")
    return Term("LS", "shift", tuple(parents))


def CS(*parents: str) -> Term:
    """Complex (MLP) shift — at least one parent."""
    if not parents:
        raise ValueError("CS() needs at least one parent.")
    return Term("CS", "shift", tuple(parents))


# explicit aliases (avoid confusion with the conditioner classes ComplexShift /
# ComplexIntercept, and give a non-single-letter option for I)
Intercept = I
LinShift = LS
CShift = CS


def term(effect: str, *parents: str) -> Term:
    """Build a :class:`Term` from an effect *label* — useful when the effect type
    is data-driven (e.g. sweeping ``"ls"`` vs ``"cs"``). Accepts both the legacy
    labels ``"ls"``/``"cs"``/``"ci"`` and the new ``"LS"``/``"CS"``/``"I"``."""
    e = _LEGACY.get(effect.lower(), effect.upper())
    if e == "I":
        return I(*parents)
    if e == "LS":
        return LS(*parents)
    if e == "CS":
        return CS(*parents)
    raise ValueError(f"unknown term effect '{effect}'.")


@dataclass
class ContinuousNode:
    """Continuous variable, modelled by a monotone 1-D transform + shifts.

    Args:
        terms: additive formula, a list of :func:`I`/:func:`LS`/:func:`CS` terms
            (``None`` / omitted = a source node).
        transform: "bernstein" (TRAM-faithful), "spline" or "affine".
        transform_kwargs: forwarded to the transform.
    """
    terms: list[Term] | None = None
    transform: str = "bernstein"
    transform_kwargs: dict = field(default_factory=dict)
    kind: str = field(default="continuous", init=False)


@dataclass
class OrdinalNode:
    """Ordinal variable with ``levels`` ordered classes (0 .. levels-1),
    modelled by increasing cutpoints (ordered logit) + shifts."""
    levels: int
    terms: list[Term] | None = None
    kind: str = field(default="ordinal", init=False)


NodeSpec = ContinuousNode | OrdinalNode


def node_terms(node: NodeSpec) -> list[Term]:
    """Canonical term list for a node (empty for a source node)."""
    return list(node.terms) if node.terms is not None else []


def node_parents(node: NodeSpec) -> list[str]:
    """Ordered, de-duplicated parent names referenced by a node's terms."""
    seen: dict[str, None] = {}
    for term in node_terms(node):
        for p in term.parents:
            seen.setdefault(p, None)
    return list(seen)


def validate_and_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Validate the spec and return a topological ordering of the nodes."""
    for name, node in spec.items():
        seen: set[str] = set()
        for term in node_terms(node):
            if term.effect not in EFFECTS:
                raise ValueError(f"Node '{name}': unknown term effect '{term.effect}'.")
            if term.effect == "LS" and len(term.parents) != 1:
                raise ValueError(f"Node '{name}': LS term must have exactly one parent.")
            for p in term.parents:
                if p not in spec:
                    raise ValueError(f"Node '{name}': unknown parent '{p}'.")
                if p in seen:
                    raise ValueError(
                        f"Node '{name}': parent '{p}' appears in more than one term "
                        "(each parent must enter through exactly one term).")
                seen.add(p)
        if isinstance(node, OrdinalNode) and node.levels < 2:
            raise ValueError(f"Node '{name}': ordinal levels must be >= 2.")

    # Kahn's algorithm over pa(x_i) = union of all term parents
    remaining = {name: set(node_parents(node)) for name, node in spec.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError(f"Graph has a cycle among: {sorted(remaining)}")
        for n in ready:
            order.append(n)
            del remaining[n]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def spec_to_dict(spec: dict[str, NodeSpec]) -> dict:
    """JSON-serializable representation (for checkpoints)."""
    out = {}
    for name, node in spec.items():
        terms = [{"effect": t.effect, "parents": list(t.parents)}
                 for t in node_terms(node)]
        d = {"kind": node.kind, "terms": terms}
        if isinstance(node, ContinuousNode):
            d["transform"] = node.transform
            d["transform_kwargs"] = dict(node.transform_kwargs)
        else:
            d["levels"] = node.levels
        out[name] = d
    return out


def _terms_from_dict(nd: dict) -> list[Term]:
    """Rebuild a term list from serialized form, accepting both the new ``terms``
    layout and the legacy ``parents`` dict (so old checkpoints still load)."""
    if "terms" in nd:
        ctor = {"I": I, "LS": LS, "CS": CS}
        return [ctor[t["effect"]](*t["parents"]) for t in nd["terms"]]
    # legacy checkpoint: {"parents": {parent: "ls"|"cs"|"ci"}}
    out: list[Term] = []
    for parent, label in nd.get("parents", {}).items():
        effect = _LEGACY[label]
        out.append(I(parent) if effect == "I" else
                   (LS(parent) if effect == "LS" else CS(parent)))
    return out


def spec_from_dict(d: dict) -> dict[str, NodeSpec]:
    spec: dict[str, NodeSpec] = {}
    for name, nd in d.items():
        terms = _terms_from_dict(nd)
        if nd["kind"] == "continuous":
            spec[name] = ContinuousNode(terms=terms,
                                        transform=nd["transform"],
                                        transform_kwargs=dict(nd["transform_kwargs"]))
        else:
            spec[name] = OrdinalNode(levels=int(nd["levels"]), terms=terms)
    return spec
