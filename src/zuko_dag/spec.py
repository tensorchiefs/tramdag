"""User-facing DAG specification.

A model is defined by one dict ``{node_name: NodeSpec}``. Each node declares its
parents together with the per-edge term type (multi-parent is the native case):

- ``"ls"`` — linear shift (scalar weight per parent feature),
- ``"cs"`` — complex shift (MLP, additive on the latent scale),
- ``"ci"`` — complex intercept (the 1-D transform's parameters depend on the
  parent; several ``ci`` parents of one node feed a single joint network,
  which reproduces tramdag's interacting-ci groups automatically).
"""

from __future__ import annotations

from dataclasses import dataclass, field

TERMS = ("ls", "cs", "ci")


@dataclass
class ContinuousNode:
    """Continuous variable, modelled by a monotone 1-D transform + shifts.

    Args:
        parents: mapping parent name -> term type ("ls" | "cs" | "ci").
        transform: "bernstein" (TRAM-faithful), "spline" or "affine".
        transform_kwargs: forwarded to the transform
            (e.g. n_coeffs=20 for bernstein, bins=8 for spline).
    """
    parents: dict[str, str] = field(default_factory=dict)
    transform: str = "bernstein"
    transform_kwargs: dict = field(default_factory=dict)
    kind: str = field(default="continuous", init=False)


@dataclass
class OrdinalNode:
    """Ordinal variable with ``levels`` ordered classes (0 .. levels-1),
    modelled by increasing cutpoints (ordered logit) + shifts."""
    levels: int
    parents: dict[str, str] = field(default_factory=dict)
    kind: str = field(default="ordinal", init=False)


NodeSpec = ContinuousNode | OrdinalNode


def validate_and_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Validate the spec and return a topological ordering of the nodes."""
    for name, node in spec.items():
        for parent, term in node.parents.items():
            if parent not in spec:
                raise ValueError(f"Node '{name}': unknown parent '{parent}'.")
            if term not in TERMS:
                raise ValueError(
                    f"Edge {parent} -> {name}: term '{term}' not in {TERMS}.")
        if isinstance(node, OrdinalNode) and node.levels < 2:
            raise ValueError(f"Node '{name}': ordinal levels must be >= 2.")

    # Kahn's algorithm
    remaining = {name: set(node.parents) for name, node in spec.items()}
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
        d = {"kind": node.kind, "parents": dict(node.parents)}
        if isinstance(node, ContinuousNode):
            d["transform"] = node.transform
            d["transform_kwargs"] = dict(node.transform_kwargs)
        else:
            d["levels"] = node.levels
        out[name] = d
    return out


def spec_from_dict(d: dict) -> dict[str, NodeSpec]:
    spec: dict[str, NodeSpec] = {}
    for name, nd in d.items():
        if nd["kind"] == "continuous":
            spec[name] = ContinuousNode(parents=dict(nd["parents"]),
                                        transform=nd["transform"],
                                        transform_kwargs=dict(nd["transform_kwargs"]))
        else:
            spec[name] = OrdinalNode(levels=int(nd["levels"]),
                                     parents=dict(nd["parents"]))
    return spec
