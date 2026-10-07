"""Figures of a TRAM-DAG: the labelled DAG, the marginals, the training curve.

matplotlib is an optional dependency: ``pip install "tramdag[plots]"``. It is
imported on the first call, so importing tramdag never needs it.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .spec import NodeSpec, node_parents, validate_and_sort

# %% global variables ------------------------------------------------------------------
__all__ = ["plot_dag", "plot_marginals", "plot_training", "plot_varying_coef"]
# how each term draws its edge; an unregistered term falls back to dotted gray
EDGE_STYLE = {
    "LS": dict(ls="-", lw=1.3),
    "CS": dict(ls="-", lw=2.4),
    "CI": dict(ls="--", lw=1.8),
    "VC": dict(ls="-", lw=2.4),
    "VC modifier": dict(ls=":", lw=1.4),
}
EDGE_LABEL = {"VCm": "VC modifier"}  # to_matrix's tag for a VC modifier
# the colours of every drawn element; `style=` picks one or overrides keys
STYLES = {
    "light": dict(
        text="black",
        muted="0.4",
        node_edge="0.3",
        node_face={"continuous": "#e3f2fd", "ordinal": "#fff3e0"},
        edge={"LS": "0.25", "CS": "C0", "CI": "C1", "VC": "C3", "VC modifier": "C3"},
        label_box="white",
        font_scale=1.0,
    ),
    "dark": dict(
        text="white",
        muted="0.75",
        node_edge="0.8",
        node_face={"continuous": "#1e3a5f", "ordinal": "#5c3d14"},
        edge={
            "LS": "0.85",
            "CS": "#4fc3f7",
            "CI": "#ffb74d",
            "VC": "#ef5350",
            "VC modifier": "#ef5350",
        },
        label_box="#202124",
        font_scale=1.0,
    ),
}
NODE_H, ROW_DY = 0.56, 1.1  # layout units: node height, distance between rows
INCH_PER_UNIT = 0.9  # a new DAG figure's size per layout unit; the fonts fit it
BULGE = 0.5  # how far an edge that skips a layer bends out, per skipped layer


# %% private functions -----------------------------------------------------------------
def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            'tramdag.plots needs matplotlib: pip install "tramdag[plots]"'
        ) from err
    return plt


def _style(style) -> dict:
    """Give the full style: a preset name, or a dict over the light preset."""
    if style is None or isinstance(style, str):
        return STYLES[style or "light"]
    base = STYLES["light"]
    merged = base | style
    for key in ("node_face", "edge"):
        merged[key] = base[key] | style.get(key, {})
    return merged


def _edge_style(name: str, st: dict) -> dict:
    """Give the line style of one edge kind in the colours of ``st``."""
    if name not in EDGE_STYLE:
        return dict(color="0.5", ls=":", lw=1.2)
    return EDGE_STYLE[name] | dict(color=st["edge"][name])


def _node_width(name: str, node: NodeSpec | None = None) -> float:
    """Wide enough for the name in bold 10 pt and the kind sub-label."""
    width = max(1.0, 0.13 * len(name) + 0.3)
    if node is not None:
        width = max(width, 0.065 * len(_kind_label(node)) + 0.2)
    return width


def _kind_label(node: NodeSpec) -> str:
    """Give the node's kind sub-label."""
    return (
        "continuous" if node.kind == "continuous" else f"ordinal · {node.levels} levels"
    )


def _layout(spec: dict[str, NodeSpec]) -> tuple[dict[str, tuple[float, float]], float]:
    """Layered left-to-right positions: a node sits one layer past its parents.

    Within a layer the nodes follow their parents' mean row (one barycenter
    sweep), which keeps most edges short and uncrossed on the DAGs this
    package is for. Rows are centered on 0. Also gives the layer distance,
    which grows with the widest node.
    """
    layer_dx = max(_node_width(n, spec[n]) for n in spec) + 1.0
    order = validate_and_sort(spec)
    depth: dict[str, int] = {}
    for name in order:
        depth[name] = 1 + max((depth[p] for p in node_parents(spec[name])), default=-1)
    layers: dict[int, list[str]] = defaultdict(list)
    for name in order:
        layers[depth[name]].append(name)
    pos: dict[str, tuple[float, float]] = {}
    for d in sorted(layers):
        names = layers[d]
        if d > 0:
            names.sort(
                key=lambda n: np.mean([pos[p][1] for p in node_parents(spec[n])])
            )
        for i, n in enumerate(names):
            pos[n] = (d * layer_dx, ((len(names) - 1) / 2 - i) * ROW_DY)
    return pos, layer_dx


def _term_edges(child: str, term) -> list[tuple[str, str, str, bool]]:
    """Give ``(parent, child, label, joint)`` for the edges one term owns.

    Read off the term's adjacency ``cells``: the tag is the term name (``CI``
    for an intercept edge, ``VCm`` for a VC modifier) and ``joint`` is the
    term's own answer; ``cells`` is the per-term authority here.
    """
    return [
        (parent, child, EDGE_LABEL.get(tag, tag), joint)
        for parent, tag, joint in term.cells()
    ]


def _draw_node(ax, name: str, node: NodeSpec, xy, st: dict, node_kind: bool):
    from matplotlib.patches import Ellipse, FancyBboxPatch

    x, y = xy
    w = _node_width(name, node if node_kind else None)
    face, edge = st["node_face"][node.kind], st["node_edge"]
    if node.kind == "ordinal":
        patch = FancyBboxPatch(
            (x - w / 2, y - NODE_H / 2),
            w,
            NODE_H,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            fc=face,
            ec=edge,
            lw=1.2,
            zorder=3,
        )
    else:
        patch = Ellipse((x, y), w, NODE_H, fc=face, ec=edge, zorder=3)
    ax.add_patch(patch)
    fs = st["font_scale"]
    name_y = y + 0.06 if node_kind else y
    ax.text(
        x,
        name_y,
        name,
        ha="center",
        va="center",
        fontsize=10 * fs,
        weight="bold",
        color=st["text"],
        zorder=4,
    )
    if node_kind:
        ax.text(
            x,
            y - 0.13,
            _kind_label(node),
            ha="center",
            va="center",
            fontsize=6.5 * fs,
            color=st["muted"],
            zorder=4,
        )
    return patch


def _bulge(pos, layer_dx: float, edge, lane: float) -> float:
    """How far an edge bends out of its chord (positive = upward).

    An edge that skips layers bends away from the middle row, where the
    nodes are, by ``BULGE`` per skipped layer; parallel edges of one pair (a
    VC modifier next to its CS) take separate lanes.
    """
    (x0, y0), (x1, y1) = pos[edge[0]], pos[edge[1]]
    skipped = round((x1 - x0) / layer_dx) - 1
    return BULGE * skipped * (1 if y0 + y1 >= 0 else -1) + lane


def _arc(pos, edge, bulge: float, t) -> np.ndarray:
    """Give the points at ``t`` along an edge's arc, shape ``(len(t), 2)``.

    The arc is matplotlib's ``arc3``, a quadratic Bezier curve whose midpoint
    lies ``bulge`` off the chord's midpoint.
    """
    p0, p2 = np.array(pos[edge[0]]), np.array(pos[edge[1]])
    d = p2 - p0
    normal = np.array([-d[1], d[0]]) / np.hypot(*d)
    control = (p0 + p2) / 2 + 2 * bulge * normal
    t = np.asarray(t, dtype=float)[:, None]
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * control + t**2 * p2


def _hits(spec, pos, pts, skip=()) -> bool:
    """Say whether any point lies on a node other than those in ``skip``."""
    return any(
        (
            (np.abs(pts[:, 0] - x) < _node_width(n, spec[n]) / 2 + 0.1)
            & (np.abs(pts[:, 1] - y) < NODE_H / 2 + 0.1)
        ).any()
        for n, (x, y) in pos.items()
        if n not in skip
    )


def _clear(spec, pos, edge, bulge: float) -> float:
    """Give the smallest bend near ``bulge`` whose arc misses every other node.

    Bends on both sides are tried, up to 2 units beyond ``bulge``; when none
    clears, the edge keeps ``bulge``.
    """
    t = np.linspace(0.1, 0.9, 25)
    for step in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0):
        if not _hits(spec, pos, _arc(pos, edge, bulge + step, t), edge[:2]):
            return bulge + step
    return bulge


def _draw_edge(
    ax, patches, spec, pos, edge, labels: bool, bulge: float, st: dict
) -> None:
    from matplotlib.patches import FancyArrowPatch

    parent, child, name, joint = edge
    (x0, y0), (x1, y1) = pos[parent], pos[child]
    style = _edge_style(name, st)
    # arc3 bulges by rad * length / 2 to the right of its direction of travel
    dist = float(np.hypot(x1 - x0, y1 - y0))
    rad = -2 * bulge / dist
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        patchA=patches[parent],
        patchB=patches[child],
        arrowstyle="-|>,head_length=6,head_width=3",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        **style,
    )
    ax.add_patch(arrow)
    if labels:
        box = st["label_box"]
        # toward the child, so crossing edges into different children separate;
        # the midpoint when that spot lies on a node
        at = _arc(pos, edge, bulge, [0.62])
        if _hits(spec, pos, at):
            at = _arc(pos, edge, bulge, [0.5])
        ax.text(
            *at[0],
            name + (" joint" if joint else ""),
            fontsize=6.5 * st["font_scale"],
            color=style["color"],
            ha="center",
            va="center",
            zorder=2.5,
            bbox=None if box is None else dict(fc=box, ec="none", pad=0.6, alpha=0.85),
        )


def _draw_modifier(ax, patches, pos, mod: str, target, treatment: str, st: dict):
    """Draw a VC modifier's arrow onto the midpoint of its treatment edge.

    The arrow bends around the treatment node when the straight line would
    cross it, as it does for a modifier in the treatment's row.
    """
    from matplotlib.patches import FancyArrowPatch

    (x0, y0), (tx, ty) = pos[mod], target
    (cx, cy) = pos[treatment]
    dx, dy = tx - x0, ty - y0
    s = np.clip(((cx - x0) * dx + (cy - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    crosses = np.hypot(x0 + s * dx - cx, y0 + s * dy - cy) < NODE_H
    rad = (0.4 if cy <= y0 else -0.4) if crosses else 0.0
    style = _edge_style("VC modifier", st)
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            target,
            patchA=patches[mod],
            arrowstyle="-|>,head_length=5,head_width=2.5",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=3,
            **style,
        )
    )
    ax.plot(*target, "o", ms=3.5, color=style["color"])


def _dag_edges(spec, modifiers: str) -> tuple[list, list]:
    """Give the drawn edges and, in ``"edge"`` mode, ``(modifier, treatment edge)``."""
    edges, mods = [], []
    for child, node in spec.items():
        for term in node.terms:
            term_edges = _term_edges(child, term)
            if modifiers == "edge" and term.name == "VC":
                treatment = term_edges[0]
                mods += [(e[0], treatment) for e in term_edges[1:]]
                term_edges = [treatment]
            edges += term_edges
    return edges, mods


def _bulges(spec, pos, layer_dx: float, edges) -> dict:
    """Give each edge its bend: parallel edges in lanes, every arc clear of nodes."""
    parallel = defaultdict(list)
    for edge in edges:
        parallel[edge[:2]].append(edge)
    bulges = {}
    for pair in parallel.values():
        for k, edge in enumerate(pair):
            lane = 0.35 * (k - (len(pair) - 1) / 2)
            bulges[edge] = _clear(spec, pos, edge, _bulge(pos, layer_dx, edge, lane))
    return bulges


def _limits(spec, pos, bulges) -> tuple[float, float, float, float]:
    """Give the axes limits: the nodes plus room for the arcs."""
    xs, ys = (np.array(v) for v in zip(*pos.values(), strict=True))
    arcs = [_arc(pos, e, b, np.linspace(0, 1, 11))[:, 1] for e, b in bulges.items()]
    arc_y = np.concatenate([ys, *arcs])
    half_w = max(_node_width(n, spec[n]) for n in spec) / 2
    return (
        xs.min() - half_w - 0.2,
        xs.max() + half_w + 0.2,
        min(ys.min() - 0.5, arc_y.min() - 0.2),
        max(ys.max() + 0.5, arc_y.max() + 0.2),
    )


def _legend(ax, names: set[str], st: dict) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], label=e, **_edge_style(e, st)) for e in EDGE_STYLE if e in names
    ]
    if handles:
        leg = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(handles),
            fontsize=7 * st["font_scale"],
            frameon=False,
        )
        for text in leg.get_texts():
            text.set_color(st["text"])


def _finish(fig, path, created: bool) -> None:
    if created:
        fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")


# %% public functions ------------------------------------------------------------------
def plot_dag(
    spec_or_flow,
    *,
    labels: bool = True,
    legend: bool = True,
    node_kind: bool = True,
    modifiers: str = "edge",
    style=None,
    ax=None,
    path=None,
    title=None,
):
    """Draw the labelled DAG of a spec (or of a fitted flow).

    Layers run left to right, a node one layer past its parents. Continuous
    nodes are ellipses, ordinal nodes rounded boxes with their level count.
    Each edge is drawn by the term that owns it: ``LS`` thin gray, ``CS`` thick
    blue, a complex intercept (``CI``) dashed orange, a ``VC`` treatment edge
    red; a multi-parent CS/CI is labelled ``joint``. A VC modifier is a dotted
    red arrow onto the treatment edge it modifies.

    Parameters
    ----------
    spec_or_flow : dict[str, NodeSpec] | CausalFlowDAG
        The DAG to draw. A flow draws its ``spec``.
    labels : bool, optional
        Write the term name on each edge, by default True.
    legend : bool, optional
        Add a legend of the terms used, by default True.
    node_kind : bool, optional
        Write the node kind under each name, by default True.
    modifiers : str, optional
        ``"edge"`` (default) points a VC modifier at the midpoint of its
        treatment edge; ``"node"`` draws it as an edge into the outcome node.
    style : str | dict | None, optional
        ``"light"`` (default) or ``"dark"``, or a dict over the light preset
        with any of the keys of ``tramdag.plots.STYLES["light"]``: ``text``,
        ``muted``, ``node_edge``, ``node_face`` (per kind), ``edge`` (per
        term), ``label_box`` (``None`` for no box) and ``font_scale``.
    ax : matplotlib.axes.Axes | None, optional
        Draw into this axes; by default a new figure sized to the layout.
    path : str | Path | None, optional
        Save the figure here (150 dpi) after drawing, by default None (not
        saved).
    title : str | None, optional
        Axes title, by default none.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.

    Raises
    ------
    ValueError
        If the spec has no node.
    """
    plt = _plt()
    spec = getattr(spec_or_flow, "spec", spec_or_flow)  # a flow draws its spec
    if not spec:
        raise ValueError("plot_dag needs a spec with at least one node")
    st = _style(style)
    pos, layer_dx = _layout(spec)
    edges, mods = _dag_edges(spec, modifiers)
    bulges = _bulges(spec, pos, layer_dx, edges)
    x_lo, x_hi, y_lo, y_hi = _limits(spec, pos, bulges)
    created = ax is None
    if created:
        _, ax = plt.subplots(
            figsize=(
                INCH_PER_UNIT * (x_hi - x_lo),
                INCH_PER_UNIT * (y_hi - y_lo) + 0.5,
            )
        )
    else:  # the fonts shrink with the space the given axes has per layout unit
        box = ax.get_position()
        w, h = ax.figure.get_size_inches()
        per_unit = min(box.width * w / (x_hi - x_lo), box.height * h / (y_hi - y_lo))
        st = st | dict(font_scale=st["font_scale"] * per_unit / INCH_PER_UNIT)
    patches = {
        name: _draw_node(ax, name, spec[name], xy, st, node_kind)
        for name, xy in pos.items()
    }
    for edge, bulge in bulges.items():
        _draw_edge(ax, patches, spec, pos, edge, labels, bulge, st)
    for mod, treatment in mods:
        target = tuple(_arc(pos, treatment, bulges[treatment], [0.5])[0])
        _draw_modifier(ax, patches, pos, mod, target, treatment[0], st)
    if legend:
        names = {e for _, _, e, _ in bulges} | ({"VC modifier"} if mods else set())
        _legend(ax, names, st)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=st["text"])
    _finish(ax.figure, path, created)
    return ax


def plot_marginals(
    flow,
    df: pd.DataFrame,
    *,
    ncols: int = 3,
    seed=None,
    colors=("C0", "C1"),
    legend: str | None = "axes",
    ax=None,
    path=None,
    title="observed vs sampled marginals",
):
    """Observed vs sampled marginal of every node, one panel each.

    Ordinal nodes compare level proportions side by side; continuous nodes a
    density histogram of the data with the flow's sample as a step outline.
    The sample has as many rows as ``df``.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow.
    df : pd.DataFrame
        The data to compare against (the validation split, typically).
    ncols : int, optional
        Panels per row of a new figure, by default 3.
    seed : int | None, optional
        Seed of the flow's sample, by default None.
    colors : tuple[str, str], optional
        Colours of the data and of the flow's sample, by default
        ``("C0", "C1")``.
    legend : str | None, optional
        ``"axes"`` (default) puts a legend in every panel, ``"figure"`` one
        legend for the figure, ``None`` none.
    ax : sequence of matplotlib.axes.Axes | None, optional
        Draw into these panels, one per node in the flow's order; by default
        a new figure.
    path : str | Path | None, optional
        Save the figure here (150 dpi) after drawing, by default None (not
        saved).
    title : str | None, optional
        Figure title, by default ``"observed vs sampled marginals"``;
        ``None`` for none.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The panels, in the flow's node order (unused panels of a new figure
        are switched off).
    """
    plt = _plt()
    sample = flow.sample(len(df), seed=seed)
    created = ax is None
    if created:
        nrows = -(-len(flow.order) // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False
        )
        for unused in axes.flat[len(flow.order) :]:
            unused.set_axis_off()
    else:
        axes = np.asarray(ax, dtype=object)
        fig = axes.flat[0].figure
    for panel, name in zip(axes.flat, flow.order, strict=False):
        node = flow.spec[name]
        if node.kind == "ordinal":
            lv, w = np.arange(node.levels), 0.4
            for x, d, label, color in [
                (lv - w / 2, df, "data", colors[0]),
                (lv + w / 2, sample, "flow", colors[1]),
            ]:
                counts = d[name].value_counts(normalize=True).reindex(lv, fill_value=0)
                panel.bar(x, counts, w, label=label, color=color)
            panel.set_xticks(lv)
            panel.set_ylabel("proportion")
        else:
            # 30 bins over the central 99.8 %, so an outlier cannot flatten the
            # panel; the density counts every row, also those outside the bins
            edges = np.linspace(*np.quantile(df[name], [0.001, 0.999]), 31)
            width = edges[1] - edges[0]
            panel.hist(
                df[name],
                bins=edges,
                weights=np.full(len(df), 1 / (len(df) * width)),
                alpha=0.5,
                color=colors[0],
                label="data",
            )
            panel.hist(
                sample[name],
                bins=edges,
                weights=np.full(len(sample), 1 / (len(sample) * width)),
                histtype="step",
                lw=1.5,
                color=colors[1],
                label="flow",
            )
            panel.set_ylabel("density")
        panel.set_title(name)
        if legend == "axes":
            panel.legend(fontsize=8, frameon=False)
    if legend == "figure":
        fig.legend(
            *axes.flat[0].get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            frameon=False,
        )
    if title:
        fig.suptitle(title)
    _finish(fig, path, created)
    return axes


def plot_training(flow, *, stops=None, ax=None, path=None, title=None):
    """Draw the summed train and validation NLL per epoch.

    ``flow.history`` accumulates across ``fit`` calls, so the curves cover
    every epoch the flow has trained, not only the last call. Each validation
    entry is drawn at the epoch it was measured in, which is what keeps the two
    curves aligned when one ``fit`` validated and another did not.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow; ``flow.history`` is read.
    stops : dict[str, int] | None, optional
        ``{node: epoch}`` of the per-node stops, each a dashed mark; a
        node's stop epoch is the length of its ``history["train"]``. By
        default no marks.
    ax : matplotlib.axes.Axes | None, optional
        Draw into this axes; by default a new figure.
    path : str | Path | None, optional
        Save the figure here (150 dpi) after drawing, by default None (not
        saved).
    title : str | None, optional
        Axes title, by default ``"training"``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.

    Raises
    ------
    ValueError
        If ``flow.history`` is empty.
    """
    plt = _plt()
    hist = flow.history
    if not hist.get("train"):
        raise ValueError("plot_training needs a fitted flow; its history is empty")
    train = np.array([sum(d.values()) for d in hist["train"]])
    curves = {"train": (np.arange(1, len(train) + 1), train)}
    if hist.get("val"):
        val = np.array([sum(d.values()) for d in hist["val"]])
        # each entry's own epoch, so an unvalidated fit in between leaves a gap
        # rather than shifting the whole curve back to epoch 1
        curves["val"] = (np.asarray(hist["val_epoch"], dtype=float), val)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    for label, (x, curve) in curves.items():
        ax.plot(x, curve, label=f"{label} NLL (total)")
    # zoom past the initial drop: the top keeps 98 % of each curve after 10 %
    # of its epochs, so a validation curve that rises later stays in view
    lo = min(c.min() for _, c in curves.values())
    hi = max(np.quantile(c[len(c) // 10 :], 0.98) for _, c in curves.values())
    if hi > lo:
        ax.set_ylim(lo - 0.05 * (hi - lo), hi)
    for name, epoch in sorted((stops or {}).items(), key=lambda kv: kv[1]):
        ax.axvline(epoch, ls="--", lw=1, color="gray")
        ax.text(
            epoch,
            1.01,
            name,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(title or "training")
    _finish(ax.figure, path, created)
    return ax


def plot_varying_coef(
    flow, df: pd.DataFrame, node: str, *, by: str, t=None, ax=None, path=None
):
    r"""Draw a VC term's effect $\beta(x)$ along one modifier.

    The grid spans ``by`` over its range in ``df``; every other modifier is
    held at its median in ``df``. The values come from
    [`varying_coef`][tramdag.flow.CausalFlowDAG.varying_coef], on the node's
    latent scale. A dotted line marks $\beta = 0$.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow.
    df : pd.DataFrame
        Rows that give the range of ``by`` and the medians of the other
        modifiers.
    node : str
        The node that carries the VC term.
    by : str
        The modifier along which to draw.
    t : str | None, optional
        The treatment of the VC term; optional when the node has one.
    ax : matplotlib.axes.Axes | None, optional
        Draw into this axes; by default a new figure.
    path : str | Path | None, optional
        Save the figure here (150 dpi) after drawing, by default None (not
        saved).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    plt = _plt()
    grid = np.linspace(df[by].min(), df[by].max(), 200)
    rows = pd.DataFrame({c: np.full(len(grid), df[c].median()) for c in df.columns})
    rows[by] = grid
    beta = flow.varying_coef(rows, node, t=t)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(grid, beta)
    ax.axhline(0.0, ls=":", lw=1, color="gray")
    ax.set_xlabel(by)
    ax.set_ylabel(r"$\beta$" + f"({by})")
    _finish(ax.figure, path, created)
    return ax
