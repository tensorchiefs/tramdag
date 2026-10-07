"""``tramdag.plots``: every term draws, and the missing extra is named."""

# %% imports ---------------------------------------------------------------------------
import sys

import matplotlib as mpl
import pandas as pd
import pytest

mpl.use("Agg")

from tramdag import CI, CS, LS, VC, CausalFlowDAG, ContinuousNode, OrdinalNode, plot_dag
from tramdag.callbacks import EarlyStopping
from tramdag.plots import plot_marginals, plot_training, plot_varying_coef


# %% private functions -----------------------------------------------------------------
def _every_term_spec():
    return {
        "x1": ContinuousNode(),
        "x2": OrdinalNode(3, CI("x1")),
        "t": OrdinalNode(2, LS("x1") + CS("x2")),
        "y": ContinuousNode(CS("x1", "x2") + VC("x1", t="t")),
    }


# %% public functions ------------------------------------------------------------------
def test_plot_dag_draws_every_node_and_edge():
    """One patch per node, one arrow per edge, a label per edge, a legend."""
    spec = _every_term_spec()
    ax = plot_dag(spec)
    n_edges = 1 + 2 + 2 + 1 + 1  # CI, LS+CS, joint CS (2 parents), VC, VC modifier
    arrows = [p for p in ax.patches if type(p).__name__ == "FancyArrowPatch"]
    assert len(ax.patches) == len(spec) + n_edges
    assert len(arrows) == n_edges
    labels = {t.get_text() for t in ax.texts}
    assert {"CI", "LS", "CS", "CS joint", "VC"} <= labels
    assert len(ax.lines) == 1  # the modifier's dot on the treatment edge
    legend = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "VC modifier" in legend
    # the modifier as an edge into the outcome, with its own label
    node_mode = plot_dag(spec, modifiers="node")
    assert "VC modifier" in {t.get_text() for t in node_mode.texts}
    assert not node_mode.lines
    # a flow draws its spec; labels and legend are optional
    flow = CausalFlowDAG(spec, seed=0)
    ax2 = plot_dag(flow, labels=False, legend=False, title="d")
    assert ax2.get_title() == "d"
    assert len(ax2.patches) == len(ax.patches)
    assert ax2.get_legend() is None
    assert {t.get_text() for t in ax2.texts} == {
        *spec,
        "continuous",
        "ordinal · 3 levels",
        "ordinal · 2 levels",
    }


def test_plot_dag_layers_children_past_their_parents():
    """A node sits one layer right of its deepest parent."""
    from tramdag.plots import _layout

    pos, layer_dx = _layout(_every_term_spec())
    assert [pos[n][0] / layer_dx for n in ("x1", "x2", "t", "y")] == [0, 1, 2, 3]


def test_marginals_and_training_draw_from_a_fitted_flow(ls_chain, tmp_path):
    """Both figures read the flow after a short per-node fit; ``path`` saves."""
    df = ls_chain["draw"](300, 0)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))}
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(
        df,
        epochs=30,
        batch_size=100,
        validation_data=df,
        callbacks=lambda name: EarlyStopping(patience=8),
    )
    axes = plot_marginals(flow, df, ncols=2, seed=0, path=tmp_path / "m.png", title="m")
    assert axes.flat[0].figure._suptitle.get_text() == "m"
    assert axes.shape == (1, 2)
    assert (tmp_path / "m.png").exists()
    ax = plot_training(flow, path=tmp_path / "t.png", title="t")
    assert ax.get_title(loc="left") == "t"
    assert len(ax.lines) == 2  # train and val, no marks without stops=
    assert (tmp_path / "t.png").exists()
    stops = {n: len(nd.history["train"]) for n, nd in flow.nodes.items()}
    ax = plot_training(flow, stops=stops)
    assert len(ax.lines) == 2 + len(stops)  # one mark per stop
    # no validation history, no marks: one line
    flow2 = CausalFlowDAG(spec, seed=0)
    flow2.fit(df, epochs=3, batch_size=100)
    assert len(plot_training(flow2).lines) == 1


def test_plots_refuse_nothing_to_draw():
    """An empty spec and an unfitted flow fail by name, not inside matplotlib."""
    with pytest.raises(ValueError, match="at least one node"):
        plot_dag({})
    with pytest.raises(ValueError, match="history is empty"):
        plot_training(CausalFlowDAG({"x1": ContinuousNode()}))


def test_plots_name_the_optional_dependency(monkeypatch):
    """Without matplotlib the error says what to install."""
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"tramdag\[plots\]"):
        plot_dag(_every_term_spec())


def test_the_validation_curve_keeps_its_own_epochs(ls_chain):
    """History accumulates across fits, so val must carry its epoch.

    After a fit without validation and one with it, the validation curve
    starts at the second fit's first epoch, aligned with the training curve.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))}
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=3, batch_size=100)
    flow.fit(df, epochs=2, batch_size=100, validation_split=0.2)
    assert flow.history["val_epoch"] == [4, 5]
    ax = plot_training(flow)
    drawn = {
        line.get_label().split()[0]: [float(v) for v in line.get_xdata()]
        for line in ax.get_lines()
        if line.get_label().startswith(("train", "val"))
    }
    assert drawn["train"] == [1, 2, 3, 4, 5]
    assert drawn["val"] == [4, 5]


def test_plots_draw_into_given_axes_and_take_a_style(ls_chain):
    """ax= draws into the caller's layout; style and node_kind change the look."""
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2)
    ax = plot_dag(_every_term_spec(), ax=left, style="dark", node_kind=False)
    assert ax is left
    assert len(fig.axes) == 2  # no new figure, no new axes
    texts = {t.get_text() for t in left.texts}
    assert "continuous" not in texts  # no kind sub-labels
    assert {t.get_color() for t in left.texts if t.get_text() == "x1"} == {"white"}
    df = ls_chain["draw"](200, 1)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))}
    flow = CausalFlowDAG(spec, seed=0).fit(df, epochs=2, validation_split=0.2)
    assert plot_training(flow, ax=right) is right
    fig2, panels = plt.subplots(1, 2)
    out = plot_marginals(flow, df, ax=panels, legend="figure", title=None)
    assert list(out) == list(panels)
    assert fig2._suptitle is None
    assert len(fig2.legends) == 1
    assert all(p.get_legend() is None for p in panels)


def test_plot_varying_coef_draws_the_vc_effect(ls_chain):
    """One line of beta along the modifier, as varying_coef gives it."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(float)
    df = pd.DataFrame({"x": x, "t": t, "y": x + (1 + x) * t + rng.logistic(size=n)})
    spec = {
        "x": ContinuousNode(),
        "t": OrdinalNode(2),
        "y": ContinuousNode(LS("x") + VC("x", t="t")),
    }
    flow = CausalFlowDAG(spec, seed=0).fit(df, epochs=2)
    ax = plot_varying_coef(flow, df, "y", by="x")
    line = ax.get_lines()[0]
    grid = pd.DataFrame({"x": line.get_xdata()})
    assert np.allclose(line.get_ydata(), flow.varying_coef(grid, "y"))


def test_plot_training_survives_a_diverged_epoch(ls_chain):
    """An infinite NLL in the history must not blow the zoom up."""
    import math

    df = ls_chain["draw"](200, 2)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))}
    flow = CausalFlowDAG(spec, seed=0).fit(df, epochs=20, validation_split=0.2)
    flow.nodes["x2"].history["val"][12] = math.inf
    lo, hi = plot_training(flow).get_ylim()
    assert math.isfinite(lo)
    assert math.isfinite(hi)
