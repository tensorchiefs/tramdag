"""A node fits on its own: alone, in a flow, in processes, and round trips."""

# %% imports ---------------------------------------------------------------------------
import math
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CI, CS, LS, VC, CausalFlowDAG, ContinuousNode, Node, OrdinalNode
from tramdag.callbacks import Callback

# %% global variables ------------------------------------------------------------------
_LINUX = pytest.mark.skipif(sys.platform != "linux", reason="n_jobs forks: Linux only")


# %% private functions -----------------------------------------------------------------
def _ls_spec():
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(LS("x1")),
        "t": OrdinalNode(2, LS("x1") + LS("x2")),
        "y": OrdinalNode(4, LS("x1") + LS("x2") + LS("t")),
    }


def _same_weights(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _scale(x, train):
    return x / train.std()


# %% public functions ------------------------------------------------------------------
def test_a_node_fitted_alone_equals_the_node_in_a_flow(ls_chain):
    """Same seeds, same rows: the same weights.

    The standalone node gets a frame with only its own and its parents'
    columns, and the seeds of the flow.
    """
    df = ls_chain["draw"](600, 0)
    flow = CausalFlowDAG(_ls_spec(), seed=0)
    schema = {"x1": "continuous", "x2": "continuous", "t": 2}
    alone = Node("y", flow.spec["y"], schema, seed=0)
    flow.fit(df, epochs=5, batch_size=128, seed=3)
    alone.fit(df[["y", "x1", "x2", "t"]], epochs=5, batch_size=128, seed=3)
    assert _same_weights(alone, flow.nodes["y"])
    assert alone.history == flow.nodes["y"].history
    assert alone.nll(df) == pytest.approx(flow.nll(df)["y"], rel=1e-6)


def test_an_unrelated_root_leaves_the_other_nodes_unchanged(ls_chain):
    """Every node shuffles from the fit seed, whatever the other nodes are."""
    df = ls_chain["draw"](600, 0)
    df["z"] = np.random.default_rng(0).normal(size=len(df))
    flow = CausalFlowDAG(_ls_spec(), seed=0).fit(df, epochs=3, batch_size=128, seed=3)
    wider = CausalFlowDAG({"z": ContinuousNode(), **_ls_spec()}, seed=0)
    wider.fit(df, epochs=3, batch_size=128, seed=3)
    assert all(_same_weights(nd, wider.nodes[n]) for n, nd in flow.nodes.items())


def test_fit_and_fit_jointly_shuffle_alike(ls_chain):
    """One node, one seed: both loops see the same rows in every epoch."""
    df = ls_chain["draw"](600, 0)[["x1"]]
    a = CausalFlowDAG({"x1": ContinuousNode()}, seed=0)
    b = CausalFlowDAG({"x1": ContinuousNode()}, seed=0)
    a.fit(df, epochs=3, batch_size=128, seed=3)
    b.fit_jointly(df, epochs=3, batch_size=128, seed=3)
    assert _same_weights(a, b)


@_LINUX
def test_processes_give_the_serial_result(ls_chain):
    """``n_jobs=2`` forks the node fits; weights and histories are identical."""
    df = ls_chain["draw"](600, 1)
    serial = CausalFlowDAG(_ls_spec(), seed=0)
    forked = CausalFlowDAG(_ls_spec(), seed=0)
    kwargs = dict(epochs=4, batch_size=128, seed=5, validation_split=0.2)
    serial.fit(df, **kwargs)
    forked.fit(df, n_jobs=2, **kwargs)
    assert _same_weights(serial, forked)
    for name in serial.order:
        assert serial.nodes[name].history == forked.nodes[name].history


def test_a_node_round_trips_through_save_and_load(ls_chain, tmp_path):
    df = ls_chain["draw"](400, 2)
    node = Node("x2", ContinuousNode(LS("x1")), {"x1": "continuous"})
    node.fit(df, epochs=3, batch_size=128)
    node.save(tmp_path / "x2.pt")
    loaded = Node.load(tmp_path / "x2.pt")
    assert _same_weights(node, loaded)
    assert loaded.history == node.history
    assert bool(loaded.calibrated)
    assert loaded.nll(df) == pytest.approx(node.nll(df), rel=1e-6)


def test_the_flow_history_repeats_a_stopped_nodes_last_entry(ls_chain):
    df = ls_chain["draw"](400, 3)[["x1", "x2"]]
    flow = CausalFlowDAG({"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))})
    flow.fit(df, epochs={"x1": 3, "x2": 6}, batch_size=128)
    train = flow.history["train"]
    assert len(train) == 6
    assert [row["x1"] for row in train[3:]] == [
        flow.nodes["x1"].history["train"][-1]
    ] * 3
    assert all(not math.isnan(v) for row in train for v in row.values())


@_LINUX
def test_a_forked_fit_loads_a_callable_transform_and_leaves_eval_mode(ls_chain):
    """The parent takes the workers' state, train columns of any shape included."""
    df = ls_chain["draw"](300, 5)[["x1", "x2"]]
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(CI("x1", units=[4], input_transform=_scale)),
    }
    flow = CausalFlowDAG(spec, seed=0).fit(
        df, epochs=2, batch_size=128, n_jobs=2, seed=0
    )
    assert not any(nd.training for nd in flow.nodes.values())
    assert bool(flow.nodes["x2"].calibrated)


def test_a_bad_frame_fails_before_any_node_fits(ls_chain):
    df = ls_chain["draw"](300, 6)
    flow = CausalFlowDAG(_ls_spec(), seed=0)
    with pytest.raises(KeyError, match="'y'"):
        flow.fit(df, epochs=2, validation_data=df.drop(columns="y"))
    assert not any(bool(nd.calibrated) for nd in flow.nodes.values())


def test_a_node_that_refuses_calibration_leaves_no_node_fitted():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=400)
    x2 = np.where(rng.random(400) < 0.97, 1.0, 2.0)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    flow = CausalFlowDAG({"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))})
    with pytest.raises(ValueError, match="quantiles coincide"):
        flow.fit(df, epochs=3)
    assert flow.nodes["x1"].history["train"] == []
    assert not any(bool(nd.calibrated) for nd in flow.nodes.values())


def test_a_generator_of_callbacks_reaches_every_node(ls_chain):
    class StopAt2(Callback):
        def on_epoch_end(self, node, epoch, opt):
            return epoch >= 2

    df = ls_chain["draw"](300, 8)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))}
    per_node = {n: (c for c in [StopAt2()]) for n in spec}
    for callbacks in (per_node, (c for c in [StopAt2()])):
        flow = CausalFlowDAG(spec, seed=0).fit(
            df, epochs=50, callbacks=callbacks, seed=0
        )
        assert [len(nd.history["train"]) for nd in flow.nodes.values()] == [2, 2]


def test_parameter_groups_give_every_parameter_once_by_role():
    """The kinds pin the split an optimizer factory decays: networks only."""
    spec = {
        "x1": ContinuousNode(),
        "t": OrdinalNode(2, LS("x1")),
        "x2": ContinuousNode(CI("x1")),
        "y": ContinuousNode(LS("x1") + CS("x2") + VC("x1", t="t")),
    }
    flow = CausalFlowDAG(spec, seed=0)
    y = flow.nodes["y"]
    groups = y.parameter_groups()
    assert list(groups) == ["intercept", "linear", "networks"]
    named = {id(p): n for n, p in y.named_parameters()}
    names = {role: [named[id(p)] for p in ps] for role, ps in groups.items()}
    assert names["intercept"] == ["intercept.theta"]
    assert names["linear"] == ["shifts.x1.fc.weight", "shifts.t.beta0"]
    assert names["networks"] == [
        n for n in named.values() if n.startswith("shifts.") and ".net." in n
    ]
    assert sorted(n for ns in names.values() for n in ns) == sorted(named.values())
    x2 = flow.nodes["x2"].parameter_groups()
    assert x2["networks"] == list(flow.nodes["x2"].parameters())
    assert x2["intercept"] == x2["linear"] == []
