"""A node fits on its own: alone, in a flow, in processes, and round trips."""

# %% imports ---------------------------------------------------------------------------
import copy
import math

import pytest
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, Node, OrdinalNode
from tramdag.callbacks import EarlyStopping
from tramdag.fitting import _node_seed


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


# %% public functions ------------------------------------------------------------------
def test_a_node_fitted_alone_equals_the_node_in_a_flow(ls_chain):
    """Same start, same rows, same shuffling seed: the same weights.

    The standalone node gets a frame with only its own and its parents'
    columns, and the flow's derived seed for its position.
    """
    df = ls_chain["draw"](600, 0)
    flow = CausalFlowDAG(_ls_spec(), seed=0)
    alone = Node("y", flow.spec["y"], {"x1": "continuous", "x2": "continuous", "t": 2})
    alone.load_state_dict(copy.deepcopy(flow.nodes["y"].state_dict()))
    flow.fit(df, epochs=5, batch_size=128, seed=3)
    index = flow.order.index("y")
    alone.fit(
        df[["y", "x1", "x2", "t"]], epochs=5, batch_size=128, seed=_node_seed(3, index)
    )
    assert _same_weights(alone, flow.nodes["y"])
    assert alone.history == flow.nodes["y"].history
    assert alone.nll(df) == pytest.approx(flow.nll(df)["y"], rel=1e-6)


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
    flow.fit(df, epochs=lambda name: 3 if name == "x1" else 6, batch_size=128)
    train = flow.history["train"]
    assert len(train) == 6
    assert [row["x1"] for row in train[3:]] == [
        flow.nodes["x1"].history["train"][-1]
    ] * 3
    assert all(not math.isnan(v) for row in train for v in row.values())


def test_per_node_early_stopping_through_the_flow(ls_chain):
    """A callbacks factory stops each node at its own epoch."""
    df = ls_chain["draw"](800, 4)
    flow = CausalFlowDAG(_ls_spec(), seed=0)
    flow.fit(
        df,
        epochs=2000,
        validation_split=0.25,
        callbacks=lambda name: EarlyStopping(patience=10),
    )
    ran = {name: len(nd.history["train"]) for name, nd in flow.nodes.items()}
    assert all(r < 2000 for r in ran.values())
    assert len(set(ran.values())) > 1  # the nodes stop at different epochs
