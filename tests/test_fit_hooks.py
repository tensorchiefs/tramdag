"""Tests for fit()'s hooks: ``optimizer=`` and the ``callbacks=`` list.

The critical guard is `test_torch_plateau_scheduler_preserves_exact_mle`: a
learning-rate schedule attached through the hooks must NOT break the exact-MLE
property of all-`ls` models, which is checked against statsmodels on the inline
all-`ls` DGP (see conftest).
"""

# %% imports ---------------------------------------------------------------------------
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode
from tramdag.callbacks import Callback, EarlyStopping


# %% private functions -----------------------------------------------------------------
def _two_node_spec():
    return {"x1": ContinuousNode(), "x2": ContinuousNode(LS("x1"))}


def _ls_spec():
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(LS("x1")),
        "t": OrdinalNode(2, LS("x1") + LS("x2")),
        "y": OrdinalNode(4, LS("x1") + LS("x2") + LS("t")),
    }


# %% public functions ------------------------------------------------------------------
def test_fit_improves_and_records_train_nll(ls_chain):
    df = ls_chain["draw"](800, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow._calibrate(df)
    nll0 = sum(flow.nll(df).values())
    flow.fit(df, epochs=60, learning_rate=1e-2)
    nll1 = sum(flow.nll(df).values())
    assert np.isfinite(nll1)
    assert nll1 < nll0
    assert len(flow.history["train"]) == 60
    assert set(flow.history["train"][-1]) == {"x1", "x2"}


def test_callbacks_all_run_and_any_stops(ls_chain):
    """Every callback runs even on the stop epoch (no short-circuit); a
    Callback instance gets all three hooks, a bare callable is on_epoch_end.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    node = flow.nodes["x2"]
    calls = []

    class Recorder(Callback):
        def on_fit_begin(self, nd, opt):
            calls.append(("begin", 0))

        def on_epoch_end(self, nd, epoch, opt):
            # also pins the callback contract: live node, 1-based epoch, optimizer
            assert nd is node
            assert isinstance(opt, torch.optim.Adam)
            calls.append(("stop?", epoch))
            return epoch == 2

        def on_fit_end(self, nd, opt):
            calls.append(("end", 0))

    def logger(nd, epoch, opt):
        calls.append(("log", epoch))

    node.fit(df, epochs=10, callbacks=[Recorder(), logger])
    assert calls == [
        ("begin", 0),
        ("stop?", 1),
        ("log", 1),
        ("stop?", 2),
        ("log", 2),  # the logger still ran on the stop epoch
        ("end", 0),
    ]


def test_flow_callbacks_factory_gives_each_node_its_own(ls_chain):
    """``callbacks=f(name)`` is called once per node, in topological order."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    seen = []

    def factory(name):
        return lambda nd, epoch, opt: seen.append((name, nd.name, epoch))

    flow.fit(df, epochs=2, callbacks=factory)
    assert seen == [("x1", "x1", 1), ("x1", "x1", 2), ("x2", "x2", 1), ("x2", "x2", 2)]


def test_user_optimizer_is_used_and_keeps_its_state(ls_chain):
    """``optimizer=`` replaces the default Adam; its lr, not ``learning_rate``,
    drives the fit, and its state survives into a second call.
    """
    df = ls_chain["draw"](400, 1)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    node = flow.nodes["x2"]
    node.calibrate(df)
    opt = torch.optim.SGD(node.parameters(), lr=0.0)  # a zero step: nothing moves
    before = copy.deepcopy(node.state_dict())
    node.fit(df, epochs=2, learning_rate=1e-2, optimizer=opt)
    assert all(torch.equal(before[k], v) for k, v in node.state_dict().items())
    opt = torch.optim.Adam(node.parameters(), lr=1e-2)
    node.fit(df, epochs=3, optimizer=opt)
    assert opt.state[node.shifts["x1"].fc.weight]["step"] == 3 * 1
    with pytest.raises(TypeError, match="factory"):
        flow.fit(df, epochs=1, optimizer=opt)


def test_an_optimizer_factory_may_split_the_parameters_into_groups(ls_chain):
    """A factory gives each node an optimizer with two groups: weight decay on
    the networks only. The decayed group moves, the other group's decay is 0.
    """
    from tramdag import CS

    df = ls_chain["draw"](400, 1)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(CS("x1"))}
    flow = CausalFlowDAG(spec, seed=0)

    def adamw(node):
        nets = list(node.shifts.parameters())
        rest = list(node.intercept.parameters())
        return torch.optim.AdamW(
            [
                {"params": nets, "weight_decay": 0.1},
                {"params": rest, "weight_decay": 0.0},
            ],
            lr=1e-2,
        )

    flow.fit(df, epochs=2, optimizer=adamw)
    assert flow.nodes["x2"].history["lr"][-1] == {0: 1e-2, 1: 1e-2}
    assert flow.nodes["x1"].history["lr"][-1] == {0: 1e-2, 1: 1e-2}


def test_restore_best_matches_the_manual_six_line_callback(ls_chain):
    """``EarlyStopping()`` (no patience) lands exactly where a manual
    snapshot callback does — restoration is automatic at fit end, and
    without patience the full budget runs.
    """
    df = ls_chain["draw"](800, 2)[["x1", "x2"]]
    val = ls_chain["draw"](400, 3)[["x1", "x2"]]
    node = CausalFlowDAG(_two_node_spec(), seed=0).nodes["x2"]
    manual = {"nll": float("inf"), "epoch": 0}

    def keep_best(nd, epoch, opt):
        nll = nd.nll(val)
        if nll < manual["nll"]:
            manual.update(nll=nll, epoch=epoch)

    best = EarlyStopping()
    node.fit(
        df,
        epochs=40,
        learning_rate=1e-2,
        validation_data=val,
        callbacks=[keep_best, best],
    )
    assert len(node.history["train"]) == 40  # no patience: full budget
    assert (best.best_nll, best.best_epoch) == (manual["nll"], manual["epoch"])
    assert node.nll(val) == pytest.approx(best.best_nll, rel=1e-6)


def test_restore_best_resets_between_fits(ls_chain):
    """A reused instance starts fresh: the second fit restores its own best,
    never the first fit's snapshot.
    """
    df = ls_chain["draw"](400, 6)[["x1", "x2"]]
    node = CausalFlowDAG(_two_node_spec(), seed=0).nodes["x2"]
    best = EarlyStopping()
    node.fit(df, epochs=5, validation_data=df, callbacks=best)
    first = (best.best_nll, best.best_epoch)
    node.fit(df, epochs=3, validation_data=df, callbacks=best)
    assert best.best_epoch <= 3  # counted within the second fit
    assert best.best_nll <= first[0] + 1e-9  # training continued, no stale state


def test_early_stopping_stops_and_restores_the_best(ls_chain):
    """One instance both halts the fit once the best epoch is ``patience``
    old and leaves the node at the best-validation weights.
    """
    df = ls_chain["draw"](800, 7)[["x1", "x2"]]
    val = ls_chain["draw"](400, 8)[["x1", "x2"]]
    node = CausalFlowDAG(_two_node_spec(), seed=0).nodes["x2"]
    early = EarlyStopping(patience=5)
    node.fit(df, epochs=4000, validation_data=val, callbacks=early)
    ran = len(node.history["train"])
    assert ran < 4000
    assert ran - early.best_epoch == 5
    assert node.nll(val) == pytest.approx(early.best_nll, rel=1e-6)


def test_early_stopping_without_restore_keeps_the_final_weights(ls_chain):
    """``restore_best=False`` stops but leaves the last epoch's weights."""
    df = ls_chain["draw"](800, 7)[["x1", "x2"]]
    val = ls_chain["draw"](400, 8)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(
        df,
        epochs=4000,
        validation_data=val,
        callbacks=lambda name: EarlyStopping(patience=5, restore_best=False),
    )
    final = sum(flow.nll(val).values())
    assert final == pytest.approx(
        sum(flow.history["val"][-1].values()), rel=1e-6
    )  # each node's last epoch, not its best
    with pytest.raises(ValueError, match="no-op"):
        EarlyStopping(restore_best=False)


def test_non_callable_callback_is_refused(ls_chain):
    """A ``callbacks=`` entry that is not callable must raise up front."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(TypeError, match="Callback instances or callables"):
        flow.fit(df, epochs=10, callbacks=[42])


def test_epochs_must_be_positive(ls_chain):
    """epochs=0 would skip the loop but still calibrate and run the
    after-fit hooks — refuse it instead.
    """
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(ValueError, match="epochs"):
        flow.fit(ls_chain["draw"](100, 0)[["x1", "x2"]], epochs=0)


def test_restore_best_without_an_epoch_refuses(ls_chain):
    """Restoring before any epoch is a bug in the caller's loop — loud."""
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(RuntimeError, match="nothing to restore"):
        EarlyStopping().on_fit_end(flow.nodes["x1"], None)


def test_callbacks_demand_fit_managed_validation(ls_chain):
    """EarlyStopping without validation_data/-split fails loudly at epoch 1."""
    df = ls_chain["draw"](100, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(RuntimeError, match="validation_data"):
        flow.fit(df, epochs=2, callbacks=EarlyStopping())


def test_verbose_prints_every_nth_and_final_epoch(ls_chain, capsys):
    """``fit(verbose=N)`` prints every Nth epoch plus the final one."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=5, verbose=2, validation_data=df.head(50))
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 6  # per node: epochs 2, 4 and the final 5
    assert all("train" in ln and "val" in ln for ln in lines)
    assert lines[0].startswith("x1:")
    assert lines[-1].startswith("x2:")
    assert "5/5" in lines[-1]


def test_validation_split_takes_the_tail(ls_chain):
    """A float split trains on the head, validates on the tail (Keras rule)."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=2, validation_split=0.25, batch_size=50)
    assert len(flow.history["val"]) == 2
    # calibration saw only the head: the range is the head's quantiles
    head = df.iloc[:150]
    lo = float(flow.nodes["x1"].ut.xmin)
    assert lo == pytest.approx(head["x1"].quantile(0.05), abs=1e-6)
    with pytest.raises(ValueError, match="not both"):
        flow.fit(df, epochs=1, validation_data=df, validation_split=0.5)


def test_each_node_stops_on_its_own_and_keeps_the_mle(ls_chain):
    """Per-node ``EarlyStopping`` stops every node before the epoch ceiling,
    at its own epoch, and still lands on the known truth (x2 <- x1 weight 1.2
    in the inline DGP).
    """
    df = ls_chain["draw"](2000, 4)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    stops = {}
    flow.fit(
        df,
        epochs=4000,
        batch_size=512,
        validation_data=df,
        callbacks=lambda name: stops.setdefault(name, EarlyStopping(patience=40)),
    )
    for name, nd in flow.nodes.items():
        ran = len(nd.history["train"])
        assert ran < 4000
        assert ran - stops[name].best_epoch == 40
        assert len(nd.history["lr"]) == ran
    # the flow view repeats a stopped node's last entry
    assert len(flow.history["train"]) == max(
        len(nd.history["train"]) for nd in flow.nodes.values()
    )
    assert float(flow.ls_coefficients()["x2"]["x1"][0]) == pytest.approx(1.2, abs=0.1)


def test_callbacks_reject_stale_validation_from_an_earlier_fit(ls_chain):
    """After a validated fit, an unvalidated fit must not let a callback read
    the old history["val"] entry as the current epoch.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=2, validation_data=df, callbacks=EarlyStopping())
    with pytest.raises(RuntimeError, match="validation_data"):
        flow.fit(df, epochs=2, callbacks=EarlyStopping())  # no validation now
    flow.fit(df, epochs=2, validation_data=df, callbacks=EarlyStopping())  # fine again


def test_callbacks_reject_the_class_instead_of_an_instance(ls_chain):
    """`callbacks=EarlyStopping` (forgotten parens) fails with the fix named."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(TypeError, match="instantiate it: EarlyStopping"):
        flow.fit(df, epochs=2, callbacks=EarlyStopping)


def test_each_node_is_restored_at_its_best_epoch(ls_chain):
    """Every node ends at the weights of its own best validation epoch.

    A complex shift on few rows keeps training past its best; the restore
    brings each node back, so its final validation NLL is its curve's minimum.
    """
    from tramdag import CS

    train = ls_chain["draw"](150, 1)[["x1", "x2"]]
    val = ls_chain["draw"](150, 2)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode(CS("x1"))}
    flow = CausalFlowDAG(spec, seed=0)
    stops = {}
    flow.fit(
        train,
        epochs=600,
        batch_size=150,
        learning_rate=3e-2,
        validation_data=val,
        callbacks=lambda name: stops.setdefault(name, EarlyStopping(patience=60)),
    )
    final = flow.nll(val)
    for name, nd in flow.nodes.items():
        curve = nd.history["val"]
        best = min(range(len(curve)), key=curve.__getitem__)
        assert stops[name].best_epoch == best + 1
        assert final[name] == pytest.approx(curve[best], rel=1e-6)
    assert len(flow.nodes["x2"].history["val"]) > stops["x2"].best_epoch


def test_early_stopping_min_delta_counts_a_small_gain_as_flat(ls_chain):
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    stopping = EarlyStopping(min_delta=1e6)
    flow.fit(df, epochs=5, validation_data=df, callbacks=stopping)
    assert stopping.best_epoch == 1  # nothing beats epoch 1 by a million


def test_two_restoring_callbacks_are_refused(ls_chain):
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(ValueError, match="restore weights"):
        flow.fit(
            df,
            epochs=2,
            validation_data=df,
            callbacks=[EarlyStopping(), EarlyStopping(patience=5)],
        )


def test_torch_plateau_scheduler_preserves_exact_mle(ls_chain):
    """The headline guard: all-`ls` + torch's ReduceLROnPlateau through the
    hooks must still land on the classical MLE (outcome coefficients vs
    statsmodels on the same data).
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    obs = ls_chain["draw"](2500, 3)
    flow = CausalFlowDAG(_ls_spec(), seed=3)
    X = flow.design_matrix(obs, "y", drop_first=True)
    res = OrderedModel(obs["y"].astype(int), X, distr="logit").fit(
        method="bfgs", disp=False
    )

    class Plateau(Callback):
        def on_fit_begin(self, node, opt):
            self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.3, patience=40, min_lr=1e-5
            )

        def on_epoch_end(self, node, epoch, opt):
            self.plateau.step(node.history["train"][-1])
            return opt.param_groups[0]["lr"] <= 1e-5 and epoch > 500

    flow.fit(obs, epochs=4000, batch_size=512, callbacks=lambda name: Plateau())
    coefs = flow.ls_coefficients()["y"]
    w_t = np.asarray(coefs["t"]).ravel()
    assert float(coefs["x1"][0]) == pytest.approx(res.params["x1"], abs=0.03)
    assert float(coefs["x2"][0]) == pytest.approx(res.params["x2"], abs=0.03)
    assert (w_t[1] - w_t[0]) == pytest.approx(res.params["t[1]"], abs=0.06)
    assert len(flow.nodes["y"].history["train"]) < 4000  # the callback stopped it


def test_history_accumulates_across_fit_calls(ls_chain):
    """A second fit continues the record instead of replacing it."""
    df = ls_chain["draw"](200, 5)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=2, batch_size=100)
    flow.fit(df, epochs=3, batch_size=100)
    assert len(flow.history["train"]) == 5


def test_a_diverged_fit_says_so_instead_of_blaming_the_callback(ls_chain):
    """EarlyStopping names a NaN validation curve, not its own wiring.

    A NaN never beats ``inf``, so nothing is ever snapshotted and fit end has
    nothing to restore; the message names divergence as a cause.
    """
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    with pytest.raises(RuntimeError, match="nothing to restore"):
        flow.fit(
            df,
            epochs=3,
            learning_rate=1e9,
            validation_split=0.2,
            callbacks=EarlyStopping(),
        )


def test_a_frame_it_cannot_batch_raises_instead_of_training_on_nothing(ls_chain):
    """A single-row frame raises instead of running epochs that change nothing."""
    flow = CausalFlowDAG({"a": OrdinalNode(2), "b": OrdinalNode(2, LS("a"))}, seed=0)
    one = pd.DataFrame({"a": [1], "b": [0]})
    with pytest.raises(ValueError, match="trained on no row"):
        flow.fit(one, epochs=5)


def test_the_epoch_nll_averages_over_the_rows_it_trained_on(ls_chain):
    """A skipped trailing row must not scale every node's epoch NLL down."""
    df = ls_chain["draw"](201, 1)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node_spec(), seed=0)
    flow.fit(df, epochs=1, batch_size=100)  # 100 + 100 + 1: the last is skipped
    trimmed = CausalFlowDAG(_two_node_spec(), seed=0)
    trimmed.fit(df.iloc[:200], epochs=1, batch_size=100)
    a = sum(flow.history["train"][-1].values())
    b = sum(trimmed.history["train"][-1].values())
    assert a == pytest.approx(b, rel=0.05)
