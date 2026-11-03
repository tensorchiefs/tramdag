# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training strategies: driving `fit` from the API side
#
# `fit` runs one minibatch Adam loop per node and keeps the final weights. It
# owns the loop, the per-epoch validation score and the progress output. The
# caller controls the rest through `optimizer=` and `callbacks=`.
#
# This notebook runs every shipped strategy on one small workload and ends
# with a runtime scoreboard. The table in
# [`docs/fitting.md`](../docs/fitting.md) tells which strategy to pick.

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import CausalFlowDAG, ContinuousNode, I, Node
from tramdag.callbacks import Callback, EarlyStopping
from tramdag.plots import plot_training

plt.rcParams["figure.dpi"] = 110

REPO = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)

# %% [markdown]
# The data is the tracked VACA sample of `notebooks/data/`, from the process
# of the Colab demo. The notebook runs from the repository root or from
# `notebooks/`.

# %%
df = pd.read_csv(REPO / "notebooks" / "data" / "vaca.csv")
train, val = df.iloc[:900], df.iloc[900:]

SPEC = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(I("x1")),
    "x3": ContinuousNode(I("x1", "x2")),
}
CEILING = 600
scoreboard = []


def build():
    """Give a fresh unfitted flow, seeded so every recipe starts identically."""
    torch.manual_seed(0)
    return CausalFlowDAG(SPEC)


def record(name, flow, seconds):
    """Add one finished recipe to the scoreboard and print its line."""
    nll = sum(flow.nll(val).values())
    node_epochs = sum(len(nd.history["train"]) for nd in flow.nodes.values())
    scoreboard.append(
        {
            "strategy": name,
            "val_nll": nll,
            "node_epochs": node_epochs,
            "seconds": seconds,
        }
    )
    print(
        f"{name:22s} val NLL {nll:.4f}   {node_epochs:4d} node-epochs   {seconds:5.1f}s"
    )
    return nll


print(f"{len(train)} train rows, {len(val)} validation rows")

# %% [markdown]
# ## 1. What `fit` records without any callback
#
# With `validation_data=`, `fit` scores the validation NLL of each node once
# per epoch. `flow.history` holds one dict of per-node values per epoch, for
# the keys that the docstring of `CausalFlowDAG.history` lists.
# It is a view over the per-node `flow.nodes[name].history`.

# %%
flow = build()
flow.fit(train, epochs=20, batch_size=256, validation_data=val)

print("history keys:", sorted(flow.history))
print("epochs recorded:", len(flow.history["train"]))
print("per-node train NLL, last epoch:")
for node, value in flow.history["train"][-1].items():
    print(f"    {node}: {value:.4f}")
print("learning rate, last epoch:", flow.history["lr"][-1])

assert len(flow.history["train"]) == len(flow.history["val"]) == 20
assert set(flow.history["train"][-1]) == set(SPEC)

# %% [markdown]
# Without `validation_data=` there is no `val` key, and `EarlyStopping`
# refuses rather than guesses.

# %%
bare = build()
bare.fit(train, epochs=5, batch_size=256)
print("history keys without validation:", sorted(bare.history))
assert "val" not in bare.history

try:
    build().fit(train, epochs=5, batch_size=256, callbacks=EarlyStopping())
except RuntimeError as err:
    print("\nEarlyStopping without validation refuses:\n   ", err)
else:
    raise AssertionError("EarlyStopping should refuse without validation data")

# %% [markdown]
# ## 2. Plain Adam
#
# One rate, one loop, and the weights of the last epoch: the baseline for
# the other recipes. An all-`LS` model trained this way to convergence
# reaches the maximum-likelihood estimate
# ([`classical_fit_tram_dag.py`](classical_fit_tram_dag.py)).

# %%
flow = build()
t0 = time.perf_counter()
flow.fit(train, epochs=CEILING, batch_size=256, learning_rate=1e-2, validation_data=val)
nll_plain = record("plain Adam", flow, time.perf_counter() - t0)
history_plain = [sum(d.values()) for d in flow.history["val"]]

# %% [markdown]
# ## 3. Two phases
#
# A second `fit` call continues the training with a new optimizer. A second
# call at a lower rate is thus a coarse-then-fine schedule. The history
# accumulates over the calls.

# %%
flow = build()
t0 = time.perf_counter()
flow.fit(train, epochs=200, batch_size=256, learning_rate=1e-2, validation_data=val)
after_coarse = sum(flow.nll(val).values())
flow.fit(train, epochs=100, batch_size=256, learning_rate=1e-3, validation_data=val)
nll_two_phase = record("two phases", flow, time.perf_counter() - t0)

print(f"    after the coarse phase: {after_coarse:.4f}")
print(f"    after the fine phase:   {nll_two_phase:.4f}")
assert len(flow.history["train"]) == 300, "the second fit did not continue the history"

# %% [markdown]
# ## 4. `EarlyStopping`: keep the best weights
#
# A flexible model can reach its best validation score part way through and
# then drift. `EarlyStopping` keeps a copy of each node's best epoch and
# loads it at the end of that node's fit. After the fit, the validation score
# of each node equals the minimum over its epochs.

# %%
flow = build()
t0 = time.perf_counter()
stopper = EarlyStopping()
flow.fit(
    train,
    epochs=CEILING,
    batch_size=256,
    learning_rate=1e-2,
    validation_data=val,
    callbacks=stopper,
)
nll_best = record("EarlyStopping", flow, time.perf_counter() - t0)

restored = flow.nll(val)
for name, nd in flow.nodes.items():
    recorded = nd.history["val"]
    best = int(np.argmin(recorded)) + 1
    print(
        f"    {name}: best epoch {best} of {len(recorded)}, "
        f"minimum {min(recorded):.4f}, last {recorded[-1]:.4f}"
    )
    assert abs(restored[name] - min(recorded)) < 1e-4, (
        f"the restored weights of {name} do not match its best recorded epoch"
    )

# %% [markdown]
# ## 5. `EarlyStopping(patience=)`: stop each node on its own
#
# With `patience`, a node also stops once its best epoch is that many epochs
# old, so each node stops at its own epoch. A function of the node name,
# `callbacks=lambda name: ...`, gives each node its own instance.

# %%
flow = build()
t0 = time.perf_counter()
stoppers = {}
flow.fit(
    train,
    epochs=CEILING,
    batch_size=256,
    learning_rate=1e-2,
    validation_data=val,
    callbacks=lambda name: stoppers.setdefault(name, EarlyStopping(patience=25)),
)
record("EarlyStopping(25)", flow, time.perf_counter() - t0)

stopped = {name: len(nd.history["train"]) for name, nd in flow.nodes.items()}
for name, spent in stopped.items():
    best = stoppers[name].best_epoch
    print(f"    {name}: stopped after {spent} of {CEILING} epochs, best was {best}")
    assert spent < CEILING, f"patience never triggered for {name}"
    assert spent - best >= 25

# %% [markdown]
# `flow.history` repeats the last entry of a node that stopped earlier, so the
# summed curve stays defined. `plot_training` marks the stop epochs and the
# best epochs.

# %%
plot_training(flow, stops=True)
plt.show()

# %% [markdown]
# ## 6. Writing your own
#
# [`docs/fitting.md`](../docs/fitting.md) gives the callback hooks. `Plateau`
# builds torch's `ReduceLROnPlateau` on the optimizer of each node and steps
# it on `node.history["val"]`.


# %%
class Plateau(Callback):
    """One torch ReduceLROnPlateau per node, driven by fit's own score."""

    def __init__(self, factor=0.3, patience=10):
        self.factor, self.patience = factor, patience
        self.scheduler = None

    def on_fit_begin(self, node, optimizer):
        """Build the scheduler here: the node's optimizer exists only now."""
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )

    def on_epoch_end(self, node, epoch, optimizer):
        """Step the scheduler, and stop once the rate has bottomed out."""
        self.scheduler.step(node.history["val"][-1])
        return optimizer.param_groups[0]["lr"] < 1e-5


flow = build()
t0 = time.perf_counter()
flow.fit(
    train,
    epochs=CEILING,
    batch_size=256,
    learning_rate=1e-2,
    validation_data=val,
    callbacks=Plateau(),
)
record("Plateau", flow, time.perf_counter() - t0)

for name, rates in flow.history["lr"][-1].items():
    print(f"    {name}: rate went 1.0e-02 -> {rates[0]:.1e}")
    assert rates[0] < 1e-2, f"the scheduler never decayed the rate of {name}"

# %% [markdown]
# A callable in a list is an `on_epoch_end` hook. This one records the
# validation NLL of each node after every epoch.

# %%
trace = []
build().fit(
    train,
    epochs=5,
    batch_size=256,
    callbacks=[lambda node, epoch, opt: trace.append((node.name, node.nll(val)))],
)
print(f"{len(trace)} entries, the first: {trace[0]}")
assert len(trace) == 5 * len(SPEC)

# %% [markdown]
# `optimizer=` takes a factory `f(node)`. This one gives the networks weight
# decay in their own parameter group. `flow.history["lr"]` shows both groups
# of every node.


# %%
def adam_with_net_decay(node):
    """Adam with weight decay on the network parameters only."""
    groups = node.parameter_groups()
    rest = groups["intercept"] + groups["linear"]
    nets = {"params": groups["networks"], "weight_decay": 1e-4}
    groups = [{"params": rest}, nets]
    return torch.optim.Adam(groups, lr=1e-2)


flow = build()
flow.fit(train, epochs=20, batch_size=256, optimizer=adam_with_net_decay)
print("rates per group, last epoch:", flow.history["lr"][-1])
assert all(set(rates) == {0, 1} for rates in flow.history["lr"][-1].values())

# %% [markdown]
# ## 7. One node alone
#
# A node is a TRAM regression of one variable on its parents and fits
# without the DAG. It takes its name, its node spec and a parent schema,
# `{parent: "continuous" | n_levels}`.

# %%
x3 = Node("x3", SPEC["x3"], {"x1": "continuous", "x2": "continuous"})
x3.fit(
    train[["x1", "x2", "x3"]],
    epochs=CEILING,
    validation_data=val[["x1", "x2", "x3"]],
    callbacks=EarlyStopping(patience=25),
)
print(f"x3 alone: {len(x3.history['train'])} epochs, val NLL {x3.nll(val):.4f}")
assert abs(x3.nll(val) - min(x3.history["val"])) < 1e-4

# %% [markdown]
# ## 8. The scoreboard: a runtime comparison
#
# Each row gives the wall-clock seconds, the node-epochs summed over the
# nodes, the validation NLL and its gap to the best recipe. A recipe that
# stops early wins on seconds only if its NLL stays near the best. The
# seconds apply to this machine only.

# %%
board = pd.DataFrame(scoreboard).set_index("strategy")
board["nll_gap"] = board["val_nll"] - board["val_nll"].min()
board["s_per_node_epoch"] = board["seconds"] / board["node_epochs"]
print(board.to_string(float_format=lambda v: f"{v:.4f}"))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.scatter(board["seconds"], board["nll_gap"])
for name, row in board.iterrows():
    ax.annotate(
        name,
        (row["seconds"], row["nll_gap"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )
ax.margins(x=0.15)
ax.set_xlabel("wall-clock seconds")
ax.set_ylabel("validation NLL above the best recipe")
ax.set_title("cost against quality, one workload")
fig.tight_layout()
plt.show()

assert nll_best <= nll_plain + 1e-6, (
    f"EarlyStopping scored {nll_best:.4f} against plain Adam's {nll_plain:.4f}"
)

# %% [markdown]
# The restored `EarlyStopping` score lies below the minimum of the plain Adam
# curve: each node restores its own best epoch, while the summed curve has
# one epoch for all nodes.

# %%
assert nll_best <= min(history_plain) + 1e-6
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(np.arange(1, len(history_plain) + 1), history_plain, label="plain Adam")
ax.axhline(nll_best, ls="--", lw=1, color="C1", label="EarlyStopping, restored")
ax.set_xlabel("epoch")
ax.set_ylabel("validation NLL (total)")
ax.set_ylim(min(nll_best, min(history_plain)) - 0.02, min(history_plain) + 0.4)
ax.legend()
ax.set_title("keeping the best epoch against keeping the last")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Which one, when
#
# The decision table lives in [`docs/fitting.md`](../docs/fitting.md).
