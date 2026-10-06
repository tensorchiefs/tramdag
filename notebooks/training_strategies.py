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
# owns three things only: the loop, the per-epoch validation score, and
# progress printing. Everything else is the caller's, through `optimizer=` and
# `callbacks=`.
#
# This notebook runs every shipped strategy on one small workload, so the
# comparison is like for like, and prints what each one leaves behind in
# `flow.history`, and closes with a runtime comparison of the recipes on this
# one workload. For which strategy to pick see the table in
# [`docs/fitting.md`](../docs/fitting.md).

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

# repo-relative data, whether the notebook runs from the repo root or notebooks/
REPO = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)

# The tracked 1000-row VACA sample, the same benchmark the Colab demo uses.
df = pd.read_csv(REPO / "notebooks" / "data" / "vaca.csv")
train, val = df.iloc[:900], df.iloc[900:]

SPEC = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(I("x1")),
    "x3": ContinuousNode(I("x1", "x2")),
}
CEILING = 600  # epoch ceiling for every recipe below
scoreboard = []


def build():
    """Give a fresh unfitted flow, seeded so every recipe starts identically."""
    torch.manual_seed(0)
    return CausalFlowDAG(SPEC)


def record(name, flow, seconds):
    """Add one finished recipe to the scoreboard and print its line."""
    nll = sum(flow.nll(val).values())
    epochs = len(flow.history["train"])
    scoreboard.append(
        {"strategy": name, "val_nll": nll, "epochs": epochs, "seconds": seconds}
    )
    print(f"{name:22s} val NLL {nll:.4f}   {epochs:4d} epochs   {seconds:5.1f}s")
    return nll


print(f"{len(train)} train rows, {len(val)} validation rows")

# %% [markdown]
# ## 1. What `fit` records without any callback
#
# Pass `validation_data=` and `fit` scores the validation set once per epoch,
# centrally. Every shipped callback reads that one computation rather than
# repeating it. Four keys appear in `flow.history`:
#
# - `train`, one dict of per-node negative log-likelihood per epoch,
# - `val`, the same for the validation rows, only when validation is on,
# - `val_epoch`, the train epoch each `val` entry belongs to,
# - `lr`, each node's rates `{group index: lr}` after each epoch, recorded
#   after the callbacks ran, so a schedule's decision for that epoch is what
#   gets stored.
#
# Each node keeps its own history in `flow.nodes[name].history`, one float
# per epoch. `flow.history` is a view over them.

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
# Without `validation_data=` there is no `val` key, and the shipped callbacks
# refuse rather than guess.

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
# One rate, one loop, and the weights at the last epoch. This is the baseline
# every other recipe is measured against. An all-`LS` model trained this way
# to convergence reaches the classical maximum-likelihood estimate, which
# [`classical_fit_tram_dag.py`](classical_fit_tram_dag.py) checks against
# `statsmodels` and R.

# %%
flow = build()
t0 = time.perf_counter()
flow.fit(train, epochs=CEILING, batch_size=256, learning_rate=1e-2, validation_data=val)
nll_plain = record("plain Adam", flow, time.perf_counter() - t0)
history_plain = [sum(d.values()) for d in flow.history["val"]]

# %% [markdown]
# ## 3. Two phases
#
# Calling `fit` again continues training. Each call builds a new optimizer,
# so a second call at a lower rate is a coarse-then-fine
# schedule with no callback at all. History accumulates across the calls.

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
# The loop keeps the final weights, which is what makes the classical
# agreement exact. A flexible model does not always want that: it can reach a
# better validation score part way through and then drift. `EarlyStopping`
# snapshots each node's best epoch and loads it back at the end of that node's
# fit.
#
# The check below is the one a static code block cannot make. After the fit,
# each node's validation score equals the *minimum* over its recorded epochs,
# not the last one.

# %%
flow = build()
t0 = time.perf_counter()
stopper = EarlyStopping()  # restore_best is on by default, no patience
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
# Without `patience` each node spends its whole budget and only the
# restoration happens. With it, a node also stops once its best epoch is that
# many epochs old. Nodes converge at different speeds, so they stop at
# different epochs.
#
# One shared instance resets at each node's fit begin, so its attributes
# describe the last node only. A factory `callbacks=lambda name: ...` gives
# each node its own instance.

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
# summed curve stays defined. `plot_training` marks the stop epochs.

# %%
plot_training(flow, stops=stopped)
plt.show()

# %% [markdown]
# ## 6. Writing your own
#
# The callback contract is in [`docs/fitting.md`](../docs/fitting.md): every
# hook receives the node, a bare callable is an `on_epoch_end` hook, and a
# `Callback` subclass gets the other two hooks. `Plateau` below builds torch's
# `ReduceLROnPlateau` on each node's optimizer and steps it on the validation
# score `fit` has already put in `node.history["val"]`.


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
# A one-line callable is often enough. This one records each node's
# validation NLL after every epoch. Put it in a list: a callable alone is a
# per-node factory `f(name)`.
#
# ```python
# trace = []
# flow.fit(train, epochs=100, validation_data=val,
#          callbacks=[lambda node, epoch, opt: trace.append(node.nll(val))])
# ```

# %% [markdown]
# ## 7. One node alone
#
# A node is a TRAM regression of one variable on its parents and fits without
# the DAG. It takes its name, its node spec and a parent schema,
# `{parent: "continuous" | n_levels}`. The frame needs only the node's own
# column and its parents' columns.

# %%
x3 = Node("x3", SPEC["x3"], {"x1": "continuous", "x2": "continuous"})
x3.fit(
    train[["x1", "x2", "x3"]],
    epochs=CEILING,
    validation_data=val[["x1", "x2", "x3"]],
    callbacks=EarlyStopping(patience=25),
)
print(f"x3 alone: {len(x3.history['train'])} epochs, val NLL {x3.nll(val):.4f}")
assert x3.nll(val) == min(x3.history["val"])  # the restored best epoch

# %% [markdown]
# ## 8. The scoreboard: a poor man's runtime comparison
#
# One workload, one seed, one machine, every recipe: the wall-clock seconds
# each one took, the epochs it spent, its validation NLL and the gap to the
# best NLL on the board. A recipe that stops itself wins on seconds only if it
# also stays near the best NLL, which is what the last two columns show side
# by side. Absolute seconds are this machine's; the ranking is what travels.

# %%
board = pd.DataFrame(scoreboard).set_index("strategy")
board["nll_gap"] = board["val_nll"] - board["val_nll"].min()
board["s_per_epoch"] = board["seconds"] / board["epochs"]
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
ax.set_xlabel("wall-clock seconds")
ax.set_ylabel("validation NLL above the best recipe")
ax.set_title("cost against quality, one workload")
fig.tight_layout()
plt.show()

# Keeping the best weights cannot score worse than keeping the last ones.
assert nll_best <= nll_plain + 1e-6, (
    f"EarlyStopping scored {nll_best:.4f} against plain Adam's {nll_plain:.4f}"
)

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(np.arange(1, len(history_plain) + 1), history_plain, label="plain Adam")
ax.axhline(nll_best, ls="--", lw=1, color="C1", label="EarlyStopping, restored")
ax.set_xlabel("epoch")
ax.set_ylabel("validation NLL (total)")
ax.set_ylim(min(history_plain) - 0.02, min(history_plain) + 0.4)
ax.legend()
ax.set_title("keeping the best epoch against keeping the last")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Which one, when
#
# The decision table lives in [`docs/fitting.md`](../docs/fitting.md).
