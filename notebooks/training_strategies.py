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
# `fit` is one minibatch Adam loop that keeps the final weights. It owns three
# things only: the loop, the per-epoch validation score, and progress
# printing. Everything else is the caller's, through `optimizer=` and
# `callbacks=`.
#
# This notebook runs every shipped strategy on one small workload, so the
# comparison is like for like, and prints what each one leaves behind in
# `flow.history`. It is about mechanics, not speed. For measured
# time-to-target on real workloads see
# [`docs/training-speed.md`](../docs/training-speed.md). For which strategy to
# pick see the table in [`docs/fitting.md`](../docs/fitting.md).
#
# This notebook is executed on every documentation build, so its recipes are
# checked against the current API. The snippets in `docs/fitting.md` are a
# quick reference copied from here; if the two ever disagree, this file is
# right.

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import CausalFlowDAG, ContinuousNode, I
from tramdag.callbacks import Callback, EarlyStopping, PerNodePlateau, per_node_adam
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
CEILING = 300  # epoch ceiling for every recipe below
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
# repeating it. Three keys appear in `flow.history`:
#
# - `train`, one dict of per-node negative log-likelihood per epoch,
# - `val`, the same for the validation rows, only when validation is on,
# - `lr`, the optimizer's rate after each epoch, recorded after the callbacks
#   ran, so a schedule's decision for that epoch is what gets stored.
#
# The per-node shape is not decoration. The joint likelihood decomposes per
# node, so a per-node score says which node is still improving.

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
# Calling `fit` again continues training. The optimizer is rebuilt unless you
# pass your own, so a second call at a lower rate is a coarse-then-fine
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
# snapshots the best epoch and loads it back at the end of the fit.
#
# The check below is the one a static code block cannot make. After the fit,
# the model's validation score equals the *minimum* over the recorded epochs,
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

recorded = [sum(d.values()) for d in flow.history["val"]]
print(f"    best epoch {stopper.best_epoch} of {len(recorded)}")
print(f"    minimum recorded: {min(recorded):.4f}")
print(f"    last epoch:       {recorded[-1]:.4f}")
assert abs(nll_best - min(recorded)) < 1e-4, (
    "the restored weights do not match the best recorded epoch"
)

# %% [markdown]
# ## 5. `EarlyStopping(patience=)`: stop as well as restore
#
# Without `patience` the fit spends its whole budget and only the restoration
# happens. With it, the fit also stops once the best epoch is that many
# epochs old.

# %%
flow = build()
t0 = time.perf_counter()
stopper = EarlyStopping(patience=25)
flow.fit(
    train,
    epochs=CEILING,
    batch_size=256,
    learning_rate=1e-2,
    validation_data=val,
    callbacks=stopper,
)
record("EarlyStopping(25)", flow, time.perf_counter() - t0)

spent = len(flow.history["train"])
print(f"    stopped after {spent} of {CEILING} epochs, best was {stopper.best_epoch}")
assert spent < CEILING, "patience never triggered, so the ceiling bound instead"
assert spent - stopper.best_epoch >= 25

# %% [markdown]
# ## 6. Per-node rates: `per_node_adam` with `PerNodePlateau`
#
# The per-node losses have independent gradients, so a rate per node is
# exactly independent per-node training. `per_node_adam` builds an Adam with
# one tagged parameter group per node. `PerNodePlateau` then decays each
# node's rate on that node's own validation score and freezes the node once
# the rate is low and flat. The fit stops when the last node freezes.
#
# Do not attach a torch scheduler to the same optimizer. Two controllers would
# steer the same group rates against each other.

# %%
flow = build()
t0 = time.perf_counter()
plateau = PerNodePlateau(patience=10, freeze=30)
flow.fit(
    train,
    epochs=CEILING,
    batch_size=256,
    validation_data=val,
    optimizer=per_node_adam(flow, lr=1e-2),
    callbacks=plateau,
)
record("PerNodePlateau", flow, time.perf_counter() - t0)

print(f"    froze at: {dict(sorted(plateau.frozen.items()))}")
print("    per-node rates at the end:", flow.history["lr"][-1])
assert set(plateau.frozen) == set(SPEC), "not every node froze"
assert len(flow.history["train"]) < CEILING, "the plateau rule did not self-stop"

# %% [markdown]
# `history["lr"]` is a dict per epoch when the groups are tagged, so the decay
# of each node is on record without a callback of your own. `plot_training`
# reads both that and the freeze epochs.

# %%
plot_training(flow, frozen=plateau.frozen)
plt.show()

# %% [markdown]
# ## 7. Writing your own
#
# A bare callable in `callbacks=` is an `on_epoch_end` hook, called as
# `cb(flow, epoch, optimizer)`. Return `True` to stop the fit. That is enough
# to carry any torch scheduler, because `fit` has already put this epoch's
# validation score in `history["val"]`.
#
# Subclass `Callback` instead when you need the other two hooks:
# `on_fit_begin` for state that must reset between fits, and `on_fit_end`,
# which runs before the varying-coefficient re-centering so restored weights
# still take part in it.


# %%
class GlobalPlateau(Callback):
    """One torch ReduceLROnPlateau over all nodes, driven by fit's own score."""

    def __init__(self, factor=0.3, patience=10):
        self.factor, self.patience = factor, patience
        self.scheduler = None

    def on_fit_begin(self, flow, optimizer):
        """Build the scheduler here: the optimizer exists only once fit runs."""
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )

    def on_epoch_end(self, flow, epoch, optimizer):
        """Step the scheduler, and stop once the rate has bottomed out."""
        self.scheduler.step(sum(flow.history["val"][-1].values()))
        return optimizer.param_groups[0]["lr"] < 1e-5


flow = build()
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
t0 = time.perf_counter()
flow.fit(
    train,
    epochs=CEILING,
    batch_size=256,
    validation_data=val,
    optimizer=optimizer,
    callbacks=GlobalPlateau(),
)
record("GlobalPlateau", flow, time.perf_counter() - t0)

final_lr = optimizer.param_groups[0]["lr"]
print(f"    rate went 1.0e-02 -> {final_lr:.1e}")
assert final_lr < 1e-2, "the scheduler never decayed the rate"

# %% [markdown]
# A one-line callable is often enough. This one records a coefficient after
# every epoch, which is how the paper replications trace convergence:
#
# ```python
# trace = []
# flow.fit(train, epochs=100, validation_data=val,
#          callbacks=lambda f, epoch, opt: trace.append(f.nll(val)["x3"]))
# ```
#
# A callable of the wrong shape is refused before the first epoch, so a
# mis-registered callback cannot waste a long run.

# %%
try:
    build().fit(train, epochs=5, batch_size=256, callbacks=lambda flow: None)
except TypeError as err:
    print("wrong arity refused up front:\n   ", err)
else:
    raise AssertionError("a one-argument callable should be refused")

# %% [markdown]
# ## 8. The scoreboard
#
# One workload, one seed, six recipes. The numbers show mechanics on 900 rows,
# not a benchmark; `docs/training-speed.md` has the measured comparison.

# %%
board = pd.DataFrame(scoreboard).set_index("strategy")
print(board.to_string(float_format=lambda v: f"{v:.4f}"))

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
# The decision table lives in [`docs/fitting.md`](../docs/fitting.md) and the
# measured time-to-target in
# [`docs/training-speed.md`](../docs/training-speed.md). Two rules are worth
# repeating here, because both are easy to get wrong:
#
# - An all-`LS` model wants the final weights, not the best ones. That is what
#   makes it match `statsmodels` and R exactly. Use `fit_classical` for it
#   instead, which is deterministic and faster.
# - A flexible model with confounded observational data can overfit at the
#   maximum-likelihood point and lose the causal effect. There
#   `EarlyStopping` is not a speed choice but a correctness one.
