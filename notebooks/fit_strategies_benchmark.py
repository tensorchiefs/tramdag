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
# # Per-node or joint fit: a benchmark
#
# The joint likelihood of a TRAM-DAG is a sum of per-node terms with disjoint
# parameters. So `fit` (one optimizer and one loop per node) and
# `fit_jointly` (one optimizer and one loop for all nodes) have the same
# optimum. They differ in the path to it: the batches, the step sizes, and
# what a schedule or a stop rule reads.
#
# This notebook runs both on one synthetic process with a known truth. It
# compares six arms: per node in series, per node in processes (`n_jobs`),
# and jointly, each with a constant learning rate and with a plateau
# schedule. The per-node plateau reads the node's own validation NLL; the
# joint plateau reads the summed one, as the reference implementation does
# (deviation D5 in [`docs/paper-replication.md`](../docs/paper-replication.md)).
#
# Each arm reports wall-clock seconds, the epoch at which the summed
# validation NLL first comes within `EPS` of its best, the best summed
# validation NLL, and the error of one intervention against the truth.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import CausalFlowDAG, ContinuousNode, I
from tramdag.callbacks import Callback, EarlyStopping

plt.rcParams["figure.dpi"] = 110
torch.set_num_threads(1)


def claim(text, value, bound, fmt="{:.3f}"):
    """Assert a computed quantity stays inside its bound, and print both."""
    ok = "ok  " if value <= bound else "FAIL"
    print(f"{ok}{text}: {fmt.format(value)} <= {fmt.format(bound)}")
    assert value <= bound, f"{text}: {fmt.format(value)} exceeds {fmt.format(bound)}"


# %% [markdown]
# ## 1. The process
#
# The bimodal VACA triangle of the Colab demo: $x_1$ is a two-component
# mixture, $x_2 = -x_1 + \varepsilon_2$, $x_3 = x_1 + 0.25\,x_2 +
# \varepsilon_3$. Under $do(x_2 = a)$, $x_3 = x_1 + 0.25\,a + \varepsilon_3$,
# so the true interventional distribution is a fresh draw of the process with
# $x_2$ clamped.


# %%
def sample_dgp(n, seed, do=None):
    """Draw n rows of the VACA triangle, optionally with x2 clamped."""
    rng = np.random.default_rng(seed)
    mix, a, b, e2, e3 = rng.uniform(size=n), *rng.normal(size=(4, n))
    x1 = np.where(mix < 0.5, -2.0 + np.sqrt(1.5) * a, 1.5 + b)
    x2 = np.full(n, float(do["x2"])) if do else -x1 + e2
    x3 = x1 + 0.25 * x2 + e3
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def w1(a, b):
    """Give the 1-Wasserstein distance of two samples of equal size."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


N_TRAIN, N_VAL = 8_000, 2_000
df = sample_dgp(N_TRAIN + N_VAL, seed=0)
train, val = df.iloc[:N_TRAIN], df.iloc[N_TRAIN:]
DO = {"x2": -1.0}
truth = sample_dgp(20_000, seed=1, do=DO)["x3"].to_numpy()

SPEC = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(I("x1")),
    "x3": ContinuousNode(I("x1", "x2")),
}

# %% [markdown]
# ## 2. The arms
#
# Every arm gets the same budget: `EPOCHS` passes, batch size 512, Adam at
# 1e-2, and `EarlyStopping()` without patience, so each fit runs its full
# budget and keeps its best weights. Torch runs on one thread, so the serial
# arms and each `n_jobs` worker use one core each. The plateau arms add torch's
# `ReduceLROnPlateau` through a callback. The callback reads the validation
# NLL of whatever it hooks: one node under `fit`, the flow under
# `fit_jointly`, where the value is the sum over the nodes.

# %%
EPOCHS = 100
SEEDS = [0, 1, 2]
EPS = 0.01
KW = dict(epochs=EPOCHS, batch_size=512, learning_rate=1e-2, validation_data=val)


class Plateau(Callback):
    """Cut the learning rate tenfold once the validation NLL stalls."""

    def on_fit_begin(self, target, optimizer):
        """Attach a fresh schedule to this fit's optimizer."""
        self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10
        )

    def on_epoch_end(self, target, epoch, optimizer):
        """Step the schedule on the node's or the flow's validation NLL."""
        last = target.history["val"][-1]
        self.schedule.step(sum(last.values()) if isinstance(last, dict) else last)


def callbacks(plateau):
    """Give the callbacks of one fit: a fresh plateau when asked, plus EarlyStopping."""
    return [Plateau(), EarlyStopping()] if plateau else [EarlyStopping()]


ARMS = {
    "per node": lambda f, p, s: f.fit(
        train, seed=s, callbacks=lambda name: callbacks(p), **KW
    ),
    "per node, n_jobs=3": lambda f, p, s: f.fit(
        train, seed=s, n_jobs=3, callbacks=lambda name: callbacks(p), **KW
    ),
    "joint": lambda f, p, s: f.fit_jointly(train, seed=s, callbacks=callbacks(p), **KW),
}

# %% [markdown]
# ## 3. The runs
#
# Each arm runs once per seed. The seed sets the initial weights and the
# minibatch order. The summed validation curve sums the per-node curves at
# each epoch; every node runs the full budget, so the epochs line up.

# %%
rows, curves = [], {}
for name, run in ARMS.items():
    for plateau in (False, True):
        arm = name + (" + plateau" if plateau else "")
        for seed in SEEDS:
            flow = CausalFlowDAG(SPEC, seed=seed)
            start = time.perf_counter()
            run(flow, plateau, seed)
            seconds = time.perf_counter() - start
            curve = np.array([sum(v.values()) for v in flow.history["val"]])
            sample = flow.sample(len(truth), do=DO, seed=seed)["x3"].to_numpy()
            final_lr = min(lr[0] for lr in flow.history["lr"][-1].values())
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "seconds": seconds,
                    "epochs_to_eps": int(np.argmax(curve <= curve.min() + EPS)) + 1,
                    "best_val_nll": curve.min(),
                    "w1_do": w1(sample, truth),
                    "final_lr": final_lr,
                }
            )
            curves[arm, seed] = curve
        print(f"{arm:28s} done")

runs = pd.DataFrame(rows)
board = runs.groupby("arm", sort=False).agg(["mean", "std"])
board = board.drop(columns="seed")
print(board.to_string(float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## 4. Convergence
#
# The summed validation NLL per epoch, seed 0, one panel per schedule.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, plateau in zip(axes, (False, True), strict=True):
    for i, name in enumerate(ARMS):
        arm = name + (" + plateau" if plateau else "")
        ax.plot(np.arange(1, EPOCHS + 1), curves[arm, 0], color=f"C{i}", label=name)
    ax.set_title("plateau schedule" if plateau else "constant learning rate")
    ax.set_xlabel("epoch")
axes[0].set_ylabel("summed validation NLL")
best = runs["best_val_nll"].min()
axes[0].set_ylim(best - 0.02, best + 0.3)
axes[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. What the numbers say
#
# The checks below hold on any machine; the seconds do not, so the notebook
# prints them but does not assert them. The serial per-node arm and the joint
# arm run one process each; the `n_jobs` arm runs three.
#
# All arms reach the same best validation NLL within a small gap. The
# optimum is the same, so this is the expected result; a large gap would
# point to a bug in one path. The intervention error varies more from seed
# to seed than from arm to arm.

# %%
means = runs.groupby("arm", sort=False).mean()
claim(
    "spread of the mean best validation NLL over the arms",
    means["best_val_nll"].max() - means["best_val_nll"].min(),
    0.003,
)
spread = runs.groupby("arm")["w1_do"].std().max()
claim(
    "spread of the mean W1 error under do(x2=-1) over the arms, per seed std",
    (means["w1_do"].max() - means["w1_do"].min()) / spread,
    2.0,
)

# %% [markdown]
# Serial and forked per-node fits with the same seed train the same nodes on
# the same batches. On one thread their curves agree to the last bits.

# %%
claim(
    "largest gap of the serial and forked per-node curves",
    max(
        np.abs(curves["per node", s] - curves["per node, n_jobs=3", s]).max()
        for s in SEEDS
    ),
    1e-6,
    fmt="{:.2e}",
)

# %% [markdown]
# ## Which one, when
#
# [`docs/fitting.md`](../docs/fitting.md) holds the decision table.
