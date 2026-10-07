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
# # Additive vs joint complex intercept: reading per-parent effects
#
# A complex intercept groups its parents jointly, `CI("x1", "x2")`, or
# additively, `CI("x1", "x2", allow_interaction=False)`.
# [`docs/model.md`](../docs/model.md) states the rule, and
# [`docs/interpretation.md`](../docs/interpretation.md) why the additive form
# needs `flow.intercept_contributions(data, node)` to be read.
#
# This notebook fits both models and shows the difference in interpretation.
# The data have an interaction, so the joint model fits them better. Only the
# additive model splits into per-parent parts, and `intercept_contributions`
# gives these parts.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tramdag import CI, CausalFlowDAG, ContinuousNode
from tramdag.callbacks import EarlyStopping

# %% [markdown]
# ## The data
#
# `x1, x2 ~ U(-2, 2)` and
# $x_3 = x_1 + x_2 + \gamma\,x_1 x_2 + \tfrac12\,\varepsilon$,
# $\varepsilon \sim \mathrm{Logistic}(0,1)$. The term $\gamma\,x_1 x_2$ is an
# interaction of the two parents.

# %%
GAMMA = 0.6


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    eps = rng.logistic(0, 1, n)
    x3 = x1 + x2 + GAMMA * x1 * x2 + 0.5 * eps
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


train = simulate(4000, seed=1)
val = simulate(1000, seed=2)
train.head()

# %% [markdown]
# ## Fit both models
#
# The data and the node are the same. Only the grouping of the two intercept
# parents is different.


# %%
def make_flow(joint: bool):
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode(
            CI("x1", "x2") if joint else CI("x1", "x2", allow_interaction=False)
        ),
    }
    return CausalFlowDAG(spec, seed=0)


flow_joint = make_flow(joint=True)
flow_add = make_flow(joint=False)

for f in (flow_joint, flow_add):
    f.fit(
        train,
        epochs=1200,
        learning_rate=1e-2,
        validation_data=val,
        callbacks=EarlyStopping(),  # keep the best-validation weights
    )

# The validation NLL is per row. The joint model can represent the interaction,
# so it has the lower validation NLL.
nll_joint = sum(flow_joint.nll(val).values())
nll_add = sum(flow_add.nll(val).values())
print(f"val NLL  joint   : {nll_joint:.4f}")
print(f"val NLL  additive: {nll_add:.4f}")
print(
    f"gap over the {len(val)} validation rows: {len(val) * (nll_add - nll_joint):.1f}"
)
assert nll_joint < nll_add, "the joint model should fit the interaction better"

# %% [markdown]
# ## `intercept_contributions`: the centered decomposition
#
# It returns the mean-centered (sum-to-zero) contribution of each parent's
# network to the transform parameters `theta`. The removed means collect in
# `baseline`. The centering is over the rows of the `df` that you pass.

# %%
res_add = flow_add.intercept_contributions(train, "x3")
res_joint = flow_joint.intercept_contributions(train, "x3")

print("additive terms :", list(res_add["contributions"]))  # 'x1', 'x2': separable
print("joint terms    :", list(res_joint["contributions"]))  # 'x1+x2': inseparable

# the centering: every column mean is about 0
col_means = {
    k: float(np.abs(v.mean(0)).max()) for k, v in res_add["contributions"].items()
}
print("per-term column means (≈0)   :", col_means)
assert max(col_means.values()) < 1e-5

# %% [markdown]
# ## The structural difference: separable curves vs an entangled cloud
#
# Pick the transform coefficient that varies most across the data and plot each
# term's centered contribution to it against a parent value.
#
# - **Additive** (`net(x1)` depends only on `x1`): its contribution to any
#   coefficient is a deterministic 1-D function of `x1`. The points collapse to
#   a **clean curve**, and the same applies to `x2`. These curves *are* the
#   per-parent partial effects, because the structure is separable.
# - **Joint** (one network over both parents): the single `x1+x2` component
#   depends on *both*. Plotted against `x1` alone it is a **cloud**, and the
#   hidden `x2` (color) sets its height. There is nothing to read off per
#   parent. The joint panel picks its own most-varying coefficient and has its
#   own y-axis.

# %%
c1 = res_add["contributions"]["x1"]
c2 = res_add["contributions"]["x2"]
cj = res_joint["contributions"]["x1+x2"]
k = int((c1.var(0) + c2.var(0)).argmax())  # most-varying coefficient, additive
kj = int(cj.var(0).argmax())  # most-varying coefficient, joint
x1v, x2v = train["x1"].values, train["x2"].values

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
o1, o2 = np.argsort(x1v), np.argsort(x2v)
axes[0].plot(x1v[o1], c1[o1, k], color="#1b9e77", lw=2.5, label="net(x1)  vs x1")
axes[0].plot(x2v[o2], c2[o2, k], color="#d95f02", lw=2.5, label="net(x2)  vs x2")
axes[0].axhline(0, color="0.7", lw=0.8)
axes[0].set_title("additive: separable per-parent curves")
axes[0].set_xlabel("parent value")
axes[0].set_ylabel(f"centered contribution to theta[{k}]")
axes[0].legend()

sc = axes[1].scatter(x1v, cj[:, kj], c=x2v, s=8, alpha=0.6, cmap="coolwarm")
axes[1].axhline(0, color="0.7", lw=0.8)
axes[1].set_title("joint: against x1 a cloud (height set by x2)")
axes[1].set_xlabel("x1")
axes[1].set_ylabel(f"centered contribution to theta[{kj}]")
fig.colorbar(sc, ax=axes[1], label="x2")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Takeaway
#
# | you want… | use | what you get |
# |---|---|---|
# | a per-parent partial-effect plot ("what does `x1` do?") | **additive** `CI("x1", "x2", allow_interaction=False)` | `intercept_contributions` → mean-centered, **separable** components |
# | interactions between parents in the transform | **joint** `CI("x1", "x2")` | one entangled network: flexible, **not** separable |
#
# The choice is between fit and interpretation. Here the joint model has the
# lower validation NLL, and only the additive model is readable per parent.
#
# *Caveat:* the contributions live in the **unconstrained** parameter space of
# the transform. The additive terms are summed there, before the monotonicity
# constraint. Thus they are partial effects on the parameters. In general,
# they are not an additive split of the curve `h` itself.
