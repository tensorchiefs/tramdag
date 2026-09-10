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
# # TRAM-DAG: one causal model, all three rungs
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
#
# A TRAM-DAG ([Sick & Dürr, CLeaR 2025](https://arxiv.org/abs/2503.16206)) is
# one normalizing flow wired like the adjacency matrix of a causal DAG. Fit it
# once on observational data. Then answer all three rungs of Pearl's ladder
# with the same fitted model:
#
# | rung | query | call |
# |---|---|---|
# | L1 association | $p(x)$, sampling | `flow.log_prob(df)`, `flow.sample(n)` |
# | L2 intervention | $p(x \mid do(x_j{=}a))$ | `flow.sample(n, do={...})` |
# | L3 counterfactual | what would $x_i$ have been, had $x_j$ been $a$? | `u = flow.abduct(df)`, then `flow.sample(do={...}, u=u)` |
#
# This notebook uses the first benchmark of the paper. It is a three-variable
# process with a bimodal source, and its ground truth is known analytically.
# Every claim below is therefore a computed check, not a statement. A failed
# check stops the notebook.
#
# The model theory is in [`docs/model.md`](../docs/model.md). How to read a
# fitted model is in [`docs/interpretation.md`](../docs/interpretation.md).
#
# The notebook runs on CPU in well under five minutes.

# %%
import importlib.util
import subprocess
import sys
import time

if importlib.util.find_spec("tramdag") is None:  # Colab: install from PyPI
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "tramdag[plots]"]
    )
    # to track active development instead of the latest release, use:
    #   pip install "git+https://github.com/tensorchiefs/tramdag.git@main"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, I, plot_dag
from tramdag.callbacks import PerNodePlateau, per_node_adam
from tramdag.plots import plot_marginals, plot_training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["figure.dpi"] = 110
t_total = time.perf_counter()
print(f"torch {torch.__version__}  device: {DEVICE}")

# %% [markdown]
# ## 1. The benchmark process
#
# The process comes from Sánchez-Martín et al. 2022, App. E.1, and from App.
# C.1 of the TRAM-DAG paper:
#
# $$
# \begin{aligned}
# x_1 &\sim \tfrac12\,\mathcal N(-2,\,1.5) + \tfrac12\,\mathcal N(1.5,\,1)
#       &&\text{(bimodal source)}\\
# x_2 &= -x_1 + \mathcal N(0,1)\\
# x_3 &= x_1 + 0.25\,x_2 + \mathcal N(0,1)
# \end{aligned}
# $$
#
# The DAG is $x_1 \to x_2$, $x_1 \to x_3$, $x_2 \to x_3$. The source is
# bimodal, which is the part that a location-scale model cannot fit. Section 6
# measures that.

# %%
# The process is written out here, so the notebook needs nothing but tramdag.
# draw_latents and simulate stay separate on purpose. Keeping the noise lets
# section 5 intervene on the same individuals, which is what makes an
# individual counterfactual checkable.


def draw_latents(n, rng):
    """Draw every noise variable of the process, n rows each."""
    return {
        "x1_mix": rng.uniform(size=n),  # which mixture component
        "x1_a": rng.normal(size=n),  # the N(-2, 1.5) branch
        "x1_b": rng.normal(size=n),  # the N(1.5, 1) branch
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    }


def simulate(latents, do=None):
    """Run the process forward. A variable named in `do` is clamped instead."""
    do = do or {}
    n = len(latents["x2"])

    if "x1" in do:
        x1 = np.full(n, float(do["x1"]))
    else:
        x1 = np.where(
            latents["x1_mix"] < 0.5,
            -2.0 + np.sqrt(1.5) * latents["x1_a"],
            1.5 + latents["x1_b"],
        )

    x2 = np.full(n, float(do["x2"])) if "x2" in do else -x1 + latents["x2"]
    x3 = x1 + 0.25 * x2 + latents["x3"]
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def sample_dgp(n, seed, do=None):
    """Draw n fresh rows, optionally under an intervention."""
    return simulate(draw_latents(n, np.random.default_rng(seed)), do)


N = 20_000
df = sample_dgp(N, seed=43)
train, val = df.iloc[: N - 2_000], df.iloc[N - 2_000 :]

fig, ax = plt.subplots(figsize=(5.5, 3))
ax.hist(df["x1"], bins=80, density=True, alpha=0.7)
ax.set_title(f"the source $x_1$ has two modes ({N:,} rows)")
ax.set_xlabel("$x_1$")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The spec is the DAG, and one fit
#
# A spec is one entry per node. Each entry names the terms through which the
# parents enter. `I(...)` puts the parents in control of the transform
# parameters, which is the most flexible term. Section 7 writes the same DAG
# with interpretable terms instead.
#
# The joint negative log-likelihood decomposes per node, so one Adam fits
# every node at once. `fit` is a plain loop. `validation_data=` adds a
# per-epoch validation score in `flow.history["val"]`, and `verbose=` prints
# progress. The strategy attaches through `callbacks=`. Here `per_node_adam`
# gives each node its own parameter group, and `PerNodePlateau` lowers that
# node's rate when its own validation score stops improving. It freezes the
# node once the rate is low and flat. The fit ends when the last node freezes,
# so `epochs=200` is a ceiling and not a budget.
#
# [`notebooks/training_strategies.py`](training_strategies.py) works through
# the other recipes.

# %%
spec = {
    "x1": ContinuousNode(),  # the source has no parents
    "x2": ContinuousNode(I("x1")),
    "x3": ContinuousNode(I("x1", "x2")),
}
plot_dag(spec)
plt.show()

# %%
torch.manual_seed(0)
flow = CausalFlowDAG(spec, device=DEVICE)
sched = PerNodePlateau(patience=10, freeze=40)

t0 = time.perf_counter()
flow.fit(
    train,
    epochs=200,
    batch_size=2048,
    validation_data=val,
    verbose=50,
    optimizer=per_node_adam(flow, lr=1e-1),
    callbacks=sched,
)
t_fit = time.perf_counter() - t0
epochs_used = len(flow.history["val"])
print(f"\nfitted on {DEVICE} in {t_fit:.1f}s, {epochs_used} epochs")
print(f"each node froze at epoch: {dict(sorted(sched.frozen.items()))}")

# The ceiling must not bind. If it does, the plateau rule never finished and
# the numbers below describe an unconverged fit.
assert epochs_used < 200, (
    "the fit used all 200 epochs, so no node froze and the plateau rule did "
    "not self-stop; raise the ceiling before trusting anything below"
)

# %%
plot_training(flow, frozen=sched.frozen)
plt.show()

# %% [markdown]
# ## 3. Rung 1: the observational fit
#
# Samples from the fitted flow must reproduce the observed joint distribution.
# The marginals are the picture. The correlation matrix is the check, because
# it also covers the dependence between variables, which a marginal plot does
# not show.

# %%
plot_marginals(flow, df, seed=1)
plt.show()

# %%
samp = flow.sample(len(df), seed=1)
corr_gap = float(np.abs(samp.corr().to_numpy() - df.corr().to_numpy()).max())
print(f"largest absolute difference between correlation matrices: {corr_gap:.4f}")
assert corr_gap < 0.05, f"the sampled dependence structure is off by {corr_gap:.4f}"

# %% [markdown]
# ## 4. Rung 2: interventions
#
# `do=` mutilates the graph. It clamps $x_2$, cuts the edge into it, and
# resamples everything downstream. Under the process,
# $x_3 \mid do(x_2{=}a) = x_1 + 0.25a + \mathcal N(0,1)$, so
# $\mathbb E[x_3] = -0.25 + 0.25a$ exactly. That analytic target is what makes
# an error visible.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
errors = []
print("E[x3 | do(x2=a)]     analytic    TRAM-DAG      error")
for ax, a in zip(axes, (-3.0, -1.0, 0.0), strict=True):
    truth = sample_dgp(len(df), seed=543, do={"x2": a})
    fitted = flow.sample(len(df), do={"x2": a}, seed=2)
    bins = np.linspace(truth["x3"].quantile(0.001), truth["x3"].quantile(0.999), 70)
    ax.hist(truth["x3"], bins=bins, density=True, alpha=0.5, label="process")
    ax.hist(
        fitted["x3"], bins=bins, density=True, histtype="step", lw=1.6, label="TRAM-DAG"
    )
    ax.set_title(f"$p(x_3 \\mid do(x_2={a:+.0f}))$")

    analytic = -0.25 + 0.25 * a
    got = float(fitted["x3"].mean())
    errors.append(abs(got - analytic))
    print(
        f"   a = {a:+.0f}:            {analytic:+.4f}     {got:+.4f}     {errors[-1]:.4f}"
    )
axes[0].legend()
fig.tight_layout()
plt.show()

# The bound is about three times the largest error measured while writing this
# notebook (0.048), which leaves room for another machine and another draw.
assert max(errors) < 0.15, f"interventional mean off by {max(errors):.4f}"

# %% [markdown]
# ## 5. Rung 3: individual counterfactuals
#
# Take 1,000 held-out individuals. Abduction inverts the flow and recovers the
# latent noise of each one. That noise is everything about the individual that
# the model does not attribute to the parents. Pushing it back through the
# flow must return the observed row, which is the first check.
#
# Action and prediction then rerun each individual under $do(x_1 = 0)$ with
# the same noise. The process keeps its own noise, so the true individual
# counterfactual is known and the flow can be scored per individual. Real data
# never permits this check, because only one of the two outcomes exists.

# %%
lat = draw_latents(1_000, np.random.default_rng(7))
factual = simulate(lat)  # the same noise on both sides,
cf_true = simulate(lat, do={"x1": 0.0})  # so these are true counterfactuals

u = flow.abduct(factual)
recon_err = float(np.abs(flow.sample(u=u).to_numpy() - factual.to_numpy()).max())
print(f"abduction then reconstruction: max |error| = {recon_err:.2e}")
# This inverts a monotone transform in float32, so the bound is a precision
# bound rather than a statistical one.
assert recon_err < 1e-4, f"abduction did not round-trip: {recon_err:.2e}"

cf_flow = flow.sample(do={"x1": 0.0}, u=u)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, c in zip(axes, ["x2", "x3"], strict=True):
    r = float(np.corrcoef(cf_true[c], cf_flow[c])[0, 1])
    ax.scatter(cf_true[c], cf_flow[c], s=4, alpha=0.4)
    lims = [cf_true[c].min(), cf_true[c].max()]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_title(f"${c[0]}_{c[1]}$ under $do(x_1{{=}}0)$   (r = {r:.4f})")
    ax.set_xlabel("true, shared noise")
    ax.set_ylabel("TRAM-DAG")
    assert r > 0.99, f"counterfactual {c} correlates only {r:.4f} with the truth"
fig.suptitle("L3: individual counterfactuals, scored one unit at a time")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Why the Bernstein transform
#
# Every continuous node owns a monotone transform, and that transform carries
# the distributional flexibility. One argument switches it: `"bernstein"` is
# the default, `"spline"` is a monotone rational-quadratic spline, and
# `"affine"` is location-scale only, which makes each node-conditional a
# logistic distribution.
#
# The affine model is the informative comparison here, because the source is
# bimodal and a location-scale transform cannot produce two modes. A smaller
# subsample is enough to show it.
#
# The spline result is a separate story about how zuko extrapolates outside
# its domain. [`docs/zuko-upstream.md`](../docs/zuko-upstream.md) covers it,
# and this notebook does not spend a third fit on it.

# %%
sub = train.iloc[:5_000]
affine_spec = {
    "x1": ContinuousNode([I(transform="affine")]),
    "x2": ContinuousNode(I("x1", transform="affine")),
    "x3": ContinuousNode(I("x1", "x2", transform="affine")),
}
torch.manual_seed(0)
affine = CausalFlowDAG(affine_spec, device=DEVICE)
affine.fit(sub, epochs=150, learning_rate=1e-2, batch_size=1024, validation_data=val)

nll_bernstein = sum(flow.nll(val).values())
nll_affine = sum(affine.nll(val).values())
print(f"held-out NLL, lower is better:  bernstein {nll_bernstein:.4f}")
print(f"                                affine    {nll_affine:.4f}")

# %%
bins = np.linspace(df["x1"].quantile(0.001), df["x1"].quantile(0.999), 70)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(df["x1"], bins=bins, density=True, alpha=0.4, color="gray", label="data")
for name, model, color in [("bernstein", flow, "C3"), ("affine", affine, "C2")]:
    ax.hist(
        model.sample(5_000, seed=3)["x1"],
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        color=color,
        label=name,
    )
ax.set_title("a location-scale transform cannot produce two modes")
ax.set_xlabel("$x_1$")
ax.legend()
fig.tight_layout()
plt.show()

# The measured gap was 0.128 nats. The bound keeps a clear margin.
assert nll_affine > nll_bernstein + 0.05, (
    f"the affine model scored {nll_affine:.4f} against {nll_bernstein:.4f}, "
    "so this comparison no longer shows what it claims"
)

# %% [markdown]
# ## 7. The same DAG, written to be interpretable
#
# Everything above used `I(...)`, which gives a node no single coefficient to
# read. That is the flexible end of the family. Writing an edge as
# `LS("parent")` instead puts one weight on that edge, and the weight is
# readable after training.
#
# An all-`LS` model is a classical transformation model per node, so
# `fit_classical` applies. It is a deterministic full-batch L-BFGS fit in
# float64, with no schedule and no minibatch noise.
#
# Two things are worth reading off the result. The flow adds its shift on the
# latent scale, so a parent that raises a child gets a negative weight. And
# the weights carry the latent scale, so their ratio is what compares against
# the process, not their size. Here $x_3 = x_1 + 0.25 x_2$, so the ratio
# should be near $1 / 0.25 = 4$.
#
# [`docs/interpretation.md`](../docs/interpretation.md) covers the ordinal
# case, where a weight is a log-odds ratio.

# %%
ls_spec = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(LS("x1")),
    "x3": ContinuousNode(LS("x1") + LS("x2")),
}
torch.manual_seed(0)
ls_flow = CausalFlowDAG(ls_spec, device=DEVICE)
report = ls_flow.fit_classical(sub)
coefs = ls_flow.ls_coefficients()

b1 = float(coefs["x3"]["x1"][0])
b2 = float(coefs["x3"]["x2"][0])
ratio = b1 / b2
nll_ls = sum(ls_flow.nll(val).values())
print(f"exact MLE in {report['seconds']:.1f}s\n")
print(f"x3 <- x1 : {b1:+.4f}")
print(f"x3 <- x2 : {b2:+.4f}")
print(f"ratio    : {ratio:.3f}   (the process has 1 / 0.25 = 4)")
print(f"\nheld-out NLL: all-LS {nll_ls:.4f}, flexible {nll_bernstein:.4f}")

# The bound is deliberately wide: this is one draw at one seed, and the
# measured deviation was 0.03.
assert abs(ratio - 4.0) < 0.25, f"the coefficient ratio came out {ratio:.3f}"

# %% [markdown]
# The all-`LS` model pays a little likelihood for being readable. The price is
# small here because the edges of this process really are linear, so the
# flexible terms have little extra structure to find. On a process with
# curved edges the gap is larger. `experiments/paper/` measures that case.

# %% [markdown]
# ## What this notebook checked
#
# | rung | call | checked against |
# |---|---|---|
# | L1 | `flow.sample(n)` | the correlation matrix of the data |
# | L2 | `flow.sample(n, do=...)` | the analytic mean $-0.25 + 0.25a$ |
# | L3 | `flow.abduct(df)`, then `flow.sample(do=..., u=u)` | the true per-unit counterfactual |
# | transform | `I(transform="affine")` | held-out likelihood against Bernstein |
# | interpretability | `LS(...)`, `fit_classical`, `ls_coefficients()` | the coefficient ratio of the process |
#
# Where to go next:
#
# - [`docs/model.md`](../docs/model.md), the model and its notation.
# - [`docs/interpretation.md`](../docs/interpretation.md), reading a fitted
#   model, including ordinal nodes and log-odds ratios.
# - [`notebooks/training_strategies.py`](training_strategies.py), the fitting
#   API and every shipped strategy.
# - [`notebooks/classical_fit_tram_dag.py`](classical_fit_tram_dag.py), the
#   agreement with `statsmodels` and R.
# - [`notebooks/varying_coefficients.py`](varying_coefficients.py), effects
#   that vary with a covariate.
# - [repo](https://github.com/tensorchiefs/tramdag) ·
#   [paper](https://arxiv.org/abs/2503.16206)

# %%
elapsed = time.perf_counter() - t_total
print(f"whole notebook: {elapsed:.0f}s")
assert elapsed < 300, f"the notebook took {elapsed:.0f}s, over its five-minute claim"
