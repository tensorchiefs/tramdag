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
# # Individual treatment effects from observational data (all-CI TRAM-DAG)
#
# This notebook estimates **individual treatment effects (ITEs)** with an
# **S-learner TRAM-DAG** on a confounded *observational* cohort, and validates the
# recovered effects against the simulator's known per-individual ground truth.
#
# The DGP (`tramdag.simulations.ITEObservational`, ported from the master's
# thesis repo `mikekr97/MA_Mike`) is a 7-variable mediation SCM with a **binary treatment
# `Tr`** confounded by `X1, X2`:
#
# ```
# X1, X2, X3 ~ N(0, Σ)            (correlated, ρ=0.1)
# Tr ~ Bernoulli(sigmoid(0.5 - 0.5·X1 + 0.3·X2))      # confounded treatment
# X5 :  2.5·X5 = logit(U5) - 0.8·Tr                    # mediator
# X6 :  4·X6   = logit(U6) + 0.5·X5                    # mediator
# Y  :  h_y(Y) = logit(U7) - [1.5·Tr + X·β + (-0.9·X2 + 0.7·X3)·Tr]
# ```
#
# Treatment affects `Y` directly **and** through the mediators `X5 → X6 → Y`, and
# the `(X2, X3)·Tr` interaction makes the effect **heterogeneous** across
# individuals (scenario 1). The baseline `h_y(y) = tan(y/2)/0.2` is nonlinear. All
# latents are logistic, so the DGP lives *inside* the flow's family — what must be
# learned are the shapes and the interaction.
#
# **"All CI":** every child node's transformation is a **joint complex intercept**
# over its parents — `I(*parents)` — the most flexible TRAM-DAG, able to absorb the
# mediation and the treatment interactions.

# %%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tramdag import CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.simulations import ITEObservational

torch.manual_seed(0)
COLS = ["X1", "X2", "X3", "Tr", "X5", "X6", "Y"]

# %% [markdown]
# ## Data
#
# Train on 20k observational rows (no ground truth seen during fitting). Hold out
# a test set that *also* carries the simulator's aligned ground truth — per
# individual, `ITE_true = Y(Tr=1) − Y(Tr=0)` at the **shared** exogenous noise
# (exactly what Pearl abduction recovers).

# %%
gen = ITEObservational(seed=123, scenario=1)
train = gen.observational(20_000)
test = gen.with_truth(4000, seed_offset=50)
truth = gen.true_ate(mc_n=200_000)
print(f"train n={len(train)}  treated rate={train['Tr'].mean():.3f}")
print(f"true ATE (median latent) = {truth['ate_median']:+.3f}   "
      f"(observed latent) = {truth['ate_true']:+.3f}")
train.head()

# %% [markdown]
# ### Confounding: the naive contrast is biased
#
# Because sicker/healthier patients (via `X1, X2`) select into treatment, the raw
# observational difference in means is **not** the causal effect.

# %%
naive = test.loc[test.Tr == 1, "Y"].mean() - test.loc[test.Tr == 0, "Y"].mean()
print(f"naive  E[Y|Tr=1] - E[Y|Tr=0] = {naive:+.3f}   (confounded)")
print(f"true ATE                     = {truth['ate_true']:+.3f}")

# %% [markdown]
# ## The all-CI S-learner TRAM-DAG
#
# One joint complex intercept per child node. `to_matrix()` shows the edge types
# (all `CI`).

# %%
spec = {"X1": ContinuousNode(), "X2": ContinuousNode(), "X3": ContinuousNode(),
        "Tr": OrdinalNode(levels=2, terms=[I("X1", "X2")]),
        "X5": ContinuousNode(terms=[I("Tr")]),
        "X6": ContinuousNode(terms=[I("X5")]),
        "Y":  ContinuousNode(terms=[I("Tr", "X1", "X2", "X3", "X5", "X6")])}
flow = CausalFlowDAG(spec, seed=1)
print(flow.to_matrix())

# %% [markdown]
# Fit jointly by maximum likelihood (~3 min on CPU; the `plateau` schedule decays
# each node's learning rate off its own validation NLL).

# %%
flow.fit(train, epochs=600, learning_rate=1e-2, schedule="plateau",
         plateau_patience=25, verbose=100)

# %% [markdown]
# ## L1 sanity — does the fitted flow reproduce the observational distribution?

# %%
sim = flow.sample(len(train), seed=0)
summary = pd.DataFrame({"data": train[COLS].mean(), "flow": sim[COLS].mean(),
                        "data sd": train[COLS].std(), "flow sd": sim[COLS].std()})
print(summary.round(3))

# %% [markdown]
# ## L3 — individual treatment effects via abduction + do
#
# For each held-out patient: **abduct** the latent noise from the observed row
# (Pearl step 1), then push it through the flow twice under `do(Tr=1)` and
# `do(Tr=0)` (steps 2–3). The mediators `X5, X6` and the outcome `Y` are
# recomputed under each counterfactual treatment, so the predicted
# `ITE = Y_cf(Tr=1) − Y_cf(Tr=0)` includes the mediated path.

# %%
u = flow.abduct(test[COLS], seed=0)
y1 = flow.sample(do={"Tr": 1.0}, u=u)["Y"].to_numpy()
y0 = flow.sample(do={"Tr": 0.0}, u=u)["Y"].to_numpy()
ite_pred = y1 - y0
ite_true = test["ITE_true"].to_numpy()

print(f"ATE  predicted {ite_pred.mean():+.3f}   true {ite_true.mean():+.3f}")
print(f"ITE  corr(pred, true) = {np.corrcoef(ite_pred, ite_true)[0, 1]:.3f}   "
      f"MAE = {np.abs(ite_pred - ite_true).mean():.3f}")

# %% [markdown]
# The model recovers each patient's effect, not just the average. Left: predicted
# vs true ITE (the diagonal is perfect recovery). Right: the learned
# **heterogeneity** — the effect varies with `X2` (and `X3`, color), exactly the
# `(X2, X3)·Tr` interaction baked into the DGP.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
lim = [min(ite_true.min(), ite_pred.min()), max(ite_true.max(), ite_pred.max())]
axes[0].plot(lim, lim, color="0.6", lw=1, ls="--")
axes[0].scatter(ite_true, ite_pred, s=6, alpha=0.3, color="#1b9e77")
axes[0].set_xlabel("true ITE")
axes[0].set_ylabel("predicted ITE")
axes[0].set_title(f"individual effects (r={np.corrcoef(ite_pred, ite_true)[0,1]:.3f})")

sc = axes[1].scatter(test["X2"], ite_pred, c=test["X3"], s=8, alpha=0.6,
                     cmap="coolwarm")
axes[1].set_xlabel("X2")
axes[1].set_ylabel("predicted ITE")
axes[1].set_title("heterogeneity: effect modified by X2, X3")
fig.colorbar(sc, ax=axes[1], label="X3")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Prospective: the ITE *distribution* for a new, untreated patient
#
# The recovery above used each patient's **full** observed row (including the
# post-treatment `X5, X6, Y`) to *abduct* their latent noise — an L3 query that
# only works once a patient has been treated and observed. The realistic
# prospective question is different: a **new patient** arrives with only the
# baseline covariates `X1, X2, X3` and **no treatment yet**, so there is nothing
# downstream to abduct.
#
# What is identifiable is then not a single number but the **conditional
# distribution of the individual effect given the covariates**,
# `P(ITE | X1, X2, X3)` — the *prior-predictive* ITE. Monte-Carlo recipe: hold
# `X1, X2, X3` fixed, draw the post-treatment latents (`X5, X6, Y`) from their
# **prior** (standard logistic), and push each draw through both `do(Tr=1)` and
# `do(Tr=0)` **using the same latents** — a within-patient contrast (common random
# numbers; *independent* draws per arm would instead give the variance of two
# unrelated patients). The mean of the resulting cloud is the estimated **CATE**;
# its spread is the patient's effect **uncertainty**.
#
# That spread (i) is **aleatoric** — it comes from the patient's unrealized noise,
# and shrinks only once they are treated and observed (then you can abduct), *not*
# with more training data; and (ii) is **conditional on the model** — it omits
# estimation error and its width encodes the flow's counterfactual coupling. Here
# we can *validate* it against the simulator's true ITE distribution.

# %%
M = 4000   # latent draws per patient


def model_ite_dist(flow, x1, x2, x3, m=M, seed=0):
    """Prior-predictive ITE samples from the fitted flow, using ONLY X1,X2,X3.
    Post-treatment latents are drawn from the prior; the same draw is pushed
    through both arms (shared latents => a within-patient contrast)."""
    rng = np.random.default_rng(seed)
    u = pd.DataFrame({"X1": 0.0, "X2": 0.0, "X3": 0.0, "Tr": 0.0,  # overwritten by do
                      "X5": rng.logistic(size=m), "X6": rng.logistic(size=m),
                      "Y": rng.logistic(size=m)})
    fixed = {"X1": x1, "X2": x2, "X3": x3}
    y1 = flow.sample(do={**fixed, "Tr": 1.0}, u=u)["Y"].to_numpy()
    y0 = flow.sample(do={**fixed, "Tr": 0.0}, u=u)["Y"].to_numpy()
    return y1 - y0                                      # shared latents per row


def true_ite_dist(gen, x1, x2, x3, m=M, seed=1):
    """The DGP's true ITE samples for a patient with these covariates."""
    rng = np.random.default_rng(seed)
    lat = {"x123": np.tile([x1, x2, x3], (m, 1)), "u_tr": rng.uniform(size=m),
           "u5": rng.uniform(size=m), "u6": rng.uniform(size=m), "u7": rng.uniform(size=m)}
    y1 = gen.simulate(latents=lat, do={"Tr": 1.0})["Y"].to_numpy()
    y0 = gen.simulate(latents=lat, do={"Tr": 0.0})["Y"].to_numpy()
    return y1 - y0


# five new patients — we use ONLY their baseline covariates X1, X2, X3.
# All three are conditioned on: X2, X3 are the explicit effect-modifiers (the
# (X2,X3)·Tr interaction), while X1 has no interaction and cancels on the *logit*
# scale — but it still shifts the patient's operating point on the saturating
# baseline h_y, so it modulates the effect *magnitude* through the nonlinearity.
patients = gen.observational(5, seed_offset=2024)[["X1", "X2", "X3"]]
print(f"{'patient':>7} {'X1':>6} {'X2':>6} {'X3':>6}  {'CATE(model)':>12} "
      f"{'CATE(true)':>11}  {'80% PI (model)':>18}")
dists = []
for i, (x1, x2, x3) in enumerate(patients.itertuples(index=False)):
    md = model_ite_dist(flow, x1, x2, x3, seed=i)
    td = true_ite_dist(gen, x1, x2, x3, seed=100 + i)
    dists.append((md, td))
    lo, hi = np.percentile(md, [10, 90])
    print(f"{i:>7} {x1:>+6.2f} {x2:>+6.2f} {x3:>+6.2f}  {md.mean():>+12.3f} "
          f"{td.mean():>+11.3f}   [{lo:+.2f}, {hi:+.2f}]")

# %% [markdown]
# Each panel is one new patient's **ITE distribution** from the fitted model
# (filled), with the simulator's **true** distribution overlaid (outline) and both
# means marked. The distributions are wide — a single patient's effect is
# genuinely uncertain from baseline data alone — yet the model tracks both the
# **location** (CATE, shifting with `X2, X3` — the interaction modifiers — and,
# more weakly, with `X1` via the operating point on the saturating baseline) and
# the **spread** of the truth. (`0` marks "no effect".)

# %%
fig, axes = plt.subplots(1, len(patients), figsize=(3.0 * len(patients), 3.4),
                         sharex=True, sharey=True)
for ax, (i, row), (md, td) in zip(axes, patients.iterrows(), dists):
    bins = np.linspace(min(md.min(), td.min()), max(md.max(), td.max()), 40)
    ax.hist(md, bins=bins, density=True, color="#1b9e77", alpha=0.55, label="model")
    ax.hist(td, bins=bins, density=True, histtype="step", color="0.2", lw=1.5,
            label="true")
    ax.axvline(md.mean(), color="#1b9e77", lw=1.5)
    ax.axvline(td.mean(), color="0.2", lw=1.0, ls="--")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_title(f"patient {i}\nX1={row.X1:+.2f}, X2={row.X2:+.2f}, X3={row.X3:+.2f}",
                 fontsize=8)
    ax.set_xlabel("ITE")
axes[0].set_ylabel("density")
axes[0].legend(fontsize=8)
fig.suptitle("Prior-predictive ITE per new patient (only X1, X2, X3 known) "
             "vs DGP truth", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Takeaway
#
# - The **naive** observational contrast is biased by confounding (treatment
#   depends on `X1, X2`).
# - A single **all-CI TRAM-DAG**, fitted once on observational data, answers the
#   L3 counterfactual query *per individual* via abduction + `do` — recovering the
#   true ITEs (here r ≈ 0.99) and the ATE, **including the mediated `X5 → X6 → Y`
#   path and the treatment-effect heterogeneity**.
# - The DGP ships with the package (`tramdag.simulations.ITEObservational`, four
#   scenarios) with per-individual ground truth, so this is a self-contained,
#   reproducible ITE benchmark. Switch `scenario=` to 2 (main effect only),
#   3 (interaction only) or 4 (null effect) to vary the truth.
# - For a **new, untreated** patient (only `X1, X2, X3` known) the sharp individual
#   counterfactual is not available — but the model still returns the
#   *prior-predictive* ITE **distribution** `P(ITE | X1, X2, X3)`: its mean is the
#   CATE driving the treat/don't-treat decision, its spread the (aleatoric)
#   per-patient uncertainty, here validated against the DGP's true ITE spread.
