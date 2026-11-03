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
# # Classical fitting of all-`ls` TRAM-DAGs (`fit_classical`)
#
# When **every edge of a TRAM-DAG is a linear shift (`LS`)**, each
# node-conditional is a classical model
#
#  * an ordered-logit / proportional-odds model for ordinal nodes
#
#  * a Colr model (R's `tram::Colr`) for continuous ones.
#
# For such a model `flow.fit_classical()` is the optimizer. It is a
# deterministic full-batch L-BFGS, and its report says when it reached the
# maximum-likelihood estimate ([`docs/fitting.md`](../docs/fitting.md)). This
# notebook checks the result against `statsmodels` and R.
#
# The R references below are constants from
# `notebooks/classical_fit_tram_dag.R`. To check them, run
# `Rscript notebooks/classical_fit_tram_dag.R` from the repository root. It
# needs `MASS` and `tram`.
#

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.miscmodels.ordinal_model import OrderedModel

from tramdag import LS, SI, CausalFlowDAG, ContinuousNode, Node, OrdinalNode
from tramdag.callbacks import EarlyStopping

HERE = [Path.cwd(), *Path.cwd().parents]
REPO = next(p for p in HERE if (p / "pyproject.toml").exists())
DATA = REPO / "experiments" / "data"
NB_DATA = REPO / "notebooks" / "data"


def claim(text, value, bound, fmt="{:.2e}"):
    """Assert a computed quantity stays inside its bound, and print both.

    Every agreement this notebook asserts goes through here, so no number is
    typed into a markdown cell where it can drift away from the code. The
    prose says what is checked. The bound is the only number in the source,
    and it leaves room for another machine.
    """
    ok = "ok  " if value <= bound else "FAIL"
    print(f"{ok}{text}: {fmt.format(value)} <= {fmt.format(bound)}")
    assert value <= bound, f"{text}: {fmt.format(value)} exceeds {fmt.format(bound)}"


# %% [markdown]
# ## 0. The zeroth example: logistic regression on `MASS::birthwt`
#
# A two-level `OrdinalNode` whose terms are all `LS` is logistic regression.
# The ordinal transform is
#
# $$P(Y \le 0 \mid x) = \sigma\!\left(\theta_0 - \textstyle\sum_p w_p x_p\right),$$
#
# and with two levels there is a single cutpoint $\theta_0$, so
#
# $$\operatorname{logit} P(Y = 1 \mid x) = -\theta_0 + \textstyle\sum_p w_p x_p .$$
#
# The shift weights are the logistic-regression coefficients. The intercept
# is minus the cutpoint, because an ordinal node subtracts its shift
# ([`docs/notation.md`](../docs/notation.md)). Section 2 uses the same model
# with $K-1$ cutpoints.
#
# The data is `birthwt` from MASS, the low-birth-weight study of Hosmer &
# Lemeshow. The outcome is `low`, and the predictors are `age`, `lwt` and
# `smoke` ([`notebooks/data/README.md`](data/README.md)). This is not a causal
# model.

# %% [markdown]
# ### The R you can copy-paste
#
# Paste this into an R session to get the reference fit:
#
# ```r
# library(MASS)
# m <- glm(low ~ age + lwt + smoke, data = birthwt, family = binomial)
# coef(m)
# logLik(m)
# ```
#
# which prints
#
# ```
# (Intercept)         age         lwt       smoke
#  1.36822527 -0.03899458 -0.01213854  0.67076374
# 'log Lik.' -111.4396765 (df=4)
# ```
#
# (R 4.2.3, MASS 7.3-58.2.) The cell below holds these numbers as `R_GLM`.

# %%
R_GLM = {
    "intercept": 1.3682252685,
    "age": -0.0389945827,
    "lwt": -0.0121385423,
    "smoke": 0.6707637407,
}
R_LOGLIK = -111.43967649

bw = pd.read_csv(NB_DATA / "birthwt.csv")
print(
    bw.head(3).to_string(index=False),
    f"\n\n{len(bw)} births, {bw['low'].sum()} of them low birth weight",
)

# %% [markdown]
# `low` is the outcome. `smoke` is a binary parent, so it is an ordinal node
# too. `age` and `lwt` are source nodes. The flow also models their
# marginals, which `glm` does not. They get the cheapest transform,
# `SI(transform="affine")`. The joint NLL decomposes per node, so the
# marginals cannot change the `low` node.

# %%
spec_bw = {
    "age": ContinuousNode(SI(transform="affine")),
    "lwt": ContinuousNode(SI(transform="affine")),
    "smoke": OrdinalNode(2),
    "low": OrdinalNode(2, LS("age") + LS("lwt") + LS("smoke")),
}

flow_bw = CausalFlowDAG(spec_bw, seed=0)
flow_bw.fit_classical(bw)

# %% [markdown]
# `age` and `lwt` are continuous parents, so they enter raw and their weights
# compare directly. `smoke` is ordinal, so it enters one-hot over both levels,
# $w_0$ and $w_1$. Only their difference is identified
# ([`docs/interpretation.md`](../docs/interpretation.md)), and it is the
# `smoke` coefficient of R. With $s \in \{0, 1\}$,
# $\mathbf{1}[s = 0] = 1 - s$, so the one-hot pair collapses to
#
# $$w_0\,\mathbf{1}[s = 0] + w_1\,\mathbf{1}[s = 1] \;=\; w_0 + (w_1 - w_0)\,s ,$$
#
# and the node reads
#
# $$\operatorname{logit} P(\texttt{low} = 1) \;=\; (-\theta_0 + w_0) \;+\; w_{\text{age}}\,\texttt{age} \;+\; w_{\text{lwt}}\,\texttt{lwt} \;+\; (w_1 - w_0)\,s .$$
#
# The first bracket is the `(Intercept)` of R. The cutpoints are
# $[-\infty, \theta_0, +\infty]$, so the code takes the finite one. The
# `statsmodels` formula is the R call above.

# %%
import statsmodels.formula.api as smf  # noqa: E402

from tramdag.transforms import ordinal_cutpoints  # noqa: E402

res_bw = smf.logit("low ~ age + lwt + smoke", data=bw).fit(disp=False)

w = flow_bw.ls_coefficients()["low"]
with torch.no_grad():
    theta_0 = float(ordinal_cutpoints(flow_bw.nodes["low"].intercept(1))[0, 1])

rows_bw = [
    (
        "intercept",
        -theta_0 + float(w["smoke"][0]),
        res_bw.params["Intercept"],
        R_GLM["intercept"],
    ),
    ("age", float(w["age"][0]), res_bw.params["age"], R_GLM["age"]),
    ("lwt", float(w["lwt"][0]), res_bw.params["lwt"], R_GLM["lwt"]),
    (
        "smoke",
        float(w["smoke"][1] - w["smoke"][0]),
        res_bw.params["smoke"],
        R_GLM["smoke"],
    ),
]
print(f"{'':<11}{'fit_classical':>14}{'sm.Logit':>10}{'R glm':>10}{'max|diff|':>11}")
spreads = []
for name, flow_value, sm_value, r_value in rows_bw:
    spreads.append(max(abs(flow_value - sm_value), abs(flow_value - r_value)))
    print(
        f"{name:<11}{flow_value:>14.4f}{sm_value:>10.4f}{r_value:>10.4f}"
        f"{spreads[-1]:>11.2e}"
    )
claim("max coefficient spread of the three fits", max(spreads), 6e-5)

# %% [markdown]
# L-BFGS on the flow, the Newton solver of `statsmodels` and the IRLS of R
# find the same maximum likelihood. `flow.pmf` gives the fitted probabilities
# of `glm` row by row. `nll()` gives the mean NLL per node; times $-n$ it is
# the log-likelihood.

# %%
ll_flow = -flow_bw.nll(bw)["low"] * len(bw)
p_flow = flow_bw.pmf(bw, node="low")[:, 1]
p_classical = res_bw.predict(bw).values
print(f"log-likelihood   flow {ll_flow:.6f}   R glm {R_LOGLIK:.6f}")
claim("|logLik_flow - logLik_glm|", abs(ll_flow - R_LOGLIK), 3e-6)
claim(
    f"max |P_flow(low=1) - P_glm(low=1)| over {len(bw)} rows",
    float(np.abs(p_flow - p_classical).max()),
    1e-5,
)

# %% [markdown]
# **The same regression without the DAG.** A lone `Node` drops the marginals
# of `age` and `lwt`. `Node.fit_classical` finds the same maximum likelihood
# as the flow, R and `statsmodels`.

# %%
low = Node(
    "low", spec_bw["low"], {"age": "continuous", "lwt": "continuous", "smoke": 2}
)
low.fit_classical(bw[["age", "lwt", "smoke", "low"]])
w_low = low.ls_coefficients()
ll_low = -low.nll(bw) * len(bw)
print(f"log-likelihood   node alone {ll_low:.6f}   R glm {R_LOGLIK:.6f}")
print(
    f"smoke   node alone {float(w_low['smoke'][1] - w_low['smoke'][0]):.4f}"
    f"   R glm {R_GLM['smoke']:.4f}"
)
assert abs(ll_low - R_LOGLIK) < 1e-4, "the lone node misses the glm likelihood"

# %% [markdown]
# ## 1. Continuous case
#
# `birthwt` also holds `bwt`, the birth weight in grams that `low` is cut
# from. With a flexible monotone transform and a logistic latent, the same
# three predictors give the Colr model of R:
#
# ```r
# tram::Colr(bwt ~ age + lwt + smoke, order = 21)
# ```
#
# `glm` cannot fit this model. The R references come from this call (R 4.2.3,
# tram 1.0.4).

# %%
R_COLR_BWT = {
    "age": -0.021971,
    "lwt": -0.010233,
    "smoke": 0.668698,
    "loglik": -1499.6678,
}

spec_bwt = {
    "age": ContinuousNode(SI(transform="affine")),
    "lwt": ContinuousNode(SI(transform="affine")),
    "smoke": OrdinalNode(2),
    "bwt": ContinuousNode(
        SI(transform_kwargs={"n_coeffs": 20}) + LS("age") + LS("lwt") + LS("smoke")
    ),
}

flow_bwt = CausalFlowDAG(spec_bwt, seed=0)
flow_bwt.fit_classical(bw)

w = flow_bwt.ls_coefficients()["bwt"]
fitted = {
    "age": w["age"][0],
    "lwt": w["lwt"][0],
    "smoke": w["smoke"][1] - w["smoke"][0],
    "loglik": -flow_bwt.nll(bw)["bwt"] * len(bw),
}

print(f"{'':<9}{'age':>9}{'lwt':>9}{'smoke':>9}{'logLik':>11}")
for name, r in [("flow", fitted), ("R Colr", R_COLR_BWT)]:
    print(
        f"{name:<9}{r['age']:>9.4f}{r['lwt']:>9.4f}{r['smoke']:>9.4f}"
        f"{r['loglik']:>11.2f}"
    )
diff = {k: abs(fitted[k] - R_COLR_BWT[k]) for k in fitted}
print(
    f"{'|diff|':<9}{diff['age']:>9.4f}{diff['lwt']:>9.4f}{diff['smoke']:>9.4f}"
    f"{diff['loglik']:>11.2f}"
)

# %% [markdown]
# The agreement is not exact, because the two libraries place the Bernstein
# basis differently. The flow puts it on the `range_q` quantiles with linear
# tails ([`docs/model.md`](../docs/model.md)). `Colr` uses the full range of
# the data. The two fits thus maximize slightly different likelihoods. The
# next cells compare the differences with the standard errors.

# %% [markdown]
# ### Standard errors and confidence intervals
#
# The sampling distribution of the maximum likelihood estimator is
#
# $$\hat{\theta} \;\overset{\cdot}{\sim}\; N(\theta, I^{-1})$$
#
# where $I$ is the Fisher information matrix. $\theta$ is unknown, so $I$ is
# evaluated at $\hat\theta$. The standard errors are on the diagonal of
# $I^{-1}$. For a contrast $c$, the standard error is $\sqrt{c^\top I^{-1} c}$.
#
# The **observed** Fisher information is the Hessian of the *negative*
# log-likelihood at the MLE,
#
# $$I_{ab} = \frac{\partial^{2}(-\ell)}{\partial\theta_a\,\partial\theta_b}\bigg|_{\hat\theta},$$
#
# which autograd gives: one backward pass per component gives a row of $I$.
#
# Here $I$ is singular, so $I^{-1}$ does not exist. The one-hot `smoke` pair
# and the Bernstein intercept have directions without curvature (see
# "weakly identified directions" in
# [`docs/fitting.md`](../docs/fitting.md#path-b-classical-optimization-with-fit_classical)).
# $I$ is symmetric, so it has an orthonormal eigendecomposition. The
# **pseudo-inverse** inverts only the directions that have curvature:
#
# $$I = \sum_{k=1}^{P} \lambda_k\, e_k e_k^{\top}
#   \qquad\Longrightarrow\qquad
#   I^{+} = \sum_{k\,:\ \lambda_k > \tau} \frac{1}{\lambda_k}\, e_k e_k^{\top},
#   \qquad \tau = 10^{-7}\,\lambda_{\max}.$$
#
# Terms with $\lambda_k \le \tau$ are dropped and not inverted. This is
# correct for a contrast that avoids the flat subspace. The `leak` column
# measures how far a contrast enters it:
#
# $$\operatorname{leak}(c) = \max_{k\,:\ \lambda_k \le \tau} \left| c^{\top} e_k \right| .$$
#
# Only interpret rows whose leak is close to zero. The spectrum has a flat
# block near zero, one weakly curved Bernstein direction below $\tau$, and
# the curved directions. The threshold drops the weak direction with the flat
# block. The cell below computes the standard errors again with a threshold
# between the two and compares. The Hessian runs in float64.

# %%
from scipy.stats import norm  # noqa: E402


def observed_information(flow, node, data):
    """Give one node's NLL Hessian, the gradient norm, and parameter offsets."""
    flow.double()
    names, params = zip(*flow.nodes[node].named_parameters(), strict=True)
    nd = flow.nodes[node]
    nll = -nd.row_log_prob(nd.tensorize(data)).sum()
    grad = torch.cat(
        [g.reshape(-1) for g in torch.autograd.grad(nll, params, create_graph=True)]
    )
    hess = torch.stack(
        [
            torch.cat(
                [
                    h.reshape(-1)
                    for h in torch.autograd.grad(grad[i], params, retain_graph=True)
                ]
            )
            for i in range(grad.numel())
        ]
    )
    offsets, at = {}, 0
    for name, p in zip(names, params, strict=True):
        offsets[name] = at
        at += p.numel()
    result = hess.detach().numpy(), float(grad.detach().norm()), offsets
    flow.float()
    return result


def spectrum_split(evals, tau):
    """Split the eigenvalues at or below ``tau`` into a flat and a weak block.

    The flat block is the directions near zero. The weak block is the
    directions above the widest log-gap among the eigenvalues at or below
    ``tau``. ``tau`` drops them too. The result gives the size of
    the flat block and its largest eigenvalue, and the smallest weak
    eigenvalue, so that a caller can place a second threshold between them.
    """
    low = np.sort(np.maximum(np.abs(evals[np.abs(evals) <= tau]), 1e-300))
    cut = int(np.argmax(np.diff(np.log(low)))) + 1 if low.size > 1 else low.size
    return {
        "n_flat": cut,
        "flat_max": float(low[cut - 1]) if cut else 0.0,
        "weak_min": float(low[cut]) if cut < low.size else 0.0,
    }


def ls_conf_int(flow, node, data, level=0.95, rcond=1e-7):
    """Give Wald intervals for the linear-shift coefficients of one node."""
    hess, grad_norm, offsets = observed_information(flow, node, data)
    evals, evecs = np.linalg.eigh(hess)
    tau = rcond * np.abs(evals).max()
    flat = evecs[:, np.abs(evals) <= tau]
    cov = np.linalg.pinv(hess, rcond=rcond)
    z = norm.ppf(0.5 + level / 2)

    rows = []
    for parent, w in flow.ls_coefficients()[node].items():
        c = np.zeros(hess.shape[0])
        key = next(n for n in offsets if n.startswith(f"shifts.{parent}"))
        if len(w) == 1:
            c[offsets[key]], estimate, term = 1.0, w[0], parent
        else:
            c[offsets[key]], c[offsets[key] + 1] = -1.0, 1.0
            estimate, term = w[1] - w[0], f"{parent} (1 vs 0)"
        se = float(np.sqrt(c @ cov @ c))
        rows.append(
            {
                "term": term,
                "estimate": estimate,
                "se": se,
                "lower": estimate - z * se,
                "upper": estimate + z * se,
                "leak": float(np.abs(c @ flat).max()) if flat.size else 0.0,
            }
        )
    diagnostics = {
        "n_params": hess.shape[0],
        "grad_norm": grad_norm,
        "lambda_max": float(np.abs(evals).max()),
        **spectrum_split(evals, tau),
    }
    return pd.DataFrame(rows).set_index("term"), diagnostics


# %%
table, diag = ls_conf_int(flow_bwt, "bwt", bw)
print(table.round(6).to_string())
print(
    f"\n{diag['n_params']} parameters, {diag['n_flat']} of them without curvature"
    f"   |grad| = {diag['grad_norm']:.1e}"
)

# %% [markdown]
# The reference standard errors come from the same `Colr` fit.

# %%
R_COLR_BWT_SE = {"age": 0.025305, "lwt": 0.004116, "smoke": 0.260131}

print()
print(
    f"{diag['n_flat']} flat directions, largest eigenvalue {diag['flat_max']:.1e}"
    f"   weak direction {diag['weak_min']:.1e}"
    f"   tau = {1e-7 * diag['lambda_max']:.1e}"
)
tau2 = np.sqrt(diag["flat_max"] * diag["weak_min"])
table2, _ = ls_conf_int(flow_bwt, "bwt", bw, rcond=tau2 / diag["lambda_max"])
claim(
    "max |SE(tau) - SE(tau2)|, tau2 between the flat block and the weak direction",
    float((table["se"] - table2["se"]).abs().max()),
    1e-3,
)
rows_se = [("age", "age"), ("lwt", "lwt"), ("smoke (1 vs 0)", "smoke")]
se_gap = max(abs(float(table.loc[t, "se"]) - R_COLR_BWT_SE[r]) for t, r in rows_se)
claim("max |SE_flow - SE_Colr|", se_gap, 1e-3)
claim(
    "max |coef_flow - coef_Colr| / SE",
    max(diff[r] / float(table.loc[t, "se"]) for t, r in rows_se),
    0.07,
)

# %% [markdown]
# The standard errors of all three parents agree with those of `Colr`, and
# the coefficient differences are a small fraction of one standard error.

# %% [markdown]
# ## 1b. A bimodal DAG: the demo data
#
# The data is the bimodal VACA triangle `x1 → x2 → x3 ← x1` of the
# [demo notebook](demo_tram_dag_colab.py). Each node of the all-`LS` model is
# a Colr model. The DGP noise is Gaussian and the latent is logistic, so the
# model is misspecified. `fit_classical` still finds its MLE.


# %%
if False:

    def sample_vaca(n, seed):
        """Draw n rows from the VACA bimodal triangle SCM."""
        rng = np.random.default_rng(seed)
        mix, a, b = rng.uniform(size=n), rng.normal(size=n), rng.normal(size=n)
        x1 = np.where(mix < 0.5, -2.0 + np.sqrt(1.5) * a, 1.5 + b)
        x2 = -x1 + rng.normal(size=n)
        x3 = x1 + 0.25 * x2 + rng.normal(size=n)
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    df = sample_vaca(1000, seed=42)
    df.to_csv(NB_DATA / "vaca.csv", index=False)

# %% [markdown]
# ### The R you can copy-paste
#
# `tram::Colr` fits the same model on the tracked `notebooks/data/vaca.csv`.
# The two libraries count the basis differently: `N_COEFFS = 20` below is
# `order = 21` in tram, because zuko adds two control points
# ([`docs/zuko-upstream.md`](../docs/zuko-upstream.md)).
#
# ```r
# library(tram)
# d <- read.csv("notebooks/data/vaca.csv")
#
# m2 <- Colr(x2 ~ x1, data = d, order = 21)
# coef(m2)     # -> x1 1.800042
# logLik(m2)   # -> -1401.213542
#
# m3 <- Colr(x3 ~ x1 + x2, data = d, order = 21)
# coef(m3)     # -> x1 -1.777289   x2 -0.455282
# logLik(m3)   # -> -1411.901958
# ```
#
# `Colr` reports the shift on the same side as the flow, so the signs
# match.

# %%
R_COLR = {
    "x2": {"coef": {"x1": 1.800042}, "loglik": -1401.213542},
    "x3": {"coef": {"x1": -1.777289, "x2": -0.455282}, "loglik": -1411.901958},
}

# %% [markdown]
# The reference values come from `classical_fit_tram_dag.R` (R 4.2.3,
# tram 1.0.4).

# %%
df = pd.read_csv(NB_DATA / "vaca.csv")
N_COEFFS = 20

spec_vaca = {
    "x1": ContinuousNode(SI(transform_kwargs={"n_coeffs": N_COEFFS})),
    "x2": ContinuousNode(SI(transform_kwargs={"n_coeffs": N_COEFFS}) + LS("x1")),
    "x3": ContinuousNode(
        SI(transform_kwargs={"n_coeffs": N_COEFFS}) + LS("x1") + LS("x2")
    ),
}

flow_c = CausalFlowDAG(spec_vaca, seed=0)
rep = flow_c.fit_classical(df)
print(f"\n{'':<12}{'fit_classical':>14}{'R Colr':>12}{'|diff|':>10}")
loglik = {k: -v * len(df) for k, v in flow_c.nll(df).items()}
for node, parents in flow_c.ls_coefficients().items():
    for p, w in parents.items():
        c, r = w[0], R_COLR[node]["coef"][p]
        print(f"{p + ' -> ' + node:<12}{c:>14.4f}{r:>12.4f}{abs(c - r):>10.4f}")
    c, r = loglik[node], R_COLR[node]["loglik"]
    print(f"{'logLik ' + node:<12}{c:>14.4f}{r:>12.4f}{abs(c - r):>10.4f}")

# %% [markdown]
# ### Why this one is not an exact-MLE check
#
# The coefficients agree less closely than in Sections 0 and 2, because of
# the basis placement of Section 1. The relative gap of the shift
# coefficients stays small.

# %%
gaps = [
    (abs(float(w[0]) - R_COLR[node]["coef"][parent]), abs(R_COLR[node]["coef"][parent]))
    for node in R_COLR
    for parent, w in flow_c.ls_coefficients()[node].items()
]
claim("max |coef_flow - coef_Colr|", max(g for g, _ in gaps), 0.01)
claim("max relative coefficient gap", max(g / r for g, r in gaps), 0.02)

# %% [markdown]
# ### The fitted conditional density
#
# `flow.density(df, node, grid)` evaluates $p(\text{node} = g \mid
# \mathrm{pa})$ at each grid value, in closed form from the transform and with
# no sampling. It is the continuous counterpart of `flow.pmf`.
#
# A density must integrate to one over a wide grid, which checks the
# log-determinant of the transform. Its mode must move as the linear shifts
# say.

# %%
grid = np.linspace(df["x3"].min() - 2, df["x3"].max() + 2, 601)
rows = df.iloc[:200]
dens = flow_c.density(rows, "x3", grid)
mass = np.trapezoid(dens, grid, axis=1)
claim("worst |integral of the density - 1|", float(np.abs(mass - 1.0).max()), 5e-3)

# %% [markdown]
# A continuous node adds the shift to $h(x)$, so a positive weight lowers the
# mode.

# %%
modes = grid[dens.argmax(axis=1)]
shift = -(
    float(flow_c.ls_coefficients()["x3"]["x1"][0]) * rows["x1"]
    + float(flow_c.ls_coefficients()["x3"]["x2"][0]) * rows["x2"]
)
corr = float(np.corrcoef(modes, shift)[0, 1])
print(f"corr(mode of the fitted density, the linear predictor) = {corr:.4f}")
assert corr > 0.95, f"the density's mode does not track the shifts: r = {corr:.4f}"

# %% [markdown]
# **Classical against Adam: the same optimum.** An Adam fit with
# `EarlyStopping`, validated on the training rows, reaches the same
# coefficients. The classical fit gives determinism and the exact MLE, not
# speed. The cell prints both timings and does not assert them.

# %%
flow_a = CausalFlowDAG(spec_vaca, seed=0)
t0 = time.perf_counter()
flow_a.fit(
    df,
    epochs=2000,
    batch_size=4096,
    learning_rate=1e-1,
    validation_data=df,
    callbacks=EarlyStopping(patience=60),
    seed=0,
)
t_adam = time.perf_counter() - t0

coef_c, coef_a = flow_c.ls_coefficients(), flow_a.ls_coefficients()
print(f"{'coef':<10}{'classical':>12}{'adam':>12}{'|diff|':>10}")
adam_gaps = []
for node, p in [("x2", "x1"), ("x3", "x1"), ("x3", "x2")]:
    c, aw = coef_c[node][p][0], coef_a[node][p][0]
    adam_gaps.append(abs(c - aw))
    print(f"{p}->{node:<6}{c:>12.4f}{aw:>12.4f}{adam_gaps[-1]:>10.4f}")
print(f"\nclassical {rep['seconds']:.2f}s  vs  adam {t_adam:.1f}s")
claim("max |coef_classical - coef_adam|", float(max(adam_gaps)), 1e-3)

# %% [markdown]
# ## 2. Ordinal case
#
# For an ordinal outcome the classical model is the ordered logit of
# `statsmodels.OrderedModel`. The data is the paper's mixed triangle
# (`experiments/data/triangle-mixed/linear`): `x3` has four levels and the
# continuous parents `x1` and `x2`
# ([`docs/paper-replication.md`](../docs/paper-replication.md)).

# %%
tm = pd.read_csv(DATA / "triangle-mixed" / "linear" / "obs.csv")

spec_tm = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(LS("x1")),
    "x3": OrdinalNode(4, LS("x1") + LS("x2")),
}

flow_o = CausalFlowDAG(spec_tm, seed=0)
flow_o.fit_classical(tm)

# %% [markdown]
# ### Refitting the same design in statsmodels
#
# `flow.design_matrix(df, node, drop_first=True)` gives the encoding of the
# flow's own shifts, so `statsmodels` fits the same design.

# %%
design = flow_o.design_matrix(tm, "x3", drop_first=True)
res = OrderedModel(tm["x3"].astype(int), design, distr="logit").fit(
    method="bfgs", disp=False
)

fitted = flow_o.ls_coefficients()["x3"]
print(f"{'coefficient':<14}{'fit_classical':>14}{'statsmodels':>13}{'|diff|':>9}")
for name in ("x1", "x2"):
    a_, b_ = float(fitted[name][0]), res.params[name]
    print(f"{name:<14}{a_:>14.4f}{b_:>13.4f}{abs(a_ - b_):>9.4f}")

# %% [markdown]
# ### Standard errors
#
# The helper of Section 1 applies unchanged, and `statsmodels` reports the
# same quantities.

# %%
table_o, diag_o = ls_conf_int(flow_o, "x3", tm)
ci_sm = res.conf_int()
print(
    f"{'':<6}{'flow SE':>9}{'sm SE':>8}{'|est|/SE':>10}   "
    f"{'flow 95%':>18}{'statsmodels 95%':>20}"
)
for term in ("x1", "x2"):
    r = table_o.loc[term]
    print(
        f"{term:<6}{r['se']:>9.4f}{res.bse[term]:>8.4f}"
        f"{abs(r['estimate']) / r['se']:>10.2f}   "
        f"[{r['lower']:+.3f}, {r['upper']:+.3f}]  "
        f"[{ci_sm.loc[term, 0]:+.3f}, {ci_sm.loc[term, 1]:+.3f}]"
    )

se_gap = max(
    abs(float(table_o.loc[term, "se"]) - float(res.bse[term])) for term in ("x1", "x2")
)
claim("max |SE_flow - SE_statsmodels|", se_gap, 1e-3)

# %% [markdown]
# The standard errors agree with `statsmodels`. The cell below checks that
# the coefficient gap is a small fraction of one standard error.

# %%
print(f"|grad| at the classical optimum: {diag_o['grad_norm']:.1e}")
for term in ("x1", "x2"):
    observed = abs(float(table_o.loc[term, "estimate"]) - float(res.params[term]))
    print(f"{term:<4} |flow - statsmodels| = {observed:.2e}")
    claim(
        f"{term}: gap against statsmodels / SE",
        observed / float(table_o.loc[term, "se"]),
        1e-3,
    )

# %% [markdown]
# ## 3. Warm start: classical fit, then further training
#
# A continued `fit()` from the classical solution stays put. This confirms
# the optimum and makes the classical fit a warm start.

# %%
before = {k: v.copy() for k, v in flow_o.ls_coefficients()["x3"].items()}

flow_o.fit(tm, epochs=300, learning_rate=1e-3, batch_size=256, seed=0)
after = flow_o.ls_coefficients()["x3"]

print("coefficient drift after 300 more Adam epochs from the classical MLE:")
for p in ["x1", "x2"]:
    d = float(np.abs(after[p] - before[p]).max())
    claim(f"{p}: max |drift| / SE", d / float(table_o.loc[p, "se"]), 0.1)
