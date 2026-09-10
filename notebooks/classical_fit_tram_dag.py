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
# node-conditional is a *classical transformation model*
#
#  * an ordered-logit / proportional-odds model for ordinal nodes
#
#  * a continuous-outcome logistic transformation model (R's `tram::Colr`) for continuous ones.
#
# For such a model, the classical optimizer `flow.fit_classical()` is best suited:
# - **full-batch, float64, L-BFGS** with a strong-Wolfe line search,
# - **deterministic** — no minibatching, so the same start gives bit-identical
#   results,
# - lands on the **exact maximum-likelihood estimate**, matching `statsmodels`
#   and R to ~1e-3 on the well-identified coefficients.
#
# Every R reference printed below is hard-coded from a real fit, and every one
# of those fits lives in `notebooks/classical_fit_tram_dag.R`. Run
# `Rscript notebooks/classical_fit_tram_dag.R` from the repo root to re-check
# them; it needs `MASS` (ships with R) and `tram`.
#

# %%
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.miscmodels.ordinal_model import OrderedModel

from tramdag import LS, SI, CausalFlowDAG, ContinuousNode, OrdinalNode
from tramdag.callbacks import PerNodePlateau, per_node_adam

warnings.filterwarnings("ignore")

# repo-relative data, whether the notebook runs from the repo root or notebooks/
HERE = [Path.cwd(), *Path.cwd().parents]
REPO = next(p for p in HERE if (p / "pyproject.toml").exists())
DATA = REPO / "experiments" / "misc" / "data"
NB_DATA = REPO / "notebooks" / "data"


def claim(text, value, bound, fmt="{:.2e}"):
    """Assert a computed quantity stays inside its bound, and print both.

    Every agreement this notebook asserts goes through here, so no number is
    typed into a markdown cell where it can drift away from the code. The
    prose says what is being checked and why the bound is what it is; the
    bound is the only number in the source.

    A bound sits at roughly 1.5 to 4 times the measured value, which is the
    band that ``experiments/*/ground_truth/*.json`` already uses. Tighter than
    that and another machine trips it; wider and it stops meaning anything.
    """
    ok = "ok  " if value <= bound else "FAIL"
    print(f"{ok}{text}: {fmt.format(value)} <= {fmt.format(bound)}")
    assert value <= bound, f"{text}: {fmt.format(value)} exceeds {fmt.format(bound)}"


# %% [markdown]
# ## 0. The zeroth example — logistic regression on `MASS::birthwt`
#
# Before any DAG, take the model everyone already knows, on a dataset every R
# user already has. A **two-level** `OrdinalNode` whose terms are all `LS` *is*
# logistic regression — not an analogue of it, the same model. The ordinal
# transform is
#
# $$P(Y \le 0 \mid x) = \sigma\!\left(\theta_0 - \textstyle\sum_p w_p x_p\right),$$
#
# and with two levels there is a single cutpoint $\theta_0$, so
#
# $$\operatorname{logit} P(Y = 1 \mid x) = -\theta_0 + \textstyle\sum_p w_p x_p .$$
#
# The shift weights **are** the logistic-regression coefficients; the intercept
# is **minus** the cutpoint, because an ordinal node *subtracts* its shift. That
# sign convention is the one thing worth internalizing here — Section 2 is this
# same picture with $K-1$ cutpoints instead of one.
#
# The data is `birthwt` from **MASS**, the low-birth-weight study of Hosmer &
# Lemeshow: 189 births, outcome `low` (birth weight under 2.5 kg), predictors
# `age` (mother's age), `lwt` (mother's weight at last period, lbs) and `smoke`
# (smoking during pregnancy). `notebooks/data/birthwt.csv` is those four columns
# exported verbatim from MASS. Please note that this is not a **causal** model.

# %% [markdown]
# ### The R you can copy-paste
#
# `birthwt` ships with MASS, so nothing needs downloading — paste this into any
# R session and you have the reference fit:
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
# (R 4.2.3, MASS 7.3-58.2.) Those four numbers are hard-coded below as `R_GLM`,
# so the notebook can show the three-way comparison without an R installation.

# %%
R_GLM = {  # coef(glm(low ~ age + lwt + smoke, birthwt, family = binomial))
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
# The spec. `low` is the outcome; `smoke` is a binary **parent**, so it is an
# ordinal node too. `age` and `lwt` are source nodes — the flow models their
# marginals as well, which a plain `glm` does not. That is a nuisance here, not
# a feature, so they get the cheapest transform (`SI(transform="affine")`), and
# it cannot disturb the outcome node in any case: the joint NLL decomposes per
# node, so the `low` node's gradient never sees the covariate marginals.

# %%
spec_bw = {
    "age": ContinuousNode([SI(transform="affine")]),
    "lwt": ContinuousNode([SI(transform="affine")]),
    "smoke": OrdinalNode(2),
    "low": OrdinalNode(2, [LS("age"), LS("lwt"), LS("smoke")]),
}

flow_bw = CausalFlowDAG(spec_bw, seed=0)
flow_bw.fit_classical(bw)

# %% [markdown]
# Now read the coefficients back out. `age` and `lwt` are continuous parents, so
# they enter raw and their weights compare directly. `smoke` is ordinal, so it
# enters **one-hot over both levels**.
#
# > **Careful — different coding from R.** R gives a binary predictor one
# > column; we give it two, $w_0$ and $w_1$. Only the difference $w_1 - w_0$ is
# > determined by the data, and it is what R's single `smoke` coefficient means.
# > Each level alone is fixed by the weight initialization: a different `seed`
# > shifts both by the same constant and moves $\theta_0$ to compensate, leaving
# > every fitted probability unchanged. Section 2 uses the same rule for `T`.
#
# The second column also moves the intercept. With $s \in \{0, 1\}$ we have
# $\mathbf{1}[s = 0] = 1 - s$, so the one-hot pair collapses to
#
# $$w_0\,\mathbf{1}[s = 0] + w_1\,\mathbf{1}[s = 1] \;=\; w_0 + (w_1 - w_0)\,s ,$$
#
# and the node reads
#
# $$\operatorname{logit} P(\texttt{low} = 1) \;=\; (-\theta_0 + w_0) \;+\; w_{\text{age}}\,\texttt{age} \;+\; w_{\text{lwt}}\,\texttt{lwt} \;+\; (w_1 - w_0)\,s .$$
#
# The first bracket is R's `(Intercept)` — it is $-\theta_0 + w_0$, not
# $-\theta_0$ — and the last is R's `smoke`.

# %%
import statsmodels.formula.api as smf  # noqa: E402

from tramdag.transforms import ordinal_cutpoints  # noqa: E402

# the same formula as the R call above, and the same names in the output
res_bw = smf.logit("low ~ age + lwt + smoke", data=bw).fit(disp=False)

w = flow_bw.ls_coefficients()["low"]
with torch.no_grad():  # cutpoints are [-inf, theta_0, +inf]; take the finite one
    theta_0 = float(ordinal_cutpoints(flow_bw.nodes["low"].intercept(1))[0, 1])

rows_bw = [
    # -theta_0 + w_smoke[0]: the one-hot level-0 column is part of the intercept
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
for name, flow_value, sm_value, r_value in rows_bw:
    spread = max(abs(flow_value - sm_value), abs(flow_value - r_value))
    print(
        f"{name:<11}{flow_value:>14.4f}{sm_value:>10.4f}{r_value:>10.4f}{spread:>11.2e}"
    )

# %% [markdown]
# Three optimizers — L-BFGS on a transformation model, `statsmodels`' Newton
# solver, R's IRLS — land on the same maximum likelihood, because there is only
# one. The agreement is not only in the parameters: the log-likelihood of the
# `low` node and R's `logLik(m)` are the same number, and `flow.pmf` reproduces
# `glm`'s fitted probabilities row by row.

# %%
# nll() is the *mean* NLL per node; times n and negated it is the log-likelihood
ll_flow = -flow_bw.nll(bw)["low"] * len(bw)
p_flow = flow_bw.pmf(bw, node="low")[:, 1]
p_classical = res_bw.predict(bw).values
print(f"log-likelihood   flow {ll_flow:.6f}   R glm {R_LOGLIK:.6f}")
print(
    f"max |P_flow(low=1) - P_glm(low=1)| over {len(bw)} rows: "
    f"{np.abs(p_flow - p_classical).max():.2e}"
)

# %% [markdown]
# ## 1. Continuous case
#
# Section 0 modelled `low`, the *dichotomized* birth weight. `birthwt` also
# carries the number it was cut from — `bwt`, the weight in grams — so the same
# three predictors can be fitted against a continuous outcome. That is a
# continuous outcome logistic transformation model, R's `tram::Colr`
# bwt ~ age + lwt + smoke
# No way to model this with glm.

# %%
R_COLR_BWT = {  # Colr(bwt ~ age + lwt + smoke, order = 21); R 4.2.3 / tram 1.0.4
    "age": -0.021971,
    "lwt": -0.010233,
    "smoke": 0.668698,
    "loglik": -1499.6678,
}

spec_bwt = {
    "age": ContinuousNode([SI(transform="affine")]),  # nuisance marginals
    "lwt": ContinuousNode([SI(transform="affine")]),
    "smoke": OrdinalNode(2),
    "bwt": ContinuousNode(
        [SI(n_coeffs=20), LS("age"), LS("lwt"), LS("smoke")]  # degree 21
    ),
}

flow_bwt = CausalFlowDAG(spec_bwt, seed=0)
flow_bwt.fit_classical(bw, max_iter=800)

w = flow_bwt.ls_coefficients()["bwt"]
fitted = {
    "age": w["age"][0],
    "lwt": w["lwt"][0],
    "smoke": w["smoke"][1] - w["smoke"][0],  # one-hot: read the difference
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
# The largest *absolute* difference is in `smoke`, but that only reflects its
# coefficient being some thirty times bigger than the others; as a fraction of
# the estimate it is the smallest of the three. Neither comparison is the useful
# one. A difference matters when it is large relative to the **sampling
# uncertainty** of the estimate — and that is a quantity we have not computed
# yet.

# %% [markdown]
# ### Standard errors and confidence intervals
#
# The sampling distribution of the maximum likelihood estimator is
#
# $$\hat{\theta} \;\overset{\cdot}{\sim}\; N(\theta, I^{-1})$$
#
# where $I$ is the Fisher information matrix. Since $\theta$ is unknown we
# evaluate $I$ at $\hat\theta$, and read the standard errors off the diagonal of
# $I^{-1}$. For a contrast $c$, we read off $\sqrt{c^\top I^{-1} c}$.
#
# The **observed** Fisher information is the Hessian of the *negative*
# log-likelihood at the MLE,
#
# $$I_{ab} = \frac{\partial^{2}(-\ell)}{\partial\theta_a\,\partial\theta_b}\bigg|_{\hat\theta},$$
#
# which autograd gives directly: one backward pass produces the gradient with
# the graph retained, then one further backward pass per component produces a
# row of $I$.
#
# Here $I$ is singular, so $I^{-1}$ does not exist. It is symmetric, so it has an
# orthonormal eigendecomposition, and the **pseudo-inverse** inverts only the
# directions that carry curvature:
#
# $$I = \sum_{k=1}^{P} \lambda_k\, e_k e_k^{\top}
#   \qquad\Longrightarrow\qquad
#   I^{+} = \sum_{k\,:\ \lambda_k > \tau} \frac{1}{\lambda_k}\, e_k e_k^{\top},
#   \qquad \tau = 10^{-7}\,\lambda_{\max}.$$
#
# Terms with $\lambda_k \le \tau$ are dropped rather than inverted. That is right
# for a contrast which avoids the flat subspace, and the `leak` column measures
# exactly how far it fails to:
#
# $$\operatorname{leak}(c) = \max_{k\,:\ \lambda_k \le \tau} \left| c^{\top} e_k \right| .$$
#
# Only interpret rows whose leak is close to zero. The choice of $\tau$ is not
# delicate, because the spectrum splits into a flat block and a curved block
# with orders of magnitude between them. Any threshold inside that gap gives
# the same answer, and the cell below measures the gap rather than asserting
# it.

# %%
from scipy.stats import norm  # noqa: E402


def observed_information(flow, node, data):
    """Give one node's NLL Hessian, the gradient norm, and parameter offsets."""
    flow.double()  # second derivatives need float64
    names, params = zip(*flow.nodes[node].named_parameters(), strict=True)
    nll = -flow.node_log_prob(flow._tensorize(data))[node].sum()
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
    flow.float()  # back to the stored precision
    return result


def spectrum_split(evals, tau):
    """Describe the curvature spectrum either side of ``tau``.

    A wide gap between the flat block and the curved block is what makes the
    threshold uncritical, so this reports both edges of the gap and lets the
    caller check it instead of describing it in prose.
    """
    absolute = np.abs(evals)
    flat, curved = absolute[absolute <= tau], absolute[absolute > tau]
    return {
        "n_flat": int(flat.size),
        "flat_max": float(flat.max()) if flat.size else 0.0,
        "curved_min": float(curved.min()) if curved.size else 0.0,
    }


def ls_conf_int(flow, node, data, level=0.95):
    """Give Wald intervals for the linear-shift coefficients of one node."""
    hess, grad_norm, offsets = observed_information(flow, node, data)
    evals, evecs = np.linalg.eigh(hess)
    rcond = 1e-7
    tau = rcond * np.abs(evals).max()  # one threshold, used for both
    flat = evecs[:, np.abs(evals) <= tau]  # the directions without curvature
    cov = np.linalg.pinv(hess, rcond=rcond)  # invert only the rest
    z = norm.ppf(0.5 + level / 2)

    rows = []
    for parent, w in flow.ls_coefficients()[node].items():
        c = np.zeros(hess.shape[0])
        key = next(n for n in offsets if n.startswith(f"shifts.{parent}"))
        if len(w) == 1:  # continuous parent: one weight
            c[offsets[key]], estimate, term = 1.0, w[0], parent
        else:  # ordinal parent: the contrast w[1] - w[0]
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
                # 0 = the contrast avoids the flat subspace, so the SE is real
                "leak": float(np.abs(c @ flat).max()) if flat.size else 0.0,
            }
        )
    diagnostics = {
        "n_params": hess.shape[0],
        "grad_norm": grad_norm,
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
# Colr(bwt ~ age + lwt + smoke, order = 21) on birthwt; R 4.2.3 / tram 1.0.4,
# produced by classical_fit_tram_dag.R. This is the oracle, not the claim.
R_COLR_BWT_SE = {"age": 0.025305, "lwt": 0.004116, "smoke": 0.260131}

print()
claim(
    "flat_max / curved_min, so smaller is a wider gap",
    diag["flat_max"] / diag["curved_min"],
    1e-4,
)
se_gap = max(
    abs(float(table.loc[t, "se"]) - r)
    for t, r in [("age", R_COLR_BWT_SE["age"]), ("lwt", R_COLR_BWT_SE["lwt"])]
)
claim("max |SE_flow - SE_Colr|, continuous parents", se_gap, 1e-3)

# %% [markdown]
# The standard errors of the two continuous parents reproduce `Colr`'s, and
# R's `confint()` on a `Colr` fit is Wald as well.
#
# Two things stand in the way of putting confidence intervals into the package
# itself:
#
# 1. The gradient norm printed above is not zero. `fit_classical` stops on
#    flatness of the likelihood rather than on the gradient norm, so the
#    Hessian is taken a little off the exact stationary point.
# 2. An interval only means something for an estimable parameter. A
#    `conf_int` that returned one row per parameter would be mostly
#    meaningless, because most rows sit in the flat subspace.

# %% [markdown]
# ## 1b. A bimodal DAG — the demo data
#
# The VACA benchmark triangle (`x1 → x2 → x3 ← x1`, the demo notebook's bimodal
# SCM): `x1` is a two-component Gaussian mixture, `x2 = -x1 + N(0,1)`,
# `x3 = x1 + 0.25 x2 + N(0,1)`. We fit an **all-`ls`** model: each node is a
# continuous logistic transformation model with a Bernstein baseline and linear
# shifts.
#
# Note this is an *honest misspecification*: the DGP noise is Gaussian while the
# TRAM latent is logistic, so the all-`ls` model is not the true generator — but
# `fit_classical` still finds its exact MLE, which is the point here.


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
# `tram::Colr` fits the same model family — a continuous outcome logistic
# transformation model with a Bernstein baseline. The cell above already wrote
# `notebooks/data/vaca.csv`, so R reads the identical rows the flow is fitted on.
#
# The two libraries **count the basis differently**, and the comparison is only
# meaningful at the same polynomial degree. The flow's `n_coeffs` unconstrained
# coefficients become `n_coeffs + 2` control points, that is a Bernstein
# polynomial of degree `n_coeffs + 1` (`order = n + 1` in `transforms.py`). So
# `N_COEFFS = 20` below is tram's `order = 21`, **not** `order = 19`.
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
# The coefficients need **no sign flip**: `Colr` reports the shift on the same
# side as the flow.

# %%
R_COLR = {  # Colr(..., order = 21) on vaca.csv; R 4.2.3 / tram 1.0.4
    "x2": {"coef": {"x1": 1.800042}, "loglik": -1401.213542},
    "x3": {"coef": {"x1": -1.777289, "x2": -0.455282}, "loglik": -1411.901958},
}

# %%
df = pd.read_csv(NB_DATA / "vaca.csv")
# n_coeffs is stated rather than left to the default, because Section 1 compares
# against R and the two libraries count the basis differently: the flow's
# n_coeffs unconstrained coefficients become n_coeffs + 2 control points, i.e. a
# Bernstein polynomial of degree n_coeffs + 1 (see `order = n + 1` in
# transforms.py). So n_coeffs=20 here is tram's `order = 21`, not `order = 19`.
N_COEFFS = 20  # -> Bernstein degree 21

spec_vaca = {
    "x1": ContinuousNode([SI(n_coeffs=N_COEFFS)]),
    "x2": ContinuousNode([SI(n_coeffs=N_COEFFS), LS("x1")]),
    "x3": ContinuousNode([SI(n_coeffs=N_COEFFS), LS("x1"), LS("x2")]),
}

flow_c = CausalFlowDAG(spec_vaca, seed=0)
rep = flow_c.fit_classical(df)  # logs iters / NLL / time
print(f"\n{'':<12}{'fit_classical':>14}{'R Colr':>12}{'|diff|':>10}")
loglik = {k: -v * len(df) for k, v in flow_c.nll(df).items()}  # nll() is a mean
for node, parents in flow_c.ls_coefficients().items():
    for p, w in parents.items():
        c, r = w[0], R_COLR[node]["coef"][p]
        print(f"{p + ' -> ' + node:<12}{c:>14.4f}{r:>12.4f}{abs(c - r):>10.4f}")
    c, r = loglik[node], R_COLR[node]["loglik"]
    print(f"{'logLik ' + node:<12}{c:>14.4f}{r:>12.4f}{abs(c - r):>10.4f}")

# %% [markdown]
# ### Why this one is not an exact-MLE check
#
# The coefficients here agree far less closely than in Sections 0 and 2, and
# neither result is the more correct one. The two libraries place the
# Bernstein basis differently: the flow puts it on the 5%/95% quantiles and
# continues as a straight line outside them, which leaves a tenth of the rows
# in the tails, while `Colr` uses the full range of the data. The shift
# coefficients survive that difference in relative terms, which is why the
# ordered logit above finds the same two numbers without any Bernstein basis
# at all.

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
# Two checks are worth making on it, and neither needs an external reference.
# A density must integrate to one over a wide enough grid, and its mode must
# track the conditional mean that the linear shifts imply. The first is the
# stronger check, because it exercises the log-determinant of the transform
# rather than only its location.

# %%
grid = np.linspace(df["x3"].min() - 2, df["x3"].max() + 2, 601)
rows = df.iloc[:200]
dens = flow_c.density(rows, "x3", grid)  # (200, 601)
mass = np.trapezoid(dens, grid, axis=1)
claim("worst |integral of the density - 1|", float(np.abs(mass - 1.0).max()), 5e-3)

# the mode should move with the parents in the direction the shifts say: the
# flow subtracts on the latent scale, so a positive weight lowers the value
modes = grid[dens.argmax(axis=1)]
shift = -(
    float(flow_c.ls_coefficients()["x3"]["x1"][0]) * rows["x1"]
    + float(flow_c.ls_coefficients()["x3"]["x2"][0]) * rows["x2"]
)
corr = float(np.corrcoef(modes, shift)[0, 1])
print(f"corr(mode of the fitted density, the linear predictor) = {corr:.4f}")
assert corr > 0.95, f"the density's mode does not track the shifts: r = {corr:.4f}"

# %% [markdown]
# **Classical against Adam: the same optimum.** A converged Adam fit with no
# early stopping reaches the same coefficients. The guarantees of the
# classical fit are determinism and the exact maximum-likelihood estimate, not
# speed. Which one is faster depends on the model. It wins clearly on
# ordinal-outcome models, as in Section 2, while on this Bernstein-heavy
# continuous DAG the flat directions of the basis slow L-BFGS down. Both
# timings are printed below rather than claimed, because a wall clock is a
# property of the machine.

# %%
flow_a = CausalFlowDAG(spec_vaca, seed=0)
t0 = time.perf_counter()
# the pre-0.4 fit(schedule="plateau", freeze_patience=) recipe, now a callback
sched = PerNodePlateau(patience=15, freeze=60)
flow_a.fit(
    df,
    epochs=2000,
    batch_size=4096,
    validation_data=df,
    optimizer=per_node_adam(flow_a, lr=1e-1),
    callbacks=sched,
)
t_adam = time.perf_counter() - t0

coef_c, coef_a = flow_c.ls_coefficients(), flow_a.ls_coefficients()
print(f"{'coef':<10}{'classical':>12}{'adam':>12}{'|diff|':>10}")
for node, p in [("x2", "x1"), ("x3", "x1"), ("x3", "x2")]:
    c, aw = coef_c[node][p][0], coef_a[node][p][0]
    print(f"{p}->{node:<6}{c:>12.4f}{aw:>12.4f}{abs(c - aw):>10.4f}")
print(f"\nclassical {rep['seconds']:.2f}s  vs  adam {t_adam:.1f}s")

# %% [markdown]
# ## 2. Ordinal case (treatment effect)
#
# For an **ordinal** outcome the classical model is the ordered-logit, which
# `statsmodels.OrderedModel` fits — so here the equivalence is checkable in
# Python. The all-`ls` stroke DAG (the synthetic `magic-mrclean/ls` cohort):

# %%
obs = pd.read_csv(DATA / "magic-mrclean" / "ls" / "obs.csv")

spec_stroke = {
    "Age": ContinuousNode(),
    "mRS_pre": OrdinalNode(6, [LS("Age")]),
    "NIHSSa": ContinuousNode([LS("Age"), LS("mRS_pre")]),
    "T": OrdinalNode(2, [LS("Age"), LS("mRS_pre"), LS("NIHSSa")]),
    "mRS_3m": OrdinalNode(7, [LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")]),
}

flow_s = CausalFlowDAG(spec_stroke, seed=0)  # the seed validate_ls.py checks
flow_s.fit_classical(obs)

# %% [markdown]
# ### Refitting the same design in statsmodels
#
# `flow.design_matrix(node, drop_first=True)` gives the encoding the flow
# feeds its own shifts, so handing it to `statsmodels` compares two fits of
# one design rather than two different designs.

# %%

design = flow_s.design_matrix(obs, "mRS_3m", drop_first=True)
res = OrderedModel(obs["mRS_3m"].astype(int), design, distr="logit").fit(
    method="bfgs", disp=False
)

fitted = flow_s.ls_coefficients()["mRS_3m"]
rows = [
    ("Age", float(fitted["Age"][0]), res.params["Age"]),
    ("NIHSSa", float(fitted["NIHSSa"][0]), res.params["NIHSSa"]),
    ("T (1 vs 0)", float(fitted["T"][1] - fitted["T"][0]), res.params["T[1]"]),
]
print(f"{'coefficient':<14}{'fit_classical':>14}{'statsmodels':>13}{'|diff|':>9}")
for name, a_, b_ in rows:
    print(f"{name:<14}{a_:>14.4f}{b_:>13.4f}{abs(a_ - b_):>9.4f}")

# %% [markdown]
# ### Standard errors, and what is actually weakly identified
#
# The helper from Section 1 needs no change: give it the node, and it builds the
# contrast for each ordinal parent. `statsmodels` reports the same quantities,
# so every number below has an external check.

# %%
table_s, diag_s = ls_conf_int(flow_s, "mRS_3m", obs)
ci_sm = res.conf_int()
print(
    f"{'':<12}{'flow SE':>9}{'sm SE':>8}{'|est|/SE':>10}   "
    f"{'flow 95%':>18}{'statsmodels 95%':>20}"
)
for term, key in [("Age", "Age"), ("NIHSSa", "NIHSSa"), ("T (1 vs 0)", "T[1]")]:
    r = table_s.loc[term]
    print(
        f"{term:<12}{r['se']:>9.4f}{res.bse[key]:>8.4f}"
        f"{abs(r['estimate']) / r['se']:>10.2f}   "
        f"[{r['lower']:+.3f}, {r['upper']:+.3f}]  "
        f"[{ci_sm.loc[key, 0]:+.3f}, {ci_sm.loc[key, 1]:+.3f}]"
    )
print(
    f"\n{diag_s['n_params']} parameters, {diag_s['n_flat']} without curvature"
    " — one per one-hot parent, mRS_pre and T"
)

# %%
se_gap = max(
    abs(float(table_s.loc[term, "se"]) - float(res.bse[key]))
    for term, key in [("Age", "Age"), ("NIHSSa", "NIHSSa"), ("T (1 vs 0)", "T[1]")]
)
claim("max |SE_flow - SE_statsmodels|", se_gap, 1e-3)

# the ranking is the claim, not the ratio: the treatment is the best determined
# coefficient of the three, and which one that is must not depend on a seed
ratios = {
    term: abs(float(table_s.loc[term, "estimate"]) / float(table_s.loc[term, "se"]))
    for term in ("Age", "NIHSSa", "T (1 vs 0)")
}
print("|estimate| / SE:", {k: round(v, 2) for k, v in ratios.items()})
# The claim the next section rests on: none of these three is the weak one, so
# the weak coefficient has to be found elsewhere. Two standard errors from zero
# is the conventional line.
assert min(ratios.values()) > 2.0, f"one of these is not well determined: {ratios}"

# %% [markdown]
# The standard errors agree with `statsmodels`, and all three coefficients sit
# well clear of zero. The treatment effect is therefore not the uncertain one
# here, whatever the ordering of the three happens to be.
#
# To find the weak one, look at every `mRS_pre` level instead of only the
# first, and print how many rows carry it.

# %%
hess_s, _, offsets_s = observed_information(flow_s, "mRS_3m", obs)
cov_s = np.linalg.pinv(hess_s, rcond=1e-7)
at = offsets_s[next(n for n in offsets_s if n.startswith("shifts.mRS_pre"))]
w_pre = flow_s.ls_coefficients()["mRS_3m"]["mRS_pre"]
counts = obs["mRS_pre"].value_counts()

print(f"{'level':>6}{'rows':>7}{'estimate':>10}{'SE':>9}{'|est|/SE':>10}   95% CI")
for level in range(1, 6):
    c = np.zeros(hess_s.shape[0])
    c[at], c[at + level] = -1.0, 1.0  # w[level] - w[0]
    est = w_pre[level] - w_pre[0]
    se = float(np.sqrt(c @ cov_s @ c))
    print(
        f"{level:>6}{counts.get(level, 0):>7}{est:>10.4f}{se:>9.4f}"
        f"{abs(est) / se:>10.2f}   [{est - 1.96 * se:+.3f}, {est + 1.96 * se:+.3f}]"
    )

# %% [markdown]
# The rarest level has the widest interval. That is what a weakly identified
# coefficient looks like, and the second column shows the cause: too few rows
# carry it. The relationship is the claim, so the cell below checks it rather
# than quoting one run's numbers.

# %%
by_level = {}
for level in range(1, 6):
    c = np.zeros(hess_s.shape[0])
    c[at], c[at + level] = -1.0, 1.0
    by_level[level] = (
        int(counts.get(level, 0)),
        float(np.sqrt(c @ cov_s @ c)),
    )
rarest = min(by_level, key=lambda k: by_level[k][0])
widest = max(by_level, key=lambda k: by_level[k][1])
print(f"rarest level: {rarest}   widest standard error: level {widest}")
assert rarest == widest, (
    f"level {rarest} has the fewest rows but level {widest} has the widest "
    "interval, so row count is no longer what drives the uncertainty"
)

# %% [markdown]
# The same slack explains the `|diff|` column above. The gap between the flow
# and `statsmodels` is not evidence about identification. It is the optimizer
# stopping while the likelihood is still flat, and the displacement that
# causes is $c^\top I^{+}\nabla\ell$. The cell below computes that
# displacement and checks it against the observed difference, which turns a
# hand-checked coincidence into a pinned identity. `Age` looks worse than `T`
# only because its standard error is smaller, so the same numerical slack is a
# larger fraction of it.

# %%
print(f"|grad| at the classical optimum: {diag_s['grad_norm']:.1e}")
for term, key in [("Age", "Age"), ("NIHSSa", "NIHSSa")]:
    observed = abs(float(table_s.loc[term, "estimate"]) - float(res.params[key]))
    print(
        f"{term:<8} |flow - statsmodels| = {observed:.2e}"
        f"   = {observed / float(table_s.loc[term, 'se']):.0%} of one SE"
    )
    claim(f"{term}: gap against statsmodels", observed, 5e-2)
#
# `experiments/misc/validate_ls.py` runs this comparison as a checked
# experiment, and adds R (`MASS::polr` / `tram`) and the treatment effect. Its
# docstring records the same finding about `mRS_pre` level 5.

# %% [markdown]
# ## 3. Warm-start handoff: classical fit → further training
#
# `fit_classical` leaves the model at the MLE in float32, ready for any normal
# operation. Two things to verify:
#
# 1. the float64→float32 round-trip didn't move the coefficients, and
# 2. continuing with `fit()` from the classical solution *stays put* — confirming
#    it really is the optimum (and showing the classical fit as a fast, principled
#    initialization for further or richer training).

# %%
before = {k: v.copy() for k, v in flow_s.ls_coefficients()["mRS_3m"].items()}

# continue training from the classical solution with a gentle Adam phase
flow_s.fit(obs, epochs=300, learning_rate=1e-3, batch_size=256)
after = flow_s.ls_coefficients()["mRS_3m"]

print("coefficient drift after 300 more Adam epochs from the classical MLE:")
for p in ["Age", "NIHSSa", "T"]:
    d = float(np.abs(after[p] - before[p]).max())
    print(f"  {p:<8} max|Δ| = {d:.4f}")
print("\n-> small drift = the classical fit was already at the optimum;")
print("   fit_classical is a valid warm start for continued / flexible training.")
