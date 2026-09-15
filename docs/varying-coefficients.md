# Varying-coefficient treatment effects: `VC(*modifiers, t=, penalty=, center=)`

A `VC` term gives a node a treatment-effect head with its own bias-variance
budget. It contributes $\beta(x)\, x_t$ with
$\beta(x) = \beta_0 + b_\Theta(x)$ to the node's shift, where $b_\Theta$ is a
deliberately small, penalized network of the modifiers. The point is not
expressiveness. The point is that the flow estimates the effect function with
care, not as a by-product. Every claim here runs in
[`notebooks/varying_coefficients.py`](../notebooks/varying_coefficients.py),
which is the one place for the code.

Modifiers may appear twice in a node, as prognostic parents through `CS` or
`LS` and as effect modifiers through `VC`. That pattern is intended. Only the
treatment named by `t=` owns its edge, and a second term that declares it
raises an error.

## Why not a joint `CS` over treatment and modifiers

For a binary treatment the multi-parent `CS` is equivalent in expressiveness,
since any shift decomposes as $s(x,t) = s(x,0) + [s(x,1) - s(x,0)]\, t$. But
the `CS` form has no effect-specific regularization. The likelihood rewards a
good average fit of $s(x,t)$, and nothing rewards a smooth difference between
the arms, so the read-out is the difference of two jointly fitted,
unregularized networks and amplifies noise. On the `vc_hetero` process the
`CS` reduced form reaches a correlation of about 0.5 with the true effect
function even though the model is in-class; the `VC` term reaches about 0.99
on the same protocol, and `tests/test_vc_term.py` pins the bar at 0.9. Causal
forests and R-learners work not because they target the effect but because
they regularize it (Nie & Wager 2021; Athey, Tibshirani & Wager 2019). `VC`
brings that ingredient into the TRAM framework.

## Semantics

- **Scale.** $\beta(x)$ lives on the node's latent scale, added for a
  continuous node and subtracted from the cutpoints for an ordinal one,
  exactly like an `LS` weight. With no modifiers, `VC(t="T")` is `LS("T")`
  bit-exactly, so `VC` against `LS` is a nested question.
- **Penalty.** The objective is $\sum_i \mathrm{NLL}_i + \lambda \lVert
  b_\Theta \rVert^2$ on the total-NLL scale, with `penalty=` as $\lambda$.
  It is a fixed Gaussian prior whose shrinkage vanishes as $n$ grows, it never
  applies to $\beta_0$, and $\lambda \to \infty$ recovers the `LS` fit. Raise
  it when the modifiers are many or $n$ is small.
- **Identification.** A constant moves freely between $\beta_0$ and
  $b_\Theta$. The head's output layer is zero-initialized, so
  $\beta(x) = \beta_0$ at step 0, and after `fit` the flow re-centers the head
  to mean zero over the training data, which preserves the function. $\beta_0$
  is therefore the main effect in the training population, the Colr reading
  when $\beta$ is constant.
- **Warm start.** Fit the all-`LS` version classically and copy the treatment
  weight into `beta0` before `fit`, so training starts at the classical answer
  and learns only deviations.
- **Treatments.** Continuous or binary ordinal, entering as its 0/1 level so
  that $\beta$ is the identified level-1-against-0 contrast. The term is
  linear in the treatment.
- **Read-out.** `flow.varying_coef(df, node)` evaluates $\beta(x)$ in closed
  form, deterministic and free of the outcome. For a binary treatment it
  equals the abduction difference $u(x,1,y) - u(x,0,y)$ identically, which a
  test pins.

## Propensity centering: `center="col"`

With `center=` the term contributes $\beta(x)\,(t - \hat e(x))$ instead of
$\beta(x)\, t$. This is Robinson's R-learner orthogonalization inside the
likelihood, the ingredient Dandl et al. (2024) found decisive for effect
estimation under confounding in model-based forests. The finding reproduces
here on the `confounded` process, where the model deliberately
under-specifies its prognostic part: the uncentered $\hat\beta$ absorbs the
confounded misfit, and centering brings it back near the truth. The notebook
reports the measured reduction and `tests/test_vc_centered.py` requires at
least a factor of two. If the prognostic part is correctly specified,
centering changes little. It is insurance against the misspecification you do
not know you have.

The design is two-stage and frozen, because the naive versions are wrong.

- **Training** uses out-of-fold propensities that you compute and pass as the
  training-frame column `center=` names, one value per row. Any propensity
  model works as long as each fold is predicted by a fit that never saw it,
  the cross-fitting requirement of double machine learning. In-sample
  propensities reintroduce the own-observation bias and can be worse than no
  centering. The values enter the loss as frozen data, so no gradient reaches
  the treatment node and the per-node factorization stays intact. `fit`
  refuses a centered spec whose frame lacks the column.
- **Inference**, in `log_prob`, `sample`, `abduct`, `pmf` and `scores`,
  recomputes $\hat e$ from the flow's own fitted treatment node on the current
  parent values, detached. Under `do(T=t)` the regressor becomes
  $t - \hat e(x)$. Nothing is cached.
- **Interpretation.** With centering, $\beta_0$ is the effect at the
  treatment margin. `varying_coef` is unchanged, because centering moves the
  regressor and not $\beta$. Centering requires a binary ordinal treatment.

## Validation

The `vc_hetero` process in [`tests/conftest.py`](../tests/conftest.py) is a
logistic-shift SCM with a known $\beta(x) = -1 + 0.8\,X_2 - 0.6\,X_3$, a
nonlinear prognostic part and confounded assignment, where $X_2$ is
confounder and modifier at once. That is the configuration in which the `CS`
reduced form fails hardest. `tests/test_vc_term.py` requires a recovery
correlation of at least 0.9 at $n = 5000$, a fitted $\beta_0$ that matches
`fit_classical` under a large penalty, and the read-out identities. The
centering claims are measured on the `confounded` process in the same file.
[scores.md](scores.md) covers the scan that shortlists modifiers before a `VC`
term is declared.
