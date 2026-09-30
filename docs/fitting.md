# Fitting a TRAM-DAG

This page is the reference for training a
[`CausalFlowDAG`](../src/tramdag/flow.py): the likelihood, the two fitting
paths `fit` and `fit_classical`, and the hooks of the Adam path. Every recipe
named here runs end to end in
[`notebooks/training_strategies.py`](../notebooks/training_strategies.py),
which is the one place for the code and closes with a runtime comparison of
the recipes on one workload.

## One module, one sub-model per node

A `CausalFlowDAG` is a single `torch.nn.Module` holding one independent
sub-model per variable: an intercept that produces the transform parameters
$\theta$, the monotone transform $h$ with range buffers and no weights of its
own, and one shift module per shift term. The nodes share no parameters. The
DAG lives entirely in which parents each node reads. There is no edge weight
matrix and no shared trunk. [code-map.md](code-map.md) says which class
implements which term.

## The likelihood

The flow maps iid standard-logistic latents $U$ to the observed $X$ in causal
order, and node $i$ reads its parents as data. The Jacobian is therefore
triangular, and the joint log-likelihood decomposes per node,
$\log p(x) = \sum_i \log p(x_i \mid \mathrm{pa}(x_i))$.
`CausalFlowDAG.node_log_prob` computes one term and `log_prob` sums them.

- **Continuous node.** Change of variables through the monotone transform:
  $u = h(x;\theta) + s$ and
  $\log p(x \mid \mathrm{pa}) = \log f_{\text{logistic}}(u) + \log |h'(x)|$.
  The transform returns the log-derivative alongside $u$.
- **Ordinal node.** The ordered-logit head of [model.md](model.md#ordinal-nodes),
  evaluated as the log of the cutpoint-interval probability in log space.

The training loss is the summed per-node mean negative log-likelihood over the
batch, plus the `VC` penalty. `log_prob` returns the per-row joint instead.

Because parents enter as data, the per-node gradients are independent, so a
joint fit of the summed loss equals a separate fit of each node. That is what
licenses per-node learning rates, freezing through a callback, and the
all-`LS` classical fit.

## Path A: stochastic optimization with `fit`

`fit` is the general-purpose trainer, and the only one for a model with a
`CS`, `CI` or `VC` term. It is one minibatch Adam loop that keeps the final
weights.

- **Optimizer.** One optimizer over all parameters, `Adam(lr=learning_rate)`
  by default; `optimizer=` takes any `torch.optim.Optimizer`, and
  `per_node_adam` builds per-node parameter groups.
- **Minibatches.** A fresh `torch.randperm` shuffle each epoch, seeded by
  `seed=`, which seeds the shuffle only and not the weight init.
- **Epochs.** `epochs` has no default: a fixed budget under-spends on one
  workload and wastes on the next, so every caller states its own.
- **Calibration.** The first `fit` calibrates the flow on `train_df`. Every term
  freezes its data-dependent state there, the intercept its `range_q`
  quantiles and each `input_transform=` its statistics. A loaded checkpoint is
  never recalibrated.
- **Marginal start.** Off by default; every simple intercept starts at zuko's
  zero. `fit(marginal_init=True)` starts every Bernstein or ordinal simple
  intercept at the empirical marginal of its column instead.
  [The marginal start](#the-marginal-start) below says how.
- **Validation.** `validation_data=` takes a frame; `validation_split=` takes
  a float and uses the last fraction of `train_df` unshuffled, so shuffle the
  frame first if its row order means anything; only the head calibrates. With
  either, `fit` writes the per-node validation NLL after
  every epoch into `flow.history["val"]`.
- **Logging.** `flow.history["lr"]` records the optimizer's rates per epoch,
  one dict keyed by a group's `node` tag or else its index: `{0: lr}` for the
  default Adam, `{node: lr}` under `per_node_adam`. `verbose=N` prints every
  Nth epoch and the last one; the default 0 is silent.
- **Callbacks.** `callbacks=` takes one `Callback` or a list. The hooks are
  `on_fit_begin`, `on_epoch_end` and `on_fit_end`; a bare callable is an
  `on_epoch_end` hook `cb(flow, epoch, optimizer)`, and any `True` return
  stops the fit. `on_fit_end` runs before the `VC` re-centering. The shipped
  callbacks are `EarlyStopping`, which restores the best-validation weights
  and takes an optional `patience` and `min_delta`, and `PerNodeEarlyStopping`
  with `per_node_adam`, the same per node: a node freezes after `patience`
  flat epochs and loads its best weights back. `fit` refuses two callbacks
  that restore weights. Both read
  `history["val"]`.
- **Centered `VC` propensities** ride the training frame as the column that
  `VC(center=)` names, and split and minibatch with it.
  [varying-coefficients.md](varying-coefficients.md) is the guide.

### The marginal start

`marginal_init=True` sets the $\boldsymbol{\vartheta}$ of each simple
intercept. At shift $s = 0$, the node then has the empirical marginal
$\hat F$ of its training column: exactly for an ordinal node, as a smooth
approximation for a continuous node. An intercept with parents (`I("x1")`) has no
single $\boldsymbol{\vartheta}$ and keeps its initialization.

**Why logit.** The latent $U$ is standard logistic, so its CDF is the
sigmoid $\sigma$ and its quantile function is $\operatorname{logit}$. At
$s = 0$ the model says $P(X \le x) = \sigma(h(x))$. For this CDF to equal
$\hat F$, the transform must be

$$
h(x) = \operatorname{logit} \hat F(x) = \log \hat F(x) - \log\bigl(1 - \hat F(x)\bigr).
$$

The code evaluates $\log(1 - p)$ as `log1p(-p)`, which stays accurate for
$p$ near 0.

**Ordinal node.** `ordinal_marginal_init_theta` takes the class counts. The
model is $P(X \le k) = \sigma(\vartheta_k - s)$, so the targets are the
cumulative class log-odds $c_k = \operatorname{logit} \hat F(k)$ for
$k = 0, \dots, K-2$. The function inverts `ordinal_cutpoints` and gives the unconstrained
parameters $\tilde\vartheta_0 = c_0$ and
$\tilde\vartheta_k = \log(c_k - c_{k-1})$. At
$s = 0$ the class probabilities are the empirical frequencies. For counts
`[50, 30, 15, 5]` the start gives the PMF `[0.50, 0.30, 0.15, 0.05]`.

**Continuous node.** `BernsteinUT.marginal_init_theta` takes the column. A
Bernstein polynomial of order $M$ with control points $\vartheta_k$ is close
to the function whose values at $k/M$ are $\vartheta_k$. zuko turns
`n_coeffs` parameters into $M + 1$ control points, with $M$ =
`n_coeffs + 1`. The start uses this:

1. It places $M + 1$ equally spaced points $x_k$ on the calibrated range,
   from the `range_q` quantile to the `1 - range_q` quantile of the column.
   With `n_coeffs = 20`, $M = 21$ and there are 22 points.
2. It sets each control point to $\operatorname{logit} \hat F(x_k)$, with
   $\hat F(x_k)$ the fraction of rows at or below $x_k$.
3. It inverts zuko's parameterization. zuko builds the control points as a
   cumulative sum of softplus steps and ties the first two and the last two
   steps. The start averages each tied pair and inverts the softplus with
   $\log(e^d - 1)$. It adds zuko's centering offset $n \log 2 / 2$, with
   $n$ = `n_coeffs`, to the first parameter.
   [zuko-upstream.md](zuko-upstream.md#3-public-inverse-of-_constrain_theta)
   lists the zuko internals this relies on.

The polynomial then starts as the Bernstein approximation of
$\operatorname{logit} \hat F$. The approximation smooths $\hat F$. On 4000
rows, the largest gap between the start's CDF and $\hat F$ is 0.010 for a
normal column, 0.080 for a lognormal column and 0.103 for a bimodal column.

The ends of the range carry the empirical quantiles `range_q` and
`1 - range_q`, so the end control points are close to
$\operatorname{logit} q$ and $\operatorname{logit}(1 - q)$ with $q$ =
`range_q`. At the default $q = 0.05$ these are $-2.944$ and $2.944$. A transform with fewer than 3 coefficients ignores the
column and takes the straight line between these two values. Its tied steps
leave no shape to fit. At `n_coeffs = 20`, zuko's zero start is also close
to a straight line, on about $[-6.9, 7.6]$, and 2.5 times steeper.

**Guards.** Two floors keep the start finite and increasing:

- The start clips $\hat F$ to $[10^{-3}, 1 - 10^{-3}]$, so a control point
  or a cutpoint stays within about $\pm 6.9$.
- Each step between adjacent points is at least $10^{-3}$, because the
  softplus and log inverses need strictly increasing points. An empty class
  or a gap in the data therefore gets a small but nonzero probability. An
  empty middle class starts at $2.5 \cdot 10^{-4}$ or less.

**Where it does not apply.** The spline and affine transforms have no
marginal start, and the flag skips them. A `range_q=0` Bernstein transform
maps the data minimum and maximum onto the range ends. Their target
$\operatorname{logit} 0$ does not exist, so `fit` raises. It also raises when
this transform belongs to an intercept with parents. Fit such a model with
`marginal_init=False`.

**What it changes.** The start moves only the initial point. For the three
columns above, zuko's zero start gives a per-row NLL of 2.35, 4.15 and 5.52.
The marginal start gives 1.42, 1.51 and 1.34. The likelihood of an all-`LS`
model has one optimum, which the fit reaches with or without the start.
`fit_classical` therefore takes no such flag. A model with a network shift or a
learning-rate anneal can end in a different basin. D4 in
[paper-replication.md](paper-replication.md#d4-the-marginal-start-measured-per-variant)
measures this per variant.

`fit` applies the start only on the call that calibrates. A second `fit`
therefore continues from the trained intercepts and does not reset them. A
multi-phase schedule depends on this.

### Which recipe

All-`LS` models train to the MLE and keep the final weights. Flexible models
with a `CI`, `CS` or `VC` term overfit observational confounding at the MLE
and need the best-validation weights to recover the causal effect.

| recipe | when |
|---|---|
| `fit_classical` | all-`LS` spec: exact, deterministic, seconds |
| plain Adam | an all-`LS` spec that `fit_classical` refuses, quick looks |
| multi-phase Adam, one `fit` call per rate | a tighter MLE without a scheduler |
| `EarlyStopping()` | any `CI`, `CS` or `VC` model; register it, `fit` has no default |
| `EarlyStopping(patience=)` | also stop once the best epoch is that old |
| a global plateau `Callback` around torch's `ReduceLROnPlateau` | one shared decaying rate |
| `PerNodeEarlyStopping(patience=)` with `per_node_adam` | nodes converge at different speeds; self-stopping |

Two details are easy to get wrong.

- A second `fit` call continues training and `history` accumulates. That is
  what makes a multi-phase schedule a loop of `fit` calls.
- A post-fit `load_state_dict` skips the `VC` re-centering. Restore weights
  from `on_fit_end` instead, as `EarlyStopping` does.

## Path B: classical optimization with `fit_classical`

`fit_classical` is the optimizer for all-`LS` models, where every
node-conditional is an ordered logit or a Colr model. It raises on any `CS`, `CI` or `VC` term.

- **Full-batch, float64, L-BFGS** with a strong-Wolfe line search; no
  minibatches, schedule or early stopping, so the same init gives
  bit-identical results. A converged fit matches `statsmodels` and R to about
  four decimals, where a converged Adam `fit` gets to about 1e-3.
- **Budget.** `max_iter=400` is a default, not a promise. Torch ends the run
  when the NLL or the parameters move by less than 1e-9. A model with several
  nodes can need thousands of iterations to stop on that rule. Give the
  budget room and read the report.
- **The report.** `stop_reason` is `"tolerance"`, `"max_iter"` or `"max_eval"`
  (torch's budget of closure calls, `max_iter * 5 // 4`). `converged` needs
  both: the run stopped on its own AND `grad_norm` is at most
  `tramdag.fitting.GRAD_TOL` (1e-2). Both conditions are necessary, because
  the same tolerance fires when the line search stalls far from the optimum.
  Such a run stops on "tolerance" with a large gradient norm and is not
  converged.
- **float64 is transient.** The fit restores float32 afterwards, and
  checkpoints stay float32.
- **Weakly identified directions never settle.** A Bernstein intercept, rare
  one-hot levels or a flat treatment-effect ridge drift along zero-curvature
  valleys after the likelihood is at its optimum, which is why `tolerance_grad`
  is 0 and the gradient bound is loose. Correctness comes from the comparison with
  classical software in
  [`notebooks/classical_fit_tram_dag.py`](../notebooks/classical_fit_tram_dag.py).

A converged `fit_classical` leaves the model at the MLE, ready for any operation. A
`fit` call from there stays put, which is both a check that the classical
solution is the optimum and a warm start; the `VC` guide uses it for `beta0`.

## Memory and disk

Neither path writes to disk. The parameters, the optimizer state and the
`history` dict live in RAM, and whatever a callback records is yours. The
only disk I/O is the explicit `save()` and `load()`.
