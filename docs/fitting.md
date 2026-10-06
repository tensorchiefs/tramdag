# Fitting a TRAM-DAG

This page is the reference for training a
[`CausalFlowDAG`](../src/tramdag/flow.py) and its nodes. It covers the
likelihood, the two fitting paths `fit` and `fit_classical`, and the hooks of
the Adam path. Every recipe named here runs end to end in
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

Because parents enter as data, the per-node terms have disjoint parameters,
and the maximum of the sum is the maximum of each term. So every node fits on
its own: [`Node.fit`](../src/tramdag/fitting.py) minimizes the node's mean
negative log-likelihood over the batch, plus the node's `VC` penalty.
`CausalFlowDAG.fit` runs `Node.fit` for each node. `log_prob` returns the
per-row joint.

## One node alone

A [`Node`](../src/tramdag/nodes.py) is a TRAM regression of one variable on
its parents, and it needs no DAG. It takes its name, its node spec and a
parent schema, `{parent: "continuous" | n_levels}`.
[`notebooks/training_strategies.py`](../notebooks/training_strategies.py) fits
one. The frame needs the node's column, its parents' columns and its side
columns; other columns are ignored. `fit`, `fit_classical`, `nll`, `save` and
`load` work as on the flow, for one node. `nll` gives one float, and the
`fit_classical` report is the per-node report. `fit` takes an optimizer
instance or a factory `f(node)`, but no `n_jobs`. A centered `VC` needs its
propensity column in every frame, because only the flow can compute it live
from the treatment node. A flow derives each node's shuffling seed from
`fit(seed=)` and the node's position. So `Node.fit(seed=)` alone does not
repeat a node of a seeded flow fit. A node built alone draws its initial
weights from torch's global RNG; call `torch.manual_seed` first for a
reproducible start. It always uses torch's default init; `init=` exists on
`CausalFlowDAG` only.

## Path A: stochastic optimization with `fit`

`fit` is the general-purpose trainer, and the only one for a model with a
`CS`, `CI` or `VC` term. Per node, it is one minibatch Adam loop that keeps
the final weights.

- **Nodes.** `CausalFlowDAG.fit` fits the nodes in topological order.
  `n_jobs=N` forks `N` worker processes (Linux and macOS), one task per node.
  A worker runs torch on one thread, so it gives the weights of a serial fit
  on one thread. `epochs=`, `learning_rate=` and `callbacks=` take one value
  for all nodes or a function of the node name.
- **Optimizer.** One optimizer per node, `Adam(lr=learning_rate)` by default.
  `optimizer=` takes a factory `f(node)`, for example to give the networks
  weight decay in their own parameter group. `Node.fit` also takes an
  optimizer instance.
- **Minibatches.** A fresh shuffle each epoch. `seed=` seeds the shuffle
  only, not the weight init; each node gets its own seed derived from
  `(seed, node index)`.
- **Epochs.** `epochs` has no default: a fixed budget under-spends on one
  workload and wastes on the next, so every caller states its own.
- **Calibration.** The first `fit` or `fit_classical` calibrates each node on
  `train_df`. Every term freezes its data-dependent state there, the
  intercept its `range_q` quantiles and each `input_transform=` its
  statistics. A loaded checkpoint is never recalibrated.
- **Marginal start.** Off by default: every simple intercept starts with all
  its parameters at zero, the zero start. `fit(marginal_init=True)` starts
  each Bernstein or ordinal simple intercept at the empirical distribution of
  its column instead. At shift zero the untrained model then already fits
  each marginal, and the fit needs fewer epochs. Only the fit that
  calibrates the node applies it, so a node that `fit_classical` calibrated
  ignores the flag. [The marginal start](#the-marginal-start) says how, and
  when the endpoint changes.
- **Validation.** `validation_data=` takes a frame; `validation_split=` takes
  a float and uses the last fraction of `train_df` unshuffled, so shuffle the
  frame first if its row order means anything; only the head calibrates. The
  flow splits once, so all nodes see the same rows. With either, `fit` writes
  the node's validation NLL after every epoch into `node.history["val"]`.
- **History.** `node.history` holds `"train"`, `"val"`, `"val_epoch"` and
  `"lr"`, the optimizer's rates per epoch as `{group index: lr}`.
  `flow.history` is a view with one `{node: value}` dict per epoch and key,
  plus the `val_epoch` list. A node that stopped earlier repeats its last
  entry. `verbose=N` prints every Nth epoch and the last one, per node; the
  default 0 is silent.
- **Callbacks.** `callbacks=` takes one `Callback` or a list. The hooks get the
  node: `on_fit_begin(node, optimizer)`, `on_epoch_end(node, epoch, optimizer)`
  and `on_fit_end(node, optimizer)`. In `Node.fit` a bare callable is an
  `on_epoch_end` hook; any `True` return stops that node's fit. `on_fit_end`
  runs before the `VC` re-centering. On the flow, a bare callable is a
  function of the node name that gives that node its callbacks. So on the
  flow, a bare hook goes into a list. The shipped callback is
  `EarlyStopping`: it restores the node's best-validation weights and takes an
  optional `patience` and `min_delta`. It reads `node.history["val"]`. `fit`
  refuses two callbacks that restore weights.
- **Centered `VC` propensities** ride the training frame as the column that
  `VC(propensity=)` names, and split and minibatch with it. On the flow, the
  validation NLL uses the live propensity of the fitted treatment node, as
  every query does. `Node.fit` alone reads the column from the validation
  frame.
  [varying-coefficients.md](varying-coefficients.md) is the guide.

### The marginal start

The marginal start puts the optimizer near the data before the first step.
It changes the parameters the optimizer starts from, not the model or the
loss.

`marginal_init=True` sets the $\boldsymbol{\vartheta}$ of each Bernstein or
ordinal simple intercept. At shift $s = 0$, the node's CDF then equals the
empirical marginal $\hat F$ of its training column. The match is exact for
an ordinal node and a smooth approximation for a continuous node. An
intercept with parents (`I("x1")`) has no single $\boldsymbol{\vartheta}$.
The flag leaves it at the seeded initialization of its network.

**Why logit.** The latent $U$ is standard logistic, so its CDF is the
sigmoid $\sigma$ and its quantile function is $\operatorname{logit}$. At
$s = 0$ the model says $P(X \le x) = \sigma(h(x))$. For this CDF to equal
$\hat F$, the transform must be

$$
h(x) = \operatorname{logit} \hat F(x) = \log \hat F(x) - \log\bigl(1 - \hat F(x)\bigr).
$$

The code evaluates $\log(1 - p)$ as `log1p(-p)`, which stays accurate for
$p$ near 0.

**Ordinal node.** `ordinal_marginal_init_theta` takes the counts of the $K$
classes $0, \dots, K-1$. The model is $P(X \le k) = \sigma(\vartheta_k - s)$,
so the targets are the cumulative class log-odds
$c_k = \operatorname{logit} \hat F(k)$ for $k = 0, \dots, K-2$.
`ordinal_cutpoints` builds increasing cutpoints from unconstrained
parameters: $\vartheta_0 = \tilde\vartheta_0$ and
$\vartheta_k = \vartheta_{k-1} + e^{\tilde\vartheta_k}$. The start inverts
this map. It sets $\tilde\vartheta_0 = c_0$ and
$\tilde\vartheta_k = \log(c_k - c_{k-1})$. At $s = 0$ the class
probabilities are then the empirical frequencies. For counts
`[50, 30, 15, 5]` the start gives the PMF `[0.50, 0.30, 0.15, 0.05]`.

**Continuous node.** `BernsteinUT.marginal_init_theta` takes the column. A
Bernstein polynomial of order $M$ with control points $\vartheta_k$ is close
to the function whose values at $k/M$ are $\vartheta_k$. The transform
pre-scales the calibrated range onto the domain of the polynomial
([model.md](model.md#the-three-knobs-on-a-term)). The point $k/M$ of the
domain then belongs to a data value $x_k$. zuko turns `n_coeffs` parameters
into $M + 1$ control points, with $M$ = `n_coeffs + 1`. Step 3 explains the
two extra points. The start uses the closeness property in three steps:

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
$\operatorname{logit} \hat F$. The approximation smooths $\hat F$, and a
sharp feature such as the dip between two modes smooths most.

The ends of the range carry the empirical quantiles `range_q` and
`1 - range_q`. The end control points are therefore close to
$\operatorname{logit} q$ and $\operatorname{logit}(1 - q)$, with $q$ =
`range_q`. At the default $q = 0.05$ these are $-2.944$ and $2.944$. With
fewer than 3 coefficients, the tied steps use all the parameters and leave
only a straight line. The start then takes the line between these two end
values.

**Guards.** Two limits keep the start finite and increasing:

- The start clips $\hat F$ to $[10^{-3}, 1 - 10^{-3}]$. A control point or
  a cutpoint then stays within about $\pm 6.9$.
- Each step between adjacent control points or cutpoints is at least
  $10^{-3}$, because the softplus and log inverses need strictly increasing
  points. An empty class or a gap in the data therefore gets a small but
  nonzero probability. An empty middle class starts at $2.5 \cdot 10^{-4}$
  or less.

**Where it does not apply.** The spline and affine transforms have no
marginal start, and the flag skips them. A `range_q=0` Bernstein transform
maps the data minimum and maximum onto the range ends. The start ties the
range ends to $\operatorname{logit} q$ and $\operatorname{logit}(1 - q)$,
which are infinite for $q = 0$. So `fit` raises before it reads the column.
It raises for such a transform also in an intercept with parents, which does
not use the start. Fit such a model with `marginal_init=False`.

**What it changes.** At `n_coeffs = 20` the zero start is close to a
straight line too. Its control points span a wider range than the line from
$-2.944$ to $2.944$. A steeper $h$ gives a narrower distribution than
the data, so the zero start has the higher NLL at shift zero.

The likelihood of an all-`LS` model has one optimum, and the fit reaches it
with or without the start. So the start only shortens the way, and
`fit_classical` takes no such flag.

A model with a network shift or a learning-rate anneal can reach a
different local optimum. For such a model the start can improve or worsen
the causal estimate. D4 in
[paper-replication.md](paper-replication.md#d4-the-marginal-start-measured-per-variant)
measures this per variant.

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
| `EarlyStopping(patience=)` | also stop each node once its best epoch is that old |
| a plateau `Callback` around torch's `ReduceLROnPlateau` | a decaying rate per node |

Three details are easy to get wrong.

- A second `fit` call continues training and `history` accumulates. That is
  what makes a multi-phase schedule a loop of `fit` calls.
- A callback instance passed to the flow serves every node in turn. A shipped
  callback resets at each node's fit begin, so its attributes describe the
  last node only. Under `n_jobs` its state stays in the workers. Use a
  function of the node name to keep one instance per node.
- A post-fit `load_state_dict` skips the `VC` re-centering. Restore weights
  from `on_fit_end` instead, as `EarlyStopping` does.

## Path B: classical optimization with `fit_classical`

`fit_classical` is the optimizer for all-`LS` models, where every
node-conditional is an ordered logit or a Colr model. It raises on any `CS`, `CI` or `VC` term.

- **Full-batch, float64, L-BFGS** with a strong-Wolfe line search; no
  minibatches, schedule or early stopping, so the same init gives
  bit-identical results. A converged fit matches `statsmodels` and R to about
  four decimals, where a converged Adam `fit` gets to about 1e-3.
- **Budget.** `max_iter=5000` per node is a default, not a promise. Torch ends
  the run when the NLL or the parameters move by less than 1e-9. A node with
  many parameters can need thousands of iterations to stop on that rule. A
  report that is not `converged` issues a `UserWarning`.
- **Per node.** The flow's `fit_classical` runs `Node.fit_classical` for each
  node. Its report holds `converged` (every node converged), the summed
  `final_nll` and `seconds`. It also holds the `coefficients` as
  `{node: {parent: array}}` and `nodes`, the per-node reports.
- **The report.** Per node, `stop_reason` is `"tolerance"`, `"max_iter"` or
  `"max_eval"` (torch's budget of closure calls, `max_iter * 5 // 4`).
  `converged` needs both: the run stopped on its own AND `grad_norm` is at
  most `tramdag.fitting.GRAD_TOL` (1e-2). Both conditions are necessary, because
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
`history` dicts live in RAM, and whatever a callback records is yours. The
only disk I/O is the explicit `save()` and `load()`.
