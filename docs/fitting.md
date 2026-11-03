# Fitting a TRAM-DAG

This page explains how to train a
[`CausalFlowDAG`][tramdag.flow.CausalFlowDAG] and its nodes. It covers the
likelihood, the Adam paths `fit` and `fit_jointly`, the classical path
`fit_classical`, and which recipe to use. The API pages hold the signatures
and defaults.
[`notebooks/training_strategies.py`](../notebooks/training_strategies.py)
runs the Adam recipes end to end and compares their runtime.
[`notebooks/fit_strategies_benchmark.py`](../notebooks/fit_strategies_benchmark.py)
compares `fit`, `fit` with `n_jobs` and `fit_jointly`.

## One module, one sub-model per node

A `CausalFlowDAG` is a single `torch.nn.Module` with one independent
sub-model per variable. A sub-model has an intercept that produces the
transform parameters $\theta$ and one shift module per shift term. It also has
the monotone transform $h$, which holds range buffers but no weights. The nodes
share no parameters. The DAG lives entirely in which parents each node reads:
there is no edge weight matrix and no shared trunk.

## The likelihood

The flow maps iid standard-logistic latents $U$ to the observed $X$ in causal
order, and node $i$ reads its parents as data. The Jacobian is therefore
triangular, and the joint log-likelihood decomposes per node,
$\log p(x) = \sum_i \log p(x_i \mid \mathrm{pa}(x_i))$.

- **Continuous node.** Change of variables through the monotone transform:
  $u = h(x;\theta) + s$ and
  $\log p(x \mid \mathrm{pa}) = \log f_{\text{logistic}}(u) + \log |h'(x)|$.
- **Ordinal node.** The ordered-logit head of
  [model.md](model.md#ordinal-nodes), the log of the cutpoint-interval
  probability.

Because parents enter as data, the per-node terms have disjoint parameters,
and maximizing each term maximizes the sum. So every node can fit on its own:
`fit` minimizes each node's mean negative log-likelihood, plus the node's `VC`
penalty, one node after the other. `fit_jointly` minimizes the sum of the
same losses in one loop.

## One node alone

A [`Node`][tramdag.nodes.Node] is a TRAM regression of one variable on its
parents, and it needs no DAG. It fits, scores and saves like the flow, for
one node. [`notebooks/training_strategies.py`](../notebooks/training_strategies.py)
fits one with Adam, and
[`notebooks/classical_fit_tram_dag.py`](../notebooks/classical_fit_tram_dag.py)
fits one classically against R `glm`. A lone node built with `seed=` starts from
the weights it has in a flow with that seed and the default `init`
([code-map.md](code-map.md#flowpy-the-model)). A fit with the flow's fit
`seed=` then gives the weights of that node in the flow.

## Path A: stochastic optimization with `fit`

`fit` is the general-purpose trainer. With `fit_jointly`, it is one of the
two Adam paths, the only trainers for a model with a `CS`, `CI` or `VC` term
(`fit_classical` is not). Per node, it is one minibatch Adam loop that keeps
the final weights unless a callback restores others. The
[`fit`][tramdag.flow.CausalFlowDAG.fit] docstring lists every option.

- **Nodes.** The nodes fit in topological order, or in parallel worker
  processes with `n_jobs`.
- **Epochs.** `epochs` has no default. A fixed budget under-spends on one
  workload and wastes on the next, so every caller states its own.
- **Shuffle.** Every node shuffles its rows from the fit's `seed=`. So a
  node fits as `Node.fit` with that seed, and `fit_jointly` sees the same
  row order.
- **Calibration.** The first fit calibrates each node on the training frame.
  Every term freezes its data-dependent state there: the intercept its
  `range_q` quantiles and each `input_transform=` its statistics. A loaded
  checkpoint is never recalibrated.
- **Validation.** With `validation_data=` or `validation_split=`, `fit`
  records each node's validation NLL after every epoch. `validation_split`
  takes the last rows unshuffled, so shuffle the frame first if its row order
  means anything.
- **Callbacks.** A callback gets the node at
  fit begin, after every epoch and at fit end. A `True` return after an epoch
  stops that node. The shipped callback is
  [`EarlyStopping`][tramdag.callbacks.EarlyStopping], which restores the
  best-validation weights. A learning-rate scheduler is a few lines of
  callback around torch's `lr_scheduler`.
- **Centered `VC` propensities** ride the training frame as a column
  ([varying-coefficients.md](varying-coefficients.md)).

### The marginal start

`fit(marginal_init=True)` puts the optimizer near the data before the first
step. It changes the parameters the optimizer starts from, not the model or
the loss. It sets the $\boldsymbol{\vartheta}$ of each Bernstein or ordinal
simple intercept so that, at shift $s = 0$, the node's CDF equals the
empirical marginal $\hat F$ of its training column. The match is exact for an
ordinal node and a smooth approximation for a continuous node. An intercept
with parents has no single $\boldsymbol{\vartheta}$ and keeps its seeded
initialization. Without the flag, every simple intercept starts with all its
parameters at zero.

The latent $U$ is standard logistic, so at $s = 0$ the model says
$P(X \le x) = \sigma(h(x))$. The start therefore targets
$h = \operatorname{logit} \hat F$.

- An ordinal node targets the cumulative class log-odds and inverts
  `ordinal_cutpoints`
  ([`ordinal_marginal_init_theta`][tramdag.transforms.ordinal_marginal_init_theta]).
- A continuous node sets the Bernstein control points to
  $\operatorname{logit} \hat F$ at equally spaced points of the calibrated
  range. Then it inverts the zuko parameterization
  ([`BernsteinUT.marginal_init_theta`][tramdag.transforms.BernsteinUT.marginal_init_theta]).
  The polynomial smooths $\hat F$, and a sharp feature such as the dip between
  two modes smooths most.

The spline and affine transforms have no marginal start, and the flag skips
them. A `range_q=0` Bernstein transform puts the range ends at the data
extremes, where the logit is infinite, so `fit` refuses the flag there.

The zero start is close to a straight line with a steeper $h$ than the data.
It gives a narrower distribution and a higher NLL at shift zero. The
likelihood of an all-`LS` model has one optimum, so the start only shortens
the way. A model with a network shift or a learning-rate anneal can reach a
different local optimum. There the start can improve or worsen the causal
estimate. D4 in
[paper-replication.md](paper-replication.md#d4-the-marginal-start-measured-per-variant)
measures this per variant.

### Which recipe

All-`LS` models train to the MLE and keep the final weights. Flexible models
with a `CI`, `CS` or `VC` term overfit observational confounding at the MLE
and need the best-validation weights to recover the causal effect.

| recipe | when |
|---|---|
| `fit_classical` | all-`LS` spec: exact, deterministic, seconds |
| plain Adam | an all-`LS` spec whose `fit_classical` report is not converged, quick looks |
| multi-phase Adam, one `fit` call per rate | a tighter MLE without a scheduler |
| `EarlyStopping()` | any `CI`, `CS` or `VC` model; register it, `fit` has no default |
| `EarlyStopping(patience=)` | also stop each node once its best epoch is that old |
| `EarlyStopping(min_epochs=, smoothing=)` | ignore epochs before `min_epochs`; judge the best epoch on a moving average of the validation NLL, for small validation sets |
| a plateau `Callback` around torch's `ReduceLROnPlateau` | a decaying rate per node |
| `fit_jointly` | one schedule or stop rule on the summed NLL, as in the reference implementation |

Three details are easy to get wrong.

- A second `fit` call continues training and `history` accumulates. That is
  what makes a multi-phase schedule a loop of `fit` calls.
- A callback instance passed to the flow's `fit` serves every node in turn,
  so its attributes describe the last node only. Use a function of the node
  name to keep one instance per node. Under `n_jobs` the callbacks stay in
  the workers; read `node.history` instead.
- A post-fit `load_state_dict` skips the `VC` re-centering. Restore weights
  from `on_fit_end` instead, as `EarlyStopping` does.

## Path B: classical optimization with `fit_classical`

`fit_classical` is the optimizer for all-`LS` models, where every
node-conditional is an ordered logit or a transformation model with linear
shifts. The [`fit_classical`][tramdag.flow.CausalFlowDAG.fit_classical]
docstring holds the options and the report.

- **Full batch, float64, L-BFGS, then Newton.** There are no minibatches or
  schedules, so the same initialization gives bit-identical results. L-BFGS
  stops short of the optimum on an ill-conditioned NLL. One cause is a parent
  on a large raw scale next to a rare level. Newton steps on the exact
  Hessian then finish the fit.
- **The model stays float32.** Read-outs use the fitted values rounded by
  about 1e-7 relative. Compare the report's float64 `coefficients` with other
  software.
- **Unidentified directions do not move.** The
  [`fit_classical`][tramdag.nodes.Node.fit_classical] docstring names them and
  says how to compare them with other software.

A converged `fit_classical` leaves the model at the MLE. A `fit` call from
there stays put, which makes it a warm start. `Node.warm_start` uses it for
the `beta0` of a `VC` ([varying-coefficients.md](varying-coefficients.md)).

## Memory and disk

Neither path writes to disk. The parameters, the optimizer state and the
`history` dicts live in RAM. The only disk access is the explicit `save()`
and `load()`. `fit(n_jobs=N)` forks `N` copy-on-write copies of the flow and
the training frame.
