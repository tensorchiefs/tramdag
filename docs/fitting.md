# Fitting a TRAM-DAG

This page is the reference for training a
[`CausalFlowDAG`](../src/tramdag/flow.py) and its nodes. It covers the
likelihood, the two fitting paths `fit` and `fit_classical`, and the hooks of
the Adam path. [`notebooks/training_strategies.py`](../notebooks/training_strategies.py)
runs the Adam recipes end to end. It closes with a runtime comparison of the
recipes on one workload. `n_jobs`, `marginal_init`, `validation_split` and
per-node `epochs` or `learning_rate` functions have no notebook example.

## One module, one sub-model per node

A `CausalFlowDAG` is a single `torch.nn.Module` with one independent
sub-model per variable. A sub-model has an intercept that produces the
transform parameters $\theta$ and one shift module per shift term. It also
has the monotone transform $h$, which holds range buffers but no weights.
The nodes share no parameters. The
DAG lives entirely in which parents each node reads. There is no edge weight
matrix and no shared trunk. [code-map.md](code-map.md) says which class
implements which term.

## The likelihood

The flow maps iid standard-logistic latents $U$ to the observed $X$ in causal
order, and node $i$ reads its parents as data. The Jacobian is therefore
triangular, and the joint log-likelihood decomposes per node,
$\log p(x) = \sum_i \log p(x_i \mid \mathrm{pa}(x_i))$.
`Node.row_log_prob` computes one term and `CausalFlowDAG.log_prob` sums them.

- **Continuous node.** Change of variables through the monotone transform:
  $u = h(x;\theta) + s$ and
  $\log p(x \mid \mathrm{pa}) = \log f_{\text{logistic}}(u) + \log |h'(x)|$.
  The transform returns the log-derivative alongside $u$.
- **Ordinal node.** The ordered-logit head of [model.md](model.md#ordinal-nodes),
  evaluated as the log of the cutpoint-interval probability in log space.

Because parents enter as data, the per-node terms have disjoint parameters,
and maximizing each term maximizes the sum. So every node fits on
its own: [`Node.fit`](../src/tramdag/fitting.py) minimizes the node's mean
negative log-likelihood over the batch, plus the node's `VC` penalty.
`CausalFlowDAG.fit` runs `Node.fit` for each node. `log_prob` returns the
per-row joint.

## One node alone

A [`Node`](../src/tramdag/nodes.py) is a TRAM regression of one variable on its
parents, and it needs no DAG. It takes its name, its node spec and a parent
schema, `{parent: "continuous" | n_levels}`.
[`notebooks/training_strategies.py`](../notebooks/training_strategies.py) fits
one with Adam, and
[`notebooks/classical_fit_tram_dag.py`](../notebooks/classical_fit_tram_dag.py)
fits one classically against R `glm`. The frame needs the node's column, its
parents' columns and its side columns; the fit ignores other columns. `fit`,
`fit_classical`, `nll`, `save` and `load` work as on the flow, for one node.
`nll` gives one float, and the `fit_classical` report is the per-node report.
`fit` takes an optimizer instance or a factory `f(node)`, but no `n_jobs`. A
centered `VC` needs its propensity column in every frame, because only the flow
can compute it live from the treatment node. A flow derives each node's
shuffling seed from `fit(seed=)` and the node's index. So `Node.fit(seed=)`
alone does not repeat a node of a seeded flow fit. A node built alone draws its
initial weights from torch's global RNG; call `torch.manual_seed` first for a
reproducible start. It always uses torch's default init; `init=` exists on
`CausalFlowDAG` only.

## Path A: stochastic optimization with `fit`

`fit` is the general-purpose trainer, and the only one for a model with a
`CS`, `CI` or `VC` term. Per node, it is one minibatch Adam loop that keeps
the final weights.

- **Nodes.** `CausalFlowDAG.fit` fits the nodes in topological order. With
  validation data, the nodes with a centered `VC` fit last.
  `n_jobs=N` forks `N` worker processes (Linux only), one task per node.
  A worker runs torch on one thread. With the same `seed=`, it gives the
  weights of a serial fit on one thread. Without `seed=`, `n_jobs > 1` draws
  a base seed from torch's global RNG.
  `epochs=`, `learning_rate=` and `callbacks=` take one value for all
  nodes or a function of the node name.
- **Optimizer.** One optimizer per node, `Adam(lr=learning_rate)` by default.
  `optimizer=` takes a factory `f(node)`, for example to give the networks
  weight decay in their own parameter group. `Node.fit` also takes an
  optimizer instance. `node.parameter_groups()` gives the parameters by kind:
  `"intercept"`, `"linear"` and `"networks"`.
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
- **Validation.** `validation_data=` takes a frame. `validation_split=` takes
  a float and uses the last fraction of `train_df` unshuffled. So shuffle the
  frame first if its row order means anything. Only the head calibrates. The
  flow splits once, so all nodes see the same rows. With either, `fit` writes
  the node's validation NLL after every epoch into `node.history["val"]`.
- **History.** `node.history` and `flow.history` are in the
  [`fit`][tramdag.flow.CausalFlowDAG.fit] docstring. A node that stopped
  earlier repeats its last entry in the flow view. `flow.history_frame()` gives
  one row per node and epoch, with no repeats.
- **Callbacks.** `callbacks=` takes one `Callback` or a list. The hooks get the
  node: `on_fit_begin(node, optimizer)`, `on_epoch_end(node, epoch, optimizer)`
  and `on_fit_end(node, optimizer)`. In `Node.fit` a bare callable is an
  `on_epoch_end` hook; any `True` return stops that node's fit. `on_fit_end`
  runs before the `VC` re-centering. On the flow, a bare callable is a
  function of the node name that gives that node its callbacks. So on the
  flow, a bare hook goes into a list. The shipped callback is
  `EarlyStopping`: it restores the node's best-validation weights and takes an
  optional `patience` and `min_delta`. It reads `node.history["val"]`. It
  writes its best epoch to `node.history["best_epoch"]`, which
  `plot_training(flow, stops=True)` marks. `fit`
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
loss. It sets the $\boldsymbol{\vartheta}$ of each Bernstein or ordinal simple
intercept so that, at shift $s = 0$, the node's CDF equals the empirical
marginal $\hat F$ of its training column. The match is exact for an ordinal
node and a smooth approximation for a continuous node. An intercept with
parents has no single $\boldsymbol{\vartheta}$ and keeps its seeded
initialization.

The latent $U$ is standard logistic, so at $s = 0$ the model says
$P(X \le x) = \sigma(h(x))$. The start therefore targets
$h = \operatorname{logit} \hat F$.

An ordinal node targets the cumulative class log-odds and inverts
`ordinal_cutpoints`
([`ordinal_marginal_init_theta`][tramdag.transforms.ordinal_marginal_init_theta]).

A continuous node sets the Bernstein control points to $\operatorname{logit}
\hat F$ at equally spaced points of the calibrated range. Then it inverts the
parameterization of zuko ([zuko-upstream.md](zuko-upstream.md#3-public-inverse-of-_constrain_theta);
[`BernsteinUT.marginal_init_theta`][tramdag.transforms.BernsteinUT.marginal_init_theta]).
The polynomial smooths $\hat F$, and a sharp feature such as the dip between
two modes smooths most.

- **Guards.** The start clips $\hat F$ away from 0 and 1 and keeps a minimum
  step between adjacent points. The logit and the inverses stay finite and
  increasing. An empty class or a gap in the data therefore gets a small but
  nonzero probability.
- **Where it does not apply.** The spline and affine transforms have no
  marginal start, and the flag skips them. A `range_q=0` Bernstein transform
  puts the range ends at the data extremes, where the logit is infinite. So
  `fit` raises before it reads the column, also for an intercept with parents.
  Fit such a model with `marginal_init=False`.
- **What it changes.** The zero start is close to a straight line with a
  steeper $h$ than the data. It gives a narrower distribution and the higher
  NLL at shift zero. The likelihood of an all-`LS` model has one optimum, so
  the start only shortens the way, and `fit_classical` takes no such flag. A
  model with a network shift or a learning-rate anneal can reach a different
  local optimum. For such a model the start can improve or worsen the causal
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

Three details are easy to get wrong.

- A second `fit` call continues training and `history` accumulates. That is
  what makes a multi-phase schedule a loop of `fit` calls.
- A callback instance passed to the flow serves every node in turn. A shipped
  callback resets at each node's fit begin, so its attributes describe the
  last node only. Use a function of the node name to keep one instance per
  node. Under `n_jobs` the state of every callback stays in the workers; read
  `node.history` instead.
- A post-fit `load_state_dict` skips the `VC` re-centering. Restore weights
  from `on_fit_end` instead, as `EarlyStopping` does.

## Path B: classical optimization with `fit_classical`

`fit_classical` is the optimizer for all-`LS` models, where every
node-conditional is an ordered logit or a transformation model with linear
shifts. It raises on any `CS`, `CI` or `VC` term. The options and the report
are in the [`fit_classical`][tramdag.flow.CausalFlowDAG.fit_classical]
docstring.

- **Full batch, float64, L-BFGS.** There are no minibatches, schedule or early
  stopping, so the same initialization gives bit-identical results. The fit
  restores float32 afterwards, and checkpoints stay float32.
- **`converged` needs two conditions.** The run must stop on its own, and
  `grad_norm` must be at most `tramdag.fitting.GRAD_TOL`. The tolerance rule
  also fires when the line search stalls far from the optimum. Such a run
  stops on `"tolerance"` with a large gradient norm and is not converged.
- **Weakly identified directions never settle.** A Bernstein intercept, rare
  one-hot levels or a flat treatment-effect ridge drift along zero-curvature
  valleys after the likelihood is at its optimum. This is why `tolerance_grad`
  is off and the gradient bound is loose. Correctness comes from the
  comparison with classical software in
  [`notebooks/classical_fit_tram_dag.py`](../notebooks/classical_fit_tram_dag.py).

A converged `fit_classical` leaves the model at the MLE, ready for any
operation. A `fit` call from there stays put. This checks that the classical
solution is the optimum, and it is a warm start. `Node.warm_start` uses it
for `beta0` of a `VC` ([varying-coefficients.md](varying-coefficients.md)).

## Memory and disk

Neither path writes to disk. The parameters, the optimizer state and the
`history` dicts live in RAM, and whatever a callback records is yours. The
only disk access is the explicit `save()` and `load()`. `fit(n_jobs=N)` forks
`N` copy-on-write copies of the flow and the training frame.
