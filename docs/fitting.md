# Fitting a TRAM-DAG: how training works

This document is the technical reference for training a
[`CausalFlowDAG`](../src/tramdag/flow.py). It covers three subjects:

- the likelihood
- the two fitting paths, `fit` and `fit_classical`
- the hooks of these two paths

## How the flow is built: one module, one sub-model per node

A `CausalFlowDAG` is a single `torch.nn.Module`. It holds **one independent
sub-model per variable**. Each sub-model has these parts:

- an intercept, which produces the transform parameters `θ`
- the monotone 1-D transform `h`, which has no learnable weights of its own and
  only range buffers
- one shift module per shift term

The nodes share **no parameters**. One module bundles them, and one optimizer
trains them. The DAG structure lives entirely in *which parents each node
reads*. There is no edge weight matrix and no shared trunk.

The [code map](code-map.md) says which class implements which term. Continuous
parent features enter raw. Ordinal parent features enter as one-hot columns.

## How the likelihood is computed

A TRAM-DAG maps iid standard-logistic latents `U` to the observed `X` in causal
order. Node `i` reads only its parents (earlier variables, as *data*). Therefore
the Jacobian of `U → X` is **triangular**, and its log-determinant is the sum of
the per-node 1-D terms. The joint log-likelihood therefore **decomposes per node**:

```
log p(x) = Σ_i log p(x_i | pa(x_i))
```

[`CausalFlowDAG.node_log_prob`](../src/tramdag/flow.py) computes one term per node
and `log_prob` sums them. For a node, given `θ, shift` from `theta_shift`:

- **continuous** — change of variables through the monotone transform:

  ```
  u = h(x; θ) + shift
  log p(x | pa) = log f_logistic(z) + log |dz/dx|
  ```

  That is, the term is the standard-logistic density at the latent `u`
  ([`StandardLogistic.log_prob`](../src/tramdag/transforms.py)) plus the
  transform's log-derivative. `ut.forward` returns this log-derivative as `ladj`.
  This is the 1-D Jacobian term that makes the result a proper density, not only
  a score.
- **ordinal** — an ordered-logit / proportional-odds head,
  `P(x ≤ k) = σ(θ_k − shift)`. [`ordinal_log_prob`](../src/tramdag/transforms.py)
  evaluates it as the log of the cutpoint-interval probability. The computation
  runs in log-space via `logsigmoid`/`log1mexp`, because the naive sigmoid
  difference underflows to exactly-zero gradients in float32.

The training loss is the summed per-node **mean** NLL over the batch
(`Σ_i mean_rows(−log p(x_i | pa))`). In contrast, `log_prob` returns the per-row
joint, which scores whole observations.

**Consequence used by both optimizers:** because parents enter as data, the
per-node gradients are independent. Therefore a joint fit of the summed loss is
identical to a separate fit of each node. This independence licenses three
things:

- per-node learning rates
- freezing, which a callback does (see below)
- the all-`ls` classical fit

## Path A — stochastic optimization (`fit`)

[`CausalFlowDAG.fit`](../src/tramdag/flow.py) is the general-purpose trainer. Any
`cs`/`ci` edge requires it. Mechanics:

- **One optimizer over all parameters**. The default is
  `Adam(lr=learning_rate)`. You can pass any `torch.optim.Optimizer` as
  `optimizer=` instead. This is exactly per-node training, as the consequence
  above explains. For per-node rates, `per_node_adam` builds the per-node
  parameter groups.
- **Minibatches**: a fresh `torch.randperm` shuffle each epoch (`seed=` seeds
  it). The loss is the summed per-node mean NLL on the batch, plus the `VC`
  penalty.
- **`calibrate(train_df)`** — the first `fit` calls it. Every term then
  freezes its own data-dependent state. The intercept maps the train
  `range_q`/`1-range_q` quantiles onto its transform's domain. Every
  term-level `input_transform=` freezes its statistics. Calibration never
  touches the weights. A checkpoint carries the flag, so no `fit` ever
  recalibrates a loaded model.
- **`flow.init_marginals(train_df)`** — the calibrated start is a separate
  step, and it is always explicit. It resets every Bernstein or ordinal
  simple intercept to the empirical marginal of its column. That value is
  `logit(F_hat)` in both cases, as control points or as cutpoints. The
  spline, the affine and the `range_q=0` transforms have no such start. The
  call is a pure init that leaves the MLE unchanged. You can call it at any
  time, including on a trained or a loaded flow.
- **Validation, Keras-shaped** — two arguments turn validation on.
  `validation_data=` takes a DataFrame. `validation_split=` takes a float, and
  it uses the LAST fraction of `train_df` with no shuffle. Only the head of
  the frame calibrates, so there is no leakage. With either argument, `fit`
  computes the per-node validation NLL once after every epoch, into
  `flow.history["val"]`. `validation_batch_size=` chunks that pass.
- **Logging** — the shipped callbacks read `flow.history["val"]` there.
  `flow.history["lr"]` records the optimizer's rate after every epoch. With
  `per_node_adam` that record is a `{node: lr}` dict. A schedule's decisions
  are therefore on record, and you need no callback of your own. `verbose=N`
  prints every Nth epoch plus the final one. The default is 0, which is
  silent.
- **`callbacks=`** — it takes one entry or a list. A
  [`tramdag.callbacks.Callback`](../src/tramdag/callbacks.py) hooks
  `on_fit_begin`, `on_epoch_end` and `on_fit_end`. Its docstring is the
  contract for the timing, the stop rule and the VC re-centering order. A
  bare callable in the list is an `on_epoch_end` hook with the signature
  `cb(flow, epoch, optimizer)`. Any `True` return stops the fit. Schedules,
  snapshots and coefficient trajectories live here.
- **The common recipes ship in `tramdag.callbacks`.** `EarlyStopping`
  restores the best-validation weights automatically, and it takes an
  optional patience. `PerNodePlateau` plus `per_node_adam` give per-node
  decay and freezing. All of them read `history["val"]`.
- **Centered `VC` propensities are a column.** `VC(center="ps")` names the
  column of the training frame that holds the out-of-fold `P(t=1|pa_t)` per
  row. That column splits and minibatches with the frame. See
  [varying-coefficients.md](varying-coefficients.md).

### Training strategies

Every strategy below is `fit` plus a callback, or `fit` plus a few lines of
your own. Pick a strategy by model class. Each strategy has a copy-paste
example. The examples assume a built `flow = CausalFlowDAG(spec)` and the
pandas frames `train_df`/`val_df`.

The empirical rule of thumb has two halves. **All-`ls` models train to the MLE
and keep the final weights. Flexible models (CI, CS or VC) validate and keep
the best weights.** Flexible models overfit observational confounding at the
MLE, as the finding below shows.

| Strategy | When |
|-----------------------------|-----------------------------------------------------------------------------------|
| exact MLE — `fit_classical` (Path B, below) | all-`ls` spec; deterministic, seconds |
| plain Adam | all-`ls` with a shift `fit_classical` refuses, quick looks |
| multi-phase Adam | a tighter MLE without a scheduler |
| best-validation weights — `EarlyStopping()` | any CI/CS/VC model; the recommended recipe (register it — fit has no default) |
| … + patience — `EarlyStopping(patience=)` | also stop once the best is that many epochs old |
| global plateau schedule | decaying one shared rate beats picking one |
| per-node plateau — `PerNodePlateau` | nodes converge at different speeds; self-stopping |

The code below gives one line for each Adam recipe, as a quick reference. The
exact-MLE path is Path B below, and the global plateau rule needs a `Callback`
of your own. Every Adam strategy runs end to end, with its output and its
checks, in
[`notebooks/training_strategies.py`](../notebooks/training_strategies.py).
Every documentation build executes that notebook, so the notebook is the
source of truth. If a snippet here disagrees with the notebook, the notebook
is right.

```python
# plain Adam: one rate, one loop, the final weights
flow.fit(train_df, epochs=500, learning_rate=1e-3, batch_size=256, verbose=100)

# two phases: a second call continues training, so this is a schedule
for epochs, lr in [(800, 1e-2), (700, 1e-3), (500, 1e-4)]:
    flow.fit(train_df, epochs=epochs, learning_rate=lr)

# best-validation weights: the flexible-model recipe, restored at fit end
flow.fit(train_df, epochs=4000, validation_data=val_df, callbacks=EarlyStopping())

# ... and stop once the best epoch is that old
flow.fit(
    train_df, epochs=4000, validation_split=0.1, callbacks=EarlyStopping(patience=200)
)

# one rate per node, each freezing on its own score; stops when all froze
flow.fit(
    train_df,
    epochs=4000,
    validation_split=0.1,
    optimizer=per_node_adam(flow, lr=1e-2),
    callbacks=PerNodePlateau(),
)  # patience=15, freeze=50
```

Import `EarlyStopping`, `PerNodePlateau` and `per_node_adam` from
`tramdag.callbacks`. For one shared rate instead of per-node rates, carry
torch's `ReduceLROnPlateau` in a `Callback` of your own. The notebook's
`GlobalPlateau` is that class, ready to copy.

Two measured facts decide between the first two. The multi-phase recipe lands
within ~1e-5 of statsmodels on an all-`ls` model. A single converged
constant-rate run gets ~1e-3. If you need the tighter agreement and the spec
is all-`ls`, use `fit_classical` instead, which is exact and faster.

`experiments/benchmarks/bench_training.py` measures the per-node recipe.

Three details are easy to get wrong:

- `validation_split` takes the **last** rows, unshuffled. If the row order of
  the frame means anything, shuffle the frame first.
- A second `fit` call **continues** training, and `history` accumulates across
  the calls. The call does not start over.
- A post-fit `load_state_dict` skips the varying-coefficient re-centering. On
  a spec with a `VC` term, restore the weights from the `on_fit_end` hook of
  a `Callback` subclass instead. That hook runs before the re-centering step.
  `EarlyStopping` already does this.

The exact-MLE and warm-start strategies are Path B, below.
[training-speed.md](training-speed.md) gives the measured time-to-target for
each recipe.

## Path B — classical optimization (`fit_classical`)

[`CausalFlowDAG.fit_classical`](../src/tramdag/flow.py) is the dedicated
optimizer for **all-`ls`** models. In such a model, every node-conditional is a
classical transformation model, that is an ordered-logit or a Colr model. It
raises on any `cs`, `ci` or `vc` term.

- **Full-batch, float64, L-BFGS** with a strong-Wolfe line search. There are
  no minibatches, no schedule and no early stopping. Therefore the fit is
  **deterministic**: the same init gives bit-identical results. The fit also
  lands on the **exact MLE**. `fit_classical` matches `statsmodels` and R to
  ~4 decimals. A converged Adam `fit` gets within ~1e-3.
- **Solver budget** — `max_iter=400`, `tol=1e-9` and `history_size=50`. The
  fit is one L-BFGS run with torch's own stopping rule. The run ends when the
  NLL or the parameters move by less than `tol`, or at `max_iter`. The
  report's `n_iter` is torch's count. The report's `converged` says whether a
  tolerance ended the run, and not the cap. `history_size` is the L-BFGS
  memory.
- **float64 is a transient compute mode.** The fit runs in double and
  restores float32 afterwards. Checkpoints stay float32.
- **Convergence** — the flag is true when torch's `tolerance_change` ended
  the run before `max_iter` did. The flag is *advisory*. A Bernstein intercept
  and weakly-identified directions continue to drift along zero-curvature
  valleys after the likelihood is at the optimum. Rare one-hot levels and a
  flat treatment-effect ridge are two such directions. Correctness comes from
  a comparison with classical software, and not from the flag. That comparison
  is `python -m misc.validate_ls classical`.
- Read the fitted coefficients with `ls_coefficients()`.

### Warm-start handoff: classical fit, then keep training

`fit_classical` leaves the model at the MLE in float32, ready for any normal
operation. A `fit()` call from there **stays put**. This behaviour is a check
that the classical solution really is the optimum. It is also a way to use the
classical fit as a fast, principled initialization:

```python
flow.fit_classical(train_df)  # exact MLE, seconds
before = flow.ls_coefficients()["y"]
flow.fit(train_df, epochs=300, learning_rate=1e-3)  # a gentle Adam phase ...
after = flow.ls_coefficients()["y"]  # ... barely moves
```

A small drift means that the classical fit was already at the optimum. The
same handoff warm-starts the `beta0` of a `VC` term.
[varying-coefficients.md](varying-coefficients.md) holds the measured recipe.

## Memory and disk during fitting

Neither fitting path writes to disk. Three things live in RAM:

- the parameters
- the optimizer state
- the `history` dict

Whatever a callback records is yours. The only disk I/O in the module is the
explicit `save()`/`load()`. The `results/` artifacts in this repository come
from the experiment scripts, and not from the library.

## Optimizer choice

Today the package uses **Adam** for flexible models. It uses **L-BFGS** in
float64 for all-`ls` models. The per-node decomposition makes optimizer swaps
cheap through `optimizer=`. Three candidates wait for a measurement:

- IRLS for the `ls` path
- per-node mixing
- modern Adam variants

`experiments/benchmarks/bench_training.py` benchmarks every candidate before
the package adopts it.
