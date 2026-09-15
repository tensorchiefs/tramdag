# Fitting a TRAM-DAG

This page is the reference for training a
[`CausalFlowDAG`](../src/tramdag/flow.py): the likelihood, the two fitting
paths `fit` and `fit_classical`, and the hooks of the Adam path. Every recipe
named here runs end to end in
[`notebooks/training_strategies.py`](../notebooks/training_strategies.py),
which is the one place for the code; the measured time-to-target of each
recipe is in [training-speed.md](training-speed.md).

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
- **Epochs.** `epochs` has no default; [training-speed.md](training-speed.md)
  shows why a fixed budget cannot be right for every workload.
- **Calibration.** The first `fit` calls `calibrate(train_df)`. Every term
  freezes its data-dependent state there, the intercept its `range_q`
  quantiles and each `input_transform=` its statistics. Calibration never
  touches the weights, and a loaded checkpoint is never recalibrated.
- **Marginal start.** `init_marginals(train_df)` is a separate, always
  explicit step. It resets every Bernstein or ordinal simple intercept to the
  empirical marginal of its column, as `logit(F_hat)` in control points or
  cutpoints. The spline, the affine and the `range_q=0` transforms have no
  such start. The call is a pure initialization and leaves the MLE unchanged.
- **Validation.** `validation_data=` takes a frame; `validation_split=` takes
  a float and uses the last fraction of `train_df` unshuffled, so shuffle the
  frame first if its row order means anything; only the head calibrates. With
  either, `fit` writes the per-node validation NLL after
  every epoch into `flow.history["val"]`.
- **Logging.** `flow.history["lr"]` records the optimizer's rate per epoch, a
  `{node: lr}` dict under `per_node_adam`. `verbose=N` prints every Nth epoch
  and the last one; the default 0 is silent.
- **Callbacks.** `callbacks=` takes one `Callback` or a list. The hooks are
  `on_fit_begin`, `on_epoch_end` and `on_fit_end`; a bare callable is an
  `on_epoch_end` hook `cb(flow, epoch, optimizer)`, and any `True` return
  stops the fit. `on_fit_end` runs before the `VC` re-centering. The shipped
  callbacks are `EarlyStopping`, which restores the best-validation weights
  and takes an optional `patience`, and `PerNodePlateau` with `per_node_adam`
  for per-node decay and freezing. All of them read `history["val"]`.
- **Centered `VC` propensities** ride the training frame as the column that
  `VC(center=)` names, and split and minibatch with it.
  [varying-coefficients.md](varying-coefficients.md) is the guide.

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
| `PerNodePlateau(patience=, freeze=)` with `per_node_adam` | nodes converge at different speeds; self-stopping |

Two details are easy to get wrong.

- A second `fit` call continues training and `history` accumulates. That is
  what makes a multi-phase schedule a loop of `fit` calls.
- A post-fit `load_state_dict` skips the `VC` re-centering. Restore weights
  from `on_fit_end` instead, as `EarlyStopping` does.

## Path B: classical optimization with `fit_classical`

`fit_classical` is the optimizer for all-`LS` models, where every
node-conditional is a classical transformation model, an ordered logit or a
Colr model. It raises on any `CS`, `CI` or `VC` term.

- **Full-batch, float64, L-BFGS** with a strong-Wolfe line search; no
  minibatches, schedule or early stopping, so the same init gives
  bit-identical results, and the fit lands on the exact MLE. It matches
  `statsmodels` and R to about four decimals, where a converged Adam `fit`
  gets to about 1e-3.
- **Budget.** `max_iter=400`, `history_size=50`. Torch's own stopping rule
  ends the run when the NLL or the parameters move by less than 1e-9. The
  report's `converged` says whether a tolerance ended the run rather than the
  cap.
- **float64 is transient.** The fit restores float32 afterwards, and
  checkpoints stay float32.
- **The flag is advisory.** A Bernstein intercept and weakly identified
  directions, rare one-hot levels or a flat treatment-effect ridge, drift
  along zero-curvature valleys after the likelihood is at its optimum.
  Correctness comes from the comparison with classical software in
  [`notebooks/classical_fit_tram_dag.py`](../notebooks/classical_fit_tram_dag.py)
  and `experiments/misc/validate_ls.py`.

`fit_classical` leaves the model at the MLE, ready for any operation. A
`fit` call from there stays put, which is both a check that the classical
solution is the optimum and a warm start; the `VC` guide uses it for `beta0`.

## Memory and disk

Neither path writes to disk. The parameters, the optimizer state and the
`history` dict live in RAM, and whatever a callback records is yours. The
only disk I/O is the explicit `save()` and `load()`.
