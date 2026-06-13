# Fitting a TRAM-DAG: how training works

Technical reference for the two fitting paths in
[`CausalFlowDAG`](../src/tramdag/flow.py) — the stochastic optimizer
([`fit`](../src/tramdag/flow.py)) and the classical one
([`fit_classical`](../src/tramdag/flow.py)) — plus the likelihood they both
optimize, the memory model, and where optimizer choice could go next.

## The objective: a per-node-decomposable likelihood

A TRAM-DAG is one triangular flow from iid standard-logistic latents to the
observed variables. Its joint negative log-likelihood **decomposes per node**,
because each node-conditional depends only on that node's own parameters (parents
enter as observed *data*, never through another node's parameters):

```
log p(x) = Σ_i log p(x_i | pa(x_i))
```

This is computed by
[`CausalFlowDAG.node_log_prob`](../src/tramdag/flow.py), which returns one
log-likelihood term per node, and summed by `log_prob`. Per node:

- **continuous**: `z = h(x; θ) + Σ shifts`, scored against the standard-logistic
  density. `h` is the monotone transform —
  [`BernsteinUT` / `SplineUT` / `AffineUT`](../src/tramdag/transforms.py).
- **ordinal**: an ordered-logit head,
  [`ordinal_log_prob`](../src/tramdag/transforms.py), computed in log-space for
  numerical stability.

The shift terms come from the per-edge conditioners in
[`conditioners.py`](../src/tramdag/conditioners.py): `LinearShift` (`ls`),
`ComplexShift` (`cs`), `ComplexIntercept` (`ci`).

**Consequence used by both optimizers:** the per-node gradients are independent,
so optimizing the summed loss jointly is identical to optimizing each node
separately. This is what licenses per-node learning rates, per-node freezing
(stochastic path), and the all-`ls` classical fit.

## Path A — stochastic optimization (`fit`)

[`CausalFlowDAG.fit`](../src/tramdag/flow.py) is the general-purpose trainer
(required for any `cs`/`ci` edge). Mechanics:

- **Adam** with **one parameter group per node** — built from
  `self.nodes[name].parameters()`, so each node can carry its own learning rate.
- **Minibatches**: a fresh `torch.randperm` shuffle each epoch; the loss is the
  summed per-node mean NLL on the batch.
- **Transform ranges** are set once at the start from the train 5%/95% quantiles
  (`_set_ranges`), defining each Bernstein/spline domain.
- **Learning-rate schedules** (`schedule=`): `None` (constant), `"onecycle"`,
  `"cosine"`, or `"plateau"` — the last decays each node's lr off *its own*
  validation NLL.
- **Per-node freezing** (`freeze_patience=`): a node whose validation NLL has
  stalled is dropped from the loss and backward pass (a real compute saving,
  valid because of the independence above); when all nodes freeze the fit stops.
- **`restore_best`**: optionally snapshots each node's best-validation weights and
  restores them at the end (early-stopping regularization). Default `False` so the
  fit sits at the training-data MLE.

Benchmarks, schedule trade-offs, and the recommended self-stopping recipe are in
[training-speed.md](training-speed.md); the worked walkthrough is
[`notebooks/intro_tram_dag.py`](../notebooks/intro_tram_dag.py).

## Path B — classical optimization (`fit_classical`)

[`CausalFlowDAG.fit_classical`](../src/tramdag/flow.py) is the dedicated optimizer
for **all-`ls`** models, where every node-conditional is a classical
transformation model (ordered-logit / Colr). It raises on any `cs`/`ci` edge.

- **Full-batch, float64, L-BFGS** (strong-Wolfe line search) — no minibatches, no
  schedule, no early stopping ⇒ **deterministic** (same init → bit-identical) and
  on the **exact MLE** (matches `statsmodels`/R to ~1e-3 on well-identified
  coefficients).
- **float64 is a transient compute mode**: `self.double()` upcasts parameters
  *and* the transforms' range buffers in one call; the fit runs in double; a
  `finally: self.float()` restores float32. The stored model — and
  `save`/`load` — stay float32. The data path (`_tensorize`, `sample`, `pmf`)
  reads the model dtype so it follows along.
- **Convergence** is judged by NLL flatness and is *advisory*: a Bernstein
  intercept and weakly-identified directions (rare one-hot levels, a flat
  treatment-effect ridge) keep drifting along zero-curvature valleys after the
  likelihood is at the optimum. Correctness is verified against classical software
  (`experiments/validate_ls.py --classical`), not the flag.
- Read the fitted coefficients with `ls_coefficients()`.

Walkthrough: [`notebooks/classical_fit_tram_dag.py`](../notebooks/classical_fit_tram_dag.py).

## Memory and disk during fitting

**Neither fitting path writes to disk.** Everything during `fit`/`fit_classical`
lives in RAM:

- model parameters and (for `fit`) the per-node Adam optimizer state;
- the `history` dict (per-epoch train/val NLL, lr, wall-clock) — an in-memory
  attribute, not a file;
- `restore_best` snapshots — `copy.deepcopy` of node `state_dict`s held in memory
  (`flow.py`), never serialized.

The **only** disk I/O in the module is the explicit, user-called
[`save`](../src/tramdag/flow.py) (`torch.save` of spec + state_dict + history) and
its counterpart `load`. So a fit produces no temp files, no checkpoints, no
scratch directory; persistence is opt-in via `flow.save(path)`. (The `results/`
and `docs/perf/` artifacts in this repo are written by the *experiment scripts*,
not by the library's fitting code.)

## Optimizer choice — current and future

Today: **Adam** for flexible models, **L-BFGS** (float64) for all-`ls`. The
per-node-param-group structure already in `fit` makes several extensions cheap,
and worth considering as the package matures:

- **IRLS / Fisher scoring for the all-`ls` path.** The classical fitting algorithm
  for proportional-odds / GLM-type models is iteratively reweighted least squares
  (Newton with the expected information). For the `ls` case it would likely reach
  the MLE in *fewer, more stable* steps than L-BFGS — and the Fisher information it
  computes is exactly the ingredient for the planned **standard-error table**
  (currently flagged as next in the CHANGELOG). Strong candidate to back
  `fit_classical` in a future version.
- **Second-order / curvature-aware methods for flexible nodes.** K-FAC or a
  Gauss-Newton approximation could help the `ci`/`cs` MLPs, though full-batch
  quasi-Newton is a poor fit for them (minibatch noise also regularizes — see why
  `fit_classical` refuses them).
- **Modern first-order variants.** AdamW (decoupled weight decay), RAdam (warmup-
  free), or Lion/Sophia are drop-in alternatives to Adam; the benchmark harness
  ([`experiments/bench_training.py`](../experiments/bench_training.py)) is built to
  evaluate exactly such swaps on time-to-target.
- **Per-node optimizer selection.** Because the loss decomposes and the optimizer
  already holds one group per node, different nodes could in principle use
  different optimizers (e.g. L-BFGS for the `ls` nodes, Adam for an MLP node in a
  mixed model) — an interesting direction for mixed-flexibility DAGs.
- **Warm-start handoffs.** `fit_classical → fit` already works (the classical MLE
  as a fast, principled initialization); the reverse (Adam to escape, L-BFGS to
  polish) is a natural pattern to formalize.

These are *directions*, not commitments. The autoresearch loop
([`docs/research/MISSION_autoresearch.md`](research/MISSION_autoresearch.md)) is a
good venue to test the speed-oriented ones empirically before adopting any.
