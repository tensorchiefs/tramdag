# How fast can a CausalFlowDAG train?

This document benchmarks five things:

- learning-rate schedules
- per-node freezing
- batch sizes
- devices
- LBFGS

The benchmark ran on an Apple-silicon Mac mini with torch 2.12, on the CPU
unless a row notes another device.

To reproduce the benchmark, run
[the training benchmark](../experiments/benchmarks/bench_training.py).
The command is `cd experiments && uv run python -m benchmarks.bench_training`.
For one seed on cpu, add `--quick`. The full grid takes ≈ 35 min.

For a quick cross-machine comparison there is
[`experiments/benchmarks/perf_machine.py`](../experiments/benchmarks/perf_machine.py).

## The recipes

The recipes this benchmark compares are training strategies, not parts of
the model. [fitting.md](fitting.md#which-recipe) says what each one is, and
[`notebooks/training_strategies.py`](../notebooks/training_strategies.py)
runs each of them. This page is only the measurement.

The plateau-and-freeze rows are `PerNodePlateau`, with `min_delta=1e-5` on
the stroke workload; the global plateau rule is torch's `ReduceLROnPlateau`
on the summed validation NLL. The L-BFGS rows are a float32 full-batch
variant; `fit_classical` runs in float64, which removes the seed fragility of
finding 2.

## Method: time-to-target, not loss-go-down

Each config runs once. The benchmark's callback records the per-epoch
validation NLL *and* the wall-clock time. From these records, we measure the
seconds until the fit is within a fixed gap of a cached long-run reference. The
reference is the median over 3 torch seeds of a long run. The two workloads and
their tolerances are:

| workload | model / data | reference NLL | tight tol | practical tol |
|---|---|---|---|---|
| **stroke-ls** | all-`ls` 5-node DAG, frozen `experiments/misc/data/magic-mrclean/ls` (n=1275, full-data MLE) | 10.3042 (train) | +1e-3 | +5e-3 |
| **vaca-ci** | all-`ci` flow, frozen `experiments/paper/data/vaca` (n=5000, 90/10 split) | 4.9632 (val) | +2e-3 | +1e-2 |

*Tight* ≈ exact-MLE equivalence, which is a statsmodels/R-polr match.
*Practical* ≈ coefficient-equivalent. A fit with gap ≈ 3e-3 already matches the
R reference coefficients within the tolerances of
[`experiments/misc/validate_ls.py`](../experiments/misc/validate_ls.py).

## Results

![stroke-ls convergence](img/nll_vs_time_stroke-ls.png)
![vaca-ci convergence](img/nll_vs_time_vaca-ci.png)

The table shows the median seconds to target, at batch 512 on cpu. A "—" entry
means that the fit never reached the target within the budget.

| config | stroke-ls practical | stroke-ls tight | vaca-ci practical | vaca-ci tight | self-stops |
|---|---|---|---|---|---|
| baseline two-phase | 9.0 | **21.4** | 2.1 | 2.8¹ | no (runs 40 s / 15 s) |
| constant 1e-2 | 9.1 | 21.5² | 2.2 | 2.8¹ | no |
| **plateau + freeze** | **8.9** | — (gap 2e-3) | **2.0** | 2.9 | **yes — 13 s / 4 s total** |
| LBFGS (full-batch) | **1.6** (2/3 seeds) | — (gap 4–8e-3) | n/a | n/a | yes |

¹ transient: the val-NLL curve dips through the target and then drifts away.
Stroke needs the 1e-3 phase to *stay*. Vaca shows mild overfitting. Final gap
for the vaca baseline is 0.037. A 520-epoch budget **underfits** vaca by ~0.03
nats. Plateau+freeze *stays* at its target.
² constant lr at batch 512 stalls at gap 3–7e-3. Only the lr-decay phase
closes the last decade. This is why the two-phase recipe pairs a constant
phase with a decay phase.

## Findings

1. **Per-node plateau decay + freezing is the best default-style trainer.** It
   reaches the same time-to-accuracy as the hand-tuned two-phase schedule. It
   needs **no budget tuning** to do so. It decays the lr of each node off that
   node's own validation curve. It freezes converged nodes, a real FLOP saving,
   because the per-node NLLs have independent gradients. And it **stops
   itself**. On stroke-ls it takes 13 s total against 40 s for the baseline. On
   vaca-ci it takes 4 s against 15 s, at an equal or better final NLL.
2. **LBFGS is fast but seed-fragile.** On 2/3 seeds, full-batch LBFGS reaches
   coefficient-level accuracy on the classical all-`ls` model in **< 2 s**. Adam
   needs 9 s for the same accuracy. The third seed stalls at gap 8e-3. An Adam
   warm start made it *worse* on every seed, because the fit then landed in a
   different basin. Use LBFGS as a fast first shot with the plateau trainer as
   the fallback, and not as the default.
3. **OneCycle is a "spend exactly this budget" scheduler.** Accuracy arrives
   only at the end of its anneal. At 1500 epochs it misses everything. At 3000
   epochs it lands gap 1–2e-3. But you must know the right budget in advance.
   That requirement is the problem that we try to remove.
4. **Full-batch loses on time-to-target**, despite ~1.6× higher epoch
   throughput. It makes too few optimizer steps per second of compute at these
   n. Batch 512 is a good default. Very large batches (16k) only improved raw
   throughput at n=50k.
5. **MPS (Apple GPU) is 3–4× slower than the M-series CPU** at these model
   sizes. The reconstruction is identical, so the MPS result is correct.
   Kernel-launch overhead dominates sub-millisecond ops. Stay on CPU locally.
   CUDA on Colab-class GPUs is a different regime.
6. **A fixed budget wastes or under-spends.** Stroke budgeted 4000 epochs, and
   the converged work is done after ~1500. Freezing recovers that difference
   automatically. Vaca budgeted 520 epochs, which is ~0.03 nats short of
   converged. Fixed budgets are wrong in both directions. Adaptive stopping
   fixes both.
7. **Freezing helps, and a parallel node loop does not.** Freezing deletes
   whole epochs. Node-level overlap only time-slices the cores that each node's
   batched BLAS operations already saturate. It measured as contention, not as
   speedup. Overlap can pay only where the per-node kernels do not fully use
   the hardware, which means tiny nodes on a large GPU. For that case the tool is a
   fusion of same-shaped nodes, not threads.

## Recommendation

The everyday recipe is a plateau callback with a generous `epochs` ceiling:
`PerNodePlateau` for per-node rates, or torch's `ReduceLROnPlateau` in a
`Callback` of your own for one shared rate.

Finding 6 is why `epochs` has no default: a fixed budget cannot be right for
every workload, so every caller states its own.
