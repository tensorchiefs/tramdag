# RESEARCH_LOG — autonomous training-speed research for tramdag

Run branch: `research/2026-06-15-workstation`
Machine: workstation — NVIDIA TITAN RTX (24 GB), torch 2.12.0+cu130, CUDA.
Metric: time-to-target as defined in `experiments/bench_training.py` (W1 stroke-ls,
W2 vaca-ci) + W3 (vaca-ci n=50k, batch 4096, GPU). See MISSION_autoresearch.md.

Format per experiment: HYPOTHESIS (predicted effect size) · CHANGE · COMMANDS ·
NUMBERS (all seeds) · VERDICT · WHAT THIS TEACHES.

---

## Experiment #0 — baseline (no hypothesis; establish the comparison point)

CHANGE: none. Run `bench_training.py` (full grid, 3 seeds) + `perf_machine.py`
unchanged on this machine. All published numbers are from an Apple-silicon Mac
mini and do not transfer; this baseline is the only valid comparison point.

COMMANDS:
- `uv run python perf_machine.py`  (cpu+cuda, 200-epoch fixed workloads)
- `uv run python bench_training.py`  (full grid, seeds 0/1/2, cpu-only — no MPS here)
- per-seed grid wall-clock ≈ 19 min → full 3-seed grid ≈ 55 min.

NUMBERS — bench_training median seconds to target over seeds 0/1/2 (cpu), the
relevant default-trainer rows (full table: `docs/research/baseline/ranking.csv`):

| workload | config | batch | practical | tight |
|---|---|---|---|---|
| stroke-ls (W1) | baseline-2phase | 512 | 34.0 | 78.9 |
| stroke-ls | constant | 512 | 33.6 | 78.8 |
| stroke-ls | plateau+freeze | 512 | 31.9 | MISS |
| stroke-ls | **lbfgs** | full | **4.6** | MISS |
| stroke-ls | adam150+lbfgs | full | 9.1 | MISS |
| vaca-ci (W2) | baseline-2phase | 512 | 7.5 | 10.9 |
| vaca-ci | plateau+freeze | 512 | **7.1** | **7.5** |
| vaca-ci | constant | 512 | 7.5 | 10.9 |

perf_machine (200-epoch fixed work; fit seconds, lower=better):

| workload | cpu | cuda |
|---|---|---|
| intro (n=5k)  | 23.8 | 87.0 |
| large (n=50k) | 42.3 | 90.8 |

Cross-machine NLL match: intro 5.3744, large 4.9154 — agree with Mac/iOS to ~1e-3
(correctness preserved; this is a genuine workstation, not a misconfig).

VERDICT: baseline established (no hypothesis). Reference NLLs reproduced exactly
(stroke 10.3042, vaca 4.9632 — same as published, cached in reference.json).

WHAT THIS TEACHES:
1. **CUDA is 2–4× SLOWER than this box's own CPU** for both workloads — these
   per-node models are tiny (sub-millisecond kernels); launch overhead dominates,
   exactly the MPS story from `docs/training-speed.md` Finding #5. The "GPU
   throughput regime" (W3) is unlikely to favor CUDA here without fusing the
   per-node Python loop into far fewer, larger kernels. This reframes the GPU
   ideas: the lever is *kernel count*, not device.
2. This x86 box is ~3× slower per-op than the Mac mini for these models
   (stroke practical 34s vs Mac 9s) — absolute times differ but the *ranking*
   is the same shape: LBFGS fastest on W1, plateau≈baseline on both.
3. Later-experiment timeout cap = min(5×baseline, 30min) = **30 min** (grid
   per-seed ≈ 19 min ≫ 6 min).

---

## Experiment #1 — profiler breakdown (mandated diagnostic; no library change)

HYPOTHESIS: training is dominated by per-node transform compute (matmuls) — to be
tested, not assumed.

CHANGE: added `experiments/profile_epoch.py` (diagnostic only). Manual 4-phase
timing + torch.profiler self-CPU op table, both workloads, batch 512.

COMMANDS: `uv run python profile_epoch.py`

NUMBERS:
- per-step wall: stroke-ls 10.7 ms, vaca-ci 12.6 ms.
- phase split (both workloads): forward 52–55%, backward 39–41%, opt.step 5–6%,
  batch-index ~1%. → forward+backward of the transform math is ~94%.
- op table: NO single op dominates. Time is spread over thousands of tiny
  elementwise ops — `mul/div/sum/where/xlogy/neg/exp/sub/add`, each 4–20 µs,
  hundreds–thousands of calls per 30 steps. `xlogy` is the priciest single op
  (~41 µs). dtype/copy overhead (`_to_copy`+`to`+`copy_`+`empty_strided`) ≈ 15%.

VERDICT: HYPOTHESIS REJECTED. Training is **op-dispatch / overhead bound**, not
matmul/compute bound. The per-node Python loop issues many sub-millisecond aten
ops; fixed per-op dispatch + (suspected) thread-launch cost dominates real FLOPs.

WHAT THIS TEACHES (re-ranks the backlog):
- This is exactly why CUDA loses (Exp #0): tiny ops, launch-overhead bound.
- The high-ceiling structural fix is **fewer/larger ops** — fuse the elementwise
  chain (torch.compile) or batch the per-node loop. High cost / autograd+MLE risk.
- Cheapest first test: **thread count.** Thousands of µs-scale ops on a 20-thread
  CPU strongly implies oversubscription (per-op fork/join > the work). → Exp #2.
- Clean medium win: kill the ~15% dtype-copy overhead (cache parent encodings,
  avoid float re-conversions each step).

---

## Experiment #2 — CPU thread-count sweep

HYPOTHESIS: tiny ops on 10 cores are oversubscribed; dropping to ~4 threads cuts
per-step time ≥20% (fork/join > work).

CHANGE: `experiments/sweep_threads.py` (diagnostic). Median per-step wall over 300
steps vs `torch.set_num_threads ∈ {1,2,4,6,8,12,16,default=10}`, both workloads.

COMMANDS: `uv run python sweep_threads.py`

NUMBERS (median ms/step):
| threads | stroke-ls | vaca-ci |
|---|---|---|
| 1  | 10.05 | 10.18 |
| 2  | 10.14 | 10.25 |
| 4  | 10.39 | 10.39 |
| 8  | 10.56 | 10.93 |
| 10 (default) | 10.72 | 11.15 |
| 16 | 12.10 | 13.01 |

VERDICT: REJECTED (as a shippable win). Effect is real but small and monotone:
1 thread is fastest, default-10 costs +6% (stroke) / +9% (vaca), and only >10
threads clearly hurts. Predicted ≥20% did not materialize.

WHAT THIS TEACHES: the ops are so small they barely parallelize — going 1→10
threads changes per-step time <10%. This is the strongest confirmation yet that
training is **single-thread-effective and dispatch-bound**: extra cores add a
little fork/join overhead but almost no op actually runs in parallel. Corollary:
the only big lever left is **reducing the number of dispatched ops** (operator
fusion via torch.compile, or batching the per-node Python loop). Threads/devices
are dead ends here. (1 thread never hurts and saves ~7% — a free env tweak, but
not a library change and below the 10% bar; noted, not shipped.)
