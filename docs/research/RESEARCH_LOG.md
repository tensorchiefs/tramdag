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

---

## Experiment #3 — torch.compile fusion of the per-node loss (the op-count lever)

HYPOTHESIS: torch.compile fuses the elementwise chain → ≥1.3× steady-state
per-step speedup (the only lever that attacks the dispatch-bound 94%).

CHANGE: `experiments/probe_compile.py` (diagnostic). Compile a `loss_fn(flow,
batch)` closure (`dynamic=True`), measure warmup + steady per-step vs eager.

COMMANDS: `uv run python probe_compile.py`

NUMBERS: none — **it does not run at all.** First failure: donated-buffer guard
(worked around with `torch._functorch.config.donated_buffer=False`). Second,
fatal failure: `RuntimeError: torch.compile with aot_autograd does not currently
support double backward`.

VERDICT: DEAD END (hard incompatibility, not a tuning issue).

WHAT THIS TEACHES: zuko's `BernsteinTransform.call_and_ladj` (and the RQS
transform) compute the log-det-Jacobian with `torch.autograd.grad` *inside* the
forward. Training then backprops through that → a **double backward**, which
torch.compile's aot_autograd backend does not support. This breaks compile for
*every continuous node* (the bulk of every workload). Compiling around it would
require rewriting the transforms to return an analytic ladj (no autograd in
forward) — a large change to the numerics core that risks the sacred MLE tests.
Out of scope for a safe opt-in speed win. **The op-count axis is closed in eager
PyTorch.** Both big per-step levers (threads, compile) are now dead → the only
remaining wins are on the **steps-to-target axis** (init, schedule, eval cadence).

---

## Experiment #4 — calibrated Bernstein warm-start init (CONFIRMED)

HYPOTHESIS: zuko's default zero-θ Bernstein init maps the pre-scaled domain
[−B,B] onto latent z∈[−6.93,+7.63] — ~2.5× steeper than the standard-logistic
5/95 quantiles (±2.944), so early training is wasted rescaling the marginal.
A calibrated linear init cuts median time-to-practical ≥15% on W1; ≤5%
regression on W2.

CHANGE (opt-in, defaults untouched): `fit(warm_start=False)` new flag.
`BernsteinUT.warm_start_theta()` returns the closed-form unconstrained θ (inverting
zuko's `_constrain_theta` cumsum-of-softplus) whose transform is the exact linear
map of [−B,B] onto [logit .05, logit .95]. Applied in `_set_ranges` under the same
`not ut._fitted` first-fit guard (so multi-phase fits don't reset a trained
intercept), only to `SimpleIntercept` Bernstein nodes (ci/ordinal untouched).
Verified numerically: warm θ → z(±5)=±2.944, z(0)=0 (default: −6.93/+7.63, z(0)=0.347).

COMMANDS: `uv run python exp_warmstart.py` (seeds 0/1/2) and `... 3 4 5`
(independent robustness triple). A/B reuses bench_training's cached reference,
tolerances, configs — measurement files untouched.

NUMBERS (median time-to-practical s, baseline → warm_start, two seed triples):

| workload / config | seeds 0–2 | seeds 3–5 |
|---|---|---|
| **vaca-ci / plateau+freeze (W2)** | 6.9 → **2.6**  (+62.6%) | 6.5 → **2.8**  (+56.2%) |
| stroke-ls / plateau+freeze (W1)  | 31.8 → 30.4  (+4.3%) | 35.2 → 32.3  (+8.2%) |
| stroke-ls / baseline-2phase (W1) | 33.3 → 31.2  (+6.2%) | 35.7 → 34.4  (+3.7%) |

W2 per-seed (3–5): 6.7→3.5, 6.1→2.8, 6.5→2.8 — all three drop (the seed-1 plateau
at 9.5s in the first triple was seed-specific, not method fragility). W1
baseline-2phase *tight* target also improves (78.0→66.1, 88.0→67.8): warm-start
reaches the **same** MLE faster, never a different optimum.

VERDICT: **CONFIRMED.** (1) ≥10% on ≥1 workload: W2 +56–63% across two
independent triples. (2) No regression >5%: W1 +4–8% (improves). (3) Full suite
`uv run pytest tests/ -q` 73 passed (44:36) incl. the sacred
`test_plateau_freeze_preserves_exact_mle` / `test_flow_matches_r_reference` — the
exact-MLE property survives (warm-start is pure init).

WHAT THIS TEACHES:
- The first confirmed win, and it's on the **steps-to-target axis** as predicted
  once both per-step axes (threads, compile) closed.
- **Why W2 ≫ W1:** the gain scales with how much of the total NLL gap is a
  continuous root's *marginal shape*. W2 (all-continuous) — root x1's miscalibrated
  marginal dominates early total NLL → warm-start ~2.5×s it away. W1 (all-`ls`) —
  the metric is train-NLL to the proportional-odds MLE, dominated by the ordinal
  outcome's `ls` shift coefficients; the continuous marginals are a small slice, so
  fixing them barely moves time-to-target. The effect is real and *general* (helps
  any Bernstein root), just unevenly leveraged per workload — not harness overfit.
- Counter-intuitive: only **1 of 3** vaca nodes is warm-started (x1; x2/x3 are ci),
  yet the workload speeds up ~2.5×. A single well-initialized root pays off when its
  marginal is on the critical path of the summed-NLL target.
- Free and safe: opt-in, never regresses, converges to the identical MLE. Strong
  candidate for the final PR (consider making it the default for Bernstein roots).

---

## Experiment #5 — extend warm_start to ordinal cutpoints (REJECTED)

HYPOTHESIS: W1 barely moved in Exp #4 because its three ordinal nodes (mRS_pre,
T, mRS_3m) stayed cold; calibrating their cutpoints to the empirical class
log-odds (vs zeros = near-uniform) cuts W1 median time-to-practical ≥15%.

CHANGE (folded into the same opt-in flag): `warm_start` now also calibrates
unconditional **ordinal** `SimpleIntercept` cutpoints. Added
`transforms.ordinal_warm_start_theta(counts)` — inverts `ordinal_cutpoints`
(`tt[0]=logit F(0)`, `tt[i]=log(logit F(i)−logit F(i−1))`). Verified numerically:
reproduces the empirical 7-level marginal to max-abs-err 0.0 (default zeros give
P(Y=0)=0.50 vs true 0.30). Bernstein/continuous path unchanged; W2 has no ordinal
nodes so is provably unaffected.

COMMANDS: `uv run python exp_ordinal_warmstart.py` — three arms in ONE run to
attribute the increment without cross-run drift: `off` (warm_start=False), `bern`
(Bernstein-only, ordinal intercepts pre-marked via the first-fit guard = Exp #4
behavior), `full` (both). W1 only (the workload under test), 3 seeds.

NUMBERS (median time-to-practical s, seeds 0/1/2):

| config | off | bern | full | full vs off | ordinal increment (full vs bern) |
|---|---|---|---|---|---|
| plateau+freeze  | 30.8 | 29.5 | 28.6 | +7.1% | **+2.8%** |
| baseline-2phase | 32.2 | 30.3 | 29.4 | +8.8% | **+3.0%** |

(baseline-2phase *tight* also improves, 75.5→64.0 — same MLE, reached faster.)

VERDICT: **REJECTED.** The ordinal increment is only ~3%; total W1 warm_start
stays +7–9%, below the 10% bar. (Kept in the codebase: opt-in, never regresses,
mathematically exact, and the coherent completion of "calibrate all unconditional
marginals" — but it is *not* a standalone confirmed win.)

WHAT THIS TEACHES (the real result):
- **W1 is coefficient-bound, not init-bound.** Cold ordinal cutpoints start badly
  wrong but Adam fixes them in a few epochs; what gates W1's time-to-target is the
  convergence of the `ls` **shift coefficients** (the proportional-odds regression
  weights) to the MLE — which *no* marginal-calibration init can touch.
- This closes the warm-start line: it speeds up workloads whose NLL gap is
  dominated by **marginal shape** (W2, +56–63%) and gives only a few % where the
  gap is dominated by **conditional coefficient estimation** (W1). Both Exp #4 and
  #5 are consistent with this single explanation.
- Next levers must attack coefficient convergence (schedule/optimizer on the shift
  params) or pure per-step wall-clock (dtype-copy overhead) — not initialization.
