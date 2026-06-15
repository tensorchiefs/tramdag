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
