# LEADERBOARD — best known config per workload

Machine: workstation (NVIDIA TITAN RTX, torch 2.12.0+cu130).
Metric: median time-to-practical-target over ≥3 torch seeds.

| workload | best config | median time-to-practical (s) | provenance |
|---|---|---|---|
| W1 stroke-ls | lbfgs (full-batch) | **4.6** | Exp #0 (default-trainer best: plateau+freeze b512 = 31.9; warm_start +4–8%, below bar) |
| W2 vaca-ci   | plateau+freeze b512 **+ warm_start** | **2.6** | Exp #4 (was 6.9–7.1 cold; +56–63% over two seed triples) |
| W3 vaca-ci-50k-gpu | (to be cached) | — | Exp #0 prep |

Notes: LBFGS wins W1 but is the known fragile fast-shot (not robust across seeds;
`docs/training-speed.md` Finding #2). plateau+freeze is the best robust default.
**warm_start (Exp #4, opt-in `fit(warm_start=True)`)** is the new W2 best — a free,
never-regressing calibrated Bernstein init; large win where a continuous root's
marginal is on the NLL critical path (W2), small on the ordinal-dominated W1.

Updated after each confirmed improvement.
