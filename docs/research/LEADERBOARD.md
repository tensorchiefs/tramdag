# LEADERBOARD — best known config per workload

Machine: workstation (NVIDIA TITAN RTX, torch 2.12.0+cu130).
Metric: median time-to-practical-target over ≥3 torch seeds.

| workload | best config | median time-to-practical (s) | provenance |
|---|---|---|---|
| W1 stroke-ls | lbfgs (full-batch) | **4.6** | Exp #0 (default-trainer best: plateau+freeze b512 = 31.9) |
| W2 vaca-ci   | plateau+freeze b512 | **7.1** | Exp #0 (tight 7.5) |
| W3 vaca-ci-50k-gpu | (to be cached) | — | Exp #0 prep |

Notes: LBFGS wins W1 but is the known fragile fast-shot (not robust across seeds;
`docs/training-speed.md` Finding #2). plateau+freeze is the best robust default.

Updated after each confirmed improvement.
