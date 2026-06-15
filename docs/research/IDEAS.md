# IDEAS — ranked backlog

Ranking score = (expected gain × plausibility) / cost. Re-rank after every
experiment. Results should change beliefs.

## Re-ranked after Exp #4 (warm-start CONFIRMED)

Established: training is **dispatch-bound**, single-thread-effective; both
per-step axes (threads, compile) are dead. Wins live on axis B (steps-to-target).
Exp #4 banked the top idea (warm-start, +56–63% on W2). The lesson it adds: gains
concentrate where a **continuous root's marginal is on the summed-NLL critical
path** — so the next ideas should target the *other* big NLL contributors
(conditional shape via ci/cs, and eval/schedule overhead).

DONE: **warm-start init** — CONFIRMED Exp #4 (W2 +56–63%, W1 +4–8%, no regression,
full suite green). Shipped as opt-in `fit(warm_start=True)`. Candidate PR default
for Bernstein roots.

Remaining, re-ranked:
1. **Eval val less often (`val_every=N`)** — every-epoch full val pass is pure
   wall-clock overhead, independent of the optimizer. Cheap, clean, helps *time*
   on every workload (incl. the ones warm-start barely moved). Opt-in. **DO NEXT.**
2. **Warm-start the ci/cs conditional too** — Exp #4 only warm-started uncond.
   `SimpleIntercept` roots; W2's x2/x3 (ci) and W1's outcome stayed cold. Init the
   ComplexIntercept's *output bias* toward the same calibrated θ (add a bias term /
   set last-layer) so conditional nodes also skip rescaling. Could extend the W2
   win to the per-seed laggards and finally move W1. Medium risk (touches ci net).
3. **Kill dtype-copy overhead (~15%)** — cache parent encodings / avoid per-step
   float re-conversion. Axis A but the *only* surviving per-step lever; medium.
4. batch-size × lr scaling for W3 throughput — only if W3/GPU is cached; Exp #0
   says GPU loses here. Low priority.
5. LBFGS polish after plateau — already characterized (fast/fragile). Low.
6. per-node Adam betas/eps — low expected gain.
7. RQS tail-slope fix — accuracy lever, may help optimization indirectly.

DEAD ENDS (tested): thread count (Exp #2, <10%), CUDA/device (Exp #0, slower),
torch.compile (Exp #3, double-backward unsupported).
Notes: defaults stay untouched (opt-in flags) until the final report.
