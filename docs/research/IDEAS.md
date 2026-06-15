# IDEAS — ranked backlog

Ranking score = (expected gain × plausibility) / cost. Re-rank after every
experiment. Results should change beliefs.

## Re-ranked after Exp #1 (profiler) + Exp #2 (threads)

Established: training is **dispatch-bound** (94% in fwd+bwd of the per-node loop,
thousands of µs ops), single-thread-effective. Two orthogonal axes to attack:
(A) per-step cost → must cut OP COUNT; (B) steps-to-target → init/schedule.

1. **Warm-start init (axis B)** — initialize each continuous node's Bernstein
   intercept θ so the transform starts near the marginal CDF (zuko inits to
   identity-ish; the data is pre-scaled to [-5,5]) instead of cold. Could delete
   a big chunk of early epochs. Shippable as opt-in `fit(..., warm_start=True)`.
   High expected gain on the *actual* metric (time-to-target), low risk (opt-in,
   doesn't touch the MLE path — converged optimum is unchanged). **DO NEXT (#3).**
2. **torch.compile per-node log-prob (axis A)** — fuse the elementwise chain to
   kill dispatch. Highest ceiling (attacks the 94%), helps every workload/user.
   Risk: autograd/dynamic batch shapes, warmup cost; keep OFF by default so the
   sacred MLE tests stay uncompiled. Cheap to micro-bench first. **#4.**
3. **Eval val less often** — full val pass every epoch is pure overhead;
   `val_every=N` opt-in. Small clean win on time (not steps). **#5.**
4. **Kill dtype-copy overhead (~15%)** — cache parent encodings / avoid per-step
   float re-conversion (`_to_copy`/`copy_`/`empty_strided`). Axis A, medium. **#6.**
5. batch-size × lr scaling for W3 throughput — only matters if W3/GPU is in play;
   Exp #0 says GPU loses here, so low priority. **#7.**
6. LBFGS polish after plateau (known fast-but-fragile) — already characterized. **#8.**
7. per-node Adam betas/eps — low expected gain. **#9.**
8. RQS tail-slope fix — accuracy lever, may help optimization indirectly. **#10.**

DEAD ENDS (tested): thread count (Exp #2, <10%), CUDA/device (Exp #0, slower).
Notes: defaults stay untouched (opt-in flags) until the final report.
