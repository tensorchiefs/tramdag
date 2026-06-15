# IDEAS — ranked backlog

Ranking score = (expected gain × plausibility) / cost. Re-rank after every
experiment. Results should change beliefs.

## Queue (to be re-ranked after the profiler in Exp #1)

1. **Profiler baseline (Exp #1)** — torch.profiler breakdown of one epoch per
   workload. Where does the time actually go? Forms every later hypothesis.
   Not optional; the mission mandates it as Exp #1.
2. **TF32 matmul on GPU** — `torch.backends.cuda.matmul.allow_tf32`. Cheap, opt-in,
   could help W3 throughput. Watch the MLE guard (W1 is exact-MLE).
3. **Warm-start Bernstein init** from marginal quantiles instead of zeros — could
   delete most of early training. Opt-in flag.
4. **Eval val less often** — per-epoch val eval over full val set is overhead,
   especially for large n. Opt-in flag (val_every=N).
5. **batch-size × lr scaling on GPU (W3)** — bigger batch + scaled lr for throughput.
6. **One-hot caching in `_tensorize`** — recomputed every epoch; cache parent encodings.
7. **Vectorize/fuse Python loop over nodes** — kernel-launch overhead at small models.
8. **torch.compile on per-node log-prob** — high cost, uncertain gain; defer.
9. **LBFGS polish phase after plateau-freeze** — already known fast-but-fragile.
10. **per-node Adam betas/eps** — low expected gain.
11. **RQS tail-slope fix** — accuracy lever; may help optimization indirectly.

Notes: defaults must stay untouched (opt-in flags) until the final report.
