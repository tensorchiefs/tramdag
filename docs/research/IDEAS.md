# IDEAS — ranked backlog

Ranking score = (expected gain × plausibility) / cost. Re-rank after every
experiment. Results should change beliefs.

## Re-ranked after Exp #5 (ordinal warm-start REJECTED)

Big picture now: **two workload regimes.** W2-like (NLL gap dominated by
*marginal shape*) → warm-start wins huge (Exp #4, +56–63%). W1-like (gap
dominated by *conditional `ls` shift-coefficient estimation*) → init is nearly
irrelevant (Exp #4 +4–8%, Exp #5 ordinal +3% on top — both below bar). The
**init axis is now exhausted.** Remaining wins must attack either (a) coefficient
convergence speed, or (b) pure per-step wall-clock.

DONE: warm-start init (Bernstein roots **+ ordinal cutpoints**) — opt-in
`fit(warm_start=True)`. CONFIRMED via W2 (+56–63%); ordinal extension kept but
only +3% on W1 (Exp #5 REJECTED as standalone). PR-scoping (Bernstein-only vs
both) deferred to the report.

Remaining, re-ranked by the new "coefficient-bound vs overhead" lens:
1. **Kill dtype-copy overhead (~15%)** — profiler (Exp #1) saw ~15% in
   `_to_copy`/`copy_`/`empty_strided`. Pure per-step wall-clock, helps EVERY
   workload incl. the coefficient-bound W1 that init can't touch. Cache parent
   encodings / avoid per-step float re-conversion. The only lever that addresses
   W1. Opt-in. **DO NEXT.**
2. **Faster coefficient convergence on `ls` shift params** — W1 is gated by the
   shift weights reaching the MLE. Try a higher/again-scheduled lr *just* for the
   `ls`/shift param groups (per-node lr already exists), or a brief LBFGS polish on
   the shift params after the Adam plateau. Directly targets the W1 bottleneck.
   Medium risk/effort.
3. **Eval val less often (`val_every=N`)** — DOWNRANKED: entangled with the
   plateau/freeze schedule (val drives lr-decay + freezing) AND the time-to-target
   metric's detection granularity (history is sampled per eval). Not the clean
   "pure overhead" lever it looked like; would need to decouple the schedule from
   val first.
4. batch-size × lr scaling for W3 throughput — only if W3/GPU cached; GPU loses
   here (Exp #0). Low.
5. RQS tail-slope fix — accuracy lever, may help optimization indirectly. Low.
6. per-node Adam betas/eps — low expected gain.

DEAD ENDS (tested): thread count (Exp #2, <10%), CUDA/device (Exp #0, slower),
torch.compile (Exp #3, double-backward unsupported), warm-start on coefficient-
bound workloads (Exp #4/#5, init can't move W1).
Notes: defaults stay untouched (opt-in flags) until the final report.
