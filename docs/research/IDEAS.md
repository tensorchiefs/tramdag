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

State of the search after Exp #6: **both big axes are now closed.** Per-step
(threads/compile/feature-cache all ~0–<10%) and init (warm-start helps only
marginal-shape-bound workloads). The only remaining headroom is coefficient-
convergence speed, which is W1-specific and fragile.

Remaining, re-ranked:
1. **Faster coefficient convergence on `ls` shift params** — W1's binding
   constraint. Per-group higher lr for shift params, or a short LBFGS polish on
   *just* the shift params after the Adam plateau (LBFGS already wins W1 outright
   in Exp #0 at 4.6s but is fragile/not-robust). Expected gain real but narrow
   (W1 only) and fragile. Borderline worth one experiment. **TOP, but <high.**
2. batch-size × lr scaling for W3 throughput — only if W3/GPU cached; GPU loses
   here (Exp #0). Low.
3. RQS tail-slope fix — accuracy lever, may help optimization indirectly. Low.
4. per-node Adam betas/eps — low expected gain.

DOWNRANKED: `val_every=N` — entangled with the plateau/freeze schedule (val
drives lr-decay + freezing) and the metric's detection granularity; not the clean
overhead lever it looked like.

DEAD ENDS (tested): thread count (Exp #2, <10%), CUDA/device (Exp #0, slower),
torch.compile (Exp #3, double-backward), feature-cache (Exp #6, ~0%), warm-start
on coefficient-bound workloads (Exp #4/#5, init can't move W1).

ASSESSMENT: expected gain at the top of the backlog is now low and narrow
(W1-only, fragile). Per the mission's stop rule (top-of-backlog expected gain
<5%, broadly), the productive search is ~exhausted. One confirmed broadly-safe
win banked (warm-start). Lean toward wrapping up: write REPORT.md + open the PR.
Notes: defaults stay untouched (opt-in flags) until the final report.
