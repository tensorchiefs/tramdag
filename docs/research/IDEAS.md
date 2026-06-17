# IDEAS — ranked backlog

Ranking score = (expected gain × plausibility) / cost. Re-rank after every
experiment. Results should change beliefs.

## RESOLVED — classical seeding of flexible models (Exp #7): SPEED yes, QUALITY null

OUTCOME (Exp #7): mechanism works (init-equivalence exact). **Speed CONFIRMED**
(seeded starts at the classical near-optimum, ~21 epochs to target vs cold never
matching it). **Quality NULL→NEGATIVE**: seeding anchors to the biased classical
ATE +0.076 (not truth +0.104) and *undermines* `restore_best` recovery (best-val
point becomes the classical anchor). Key lesson: lower observational NLL is
anti-correlated with causal recovery here — anchoring cements the confounded fit.
NOT shipped to main (headline failed + causal footgun); mechanism kept opt-in on
the research branch. Original plan below for provenance.

## (original) NEW DIRECTION — classical seeding of flexible models

Distinct from the merged `marginal_init` (which only calibrates *root* marginals).
Idea: fit the all-`ls` model with `fit_classical` (deterministic exact MLE, fast),
then **seed a flexible (cs/ci) model from that solution** and continue with Adam —
the flexible model starts at the well-understood classical fit and only learns the
nonlinear *deviations*.

Two hypotheses (the 2nd matters more):
- **Speed:** classically-seeded flexible reaches target in fewer steps than cold
  (doesn't re-discover the linear/log-odds structure).
- **Quality (headline):** does it land in a *better basin*? The project's known
  finding: the flexible MLE overfits observational confounding (needs
  `restore_best=True`; on magic-mrclean/nl the flexible MLE undershoots ATE +0.076
  while early-stopping recovers ~+0.10 vs truth +0.104). Does classical-anchoring
  recover the true ATE *without* early stopping? That's a methodological result.

Mechanism (opt-in, defaults untouched): `flow.seed_from_classical(classical_flow)`.
Can't `load_state_dict` across models (upgraded edges use different modules), so:
direct-copy matching `ls`/SimpleIntercept/Bernstein parts; for upgraded `cs`/`ci`
edges use a **residual parametrization** — keep a linear/constant base seeded to the
classical solution + MLP initialized near-zero (learns a correction). Key fact:
adding a classical `ls` shift `w·parent` ≡ adding `w·parent` to the *first* intercept
param (Bernstein θ[0], continuous +; ordinal t0, −) → `ci` edges ARE transferable.
Correctness gate: seeded-flexible `log_prob` must equal classical `log_prob` at init.

Plan: (a) minimal all-`ls`-except-one-`cs` for a clean signal on both hypotheses;
(b) generalize to the full magic-mrclean/nl flexible spec (Age `ci`, NIHSSa `cs`)
for the headline ATE-recovery quality test. Eval: speed = time-to-target ≥3 seeds;
quality = recovered ATE for {cold MLE, cold+restore_best, seeded} vs `true_ate`.
Must generalize off-harness; full suite green before any confirmed win; PR with the
opt-in change + pending-off-machine-test note. Null results are still findings.

---

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
2. **Large-batch + lr-scaled GPU throughput config** — UPRANKED after Verification
   V1: "GPU loses here" was wrong (it loses only at batch ≤~8k; CUDA wins 3–14× at
   batch ≥~64k, util ~23% at small batch = dispatch-bound). For big TRAM-DAGs a
   large-batch CUDA config is a real, broadly-applicable throughput win — the one
   genuinely live lever left. Open risk: lr/schedule must be retuned for the huge
   batch (time-to-target, not just per-step throughput). Medium.
3. RQS tail-slope fix — accuracy lever, may help optimization indirectly. Low.
4. per-node Adam betas/eps — low expected gain.

DOWNRANKED: `val_every=N` — entangled with the plateau/freeze schedule (val
drives lr-decay + freezing) and the metric's detection granularity; not the clean
overhead lever it looked like.

DEAD ENDS (tested): thread count (Exp #2, <10%), torch.compile (Exp #3,
double-backward), feature-cache (Exp #6, ~0%), warm-start on coefficient-bound
workloads (Exp #4/#5, init can't move W1).
CORRECTED (Verification V1): CUDA/device is NOT a dead end — GPU loses only at
small batch (≤~8k, the harness default 512); it wins 3–14× at large batch. Moved
to live idea #2. The dispatch-bound diagnosis stands; the "device is hopeless"
conclusion was over-generalized from one batch size.

ASSESSMENT: the per-step *CPU* search is exhausted and one broadly-safe CPU win is
banked (warm-start, PR #9). Verification V1 reopened one live lever: large-batch
GPU throughput (untested as a time-to-target config). Otherwise the high-value
space is mapped. Notes: defaults stay untouched (opt-in flags).
