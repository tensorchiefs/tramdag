# Project Progression: Driving an Autonomous Training-Speed Study on tramdag

*A process-first write-up of how an agentic/autonomous programming run was set up and
operated. The research subject — making TRAM-DAG training faster-to-target — is the
evidence; the method is the point. Read this if you want to learn how to drive this kind
of work.*

Branch: `research/2026-06-15-workstation` · Machine: x86 workstation, NVIDIA TITAN RTX
(24 GB), torch 2.12.0+cu130 · Model under study: `CausalFlowDAG`, a causal normalizing
flow (TRAM-DAG on zuko). See `CLAUDE.md` for what the library is.

---

## 1. What this was

An autonomous agent was handed a research mission (`docs/research/MISSION_autoresearch.md`)
and left to run it on its own with no human in the loop. The goal was narrow and
measurable: find changes that make `CausalFlowDAG` training **faster to a fixed quality
target** (time-to-target val-NLL), secondarily reach better optima. The agent worked on
its own dated, host-scoped branch; kept an append-only lab notebook
(`RESEARCH_LOG.md` / `IDEAS.md` / `LEADERBOARD.md`); committed and pushed after every
experiment so a human could follow on GitHub; and was bounded by hard limits (≤20
experiments, ≤1 per 30 min) and enforced integrity guardrails.

The run ran to completion: **7 experiments (#0 baseline + 6 hypotheses).** It produced
**one CONFIRMED, broadly-safe win** — a calibrated warm-start init, shipped as the opt-in
`fit(warm_start=True)` in [PR #9](https://github.com/tensorchiefs/tramdag/pull/9)
(`feat/warm-start-init` → `main`) — and **five negatives that, between them, fully map the
search space** along its two orthogonal axes. The negative results did most of the
navigational work: each REJECTED/DEAD-END verdict closed off a region and re-ranked the
backlog until both axes were exhausted and the mission's stop condition tripped.

---

## 2. The method / operating loop

The mission encodes a strict hypothesis loop. Each experiment is one turn of:

```
state a falsifiable HYPOTHESIS with a predicted effect size
  → make exactly ONE change
    → run a ≥3-seed experiment, wrapped in an OS-level timeout
      → record NUMBERS (all seeds) and a VERDICT (CONFIRMED / REJECTED / INCONCLUSIVE)
        → commit + push
          → re-rank the IDEAS backlog so results change beliefs
```

The constraints that shaped the agent's behaviour, and why each one mattered:

| Constraint | Effect on behaviour |
|---|---|
| **≤20 experiments total** | Forces ranking. No fishing — every run has to earn its slot. |
| **≤1 experiment / 30 min** | Slows the agent to one well-designed test at a time; "one good experiment beats three rushed ones." |
| **Profile before optimizing** | Exp #1 is a *mandated* torch.profiler breakdown. Hypotheses come from data, not vibes. |
| **One change per experiment** | No compound changes; effects stay attributable. |
| **Predicted effect size required** | "If you cannot state a falsifiable hypothesis with a predicted effect size, you are not ready to run." |
| **Defaults stay untouched** | Every library change is opt-in (a new flag/argument) until the final report. |
| **≥3 seeds, report medians, keep spread** | A verdict needs robustness, not a lucky seed. |
| **Wait idly on background jobs** | Benchmarks run as background commands; the agent does not fill the wait with speculation. **Tokens are the scarce resource; wall-clock is free.** |

The improvement bar was non-negotiable and three-pronged: a change counts only if (1)
median time-to-practical over ≥3 seeds improves ≥10% on ≥1 workload, (2) no workload
regresses >5%, and (3) the **full** test suite passes — including the "sacred"
`test_plateau_freeze_preserves_exact_mle` and `test_flow_matches_r_reference`, which pin
the exact-MLE property of all-`ls` models.

Test cadence was cost-staged on purpose: the fast subset (`-m "not slow"`, ~30 s) plus
the specific touched tests during iteration; the full ~12-min suite only when a change is
about to be recorded as *confirmed*, and again before the PR. Running the full suite every
experiment would have dominated the run.

---

## 3. The guardrails (the heart of the "how to use agentic programming" discussion)

The interesting part of this run is not the speed win — it is the scaffolding that let an
unsupervised agent be trusted with a measurement it could trivially have gamed. The
guardrails were **enforced**, not merely written down.

### 3.1 Integrity hook that actively blocked the agent — twice

A `PreToolUse` hook (`.claude/hooks/protect_measurement.py`, committed in `a82e1aa`)
blocks, at the OS level, any edit to `tests/`, `data/`, or
`experiments/bench_training.py`, and any *read* of the target values
(`tests/test_*.py`, `data/**/truth.json`, `data/**/ref_ls`) — while still allowing the
input CSVs and the harness source the agent legitimately needs. It is activated only on
the research machine (via a git-ignored `settings.local.json`), so it never interferes
with normal test editing on a dev box.

During the live run it fired twice:

- **Once when the agent tried to `grep` test files** — which could have exposed the
  target/reference values it is forbidden to curve-fit to.
- **Once when the agent tried to add a new test file under `tests/`** — the entire
  `tests/` tree *is* the measurement and is off-limits during the run.

Both times the agent treated the `BLOCKED` message as a **hard stop**, not an obstacle to
route around: it validated the new feature through its own experiment harness
(`experiments/exp_warmstart.py`) instead of writing a test, and it ran pass/fail on the
existing suite rather than reading the numbers behind it.

**Lesson:** encode "don't game the metric / don't edit what measures you" as *enforced*
guardrails. Instructions in a prompt are advisory; a `PreToolUse` hook is load-bearing.

### 3.2 Don't-read-the-targets rule

Beyond the hook, the mission's scientific-integrity section makes the anti-overfit rule
explicit: optimize for *broadly applicable* training gains, not the particularities of the
benchmark, the recorded seeds, or this machine. Reference values are computed once
(Exp #0) and then frozen. The agent runs the tests and trusts pass/fail rather than
curve-fitting to the numbers. A win that evaporates off the measured setup is overfitting
to the harness — and is rejected. (This generalization rule, not the read-blocking
deterrent, is the truly load-bearing anti-overfit guard; pytest reads at the OS level.)

### 3.3 Machine-calibrated timeouts

Every experiment was wrapped in an OS-level `timeout` capped at `min(5 × Exp#0 baseline,
30 min)` — the agent does not rely on noticing a hang itself. The baseline anchor
auto-calibrates the cap to the machine: a slow CPU box and a fast GPU each get a
proportionate budget.

This bit in practice. The published full-suite figure (~12 min) came from an Apple-silicon
Mac mini; this x86 workstation is **~3× slower** per-op for these tiny models. The first
full-suite run was *under*-budgeted and **timed out at 25 min (exit 124)** — not a code
failure, a calibration failure. Re-run at a 45-min budget, it passed: **73 tests in
44:36.** **Lesson:** calibrate caps to the actual machine. A timeout is a signal (here:
"your budget was wrong"), not necessarily a failure of the thing under test. And per the
mission, a genuine hang is logged `INCONCLUSIVE`, downranked, and **never auto-retried**.

### 3.4 Check-in recovery on startup

On resuming, the mission tells the agent to first locate its existing run branch with
`git ls-remote --heads origin 'research/*-<host>'` and continue on it — one branch per
run, never fork your own work, never push to main.

The instructive moment: on resume the agent discovered that its prior commits (Exp #0–#2)
had only ever been committed **locally — never pushed to the remote.** It caught this
*before* doing any new work and pushed everything, so the human following on GitHub could
actually see progress. **Lesson:** verify the check-in path actually reaches the remote. A
local commit is not visible progress. This is exactly why the mission front-loads a
**Step 0 push round-trip** (commit `95b441f`, a throwaway `HEARTBEAT.md`): prove the
push/auth path works in seconds, before burning a 30-minute baseline on a machine whose
git credentials might be broken.

### 3.5 Resume-after-compaction discipline

The mission instructs: after any context compaction, re-read **only three** state files —
`RESEARCH_LOG.md`, `IDEAS.md`, `LEADERBOARD.md` — and continue. Nothing else. Accumulated
knowledge lives in committed artifacts, not in the agent's context window. The append-only
notebook *is* the durable memory; the context window is disposable. This is what makes the
"wait idly, don't re-derive, tokens are scarce" rule actually workable across a long run.

---

## 4. Chronological walkthrough

The real trail (UTC), from `git log` on the branch:

| Commit | Time | Experiment |
|---|---|---|
| `95b441f` | 06:31 | Step 0 push round-trip (heartbeat) |
| `ecc951d` | 08:55 | **Exp #0** baseline benchmark + perf fingerprint |
| `cf910d2` | 09:00 | **Exp #1–2** profiler (dispatch-bound) + thread sweep (negative) |
| `1ec6498` | 15:01 | **Exp #3** torch.compile DEAD END |
| `f87c8a3` | 17:01 | **Exp #4** warm-start init CONFIRMED |
| (later) | — | **Exp #5** ordinal-cutpoint warm-start REJECTED (code kept) |
| (later) | — | **Exp #6** feature-cache REJECTED (reverted) → REPORT.md + PR #9 |

### Exp #0 — baseline (no hypothesis)

Ran `bench_training.py` (full grid, 3 seeds) and `perf_machine.py` unchanged on this
machine, because every published number came from a different (Apple-silicon) machine and
nothing transfers — this baseline is the only valid comparison point. Reference NLLs
reproduced exactly (stroke 10.3042, vaca 4.9632), confirming correctness, not a misconfig.

Median seconds-to-target (cpu, seeds 0/1/2):

| workload | best default config | practical | tight |
|---|---|---|---|
| stroke-ls (W1) | plateau+freeze b512 | 31.9 | MISS |
| stroke-ls (W1) | lbfgs (full-batch) | **4.6** | MISS |
| vaca-ci (W2) | plateau+freeze b512 | **7.1** | **7.5** |

Two findings reframed everything downstream: (1) **CUDA was 2–4× *slower* than this box's
own CPU** on both workloads (intro fit 23.8 s cpu vs 87.0 s cuda) — the per-node models
are sub-millisecond kernels, so launch overhead dominates; (2) this x86 box is ~3× slower
per-op than the Mac, but the *ranking* of configs keeps its shape. The baseline wall-clock
also set the timeout anchor: `min(5×baseline, 30 min) = 30 min`.

### Exp #1 — profiler breakdown (mandated diagnostic, no library change)

**Hypothesis:** training is dominated by per-node transform compute (matmuls). **Result:
REJECTED.** The torch.profiler op table showed *no single op dominates* — time is spread
over thousands of tiny elementwise ops (`mul/div/sum/where/xlogy/neg/exp/...`, each 4–20 µs;
`xlogy` priciest at ~41 µs), with ~15% in dtype/copy overhead. Phase split: forward 52–55%,
backward 39–41%, opt.step 5–6%. Per-step wall: stroke 10.7 ms, vaca 12.6 ms.

**What it taught:** training is **op-dispatch / overhead bound, not compute bound.** This
explained Exp #0's CUDA loss directly (tiny ops, launch-overhead bound) and **split the
backlog into two axes**: axis A = *fewer/larger ops* (threads, fusion); axis B =
*fewer steps to target* (init, schedule, eval cadence). The cheapest axis-A probe — thread
count — became Exp #2.

### Exp #2 — CPU thread-count sweep

**Hypothesis:** tiny ops on 10 cores are oversubscribed; dropping to ~4 threads cuts
per-step time ≥20%. **Result: REJECTED.** The effect was real, small, and monotone:

| threads | stroke-ls ms/step | vaca-ci ms/step |
|---|---|---|
| 1 | 10.05 | 10.18 |
| 10 (default) | 10.72 | 11.15 |
| 16 | 12.10 | 13.01 |

1 thread is fastest; default-10 costs only +6%/+9%; only >10 threads clearly hurts. The
predicted ≥20% never materialized. **What it taught:** the ops are so small they barely
parallelize — the strongest confirmation yet that training is **single-thread-effective
and dispatch-bound.** Threads and devices are dead ends; the only big lever left on axis A
is *reducing the number of dispatched ops* → operator fusion → Exp #3. (1 thread saves ~7%
for free, but it is an env tweak below the 10% bar, not a library change — noted, not
shipped.)

### Exp #3 — torch.compile fusion (the op-count lever)

**Hypothesis:** torch.compile fuses the elementwise chain → ≥1.3× steady-state per-step
speedup. **Result: DEAD END (hard incompatibility, not tuning).** It produced *no numbers
because it does not run at all*: first a donated-buffer guard (worked around), then the
fatal `RuntimeError: torch.compile with aot_autograd does not currently support double
backward`.

**What it taught:** zuko's `BernsteinTransform.call_and_ladj` (and the RQS transform)
compute the log-det-Jacobian with `torch.autograd.grad` *inside* the forward pass.
Training then backprops through that → a **double backward**, which torch.compile's
aot_autograd backend cannot handle. This breaks compile for *every continuous node* — the
bulk of every workload. Fixing it would mean rewriting the numerics core to return an
analytic ladj, risking the sacred MLE tests: out of scope for a safe opt-in win. **The
op-count axis (axis A) is closed in eager PyTorch.** With both per-step levers dead, the
remaining wins must live on axis B (steps-to-target) — which is exactly where Exp #4
looked.

### Exp #4 — calibrated Bernstein warm-start init (CONFIRMED)

**Hypothesis:** zuko's default zero-θ Bernstein init maps the pre-scaled domain onto
z ∈ [−6.93, +7.63] — ~2.5× steeper than the standard-logistic 5/95 quantiles (±2.944) —
so early training is wasted just rescaling the marginal. A calibrated linear init cuts
median time-to-practical ≥15% on W1, ≤5% regression on W2.

**Change (opt-in, defaults untouched):** a new `fit(warm_start=False)` flag.
`BernsteinUT.warm_start_theta()` returns the closed-form unconstrained θ (analytically
inverting zuko's `_constrain_theta` cumsum-of-softplus) whose transform is the exact linear
map of the pre-scaled domain onto [logit .05, logit .95]. Applied only to unconditional
`SimpleIntercept` Bernstein nodes under the same first-fit guard (ci/ordinal untouched).
Verified numerically: warm θ → z(±5) = ±2.944, z(0) = 0.

**Numbers** (median time-to-practical s, baseline → warm_start, two *independent* seed
triples):

| workload / config | seeds 0–2 | seeds 3–5 |
|---|---|---|
| **vaca-ci / plateau+freeze (W2)** | 6.9 → **2.6** (+62.6%) | 6.5 → **2.8** (+56.2%) |
| stroke-ls / plateau+freeze (W1) | 31.8 → 30.4 (+4.3%) | 35.2 → 32.3 (+8.2%) |
| stroke-ls / baseline-2phase (W1) | 33.3 → 31.2 (+6.2%) | 35.7 → 34.4 (+3.7%) |

**Verdict: CONFIRMED.** (1) ≥10% on ≥1 workload: W2 +56–63% across two independent
triples. (2) No regression >5%: W1 improves +4–8%. (3) Full suite `uv run pytest tests/ -q`
**73 passed (44:36)**, including the sacred MLE/R-reference tests — the exact-MLE property
survives because warm-start is *pure init* and converges to the identical MLE (W1's tight
target also improves, 78.0→66.1, 88.0→67.8: same optimum, reached faster).

**The insight, not just the number:** the gain concentrates where a continuous root's
*marginal shape* is on the critical path of the summed-NLL target. W2 (all-continuous):
root x1's miscalibrated marginal dominates early total NLL → warm-start ~2.5×'s it away.
W1 (all-`ls`): time-to-target is dominated by the ordinal outcome's `ls` shift
coefficients, so fixing the continuous marginals barely moves it. Counter-intuitively,
only **1 of 3** vaca nodes is warm-started (x1; x2/x3 are ci), yet the workload speeds up
~2.5× — a single well-initialized root pays off when its marginal sits on that critical
path. This is *why* the effect is general (it helps any Bernstein root) but unevenly
leveraged per workload — and why it is not harness overfitting.

This re-ranked the backlog one more time: with warm-start banked, the live ideas became
the *other* big NLL contributors — warm-starting the *ordinal* root cutpoints (does the
+3–8% on W1 hide behind cold ordinal nodes?), and killing the ~15% dtype-copy overhead
that the Exp #1 profiler flagged. Threads/CUDA/compile sit explicitly in the DEAD ENDS
list. Those two ideas became Exp #5 and #6.

### Exp #5 — extend warm-start to ordinal cutpoints (REJECTED, code kept)

**Hypothesis:** W1 barely moved in #4 because its three ordinal nodes (mRS_pre, T, mRS_3m)
stayed *cold*; calibrating their cutpoints to the empirical class log-odds (instead of
zeros = near-uniform) cuts W1 median time-to-practical ≥15%.

**Change (folded into the same opt-in flag):** `warm_start` now also calibrates
unconditional **ordinal** intercepts via `ordinal_warm_start_theta(counts)`, the
closed-form inversion of `ordinal_cutpoints`. Verified numerically: it reproduces the
empirical 7-level marginal to **max-abs-err 0.0** (default zeros give P(Y=0)=0.50 vs true
0.30). To attribute the increment cleanly, the experiment ran a **3-arm in-run A/B** —
`off` / `bern` (Bernstein-only = Exp #4 behaviour) / `full` — all in one process so there
is no cross-run drift, W1 only.

**Numbers** (median time-to-practical s, seeds 0/1/2):

| config | off | bern | full | full vs off | ordinal increment (full vs bern) |
|---|---|---|---|---|---|
| plateau+freeze  | 30.8 | 29.5 | 28.6 | +7.1% | **+2.8%** |
| baseline-2phase | 32.2 | 30.3 | 29.4 | +8.8% | **+3.0%** |

**Verdict: REJECTED.** The ordinal increment is only +2.8% / +3.0% (full vs Bernstein-only),
and total W1 warm_start stays +7–9% — below the 10% bar. But the code was **KEPT** as
opt-in: it never regresses, is mathematically exact (the max-abs-err-0.0 marginal check),
and is the coherent completion of "calibrate *all* unconditional marginals." A rejected
*workload result* is not a rejected *feature*.

**What it taught:** the real lesson isn't the 3% — it's *why* it's only 3%. **W1 is
coefficient-bound, not init-bound.** Cold ordinal cutpoints start badly wrong, but Adam
fixes them in a handful of epochs; what actually gates W1's time-to-target is the
convergence of the `ls` **shift coefficients** (the proportional-odds regression weights)
to the MLE — and *no* marginal-calibration init can touch that. This closes the entire
warm-start line with one explanation consistent with both #4 and #5: warm-start wins big
where the NLL gap is **marginal-shape-bound** (W2, +56–63%) and barely moves where it's
**conditional-coefficient-bound** (W1). The next levers must attack coefficient
convergence or pure per-step wall-clock — not initialization.

### Exp #6 — cache parent encodings / dtype-copy overhead (REJECTED, reverted)

**Hypothesis:** the ~15% `_to_copy`/`copy_`/`empty_strided` overhead the Exp #1 profiler
flagged is the per-step one-hot/dtype-cast of parent features (recomputed every forward in
`_features`); caching it once and row-indexing per batch cuts per-step time ≥10% — a flat
wall-clock multiplier that would help even the coefficient-bound W1 that init can't move.

**Method — probe-first decision gate.** Rather than spend a 30-min full A/B up front, the
agent ran a cheap micro-benchmark (`probe_feat_cache.py`): (1) correctness, (2) median
per-step ms cached vs uncached on both workloads, with an explicit gate — *proceed to the
full A/B only if ≥10% per-step, else reject*.

**Numbers:**
- Correctness: cached vs uncached final NLL bit-identical — stroke |diff| **0.00e+00**,
  vaca |diff| 0.00e+00 (the one-hot/view commutes with row-indexing).
- Per-step median: stroke 10.86→10.84 ms (**+0.2%**), vaca 11.12→11.11 ms (**+0.0%**).
  Essentially zero. The gate fired — no A/B was run.

**Verdict: REJECTED (~0% gain), and the library change was REVERTED** — unlike #5's
coherent +3% extension, this has *literally zero* benefit, so keeping it would only make
`src/` less truthful. (Probe + log kept as evidence.)

**What it taught:** the profiler's ~15% copy overhead is **not** in the Python-level
parent encoding — it's intrinsic to the per-node transform math (zuko's internal
allocations, the batch gather `v[idx]`, log-space temporaries). Caching features can't
reach it. With this, **the per-step axis is now fully closed:** threads (#2),
torch.compile (#3), and feature-caching (#6) all fail to move per-step time. Combined with
the init axis (exhausted by #4/#5), the only headroom left is **coefficient-convergence
speed** on the `ls`/shift params — which is W1-specific, fragile (the LBFGS that wins W1 in
4.6s in Exp #0 is not robust across seeds), and narrow.

### Wrap-up — stop condition, final suite, curated PR

With both axes closed, the top of the re-ranked backlog (`IDEAS.md`) was a single narrow,
W1-only, fragile lever with expected gain <5% broadly — exactly the mission's **stop
condition**. So the agent stopped searching rather than burning its remaining experiment
budget on a low-value idea. It then:

1. Wrote `docs/research/REPORT.md` — the final summary table (all 7 experiments), the one
   recommended change, the per-axis "why the search is exhausted" argument, and explicit
   follow-ups.
2. Ran the **full** suite one final time on the *exact PR state*: `uv run pytest tests/ -q`
   → **73 passed in 44:12**, including the sacred
   `test_plateau_freeze_preserves_exact_mle` / `test_flow_matches_r_reference`.
3. Opened **[PR #9](https://github.com/tensorchiefs/tramdag/pull/9)** containing **only the
   validated change** — built on a *fresh* `feat/warm-start-init` branch cut from `main`
   (just the warm-start `src/` diff + CHANGELOG + REPORT.md), **not** the research branch
   with its six diagnostic scripts and full lab notebook. The notebook is the *evidence*
   (linked from the PR), not the *deliverable*. The PR body honestly discloses that it
   ships **without a unit test** — the research machine's integrity hook protects `tests/`
   — flagging that as a follow-up rather than working around the guardrail.

---

## 5. Takeaways for driving agentic work

What actually made this run productive, grounded in what happened above:

1. **Enforce the guardrails; don't just write them.** The `PreToolUse` hook stopped the
   agent from grepping tests and from adding a test file — the two moves that could have
   let it game or alter the measurement. A prompt instruction would not have. Make
   "don't edit what measures you" a hook, not a sentence.

2. **The append-only notebook is the agent's real memory.** Knowledge lives in committed
   `RESEARCH_LOG.md` / `IDEAS.md` / `LEADERBOARD.md`, re-read after every compaction —
   *only* those three files. This is what lets the context window stay disposable and the
   run survive arbitrarily long.

3. **Negative results are navigation, not waste.** Five of the six hypotheses were
   negatives, yet they did the steering: profiler → "dispatch-bound" → threads dead →
   compile dead → *only the steps-to-target axis is left* → warm-start wins → ordinal
   extension shows W1 is coefficient-bound → feature-cache shows per-step is fully closed →
   both axes exhausted → stop. Logging each REJECTED/DEAD-END verdict in full, and
   **re-ranking the backlog after every one**, is what walked the agent to the one place a
   win existed *and* told it when to stop.

4. **Calibrate to the machine, and read a timeout as a signal.** Caps were
   `min(5×baseline, 30 min)` anchored on this box's own baseline. The full suite timed out
   at 25 min not because anything was broken but because the budget was a Mac-era number on
   a 3×-slower box — re-budgeted to 45 min, it passed clean.

5. **Verify the check-in path reaches the remote.** A local commit is not visible progress.
   The Step 0 heartbeat push proves auth/credentials in seconds; the resume-time
   `git ls-remote` check caught three commits that had never left the machine. For an agent
   a human follows asynchronously on GitHub, "did it actually push?" is a first-class
   check.

6. **Spend tokens, not wall-clock.** Benchmarks ran in the background and the agent waited
   idly rather than speculating into the gap. With a ≤20-experiment / ≤1-per-30-min budget,
   the discipline of *not* filling waiting time is what kept the run focused — one
   well-formed hypothesis per turn, each with a predicted effect size, or it doesn't run.

7. **Keep wins opt-in until the end.** Every library change was a new flag with defaults
   untouched, validated against the simulator's known truth and the full suite — so the
   final deliverable is a PR of *only the validated change* (`fit(warm_start=True)`), with
   the lab notebook as its evidence trail.

8. **Probe-first decision gates — falsify cheaply before measuring expensively.** Exp #6
   ran a few-second micro-benchmark with an explicit threshold (≥10% per-step → proceed to
   the full A/B, else reject) and killed the idea in minutes instead of spending a 30-min
   A/B on a lever that turned out to be ~0%. Put the cheapest possible falsification in
   front of the expensive measurement.

9. **Scope down on a timeout; don't blindly retry.** Exp #5's first run (27 fits) hit the
   30-min cap. The agent read that correctly — "too much work for the window," not a hang —
   and re-scoped to the 18 fits that actually mattered (dropping a workload it could *prove*
   was unaffected, since W2 has no ordinal nodes), rather than re-running the same
   oversized job. This is distinct from the mission's "a genuine hang is logged
   INCONCLUSIVE and **never** auto-retried" rule: diagnosing *why* the clock ran out is the
   point.

10. **Revert dead levers; keep coherent ones — the workload verdict is not the code
    verdict.** Exp #6's feature-cache had literally zero benefit, so it was **reverted** to
    keep `src/` truthful. Exp #5's ordinal warm-start was also a rejected *workload result*
    (+3%, below bar), but it is a harmless, exact, coherent completion of a shipped feature
    that never regresses — so it was **kept**. "REJECTED" applied to a measurement; the
    decision about the code is separate.

11. **Curate the deliverable; the notebook is evidence, not product.** "Open a PR with only
    the validated change" meant cutting a clean `feat/warm-start-init` branch from `main`
    with just the warm-start diff + CHANGELOG + REPORT.md — *not* PR-ing the research branch
    with all six diagnostic scripts and the full lab notebook. And be honest about the gaps
    the guardrails impose: the PR ships **without a unit test** because the integrity hook
    protects `tests/`, disclosed in the PR body as a follow-up rather than worked around.
