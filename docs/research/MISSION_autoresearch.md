# MISSION: autonomous training-speed research for tramdag

<!--
Launch notes (for the human):
  ssh -t <server> tmux new -As GPU_Rack_tramdag
  git clone https://github.com/tensorchiefs/tramdag && cd tramdag
  claude --remote-control GPU_Rack_tramdag        # /effort medium recommended
  first message: "Read docs/research/MISSION_autoresearch.md and begin."
Prerequisites on the server: claude + gh authenticated, uv or pip available.
-->

You are running a self-directed research loop. Goal: find changes that make
`CausalFlowDAG` training **faster to a fixed quality target** (secondarily: reach
better optima). You have the full repository and this machine (check its GPU). No
human is in the loop — work autonomously, with rigor.

Read first: CLAUDE.md, docs/training-speed.md, experiments/bench_training.py,
src/tramdag/flow.py. The June 2026 benchmark defines your methodology.

## Hard limits

- **At most 20 experiments total.** Then stop, write the report, open the PR.
- **At most one experiment per 30 minutes**, and fewer is better: one well-designed
  experiment beats three rushed ones. If you cannot state a falsifiable hypothesis
  with a predicted effect size, you are not ready to run.
- Run benchmarks as background commands and **wait idly** for them. Do not fill
  waiting time with speculation, re-reading files you already know, or rewriting
  logs. Tokens are the scarce resource; wall-clock is free.
- Keep context lean: your accumulated knowledge lives in RESEARCH_LOG.md and
  IDEAS.md, not in re-derivation. After a context compaction, re-read those two
  files and LEADERBOARD.md — nothing else — and continue.

## Branch & push rules

- Work exclusively on branch `research/training-speed`. **Never push to main.**
- Commit after every experiment (confirmed or rejected) and **push after every
  commit** — the human follows your progress on GitHub, not by asking you.
- Never touch the frozen CSVs in `data/`. Library changes stay **opt-in**
  (new flag/argument, defaults untouched) until the final report.

## Experiment #0 (mandatory first action — the baseline)

Before changing anything: run the existing benchmark suite unchanged on THIS
machine — `experiments/bench_training.py` (full grid, 3 seeds) and
`experiments/perf_machine.py` (hardware fingerprint). All published numbers are
from an Apple-silicon Mac mini; nothing transfers — your baseline is the only
valid comparison point. Commit the results (bench `results.csv`/`ranking.csv`
into `docs/research/baseline/`, the perf JSON into `docs/perf/`) and push.
This is your first commit, before any hypothesis.

## The metric (non-negotiable)

- **Time-to-target** as defined in `experiments/bench_training.py`: wall-clock
  seconds until val-NLL is within tolerance of the cached long-run reference
  (W1 stroke-ls: tight 1e-3 / practical 5e-3; W2 vaca-ci: 2e-3 / 1e-2).
- Add **W3**: vaca-ci at n=50,000, batch 4096, on this machine's GPU — the
  throughput regime. Cache its reference the same way.
- A change is an IMPROVEMENT only if ALL hold:
  1. median time-to-practical over ≥3 torch seeds improves ≥10% on ≥1 workload,
  2. no workload regresses >5%,
  3. `uv run pytest tests/ -q` fully green —
     `test_plateau_freeze_preserves_exact_mle` and
     `test_flow_matches_r_reference` are sacred: the exact-MLE property of
     all-`ls` models must survive every change.
- Wall-clock honesty: nothing else heavy on the machine; rerun the baseline
  whenever you doubt machine state; never compare numbers across machine states.

## Method: hypothesis loop with a lab notebook

Maintain three append-only files in `docs/research/`:

- `RESEARCH_LOG.md` — per experiment: HYPOTHESIS (one sentence, predicted effect
  size), CHANGE (diff summary), COMMANDS, NUMBERS (all seeds), VERDICT
  (CONFIRMED / REJECTED / INCONCLUSIVE), WHAT THIS TEACHES. Negative results are
  first-class citizens — log them fully.
- `IDEAS.md` — ranked backlog: (expected gain × plausibility) / cost. Re-rank
  after every experiment; results should change your beliefs.
- `LEADERBOARD.md` — best known config per workload, with provenance.

Discipline:

- **Profile before optimizing.** Experiment #1 is a torch.profiler breakdown of
  one epoch per workload — where does the time actually go? Form hypotheses from
  data, not vibes.
- One change per experiment. No compound changes; if unavoidable, ablate after.
- Three seeds minimum before any verdict; report medians, keep the spread.

## Starting backlog (seed it, then own it)

profiler baseline → where is the time? · torch.compile on the per-node log-prob ·
TF32 / bf16 autocast on GPU (watch the MLE guard!) · batch-size × lr scaling on
GPU · **warm-start init**: initialize Bernstein θ from the node's marginal
quantiles instead of zeros — could delete most of early training · per-node Adam
betas/eps · LBFGS polish phase after plateau-freeze · evaluate val less often
(per-epoch val eval is overhead) · vectorize/fuse the Python loop over nodes ·
one-hot caching in `_tensorize` · the RQS tail-slope fix from
`notebooks/transforms_tram_dag.py` (accuracy lever, may also help optimization) ·
gradient accumulation vs huge batches.

## Cadence and end state

After every confirmed improvement (or every 5 experiments): update LEADERBOARD,
write a 5-line status section into the log. Stop at 20 experiments, when the top
of the backlog has expected gain <5%, or when told to stop. Then write
`docs/research/REPORT.md` (what worked, what didn't and why, recommended changes
with evidence) and open a PR via `gh` containing ONLY the validated changes, PR
description = the report's summary table. A human reviews the PR; the log is
your evidence.
