"""Exp #4 — calibrated Bernstein warm-start init (opt-in ``fit(warm_start=True)``).

A/B harness that reuses bench_training's cached reference NLLs, tolerances and
configs, toggling ONLY ``warm_start``. Does not touch the measurement files.

Hypothesis: zuko's default zero-theta Bernstein init maps the pre-scaled domain
[-B,B] onto latent z in [-6.93,+7.63] (~2.5x too steep vs the standard-logistic
5/95 quantiles +-2.944). A calibrated linear init removes the early rescaling
phase -> median time-to-practical improves >=15% on W1 stroke-ls (all-ls, all
SimpleIntercept Bernstein nodes); no regression >5% on W2 vaca-ci (mostly ci,
only root x1 warm-started).

Runs the candidate-default config per workload across 3 seeds, baseline vs
warm_start. Usage:  uv run python exp_warmstart.py
"""

from __future__ import annotations

import time
import numpy as np
import torch

from bench_training import (WORKLOADS, CONFIGS, TOL_TIGHT, TOL_PRACT,
                            reference_nll, total_val)

# (workload, config-label) pairs to A/B. plateau+freeze is the robust default
# for both; baseline-2phase added on W1 to check the win is schedule-agnostic.
import sys

CASES = [("stroke-ls", "plateau+freeze"),
         ("stroke-ls", "baseline-2phase"),
         ("vaca-ci", "plateau+freeze")]
# default seeds 0/1/2; override via CLI, e.g. `... 3 4 5` for a robustness re-run
SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
BATCH = 512


def get_config(workload, label):
    for lbl, phases, extra in CONFIGS[workload]:
        if lbl == label:
            return phases, extra
    raise KeyError(label)


def run(workload, phases, extra, seed, warm_start):
    train, val = WORKLOADS[workload]["data"]()
    torch.manual_seed(seed)
    flow = CausalFlowDAG_factory(workload)
    for i, (epochs, lr) in enumerate(phases):
        flow.fit(train, val, epochs=epochs, learning_rate=lr, batch_size=BATCH,
                 verbose=0, warm_start=(warm_start and i == 0), **extra)
    nll = total_val(flow.history)
    times = np.array(flow.history["time"])
    ref = reference_nll(workload)
    hit_p = np.nonzero(nll <= ref + TOL_PRACT[workload])[0]
    hit_t = np.nonzero(nll <= ref + TOL_TIGHT[workload])[0]
    return (float(times[hit_p[0]]) if len(hit_p) else None,
            float(times[hit_t[0]]) if len(hit_t) else None,
            float(nll[-1]))


def CausalFlowDAG_factory(workload):
    from tramdag import CausalFlowDAG
    return CausalFlowDAG(WORKLOADS[workload]["spec"]())


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main():
    print(f"{'workload/config':28s} {'mode':10s} "
          f"{'practical(med)':>15s} {'tight(med)':>12s}  per-seed practical")
    summary = []
    for workload, label in CASES:
        phases, extra = get_config(workload, label)
        ref = reference_nll(workload)
        for warm in (False, True):
            ps, ts = [], []
            for seed in SEEDS:
                p, t, _ = run(workload, phases, extra, seed, warm)
                ps.append(p); ts.append(t)
            mp, mt = med(ps), med(ts)
            summary.append((workload, label, warm, mp, mt))
            tag = "warm_start" if warm else "baseline"
            pp = " ".join(f"{x:.1f}" if x else "MISS" for x in ps)
            print(f"{workload+'/'+label:28s} {tag:10s} "
                  f"{(f'{mp:.1f}s' if mp else 'MISS'):>15s} "
                  f"{(f'{mt:.1f}s' if mt else 'MISS'):>12s}  [{pp}]")

    print("\n=== delta (warm_start vs baseline, median time-to-practical) ===")
    by = {}
    for w, l, warm, mp, mt in summary:
        by.setdefault((w, l), {})[warm] = mp
    for (w, l), d in by.items():
        base, warm = d.get(False), d.get(True)
        if base and warm:
            pct = 100 * (base - warm) / base
            verdict = "IMPROVE" if pct >= 10 else ("regress" if pct < -5 else "~flat")
            print(f"  {w+'/'+l:28s} {base:.1f}s -> {warm:.1f}s  "
                  f"{pct:+.1f}%  [{verdict}]")
        else:
            print(f"  {w+'/'+l:28s} base={base} warm={warm} (a MISS)")


if __name__ == "__main__":
    main()
