"""Exp #5 — extend warm_start to unconditional ORDINAL cutpoints.

Exp #4 warm-started only Bernstein roots; W1 stroke-ls barely moved (+4-8%)
because its three ordinal nodes (mRS_pre, T, mRS_3m) stayed cold. warm_start now
also calibrates ordinal SimpleIntercept cutpoints to the empirical class log-odds.

Three-arm A/B in one run to attribute the ordinal increment without cross-run
machine drift:
  off  = warm_start=False
  bern = warm_start=True but ordinal intercepts pre-marked done (Bernstein-only,
         == Exp #4 behavior)
  full = warm_start=True (Bernstein roots + ordinal cutpoints)

Hypothesis: full cuts W1 median time-to-practical >=15% vs off (and clearly beats
bern), by calibrating the outcome marginal at init. W2 vaca-ci has no ordinal
nodes -> full == bern (no-op control, no regression).

Usage:  uv run python exp_ordinal_warmstart.py [seeds...]
"""

from __future__ import annotations

import sys
import numpy as np
import torch

from bench_training import (WORKLOADS, CONFIGS, TOL_TIGHT, TOL_PRACT,
                            reference_nll, total_val)
from tramdag import CausalFlowDAG
from tramdag.conditioners import SimpleIntercept

# W1 only: the ordinal change cannot affect W2 (vaca has no ordinal nodes;
# continuous code path unchanged), so the 30-min budget goes to the workload
# under test, all 3 arms, for in-run attribution without cross-run drift.
CASES = [("stroke-ls", "plateau+freeze"),
         ("stroke-ls", "baseline-2phase")]
SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
BATCH = 512
ARMS = ["off", "bern", "full"]


def get_config(workload, label):
    for lbl, phases, extra in CONFIGS[workload]:
        if lbl == label:
            return phases, extra
    raise KeyError(label)


def run(workload, phases, extra, seed, arm):
    train, val = WORKLOADS[workload]["data"]()
    torch.manual_seed(seed)
    flow = CausalFlowDAG(WORKLOADS[workload]["spec"]())
    if arm == "bern":  # disable ordinal warm-start via the first-fit guard
        for name in flow.order:
            node = flow.nodes[name]
            if node.kind == "ordinal" and isinstance(node.intercept, SimpleIntercept):
                node.intercept._warm_started = True
    warm = arm in ("bern", "full")
    for i, (epochs, lr) in enumerate(phases):
        flow.fit(train, val, epochs=epochs, learning_rate=lr, batch_size=BATCH,
                 verbose=0, warm_start=(warm and i == 0), **extra)
    nll = total_val(flow.history)
    times = np.array(flow.history["time"])
    ref = reference_nll(workload)
    hit_p = np.nonzero(nll <= ref + TOL_PRACT[workload])[0]
    hit_t = np.nonzero(nll <= ref + TOL_TIGHT[workload])[0]
    return (float(times[hit_p[0]]) if len(hit_p) else None,
            float(times[hit_t[0]]) if len(hit_t) else None)


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main():
    print(f"{'workload/config':28s} {'arm':5s} {'practical(med)':>15s} "
          f"{'tight(med)':>12s}  per-seed practical")
    table = {}
    for workload, label in CASES:
        phases, extra = get_config(workload, label)
        for arm in ARMS:
            ps, ts = [], []
            for seed in SEEDS:
                p, t = run(workload, phases, extra, seed, arm)
                ps.append(p); ts.append(t)
            mp, mt = med(ps), med(ts)
            table[(workload, label, arm)] = mp
            pp = " ".join(f"{x:.1f}" if x else "MISS" for x in ps)
            print(f"{workload+'/'+label:28s} {arm:5s} "
                  f"{(f'{mp:.1f}s' if mp else 'MISS'):>15s} "
                  f"{(f'{mt:.1f}s' if mt else 'MISS'):>12s}  [{pp}]")

    print("\n=== deltas (median time-to-practical) ===")
    for workload, label in CASES:
        off = table[(workload, label, "off")]
        bern = table[(workload, label, "bern")]
        full = table[(workload, label, "full")]
        def pct(a, b):
            return f"{100*(a-b)/a:+.1f}%" if a and b else "n/a"
        print(f"  {workload+'/'+label:28s} off={off}s "
              f"full vs off {pct(off, full)}  | ordinal increment "
              f"(full vs bern) {pct(bern, full)}")


if __name__ == "__main__":
    main()
