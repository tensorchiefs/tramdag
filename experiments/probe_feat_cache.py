"""Exp #6 probe — feature-caching (opt-in fit(cache_features=True)).

(1) correctness: cached vs uncached fit must be bit-identical (the encoding
    commutes with row-indexing). (2) micro-bench: median per-step wall-clock,
    cached vs uncached, both workloads. Decision gate: proceed to a full
    time-to-target A/B only if per-step improves >=10% on >=1 workload.

Usage:  uv run python probe_feat_cache.py
"""

from __future__ import annotations

import time
import numpy as np
import torch

from bench_training import WORKLOADS
from tramdag import CausalFlowDAG


def build(workload, seed=0):
    train, val = WORKLOADS[workload]["data"]()
    torch.manual_seed(seed)
    flow = CausalFlowDAG(WORKLOADS[workload]["spec"]())
    return flow, train, val


def final_nll(workload, cache, epochs=60, seed=0):
    flow, train, val = build(workload, seed)
    flow.fit(train, val, epochs=epochs, learning_rate=1e-2, batch_size=512,
             verbose=0, cache_features=cache)
    return float(flow.log_prob(train).sum())


def per_step_ms(workload, cache, steps=300):
    flow, train, _ = build(workload)
    vals = flow._tensorize(train)
    flow._set_ranges(train)
    feats_full = flow._features(vals) if cache else None
    n = len(train); bs = 512
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    flow.train()
    ts = []
    for s in range(steps):
        idx = torch.randperm(n)[:bs]
        batch = {k: v[idx] for k, v in vals.items()}
        bf = ({k: v[idx] for k, v in feats_full.items()}
              if feats_full is not None else None)
        t0 = time.perf_counter()
        per_node = flow.node_log_prob(batch, feats=bf)
        loss = torch.stack([-v.mean() for v in per_node.values()]).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def main():
    print("=== correctness (cached vs uncached final NLL, must match) ===")
    for w in WORKLOADS:
        a = final_nll(w, False); b = final_nll(w, True)
        print(f"  {w:10s} uncached {a:.6f}  cached {b:.6f}  |diff| {abs(a-b):.2e}")

    print("\n=== per-step median ms (uncached -> cached) ===")
    for w in WORKLOADS:
        u = per_step_ms(w, False); c = per_step_ms(w, True)
        print(f"  {w:10s} {u:.2f} -> {c:.2f} ms   ({100*(u-c)/u:+.1f}%)")


if __name__ == "__main__":
    main()
