"""Exp #1 — where does a training epoch's time actually go?

Diagnostic only (no library change). For each benchmark workload we:
  (1) manually time the four phases of a training step — batch indexing,
      forward (node_log_prob), backward, optimizer.step — averaged over many
      steps, and
  (2) dump a torch.profiler self-CPU-time op table.

Run from experiments/:  uv run python profile_epoch.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from common import build_spec
from tramdag import CausalFlowDAG, ContinuousNode

REPO = Path(__file__).resolve().parents[1]


def stroke():
    obs = pd.read_csv(REPO / "data" / "magic-mrclean" / "ls" / "obs.csv")
    return obs, build_spec("ls"), 512


def vaca():
    obs = pd.read_csv(REPO / "data" / "vaca" / "obs.csv")
    spec = {"x1": ContinuousNode(),
            "x2": ContinuousNode(parents={"x1": "ci"}),
            "x3": ContinuousNode(parents={"x1": "ci", "x2": "ci"})}
    return obs.iloc[:int(len(obs) * 0.9)], spec, 512


def manual_breakdown(flow, vals, n, bs, n_steps=200):
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    order = flow.order
    flow.train()
    t = {"index": 0.0, "forward": 0.0, "backward": 0.0, "step": 0.0}
    # warm up
    for _ in range(5):
        idx = torch.randperm(n)[:bs]
        batch = {k: v[idx] for k, v in vals.items()}
        loss = torch.stack([-lp.mean() for lp in
                            flow.node_log_prob(batch).values()]).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(n_steps):
        t0 = time.perf_counter()
        idx = torch.randperm(n)[:bs]
        batch = {k: v[idx] for k, v in vals.items()}
        t1 = time.perf_counter()
        per = flow.node_log_prob(batch)
        loss = torch.stack([-lp.mean() for lp in per.values()]).sum()
        t2 = time.perf_counter()
        opt.zero_grad(); loss.backward()
        t3 = time.perf_counter()
        opt.step()
        t4 = time.perf_counter()
        t["index"] += t1 - t0
        t["forward"] += t2 - t1
        t["backward"] += t3 - t2
        t["step"] += t4 - t3
    tot = sum(t.values())
    return {k: (v / n_steps * 1e3, 100 * v / tot) for k, v in t.items()}, tot / n_steps * 1e3


def main():
    for name, loader in [("stroke-ls", stroke), ("vaca-ci", vaca)]:
        train, spec, bs = loader()
        torch.manual_seed(0)
        flow = CausalFlowDAG(spec)
        flow._set_ranges(train)
        vals = flow._tensorize(train)
        n = len(train)
        print(f"\n{'='*64}\n{name}  (n={n}, batch={bs}, {len(flow.order)} nodes, "
              f"steps/epoch={ -(-n//bs) })\n{'='*64}")
        bd, per_step = manual_breakdown(flow, vals, n, bs)
        print(f"per-step wall: {per_step:.3f} ms")
        for k, (ms, pct) in bd.items():
            print(f"  {k:9s} {ms:7.3f} ms  {pct:5.1f}%")

        # torch.profiler op table over a handful of steps
        opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            for _ in range(30):
                idx = torch.randperm(n)[:bs]
                batch = {k: v[idx] for k, v in vals.items()}
                loss = torch.stack([-lp.mean() for lp in
                                    flow.node_log_prob(batch).values()]).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
