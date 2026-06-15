"""Exp #2 — does CPU thread count matter for these tiny per-node models?

Exp #1 showed training is op-dispatch bound: thousands of µs-scale elementwise
ops per step. On a 20-core box, parallelizing each tiny op across all threads may
cost more in fork/join than it saves. Sweep torch.set_num_threads and time the
median per-step wall-clock for both workloads.

Run from experiments/:  uv run python sweep_threads.py
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import pandas as pd
import torch

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


def time_per_step(loader, n_steps=300):
    train, spec, bs = loader()
    torch.manual_seed(0)
    flow = CausalFlowDAG(spec)
    flow._set_ranges(train)
    vals = flow._tensorize(train)
    n = len(train)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    flow.train()
    for _ in range(10):  # warm up
        idx = torch.randperm(n)[:bs]
        batch = {k: v[idx] for k, v in vals.items()}
        loss = torch.stack([-lp.mean() for lp in
                            flow.node_log_prob(batch).values()]).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    dt = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        idx = torch.randperm(n)[:bs]
        batch = {k: v[idx] for k, v in vals.items()}
        loss = torch.stack([-lp.mean() for lp in
                            flow.node_log_prob(batch).values()]).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        dt.append(time.perf_counter() - t0)
    return statistics.median(dt) * 1e3  # ms


def main():
    default = torch.get_num_threads()
    print(f"default torch threads = {default}\n")
    print(f"{'threads':>7} | {'stroke-ls ms':>12} | {'vaca-ci ms':>11}")
    print("-" * 36)
    for nt in [1, 2, 4, 6, 8, 12, 16, default]:
        torch.set_num_threads(nt)
        s = time_per_step(stroke)
        v = time_per_step(vaca)
        print(f"{nt:>7} | {s:>12.3f} | {v:>11.3f}")


if __name__ == "__main__":
    main()
