"""Exp #3 probe — can torch.compile fuse the dispatch-bound per-node loop?

Exp #1 showed training is op-dispatch bound. torch.compile should fuse the
elementwise chain into far fewer kernels. Two questions decide viability:
  (1) steady-state per-step speedup vs eager, and
  (2) one-time compile warmup cost (must amortize within a short fit).

Measures both for each workload on CPU. Diagnostic only (no library change).
Run from experiments/:  uv run python probe_compile.py
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import pandas as pd
import torch

import torch._functorch.config as _ffconfig
_ffconfig.donated_buffer = False  # zuko differentiates inside forward (double backward)

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


def loss_fn(flow, batch):
    return torch.stack([-lp.mean() for lp in
                        flow.node_log_prob(batch).values()]).sum()


def bench(loader, compiled: bool, n_steps=300):
    train, spec, bs = loader()
    torch.manual_seed(0)
    flow = CausalFlowDAG(spec)
    flow._set_ranges(train)
    vals = flow._tensorize(train)
    n = len(train)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    flow.train()

    fn = torch.compile(loss_fn, dynamic=True) if compiled else loss_fn

    # fixed-size batch (full 512) to avoid recompiles on the ragged last batch
    def one_step():
        idx = torch.randperm(n)[:bs]
        batch = {k: v[idx] for k, v in vals.items()}
        loss = fn(flow, batch)
        opt.zero_grad(); loss.backward(); opt.step()
        return float(loss)

    t0 = time.perf_counter()
    one_step()                      # includes compile warmup if compiled
    warmup = time.perf_counter() - t0
    for _ in range(5):
        one_step()
    dt = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        one_step()
        dt.append(time.perf_counter() - t0)
    return statistics.median(dt) * 1e3, warmup


def main():
    torch.set_num_threads(torch.get_num_threads())
    for name, loader in [("stroke-ls", stroke), ("vaca-ci", vaca)]:
        eager_ms, _ = bench(loader, compiled=False)
        comp_ms, warmup = bench(loader, compiled=True)
        speedup = eager_ms / comp_ms
        print(f"\n{name}:")
        print(f"  eager    {eager_ms:7.3f} ms/step")
        print(f"  compiled {comp_ms:7.3f} ms/step   (speedup {speedup:.2f}x)")
        print(f"  compile warmup (first step): {warmup:.1f} s")
        be = warmup / max(eager_ms - comp_ms, 1e-9) * 1e3
        print(f"  break-even: ~{be:.0f} steps to pay back warmup" if speedup > 1
              else "  no steady-state speedup -> never breaks even")


if __name__ == "__main__":
    main()
