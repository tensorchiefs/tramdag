"""Verification V1b — CPU-parallelism control for the GPU crossover.

Follow-up to verify_gpu.py, answering: does the CPU side win the small-batch
comparison only because it quietly uses many cores? Pins torch to 1 vs all-20
threads across the batch sweep, and samples actual CPU utilization (/proc/stat,
overall + per-core) during the runs — the CPU analogue of the GPU's 23% SM number.

If the crossover batch barely moves between 1 and 20 threads, the GPU result is
not an artifact of CPU multicore. Default (3-node) model — the regime the original
'tiny models lose on GPU' claim is about.

Usage:  uv run python verify_gpu_cpu_cores.py
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from verify_gpu import make_spec, make_data, iters_for

OUT = Path(__file__).resolve().parents[1] / "docs" / "research" / "gpu-verify"
BATCHES = [4096, 16384, 65536, 262144]
THREADS = [1, 10, 20]              # 10 = this box's torch default (the sweet spot)
SEEDS = [0, 1, 2]
NCORES = __import__("os").cpu_count()


def _build(size, batch, device, seed):
    torch.manual_seed(seed)
    spec = make_spec(size)
    flow = CausalFlowDAG_on(spec, device)
    df = make_data(spec, batch, seed)
    flow._set_ranges(df)
    vals = flow._tensorize(df)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    flow.train()

    def step():
        opt.zero_grad()
        torch.stack([-v.mean() for v in flow.node_log_prob(vals).values()]).sum().backward()
        opt.step()
    return step


def CausalFlowDAG_on(spec, device):
    from tramdag import CausalFlowDAG
    return CausalFlowDAG(spec, device=device)


def per_step_ms(size, batch, device, seed, threads=None):
    if device == "cpu" and threads:
        torch.set_num_threads(threads)
    step = _build(size, batch, device, seed)
    warmup, timed = iters_for(batch)
    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / timed * 1e3


# ---- /proc/stat CPU utilization sampler -----------------------------------
def _snapshot():
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            p = line.split()
            if p[0] == "cpu" or p[0][3:].isdigit():
                v = list(map(int, p[1:]))
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                out[p[0]] = (idle, sum(v))
    return out


def sample_cpu_util(size, batch, threads, seconds=6.0):
    """Run a sustained CPU loop while sampling /proc/stat; return (overall%,
    busy-core-equivalent, n_cores_>50%)."""
    torch.set_num_threads(threads)
    step = _build(size, batch, "cpu", 0)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            step()
    a = _snapshot()
    th = threading.Thread(target=worker)
    th.start()
    time.sleep(seconds)
    b = _snapshot()
    stop.set(); th.join()
    overall = util(a["cpu"], b["cpu"])
    per = [util(a[k], b[k]) for k in a if k != "cpu"]
    busy_equiv = overall / 100 * NCORES
    n_busy = sum(1 for u in per if u > 50)
    return overall, busy_equiv, n_busy


def util(a, b):
    di, dt = b[0] - a[0], b[1] - a[1]
    return 100.0 * (1 - di / dt) if dt else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"cores={NCORES}  torch default threads={torch.get_num_threads()}\n")
    print("=== per-step ms: cpu@{1,10,20} threads vs cuda (default model); "
          "speedup = best-cpu/cuda ===")
    hdr = " ".join(f"cpu@{t:<2}".rjust(9) for t in THREADS)
    print(f"{'batch':>8} {hdr} {'cuda':>9}  {'bestcpu/gpu':>11}")
    rows = []
    for batch in BATCHES:
        cpu = {t: float(np.median([per_step_ms("default", batch, "cpu", s, t)
                                   for s in SEEDS])) for t in THREADS}
        gpu = float(np.median([per_step_ms("default", batch, "cuda", s) for s in SEEDS]))
        best = min(cpu.values())
        rows.append((batch, cpu, gpu))
        cols = " ".join(f"{cpu[t]:9.2f}" for t in THREADS)
        print(f"{batch:>8} {cols} {gpu:9.2f}  {best/gpu:11.2f}"
              f"  {'GPU WINS' if best/gpu > 1 else 'cpu faster'}", flush=True)

    print("\n=== CPU utilization during a sustained CPU loop (overall% / "
          "busy-core-equiv / #cores>50%) ===")
    for (batch, threads) in [(4096, 1), (4096, 20), (262144, 1), (262144, 20)]:
        ov, eq, nb = sample_cpu_util("default", batch, threads)
        print(f"  batch={batch:>7} threads={threads:>2}: "
              f"{ov:5.1f}%  ~{eq:4.1f} cores busy  ({nb} cores >50%)", flush=True)

    # crossover batch per thread setting (first batch where cpu_t/gpu < 1)
    cross = {}
    for t in THREADS:
        cross[t] = next((b for b, cpu, g in rows if cpu[t] / g <= 1), None)
    same = len(set(cross.values())) == 1
    verdict = ("GPU-crossover batch (first batch GPU wins) by CPU threads: "
               + ", ".join(f"{t}t->{cross[t]}" for t in THREADS) + ". "
               + ("Same bucket across thread counts -> CPU multicore does NOT "
                  "explain/move the GPU win."
                  if same else
                  "Crossover shifts with CPU threads -> CPU parallelism matters "
                  "near the boundary; GPU still wins decisively at large batch."))
    print("\n=== VERDICT ===\n" + verdict)
    (OUT / "cpu_cores_verdict.txt").write_text(verdict + "\n")
    torch.set_num_threads(10)


if __name__ == "__main__":
    main()
