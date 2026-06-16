"""Independent re-test of the 'CUDA slower than CPU' finding (Exp #0).

Standalone — does NOT import bench_training.py (the point is a measurement
independent of the harness under question). Controls for the artifacts that would
make a GPU look artificially slow, then runs the decisive work-per-step crossover
sweep.

Artifact controls
  - untimed warm-up iterations (CUDA context + lazy autotune happen on first calls)
  - torch.cuda.synchronize() bracketing every timed region (measure execution, not
    async launch)
  - assert model params AND data tensors live on cuda (no per-step H<->D copies)

Sweep
  - batch in {4096, 16384, 65536, 262144}
  - model size in {default (3 nodes, 20-coeff Bernstein, ci), scaled (6 nodes,
    32-coeff Bernstein, dense ci)}  -> grows work-per-step on both axes
  - per (batch, size): median per-step wall over >=3 seeds, cpu vs cuda
  - speedup = cpu_per_step / cuda_per_step  (>1 == GPU wins)

Mechanism
  - one sustained cuda loop on the DEFAULT spec (the regime the original claim is
    about) sampled with `nvidia-smi dmon -s u`; low SM% with gaps == dispatch-bound.

Usage:  uv run python verify_gpu.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tramdag import CausalFlowDAG, ContinuousNode

OUT = Path(__file__).resolve().parents[1] / "docs" / "research" / "gpu-verify"
BATCHES = [4096, 16384, 65536, 262144]
SIZES = ["default", "scaled"]
SEEDS = [0, 1, 2]


def make_spec(size: str) -> dict:
    if size == "default":   # == the vaca-ci workload shape (3 nodes)
        return {"x1": ContinuousNode(),
                "x2": ContinuousNode(parents={"x1": "ci"}),
                "x3": ContinuousNode(parents={"x1": "ci", "x2": "ci"})}
    # scaled: more nodes, more Bernstein coeffs, dense ci -> more work-per-step
    kw = dict(transform="bernstein", transform_kwargs={"n_coeffs": 32})
    spec = {"x1": ContinuousNode(**kw)}
    prev = ["x1"]
    for i in range(2, 7):                       # x2..x6 -> 6 nodes total
        name = f"x{i}"
        spec[name] = ContinuousNode(parents={p: "ci" for p in prev}, **kw)
        prev.append(name)
    return spec


def make_data(spec: dict, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({name: rng.standard_normal(n).astype(np.float32)
                         for name in spec})


def iters_for(batch: int) -> tuple[int, int]:
    """(warmup, timed) — fewer timed steps at huge batch to bound wall-clock."""
    timed = max(5, min(30, 2_000_000 // batch))
    return max(3, timed // 3), timed


def per_step_seconds(size: str, batch: int, device: str, seed: int) -> float:
    torch.manual_seed(seed)
    spec = make_spec(size)
    flow = CausalFlowDAG(spec, device=device)
    df = make_data(spec, batch, seed)
    flow._set_ranges(df)
    vals = flow._tensorize(df)                  # tensors created on flow.device
    if device == "cuda":                        # artifact guard: everything on GPU
        assert next(flow.parameters()).is_cuda, "params not on cuda"
        assert all(v.is_cuda for v in vals.values()), "data not on cuda"
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    flow.train()

    def step():
        opt.zero_grad()
        per_node = flow.node_log_prob(vals)
        loss = torch.stack([-v.mean() for v in per_node.values()]).sum()
        loss.backward()
        opt.step()

    warmup, timed = iters_for(batch)
    try:
        for _ in range(warmup):                 # exclude CUDA context/autotune
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(timed):
            step()
        if device == "cuda":
            torch.cuda.synchronize()            # measure execution, not launch
        return (time.perf_counter() - t0) / timed
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")


def sweep() -> pd.DataFrame:
    rows = []
    for size in SIZES:
        for batch in BATCHES:
            rec = {"size": size, "batch": batch}
            for device in ("cpu", "cuda"):
                ts = []
                for seed in SEEDS:
                    try:
                        ts.append(per_step_seconds(size, batch, device, seed))
                    except RuntimeError as e:
                        print(f"  ! {size} b={batch} {device} seed{seed}: {e}")
                        ts.append(float("nan"))
                rec[f"{device}_ms"] = float(np.nanmedian(ts)) * 1e3
            su = rec["cpu_ms"] / rec["cuda_ms"] if rec["cuda_ms"] > 0 else float("nan")
            rec["speedup"] = su
            rows.append(rec)
            print(f"  {size:8s} batch={batch:7d}: cpu {rec['cpu_ms']:8.2f} ms  "
                  f"cuda {rec['cuda_ms']:8.2f} ms  speedup(cpu/cuda) {su:5.2f}"
                  f"  {'GPU WINS' if su > 1 else 'cpu faster'}", flush=True)
    return pd.DataFrame(rows)


def gpu_utilization() -> str:
    """Sample SM utilization during a sustained DEFAULT-spec cuda loop (the regime
    the original 'tiny models lose on GPU' claim is about)."""
    spec = make_spec("default")
    torch.manual_seed(0)
    flow = CausalFlowDAG(spec, device="cuda")
    df = make_data(spec, 4096, 0)
    flow._set_ranges(df)
    vals = flow._tensorize(df)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    flow.train()
    for _ in range(20):                          # warmup
        opt.zero_grad()
        torch.stack([-v.mean() for v in flow.node_log_prob(vals).values()]).sum().backward()
        opt.step()
    torch.cuda.synchronize()
    proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-c", "8"],
                            stdout=subprocess.PIPE, text=True)
    t_end = time.perf_counter() + 9
    while time.perf_counter() < t_end:           # keep the GPU fed for the sample
        for _ in range(50):
            opt.zero_grad()
            torch.stack([-v.mean() for v in flow.node_log_prob(vals).values()]).sum().backward()
            opt.step()
        torch.cuda.synchronize()
    out, _ = proc.communicate(timeout=20)
    sms = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                sms.append(int(parts[1]))        # 'sm%' column
            except ValueError:
                pass
    mean_sm = float(np.mean(sms)) if sms else float("nan")
    return f"mean SM utilization during default-spec b=4096 cuda loop: {mean_sm:.0f}% " \
           f"(samples {sms})\n{out}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0)}")
    print("\n=== work-per-step crossover sweep (per-step ms; speedup = cpu/cuda) ===")
    df = sweep()
    df.to_csv(OUT / "crossover.csv", index=False)

    print("\n=== mechanism: GPU utilization ===")
    util = gpu_utilization()
    print(util)
    (OUT / "gpu_util.txt").write_text(util)

    won = df[df["speedup"] > 1]
    if len(won):
        first = won.sort_values("batch").iloc[0]
        verdict = (f"OVERTURNED/qualified: CUDA beats CPU at size={first['size']}, "
                   f"batch={int(first['batch'])} (speedup {first['speedup']:.2f}); "
                   f"crossover exists as work-per-step grows.")
    else:
        verdict = ("CONFIRMED: CUDA never beats CPU across batch up to "
                   f"{max(BATCHES)} x {SIZES} model sizes (max speedup "
                   f"{df['speedup'].max():.2f}); the finding holds.")
    print("\n=== VERDICT ===\n" + verdict)
    (OUT / "verdict.txt").write_text(verdict + "\n")


if __name__ == "__main__":
    main()
