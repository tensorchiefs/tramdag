"""Exp #7 — classical seeding of the flexible model (seed_from_classical).

Fit the all-`ls` model with fit_classical (exact MLE, fast), seed a flexible
(ci/cs) model from it (residual base = classical solution, MLP near-zero), then
continue with Adam. Two hypotheses:

  SPEED   — seeded reaches a target val-NLL in fewer epochs than cold (it doesn't
            re-discover the linear/log-odds structure).
  QUALITY — does it land in a better basin? On magic-mrclean/nl the flexible MLE
            overfits observational confounding (ATE ~+0.076; needs restore_best to
            recover ~+0.10 vs truth +0.104). Does classical-anchoring recover the
            true ATE WITHOUT early stopping? (headline.)

Reads only obs.csv/rct.csv; true ATE is the documented constant (CLAUDE.md /
simulator known truth), NOT read from truth.json.

Usage:  uv run python exp_seed_classical.py [seeds...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import build_spec, NODES, split
from tramdag import CausalFlowDAG

REPO = Path(__file__).resolve().parents[1]
NL = REPO / "data" / "magic-mrclean" / "nl"
TRUE_ATE = 0.104          # nl, documented known truth (CLAUDE.md); NOT from truth.json
REF_COLD, REF_RESTORE = 0.076, 0.10
SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
PHASES = [(1500, 1e-2), (500, 1e-3)]


def load():
    obs = pd.read_csv(NL / "obs.csv")[NODES]
    rct = pd.read_csv(NL / "rct.csv")[NODES].astype(
        {"NIHSSa": float, "mRS_pre": int, "T": int, "mRS_3m": int})
    return obs, rct


def ate(flow, rct):
    p0 = flow.pmf(rct, node="mRS_3m", do={"T": 0})
    p1 = flow.pmf(rct, node="mRS_3m", do={"T": 1})
    return float((p1[:, :3].sum(1) - p0[:, :3].sum(1)).mean())


def make_flexible(train, seed, seeded):
    torch.manual_seed(seed)
    flow = CausalFlowDAG(build_spec("flexible"))
    if seeded:
        clf = CausalFlowDAG(build_spec("ls"))
        clf.fit_classical(train, verbose=False)
        flow.seed_from_classical(clf)
    return flow


def run_arm(train, val, rct, seed, seeded, restore_best):
    flow = make_flexible(train, seed, seeded)
    for ep, lr in PHASES:
        flow.fit(train, val, epochs=ep, learning_rate=lr, batch_size=512,
                 verbose=0, restore_best=restore_best)
    val_nll = np.array([sum(e.values()) for e in flow.history["val"]])
    return ate(flow, rct), val_nll


def main():
    obs, rct = load()
    train, val, _ = split(obs)
    print(f"magic-mrclean/nl  n_train={len(train)}  n_rct={len(rct)}  "
          f"TRUE ATE={TRUE_ATE:+.3f} (refs: cold~{REF_COLD:+.3f}, "
          f"restore~{REF_RESTORE:+.3f})\n")

    arms = [("cold-MLE", False, False),
            ("cold+restore_best", False, True),
            ("seeded-MLE", True, False),
            ("seeded+restore_best", True, True)]
    results = {}
    speed = {}
    for label, seeded, restore in arms:
        ates = []
        for seed in SEEDS:
            a, vnll = run_arm(train, val, rct, seed, seeded, restore)
            ates.append(a)
            if not restore:                       # speed readout on the MLE arms
                speed.setdefault(label, []).append(vnll)
        results[label] = ates
        med = float(np.median(ates))
        err = med - TRUE_ATE
        print(f"  {label:22s} ATE = {med:+.4f}  (err {err:+.4f} vs truth)  "
              f"per-seed {[round(x,4) for x in ates]}")

    # --- speed: epochs to reach (cold final val-NLL + tol), cold vs seeded ---
    print("\n=== speed: epochs to target val-NLL (cold-MLE vs seeded-MLE) ===")
    cold = speed.get("cold-MLE"); seed_ = speed.get("seeded-MLE")
    if cold and seed_:
        for tol in (5e-3, 1e-2):
            def eptt(traj, ref):
                hit = np.nonzero(traj <= ref + tol)[0]
                return int(hit[0]) + 1 if len(hit) else None
            refs = [min(c.min(), s.min()) for c, s in zip(cold, seed_)]
            ce = [eptt(c, r) for c, r in zip(cold, refs)]
            se = [eptt(s, r) for s, r in zip(seed_, refs)]
            def med(xs):
                xs = [x for x in xs if x is not None]
                return int(np.median(xs)) if xs else None
            print(f"  tol {tol:.0e}: cold epochs {med(ce)} ({ce})  "
                  f"seeded epochs {med(se)} ({se})")
        print(f"  val-NLL at epoch 1: cold {[round(float(c[0]),3) for c in cold]}  "
              f"seeded {[round(float(s[0]),3) for s in seed_]}")

    print("\n=== VERDICT ===")
    cm = float(np.median(results["cold-MLE"]))
    sm = float(np.median(results["seeded-MLE"]))
    print(f"  cold MLE {cm:+.4f} -> seeded MLE {sm:+.4f} vs truth {TRUE_ATE:+.3f}. "
          + ("Seeding recovers ATE without early stopping."
             if abs(sm - TRUE_ATE) < abs(cm - TRUE_ATE) - 0.01
             else "Seeding does NOT materially improve ATE recovery (null on quality)."))


if __name__ == "__main__":
    main()
