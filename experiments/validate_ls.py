"""Validation: the all-`ls` flow's outcome node is exactly a proportional-odds
model, so its SGD fit must agree with the statsmodels MLE on the same data.

Fits OrderedModel(mRS_3m ~ Age + mRS_pre + NIHSSa + T) on the identical 80%
train split (mRS_pre one-hot, as the flow encodes ordinal parents), compares
coefficients, and computes the MLE's analytic ATE on the RCT covariates —
the repo's established equal-data reference (~ +0.055).

Usage: uv run python validate_ls.py [all_ls|all_ls_long]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from common import RESULTS_ROOT, load_magic, load_rct, split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zuko_dag import CausalFlowDAG  # noqa: E402


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Same encoding as the flow: continuous raw, ordinal one-hot (all levels)."""
    X = pd.DataFrame(index=df.index)
    X["Age"] = df["Age"].values
    for k in range(6):
        X[f"mRS_pre_{k}"] = (df["mRS_pre"].values == k).astype(float)
    X["NIHSSa"] = df["NIHSSa"].values
    X["T"] = df["T"].values
    # drop one one-hot column: with cutpoints, the full one-hot is unidentified
    # in the MLE (the flow absorbs this into the cutpoints; shifts only matter
    # up to a constant, so coefficient *differences* are the comparable thing)
    return X.drop(columns=["mRS_pre_0"])


def main(name: str = "all_ls"):
    train_df, _, _ = split(load_magic())
    res = OrderedModel(train_df["mRS_3m"].astype(int), design(train_df),
                       distr="logit").fit(method="bfgs", disp=False)
    print("=== statsmodels OrderedModel (same 80% train split) ===")
    print(res.params[["Age", "NIHSSa", "T"]].round(4).to_string())

    # analytic ATE on RCT covariates: P(good | do(T=t)) = P(mRS_3m <= 2 | X, T=t)
    rct = load_rct()
    p0 = res.model.predict(res.params, exog=design(rct.assign(T=0)).values, which="prob")
    p1 = res.model.predict(res.params, exog=design(rct.assign(T=1)).values, which="prob")
    ate_mle = (p1[:, :3].sum(axis=1) - p0[:, :3].sum(axis=1)).mean()
    print(f"\nMLE ATE on RCT covariates: {ate_mle:+.4f}   (repo reference ~ +0.055)")

    # --- compare with the fitted all-ls flow, if available ---
    ckpt = RESULTS_ROOT / name / "flow.pt"
    if not ckpt.exists():
        print(f"\n(no {ckpt} yet - run the experiment first for the comparison)")
        return
    print(f"\n(flow checkpoint: {ckpt})")
    flow = CausalFlowDAG.load(ckpt)
    node = flow.nodes["mRS_3m"]
    w_age = float(node.shifts["Age"].weight.detach())
    w_nih = float(node.shifts["NIHSSa"].weight.detach())
    # ordinal parents: shift = w @ onehot; only differences to level 0 identified
    w_pre = node.shifts["mRS_pre"].weight.detach().numpy().ravel()
    w_t = node.shifts["T"].weight.detach().numpy().ravel()

    print("\n=== all-ls flow (outcome node mRS_3m) vs MLE ===")
    rows = [("Age", w_age, res.params["Age"]),
            ("NIHSSa", w_nih, res.params["NIHSSa"]),
            ("T (=1 vs 0)", w_t[1] - w_t[0], res.params["T"])]
    for k in range(1, 6):
        rows.append((f"mRS_pre_{k} (vs 0)", w_pre[k] - w_pre[0],
                     res.params[f"mRS_pre_{k}"]))
    print(f"{'coefficient':<20}{'flow':>10}{'MLE':>10}{'diff':>10}")
    for name, a, b in rows:
        print(f"{name:<20}{a:>10.4f}{b:>10.4f}{a - b:>10.4f}")

    pf0 = flow.pmf(rct, node="mRS_3m", do={"T": 0})
    pf1 = flow.pmf(rct, node="mRS_3m", do={"T": 1})
    ate_flow = (pf1[:, :3].sum(axis=1) - pf0[:, :3].sum(axis=1)).mean()
    print(f"\nATE on RCT covariates:  flow {ate_flow:+.4f}   MLE {ate_mle:+.4f}")
    print(f"max per-patient |P(good|do(T=1)) diff|: "
          f"{np.abs(pf1[:, :3].sum(axis=1) - p1[:, :3].sum(axis=1)).max():.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all_ls")
