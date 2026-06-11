"""Tests for the magic-mrclean synthetic cohort and the flow's recovery of its
known ground truth, plus a regression check against the committed R reference."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from zuko_dag import CausalFlowDAG, ContinuousNode, OrdinalNode
from zuko_dag.simulations import MagicMrClean

DATA = Path(__file__).resolve().parents[1] / "data" / "magic-mrclean"


def _spec(style: str) -> dict:
    if style == "ls":
        t = {"Age": "ls", "mRS_pre": "ls", "NIHSSa": "ls", "T": "ls"}
    else:
        t = {"Age": "ci", "mRS_pre": "ls", "NIHSSa": "cs", "T": "ls"}
    return {
        "Age": ContinuousNode(transform="bernstein"),
        "mRS_pre": OrdinalNode(levels=6, parents={"Age": t["Age"]}),
        "NIHSSa": ContinuousNode(transform="bernstein",
                                 parents={"Age": t["Age"], "mRS_pre": t["mRS_pre"]}),
        "T": OrdinalNode(levels=2, parents={"Age": t["Age"], "mRS_pre": t["mRS_pre"],
                                            "NIHSSa": t["NIHSSa"]}),
        "mRS_3m": OrdinalNode(levels=7, parents={"Age": t["Age"], "mRS_pre": t["mRS_pre"],
                                                 "NIHSSa": t["NIHSSa"], "T": t["T"]}),
    }


# ----------------------------------------------------------------- generator
def test_generator_schema_and_ranges():
    gen = MagicMrClean(variant="nl", seed=7)
    df = gen.observational(2000)
    assert list(df.columns) == ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]
    assert df["Age"].between(20, 103).all()
    assert df["NIHSSa"].between(6, 42).all()
    assert set(df["T"].unique()) <= {0, 1}
    assert df["mRS_pre"].between(0, 5).all()
    assert df["mRS_3m"].between(0, 6).all()


def test_generator_reproducible():
    a = MagicMrClean(variant="nl", seed=7).observational(500)
    b = MagicMrClean(variant="nl", seed=7).observational(500)
    assert a.equals(b)


def test_rct_breaks_confounding():
    """In the RCT draw, T is independent of its parents (Bernoulli 1/2)."""
    gen = MagicMrClean(variant="nl", seed=7)
    rct = gen.rct(20000)
    assert rct["T"].mean() == pytest.approx(0.5, abs=0.03)
    # T uncorrelated with Age under randomization
    assert abs(np.corrcoef(rct["T"], rct["Age"])[0, 1]) < 0.05


def test_true_ate_positive_and_confounded():
    truth_nl = MagicMrClean(variant="nl", seed=7).true_ate(100_000)
    # known true benefit is positive but much smaller than the naive contrast
    assert truth_nl["true_ate"] > 0.04
    assert truth_nl["naive_obs_diff"] > truth_nl["true_ate"] + 0.1


def test_counterfactual_pair_shares_latents():
    gen = MagicMrClean(variant="nl", seed=7)
    fact, cf = gen.counterfactual_pair(1000, do={"T": 1})
    # non-descendants of T are untouched; T is clamped
    assert (fact["Age"] == cf["Age"]).all()
    assert (fact["NIHSSa"] == cf["NIHSSa"]).all()
    assert (cf["T"] == 1).all()


# ---------------------------------------------------------- frozen CSVs exist
@pytest.mark.parametrize("variant", ["ls", "nl"])
def test_frozen_csvs_present_and_match_truth(variant):
    base = DATA / variant
    obs = pd.read_csv(base / "obs.csv")
    rct = pd.read_csv(base / "rct.csv")
    truth = json.loads((base / "truth.json").read_text())
    assert len(obs) == truth["n_obs"]
    assert len(rct) == truth["n_rct"]
    assert truth["true_ate"] > 0


def _fit_and_ate(style, obs, rct, epochs=(1500, 500)):
    n = len(obs)
    tr, va = obs.iloc[:int(0.85 * n)], obs.iloc[int(0.85 * n):]
    torch.manual_seed(0)
    flow = CausalFlowDAG(_spec(style))
    flow.fit(tr, va, epochs=epochs[0], learning_rate=1e-2, batch_size=256, verbose=0, seed=0)
    flow.fit(tr, va, epochs=epochs[1], learning_rate=1e-3, batch_size=256, verbose=0)
    p0 = flow.pmf(rct, node="mRS_3m", do={"T": 0})[:, :3].sum(axis=1)
    p1 = flow.pmf(rct, node="mRS_3m", do={"T": 1})[:, :3].sum(axis=1)
    return flow, float((p1 - p0).mean())


def _load(variant):
    base = DATA / variant
    cols = ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]
    return (pd.read_csv(base / "obs.csv")[cols],
            pd.read_csv(base / "rct.csv")[cols],
            json.loads((base / "truth.json").read_text()))


# ----------------------------------------------- flow recovers the true ATE
@pytest.mark.parametrize("variant,style,tol", [
    ("ls", "ls", 0.04),        # all-ls DGP, all-ls model: exact up to finite-sample
    ("nl", "flexible", 0.04),  # nl DGP needs the flexible model to recover truth
])
def test_flow_recovers_true_ate(variant, style, tol):
    obs, rct, truth = _load(variant)
    _, ate = _fit_and_ate(style, obs, rct)
    assert ate == pytest.approx(truth["true_ate"], abs=tol)


def test_nl_storyline_all_ls_underestimates_flexible_recovers():
    """The headline simulation result: on the heterogeneous-effect `nl` cohort the
    all-`ls` model (which cannot extrapolate tau(Age) from the older observational
    cohort to the younger trial) undershoots the true ATE, while the flexible
    ci/cs flow recovers it. Both massively de-confound the naive contrast."""
    obs, rct, truth = _load("nl")
    _, ate_ls = _fit_and_ate("ls", obs, rct)
    _, ate_flex = _fit_and_ate("flexible", obs, rct)
    true = truth["true_ate"]
    assert ate_ls < true - 0.015                       # all-ls biased low
    assert abs(ate_flex - true) < abs(ate_ls - true)   # flexible closer to truth
    assert ate_flex == pytest.approx(true, abs=0.04)    # ...and close in absolute terms
    assert ate_ls < truth["naive_obs_diff"] - 0.1      # both far below the confounded naive


# ------------------------------------------------ regression vs R reference
@pytest.mark.parametrize("variant", ["ls", "nl"])
def test_flow_matches_r_reference(variant):
    """The all-ls flow must agree with the committed classical R fit (fit_ls.R)
    on the outcome-node coefficients and the ATE. Skips if R outputs are absent."""
    ref = DATA / variant / "ref_ls"
    if not (ref / "ate.csv").exists():
        pytest.skip(f"R reference not generated yet (run Rscript fit_ls.R {variant})")

    obs, rct, _ = _load(variant)
    flow, ate_flow = _fit_and_ate("ls", obs, rct, epochs=(2000, 800))
    ate_r = pd.read_csv(ref / "ate.csv")["ate"].iloc[0]
    assert ate_flow == pytest.approx(ate_r, abs=0.03)

    # outcome-node continuous-parent coefficients (Age, NIHSSa, T) vs polr
    coefs = pd.read_csv(ref / "coefficients.csv")
    y = coefs[coefs["node"] == "mRS_3m"].set_index("term")["estimate"]
    w_age = float(flow.nodes["mRS_3m"].shifts["Age"].weight.detach())
    w_nih = float(flow.nodes["mRS_3m"].shifts["NIHSSa"].weight.detach())
    w_t = flow.nodes["mRS_3m"].shifts["T"].weight.detach().numpy().ravel()
    assert w_age == pytest.approx(y["Age"], abs=0.03)
    assert w_nih == pytest.approx(y["NIHSSa"], abs=0.03)
    assert (w_t[1] - w_t[0]) == pytest.approx(y["T"], abs=0.06)
