"""Tests for the observational ITE benchmark DGP (simulations/ite_observational).

Pins the generator (reproducibility, frozen-CSV contract, the TRAM latent
identities and h_y round-trip, the Monte-Carlo ATE) and — marked slow — checks
that an all-CI TRAM-DAG recovers individual treatment effects.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tramdag.simulations import REGISTRY, ITEObservational
from tramdag.simulations.ite_observational import COLUMNS, h_y, h_y_inverse

DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------- generator
def test_registry_contains_ite():
    assert REGISTRY["ite-observational"] is ITEObservational


def test_generator_reproducible():
    a = ITEObservational(seed=42).observational(500)
    b = ITEObservational(seed=42).observational(500)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["X1", "X2", "X3", "Tr", "X5", "X6", "Y"]


def test_h_y_round_trip():
    x = np.linspace(-4, 4, 401)            # spans both tails and the core
    np.testing.assert_allclose(h_y_inverse(h_y(x)), x, atol=1e-9)


def test_do_clamps_and_propagates_through_mediators():
    gen = ITEObservational(seed=0, scenario=1)
    lat = gen.draw_latents(3000, np.random.default_rng(0))
    base = gen.simulate(latents=lat)
    cf = gen.simulate(latents=lat, do={"Tr": 1.0})
    assert (cf["Tr"] == 1.0).all()
    # treatment flows through the mediators X5 -> X6 and into Y
    assert not np.allclose(cf["X5"], base["X5"])
    assert not np.allclose(cf["X6"], base["X6"])
    assert not np.allclose(cf["Y"], base["Y"])


def test_tram_latent_identities_are_logistic():
    """Each node's structural equation inverts to a standard-logistic latent."""
    gen = ITEObservational(seed=1, scenario=1)
    df = gen.observational(20_000)
    Tr = df["Tr"].to_numpy()
    u5 = 2.5 * df["X5"].to_numpy() + 0.8 * Tr
    u6 = 4.0 * df["X6"].to_numpy() - 0.5 * df["X5"].to_numpy()
    X = df[["X1", "X2", "X3", "X5", "X6"]].to_numpy()
    u7 = h_y(df["Y"].to_numpy()) + gen._logit_outcome(X, Tr)
    for u in (u5, u6, u7):
        assert stats.kstest(u, "logistic").pvalue > 0.01


def test_treatment_assignment_rate():
    df = ITEObservational(seed=2).observational(50_000)
    assert 0.58 < df["Tr"].mean() < 0.63   # sigmoid(0.5 - 0.5 X1 + 0.3 X2), E~0.6


def test_ite_median_matches_definition():
    gen = ITEObservational(seed=3, scenario=1)
    df = gen.with_truth(5000)
    assert {"ITE_true", "ITE_median"} <= set(df.columns)
    # both ITE notions are heterogeneous (interaction scenario) and correlated
    assert df["ITE_true"].std() > 0.1
    assert np.corrcoef(df["ITE_true"], df["ITE_median"])[0, 1] > 0.5


def test_scenario_4_is_null_effect():
    df = ITEObservational(seed=4, scenario=4).with_truth(20_000)
    # no direct effect and no interaction, but treatment still acts through the
    # mediators X5 -> X6 -> Y, so the ITE is a small nonzero mediated effect
    assert abs(df["ITE_true"].mean()) < 0.1


def test_true_ate_pinned():
    t = ITEObservational(seed=123, scenario=1).true_ate(mc_n=200_000)
    assert abs(t["ate_true"] - (-0.576)) < 0.02
    assert abs(t["ate_median"] - (-0.621)) < 0.02


# ------------------------------------------------------------ frozen contract
def test_frozen_csv_contract():
    vdir = DATA / "ite-observational"
    truth = json.loads((vdir / "truth.json").read_text())
    frozen = pd.read_csv(vdir / "obs.csv")
    regen = ITEObservational(seed=truth["seed"],
                             scenario=truth["scenario"]).observational(truth["n_obs"])
    assert len(frozen) == truth["n_obs"]
    for c in frozen.columns:
        np.testing.assert_allclose(frozen[c].to_numpy(dtype=float),
                                   regen[c].to_numpy(dtype=float), atol=1e-9)


# ------------------------------------------------------------------ fit test
@pytest.mark.slow
def test_all_ci_flow_recovers_ite():
    """An all-CI S-learner TRAM-DAG recovers individual + average effects."""
    from tramdag import CausalFlowDAG, ContinuousNode, I, OrdinalNode

    gen = ITEObservational(seed=123, scenario=1)
    train = gen.observational(20_000)
    test = gen.with_truth(4000, seed_offset=50)

    spec = {"X1": ContinuousNode(), "X2": ContinuousNode(), "X3": ContinuousNode(),
            "Tr": OrdinalNode(levels=2, terms=[I("X1", "X2")]),
            "X5": ContinuousNode(terms=[I("Tr")]),
            "X6": ContinuousNode(terms=[I("X5")]),
            "Y":  ContinuousNode(terms=[I("Tr", "X1", "X2", "X3", "X5", "X6")])}
    flow = CausalFlowDAG(spec, seed=1)
    flow.fit(train, epochs=600, learning_rate=1e-2, schedule="plateau",
             plateau_patience=25, verbose=0)

    u = flow.abduct(test[list(COLUMNS)], seed=0)
    ite_pred = (flow.sample(do={"Tr": 1.0}, u=u)["Y"].to_numpy()
                - flow.sample(do={"Tr": 0.0}, u=u)["Y"].to_numpy())
    ite_true = test["ITE_true"].to_numpy()

    assert abs(ite_pred.mean() - ite_true.mean()) < 0.05          # ATE
    assert np.corrcoef(ite_pred, ite_true)[0, 1] > 0.8            # individual
