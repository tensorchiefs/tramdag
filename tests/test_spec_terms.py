"""Round-1 tests for the term-formula notation (I/LS/CS) and its equivalence
to the deprecated ``parents={...}`` dict. Single-parent only — multi-parent
(joint) terms must raise ``NotImplementedError`` until a later round."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict


def _toy_df(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = -x1 + rng.normal(size=n)
    x3 = x1 + 0.25 * x2 + rng.normal(size=n)
    y = rng.integers(0, 4, size=n).astype(float)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "Y": y})


def _legacy_spec():
    with warnings.catch_warnings():           # silence the intended deprecation
        warnings.simplefilter("ignore", DeprecationWarning)
        return {
            "X1": ContinuousNode(),
            "X2": ContinuousNode(parents={"X1": "ls"}),
            "X3": ContinuousNode(parents={"X1": "ci", "X2": "cs"}),
            "Y":  OrdinalNode(levels=4, parents={"X3": "ls"}),
        }


def _terms_spec():
    return {
        "X1": ContinuousNode(),
        "X2": ContinuousNode(terms=[LS("X1")]),
        "X3": ContinuousNode(terms=[I("X1"), CS("X2")]),
        "Y":  OrdinalNode(levels=4, terms=[LS("X3")]),
    }


# --------------------------------------------------------------- equivalence
def test_terms_equivalent_to_legacy_dict():
    """Both front-ends build the *same* flow: identical parameter structure and
    identical per-node log-likelihood on a fixed batch."""
    df = _toy_df()
    flow_legacy = CausalFlowDAG(_legacy_spec(), seed=0)
    flow_terms = CausalFlowDAG(_terms_spec(), seed=0)

    sd_a, sd_b = flow_legacy.state_dict(), flow_terms.state_dict()
    assert sd_a.keys() == sd_b.keys()

    vals_a = flow_legacy.node_log_prob(flow_legacy._tensorize(df))
    vals_b = flow_terms.node_log_prob(flow_terms._tensorize(df))
    assert vals_a.keys() == vals_b.keys()
    for k in vals_a:
        assert torch.allclose(vals_a[k], vals_b[k]), k


def test_node_internal_structure_matches():
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    assert flow.nodes["X3"].ci_parents == ["X1"]          # I("X1") -> ci-parent
    assert set(flow.nodes["X3"].shifts) == {"X2"}         # CS("X2") -> shift
    assert flow.nodes["X3"].parents == ("X1", "X2")       # ordered parent names


# ---------------------------------------------------------------- validation
def test_deprecation_warning_on_parents_dict():
    with pytest.warns(DeprecationWarning):
        ContinuousNode(parents={"X1": "ls"})


def test_terms_and_parents_mutually_exclusive():
    with pytest.raises(ValueError):
        ContinuousNode(terms=[LS("X1")], parents={"X1": "ls"})


def test_ls_requires_exactly_one_parent():
    with pytest.raises(ValueError):
        LS("X1", "X2")


@pytest.mark.parametrize("term,n_shift,n_ci", [(CS("X1", "X2"), 1, 0),
                                               (I("X1", "X2"), 0, 2)])
def test_joint_terms_build(term, n_shift, n_ci):
    """Joint (multi-parent) terms build one network over the parent group."""
    spec = {"X1": ContinuousNode(), "X2": ContinuousNode(),
            "X3": ContinuousNode(terms=[term])}
    node = CausalFlowDAG(spec, seed=0).nodes["X3"]
    assert len(node.shifts) == n_shift          # joint CS -> a single shift module
    assert len(node.ci_parents) == n_ci         # joint I  -> both parents pooled


def test_duplicate_parent_across_terms_raises():
    spec = {"X1": ContinuousNode(),
            "X3": ContinuousNode(terms=[LS("X1"), CS("X1")])}
    with pytest.raises(ValueError):
        CausalFlowDAG(spec)


def test_cycle_detected():
    spec = {"A": ContinuousNode(terms=[LS("B")]),
            "B": ContinuousNode(terms=[LS("A")])}
    with pytest.raises(ValueError):
        CausalFlowDAG(spec)


# ------------------------------------------------------------- serialization
def test_serialization_roundtrip_terms():
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    spec2 = spec_from_dict(spec_to_dict(flow.spec))
    flow2 = CausalFlowDAG(spec2, seed=0)
    df = _toy_df()
    a = flow.node_log_prob(flow._tensorize(df))
    b = flow2.node_log_prob(flow2._tensorize(df))
    for k in a:
        assert torch.allclose(a[k], b[k]), k


def test_legacy_parents_checkpoint_still_loads():
    """A checkpoint serialized in the old ``parents``-dict layout must still
    rebuild (no deprecation warning, since we translate to terms directly)."""
    legacy = {
        "X1": {"kind": "continuous", "parents": {}, "transform": "bernstein",
               "transform_kwargs": {}},
        "X2": {"kind": "continuous", "parents": {"X1": "ci"}, "transform": "bernstein",
               "transform_kwargs": {}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # must NOT warn
        spec = spec_from_dict(legacy)
    flow = CausalFlowDAG(spec, seed=0)
    assert flow.nodes["X2"].ci_parents == ["X1"]


# -------------------------------------------------------------------- matrix
def test_to_matrix():
    m = CausalFlowDAG(_terms_spec(), seed=0).to_matrix()
    assert m.loc["X1", "X2"] == "LS"
    assert m.loc["X1", "X3"] == "CI"     # I("X1")
    assert m.loc["X2", "X3"] == "CS"
    assert m.loc["X3", "Y"] == "LS"
    assert m.loc["X1", "X1"] == ""       # no self-edge
