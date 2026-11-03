"""Tests for the term-formula notation (I/LS/CS) — construction, validation,
serialization, and the meta-adjacency view.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CI, CS, LS, VC, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict


# %% private functions -----------------------------------------------------------------
def _toy_df(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = -x1 + rng.normal(size=n)
    x3 = x1 + 0.25 * x2 + rng.normal(size=n)
    y = rng.integers(0, 4, size=n).astype(float)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "Y": y})


def _terms_spec():
    return {
        "X1": ContinuousNode(),
        "X2": ContinuousNode(LS("X1")),
        "X3": ContinuousNode(I("X1") + CS("X2")),
        "Y": OrdinalNode(4, LS("X3")),
    }


# %% public functions ------------------------------------------------------------------
def test_terms_spec_builds_and_scores():
    """A term spec builds a flow whose per-node log-likelihood is finite."""
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    per_node = flow._node_log_prob(flow._tensorize(_toy_df()))
    assert set(per_node) == {"X1", "X2", "X3", "Y"}
    for v in per_node.values():
        assert torch.isfinite(v).all()


def test_node_internal_structure_matches():
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    assert flow.nodes["X3"].intercept.ci_parents == ["X1"]
    assert set(flow.nodes["X3"].shifts) == {"X2"}
    assert flow.nodes["X3"].parents == ("X1", "X2")


def test_ls_requires_exactly_one_parent():
    with pytest.raises(ValueError, match=r"LS\(\) takes exactly one parent"):
        LS("X1", "X2")


@pytest.mark.parametrize(
    ("term", "n_shift", "n_ci"), [(CS("X1", "X2"), 1, 0), (I("X1", "X2"), 0, 2)]
)
def test_joint_terms_build(term, n_shift, n_ci):
    """Joint (multi-parent) terms build one network over the parent group."""
    spec = {
        "X1": ContinuousNode(),
        "X2": ContinuousNode(),
        "X3": ContinuousNode([term]),
    }
    node = CausalFlowDAG(spec, seed=0).nodes["X3"]
    assert len(node.shifts) == n_shift
    assert len(node.intercept.ci_parents) == n_ci


def test_duplicate_parent_across_terms_raises():
    spec = {"X1": ContinuousNode(), "X3": ContinuousNode(LS("X1") + CS("X1"))}
    with pytest.raises(ValueError, match="more than one term"):
        CausalFlowDAG(spec)


def test_cycle_detected():
    spec = {"A": ContinuousNode(LS("B")), "B": ContinuousNode(LS("A"))}
    with pytest.raises(ValueError, match="cycle"):
        CausalFlowDAG(spec)


def test_serialization_roundtrip_terms():
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    spec2 = spec_from_dict(spec_to_dict(flow.spec))
    flow2 = CausalFlowDAG(spec2, seed=0)
    df = _toy_df()
    a = flow._node_log_prob(flow._tensorize(df))
    b = flow2._node_log_prob(flow2._tensorize(df))
    for k in a:
        assert torch.allclose(a[k], b[k]), k


def test_to_matrix_labels_every_term_and_leaves_non_edges_empty():
    """The paper's meta-adjacency view: rows are parents, columns children."""
    spec = {
        "a": ContinuousNode(),
        "b": ContinuousNode(),
        "y": OrdinalNode(3, I("a") + CS("b")),
    }
    m = CausalFlowDAG(spec, seed=0).to_matrix()
    assert list(m.index) == list(m.columns)
    assert m.loc["a", "y"] == "CI"
    assert m.loc["b", "y"] == "CS"
    assert m.loc["y", "a"] == ""
    assert m.loc["a", "b"] == ""

    joint = CausalFlowDAG(
        {**spec, "y": OrdinalNode(3, CS("a", "b"))}, seed=0
    ).to_matrix()
    assert joint.loc["a", "y"] == "CS['a', 'b']"
    additive = CausalFlowDAG(
        {**spec, "y": OrdinalNode(3, I("a", "b", allow_interaction=False))}, seed=0
    ).to_matrix()
    assert additive.loc["a", "y"] == "CI"


def test_ls_coefficients_shape_and_agreement_with_the_modules():
    """ls_coefficients is the public spelling of the LS weights.

    An ordinal parent contributes one weight per one-hot level, and only
    differences against level 0 are identified — so the raw vector has
    ``levels`` entries. Nodes without shift terms do not appear.
    """
    spec = {
        "x": ContinuousNode(),
        "t": OrdinalNode(3),
        "y": ContinuousNode(LS("x") + LS("t")),
    }
    flow = CausalFlowDAG(spec, seed=0)
    coefs = flow.ls_coefficients()

    assert set(coefs) == {"y"}
    assert set(coefs["y"]) == {"x", "t"}
    assert coefs["y"]["x"].shape == (1,)
    assert coefs["y"]["t"].shape == (3,)


def test_ordinal_node_refuses_a_transform():
    """An ordinal intercept is the cutpoint vector; I(transform=) has no meaning."""
    with pytest.raises(ValueError, match="cutpoint vector"):
        OrdinalNode(3, I(transform="spline"))
    with pytest.raises(ValueError, match="cutpoint vector"):
        OrdinalNode(3, I(transform_kwargs={"n_coeffs": 5}))


def test_a_misspelt_intercept_option_is_a_type_error():
    with pytest.raises(TypeError, match="activaton"):
        CI("a", activaton="tanh")


def test_strict_spec_loading_names_the_omitted_option():
    """``strict=True`` refuses an entry that omits an option that shapes its term."""
    d = spec_to_dict(_terms_spec())
    assert spec_from_dict(d, strict=True) == _terms_spec()
    del d["X3"]["terms"][1]["options"]["units"]
    assert spec_from_dict(d) == spec_from_dict(d, strict=False)
    with pytest.raises(ValueError, match=r"node 'X3'.*'CS'.*'units'"):
        spec_from_dict(d, strict=True)


def test_strict_spec_loading_skips_options_that_cannot_act():
    """A parentless or ordinal intercept needs no network or transform options."""
    d = spec_to_dict({"A": ContinuousNode(), "Y": OrdinalNode(3, [CI("A")])})
    del d["A"]["terms"][0]["options"]["units"]
    del d["Y"]["terms"][0]["options"]["transform"]
    spec_from_dict(d, strict=True)


def test_strict_spec_loading_requires_the_switches():
    """An off-by-default switch that acts on the term must be written too."""
    d = spec_to_dict(_terms_spec())
    del d["X3"]["terms"][1]["options"]["batch_norm"]
    with pytest.raises(ValueError, match=r"'CS'.*'batch_norm'"):
        spec_from_dict(d, strict=True)


def test_strict_spec_loading_skips_input_transform_without_continuous_parent():
    """``input_transform`` acts on continuous parents only."""
    spec = {
        "sex": OrdinalNode(2, [I()]),
        "age": ContinuousNode([I()]),
        "t": OrdinalNode(2, [I()]),
        "y": OrdinalNode(3, [I(), CS("sex", units=(4,)), VC("sex", t="t", units=(4,))]),
    }
    d = spec_to_dict(spec)
    for term in d["y"]["terms"][1:]:
        del term["options"]["input_transform"]
    assert spec_from_dict(d, strict=True)["y"].terms[1] == spec["y"].terms[1]
    d["y"]["terms"][1]["parents"] = ["age"]
    with pytest.raises(ValueError, match=r"'CS'.*'input_transform'"):
        spec_from_dict(d, strict=True)


def test_strict_spec_loading_skips_network_options_without_units():
    """An intercept with parents and ``units=()`` has no network."""
    spec = {"x": ContinuousNode([I()]), "y": OrdinalNode(2, [I("x", units=())])}
    d = spec_to_dict(spec)
    for k in ("activation", "batch_norm", "input_transform"):
        del d["y"]["terms"][0]["options"][k]
    assert spec_from_dict(d, strict=True) == spec
