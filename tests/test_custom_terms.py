"""Custom shift terms: the two-class term contract.

The extension contract of 1.0: a new term is a ``tramdag.Term`` subclass (its
options and checks) whose ``module`` is a ``tramdag.modules.ShiftModule``
subclass. A checkpoint carries the term's own import path, so loading a custom
spec imports it from there.
"""

# %% imports ---------------------------------------------------------------------------
import pytest
import torch
from torch import nn

from tramdag import CausalFlowDAG, ContinuousNode, Term, spec_from_dict
from tramdag.modules import ShiftModule


# %% private functions -----------------------------------------------------------------
def _two_node(term):
    return {"x1": ContinuousNode(), "x2": ContinuousNode([term])}


# %% public functions ------------------------------------------------------------------


def test_custom_term_builds_fits_and_round_trips(ls_chain, tmp_path):
    """A Term subclass naming its ShiftModule is a whole term:
    it validates, builds, fits, serializes by name and loads back.
    """
    df = ls_chain["draw"](600, 0)[["x1", "x2"]]
    term = SLS("x1", scale=3.0)
    assert term.module is _ScaledLS
    assert term.name == "SLS"
    assert repr(term) == "SLS(parents=('x1',), scale=3.0)"
    with pytest.raises(ValueError, match="exactly one parent"):
        SLS("x1", "x2")
    with pytest.raises(TypeError, match="unexpected keyword argument 'scael'"):
        SLS("x1", scael=3.0)
    flow = CausalFlowDAG(_two_node(term), seed=0)
    flow.fit(df, epochs=10, batch_size=200, learning_rate=1e-1)
    assert float(flow.nodes["x2"].shifts["x1"].w) != 0.0
    flow.save(tmp_path / "m.pt")
    loaded = CausalFlowDAG.load(tmp_path / "m.pt")
    assert loaded.spec == flow.spec
    assert torch.equal(loaded.log_prob(df), flow.log_prob(df))


def test_unknown_term_and_orphan_term_fail_by_name():
    """A serialized term name no class carries, and a term class no module
    builds, both say what to define.
    """
    d = {
        "x1": {"kind": "continuous", "terms": []},
        "x2": {
            "kind": "continuous",
            "terms": [{"term": "NOPE", "parents": ["x1"], "options": {}}],
        },
    }
    with pytest.raises(ValueError, match="unknown term 'NOPE'"):
        spec_from_dict(d)
    with pytest.raises(TypeError, match="sets `name"):

        class Nameless(Term):
            module = _ScaledLS

    with pytest.raises(TypeError, match="sets `module"):

        class Orphan(Term):
            name = "Orphan"


def test_custom_regularizer_joins_the_loss(ls_chain):
    """Fit adds regularizer()/n — the hook, not VC's internals."""
    df = ls_chain["draw"](300, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node(PEN("x1")), seed=0)
    flow.fit(df, epochs=2, batch_size=150)
    assert flow.nodes["x2"].shifts["x1"].calls >= 4  # every minibatch


# %% private classes -------------------------------------------------------------------
class _ScaledLS(ShiftModule, nn.Module):
    def __init__(self, term, spec):
        nn.Module.__init__(self)
        self.scale = term.scale
        self.w = nn.Parameter(torch.zeros(()))
        self.key = term.parents[0]
        self.parents = tuple(term.parents)

    def shift_value(self, node, feats):
        return self.scale * self.w * feats[self.parents[0]][:, 0]


class SLS(Term):
    """A minimal custom term: ``w * x`` with a fixed scale option."""

    name = "SLS"
    module = _ScaledLS

    def __init__(self, *parents, scale: float = 1.0):
        super().__init__(*parents)
        self.scale = scale
        if len(self.parents) != 1:
            raise ValueError("SLS() takes exactly one parent.")


class _PenShift(ShiftModule, nn.Module):
    def __init__(self, term, spec):
        nn.Module.__init__(self)
        self.w = nn.Parameter(torch.zeros(()))
        self.calls = 0
        self.key = term.parents[0]
        self.parents = tuple(term.parents)

    def shift_value(self, node, feats):
        return self.w * feats[self.parents[0]][:, 0]

    def regularizer(self):
        self.calls += 1
        return self.w**2


class PEN(Term):
    """A custom penalized term: the regularizer hook must reach the loss."""

    name = "PEN"
    module = _PenShift
