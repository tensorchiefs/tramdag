"""The ordinal encoding contract: one-hot parents, level indices, identifiability.

An ordinal parent enters every shift and every intercept net as a full
``levels``-wide one-hot, the encoding of the original TRAM-DAG
implementation. Three consequences are pinned here:

- the values must be level indices ``0..levels-1``, at *every* entry point —
  training rejected 1.5 already, inference used to truncate it silently;
- the full one-hot plus a node intercept is over-complete by exactly one
  parameter per ordinal ``LS`` parent, so only the level *differences* are
  identified (`design_matrix(drop_first=True)` and `w[k] - w[0]` are the two
  ways to read them);
- the numeric level itself is never a feature, so the fit is nominal in the
  parent, not linear in its level.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from tramdag import LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode


# %% private functions -----------------------------------------------------------------
def _ordinal_parent_frame(n=3000, seed=0):
    """A 4-level ordinal root with a linear-in-level effect on a continuous child."""
    rng = np.random.default_rng(seed)
    level = rng.integers(0, 4, n)
    df = pd.DataFrame(
        {"p": level.astype(float), "y": 0.7 * level + rng.logistic(size=n)}
    )
    spec = {"p": OrdinalNode(4), "y": ContinuousNode([I(), LS("p")])}
    return spec, df


# %% public functions ------------------------------------------------------------------
def test_one_hot_ls_weights_are_identified_only_up_to_a_constant():
    """Three seeds reach the same NLL with weights that differ by a constant.

    The one-hot columns sum to 1 in every row, so adding c to all of a
    parent's weights and subtracting it from the child's intercept leaves the
    likelihood untouched. Only ``w[k] - w[0]`` is an estimate.
    """
    spec, df = _ordinal_parent_frame()
    fits = []
    for seed in (0, 1, 2):
        flow = CausalFlowDAG(spec, seed=seed)
        flow.fit_classical(df)
        weights = flow.ls_coefficients()["y"]["p"]
        fits.append((weights, float(sum(flow.nll(df).values()))))

    nlls = [nll for _, nll in fits]
    assert max(nlls) - min(nlls) < 1e-4  # the same optimum

    first = fits[0][0]
    assert not all(np.allclose(w, first) for w, _ in fits[1:])  # different weights
    for weights, _ in fits[1:]:  # identical once the flat direction is removed
        np.testing.assert_allclose(
            weights - weights.mean(), first - first.mean(), atol=1e-3
        )

    # the identified contrasts recover the DGP's per-level step (latent sign
    # convention: a positive effect on y is a negative shift)
    steps = -np.diff(first)
    np.testing.assert_allclose(steps, 0.7, atol=0.1)


def test_design_matrix_drops_the_flat_direction():
    """``drop_first`` removes exactly one column per ordinal parent."""
    spec, df = _ordinal_parent_frame(n=50)
    flow = CausalFlowDAG(spec, seed=0)
    flow.calibrate(df)
    full = flow.design_matrix(df, "y")
    reduced = flow.design_matrix(df, "y", drop_first=True)
    assert list(full.columns) == [f"p[{k}]" for k in range(4)]
    assert list(reduced.columns) == [f"p[{k}]" for k in range(1, 4)]
    assert (full.sum(axis=1) == 1).all()  # the row sum that makes it flat


@pytest.mark.parametrize("bad", [1.5, 9.0, -1.0])
def test_a_non_level_ordinal_value_is_refused_at_every_entry_point(bad):
    """1.5 used to truncate to level 1 in silence; 9 raised without a name."""
    spec, df = _ordinal_parent_frame(n=200)
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=2, batch_size=200, learning_rate=1e-2)

    off = df.assign(p=bad)
    for call in (flow.log_prob, flow.abduct, lambda d: flow.nll(d)):
        with pytest.raises(ValueError, match=r"level indices 0\.\.3"):
            call(off)
    with pytest.raises(ValueError, match=r"level indices 0\.\.3"):
        flow.sample(10, do={"p": bad})

    assert len(flow.sample(10, do={"p": 2})) == 10  # a level still passes


def test_a_latent_frame_skips_the_level_check():
    """``sample(u=...)`` carries node-named columns of real latents."""
    spec, df = _ordinal_parent_frame(n=200)
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=2, batch_size=200, learning_rate=1e-2)
    u = flow.abduct(df, seed=0)
    assert not np.allclose(u["p"], np.round(u["p"]))  # latents, not levels
    back = flow.sample(u=u)
    np.testing.assert_array_equal(back["p"].to_numpy(), df["p"].to_numpy())
