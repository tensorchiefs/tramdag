r"""Per-observation scores of the shift coefficients, and the effect-modifier scan.

The scores $\psi_i = \partial \ell_i / \partial \beta$ are analytic: every shift
coefficient enters the latent additively, so
$\partial \ell_i / \partial \beta = (\partial \ell_i / \partial s_i)\, x_i$ with the
latent-scale derivative in closed form (``_dl_ds``). The scan orders
one coefficient's scores by a candidate covariate and reports a CUSUM
statistic with its p-value. The ``CausalFlowDAG`` methods [`scores`][] and
[`effect_modifier_scan`][] delegate here; ``docs/scores.md`` is the guide.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .transforms import ordinal_bounds

# %% global variables ------------------------------------------------------------------
__all__ = ["effect_modifier_scan", "node_scores", "sup_bb_pvalue"]

# 5% critical value of sup |Brownian bridge| (Kolmogorov distribution)
CRIT_5PCT = 1.3581


# %% private functions -----------------------------------------------------------------
def _dl_ds(nd, feats: dict, x: torch.Tensor) -> torch.Tensor:
    r"""Give $\partial \ell_i / \partial s_i$, shape ``(n,)``.

    This is the closed-form derivative of the per-row log-likelihood with
    respect to the total shift of the node.
    """
    theta, shift = nd.theta_shift(feats, x.shape[0])
    if nd.kind == "continuous":
        z0, _ = nd.ut.forward(theta, x)
        return 1.0 - 2.0 * torch.sigmoid(z0 + shift)
    lower, upper = ordinal_bounds(theta, shift, x)  # already include -s
    sl, su = torch.sigmoid(lower), torch.sigmoid(upper)
    return (sl * (1 - sl) - su * (1 - su)) / (su - sl)


# %% public functions ------------------------------------------------------------------
@torch.no_grad()
def node_scores(flow, df: pd.DataFrame, node: str) -> pd.DataFrame:
    """Compute the per-observation scores of the interpretable coefficients.

    The scores cover every ``LS`` weight and the ``beta0`` of every ``VC``
    term. ``CS`` terms carry no interpretable coefficient, so this function
    skips them.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow.
    df : pd.DataFrame
        Observations. Must contain the node, its parents, and the
        propensity inputs of centered VC terms.
    node : str
        Name of the node whose coefficients are scored.

    Returns
    -------
    pd.DataFrame
        One column per coefficient, shape ``(n, k)``, indexed like ``df``.
        A continuous ``LS`` parent gives one column, named after the
        parent. An ordinal ``LS`` parent gives one column per one-hot
        level, named ``"{parent}[{k}]"``. A ``VC`` term gives one column
        named after its treatment, which holds the ``beta0`` score. For a
        binary ordinal treatment that score belongs to the identified
        contrast of level 1 against level 0.

    Raises
    ------
    KeyError
        If ``node`` is unknown, or if a needed column is missing from
        ``df``.
    ValueError
        If the node has no ``LS`` or ``VC`` term.
    """
    nd = flow._node(node)
    scored = [m for m in nd.shifts.values() if m.scored]
    if not scored:
        raise ValueError(
            f"node {node!r} has no LS or VC terms. Shift scores need "
            "at least one interpretable shift coefficient."
        )

    # not y-free: l_i needs x. Plus the e_hat inputs of centered terms.
    needed = [*nd.parents, node, *flow._query_side_columns(nd)]
    values = flow._tensorize(df, needed)  # names a missing column
    feats = flow._parent_feats(nd, values)
    feats |= flow._side_feats(nd, values, len(df))
    dlds = _dl_ds(nd, feats, values[node])

    cols: dict[str, np.ndarray] = {}
    for m in scored:
        cols.update(m.score_columns(nd, flow, feats, dlds))
    return pd.DataFrame(cols, index=df.index)


def sup_bb_pvalue(stat: float) -> float:
    """Give ``P(sup |Brownian bridge| > stat)``, the Kolmogorov series.

    Parameters
    ----------
    stat : float
        Observed supremum statistic.

    Returns
    -------
    float
        The p-value, clipped to [0, 1].
    """
    if stat <= 0:
        return 1.0  # the series alternates to 0.0 here, which is the wrong tail
    # 100 terms, and they are all needed. The k-th is exp(-2k^2 stat^2), which
    # underflows past k ~ 10 only for a LARGE statistic. A small one converges
    # slowly: at stat = 0.02 the truncation at k = 10 returns 0.084 where the
    # series gives 0.9997, so a perfectly stable coefficient would be reported
    # as significant. stat = 0.2 still needs 20 terms.
    s = sum(
        (-1) ** (k + 1) * math.exp(-2.0 * k * k * stat * stat) for k in range(1, 101)
    )
    return min(1.0, max(0.0, 2.0 * s))


@torch.no_grad()
def effect_modifier_scan(
    flow,
    df: pd.DataFrame,
    node: str,
    *,
    t: str,
    candidates: list[str] | None = None,
    column: str | None = None,
) -> pd.DataFrame:
    r"""Scan the ``t``-coefficient scores for effect-modifier drift.

    For each candidate covariate the scan orders the treatment coefficient's
    scores by it, forms the scaled cumulative sum
    $B_j = \sum_{i \le j} \psi_{(i)} / (\mathrm{sd}(\psi)\sqrt{n})$ and reports
    $\sup_j |B_j|$ with its Kolmogorov p-value and the 5% critical value.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow.
    df : pd.DataFrame
        Observations, as for [`node_scores`][tramdag.scores.node_scores].
    node : str
        Name of the outcome node.
    t : str
        Name of the treatment. Its scores column is ``t`` itself for a
        continuous parent or a VC term, and the identified level-1 column
        ``"{t}[1]"`` for a binary ordinal LS parent.
    candidates : list[str] | None, optional
        Candidate covariates. Defaults to every column of ``df`` except
        ``node`` and ``t``.
    column : str | None, optional
        Score column to scan, overriding the ``t``-derived choice — the
        way to scan one level contrast of a multi-level ordinal
        treatment (e.g. ``"t[2]"``), which has no single default column.

    Returns
    -------
    pd.DataFrame
        Indexed by candidate, sorted by ``stat`` descending, with columns
        ``stat``, ``p_value``, ``crit_5pct`` and ``flag``
        (``stat > crit_5pct``).

    Raises
    ------
    KeyError
        If no score column exists for ``t`` on ``node``.
    ValueError
        If the score column is constant.
    """
    psi_df = node_scores(flow, df, node)
    if column is not None:
        if column not in psi_df.columns:
            raise KeyError(
                f"no score column {column!r} on node {node!r} "
                f"(have {list(psi_df.columns)})"
            )
        col = column
    elif t in psi_df.columns:
        col = t
    elif (
        t in flow.spec
        and flow.spec[t].kind == "ordinal"
        and flow.spec[t].levels == 2
        and f"{t}[1]" in psi_df.columns
    ):
        col = f"{t}[1]"  # the identified contrast of a binary ordinal LS parent
    else:
        raise KeyError(
            f"no score column for treatment {t!r} on node {node!r} "
            f"(have {list(psi_df.columns)}). For a multi-level ordinal "
            "treatment pass column= with the level contrast to scan."
        )
    psi = psi_df[col].to_numpy()
    n = len(psi)
    sd = psi.std()
    if sd == 0:
        raise ValueError(f"score column {col!r} is constant; there is nothing to scan")

    if candidates is None:
        candidates = [c for c in df.columns if c not in (node, t)]
    rows = {}
    for c in candidates:
        order = np.argsort(df[c].to_numpy(), kind="stable")
        b = np.cumsum(psi[order]) / (sd * math.sqrt(n))
        stat = float(np.abs(b).max())
        rows[c] = {
            "stat": stat,
            "p_value": sup_bb_pvalue(stat),
            "crit_5pct": CRIT_5PCT,
            "flag": stat > CRIT_5PCT,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out.sort_values("stat", ascending=False)
