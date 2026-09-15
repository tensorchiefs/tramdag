"""Read-outs of a fitted flow: ``ReadoutsMixin``, composed into ``CausalFlowDAG``.

Coefficients (`ls_coefficients`), effect curves (`shift_curve`,
`varying_coef`), the intercept decomposition (`intercept_contributions`),
the meta-adjacency view (`to_matrix`) and the classical design matrix
(`design_matrix`).
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .modules import LinearShiftModule, VaryingCoefficientModule

# %% global variables ------------------------------------------------------------------
__all__ = ["ReadoutsMixin"]


# %% public classes --------------------------------------------------------------------
class ReadoutsMixin:
    """The stateless read-outs, mixed into [`CausalFlowDAG`][tramdag.CausalFlowDAG]."""

    @torch.no_grad()
    def shift_curve(self, node: str, parent: str, grid) -> np.ndarray:
        """Evaluate one fitted shift term on a 1-D grid of parent values.

        The parent runs over ``grid`` with the term's other inputs absent, so the
        term must read exactly this one parent (an ``LS`` or one-parent ``CS``).

        Parameters
        ----------
        node : str
            Name of the node that carries the term.
        parent : str
            The term's key in the node's shifts: the parent's name.
        grid : array-like
            Parent values at which to evaluate the term, shape ``(m,)``.

        Returns
        -------
        np.ndarray
            The shift values, shape ``(m,)`` — the curve a replication plots
            against the data-generating truth.
        """
        nd = self._node(node)
        if parent not in nd.shifts:
            raise KeyError(
                f"node {node!r} has no shift term keyed {parent!r}; "
                f"available: {sorted(nd.shifts)}"
            )
        if self.spec[parent].kind == "ordinal":
            # a domain error (wrong parent kind), not a Python type error
            raise ValueError(
                f"{parent!r} is an ordinal parent, so it enters the term "
                "one-hot over all its levels and a 1-D grid of level indices "
                "is not its input. Read its effect with ls_coefficients() "
                "instead, whose weights are the level contrasts."
            )
        # copy, like `_tensorize`: torch warns on a non-writable numpy array,
        # which a broadcast or read-only grid is
        x = torch.as_tensor(np.array(grid), dtype=self._dtype, device=self.device)
        x = x.view(-1, 1)
        # through the term's own evaluation, so Fn and custom terms work too
        curve = nd.shifts[parent].shift_value(nd, {parent: x})
        return curve.cpu().numpy().ravel()

    @torch.no_grad()
    def varying_coef(
        self, df: pd.DataFrame, node: str, *, t: str | None = None
    ) -> np.ndarray:
        """Evaluate the fitted effect function ``beta(x)`` of a ``VC`` term.

        The value is ``beta0 + b_theta(modifiers)``, in closed form from the
        fitted term; only the modifier columns of ``df`` are read. It lives on
        the node's latent scale.

        Parameters
        ----------
        df : pd.DataFrame
            Rows at which to evaluate ``beta``. Must contain every modifier
            column of the term.
        node : str
            Name of the node that carries the VC term.
        t : str | None, optional
            Treatment name of the VC term. Optional when the node has
            exactly one VC term.

        Returns
        -------
        np.ndarray
            The ``beta`` values, shape ``(n,)``. Constant when the term has
            no modifiers.

        Raises
        ------
        KeyError
            If ``node`` is unknown, if the node has no VC term on ``t``, or
            if a modifier column is missing from ``df``.
        ValueError
            If the node has no VC term, or if ``t`` is omitted while the
            node has several VC terms.
        """
        nd = self._node(node)
        vcs = {
            m.key: m.mods
            for m in nd.shifts.values()
            if isinstance(m, VaryingCoefficientModule)
        }
        if not vcs:
            raise ValueError(f"node {node!r} has no VC term.")
        if t is None:
            if len(vcs) > 1:
                raise ValueError(
                    f"node {node!r} has several VC terms ({sorted(vcs)}); "
                    "pass t=<treatment name>"
                )
            t = next(iter(vcs))
        if t not in vcs:
            raise KeyError(f"node {node!r} has no VC term on {t!r} (has {sorted(vcs)})")
        mods = vcs[t]
        mod_feat = None
        if mods:
            feats = self._features(self._tensorize(df, mods))
            mod_feat = nd.net_input(feats, mods, t)
        return nd.shifts[t].beta(mod_feat, len(df)).cpu().numpy()

    @torch.no_grad()
    def ls_coefficients(self) -> dict[str, dict[str, np.ndarray]]:
        """Give the per-node linear-shift weights.

        A continuous parent has one weight. An ordinal parent has one weight
        per level (its one-hot encoding), identified only up to a common
        constant, so read them as differences ``w[k] - w[0]``. Only ``LS``
        terms have a weight; ``CS`` and ``VC`` shifts are skipped.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            The weights, as ``{node: {parent: array}}``. A node without
            linear-shift terms is absent.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for name in self.order:
            linear = {
                parent: module.weight.cpu().numpy().ravel().copy()
                for parent, module in self.nodes[name].shifts.items()
                if isinstance(module, LinearShiftModule)
            }
            if linear:
                out[name] = linear
        return out

    def to_matrix(self) -> pd.DataFrame:
        """Give the labeled adjacency matrix of term effects.

        Returns
        -------
        pd.DataFrame
            Rows are parents and columns are children. A cell holds the
            term tag: ``"LS"``, ``"CS"``, ``"CI"``, ``"VC"`` for a VC
            treatment, or ``"VCm"`` for a VC modifier. An empty cell means
            there is no edge. A multi-parent term carries its parent group
            as a suffix. When several terms share a cell, their tags join
            with ``"+"``.
        """
        m = pd.DataFrame("", index=list(self.order), columns=list(self.order))
        for child in self.order:
            for term in self.spec[child].terms:
                # a VC modifier may share its cell with an edge-owning term
                for p, tag, joint in term.cells():
                    if joint:
                        tag = f"{tag}{list(term.parents)}"
                    cur = m.loc[p, child]
                    m.loc[p, child] = f"{cur}+{tag}" if cur else tag
        return m

    @torch.no_grad()
    def intercept_contributions(self, df: pd.DataFrame, node: str) -> dict:
        r"""Decompose a complex intercept into mean-centered per-parent parts.

        Each network's output over the rows of ``df`` is centered to mean zero
        per parameter; the removed means collect into ``baseline``, so
        $\vartheta(\mathrm{pa}) = \text{baseline} + \sum \text{contributions}$ exactly.
        Post-hoc only: it reads the fitted weights and changes nothing.

        Parameters
        ----------
        df : pd.DataFrame
            Rows over which to center and at which to evaluate the
            contributions. Must contain every intercept-parent column.
        node : str
            Name of a node with at least one complex-intercept (``I``) term
            that has parents.

        Returns
        -------
        dict
            Three keys. ``"baseline"`` is the absorbed constant, a ``(P,)``
            array — the sum of the per-term means. ``P`` is the node's
            transform-parameter count: ``ut.n_params`` for a continuous
            node, ``levels - 1`` cutpoint parameters for an ordinal node.
            ``"contributions"`` is ``{term_label: (n, P) array}`` — each
            term's mean-centered contribution at each row, columns summing
            to about zero over the rows. ``term_label`` is the term's
            parents joined by ``"+"``. ``"parents"`` is
            ``{term_label: tuple(parent_names)}``.

        Raises
        ------
        KeyError
            If ``node`` is unknown, or if an intercept-parent column is
            missing from ``df``.
        ValueError
            If the node has no complex-intercept term with parents.

        Notes
        -----
        An additive intercept holds one network per parent group in its
        ``nets``; a single intercept term, joint or not, is itself the one
        network. The contributions live in the transform's unconstrained
        parameter space, before the monotonicity constraint.
        """
        nd = self._node(node)
        groups = nd.intercept.groups
        if not groups:
            raise ValueError(
                f"node {node!r} has no complex-intercept (I) terms with parents. "
                "Its intercept is unconditional, so there is nothing to decompose."
            )
        feats = self._features(self._tensorize(df, nd.intercept.ci_parents))
        # one net per group: the additive intercept holds them in `nets`, every
        # other intercept is itself the one network
        nets = getattr(nd.intercept, "nets", [nd.intercept])

        contributions: dict[str, np.ndarray] = {}
        parents: dict[str, tuple] = {}
        baseline = None
        for net, grp in zip(nets, groups, strict=True):
            raw = net(nd.net_input(feats, grp, "@I"))  # (n, P)
            mean = raw.mean(dim=0, keepdim=True)  # (1, P)
            label = "+".join(grp)
            contributions[label] = (raw - mean).cpu().numpy()
            parents[label] = grp
            baseline = mean if baseline is None else baseline + mean
        return {
            "baseline": baseline.cpu().numpy().ravel(),
            "contributions": contributions,
            "parents": parents,
        }

    @torch.no_grad()
    def design_matrix(
        self, df: pd.DataFrame, node: str, *, drop_first: bool = False
    ) -> pd.DataFrame:
        """Encode a node's parents the way the flow feeds them to its shifts.

        A continuous parent stays raw in one column named after it. An
        ordinal parent becomes one column per level, named
        ``"{parent}[{k}]"`` — the same one-hot the flow builds internally.

        Use ``drop_first=True`` to get the design a classical reference
        expects (``statsmodels`` ``OrderedModel``, R ``polr``): with
        cutpoints the full one-hot is unidentified, so each ordinal parent's
        level-0 column drops out and its remaining coefficients read as
        differences against level 0 — exactly what ``w[k] - w[0]`` gives on
        the flow side.

        Parameters
        ----------
        df : pd.DataFrame
            Rows to encode. Must contain every parent column of ``node``.
        node : str
            Name of the node whose parents are encoded.
        drop_first : bool, optional
            Drop each ordinal parent's level-0 column, by default ``False``.

        Returns
        -------
        pd.DataFrame
            One column per encoded feature, indexed like ``df``.
        """
        nd = self._node(node)
        feats = self._features(self._tensorize(df, nd.parents))
        cols: dict[str, np.ndarray] = {}
        for p in nd.parents:
            arr = feats[p].cpu().numpy()
            if self.spec[p].kind == "continuous":  # raw
                cols[p] = arr[:, 0]
            else:
                for k in range(1 if drop_first else 0, arr.shape[1]):
                    cols[f"{p}[{k}]"] = arr[:, k]
        return pd.DataFrame(cols, index=df.index)
