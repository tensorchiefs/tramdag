"""CausalFlowDAG — a single triangular normalizing flow on a user-defined DAG.

The flow maps iid standard-logistic latents ``U`` to the observed variables ``X``
in topological order. ``sample``, ``abduct``, ``pmf`` and ``density`` answer the
observational, interventional (``do=``) and counterfactual queries; the read-outs
that only read fitted weights live in ``readouts.py``.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from . import scores
from .fitting import FitMixin
from .modules import ShiftModule
from .nodes import (
    Node,
    check_columns,
    check_level_values,
    load_weights,
    tensorize,
    write_checkpoint,
)
from .readouts import ReadoutsMixin
from .spec import (
    NodeSpec,
    node_parents,
    spec_from_dict,
    spec_to_dict,
    validate_and_sort,
)
from .transforms import (
    StandardLogistic,
    ordinal_pmf,
)

# %% global variables ------------------------------------------------------------------
__all__ = ["CausalFlowDAG"]


# %% private functions -----------------------------------------------------------------
def _init_linear(m: nn.Linear, init: str) -> None:
    """Keras' two initializers on one linear layer: ``glorot`` or ``normal``."""
    if init == "glorot":
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    else:
        nn.init.normal_(m.weight, std=0.05)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=0.05)


# %% public classes --------------------------------------------------------------------
class CausalFlowDAG(FitMixin, ReadoutsMixin, nn.Module):
    """A causal normalizing flow defined by a DAG specification.

    Parameters
    ----------
    spec : dict[str, NodeSpec]
        The DAG specification, ``{name: ContinuousNode | OrdinalNode}``.
    device : str, optional
        Torch device, by default ``"cpu"``.
    seed : int | None, optional
        If given, seeds the weight initialization deterministically
        (``torch.manual_seed`` runs before the nodes are built). Weight
        initialization happens at construction, so this is the one knob
        for a reproducible model. ``fit(seed=...)`` only seeds the
        minibatch shuffling.
    init : str, optional
        Weight initialization of every linear layer (the LS weights and the
        CI/CS/VC networks): ``"torch"`` (default, ``nn.Linear``'s
        Kaiming-uniform), ``"glorot"`` (glorot-uniform weights, zero biases)
        or ``"normal"`` (N(0, 0.05^2) weights and biases). A VC head's
        output layer stays zero either way. Stored in the checkpoint.
    """

    def __init__(
        self,
        spec: dict[str, NodeSpec],
        device: str = "cpu",
        seed: int | None = None,
        init: str = "torch",
    ):
        super().__init__()
        if init not in ("torch", "glorot", "normal"):
            raise ValueError(
                f"init must be 'torch', 'glorot' or 'normal', got {init!r}"
            )
        if seed is not None:
            torch.manual_seed(seed)
        self.spec = spec
        self.order = validate_and_sort(spec)
        self.init = init
        self.nodes = nn.ModuleDict(
            {name: Node(name, spec[name], self._schema(name)) for name in self.order}
        )
        self._apply_init(init)
        self.device = torch.device(device)
        self.meta: dict = {}  # provenance attached at save() (version, time)
        self.to(self.device)

    def _schema(self, name: str) -> dict[str, str | int]:
        """Give a node's parent schema, ``{parent: "continuous" | n_levels}``."""
        spec = self.spec
        return {
            p: "continuous" if spec[p].kind == "continuous" else spec[p].levels
            for p in node_parents(spec[name])
        }

    def _apply_init(self, init: str) -> None:
        """Re-initialize every linear layer, if asked; VC heads re-zero their output."""
        if init == "torch":
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m, init)
        for m in self.modules():
            if isinstance(m, ShiftModule):
                m.post_init()

    @property
    def _dtype(self) -> torch.dtype:
        """Current model dtype: float32 (float64 only inside ``fit_classical``).

        Every tensor built from a frame takes this dtype, so the read-outs work
        in both modes without carrying a dtype argument.
        """
        return next(self.parameters()).dtype

    def _tensorize(
        self,
        df: pd.DataFrame,
        cols: list[str] | tuple[str, ...] | None = None,
        *,
        levels: bool = True,
    ) -> dict[str, Tensor]:
        """DataFrame columns -> one ``(n,)`` tensor each, in the model dtype.

        ``cols=None`` takes every node, in topological order. ``levels=False``
        skips the ordinal level check, for a frame of *latents* whose columns
        carry node names but real values.

        Raises
        ------
        KeyError
            If the frame lacks one of the columns, by name — a spec/data
            mismatch would otherwise surface deep inside a tensor op.
        ValueError
            If an ordinal column is not a level index of its node. The
            one-hot encoding would otherwise truncate 1.5 to level 1 in
            silence, or fail inside ``one_hot`` without naming the node.
        """
        cols = self.order if cols is None else cols
        kinds = {n: nd.levels for n, nd in self.nodes.items() if nd.kind == "ordinal"}
        return tensorize(df, cols, kinds if levels else {}, self._dtype, self.device)

    def _to_frame(self, values: dict[str, Tensor]) -> pd.DataFrame:
        """Tensors -> DataFrame; an ordinal column goes back as a level index."""
        out = {}
        for k, v in values.items():
            arr = v.cpu().numpy()
            out[k] = arr.astype(np.int64) if self.nodes[k].kind == "ordinal" else arr
        return pd.DataFrame(out)

    def _generator(self, seed: int | None) -> torch.Generator | None:
        """Give a seeded generator on this flow's device, or None for unseeded."""
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(seed)

    def _node(self, name: str) -> Node:
        """Look a node up by name, with the same error everywhere."""
        if name not in self.nodes:
            raise KeyError(f"unknown node {name!r}")
        return self.nodes[name]

    def _features(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        # spec columns only; side columns travel through _side_feats
        return {
            name: self.nodes[name].encode(v)
            for name, v in values.items()
            if name in self.spec
        }

    def _theta_shift(
        self, nd: Node, feats: dict[str, Tensor], values: dict[str, Tensor], n: int
    ) -> tuple[Tensor, Tensor]:
        """Evaluate one node's transform parameters and shift.

        The side columns join here, so every query that reaches a node's
        parameters gets them. A query that composed the call itself would drop
        a centered term's propensity column by omission, which is the failure
        ``VaryingCoefficientModule`` has to catch at run time.

        ``feats`` is passed in rather than derived, because a caller looping
        over the nodes encodes the parents once for all of them.
        """
        return nd.theta_shift(feats | self._side_feats(nd, values, n), n)

    def _side_feats(
        self, nd: Node, values: dict[str, Tensor], n: int
    ) -> dict[str, Tensor]:
        """Give the node's side columns: frozen from ``values``, else live.

        Training frames carry them as ordinary columns; queries recompute
        them from the fitted flow (``ShiftModule.live_side``).
        """
        out = {}
        for m in nd.shifts.values():
            cols = m.side_columns()
            if all(c in values for c in cols):
                out.update({c: values[c] for c in cols})
            else:
                out.update(m.live_side(self, values, n))
        return out

    def _query_side_columns(self, nd: Node) -> list[str]:
        """List the extra columns a query needs to recompute live side inputs.

        These are the columns beyond ``nd.parents``: the parents of the
        treatment nodes (which cannot be centered themselves, so one level
        is all there is).
        """
        cols = [p for m in nd.shifts.values() for p in m.extra_columns(self)]
        return [c for c in dict.fromkeys(cols) if c not in nd.parents]

    @torch.no_grad()
    def _propensity(self, nd: Node, values: dict[str, Tensor], n: int) -> Tensor:
        r"""Give $P(\text{node} = 1 \mid \mathrm{pa})$ for a binary ordinal treatment.

        $P(x \le 0) = \sigma(\vartheta_0 - s)$, so the answer is
        $\sigma(s - \vartheta_0)$. No side columns: chained centering is refused
        by the spec, so a treatment node never carries a centered term itself.
        """
        theta, shift = nd.theta_shift(nd.features(values), n)
        return torch.sigmoid(shift - theta[:, 0])

    def _conditional(
        self, df: pd.DataFrame, node: str, do: dict[str, float] | None
    ) -> tuple[Node, Tensor, Tensor, int]:
        """Evaluate one node's conditional at the rows of ``df``.

        ``do`` overrides columns before the parents are read, which is what
        makes [`pmf`][] and [`density`][] interventional. Gives the node,
        its transform parameters, its shift and the row count.
        """
        nd = self._node(node)
        df = df.assign(**(do or {}))
        n = len(df)
        values = self._tensorize(df, list(nd.parents) + self._query_side_columns(nd))
        theta, shift = self._theta_shift(nd, nd.features(values), values, n)
        return nd, theta, shift, n

    @torch.no_grad()
    def _mean_nll(self, values: dict[str, Tensor]) -> dict[str, float]:
        """Give the per-node mean NLL of already tensorized columns."""
        return {k: float(-v.mean()) for k, v in self.node_log_prob(values).items()}

    def _calibrate(
        self, train_df: pd.DataFrame, *, marginal_init: bool = False
    ) -> CausalFlowDAG:
        """Calibrate every node on the training rows ([`Node.calibrate`][]).

        The columns are checked for all nodes first, so a frame that lacks one
        does not leave the flow half calibrated.
        """
        check_columns(train_df, self.order)
        for nd in self.nodes.values():
            nd.calibrate(train_df, marginal_init=marginal_init)
        return self

    def node_log_prob(
        self,
        values: dict[str, Tensor],
        nodes: list[str] | None = None,
    ) -> dict[str, Tensor]:
        """Compute the per-node log-likelihood contributions.

        Parameters
        ----------
        values : dict[str, Tensor]
            Raw node values, keyed by node name, each shape ``(n,)``.
        nodes : list[str] | None, optional
            Restrict the computation to these nodes. A subset is exact
            because the per-node losses
            are independent. ``None`` (default) computes every node.

        Returns
        -------
        dict[str, Tensor]
            One log-likelihood tensor per node, each shape ``(n,)``.
        """
        feats = self._features(values)
        n = next(iter(values.values())).shape[0]
        out = {}
        for name in self.order if nodes is None else nodes:
            nd = self.nodes[name]
            theta, shift = self._theta_shift(nd, feats, values, n)
            out[name] = nd.log_prob(theta, shift, values[name])
        return out

    @torch.no_grad()
    def log_prob(self, df: pd.DataFrame, *, nodes: list[str] | None = None) -> Tensor:
        """Compute the joint log-likelihood per row.

        Parameters
        ----------
        df : pd.DataFrame
            Observations, one column per node.
        nodes : list[str] | None, optional
            Sum only these nodes' contributions. A subset is exact, because
            the per-node losses are independent — ``nodes=["Y"]`` is the
            conditional log-likelihood of ``Y`` given its parents, per row,
            in log space (safer than the log of [`pmf`][], which
            underflows in the tail). ``None`` (default) is the joint.

        Returns
        -------
        Tensor
            ``log p(x)`` per row, shape ``(n,)``.
        """
        if nodes is not None and not nodes:
            raise ValueError("nodes=[] sums nothing; omit it for the joint")
        for name in nodes or ():
            self._node(name)  # name the unknown node, not its KeyError
        per_node = self.node_log_prob(self._tensorize(df), nodes)
        return torch.stack(list(per_node.values()), dim=0).sum(dim=0)

    def node_negative_log_prob(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute the mean negative log-likelihood per node (a diagnostic).

        Parameters
        ----------
        df : pd.DataFrame
            Observations, one column per node.

        Returns
        -------
        dict[str, float]
            The mean NLL, keyed by node name.
        """
        return self._mean_nll(self._tensorize(df))

    nll = node_negative_log_prob  # the short name every notebook uses

    @torch.no_grad()
    def sample(
        self,
        n: int | None = None,
        *,
        do: dict[str, float] | None = None,
        u: pd.DataFrame | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Sample from the flow, with optional interventions.

        Parameters
        ----------
        n : int | None, optional
            Number of samples. Ignored if ``u`` is given.
        do : dict[str, float] | None, optional
            Interventions, as ``{node: value}``. An intervened node is
            clamped and its parent dependence removed (graph mutilation).
        u : pd.DataFrame | None, optional
            Latent variables, as returned by [`abduct`][]. If given,
            they are pushed through the flow. With ``do``, this gives a
            counterfactual.
        seed : int | None, optional
            If given, seeds the latent draw. Ignored if ``u`` is given.

        Returns
        -------
        pd.DataFrame
            The samples, one column per node.

        Raises
        ------
        ValueError
            If both ``n`` and ``u`` are omitted.
        """
        do = do or {}
        for name, value in do.items():
            if self._node(name).kind == "ordinal":
                check_level_values(name, [value], self.spec[name].levels)
        if u is not None:
            n = len(u)
            u_vals = self._tensorize(u, levels=False)  # latents, not levels
        elif n is not None:
            gen = self._generator(seed)
            u_vals = {
                name: StandardLogistic.sample((n,), device=self.device, generator=gen)
                for name in self.order
            }
        else:
            raise ValueError("sample() needs n or u")

        values: dict[str, Tensor] = {}
        for name in self.order:
            if name in do:
                values[name] = torch.full(
                    (n,), float(do[name]), dtype=self._dtype, device=self.device
                )
                continue
            nd = self.nodes[name]
            # under do, a centered VC re-derives t_do - e_hat(x); never cached
            feats = nd.features(values)
            theta, shift = self._theta_shift(nd, feats, values, n)
            values[name] = nd.sample(theta, shift, u_vals[name])
        return self._to_frame(values)

    @torch.no_grad()
    def abduct(self, df: pd.DataFrame, *, seed: int | None = None) -> pd.DataFrame:
        r"""Recover the latent variables ``u`` from observations (the abduction step).

        A continuous node inverts exactly: $u = h(x) + s$. This is
        $\operatorname{logit} P(X \le x \mid \text{parents})$, an
        interventional quantity at the given parent values. For an
        ordinal node the latent is only interval-identified, so it is
        sampled from the standard logistic truncated to the observed
        level's interval.

        Parameters
        ----------
        df : pd.DataFrame
            Observations, one column per node.
        seed : int | None, optional
            If given, seeds the truncated draw for the ordinal nodes.

        Returns
        -------
        pd.DataFrame
            The latents, one column per node, aligned with the rows of
            ``df``.
        """
        gen = self._generator(seed)
        values = self._tensorize(df)
        feats = self._features(values)
        n = len(df)
        u = {}
        for name in self.order:
            nd = self.nodes[name]
            theta, shift = self._theta_shift(nd, feats, values, n)
            u[name] = nd.abduct(theta, shift, values[name], generator=gen)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in u.items()}, index=df.index)

    @torch.no_grad()
    def pmf(
        self, df: pd.DataFrame, node: str, *, do: dict[str, float] | None = None
    ) -> np.ndarray:
        """Give the analytic class probabilities of an ordinal node.

        Parameters
        ----------
        df : pd.DataFrame
            Rows that supply the parent values of the node.
        node : str
            Name of the ordinal node.
        do : dict[str, float] | None, optional
            Column overrides, applied to ``df`` before the evaluation.

        Returns
        -------
        np.ndarray
            The class probabilities, shape ``(n, levels)``.

        Raises
        ------
        ValueError
            If ``node`` is continuous.
        """
        if self._node(node).kind != "ordinal":  # a domain error, not a type error
            raise ValueError(
                f"pmf() requires an ordinal node, {node!r} is continuous; use density()"
            )
        _, theta, shift, _ = self._conditional(df, node, do)
        return ordinal_pmf(theta, shift).cpu().numpy()

    @torch.no_grad()
    def density(
        self,
        df: pd.DataFrame,
        node: str,
        grid,
        *,
        do: dict[str, float] | None = None,
    ) -> np.ndarray:
        r"""Give the analytic conditional density of a continuous node on a grid.

        The continuous counterpart of [`pmf`][]: for every row of ``df``
        the density $p(\text{node} = g \mid \mathrm{pa})$ at each grid value $g$, in
        closed form from the transform — no sampling.

        Parameters
        ----------
        df : pd.DataFrame
            Rows that supply the parent values of the node.
        node : str
            Name of the continuous node.
        grid : array-like
            Values of ``node`` at which to evaluate the density, shape ``(m,)``.
        do : dict[str, float] | None, optional
            Column overrides, applied to ``df`` before the evaluation.

        Returns
        -------
        np.ndarray
            The densities, shape ``(n, m)``.

        Raises
        ------
        ValueError
            If ``node`` is ordinal; use ``pmf`` for it.
        """
        if self._node(node).kind != "continuous":  # a domain error, not a type error
            raise ValueError(
                f"density() requires a continuous node, {node!r} is ordinal; use pmf()"
            )
        nd, theta, shift, n = self._conditional(df, node, do)
        y = torch.tensor(np.asarray(grid), dtype=self._dtype, device=self.device)
        m = y.numel()
        # one (row, grid value) pair per evaluation: rows repeat, the grid tiles
        u0, ladj = nd.ut.forward(theta.repeat_interleave(m, 0), y.repeat(n))
        log_p = StandardLogistic.log_prob(u0 + shift.repeat_interleave(m)) + ladj
        return log_p.exp().view(n, m).cpu().numpy()

    @torch.no_grad()
    def scores(self, df: pd.DataFrame, node: str) -> pd.DataFrame:
        r"""Give the scores $\psi_i = \partial \ell_i / \partial \beta$.

        The method form of [`node_scores`][tramdag.scores.node_scores], which
        documents the arguments and the column naming.
        """
        return scores.node_scores(self, df, node)

    @torch.no_grad()
    def effect_modifier_scan(
        self,
        df: pd.DataFrame,
        node: str,
        *,
        t: str,
        candidates: list[str] | None = None,
        column: str | None = None,
    ) -> pd.DataFrame:
        """Rank candidate effect modifiers with a fluctuation scan.

        The method form of
        [`effect_modifier_scan`][tramdag.scores.effect_modifier_scan], which
        documents the method, the arguments and the result columns.
        """
        return scores.effect_modifier_scan(
            self, df, node, t=t, candidates=candidates, column=column
        )

    def save(self, path: str | Path) -> None:
        """Write the model, its history and its provenance to a checkpoint.

        The file holds the spec and the weights, the nodes' training
        ``history``, and a ``meta`` block with the tramdag version, the save
        time and the device.

        Parameters
        ----------
        path : str | Path
            Target file. Parent directories are created when missing.
        """
        write_checkpoint(
            path,
            {
                "spec": spec_to_dict(self.spec),
                "init": self.init,
                "state_dict": self.state_dict(),
                "history": {n: nd.history for n, nd in self.nodes.items()},
            },
            self.device,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CausalFlowDAG:
        """Restore a model from a checkpoint.

        The nodes' ``history`` and ``flow.meta`` are refilled.

        Parameters
        ----------
        path : str | Path
            Checkpoint file written by [`save`][].
        device : str, optional
            Torch device to load onto, by default ``"cpu"``.

        Returns
        -------
        CausalFlowDAG
            The restored model, in eval mode.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        flow = cls(
            spec_from_dict(ckpt["spec"]),
            device=device,
            init=ckpt["init"],
        )
        load_weights(flow, ckpt["state_dict"])  # with the `calibrated` flags
        for name, history in ckpt["history"].items():
            flow.nodes[name].history = history
        flow.meta = ckpt["meta"]
        flow.eval()
        return flow
