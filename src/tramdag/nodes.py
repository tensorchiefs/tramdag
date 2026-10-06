"""The node model: one TRAM regression per variable, alone or in a flow.

`Node` bundles a variable's intercept (transform parameters), monotone
transform and shift modules, and fits itself; `CausalFlowDAG` holds one per
node and the DAG lives in which parents each node reads. ``scores.py``, the
read-outs and the tests read ``kind``, ``parents``, ``shifts``, ``intercept``, ``ut``,
``net_input`` and ``theta_shift`` by name.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .fitting import NodeFitMixin
from .modules import LinearShiftModule
from .spec import (
    NodeSpec,
    node_parents,
    spec_from_dict,
    spec_to_dict,
)
from .transforms import (
    StandardLogistic,
    make_univariate_transform,
    ordinal_abduct,
    ordinal_log_prob,
    ordinal_marginal_init_theta,
    ordinal_sample,
)


# %% public functions ------------------------------------------------------------------
def encode(values: Tensor, kind: str | int) -> Tensor:
    """Encode one column as a parent feature.

    ``kind`` is a schema entry: ``"continuous"`` stays raw, shape ``(n, 1)``;
    a level count one-hot encodes, shape ``(n, levels)``.
    """
    if kind == "continuous":
        return values.view(-1, 1)
    one_hot = nn.functional.one_hot(values.long(), num_classes=kind)
    return one_hot.to(values.dtype)


def schema_entry(node_spec: NodeSpec) -> str | int:
    """Give a node's schema entry: ``"continuous"`` or its level count."""
    return "continuous" if node_spec.kind == "continuous" else node_spec.levels


def check_columns(df: pd.DataFrame, cols) -> None:
    """Name the columns ``df`` lacks, before any tensor op would."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"the data frame lacks the column(s) {missing}; this needs "
            f"{list(cols)}, the frame has {list(df.columns)}"
        )


def check_level_values(name: str, values, levels: int) -> None:
    """Reject ordinal values that are not level indices of their node.

    ``bincount``, the cutpoint likelihood and the one-hot parent
    encoding all take the values as ``0..levels-1``; a 1-based or
    non-integer value would silently be truncated instead of failing.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return
    fractional = bool((v != np.round(v)).any())
    if fractional or v.min() < 0 or v.max() >= levels:
        raise ValueError(
            f"node {name!r}: an ordinal column holds the level indices "
            f"0..{levels - 1}, got values in [{v.min()}, {v.max()}]"
            f"{' (non-integer)' if fractional else ''}"
        )


def tensorize(
    df: pd.DataFrame, cols, levels: dict[str, int], dtype, device
) -> dict[str, Tensor]:
    """DataFrame columns -> one ``(n,)`` tensor each, in ``dtype`` on ``device``.

    A column named in ``levels`` is checked to hold level indices first.
    """
    check_columns(df, cols)
    out = {}
    for c in cols:
        values = df[c].to_numpy(dtype=float)
        if c in levels:
            check_level_values(c, values, levels[c])
        out[c] = torch.tensor(values, dtype=dtype, device=device)
    return out


def write_checkpoint(path: str | Path, payload: dict, device) -> None:
    """Save ``payload`` plus a ``meta`` block with ``torch.save``.

    ``meta`` holds the tramdag version, the save time and the device.

    Raises
    ------
    ValueError
        If the spec does not pickle (a callable ``input_transform`` that is not
        a module-level function).
    """
    from . import __version__  # lazy: circular through the package root

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "tramdag_version": __version__,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
    }
    try:
        torch.save(payload | {"meta": meta}, path)
    except (pickle.PicklingError, AttributeError) as err:
        raise ValueError(
            "the spec does not serialize: a callable input_transform "
            "must be a picklable module-level function "
            "— use 'minmax'/'standardize', or def the function at "
            "module level."
        ) from err


def load_weights(module: nn.Module, state_dict: dict) -> None:
    """Load a checkpoint's state dict into a freshly built module.

    A callable input transform's frozen train columns take the checkpoint's
    shape first; their buffer is empty until calibration.
    """
    for name, t in state_dict.items():
        if not name.endswith(".train_cols"):
            continue
        buf = module.get_buffer(name)
        if buf.shape != t.shape:
            mod_path, _, buf_name = name.rpartition(".")
            module.get_submodule(mod_path).register_buffer(
                buf_name, torch.empty_like(t)
            )
    module.load_state_dict(state_dict)


# %% public classes --------------------------------------------------------------------
class Node(NodeFitMixin, nn.Module):
    r"""One dimension of the flow: an intercept plus additive shift terms.

    The intercept produces the transform parameters $\vartheta$. The shift
    terms add up on the latent scale. The likelihood, sampling and encoding
    branches on the node kind live in the five methods ``log_prob``,
    ``sample``, ``abduct``, ``marginal_theta`` and ``encode``, and in the
    module functions ``encode`` and ``schema_entry``; the rest of the package
    reads ``kind`` for dispatch and display only.

    Parameters
    ----------
    name : str
        The node's column name.
    node_spec : NodeSpec
        Specification of the node.
    schema : dict[str, str | int]
        The parent schema, ``{parent: "continuous" | n_levels}``. It sizes
        the parent features: a continuous parent enters raw, an ordinal one
        one-hot over its levels.
    """

    def __init__(self, name: str, node_spec: NodeSpec, schema: dict[str, str | int]):
        super().__init__()
        self.name = name
        self.node_spec = node_spec
        self.kind = node_spec.kind
        terms = node_spec.terms
        self.parents = tuple(node_parents(node_spec))
        missing = [p for p in self.parents if p not in schema]
        if missing:
            raise ValueError(f"node {name!r}: the schema lacks the parent(s) {missing}")
        self.schema = {p: schema[p] for p in self.parents}
        if node_spec.kind == "continuous":
            self.ut = make_univariate_transform(
                node_spec.transform, **node_spec.transform_kwargs
            )
            n_params = self.ut.n_params
        else:
            self.ut = None
            self.levels = node_spec.levels
            n_params = node_spec.levels - 1
        self.encoding = schema_entry(node_spec)  # this node's own schema entry
        # the intercept slot: the free theta_0, one joint net, or one net per
        # parent summed in coefficient space — `intercept_module` picks
        self.intercept = terms[0].module(terms[0], self.schema, n_params)
        # one module per shift term, built in formula order (the seeded RNG
        # stream is pinned to it); each names its own key: the parent, "a+b"
        # for a joint CS, the treatment for a VC
        self.shifts = nn.ModuleDict()
        for term in terms[1:]:
            m = term.module(term, self.schema)
            self.shifts[m.key] = m
        # calibrate() takes the data-dependent state once; a buffer, not a Python
        # bool, so the flag rides in the state dict and a loaded node does not
        # recalibrate on its next fit
        self.register_buffer("calibrated", torch.tensor(False))
        self.history: dict = {"train": []}  # mean train NLL per epoch
        self.meta: dict = {}  # provenance a load() fills (version, time)

    def encode(self, values: Tensor) -> Tensor:
        """Encode this node's values for use as a parent feature.

        A continuous parent stays raw, shape ``(n, 1)``; an ordinal parent is
        one-hot encoded, shape ``(n, levels)``.
        """
        return encode(values, self.encoding)

    def features(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode this node's parents out of a raw tensor dict."""
        return {p: encode(values[p], self.schema[p]) for p in self.parents}

    def tensorize(self, df: pd.DataFrame, cols=None) -> dict[str, Tensor]:
        """DataFrame columns -> one ``(n,)`` tensor each, in the node's dtype.

        ``cols=None`` takes the node's own column and its parents. Ordinal
        columns, own or parent, are checked to hold level indices.

        Raises
        ------
        KeyError
            If the frame lacks one of the columns.
        ValueError
            If an ordinal column is not a level index.
        """
        cols = (self.name, *self.parents) if cols is None else cols
        kinds = self.schema | {self.name: self.encoding}
        levels = {c: k for c, k in kinds.items() if k != "continuous"}
        p = next(self.parameters())
        return tensorize(df, cols, levels, p.dtype, p.device)

    def calibrate(self, train_df: pd.DataFrame, *, marginal_init: bool = False) -> Node:
        r"""Take the data-dependent state from the training rows, once.

        Every term calibrates itself: the intercept term maps the node's
        train ``range_q``/``1 - range_q`` quantiles (an intercept option,
        default 5%/95%; ``0.0`` is the min/max) onto the transform's pre-scaled
        domain, and every term with an ``input_transform=`` freezes its
        statistics (minmax lo/hi, standardize mean/std, a callable's frozen
        train columns).

        The first ``fit`` or ``fit_classical`` calls this when it has not run
        yet; a loaded node is already calibrated, and later fits on other rows
        reuse this state. Data on a new scale needs a new node.

        ``marginal_init`` additionally starts a simple intercept at the
        column's marginal: a Bernstein intercept at the Bernstein
        approximation of $\operatorname{logit} \hat F(y)$, an ordinal one at
        the marginal class log-odds; spline/affine intercepts and intercepts
        with parents are untouched. The start rides on this method's guard, so
        a second ``fit`` (the next phase of a schedule) continues training
        instead of discarding the intercept it just trained.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training rows with the node's column and its parents' columns.
        marginal_init : bool, optional
            Also set the calibrated start, by default ``False``: an
            uninitialized intercept starts at zuko's zero instead.

        Returns
        -------
        Node
            ``self``.
        """
        if bool(self.calibrated):
            return self
        self.tensorize(train_df)  # the columns and the ordinal levels
        self.intercept.calibrate_intercept(train_df, train_df[self.name], self.ut)
        for m in self.shifts.values():
            m.calibrate(train_df)
        if marginal_init:
            theta = self.marginal_theta(train_df[self.name].to_numpy())
            if theta is not None:  # a spline or affine transform has no start
                self.intercept.marginal_start(theta)
        self.calibrated.fill_(True)
        return self

    def log_prob(self, theta: Tensor, shift: Tensor, x: Tensor) -> Tensor:
        """``log p(x | pa)`` from the transform parameters and the shift."""
        if self.kind == "continuous":
            u0, ladj = self.ut.forward(theta, x)
            return StandardLogistic.log_prob(u0 + shift) + ladj
        return ordinal_log_prob(theta, shift, x)

    def sample(self, theta: Tensor, shift: Tensor, u: Tensor) -> Tensor:
        """Push the latent ``u`` forward to an observed value."""
        if self.kind == "continuous":
            return self.ut.inverse(theta, u - shift)
        return ordinal_sample(theta, shift, u)

    def abduct(self, theta: Tensor, shift: Tensor, x: Tensor, generator=None) -> Tensor:
        """Recover the latent: exact (continuous) or truncated-sampled (ordinal)."""
        if self.kind == "continuous":
            u0, _ = self.ut.forward(theta, x)
            return u0 + shift
        return ordinal_abduct(theta, shift, x, generator=generator)

    def marginal_theta(self, column: np.ndarray):
        r"""Give the node's marginal-start $\vartheta$, or ``None`` when there is none.

        Ordinal: the empirical class log-odds. Continuous: the transform's own
        marginal start over the same column (``None`` for spline/affine —
        nothing to set). Only a free intercept applies it.
        """
        if self.kind == "ordinal":
            counts = np.bincount(column.astype(np.int64), minlength=self.levels)
            return ordinal_marginal_init_theta(counts)
        return self.ut.marginal_init_theta(column)

    def net_input(self, feats: dict[str, Tensor], parents, key: str) -> Tensor:
        """Concatenate parent features for one term's network.

        ``key`` names the term ("@I" for the intercept, the shift key
        otherwise); a term with an ``input_transform`` gets its continuous
        parent columns transformed with the statistics frozen at
        calibration. Every network input goes through here — training and
        the read-outs (``varying_coef``, ``intercept_contributions``) alike —
        so the model seen at inference is the model that was fitted. Linear
        shifts and the VC treatment column are not network inputs and never
        pass through.
        """
        term = self.intercept if key == "@I" else self.shifts[key]
        tr = term.input_transform
        cols = []
        for p in parents:
            x = feats[p]
            if tr is not None and p in tr.cols:
                x = tr(x, tr.cols.index(p))
            cols.append(x)
        return torch.cat(cols, dim=1)

    def theta_shift(self, feats: dict[str, Tensor], n: int) -> tuple[Tensor, Tensor]:
        """Compute the transform parameters and the total shift of the node.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Encoded parent features keyed by parent name, plus the terms'
            side columns (a centered VC's propensities — frozen from the
            training frame, injected live by the flow at query time).
        n : int
            Batch size.

        Returns
        -------
        tuple[Tensor, Tensor]
            The transform parameters, shape ``(n, P)``, and the total
            shift, shape ``(n,)``.

        Raises
        ------
        RuntimeError
            If a centered VC term is evaluated without its propensity
            column in ``feats``.
        """
        theta = self.intercept.theta_value(self, feats, n)
        shift = torch.zeros(n, dtype=theta.dtype, device=theta.device)
        # plain shifts first, then VC (stable sort keeps the pinned order)
        for m in sorted(self.shifts.values(), key=lambda m: m.order):
            shift = shift + m.shift_value(self, feats)
        return theta, shift

    def ls_weights(self) -> dict[str, np.ndarray]:
        """Give the ``LS`` weights as ``{parent: array}``; empty without ``LS``."""
        return {
            parent: m.weight.detach().cpu().numpy().ravel().copy()
            for parent, m in self.shifts.items()
            if isinstance(m, LinearShiftModule)
        }

    def save(self, path: str | Path) -> None:
        """Write the node, its history and its provenance to a checkpoint.

        The file holds the node's name, spec, parent schema and weights, its
        training ``history``, and a ``meta`` block with the tramdag version,
        the save time and the device.

        Parameters
        ----------
        path : str | Path
            Target file. Parent directories are created when missing.
        """
        write_checkpoint(
            path,
            {
                "name": self.name,
                "node_spec": spec_to_dict({self.name: self.node_spec})[self.name],
                "schema": self.schema,
                "state_dict": self.state_dict(),
                "history": self.history,
            },
            next(self.parameters()).device,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Node:
        """Restore a node from a checkpoint written by [`save`][].

        Parameters
        ----------
        path : str | Path
            Checkpoint file.
        device : str, optional
            Torch device to load onto, by default ``"cpu"``.

        Returns
        -------
        Node
            The restored node, in eval mode.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        name = ckpt["name"]
        spec = spec_from_dict({name: ckpt["node_spec"]})[name]
        node = cls(name, spec, ckpt["schema"]).to(device)
        load_weights(node, ckpt["state_dict"])
        node.history = ckpt["history"]
        node.meta = ckpt["meta"]
        return node.eval()
