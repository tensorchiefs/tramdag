"""The internal node model: one sub-model per variable.

`Node` bundles a variable's intercept (transform parameters), monotone
transform and shift modules; `CausalFlowDAG` holds one per node and the DAG
lives in which parents each node reads.

Internal-but-stable surface
---------------------------
scores.py, callbacks recipes, the read-outs and the test suite read these
names by design; renaming any of them is an API change, not a cleanup:
``kind``, ``parents``, ``shifts`` (term modules with ``key``/``parents``/
``mods``…), ``intercept`` (+ ``.groups``/``.ci_parents``/``.nets``), ``ut``,
``net_input``, ``theta_shift``.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .spec import (
    NodeSpec,
    node_parents,
)
from .transforms import (
    StandardLogistic,
    make_univariate_transform,
    ordinal_abduct,
    ordinal_log_prob,
    ordinal_marginal_init_theta,
    ordinal_sample,
)


# %% public classes --------------------------------------------------------------------
class Node(nn.Module):
    """One dimension of the flow: an intercept plus additive shift terms.

    The intercept produces the transform parameters ``theta``. The shift
    terms add up on the latent scale. The likelihood, sampling and encoding
    branches on the node kind live in the five methods ``log_prob``,
    ``sample``, ``abduct``, ``marginal_theta`` and ``encode``, and nowhere
    else; the rest of the package reads ``kind`` for dispatch and display
    only. A third node kind earns a protocol; two stay an if/else in one
    place.

    Parameters
    ----------
    node : NodeSpec
        Specification of the node.
    spec : dict[str, NodeSpec]
        The full DAG specification. Needed for the parent feature widths.
    """

    def __init__(self, node: NodeSpec, spec: dict[str, NodeSpec]):
        super().__init__()
        self.kind = node.kind
        terms = node.terms
        self.parents = tuple(node_parents(node))
        if node.kind == "continuous":
            self.ut = make_univariate_transform(node.transform, **node.transform_kwargs)
            n_params = self.ut.n_params
        else:
            self.ut = None
            self.levels = node.levels
            n_params = node.levels - 1
        # the intercept slot: the free theta_0, one joint net, or one net per
        # parent summed in coefficient space — `intercept_module` picks
        self.intercept = terms[0].module(terms[0], spec, n_params)
        # one module per shift term, built in formula order (the seeded RNG
        # stream is pinned to it); each names its own key: the parent, "a+b"
        # for a joint CS, the treatment for a VC
        self.shifts = nn.ModuleDict()
        for term in terms[1:]:
            m = term.module(term, spec)
            self.shifts[m.key] = m

    def encode(self, values: Tensor) -> Tensor:
        """Encode this node's values for use as a parent feature.

        The original TRAM-DAG convention: a continuous parent stays raw, shape
        ``(n, 1)``; an ordinal parent is one-hot encoded, shape ``(n, levels)``.
        """
        if self.kind == "ordinal":
            one_hot = nn.functional.one_hot(values.long(), num_classes=self.levels)
            return one_hot.to(values.dtype)
        return values.view(-1, 1)

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
        """Give the node's marginal-start theta, or ``None`` when there is none.

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
        ``calibrate``. Every network input goes through here — training and
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
