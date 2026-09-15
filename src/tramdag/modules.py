"""The term modules: one ``nn.Module`` per term, built from the term's spec class.

A spec term ([`Term`][] subclass — ``LS``, ``CS``, ``VC``, ``Fn``, ``I``) is
plain data and carries the spec-level rules, and its ``module`` attribute is
the class here that trains it (``LinearShift.module is LinearShiftModule``).
A module is constructed from its term and the spec, ``module(term, spec)``
(the intercept slot adds ``n_params``), holds the term's network and owns the
runtime behaviour: ``shift_value``/``theta_value``, ``post_init``,
``regularizer``, ``finalize``, ``score_columns`` and the side-input contract.
This module imports nothing from [`spec`][tramdag.spec]: it reads a spec
node's ``kind``, ``levels`` and a term's ``parents`` and options, so ``spec``
can import it.

A custom term is two classes: a [`ShiftModule`][tramdag.modules.ShiftModule]
subclass with ``__init__(term, spec)`` and ``shift_value``, and a ``Term``
subclass for the options and checks whose ``module`` is that class.

The widths and the activation of every network are options of the term
classes, written once in the signatures in [`spec`][tramdag.spec]; the modules
here take what they are given.

| Module | Network | Term |
|--------------------------|-------------------------------------------|------|
| `LinearShiftModule` | `Linear(n, 1, bias=False)` | `LS` |
| `ComplexShiftModule` | hidden stack to 1 output, no bias | `CS` |
| `ComplexInterceptModule` | hidden stack to `n_params` outputs, no bias | `I(...)` |
| `AdditiveInterceptModule` | one stack per parent, outputs summed | additive `I` |
| `SimpleInterceptModule` | free parameter vector, no parent | `I()` |
| `VaryingCoefficientModule` | `beta0` + penalized hidden stack to 1 | `VC` |
| `FnShiftModule` | a user callable or `nn.Module` | `Fn` |

Parent features: a continuous parent enters raw, in one column; an ordinal
parent one-hot, in ``levels`` columns. ``ACTIVATIONS`` maps the activation
names a term may name to their ``torch.nn`` classes.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    import pandas as pd

    from .nodes import Node
    from .spec import NodeSpec, Term

# %% global variables ------------------------------------------------------------------
ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


# %% private functions -----------------------------------------------------------------
def _nn(
    n_in: int,
    units: tuple[int, ...],
    n_out: int,
    *,
    activation: str,
    batch_norm: bool,
    zero_init_last: bool = False,
) -> nn.Sequential:
    """Build the one NN shape every networked term uses.

    Hidden layers of the given ``units``, each followed by ``activation``,
    then a bias-free output layer. With ``batch_norm`` a ``BatchNorm1d`` sits
    between each hidden layer and its activation.

    Parameters
    ----------
    n_in : int
        Input width.
    units : tuple[int, ...]
        Hidden layer widths.
    n_out : int
        Output width.
    activation : str
        Key of ``ACTIVATIONS``.
    batch_norm : bool
        Normalize each hidden layer before its activation. It needs more than
        one row per batch, and the fitted function depends on the training
        batch statistics, so inference runs in ``eval()`` mode.
    zero_init_last : bool, optional
        Zero the output layer, by default ``False``.

    Returns
    -------
    nn.Sequential
        The network.

    Raises
    ------
    ValueError
        If ``activation`` is not a key of ``ACTIVATIONS``.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; choose one of {sorted(ACTIVATIONS)}"
        )
    make_activation = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    width = n_in
    for u in units:
        layers.append(nn.Linear(width, u))
        if batch_norm:
            layers.append(nn.BatchNorm1d(u))
        layers.append(make_activation())
        width = u
    out = nn.Linear(width, n_out, bias=False)
    if zero_init_last:
        nn.init.zeros_(out.weight)
    return nn.Sequential(*layers, out)


def _attach_input_transform(
    m: nn.Module, term: Term, spec: dict[str, NodeSpec], parents: tuple[str, ...]
) -> None:
    """Register the term's input transform over its continuous parents.

    Ordinal one-hots pass through untransformed, so a term whose network
    parents are all ordinal carries none.
    """
    if term.input_transform is None:
        return
    cps = tuple(p for p in parents if spec[p].kind == "continuous")
    if cps:
        m.add_module("_input_transform", _InputTransform(term.input_transform, cps))


# %% public functions ------------------------------------------------------------------
def feat_width(spec: dict[str, NodeSpec], parents: tuple[str, ...]) -> int:
    """Total feature width of the parents (ordinal one-hot, continuous raw)."""
    return sum(spec[p].levels if spec[p].kind == "ordinal" else 1 for p in parents)


def intercept_module(
    term: Term, spec: dict[str, NodeSpec], n_params: int
) -> InterceptModule:
    """Construct the node's intercept module from its ``I`` term.

    One ``I`` term becomes one of three modules: the free theta without
    parents, one joint net over all parents, or one net per parent with the
    coefficient vectors summed (``allow_interaction=False``).
    """
    if not term.parents:
        return SimpleInterceptModule(term, spec, n_params)
    if term.allow_interaction:
        return ComplexInterceptModule(term, spec, n_params)
    return AdditiveInterceptModule(term, spec, n_params)


# %% private classes -------------------------------------------------------------------
class _InputTransform(nn.Module):
    """One term's frozen network-input transform.

    ``calibrate`` takes the statistics from the training rows once:
    ``"minmax"`` freezes per-column lo/hi, ``"standardize"`` mean/std, and a
    callable keeps the raw training columns and is applied per batch as
    ``fn(x, train)`` — so train statistics inside the callable are always the
    frozen training data, never the batch's.
    """

    def __init__(self, value, cols: tuple[str, ...]):
        super().__init__()
        self.method = "callable" if callable(value) else value
        self.fn = value if callable(value) else None
        self.cols = cols  # the term's continuous parents, in parent order
        k = len(cols)
        if self.method == "minmax":
            self.register_buffer("lo", torch.zeros(k))
            self.register_buffer("hi", torch.ones(k))
        elif self.method == "standardize":
            self.register_buffer("mean", torch.zeros(k))
            self.register_buffer("std", torch.ones(k))
        else:  # callable: the raw train columns, shaped at calibrate
            self.register_buffer("train_cols", torch.zeros(0, k))

    def set_stats(self, cols: Tensor) -> None:
        """Freeze the statistics from the raw ``(n_train, k)`` train columns."""
        if self.method == "minmax":
            self.lo.copy_(cols.min(0).values)
            self.hi.copy_(cols.max(0).values)
        elif self.method == "standardize":
            self.mean.copy_(cols.mean(0))
            self.std.copy_(cols.std(0))
        else:
            self._buffers["train_cols"] = cols.detach().to(self.train_cols.device)

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Transform one continuous parent column ``(n, 1)``."""
        if self.method == "minmax":
            return (x - self.lo[i]) / (self.hi[i] - self.lo[i])
        if self.method == "standardize":
            return (x - self.mean[i]) / self.std[i]
        return self.fn(x, self.train_cols[:, i : i + 1])


# %% public classes --------------------------------------------------------------------
class TermModule:
    """What every term module shares: the input transform and its calibration."""

    @property
    def input_transform(self):
        """The term's frozen network-input transform, or ``None``.

        A module's ``__init__`` registers one (``_attach_input_transform``) when
        the term declares ``input_transform=`` over continuous parents; a plain
        class attribute would shadow the registered submodule.
        """
        return self._modules.get("_input_transform")

    def calibrate(self, train_df: pd.DataFrame) -> None:
        """Freeze this term's data-dependent state: the input-transform stats.

        ``CausalFlowDAG.calibrate`` calls this once per term; a term without
        an ``input_transform`` has nothing to freeze. The intercept slot has a
        second step on top of this one — see
        [`calibrate_intercept`][tramdag.modules.InterceptModule.calibrate_intercept].
        """
        tr = self.input_transform
        if tr is None:
            return
        cols = torch.stack(
            [
                torch.as_tensor(train_df[p].to_numpy(dtype=np.float32).copy())
                for p in tr.cols
            ],
            dim=1,
        )
        tr.set_stats(cols)


class ShiftModule(TermModule, ABC):
    """A shift term's behavior hooks, on top of its network.

    ``__init__(term, spec)`` builds the network from the term's options and the
    parents' widths in the spec, and sets ``key`` (the node's ModuleDict key)
    and ``parents`` (the term's written parents); a subclass may keep more
    (``VaryingCoefficientModule`` keeps ``mods``/``t_is_ord``/``center_col``).
    Layers are built in a fixed order under fixed attribute names, so
    state-dict paths and the seeded RNG stream stay bit-stable.
    """

    scored = False  # True when score_columns gives coefficients
    order = 0  # shifts sum in this order (VC last: the pinned order)

    key: str
    parents: tuple[str, ...]

    @abstractmethod
    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """Give this term's contribution to the node's shift, shape ``(n,)``.

        ``feats`` holds the node's encoded parents plus this term's side
        columns (frozen from the data frame during training, injected live
        by the flow at query time).
        """

    def post_init(self) -> None:
        """Re-apply construction-time invariants after a global weight init."""

    def regularizer(self) -> Tensor | None:
        """Give the term's penalty on the total-likelihood scale, or ``None``."""
        return None

    def finalize(self, node: Node, feats: dict) -> None:
        """Run the term's post-fit step (after the after-fit callbacks)."""

    def score_columns(self, node: Node, flow, feats: dict, dlds) -> dict:
        """Give the per-observation score columns of this term's coefficients.

        Empty for a term with no interpretable coefficient (``CS``).
        """
        return {}

    def side_columns(self) -> tuple[str, ...]:
        """Name the data-frame columns this term needs beyond the parents.

        Training reads them from ``train_df`` like any other column (so the
        validation split and the minibatch slicing come for free).
        """
        return ()

    def check_column(self, node_name: str, col: str, values) -> None:
        """Validate one side column's values at fit time."""

    def live_side(self, flow, values: dict, n: int) -> dict:
        """Recompute the side columns from the fitted flow, at query time."""
        return {}

    def extra_columns(self, flow) -> list[str]:
        """List the extra columns queries must tensorize for ``live_side``."""
        return []


class InterceptModule(TermModule, ABC):
    """The intercept slot's behavior hooks, on top of its network.

    A node has exactly one intercept term (normalization guarantees
    ``node.terms[0]``); it produces the transform parameters ``theta``.
    ``groups`` carries the parent groups — empty for a simple intercept,
    one tuple for a joint net, one per parent for an additive one — and
    ``ci_parents`` their flat order. [`intercept_module`][] picks the class
    for an ``I`` term; each class constructs from ``(term, spec, n_params)``.
    """

    groups: list[tuple[str, ...]]
    ci_parents: list[str]

    def calibrate_intercept(self, train_df: pd.DataFrame, own, ut) -> None:
        """Freeze the input stats and set the transform's domain, once.

        The intercept slot calibrates one thing more than every other term, so
        it says so in its own name rather than widening ``calibrate``: ``own``
        is the node's training column and ``ut`` its monotone transform
        (``None`` for an ordinal node — cutpoints have no domain), whose
        ``range_q``/``1 - range_q`` quantiles map onto the pre-scaled domain.
        """
        self.calibrate(train_df)
        if ut is None:
            return
        q = own.quantile([ut.range_q, 1.0 - ut.range_q])
        if q.iloc[1] <= q.iloc[0]:
            raise ValueError(
                f"node {own.name!r}: the {ut.range_q:.0%} and "
                f"{1 - ut.range_q:.0%} quantiles coincide at {q.iloc[0]}. A "
                "continuous node needs a spread of values; a level index "
                "needs OrdinalNode()."
            )
        ut.set_range(q.iloc[0], q.iloc[1])

    @abstractmethod
    def theta_value(self, node: Node, feats: dict, n: int) -> Tensor:
        """Give the transform parameters, shape ``(n, P)``."""

    def marginal_start(self, theta: Tensor) -> None:
        """Set the calibrated marginal start; only a free intercept has one."""


class SimpleInterceptModule(InterceptModule, nn.Module):
    """The free simple intercept: one theta vector, no parents.

    Parameters
    ----------
    term : Intercept
        The parentless intercept term.
    spec : dict[str, NodeSpec]
        The DAG specification (unused: no parents to size).
    n_params : int
        Number of transform parameters.
    """

    def __init__(self, term: Term, spec: dict[str, NodeSpec], n_params: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_params))
        self.groups, self.ci_parents = [], []

    def forward(self, n: int) -> Tensor:
        """Broadcast the parameters over a batch of ``n`` rows, shape ``(n, P)``."""
        return self.theta.unsqueeze(0).expand(n, -1)

    def theta_value(self, node: Node, feats: dict, n: int) -> Tensor:
        """Broadcast the free theta over the batch."""
        return self(n)

    def marginal_start(self, theta: Tensor) -> None:
        """Start at the node's data marginal."""
        with torch.no_grad():
            self.theta.copy_(theta)


class ComplexInterceptModule(InterceptModule, nn.Module):
    """A single (possibly joint multi-parent) complex intercept net.

    Several parents given to one term feed a single network, so they interact.
    The term's ``units``, ``activation`` and ``batch_norm`` size the network.

    Parameters
    ----------
    term : Intercept
        The intercept term, with at least one parent.
    spec : dict[str, NodeSpec]
        The DAG specification, for the parents' feature widths.
    n_params : int
        Number of transform parameters to produce.
    """

    def __init__(self, term: Term, spec: dict[str, NodeSpec], n_params: int):
        super().__init__()
        self.net = _nn(
            feat_width(spec, term.parents),
            term.units,
            n_params,
            activation=term.activation,
            batch_norm=term.batch_norm,
        )
        self.groups = [tuple(term.parents)]
        self.ci_parents = list(term.parents)
        _attach_input_transform(self, term, spec, tuple(term.parents))

    def forward(self, x: Tensor) -> Tensor:
        """Map parent features ``(n, n_features)`` to parameters ``(n, n_params)``."""
        return self.net(x)

    def theta_value(self, node: Node, feats: dict, n: int) -> Tensor:
        """Run the one net over the joint parent features."""
        return self(node.net_input(feats, self.ci_parents, "@I"))


class AdditiveInterceptModule(InterceptModule, nn.Module):
    """``allow_interaction=False``: one net per parent, outputs summed.

    Each parent reshapes the transform independently, in unconstrained
    coefficient space.

    Parameters
    ----------
    term : Intercept
        The intercept term with ``allow_interaction=False`` and two or more
        parents.
    spec : dict[str, NodeSpec]
        The DAG specification, for the parents' feature widths.
    n_params : int
        Number of transform parameters to produce.
    """

    def __init__(self, term: Term, spec: dict[str, NodeSpec], n_params: int):
        super().__init__()
        self.groups = [(p,) for p in term.parents]
        self.ci_parents = list(term.parents)
        # the submodule stays named `nets`: tests/test_statedict_stability.py
        # pins the state-dict path
        self.nets = nn.ModuleList(
            _nn(
                feat_width(spec, grp),
                term.units,
                n_params,
                activation=term.activation,
                batch_norm=term.batch_norm,
            )
            for grp in self.groups
        )
        _attach_input_transform(self, term, spec, tuple(term.parents))

    def theta_value(self, node: Node, feats: dict, n: int) -> Tensor:
        """Sum the per-parent nets in coefficient space."""
        return sum(
            net(node.net_input(feats, grp, "@I"))
            for net, grp in zip(self.nets, self.groups, strict=True)
        )


class LinearShiftModule(ShiftModule, nn.Module):
    r"""``LS`` — one raw-unit coefficient per feature of the single parent, no bias.

    For an ordinal child, $\exp(\beta)$ is an odds ratio. Keyed by the parent's
    name.

    Parameters
    ----------
    term : LinearShift
        The term, with its one parent.
    spec : dict[str, NodeSpec]
        The DAG specification, for the parent's feature width.
    """

    scored = True

    def __init__(self, term: Term, spec: dict[str, NodeSpec]):
        super().__init__()
        self.fc = nn.Linear(feat_width(spec, term.parents), 1, bias=False)
        self.key = term.parents[0]
        self.parents = tuple(term.parents)

    @property
    def weight(self) -> Tensor:
        """Tensor: the shift coefficients, shape ``(n_features,)``."""
        return self.fc.weight.squeeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Give the shift ``(n,)`` from the encoded features ``(n, n_features)``."""
        return self.fc(x).squeeze(-1)

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """Give the raw parent column times the weight — no input transform."""
        return self(torch.cat([feats[p] for p in self.parents], dim=1))

    def score_columns(self, node: Node, flow, feats: dict, dlds) -> dict:
        r"""One column per weight: the parent (continuous) or its one-hot levels.

        $\partial \ell_i / \partial \beta = (\partial \ell_i / \partial s_i)\, x_i$,
        analytic and exact.
        """
        (parent,) = self.parents  # an LS term has exactly one parent
        psi = (dlds.unsqueeze(1) * feats[parent]).cpu().numpy()
        if flow.spec[parent].kind == "ordinal":
            return {f"{parent}[{k}]": psi[:, k] for k in range(psi.shape[1])}
        return {self.key: psi[:, 0]}


class ComplexShiftModule(ShiftModule, nn.Module):
    """``CS`` — an additive network shift ``g(x)`` over its parents.

    One net over the concatenated parents, keyed ``'a'`` or ``'a+b'``. The
    term's ``units``, ``activation`` and ``batch_norm`` size the network.

    Parameters
    ----------
    term : ComplexShift
        The term, with its parents.
    spec : dict[str, NodeSpec]
        The DAG specification, for the parents' feature widths.
    """

    def __init__(self, term: Term, spec: dict[str, NodeSpec]):
        super().__init__()
        self.parents = tuple(term.parents)
        self.net = _nn(
            feat_width(spec, self.parents),
            term.units,
            1,
            activation=term.activation,
            batch_norm=term.batch_norm,
        )
        self.key = "+".join(self.parents)
        _attach_input_transform(self, term, spec, self.parents)

    def forward(self, x: Tensor) -> Tensor:
        """Give the shift ``(n,)`` from the encoded features ``(n, n_features)``."""
        return self.net(x).squeeze(-1)

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """Give the net over this term's (possibly input-transformed) features."""
        return self(node.net_input(feats, self.parents, self.key))


class VaryingCoefficientModule(ShiftModule, nn.Module):
    r"""``VC``: $\beta(\text{mod})\, x_t$ with $\beta(x) = \beta_0 + b_\Theta(x)$.

    The weights of $b_\Theta$ carry the L2 ``penalty`` (``l2``; ``fit`` adds
    ``penalty * l2()`` to the summed NLL); $\beta_0$ is not penalized. The
    output layer starts at zero, so $\beta(x) = \beta_0$ at construction. With
    ``n_features == 0`` there is no network and the term is ``LS(t)``. After
    the fit, ``recenter`` moves the training-mean of ``b_theta`` into
    ``beta0`` through the ``center`` buffer, a reparameterization that leaves
    the modelled function unchanged. Keyed by the treatment's name.

    Parameters
    ----------
    term : VaryingCoefficient
        The term: treatment first in ``parents``, then the modifiers.
    spec : dict[str, NodeSpec]
        The DAG specification, for the modifiers' feature widths and the
        treatment's kind.
    """

    scored = True
    order = 1

    def __init__(self, term: Term, spec: dict[str, NodeSpec]):
        super().__init__()
        t, mods = term.parents[0], tuple(term.parents[1:])
        self.penalty = float(term.penalty)
        self.beta0 = nn.Parameter(torch.zeros(()))
        self.register_buffer("center", torch.zeros(()))
        n_features = feat_width(spec, mods)
        if n_features > 0:
            # zero-initialised output: beta(x) == beta0 at init
            self.net = _nn(
                n_features,
                term.units,
                1,
                activation=term.activation,
                batch_norm=term.batch_norm,
                zero_init_last=True,
            )
        else:
            self.net = None
        self.key = t
        self.parents = tuple(term.parents)
        self.mods = mods
        self.t_is_ord = spec[t].kind == "ordinal"
        self.center_col = term.center
        _attach_input_transform(self, term, spec, mods)

    def beta(self, mod_feats: Tensor | None, n: int) -> Tensor:
        """Give the effect values ``beta(x)``, shape ``(n,)``.

        ``mod_feats`` is ``None`` if, and only if, the term has no modifiers;
        ``n`` is the batch size, used in that case.
        """
        if self.net is None:
            return (self.beta0 - self.center).expand(n)
        return self.beta0 + self.net(mod_feats).squeeze(-1) - self.center

    def forward(self, t: Tensor, mod_feats: Tensor | None) -> Tensor:
        r"""Give the shift $\beta(\text{modifiers})\, t$, shape ``(n,)``.

        ``t`` is the raw treatment column ``(n, 1)``, ``mod_feats`` the encoded
        modifier features or ``None`` without modifiers.
        """
        return self.beta(mod_feats, t.shape[0]) * t.squeeze(-1)

    def l2(self) -> Tensor:
        """Sum the squared ``b_theta`` weights, the penalized quantity.

        ``beta0`` is never included; 0 without modifiers.
        """
        if self.net is None:
            return torch.zeros((), device=self.beta0.device, dtype=self.beta0.dtype)
        return sum(p.pow(2).sum() for p in self.net.parameters())

    @torch.no_grad()
    def recenter(self, mod_feats: Tensor | None) -> None:
        """Re-split ``beta0`` and ``b_theta`` so ``b_theta`` has mean zero.

        The mean is taken over ``mod_feats``. The removed constant moves into
        ``beta0``, so the modelled function does not change. Without
        modifiers this does nothing.
        """
        if self.net is None:
            return
        delta = (self.net(mod_feats).squeeze(-1) - self.center).mean()
        self.center += delta
        self.beta0 += delta

    def regressor(self, feats: dict) -> Tensor:
        """Give the ``(n, 1)`` column ``beta`` multiplies — the treatment, raw.

        The one-hot level-1 indicator for a binary ordinal treatment, the
        value itself for a continuous one; a centered term subtracts its
        propensity column (the Robinson regressor ``t - e_hat(x)``). It is
        also the score of ``beta0``, so ``score_columns`` reads it here.
        """
        if self.center_col and self.center_col not in feats:
            raise RuntimeError(
                f"centered VC term on {self.key!r} needs its propensity "
                f"column {self.center_col!r}. Internal callers inject it; "
                "never evaluate a centered term without its propensity."
            )
        t = feats[self.key][:, -1:] if self.t_is_ord else feats[self.key]
        if self.center_col:
            t = t - feats[self.center_col].view(-1, 1)
        return t

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        r"""$\beta(\text{modifiers})$ times the regressor, guarding a centered term."""
        t = self.regressor(feats)
        mod_feat = node.net_input(feats, self.mods, self.key) if self.mods else None
        return self(t, mod_feat)

    def post_init(self) -> None:
        r"""Re-zero the head's output layer: $\beta(x) = \beta_0$ at start."""
        if self.net is not None:
            nn.init.zeros_(self.net[-1].weight)

    def regularizer(self) -> Tensor | None:
        r"""Give the penalty $\lambda \lVert b_\Theta \rVert^2$ on the NLL scale.

        ``None`` without a head to shrink (no modifiers, or penalty 0).
        """
        if self.net is None or self.penalty == 0:
            return None
        return self.penalty * self.l2()

    def finalize(self, node: Node, feats: dict) -> None:
        """Re-split ``beta0``/``b_theta``: the head sums to zero over train."""
        if self.mods:
            self.recenter(node.net_input(feats, self.mods, self.key))

    def score_columns(self, node: Node, flow, feats: dict, dlds) -> dict:
        r"""One column, keyed by the treatment: the ``beta0`` score.

        $\partial s / \partial \beta_0$ is the term's own ``regressor``, so forward
        and score share one definition by construction.
        """
        t = self.regressor(feats)
        return {self.key: (dlds * t.squeeze(-1)).cpu().numpy()}

    def side_columns(self) -> tuple[str, ...]:
        """Name the propensity column a centered term reads from the frame."""
        return (self.center_col,) if self.center_col else ()

    def check_column(self, node_name: str, col: str, values) -> None:
        """Propensities are probabilities."""
        if not ((values >= 0) & (values <= 1)).all():
            raise ValueError(
                f"column {col!r} (centered VC on node {node_name!r}) must "
                "hold probabilities in [0, 1]"
            )

    def live_side(self, flow, values: dict, n: int) -> dict:
        """Give the full-data propensity from the flow's own treatment node.

        Detached — no gradient reaches the treatment node from this node's
        loss — and derived from the current parent values, so
        ``do``-mutilated sampling centers with the intervened ``t`` (the DML
        prediction convention; training uses the frozen out-of-fold column).
        """
        if not self.center_col:
            return {}
        p1 = flow._propensity(flow.nodes[self.key], values, n).detach()
        return {self.center_col: p1}

    def extra_columns(self, flow) -> list[str]:
        """Give the treatment's parents (a treatment cannot be centered itself)."""
        return list(flow.nodes[self.key].parents) if self.center_col else []


class FnShiftModule(ShiftModule, nn.Module):
    """``Fn`` — a user-supplied shift function over the parent features.

    A plain function contributes a fixed (non-trained) offset; an
    ``nn.Module`` registers as a submodule and trains with the flow. Keyed
    like a CS (``'a'`` or ``'a+b'``).

    Parameters
    ----------
    term : FnShift
        The term, with its callable and its parents.
    spec : dict[str, NodeSpec]
        The DAG specification, for the parents' kinds.
    """

    def __init__(self, term: Term, spec: dict[str, NodeSpec]):
        super().__init__()
        self.fn = term.fn  # an nn.Module registers as a submodule here
        self.parents = tuple(term.parents)
        self.key = "+".join(self.parents)
        _attach_input_transform(self, term, spec, self.parents)

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """Run ``fn`` on the term's features; accept ``(n,)`` or ``(n, 1)``."""
        out = self.fn(node.net_input(feats, self.parents, self.key))
        return out.squeeze(-1) if out.dim() > 1 else out
