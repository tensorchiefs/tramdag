"""The term modules: one ``nn.Module`` per term, built from the term's spec class.

A spec term ([`Term`][] subclass — ``LS``, ``CS``, ``VC``, ``Fn``, ``I``) is
plain data and carries the spec-level rules; the module here declares which
term class it builds (``data = CS``), holds the term's network and owns the
runtime behaviour: ``build``, ``shift_value``/``theta_value``, ``post_init``,
``regularizer``, ``finalize``, ``score_columns`` and the side-input contract.
[`module_for`][] finds the module of a term by that declaration, so
subclassing is the whole registration.

A custom term is two classes: a ``Term`` subclass for the options and checks,
and a [`ShiftModule`][tramdag.modules.ShiftModule] subclass with ``data =``
that term class, ``build`` and ``shift_value``.

The networks copy the defaults of the PyTorch reference this package grew out
of, ``tramdag/models/tram_models.py`` in https://github.com/buehlpa/TramDag:
``ComplexShiftDefaultTabular`` is 64-128-64 ReLU into a bias-free
``Linear(64, 1)``, ``ComplexInterceptDefaultTabular`` is 8-8 ReLU into a
bias-free ``Linear(8, n_thetas)`` with ``n_thetas=20``. A fitted model is
therefore directly comparable with that implementation. Those defaults are
**not** the TRAM-DAG paper's own nets: the paper's R implementation
(https://github.com/tensorchiefs/tram-dag) uses
``hidden_features_I = hidden_features_CS = c(2, 25, 25, 2)`` with sigmoid
activations for the triangle experiments, and a 10-100 tanh net for the
CAREFL/VACA comparisons, so every replication in ``experiments/paper/`` sets
``units=`` and ``activation=`` from its own reference script. The widths and
the activation are defaults of the term classes, written once in the
signatures in [`spec`][tramdag.spec]; the modules here take what they are
given.

| Module | Network | Term |
|--------------------------|-------------------------------------------|------|
| `LinearShiftModule` | `Linear(n, 1, bias=False)` | `LS` |
| `ComplexShiftModule` | 64-128-64 ReLU NN to 1, no bias | `CS` |
| `ComplexInterceptModule` | 8-8 ReLU NN to `n_params`, bias-free out | `I` |
| `SimpleInterceptModule` | free parameter vector, no parent | `I()` |
| `VaryingCoefficientModule` | `beta0` + penalized 16-unit NN | `VC` |

Parent features use the encoding of the original implementation: a continuous
parent enters raw, in one column; an ordinal parent one-hot, in ``levels``
columns. ``ACTIVATIONS`` holds the three activations the reference
implementations use: ``relu`` in the PyTorch reference's default classes,
``sigmoid`` in the paper's ``create_param_net``, and ``tanh`` in the paper's
``make_model`` for the CAREFL and VACA comparisons.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import Tensor, nn

from .spec import (
    CS,
    LS,
    VC,
    ContinuousNode,
    Fn,
    I,
    OrdinalNode,
    Term,
    feat_width,
)

if TYPE_CHECKING:
    import pandas as pd

    from .nodes import Node
    from .spec import NodeSpec

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
        Key of ``ACTIVATIONS``: ``"relu"`` (what the PyTorch reference's
        default classes use), ``"sigmoid"`` (the paper's ``create_param_net``)
        or ``"tanh"`` (the paper's ``make_model``, used for its CAREFL/VACA
        comparisons).
    batch_norm : bool
        Normalize each hidden layer before its activation — neither reference
        implementation uses it. It needs more than one row per batch and makes
        the fitted function depend on the training batch statistics, so ``fit``
        must leave the flow in ``eval()`` mode for inference to be reproducible
        (it does).
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


def _attach_input_transform(m, term: Term, parents: tuple, spec: dict) -> None:
    """Register the term's input transform over its continuous parents.

    Ordinal one-hots pass through untransformed, so a term whose network
    parents are all ordinal carries none.
    """
    if term.input_transform is None:
        return
    cps = tuple(p for p in parents if isinstance(spec[p], ContinuousNode))
    if cps:
        m.add_module("_input_transform", _InputTransform(term.input_transform, cps))


# %% public functions ------------------------------------------------------------------
def module_for(term: Term) -> type[TermModule]:
    """Give the module class that builds ``term``.

    A [`TermModule`][] subclass that declares ``data = <Term subclass>``
    stamps itself onto that class as ``module`` when it is defined
    (``__init_subclass__``), so subclassing is the registration.

    Raises
    ------
    ValueError
        If no module class declares the term's class.
    """
    module = getattr(type(term), "module", None)
    if module is None:
        raise ValueError(
            f"no module builds a {type(term).__name__} term. Subclass "
            f"tramdag.modules.ShiftModule with `data = {type(term).__name__}` "
            "and implement build and shift_value."
        )
    return module


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
        self.kind = "callable" if callable(value) else value
        self.fn = value if callable(value) else None
        self.cols = cols  # the term's continuous parents, in parent order
        k = len(cols)
        if self.kind == "minmax":
            self.register_buffer("lo", torch.zeros(k))
            self.register_buffer("hi", torch.ones(k))
        elif self.kind == "standardize":
            self.register_buffer("mean", torch.zeros(k))
            self.register_buffer("std", torch.ones(k))
        else:  # callable: the raw train columns, shaped at calibrate
            self.register_buffer("train_cols", torch.zeros(0, k))

    def set_stats(self, cols: Tensor) -> None:
        """Freeze the statistics from the raw ``(n_train, k)`` train columns."""
        if self.kind == "minmax":
            self.lo.copy_(cols.min(0).values)
            self.hi.copy_(cols.max(0).values)
        elif self.kind == "standardize":
            self.mean.copy_(cols.mean(0))
            self.std.copy_(cols.std(0))
        else:
            self._buffers["train_cols"] = cols.detach().to(self.train_cols.device)

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Transform one continuous parent column ``(n, 1)``."""
        if self.kind == "minmax":
            return (x - self.lo[i]) / (self.hi[i] - self.lo[i])
        if self.kind == "standardize":
            return (x - self.mean[i]) / self.std[i]
        return self.fn(x, self.train_cols[:, i : i + 1])


# %% public classes --------------------------------------------------------------------
class TermModule:
    """What every term module shares: the input transform and its calibration.

    ``data`` names the [`Term`][] subclass the module builds;
    [`module_for`][] dispatches on it.
    """

    data: ClassVar[type[Term]]

    def __init_subclass__(cls, **kwargs):
        """Stamp the module onto the term class it declares with ``data =``."""
        super().__init_subclass__(**kwargs)
        if "data" in cls.__dict__:
            cls.data.module = cls

    @property
    def input_transform(self):
        """The term's frozen network-input transform, or ``None``.

        Builds register one (``_attach_input_transform``) when the term
        declares ``input_transform=`` over continuous parents; a plain class
        attribute would shadow the registered submodule.
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

    A built module carries ``key`` (its ModuleDict key, set by ``build``) and
    ``parents`` (the term's written parents, set by the node); subclasses may
    add term-specific attributes (``VaryingCoefficientModule`` keeps
    ``mods``/``on_is_ord``/``center_col``). ``build`` constructs the module
    exactly as the node used to, so state-dict paths and the seeded RNG
    stream stay bit-stable.
    """

    scored: ClassVar[bool] = False  # True when score_columns gives coefficients
    order: ClassVar[int] = 0  # shifts sum in this order (VC last: the pinned order)

    key: str
    parents: tuple

    @classmethod
    @abstractmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> ShiftModule:
        """Construct the term module from its spec Term."""

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
    ``ci_parents`` their flat order.
    """

    data = I

    groups: list[tuple[str, ...]]
    ci_parents: list[str]

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec], n_params: int):
        """Construct the node's intercept module from its intercept Term.

        One ``I`` term becomes one of three modules: the free theta without
        parents, one joint net over all parents, or one net per parent with
        the coefficient vectors summed (``allow_interaction=False``).
        """
        if not term.parents:
            m = SimpleInterceptModule(n_params)
            m.groups, m.ci_parents = [], []
            return m
        groups = (
            [tuple(term.parents)]
            if term.allow_interaction
            else [(p,) for p in term.parents]
        )
        if len(groups) == 1:
            m = ComplexInterceptModule(
                feat_width(spec, groups[0]),
                n_params,
                units=term.units,
                activation=term.activation,
                batch_norm=term.batch_norm,
            )
        else:  # additive intercept: one net per parent, coefficients summed
            m = AdditiveInterceptModule(
                groups,
                n_params,
                spec,
                units=term.units,
                activation=term.activation,
                batch_norm=term.batch_norm,
            )
        m.groups = groups
        m.ci_parents = [p for grp in groups for p in grp]
        _attach_input_transform(m, term, tuple(term.parents), spec)
        return m

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
    n_params : int
        Number of transform parameters.
    """

    def __init__(self, n_params: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_params))

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

    Parameters
    ----------
    n_features : int
        Width of the encoded parent features.
    n_params : int
        Number of transform parameters to produce.
    units : tuple[int, ...]
        Hidden layers of the network. The term class holds the default: the
        reference's ``ComplexInterceptDefaultTabular`` widths (module
        docstring). The paper's own nets are wider; a replication sets this
        explicitly.
    activation : str
        Key of ``ACTIVATIONS``.
    batch_norm : bool
        Normalize the hidden layers — see ``_nn``.
    """

    def __init__(
        self,
        n_features: int,
        n_params: int,
        units: tuple[int, ...],
        activation: str,
        batch_norm: bool,
    ):
        super().__init__()
        self.net = _nn(
            n_features,
            units,
            n_params,
            activation=activation,
            batch_norm=batch_norm,
        )

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
    """

    def __init__(
        self,
        groups,
        n_params: int,
        spec,
        *,
        units: tuple[int, ...],
        activation: str,
        batch_norm: bool,
    ):
        super().__init__()
        # the submodule must stay named `nets`: it is part of the state-dict
        # path that tests/test_statedict_stability.py pins
        self.nets = nn.ModuleList(
            ComplexInterceptModule(
                feat_width(spec, grp), n_params, units, activation, batch_norm
            )
            for grp in groups
        )

    def theta_value(self, node: Node, feats: dict, n: int) -> Tensor:
        """Sum the per-parent nets in coefficient space."""
        return sum(
            net(node.net_input(feats, grp, "@I"))
            for net, grp in zip(self.nets, self.groups, strict=True)
        )


class LinearShiftModule(ShiftModule, nn.Module):
    """``LS`` — one raw-unit coefficient per feature of the single parent, no bias.

    For an ordinal child, ``exp(beta)`` is an odds ratio.

    Parameters
    ----------
    n_features : int
        Width of the encoded parent features.
    """

    data = LS
    scored = True

    def __init__(self, n_features: int):
        super().__init__()
        self.fc = nn.Linear(n_features, 1, bias=False)

    @property
    def weight(self) -> Tensor:
        """Tensor: the shift coefficients, shape ``(n_features,)``."""
        return self.fc.weight.squeeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Give the shift ``(n,)`` from the encoded features ``(n, n_features)``."""
        return self.fc(x).squeeze(-1)

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> LinearShiftModule:
        """One weight per feature of the single parent; keyed by its name."""
        m = cls(feat_width(spec, term.parents))
        m.key = term.parents[0]
        return m

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """Give the raw parent column times the weight — no input transform."""
        return self(torch.cat([feats[p] for p in self.parents], dim=1))

    def score_columns(self, node: Node, flow, feats: dict, dlds) -> dict:
        """One column per weight: the parent (continuous) or its one-hot levels.

        ``d l_i / d beta = (d l_i / d s_i) * x_i`` — analytic and exact.
        """
        (parent,) = self.parents  # an LS term has exactly one parent
        psi = (dlds.unsqueeze(1) * feats[parent]).cpu().numpy()
        if isinstance(flow.spec[parent], OrdinalNode):
            return {f"{parent}[{k}]": psi[:, k] for k in range(psi.shape[1])}
        return {self.key: psi[:, 0]}


class ComplexShiftModule(ShiftModule, nn.Module):
    """``CS`` — an additive network shift ``g(x)`` over its parents.

    Parameters
    ----------
    n_features : int
        Width of the encoded parent features.
    units : tuple[int, ...]
        Hidden layers of the network. The term class holds the default: the
        reference's ``ComplexShiftDefaultTabular`` widths (module docstring).
        The paper's own nets are narrower; a replication sets this explicitly.
    activation : str
        Key of ``ACTIVATIONS``.
    batch_norm : bool
        Normalize the hidden layers — see ``_nn``.
    """

    data = CS

    def __init__(
        self,
        n_features: int,
        units: tuple[int, ...],
        activation: str,
        batch_norm: bool,
    ):
        super().__init__()
        self.net = _nn(
            n_features,
            units,
            1,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Give the shift ``(n,)`` from the encoded features ``(n, n_features)``."""
        return self.net(x).squeeze(-1)

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> ComplexShiftModule:
        """One net over the concatenated parents; keyed 'a' or 'a+b'."""
        ps = tuple(term.parents)
        m = cls(
            feat_width(spec, ps),
            units=term.units,
            activation=term.activation,
            batch_norm=term.batch_norm,
        )
        m.key = "+".join(ps)  # the parent itself for a single-parent term
        _attach_input_transform(m, term, ps, spec)
        return m

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """Give the net over this term's (possibly input-transformed) features."""
        return self(node.net_input(feats, self.parents, self.key))


class VaryingCoefficientModule(ShiftModule, nn.Module):
    """``VC`` — ``beta(modifiers) * x_t`` with ``beta(x) = beta0 + b_theta(x)``.

    ``b_theta`` is deliberately small: one hidden layer by default. Its
    weights carry an L2 ``penalty`` in the fitting objective (see ``l2``;
    ``fit`` adds ``penalty * l2()`` on the total-NLL scale). ``beta0`` is not
    penalized.

    The output layer starts at zero, so ``beta(x)`` equals ``beta0`` exactly
    at construction. The head therefore learns only the deviation from a
    constant effect, which makes the arm difference an estimate instead of a
    by-product. The unpenalized reduced form ``CS(on, x...)`` reaches a
    correlation of only about 0.5 against the true effect function (issue
    #28). With ``n_features == 0`` there are no modifiers, there is no
    network, and the term is exactly ``LS(on)``. Only the treatment owns an
    edge.

    Parameters
    ----------
    n_features : int
        Width of the encoded modifier features. Use 0 for no modifiers.
    penalty : float
        L2 weight on ``b_theta``; the term class holds the default.
    units : tuple[int, ...]
        Hidden layers of ``b_theta``; the term class holds the default, one
        layer of 16. That is the head ``tests/test_vc_term.py`` recovers a
        known ``beta(x)`` with at corr ~ 0.99; this term has no counterpart in
        the reference implementations, so the size comes from that
        measurement.
    activation : str
        Key of ``ACTIVATIONS``.
    batch_norm : bool
        Normalize the hidden layers — see ``_nn``.

    Notes
    -----
    A constant can move freely between ``beta0`` and ``b_theta``, so the split
    is not identified by the likelihood alone. The penalty resolves it during
    training, because it shrinks ``b_theta`` toward the zero function. After
    training, ``recenter`` re-splits the two exactly: ``b_theta`` then sums to
    zero over the training data, the GAM convention that
    ``intercept_contributions`` also uses. Recentering is a reparameterization
    through the ``center`` buffer and leaves the modelled function unchanged.
    """

    data = VC
    scored = True
    order = 1

    def __init__(
        self,
        n_features: int,
        penalty: float,
        units: tuple[int, ...],
        activation: str,
        batch_norm: bool,
    ):
        super().__init__()
        self.penalty = float(penalty)
        self.beta0 = nn.Parameter(torch.zeros(()))
        self.register_buffer("center", torch.zeros(()))
        if n_features > 0:
            # zero-initialised output: beta(x) == beta0 at init
            self.net = _nn(
                n_features,
                units,
                1,
                activation=activation,
                batch_norm=batch_norm,
                zero_init_last=True,
            )
        else:
            self.net = None

    def beta(self, mod_feats: Tensor | None, n: int) -> Tensor:
        """Give the effect values ``beta(x)``, shape ``(n,)``.

        ``mod_feats`` is ``None`` if, and only if, the term has no modifiers;
        ``n`` is the batch size, used in that case.
        """
        if self.net is None:
            return (self.beta0 - self.center).expand(n)
        return self.beta0 + self.net(mod_feats).squeeze(-1) - self.center

    def forward(self, t: Tensor, mod_feats: Tensor | None) -> Tensor:
        """Give the shift ``beta(mod_feats) * t``, shape ``(n,)``.

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

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> VaryingCoefficientModule:
        """Build the effect head over the modifiers; keyed by the treatment name."""
        on, mods = term.parents[0], tuple(term.parents[1:])
        m = cls(
            feat_width(spec, mods),
            penalty=term.penalty,
            units=term.units,
            activation=term.activation,
            batch_norm=term.batch_norm,
        )
        m.key = on
        m.mods = mods
        m.on_is_ord = isinstance(spec[on], OrdinalNode)
        m.center_col = term.center
        _attach_input_transform(m, term, mods, spec)
        return m

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
        t = feats[self.key][:, -1:] if self.on_is_ord else feats[self.key]
        if self.center_col:
            t = t - feats[self.center_col].view(-1, 1)
        return t

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """``beta(modifiers) * regressor``, with the centered-term guard."""
        t = self.regressor(feats)
        mod_feat = node.net_input(feats, self.mods, self.key) if self.mods else None
        return self(t, mod_feat)

    def post_init(self) -> None:
        """Re-zero the head's output layer: ``beta(x) == beta0`` at start."""
        if self.net is not None:
            nn.init.zeros_(self.net[-1].weight)

    def regularizer(self) -> Tensor | None:
        """``penalty * ||b_theta weights||^2`` on the total-likelihood scale.

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
        """One column, keyed by the treatment: the ``beta0`` score.

        ``d s / d beta0`` is the term's own ``regressor``, so forward
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
        p1 = flow._binary_p1(flow.nodes[self.key], values, n).detach()
        return {self.center_col: p1}

    def extra_columns(self, flow) -> list[str]:
        """Give the treatment's parents (a treatment cannot be centered itself)."""
        return list(flow.nodes[self.key].parents) if self.center_col else []


class FnShiftModule(ShiftModule, nn.Module):
    """``Fn`` — a user-supplied shift function over the parent features.

    A plain function contributes a fixed (non-trained) offset; an
    ``nn.Module`` registers as a submodule and trains with the flow. Built
    by [`FnShift`][] (``Fn``).
    """

    data = Fn

    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # an nn.Module registers as a submodule here

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> FnShiftModule:
        """Wrap the callable; keyed like a CS ('a' or 'a+b')."""
        ps = tuple(term.parents)
        m = cls(term.fn)
        m.key = "+".join(ps)  # the parent itself for a single-parent term
        _attach_input_transform(m, term, ps, spec)
        return m

    def shift_value(self, node: Node, feats: dict) -> Tensor:
        """Run ``fn`` on the term's features; accept ``(n,)`` or ``(n, 1)``."""
        out = self.fn(node.net_input(feats, self.parents, self.key))
        return out.squeeze(-1) if out.dim() > 1 else out
