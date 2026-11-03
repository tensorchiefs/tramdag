"""The fitting paths: ``NodeFitMixin`` for one node, ``FitMixin`` for the flow.

The joint likelihood is a sum of per-node terms with disjoint parameters, so
every node can fit on its own. ``NodeFitMixin.fit`` is one minibatch Adam loop
over one node (validation, verbose printing and the callback hooks included);
``NodeFitMixin.fit_classical`` is the float64 full-batch L-BFGS exact-MLE
route for a simple intercept with ``LS`` terms. ``FitMixin`` runs them for
every node of the flow, in topological order or in processes;
``FitMixin.fit_jointly`` trains all nodes in one loop with one optimizer.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import math
import multiprocessing
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .callbacks import Callback
from .spec import VaryingCoefficient

if TYPE_CHECKING:
    from .flow import CausalFlowDAG
    from .nodes import Node


# %% global variables ------------------------------------------------------------------
__all__ = ["FitMixin", "NodeFitMixin"]

DECREMENT_TOL = 1e-9

_FORK_JOB: tuple | None = None


# %% private functions -----------------------------------------------------------------
def _check_learning_rate(learning_rate, optimizer) -> None:
    """Refuse ``learning_rate=`` together with ``optimizer=``."""
    if learning_rate is not None and optimizer is not None:
        raise ValueError("pass learning_rate= or optimizer=, not both")


def _make_optimizer(target, optimizer, learning_rate):
    """Give the fit's optimizer: the given one, the factory's, or Adam at 1e-2."""
    if callable(optimizer):
        return optimizer(target)
    if optimizer is not None:
        return optimizer
    lr = 1e-2 if learning_rate is None else learning_rate
    return torch.optim.Adam(target.parameters(), lr=lr)


def _split_validation(
    train_df: pd.DataFrame,
    validation_data: pd.DataFrame | None,
    validation_split: float | None,
):
    """Resolve fit's validation arguments (the Keras rules).

    ``validation_split`` takes the LAST fraction of ``train_df`` as
    validation without shuffling, exactly like Keras: deterministic, no
    hidden RNG. Side columns (a centered VC's propensities) are ordinary
    columns of the frame, so they split with it.
    """
    if validation_split is None:
        return train_df, validation_data
    if validation_data is not None:
        raise ValueError("pass validation_data OR validation_split, not both")
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            f"validation_split must be in (0, 1), got {validation_split!r}"
        )
    cut = round(len(train_df) * (1.0 - validation_split))
    if cut < 1 or cut >= len(train_df):
        raise ValueError(
            f"validation_split={validation_split} leaves no rows on one side "
            f"of the {len(train_df)}-row frame"
        )
    return train_df.iloc[:cut], train_df.iloc[cut:]


def _normalize_callbacks(cbs) -> list[Callback]:
    """Give ``callbacks=`` as a list of ``Callback``s.

    A [`Callback`][tramdag.callbacks.Callback] instance is trusted: the base
    class defines all three hooks. A bare callable is an ``on_epoch_end``
    hook, called as ``cb(node, epoch, optimizer)``.
    """
    if cbs is None:
        return []
    if isinstance(cbs, Callback) or callable(cbs):
        cbs = [cbs]
    out = []
    for cb in cbs:
        if isinstance(cb, Callback):
            out.append(cb)
            continue
        out.append(_FnCallback(cb))
    restoring = [type(cb).__name__ for cb in out if getattr(cb, "restore_best", False)]
    if len(restoring) > 1:
        raise ValueError(
            f"callbacks= has two callbacks that restore weights ({restoring}); "
            "they would overwrite each other at fit end, so keep one"
        )
    return out


def _learning_rates(opt) -> dict[int, float]:
    """Give the optimizer's current rates, keyed by group index."""
    return {i: float(g["lr"]) for i, g in enumerate(opt.param_groups)}


def _log_epoch(
    label: str,
    train: float,
    val: float | None,
    epoch: int,
    epochs: int,
    verbose: int,
    stopped: bool,
):
    """Print one ``verbose`` progress line on the Nth and the final epoch.

    ``val`` is None when this fit does not validate, so the line never shows
    a stale validation value of an earlier fit.
    """
    last = stopped or epoch == epochs
    if not verbose or (epoch % verbose and not last):
        return
    line = f"{label}: epoch {epoch}/{epochs}  train {train:.4f}"
    if val is not None:
        line += f"  val {val:.4f}"
    print(line)


def _penalized(nodes: list) -> list:
    """List the shift modules of the nodes that add a penalty to the loss."""
    return [
        m for nd in nodes for m in nd.shifts.values() if m.regularizer() is not None
    ]


def _fit_epoch(
    nodes: list,
    values: dict[str, Tensor],
    opt: torch.optim.Optimizer,
    batch_size: int,
    penalized: list,
    generator: torch.Generator | None,
) -> dict[str, float]:
    """One shuffled pass over the rows; give the epoch-mean train NLL per node.

    The loss of a step is the sum of the nodes' mean NLLs, so one node is
    ``Node.fit`` and all nodes are ``fit_jointly``. The mean runs over the
    rows stepped on, not over ``n``. A skipped trailing row would otherwise
    scale the epoch NLL down by 1/n. The penalty joins the loss but not the
    history.
    """
    column = values[nodes[0].name]
    n = len(column)
    acc = dict.fromkeys([nd.name for nd in nodes], 0.0)
    trained = 0
    perm = torch.randperm(n, generator=generator).to(column.device)
    for idx in perm.split(batch_size):
        if idx.numel() < 2:
            continue
        trained += int(idx.numel())
        batch = {k: v[idx] for k, v in values.items()}
        nlls = {nd.name: -nd.row_log_prob(batch).mean() for nd in nodes}
        loss = sum(nlls.values())
        for m in penalized:
            loss = loss + m.regularizer() / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        for k, v in nlls.items():
            acc[k] += float(v.detach()) * idx.numel()
    if trained == 0:
        raise ValueError(
            f"fit() trained on no row: {n} row(s) split at batch_size="
            f"{batch_size} leaves only single-row batches, and the loop skips "
            "those, because a one-row batch carries no gradient and batch_norm "
            "cannot normalize over it. Give at least two rows."
        )
    return {k: v / trained for k, v in acc.items()}


def _record(nodes: list, train: dict, val: dict | None) -> None:
    """Append one epoch's train and validation NLL to each node's history.

    ``history["val_epoch"]`` holds the train epoch of each validation entry,
    because an unvalidated fit would otherwise shift the later entries.
    """
    for nd in nodes:
        nd.history["train"].append(train[nd.name])
        if val is not None:
            nd.history.setdefault("val", []).append(val[nd.name])
            nd.history.setdefault("val_epoch", []).append(len(nd.history["train"]))


def _train_loop(
    target,
    label: str,
    nodes: list,
    values: dict[str, Tensor],
    validate: Callable[[], dict[str, float] | None],
    opt: torch.optim.Optimizer,
    cbs: list[Callback],
    *,
    epochs: int,
    batch_size: int,
    verbose: int,
    generator: torch.Generator | None,
) -> None:
    """Run the epoch loop of ``Node.fit`` (one node) and ``fit_jointly`` (all).

    ``target`` is what the callbacks hook: the node or the flow; ``label``
    names it in the progress line. ``validate`` gives the validation NLL per
    node, or None without validation. Each node's history gets its own
    entries.
    """
    penalized = _penalized(nodes)
    for nd in nodes:
        nd.history.pop("best_epoch", None)
    for cb in cbs:
        cb.on_fit_begin(target, opt)
    for epoch in range(1, epochs + 1):
        target.train()
        train = _fit_epoch(nodes, values, opt, batch_size, penalized, generator)
        target.eval()
        val = validate()
        _record(nodes, train, val)
        stops = [bool(cb.on_epoch_end(target, epoch, opt)) for cb in cbs]
        for nd in nodes:
            nd.history.setdefault("lr", []).append(_learning_rates(opt))
        _log_epoch(
            label,
            sum(train.values()),
            None if val is None else sum(val.values()),
            epoch,
            epochs,
            verbose,
            stopped=any(stops),
        )
        if any(stops):
            break
    for cb in cbs:
        cb.on_fit_end(target, opt)
    for nd in nodes:
        nd._recenter(values)
    target.eval()


def _newton(node: Node, values: dict[str, Tensor]) -> tuple[float, float, Tensor]:
    """Give the mean NLL, the Newton decrement and the Newton step of a node.

    The step inverts the exact Hessian with the absolute values of its
    eigenvalues, so it always points downhill, and it skips the eigenvalues at
    float64 noise level, so it does not move along a direction the data do not
    identify. The decrement is half the gradient times the step: the NLL
    decrease that the quadratic model of the NLL still expects.
    """
    params = list(node.parameters())
    loss = -node.row_log_prob(values).mean()
    grad = torch.cat(
        [g.flatten() for g in torch.autograd.grad(loss, params, create_graph=True)]
    )
    rows = [torch.autograd.grad(g, params, retain_graph=True) for g in grad]
    hess = torch.stack([torch.cat([h.flatten() for h in row]) for row in rows])
    evals, evecs = torch.linalg.eigh(hess)
    floor = evals.abs().max() * len(evals) * torch.finfo(evals.dtype).eps
    inv = torch.where(evals.abs() > floor, 1 / evals.abs(), 0)
    grad = grad.detach()
    step = evecs @ (inv * (evecs.T @ grad))
    return float(loss.detach()), float(grad @ step) / 2, step


def _newton_polish(
    node: Node, values: dict[str, Tensor], max_steps: int
) -> tuple[float, int, bool]:
    """Take Newton steps until the decrement is at most ``DECREMENT_TOL``.

    Each step halves its length up to 20 times until the NLL goes down. A step
    that cannot lower the NLL also ends the polish: the NLL is then at its
    float64 resolution, or at a kink of a piecewise smooth transform. Give the
    last decrement, the number of steps, and whether the polish ended on its
    own before ``max_steps``.
    """
    params = list(node.parameters())
    for steps in range(max_steps):
        nll, decrement, step = _newton(node, values)
        if decrement <= DECREMENT_TOL:
            return decrement, steps, True
        x = parameters_to_vector(params).detach()
        with torch.no_grad():
            for k in range(20):
                vector_to_parameters(x - 0.5**k * step, params)
                if float(-node.row_log_prob(values).mean()) < nll:
                    break
            else:
                vector_to_parameters(x, params)
                return decrement, steps, True
    return _newton(node, values)[1], max_steps, False


def _node_seed(seed: int, index: int) -> int:
    """Give node ``index`` its own shuffling seed, derived from the fit's ``seed``."""
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def _per_node(value, name: str):
    """Give a per-node argument: ``value(name)`` for a function, else ``value``."""
    return value(name) if callable(value) else value


def _fit_in_child(name: str):
    """Fit one node in a forked worker; give its state and history back.

    The worker reads the flow and the per-node fit arguments from
    ``_FORK_JOB``.

    The worker uses one thread, because a forked child that starts OpenMP
    threads can deadlock.
    """
    flow, train_df, jobs = _FORK_JOB
    torch.set_num_threads(1)
    nd = flow.nodes[name]
    nd.fit(train_df, **jobs[name])
    return nd.state_dict(), nd.history


def _pad_last(rows: dict[str, list]) -> list[dict]:
    """Align per-node lists by index; a shorter list repeats its last entry."""
    length = max(len(v) for v in rows.values())
    return [
        {k: v[min(i, len(v) - 1)] if v else math.nan for k, v in rows.items()}
        for i in range(length)
    ]


# %% private classes -------------------------------------------------------------------
class _FnCallback(Callback):
    """A bare callable in ``callbacks=``, adapted to an ``on_epoch_end`` hook."""

    def __init__(self, fn):
        self.fn = fn

    def on_epoch_end(self, node, epoch: int, optimizer):
        return self.fn(node, epoch, optimizer)


# %% public classes --------------------------------------------------------------------
class NodeFitMixin:
    """The two fitting paths of one node, mixed into [`Node`][tramdag.nodes.Node]."""

    def side_columns(self) -> list[str]:
        """Name the frame columns the node's terms need beyond the parents."""
        cols = [c for m in self.shifts.values() for c in m.side_columns()]
        return list(dict.fromkeys(cols))

    def _check_side_columns(self, train_df: pd.DataFrame) -> list[str]:
        r"""Check the terms' side columns in the frame; give their names.

        A centered ``VC`` needs its propensity column
        $P(t = 1 \mid \mathrm{pa}_t)$ per row, merged into the frame as an
        ordinary column. ``docs/varying-coefficients.md`` says how to compute
        the column out of fold.
        """
        for m in self.shifts.values():
            for col in m.side_columns():
                if col not in train_df.columns:
                    raise ValueError(
                        f"the centered VC on node {self.name!r} needs its "
                        f"propensity column {col!r} in the training "
                        "frame; compute P(t=1|pa_t) out of fold and "
                        "merge it as a column"
                    )
                m.check_column(self.name, col, train_df[col].to_numpy())
        return self.side_columns()

    def row_log_prob(self, values: dict[str, Tensor]) -> Tensor:
        """Give ``log p(x | pa)`` per row of already tensorized columns.

        ``values`` holds the node's column, its parents' columns and its side
        columns, each shape ``(n,)``.
        """
        n = len(values[self.name])
        feats = self.features(values) | {c: values[c] for c in self.side_columns()}
        theta, shift = self.theta_shift(feats, n)
        return self.log_prob(theta, shift, values[self.name])

    @torch.no_grad()
    def nll(self, df: pd.DataFrame) -> float:
        """Give the node's mean negative log-likelihood over the rows of ``df``.

        ``df`` holds the node's column, its parents' columns and its side
        columns (a centered VC's propensity).
        """
        values = self.tensorize(df, (self.name, *self.parents, *self.side_columns()))
        return float(-self.row_log_prob(values).mean())

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        epochs: int,
        learning_rate: float | None = None,
        batch_size: int = 512,
        validation_data: pd.DataFrame | None = None,
        validation_split: float | None = None,
        verbose: int = 0,
        seed: int | None = None,
        marginal_init: bool = False,
        optimizer: torch.optim.Optimizer
        | Callable[[Node], torch.optim.Optimizer]
        | None = None,
        callbacks=None,
    ) -> Node:
        """Fit the node by maximum likelihood with one minibatch loop, Adam by default.

        The loss is the node's mean NLL plus its own ``VC`` penalties. The
        loop keeps the **final** weights, and a second ``fit`` call continues
        the training. The loop computes the validation NLL and prints the
        progress; learning-rate schedules, early stopping and best-weight
        restoration are the caller's, through ``optimizer`` and
        ``callbacks``; [`callbacks`][tramdag.callbacks] ships ``EarlyStopping``.
        The ``VC`` penalty never enters ``history["train"]``, and a ``VC``
        term is re-centered after the loop.

        ``node.history`` accumulates across fits: ``"train"`` and ``"val"``
        hold one mean NLL per epoch, ``"val_epoch"`` the train epoch each
        validation entry belongs to, ``"lr"`` the optimizer's rates per epoch
        as ``{group index: lr}``. ``"best_epoch"`` is the best epoch that an
        ``EarlyStopping`` of the last fit found, counted like ``"val_epoch"``;
        a fit without one removes it.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training rows: the node's column, its parents' columns and its
            side columns. Other columns are ignored.
        epochs : int
            Number of passes over the data. Required; there is no default.
        learning_rate : float, optional
            Adam step size of the default optimizer. The default (None) is
            1e-2. Exclusive with ``optimizer``.
        batch_size : int, optional
            Rows per gradient step, by default 512. ``len(train_df)`` is one
            full-batch step per epoch.
        validation_data : pd.DataFrame | None, optional
            Validation rows, with the same columns as ``train_df``, by
            default None. When given (or split off), ``fit`` appends the
            validation NLL to ``node.history["val"]`` after every epoch;
            ``EarlyStopping`` reads it there. A centered ``VC`` reads its
            propensity column from this frame; the flow's ``fit`` puts the
            live propensity of the fitted treatment node there.
        validation_split : float | None, optional
            Keras' rule: the LAST fraction of ``train_df`` becomes the
            validation set, without shuffling, and only the remaining rows
            train and calibrate. Mutually exclusive with ``validation_data``.
        verbose : int, optional
            0 (default) is silent. ``N >= 1`` prints one line every ``N``
            epochs and on the final epoch: node, epoch counter, train NLL,
            validation NLL when validation is configured.
        seed : int | None, optional
            Seeds the minibatch shuffling with a private generator, by
            default None (torch's global RNG). ``Node(seed=)`` and
            ``CausalFlowDAG(seed=)`` seed the weight initialization. A flow
            fit derives each node's seed from ``(seed, node index)``, so this
            seed alone does not repeat a node of a seeded flow fit.
        marginal_init : bool, optional
            Start a simple intercept at its column's marginal, by default
            ``False``. Applied only by the fit that calibrates the node. A
            spline or affine transform has no start and skips it.
        optimizer : torch.optim.Optimizer | callable | None, optional
            A torch optimizer over ``node.parameters()``, or a factory
            ``f(node) -> Optimizer`` (for example to split the parameters into
            groups). The default (None) is ``Adam(lr=1e-2)``.
        callbacks : Callback | callable | list | None, optional
            One entry or a list, by default None. A
            [`Callback`][tramdag.callbacks.Callback]
            hooks all three points of the fit; its docstring is the contract.
            A bare callable is an ``on_epoch_end`` hook, ``cb(node, epoch,
            optimizer)``; a ``True`` return stops the fit. Two callbacks that
            restore weights are refused.

        Returns
        -------
        Node
            ``self``, fitted, in eval mode.

        Notes
        -----
        ``fit`` validates the frames before it calibrates, so a malformed frame
        cannot half-mutate the node. A batch of one row is skipped: batch norm
        needs two rows and one row gives no gradient.

        ``history["val_epoch"]`` holds the train epoch of each validation
        entry. ``history`` accumulates across ``fit`` calls, so an unvalidated
        fit would otherwise shift every later validation entry away from its
        own epoch.

        Every callback runs at each epoch end, so a stop does not skip a
        monitoring callback. The learning rate is read after the callbacks, so
        a scheduler's decision for this epoch shows. ``on_fit_end`` runs before
        the VC re-centering, so weights that a callback restores (such as
        ``EarlyStopping``) still take part in it.

        Raises
        ------
        ValueError
            If ``learning_rate`` and ``optimizer`` are both given, the
            validation arguments clash, a centered VC term's
            propensity column is missing or out of [0, 1], two callbacks
            restore weights, an ordinal value is not a level index, the
            node's calibration refuses the frame, or ``marginal_init`` meets
            a ``range_q=0`` Bernstein transform.
        KeyError
            If a frame lacks a column the node reads.
        """
        _check_learning_rate(learning_rate, optimizer)
        cbs = _normalize_callbacks(callbacks)
        train_df, validation_data = _split_validation(
            train_df, validation_data, validation_split
        )
        cols = (self.name, *self.parents, *self._check_side_columns(train_df))
        values = self.tensorize(train_df, cols)
        val_values = (
            self.tensorize(validation_data, cols)
            if validation_data is not None
            else None
        )
        self.calibrate(train_df, marginal_init=marginal_init)
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        opt = _make_optimizer(self, optimizer, learning_rate)

        def validate():
            if val_values is None:
                return None
            with torch.no_grad():
                return {self.name: float(-self.row_log_prob(val_values).mean())}

        _train_loop(
            self,
            self.name,
            [self],
            values,
            validate,
            opt,
            cbs,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            generator=generator,
        )
        return self

    def _recenter(self, values: dict[str, Tensor]) -> None:
        r"""Run every shift term's post-fit ``finalize`` (the VC re-centering).

        A VC term re-splits $\beta_0$ and $b_\Theta$ so the head sums to zero
        over the train rows; the modelled function does not change.
        """
        feats = self.features(values)
        for m in self.shifts.values():
            m.finalize(self, feats)

    def fit_classical(
        self,
        train_df: pd.DataFrame,
        *,
        max_iter: int = 5000,
        history_size: int = 50,
    ) -> dict:
        """Fit a simple intercept with ``LS`` terms the classical way.

        The fit uses full batches, float64, and L-BFGS with a strong-Wolfe
        line search, then Newton steps on the exact Hessian. There are no
        minibatches, no schedule and no early stopping, so the fit is
        deterministic; the report says whether it reached the
        maximum-likelihood estimate. Any other term raises.
        ``docs/fitting.md`` says why.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training rows: the node's column and its parents' columns.
        max_iter : int, optional
            Upper limit on L-BFGS iterations and Newton steps together, by
            default 5000.
        history_size : int, optional
            L-BFGS memory, by default 50.

        Returns
        -------
        dict
            A convergence report: ``converged``, ``stop_reason``, ``n_iter``,
            ``final_nll``, ``grad_norm``, ``decrement``, ``seconds``, and the
            fitted ``LS`` weights as ``coefficients``, ``{parent: array}``.

        Warns
        -----
        UserWarning
            If the report is not ``converged``, with its ``stop_reason`` and
            ``decrement``.

        Raises
        ------
        ValueError
            If a term is not classical: anything but the simple intercept and
            ``LS``, if an ordinal value is not a level index, or if the node's
            calibration refuses the frame.
        KeyError
            If the frame lacks a column the node reads.

        Notes
        -----
        L-BFGS stops when one iteration lowers the NLL by less than 1e-9. On an
        ill-conditioned NLL, for example a parent on a large raw scale next to a
        rare level, this happens short of the optimum. Newton steps on the exact
        Hessian then finish the fit. The coefficients are read while the node
        is float64. The cast back to float32 loses about 1e-7 relative.

        float64 is a transient compute mode. The node is upcast for the fit,
        and ``self.double()`` converts the parameters and the range buffers of
        the transform in one call. Afterwards the node returns to float32, so
        the stored model and ``save``/``load`` stay float32. Double precision
        is what lets the line search resolve the optimum cleanly.

        ``stop_reason`` is what ended the run: ``"tolerance"`` when L-BFGS and
        the Newton steps stopped on their own, ``"max_iter"`` when the
        iteration budget ran out, and ``"max_eval"`` when torch's evaluation
        budget (``max_iter * 5 // 4`` closure calls, line searches included)
        ran out first. ``n_iter`` counts the L-BFGS iterations and the Newton
        steps. ``tolerance_grad`` is 0, so the gradient never ends the run.
        Only a ``"tolerance"`` stop of L-BFGS gets the Newton steps.

        The Newton steps stop when the ``decrement`` is at most
        ``DECREMENT_TOL`` (1e-9), or when no step lowers the NLL. The decrement
        is the decrease of the mean NLL that a Newton step still expects. It
        does not depend on the scale of the parents, which the gradient norm
        does. A Bernstein NLL converges only linearly in its weakly curved
        tail coefficients, so this can take many steps. A spline NLL
        is only piecewise smooth: at a kink its Hessian is indefinite, no step
        lowers the NLL, and the decrement is above the bound.

        ``converged`` is ``stop_reason == "tolerance"``: a stall short of the
        optimum is not converged, because the Newton steps continue from it
        until they stop on their own or spend the budget.

        The Newton step skips the eigenvalues of the Hessian at float64 noise
        level. So it does not move along a direction with zero curvature,
        such as the one that the one-hot of an ordinal parent shares with the
        cutpoints. Compare the contrasts ``w[k] - w[0]`` with other software.
        """
        self._check_classical()
        self.calibrate(train_df)
        self.double()
        t0 = time.perf_counter()
        try:
            values = self.tensorize(train_df)
            self.train()
            opt = torch.optim.LBFGS(
                self.parameters(),
                lr=1.0,
                max_iter=max_iter,
                history_size=history_size,
                tolerance_grad=0.0,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                loss = -self.row_log_prob(values).mean()
                loss.backward()
                return loss

            opt.step(closure)
            state = next(iter(opt.state.values()))
            n_iter = state["n_iter"]
            if n_iter >= max_iter:
                stop_reason = "max_iter"
            elif state["func_evals"] >= opt.param_groups[0]["max_eval"]:
                stop_reason = "max_eval"
            else:
                stop_reason = "tolerance"
            if stop_reason == "tolerance":
                decrement, steps, settled = _newton_polish(
                    self, values, max_iter - n_iter
                )
                n_iter += steps
                if not settled:
                    stop_reason = "max_iter"
            else:
                decrement = _newton(self, values)[1]
            final_nll = float(closure().detach())
            grad_norm = float(
                torch.nn.utils.get_total_norm(
                    [p.grad for p in self.parameters() if p.grad is not None]
                )
            )
            converged = stop_reason == "tolerance"
            coefs = self.ls_coefficients()
        finally:
            self.float()
        self.eval()

        if not converged:
            warnings.warn(
                f"fit_classical did not converge on node {self.name!r}: "
                f"stop_reason={stop_reason!r} after {n_iter} iterations, "
                f"decrement={decrement:.2e} (bound {DECREMENT_TOL}). Raise max_iter "
                "or read the report.",
                UserWarning,
                stacklevel=2,
            )
        return {
            "converged": converged,
            "stop_reason": stop_reason,
            "n_iter": n_iter,
            "final_nll": final_nll,
            "grad_norm": grad_norm,
            "decrement": decrement,
            "seconds": time.perf_counter() - t0,
            "coefficients": coefs,
        }

    def _check_classical(self) -> None:
        """Refuse a term the classical fit cannot take."""
        other = sorted({t.name for t in self.node_spec.terms if not t.classical})
        if other:
            raise ValueError(
                "fit_classical requires an all-`ls` spec, that is a simple "
                f"intercept and LS terms only; node {self.name!r} has {other} "
                "terms. Use fit() for flexible models."
            )


class FitMixin:
    """The flow's fitting paths, mixed into [`CausalFlowDAG`][tramdag.CausalFlowDAG].

    `fit` and `fit_classical` loop over the nodes; `fit_jointly` trains them
    in one loop. Each node fits itself ([`Node.fit`][tramdag.nodes.Node.fit]).
    """

    @property
    def history(self) -> dict[str, list]:
        """The nodes' histories, one ``{node: value}`` dict per epoch per key.

        Nodes stop at different epochs. A node that stopped earlier repeats
        its last entry, so a summed curve stays defined. A node without
        entries for a key shows NaN there. ``"val_epoch"`` is the longest
        node's, which is exact while every fit validates. Each node's best
        epoch is in ``node.history["best_epoch"]``.
        """
        out = {}
        for key in ("train", "val", "lr"):
            rows = {name: nd.history.get(key, []) for name, nd in self.nodes.items()}
            if key == "train" or any(rows.values()):
                out[key] = _pad_last(rows)
        if "val" in out:
            out["val_epoch"] = max(
                (nd.history.get("val_epoch", []) for nd in self.nodes.values()),
                key=len,
            )
        return out

    def history_frame(self) -> pd.DataFrame:
        """Give the nodes' histories as one row per node and train epoch.

        Returns
        -------
        pd.DataFrame
            Columns ``node``, ``epoch`` (from 1), ``train``, ``val`` (NaN in an
            epoch without validation) and one ``lr[k]`` per optimizer group
            ``k`` (NaN in a node without that group), and the boolean
            ``best``, true in the epoch that ``EarlyStopping`` restored
            (``node.history["best_epoch"]``) and false everywhere for a node
            without one. The minimum of ``val`` is not the restored epoch
            when ``min_delta`` is above 0. A node that no ``fit`` trained has
            no rows, and a node that stopped early does not repeat its last
            entry.
        """
        frames = []
        for name, nd in self.nodes.items():
            h = nd.history
            epochs = range(1, len(h["train"]) + 1)
            val = dict(zip(h.get("val_epoch", []), h.get("val", []), strict=True))
            frame = pd.DataFrame(
                {
                    "node": name,
                    "epoch": epochs,
                    "train": h["train"],
                    "val": [val.get(e, math.nan) for e in epochs],
                    "best": [e == h.get("best_epoch") for e in epochs],
                }
            )
            lr = pd.DataFrame(h.get("lr", [])).rename(columns="lr[{}]".format)
            frames.append(pd.concat([frame, lr], axis=1))
        return pd.concat(frames, ignore_index=True)

    def warm_start(self, train_df: pd.DataFrame) -> dict[str, dict]:
        """Warm-start every node that has a ``VC`` term.

        Each such node runs [`Node.warm_start`][tramdag.nodes.Node.warm_start].
        Call it before ``fit``.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training rows with every node's column.

        Returns
        -------
        dict[str, dict]
            The reports of the nodes with a ``VC`` term, keyed by node name.
            The dict is empty when no node has one.
        """
        return {
            name: self.nodes[name].warm_start(train_df)
            for name, node_spec in self.spec.items()
            if any(isinstance(t, VaryingCoefficient) for t in node_spec.terms)
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        epochs: int | Callable[[str], int],
        learning_rate: float | Callable[[str], float] | None = None,
        batch_size: int = 512,
        validation_data: pd.DataFrame | None = None,
        validation_split: float | None = None,
        verbose: int = 0,
        seed: int | None = None,
        marginal_init: bool = False,
        optimizer: Callable[[Node], torch.optim.Optimizer] | None = None,
        callbacks=None,
        n_jobs: int = 1,
    ) -> CausalFlowDAG:
        """Fit every node by maximum likelihood, node by node.

        The joint likelihood is a sum of per-node terms with disjoint
        parameters, so fitting the nodes one by one is exact. Each node runs
        [`Node.fit`][tramdag.nodes.Node.fit], whose docstring describes the
        loop, the history and the callback contract. The validation split
        happens once here, so all nodes see the same rows. The nodes fit in
        topological order; with validation data, the nodes with a centered
        ``VC`` fit last, so their validation NLL uses the fitted treatment
        node.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node, plus the side columns.
        epochs : int | callable
            Passes over the data, or ``f(name) -> int`` per node. Required;
            there is no default.
        learning_rate : float | callable, optional
            Adam step size of the default optimizer, or ``f(name) -> float``.
            The default (None) is 1e-2. Exclusive with ``optimizer``.
        batch_size : int, optional
            Rows per gradient step, by default 512.
        validation_data : pd.DataFrame | None, optional
            Validation rows, with the same columns as ``train_df``, by
            default None.
        validation_split : float | None, optional
            Keras' rule: the LAST fraction of ``train_df`` becomes the
            validation set, without shuffling, and only the remaining rows
            train and calibrate. Shuffle the frame first if its row order
            means anything. Mutually exclusive with ``validation_data``.
        verbose : int, optional
            0 (default) is silent; ``N >= 1`` prints every ``N`` epochs.
        seed : int | None, optional
            Seeds the minibatch shuffling, by default None. Each node gets its
            own seed derived from ``(seed, node index)``, so a seeded result
            does not depend on whether the nodes run serially or in
            processes. Unseeded with ``n_jobs > 1``, a base seed is drawn
            from torch's global RNG; see ``n_jobs``.
        marginal_init : bool, optional
            Start every simple intercept at its column's marginal, by default
            ``False``. Applied only by the fit that calibrates a node. See
            [`Node.fit`][tramdag.nodes.Node.fit].
        optimizer : callable | None, optional
            A factory ``f(node) -> Optimizer``, called once per node fit, for
            example ``lambda node: torch.optim.AdamW(node.parameters())``. The
            default (None) is ``Adam(lr=1e-2)``.
        callbacks : Callback | list | callable | None, optional
            The callbacks of every node, or a function of the node name that
            gives them per node, by default None. A bare callable is that
            function, so a bare hook goes into a list. A shared shipped
            callback resets at every node's fit begin, so its attributes
            describe the last node only; a function of the node name keeps
            one instance per node.
        n_jobs : int, optional
            Number of processes, by default 1 (serial, topological order).
            ``n_jobs > 1`` forks ``n_jobs`` workers, one task per node (Linux
            only). A worker runs torch on one thread. With the same ``seed``,
            its result equals a serial fit on one thread; with more threads a
            network can differ in the last bits. Callback state stays in the workers;
            read ``node.history`` instead.

        Returns
        -------
        CausalFlowDAG
            ``self``, fitted, in eval mode.

        Notes
        -----
        Without a ``seed``, ``n_jobs > 1`` draws one from torch's RNG, because
        forked workers would otherwise replay one RNG state. A centered VC
        validates against the live propensity of its fitted treatment node, as
        every query does. Those nodes therefore fit second. The callbacks
        normalize once here, so an iterator of callbacks survives a single pass.

        Raises
        ------
        TypeError
            If ``optimizer`` is an optimizer instance instead of a factory.
        KeyError
            If a frame lacks a node column.
        ValueError
            See [`Node.fit`][tramdag.nodes.Node.fit]. Every frame and every
            node's callbacks are checked before the first node fits. Also if
            ``learning_rate`` and ``optimizer`` are both given.
        """
        if isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer= of CausalFlowDAG.fit is a factory f(node) -> "
                "Optimizer, called once per node; got an optimizer instance"
            )
        _check_learning_rate(learning_rate, optimizer)
        train_df, validation_data = _split_validation(
            train_df, validation_data, validation_split
        )
        self._check_frames(train_df, validation_data)
        if seed is None and n_jobs > 1:
            seed = int(torch.randint(2**62, ()))
        if not callable(callbacks):
            callbacks = _normalize_callbacks(callbacks)
        jobs = {
            name: dict(
                epochs=_per_node(epochs, name),
                learning_rate=_per_node(learning_rate, name),
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=verbose,
                seed=None if seed is None else _node_seed(seed, i),
                marginal_init=marginal_init,
                optimizer=optimizer,
                callbacks=_per_node(callbacks, name),
            )
            for i, name in enumerate(self.order)
        }
        for kwargs in jobs.values():
            kwargs["callbacks"] = _normalize_callbacks(kwargs["callbacks"])
        self._calibrate(train_df, marginal_init=marginal_init)
        later = [n for n in self.order if self.nodes[n].side_columns()]
        if validation_data is None:
            later = []
        self._fit_nodes(train_df, {n: jobs[n] for n in jobs if n not in later}, n_jobs)
        for name in later:
            jobs[name]["validation_data"] = self._with_live_side(name, validation_data)
        self._fit_nodes(train_df, {n: jobs[n] for n in later}, n_jobs)
        return self.eval()

    def _check_frames(
        self, train_df: pd.DataFrame, validation_data: pd.DataFrame | None
    ) -> None:
        """Check every frame for every node, so a bad one fails before any fit."""
        self._tensorize(train_df)
        if validation_data is not None:
            self._tensorize(validation_data)
        for nd in self.nodes.values():
            nd._check_side_columns(train_df)

    def _fit_nodes(self, train_df: pd.DataFrame, jobs: dict, n_jobs: int) -> None:
        """Run ``Node.fit`` for each job, serially or in forked workers.

        The parent calibrates every node before the fork, so the shapes of the
        returned states match.
        """
        global _FORK_JOB
        if n_jobs == 1 or len(jobs) < 2:
            for name, kwargs in jobs.items():
                self.nodes[name].fit(train_df, **kwargs)
            return
        _FORK_JOB = (self, train_df, jobs)
        try:
            with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
                results = pool.map(_fit_in_child, list(jobs))
        finally:
            _FORK_JOB = None
        for name, (state, history) in zip(jobs, results, strict=True):
            self.nodes[name].load_state_dict(state)
            self.nodes[name].history = history
            self.nodes[name].eval()

    def _with_live_side(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Set the node's side columns in ``df`` to their live values."""
        nd = self.nodes[name]
        cols = dict.fromkeys([*nd.parents, *self._query_side_columns(nd)])
        live = self._side_feats(nd, self._tensorize(df, list(cols)), len(df))
        return df.assign(**{c: live[c].cpu().numpy() for c in nd.side_columns()})

    def fit_jointly(
        self,
        train_df: pd.DataFrame,
        *,
        epochs: int,
        learning_rate: float | None = None,
        batch_size: int = 512,
        validation_data: pd.DataFrame | None = None,
        validation_split: float | None = None,
        verbose: int = 0,
        seed: int | None = None,
        marginal_init: bool = False,
        optimizer: torch.optim.Optimizer
        | Callable[[CausalFlowDAG], torch.optim.Optimizer]
        | None = None,
        callbacks=None,
    ) -> CausalFlowDAG:
        """Fit all nodes together: one optimizer, one minibatch loop.

        Each step draws one batch of rows and minimizes the sum of the nodes'
        mean NLLs. The optimum is that of [`fit`][tramdag.flow.CausalFlowDAG.fit];
        the training dynamics differ. One learning-rate schedule and one stop
        rule act on the summed NLL, which is the protocol of the reference
        implementation. Each node's history gets its own train and validation
        NLL per epoch, so ``flow.history``, ``history_frame`` and
        ``plot_training`` read a joint fit as they read a per-node one.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node, plus the side columns.
        epochs : int
            Passes over the data. Required; there is no default.
        learning_rate : float, optional
            Adam step size of the default optimizer. The default (None) is
            1e-2. Exclusive with ``optimizer``.
        batch_size : int, optional
            Rows per gradient step, by default 512.
        validation_data : pd.DataFrame | None, optional
            Validation rows, one column per node, by default None. A centered
            ``VC`` validates against the live propensity of its treatment
            node.
        validation_split : float | None, optional
            Keras' rule, as in ``fit``. Mutually exclusive with
            ``validation_data``.
        verbose : int, optional
            0 (default) is silent; ``N >= 1`` prints the summed NLLs every
            ``N`` epochs and on the final epoch.
        seed : int | None, optional
            Seeds the minibatch shuffling, by default None.
        marginal_init : bool, optional
            Start every simple intercept at its column's marginal, by default
            ``False``. Applied only by the fit that calibrates a node. See
            [`Node.fit`][tramdag.nodes.Node.fit].
        optimizer : torch.optim.Optimizer | callable | None, optional
            A torch optimizer over ``flow.parameters()``, or a factory
            ``f(flow) -> Optimizer``. The default (None) is
            ``Adam(lr=1e-2)``.
        callbacks : Callback | callable | list | None, optional
            As in [`Node.fit`][tramdag.nodes.Node.fit], but every hook gets
            the flow instead of a node. ``EarlyStopping`` then tracks the
            summed validation NLL and restores the whole flow.

        Returns
        -------
        CausalFlowDAG
            ``self``, fitted, in eval mode.

        Raises
        ------
        ValueError
            See [`Node.fit`][tramdag.nodes.Node.fit].
        KeyError
            If a frame lacks a node column.
        """
        _check_learning_rate(learning_rate, optimizer)
        cbs = _normalize_callbacks(callbacks)
        train_df, validation_data = _split_validation(
            train_df, validation_data, validation_split
        )
        self._check_frames(train_df, validation_data)
        self._calibrate(train_df, marginal_init=marginal_init)
        nodes = [self.nodes[name] for name in self.order]
        side = [c for nd in nodes for c in nd.side_columns()]
        values = self._tensorize(train_df, [*self.order, *dict.fromkeys(side)])
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        opt = _make_optimizer(self, optimizer, learning_rate)
        _train_loop(
            self,
            "flow",
            nodes,
            values,
            lambda: None if validation_data is None else self.nll(validation_data),
            opt,
            cbs,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            generator=generator,
        )
        return self

    def fit_classical(
        self,
        train_df: pd.DataFrame,
        *,
        max_iter: int = 5000,
        history_size: int = 50,
    ) -> dict:
        """Fit an all-``ls`` model the classical way, node by node.

        Each node runs [`Node.fit_classical`][tramdag.nodes.Node.fit_classical]:
        float64, full batch, L-BFGS and Newton. The joint objective is a sum of per-node
        terms with disjoint parameters, so the per-node optima are the joint
        one. Any term other than a simple intercept or ``LS`` raises before a
        node is fitted.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node.
        max_iter : int, optional
            Upper limit on L-BFGS iterations and Newton steps per node, by
            default 5000.
        history_size : int, optional
            L-BFGS memory, by default 50.

        Returns
        -------
        dict
            ``converged`` (every node converged), ``final_nll`` (the sum of
            the per-node mean NLLs), ``seconds``, the fitted ``coefficients``
            as ``{node: {parent: array}}`` (a node without ``LS`` terms is
            absent), and ``nodes``, the per-node reports.

        Notes
        -----
        The report holds the float64 values of the fit. The flow keeps float32
        parameters, so ``ls_coefficients``, ``pmf`` and ``log_prob`` use these
        values rounded to float32, about 1e-7 relative. Compare the report with
        other software.

        Warns
        -----
        UserWarning
            For each node that did not converge.

        Raises
        ------
        ValueError
            If a term is not classical, an ordinal value is not a level index,
            or a node's calibration refuses the frame.
        KeyError
            If the frame lacks a node column.
        """
        for nd in self.nodes.values():
            nd._check_classical()
        self._tensorize(train_df)
        self._calibrate(train_df)
        reports = {
            name: self.nodes[name].fit_classical(
                train_df, max_iter=max_iter, history_size=history_size
            )
            for name in self.order
        }
        return {
            "converged": all(r["converged"] for r in reports.values()),
            "final_nll": sum(r["final_nll"] for r in reports.values()),
            "seconds": sum(r["seconds"] for r in reports.values()),
            "coefficients": {
                n: r["coefficients"] for n, r in reports.items() if r["coefficients"]
            },
            "nodes": reports,
        }
