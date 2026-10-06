"""The fitting paths: ``NodeFitMixin`` for one node, ``FitMixin`` for the flow.

The joint likelihood is a sum of per-node terms with disjoint parameters, so
every node fits on its own. ``NodeFitMixin.fit`` is one minibatch Adam loop
over one node (validation, verbose printing and the callback hooks included);
``NodeFitMixin.fit_classical`` is the float64 full-batch L-BFGS exact-MLE
route for a simple intercept with ``LS`` terms. ``FitMixin`` runs them for
every node of the flow, in topological order or in processes.
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

from .callbacks import Callback

if TYPE_CHECKING:
    from .flow import CausalFlowDAG
    from .nodes import Node


# %% global variables ------------------------------------------------------------------
__all__ = ["FitMixin", "NodeFitMixin"]

# gradient norm at which a self-stopped classical fit counts as converged
GRAD_TOL = 1e-2

# the flow and the per-node fit arguments a forked worker reads (n_jobs > 1)
_FORK_JOB: tuple | None = None


# %% private functions -----------------------------------------------------------------
def _check_fit_sizes(epochs: int, batch_size: int, verbose: int) -> None:
    """Reject a non-positive epoch, batch or verbose value before anything runs."""
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if verbose < 0:
        raise ValueError(f"verbose must be a non-negative int, got {verbose!r}")


def _split_validation(
    train_df: pd.DataFrame,
    validation_data: pd.DataFrame | None,
    validation_split: float | None,
):
    """Resolve fit's validation arguments (the Keras rules).

    ``validation_split`` takes the LAST fraction of ``train_df`` as
    validation without shuffling, exactly like Keras — deterministic, no
    hidden RNG. Side columns (a centered VC's propensities) are ordinary
    columns of the frame, so they split with it.
    """
    if validation_split is None:
        return train_df, validation_data
    if validation_data is not None:
        raise ValueError("pass validation_data OR validation_split, not both")
    if not 0.0 < validation_split < 1.0:
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")
    cut = round(len(train_df) * (1.0 - validation_split))
    if cut < 1 or cut >= len(train_df):
        raise ValueError(
            f"validation_split={validation_split} leaves no rows on one side "
            f"of the {len(train_df)}-row frame"
        )
    return train_df.iloc[:cut], train_df.iloc[cut:]


def _normalize_callbacks(cbs) -> list[Callback]:
    """Give ``callbacks=`` as a list of ``Callback``s, or fail loudly now.

    A [`Callback`][tramdag.callbacks.Callback] instance is trusted — the base
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
        if isinstance(cb, type) and issubclass(cb, Callback):
            raise TypeError(
                f"callbacks= got the class {cb.__name__} — instantiate it: "
                f"{cb.__name__}()"
            )
        if not callable(cb):
            raise TypeError(
                f"callbacks entries must be Callback instances or callables, got {cb!r}"
            )
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
    node, epoch: int, epochs: int, verbose: int, stopped: bool, has_val: bool
):
    """Print one ``verbose`` progress line on the Nth and the final epoch."""
    last = stopped or epoch == epochs
    if not verbose or (epoch % verbose and not last):
        return
    line = f"{node.name}: epoch {epoch}/{epochs}"
    line += f"  train {node.history['train'][-1]:.4f}"
    if has_val:  # THIS fit's validation, not a stale earlier one
        line += f"  val {node.history['val'][-1]:.4f}"
    print(line)


def _fit_epoch(
    node,
    values: dict[str, Tensor],
    opt: torch.optim.Optimizer,
    batch_size: int,
    penalized: list,
    generator: torch.Generator | None,
) -> float:
    """One shuffled pass over the rows; give the epoch-mean train NLL."""
    n = len(values[node.name])
    acc = 0.0
    trained = 0
    perm = torch.randperm(n, generator=generator).to(values[node.name].device)
    for idx in perm.split(batch_size):
        if idx.numel() < 2:
            continue  # batch norm needs two rows, and one row is no gradient
        trained += int(idx.numel())
        batch = {k: v[idx] for k, v in values.items()}
        nll = -node.row_log_prob(batch).mean()
        loss = nll
        for m in penalized:  # the penalty joins the loss, not the history
            loss = loss + m.regularizer() / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc += float(nll.detach()) * idx.numel()
    if trained == 0:
        raise ValueError(
            f"fit() trained on no row: {n} row(s) split at batch_size="
            f"{batch_size} leaves only single-row batches, and the loop skips "
            "those, because a one-row batch carries no gradient and batch_norm "
            "cannot normalize over it. Give at least two rows."
        )
    # over the rows actually stepped on, not over n: a skipped trailing row
    # would otherwise scale the epoch NLL down by 1/n
    return acc / trained


def _node_seed(seed: int, index: int) -> int:
    """Give node ``index`` its own shuffling seed, derived from the fit's ``seed``."""
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def _per_node(value, name: str):
    """Give a per-node argument: ``value(name)`` for a function, else ``value``.

    A class is not a factory: ``callbacks=EarlyStopping`` reaches
    ``_normalize_callbacks``, which names the missing parentheses.
    """
    return value(name) if callable(value) and not isinstance(value, type) else value


def _fit_in_child(name: str):
    """Fit one node in a forked worker; give its state and history back."""
    flow, train_df, jobs = _FORK_JOB
    # one thread: a forked child that starts OpenMP threads can deadlock
    torch.set_num_threads(1)
    nd = flow.nodes[name]
    nd.fit(train_df, **jobs[name])
    return nd.state_dict(), nd.history


def _pad_last(rows: dict[str, list]) -> list[dict]:
    """Align per-node lists by index; a shorter list repeats its last entry."""
    length = max((len(v) for v in rows.values()), default=0)
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
                        "frame — compute P(t=1|pa_t) out of fold and "
                        "merge it as a column."
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
        learning_rate: float = 1e-2,
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
        """Fit the node by maximum likelihood with one minibatch Adam loop.

        The loss is the node's mean NLL plus its own ``VC`` penalties. The
        loop keeps the **final** weights, and a second ``fit`` call continues
        the training. The loop computes the validation NLL and prints the
        progress; learning-rate schedules, early stopping and best-weight
        restoration are the caller's, through ``optimizer`` and
        ``callbacks``; [`callbacks`][tramdag.callbacks] ships the common
        recipes. A ``VC`` term adds its penalty to the loss,
        never to ``history["train"]``, and is re-centered after the loop.

        ``node.history`` accumulates across fits: ``"train"`` and ``"val"``
        hold one mean NLL per epoch, ``"val_epoch"`` the train epoch each
        validation entry belongs to, ``"lr"`` the optimizer's rates per epoch
        as ``{group index: lr}``.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training rows: the node's column, its parents' columns and its
            side columns. Other columns are ignored.
        epochs : int
            Number of passes over the data. Required; there is no default.
        learning_rate : float, optional
            Adam step size of the default optimizer, by default 1e-2.
            Ignored when ``optimizer`` is given.
        batch_size : int, optional
            Rows per gradient step, by default 512. ``len(train_df)`` is one
            full-batch step per epoch.
        validation_data : pd.DataFrame | None, optional
            Validation rows, with the same columns as ``train_df``. When given
            (or split off), the validation NLL is appended to
            ``node.history["val"]`` after every epoch; ``EarlyStopping``
            reads it there.
        validation_split : float | None, optional
            Keras' rule: the LAST fraction of ``train_df`` becomes the
            validation set, without shuffling, and only the remaining rows
            train and calibrate. Mutually exclusive with ``validation_data``.
        verbose : int, optional
            0 (default) is silent. ``N >= 1`` prints one line every ``N``
            epochs and on the final epoch: node, epoch counter, train NLL,
            validation NLL when validation is configured.
        seed : int | None, optional
            Seeds the minibatch shuffling with a private generator. Weight
            initialization draws from torch's global RNG at construction;
            ``CausalFlowDAG(seed=)`` seeds it for a flow.
        marginal_init : bool, optional
            Start a simple intercept at its column's marginal, by default
            ``False``. Applied only by the fit that calibrates the node.
        optimizer : torch.optim.Optimizer | callable | None, optional
            A torch optimizer over ``node.parameters()``, or a factory
            ``f(node) -> Optimizer`` (for example to split the parameters into
            groups). The default is ``Adam(lr=learning_rate)``.
        callbacks : Callback | callable | list | None, optional
            One entry or a list. A [`Callback`][tramdag.callbacks.Callback]
            hooks all three points of the fit; its docstring is the contract.
            A bare callable is an ``on_epoch_end`` hook, ``cb(node, epoch,
            optimizer)``. Two callbacks that restore weights are refused.

        Returns
        -------
        Node
            ``self``, fitted, in eval mode.

        Raises
        ------
        ValueError
            If ``epochs`` or ``batch_size`` is below 1, ``verbose`` is
            negative, both validation arguments are given, the split leaves
            an empty side, a centered VC term's propensity column is missing
            from the training frame or out of [0, 1], or two callbacks
            restore weights, or an ordinal value is not a level index.
        KeyError
            If a frame lacks a column the node reads.
        TypeError
            If a ``callbacks`` entry is neither a ``Callback`` nor a callable,
            or is a ``Callback`` class instead of an instance.
        """
        _check_fit_sizes(epochs, batch_size, verbose)
        cbs = _normalize_callbacks(callbacks)
        train_df, validation_data = _split_validation(
            train_df, validation_data, validation_split
        )
        # validate BEFORE calibrate: a malformed frame must not half-mutate the node
        cols = (self.name, *self.parents, *self._check_side_columns(train_df))
        values = self.tensorize(train_df, cols)
        val_values = (
            self.tensorize(validation_data, cols)
            if validation_data is not None
            else None
        )
        self.calibrate(train_df, marginal_init=marginal_init)
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        if callable(optimizer):
            optimizer = optimizer(self)
        opt = optimizer or torch.optim.Adam(self.parameters(), lr=learning_rate)
        penalized = [m for m in self.shifts.values() if m.regularizer() is not None]
        for cb in cbs:
            cb.on_fit_begin(self, opt)
        for epoch in range(1, epochs + 1):
            self.train()
            nll = _fit_epoch(self, values, opt, batch_size, penalized, generator)
            self.history["train"].append(nll)
            self.eval()
            if val_values is not None:
                with torch.no_grad():
                    val = float(-self.row_log_prob(val_values).mean())
                self.history.setdefault("val", []).append(val)
                # which train epoch this entry belongs to: `history` accumulates
                # across `fit` calls, so an unvalidated fit would shift every
                # later validation entry away from its own epoch
                n_train_epochs = len(self.history["train"])
                self.history.setdefault("val_epoch", []).append(n_train_epochs)
            # every callback runs (a stop must not skip a monitoring one)
            stops = [bool(cb.on_epoch_end(self, epoch, opt)) for cb in cbs]
            # after the callbacks, so a scheduler's decision for this epoch shows
            self.history.setdefault("lr", []).append(_learning_rates(opt))
            _log_epoch(
                self,
                epoch,
                epochs,
                verbose,
                stopped=any(stops),
                has_val=val_values is not None,
            )
            if any(stops):
                break
        for cb in cbs:
            # before the VC re-centering, so weights a callback restores
            # (EarlyStopping) still take part in it
            cb.on_fit_end(self, opt)
        self._recenter(values)
        self.eval()
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
        line search. There are no minibatches, no schedule and no early
        stopping, so the fit is deterministic; the report says whether it
        reached the maximum-likelihood estimate. Any other term raises.
        ``docs/fitting.md`` says why.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training rows: the node's column and its parents' columns.
        max_iter : int, optional
            Upper limit on L-BFGS iterations, by default 5000.
        history_size : int, optional
            L-BFGS memory, by default 50.

        Returns
        -------
        dict
            A convergence report: ``converged``, ``stop_reason``, ``n_iter``,
            ``final_nll``, ``grad_norm``, ``seconds``, and the fitted ``LS``
            weights as ``coefficients``, ``{parent: array}``.

        Warns
        -----
        UserWarning
            If the report is not ``converged``, with its ``stop_reason`` and
            ``grad_norm``.

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
        float64 is a transient compute mode. The node is upcast for the fit,
        and ``self.double()`` converts the parameters and the range buffers of
        the transform in one call. Afterwards the node returns to float32, so
        the stored model and ``save``/``load`` stay float32. Double precision
        is what lets the line search resolve the optimum cleanly.

        ``stop_reason`` is what ended the run: ``"tolerance"`` when L-BFGS
        stopped on its own, because the NLL or the parameters moved by less
        than 1e-9, ``"max_iter"`` when the iteration budget ran out, and
        ``"max_eval"`` when torch's evaluation budget (``max_iter * 5 // 4``
        closure calls, line searches included) ran out first.
        ``tolerance_grad`` is 0, so the gradient never ends the run.

        ``converged`` needs BOTH: the run stopped on its own AND the gradient
        norm is at most ``GRAD_TOL`` (1e-2). The objective is the mean NLL, so
        its gradient does not scale with the number of rows and an absolute
        bound is meaningful.
        """
        self._check_classical()
        self.calibrate(train_df)
        self.double()  # parameters + buffers (xmin/xmax) -> float64, one call
        t0 = time.perf_counter()
        try:
            values = self.tensorize(train_df)
            self.train()
            opt = torch.optim.LBFGS(
                self.parameters(),
                lr=1.0,
                max_iter=max_iter,
                history_size=history_size,
                tolerance_grad=0.0,  # |grad| never settles on the flat ridges
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
            with torch.no_grad():
                final_nll = float(-self.row_log_prob(values).mean())
            grad_norm = float(
                torch.nn.utils.get_total_norm(
                    [p.grad for p in self.parameters() if p.grad is not None]
                )
            )
            # a stalled line search stops on the same tolerance as an arrival,
            # so the gradient is what tells the two apart
            converged = stop_reason == "tolerance" and grad_norm <= GRAD_TOL
            coefs = self.ls_weights()  # read while still float64
        finally:
            self.float()  # restore canonical float32 (lossy ~1e-7, harmless)
        self.eval()

        if not converged:
            warnings.warn(
                f"fit_classical did not converge on node {self.name!r}: "
                f"stop_reason={stop_reason!r} after {n_iter} iterations, "
                f"grad_norm={grad_norm:.2e} (bound {GRAD_TOL}). Raise max_iter "
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

    Both loop over the nodes; each node fits itself
    ([`Node.fit`][tramdag.nodes.Node.fit]).
    """

    @property
    def history(self) -> dict[str, list]:
        """The nodes' histories, one ``{node: value}`` dict per epoch per key.

        Nodes stop at different epochs. A node that stopped earlier repeats
        its last entry, so a summed curve stays defined. ``"val_epoch"`` is
        the longest node's.
        """
        out = {}
        for key in ("train", "val", "lr"):
            rows = {name: nd.history.get(key, []) for name, nd in self.nodes.items()}
            if key == "train" or any(rows.values()):
                out[key] = _pad_last(rows)
        if "val" in out:
            # ponytail: one node's val epochs; exact while every fit validates
            out["val_epoch"] = max(
                (nd.history.get("val_epoch", []) for nd in self.nodes.values()),
                key=len,
            )
        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        epochs: int | Callable[[str], int],
        learning_rate: float | Callable[[str], float] = 1e-2,
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
        """Fit every node by maximum likelihood, one node at a time.

        The joint likelihood is a sum of per-node terms with disjoint
        parameters, so fitting the nodes one by one is exact. Each node runs
        [`Node.fit`][tramdag.nodes.Node.fit], whose docstring describes the
        loop, the history and the callback contract. The validation split
        happens once here, so all nodes see the same rows.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node, plus the side columns.
        epochs : int | callable
            Passes over the data, or ``f(name) -> int`` per node.
        learning_rate : float | callable, optional
            Adam step size of the default optimizer, or ``f(name) -> float``,
            by default 1e-2. Ignored when ``optimizer`` is given.
        batch_size : int, optional
            Rows per gradient step, by default 512.
        validation_data : pd.DataFrame | None, optional
            Validation rows, with the same columns as ``train_df``.
        validation_split : float | None, optional
            Keras' rule: the LAST fraction of ``train_df`` becomes the
            validation set. Mutually exclusive with ``validation_data``.
        verbose : int, optional
            0 (default) is silent; ``N >= 1`` prints every ``N`` epochs.
        seed : int | None, optional
            Seeds the minibatch shuffling. Each node gets its own seed derived
            from ``(seed, node index)``, so the result does not depend on
            whether the nodes run serially or in processes.
        marginal_init : bool, optional
            Start every simple intercept at its column's marginal, by default
            ``False``. Applied only by the fit that calibrates a node.
        optimizer : callable | None, optional
            A factory ``f(node) -> Optimizer``, called once per node fit, for
            example ``lambda node: torch.optim.AdamW(node.parameters())``. The
            default is ``Adam(lr=learning_rate)``.
        callbacks : Callback | list | callable | None, optional
            The callbacks of every node, or a function of the node name that
            gives them per node. A shared shipped callback resets at every node's fit
            begin, so its attributes describe the last node only.
        n_jobs : int, optional
            Number of processes, by default 1 (serial, topological order).
            ``n_jobs > 1`` forks ``n_jobs`` workers, one task per node (Linux
            and macOS only). A worker runs torch on one thread, so its result
            equals a serial fit on one thread; with more threads a network
            can differ in the last bits. Callback state stays in the workers.

        Returns
        -------
        CausalFlowDAG
            ``self``, fitted, in eval mode.

        Raises
        ------
        TypeError
            If ``optimizer`` is an optimizer instance instead of a factory,
            or a node's ``callbacks`` entry is not a callback.
        KeyError
            If a frame lacks a node column.
        ValueError
            See [`Node.fit`][tramdag.nodes.Node.fit]. Every frame and every
            node's arguments are checked before the first node fits.
        """
        if isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer= of CausalFlowDAG.fit is a factory f(node) -> "
                "Optimizer, called once per node; got an optimizer instance"
            )
        train_df, validation_data = _split_validation(
            train_df, validation_data, validation_split
        )
        self._check_frames(train_df, validation_data)
        if not callable(callbacks) or isinstance(callbacks, type):
            callbacks = _normalize_callbacks(callbacks)  # one pass over an iterator
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
            _check_fit_sizes(kwargs["epochs"], batch_size, verbose)
            kwargs["callbacks"] = _normalize_callbacks(kwargs["callbacks"])
        self._calibrate(train_df, marginal_init=marginal_init)
        # a validation frame without a centered VC's propensity column gets the
        # live one from the fitted treatment node, so those nodes fit second
        later = [
            name
            for name in self.order
            if validation_data is not None
            and any(c not in validation_data for c in self.nodes[name].side_columns())
        ]
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
        """Run ``Node.fit`` for each job, serially or in forked workers."""
        global _FORK_JOB
        from .nodes import load_weights  # lazy: nodes imports this module

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
            load_weights(self.nodes[name], state)
            self.nodes[name].history = history
            self.nodes[name].eval()

    def _with_live_side(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Add the node's missing side columns to ``df``, computed live."""
        nd = self.nodes[name]
        cols = dict.fromkeys([*nd.parents, *self._query_side_columns(nd)])
        live = self._side_feats(nd, self._tensorize(df, list(cols)), len(df))
        missing = [c for c in nd.side_columns() if c not in df]
        return df.assign(**{c: live[c].cpu().numpy() for c in missing})

    def fit_classical(
        self,
        train_df: pd.DataFrame,
        *,
        max_iter: int = 5000,
        history_size: int = 50,
    ) -> dict:
        """Fit an all-``ls`` model the classical way, node by node.

        Each node runs [`Node.fit_classical`][tramdag.nodes.Node.fit_classical]:
        float64, full batch, L-BFGS. The joint objective is a sum of per-node
        terms with disjoint parameters, so the per-node optima are the joint
        one. Any term other than a simple intercept or ``LS`` raises before a
        node is fitted.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node.
        max_iter : int, optional
            Upper limit on L-BFGS iterations per node, by default 5000.
        history_size : int, optional
            L-BFGS memory, by default 50.

        Returns
        -------
        dict
            ``converged`` (every node converged), ``final_nll`` (the sum of
            the per-node mean NLLs), ``seconds``, the fitted ``coefficients``
            as ``{node: {parent: array}}`` (a node without ``LS`` terms is
            absent), and ``nodes``, the per-node reports.

        Warns
        -----
        UserWarning
            For each node that did not converge.

        Raises
        ------
        ValueError
            If a term is not classical.
        """
        for nd in self.nodes.values():
            nd._check_classical()
        self._tensorize(train_df)  # every column and level, before a node fits
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
