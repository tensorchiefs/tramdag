"""The two fitting paths: ``FitMixin``, composed into ``CausalFlowDAG``.

`fit` is one minibatch Adam loop (validation, verbose printing and the
callback hooks included); `fit_classical` is the float64 full-batch L-BFGS
exact-MLE route for all-`ls` specs.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch import Tensor

from .callbacks import Callback

if TYPE_CHECKING:
    from .flow import CausalFlowDAG


# %% global variables ------------------------------------------------------------------
__all__ = ["FitMixin"]


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
    hook, called as ``cb(flow, epoch, optimizer)``.
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
    return out


def _learning_rates(opt) -> dict[str, float] | float | list[float]:
    """Give the optimizer's current rate(s): per node when the groups are tagged."""
    groups = opt.param_groups
    if all("node" in g for g in groups):
        return {g["node"]: float(g["lr"]) for g in groups}
    if len(groups) == 1:
        return float(groups[0]["lr"])
    return [float(g["lr"]) for g in groups]  # a hand-built untagged optimizer


def _log_epoch(
    flow, epoch: int, epochs: int, verbose: int, stopped: bool, has_val: bool
):
    """Print one ``verbose`` progress line on the Nth and the final epoch."""
    last = stopped or epoch == epochs
    if not verbose or (epoch % verbose and not last):
        return
    line = f"epoch {epoch}/{epochs}"
    line += f"  train {sum(flow.history['train'][-1].values()):.4f}"
    if has_val:  # THIS fit's validation, not a stale earlier one
        line += f"  val {sum(flow.history['val'][-1].values()):.4f}"
    print(line)


def _fit_epoch(
    flow,
    values: dict[str, Tensor],
    opt: torch.optim.Optimizer,
    batch_size: int,
    penalized: list,
) -> dict[str, float]:
    """One shuffled pass over the rows; give the epoch-mean train NLL per node."""
    n = len(next(iter(values.values())))
    acc = dict.fromkeys(flow.order, 0.0)
    trained = 0
    for idx in torch.randperm(n, device=flow.device).split(batch_size):
        if idx.numel() < 2:
            continue  # batch norm needs two rows, and one row is no gradient
        trained += int(idx.numel())
        batch = {k: v[idx] for k, v in values.items()}
        per_node = flow.node_log_prob(batch)
        nlls = {k: -v.mean() for k, v in per_node.items()}
        loss = torch.stack(list(nlls.values())).sum()
        for m in penalized:  # the penalty joins the loss, not the history
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
    # over the rows actually stepped on, not over n: a skipped trailing row
    # would otherwise scale every node's epoch NLL down by 1/n
    return {k: v / trained for k, v in acc.items()}


# %% private classes -------------------------------------------------------------------
class _FnCallback(Callback):
    """A bare callable in ``callbacks=``, adapted to an ``on_epoch_end`` hook."""

    def __init__(self, fn):
        self.fn = fn

    def on_epoch_end(self, flow, epoch: int, optimizer):
        return self.fn(flow, epoch, optimizer)


# %% public classes --------------------------------------------------------------------
class FitMixin:
    """The two fitting paths, mixed into [`CausalFlowDAG`][tramdag.CausalFlowDAG]."""

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
        optimizer: torch.optim.Optimizer | None = None,
        callbacks=None,
    ) -> CausalFlowDAG:
        """Fit all nodes jointly by maximum likelihood — one minibatch Adam loop.

        The joint NLL decomposes per node, so one optimizer over all
        parameters fits every node at once. The loop keeps the **final**
        weights, and a second ``fit`` call continues the training. Validation
        monitoring, learning-rate schedules, early stopping, best-weight
        restoration and logging are the caller's, through ``optimizer`` and
        ``callbacks``; [`callbacks`][tramdag.callbacks] ships the common
        recipes. A ``VC`` term adds its penalty to the loss, never to
        ``history["train"]``, and is re-centered after the loop.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node.
        epochs : int
            Number of passes over the data. Required; there is no default.
        learning_rate : float, optional
            Adam step size of the default optimizer, by default 1e-2.
            Ignored when ``optimizer`` is given.
        batch_size : int, optional
            Rows per gradient step, by default 512. ``len(train_df)`` is one
            full-batch step per epoch.
        validation_data : pd.DataFrame | None, optional
            Validation rows, one column per node. When given (or split off), the
            per-node validation NLL is appended to ``flow.history["val"]`` after
            every epoch; the shipped callbacks read it there. ``history["lr"]``
            records the optimizer's rate per epoch (``{node: lr}`` for tagged
            groups, a float for one group, a list for several).
        validation_split : float | None, optional
            Keras' rule: the LAST fraction of ``train_df`` becomes the
            validation set, without shuffling, and only the remaining rows
            train (and calibrate — no leakage into the frozen statistics).
            Mutually exclusive with ``validation_data``.
        verbose : int, optional
            0 (default) is silent. ``N >= 1`` prints one line every ``N``
            epochs and on the final epoch: epoch counter, summed train NLL,
            summed validation NLL when validation is configured. No
            progress bars.
        seed : int | None, optional
            Seeds torch's global RNG before the loop, for the minibatch
            shuffling. Weight initialization is seeded at construction
            (``CausalFlowDAG(spec, seed=...)``).
        optimizer : torch.optim.Optimizer | None, optional
            Any torch optimizer over ``flow.parameters()``; the default is
            ``Adam(lr=learning_rate)``. Build it yourself to attach a
            ``torch.optim.lr_scheduler`` or to continue with its state.
        callbacks : Callback | callable | list | None, optional
            One entry or a list. A [`Callback`][tramdag.callbacks.Callback]
            hooks all three points of the fit — its docstring is the
            contract (begin/epoch/end timing, the stop rule, the VC
            re-centering order). A bare callable is an ``on_epoch_end``
            hook, ``cb(flow, epoch, optimizer)`` — use it for schedules and
            coefficient trajectories. ``callbacks`` ships
            ``EarlyStopping`` and ``PerNodePlateau``, all
            reading ``history["val"]``.

        Returns
        -------
        CausalFlowDAG
            ``self``, fitted, in eval mode.

        Raises
        ------
        ValueError
            If ``epochs`` or ``batch_size`` is below 1, ``verbose`` is
            negative, both validation arguments are given, the split leaves
            an empty side, or a centered VC term's propensity column is
            missing from the training frame or out of [0, 1].
        TypeError
            If a ``callbacks`` entry is neither a ``Callback`` nor a callable,
            or is a ``Callback`` class instead of an instance.
        """
        _check_fit_sizes(epochs, batch_size, verbose)
        cbs = _normalize_callbacks(callbacks)
        if seed is not None:
            torch.manual_seed(seed)
        train_df, validation_data = _split_validation(
            train_df, validation_data, validation_split
        )
        # validate BEFORE calibrate: a malformed frame must not half-mutate the flow
        side_cols = self._check_side_columns(train_df)
        self.calibrate(train_df)
        values = self._tensorize(train_df, list(self.order) + side_cols)
        val_values = (
            self._tensorize(validation_data) if validation_data is not None else None
        )
        opt = optimizer or torch.optim.Adam(self.parameters(), lr=learning_rate)
        penalized = [
            m
            for nd in self.nodes.values()
            for m in nd.shifts.values()
            if m.regularizer() is not None
        ]
        for cb in cbs:
            cb.on_fit_begin(self, opt)
        for epoch in range(1, epochs + 1):
            self.train()
            epoch_nll = _fit_epoch(self, values, opt, batch_size, penalized)
            self.history["train"].append(epoch_nll)
            self.eval()
            if val_values is not None:
                self.history.setdefault("val", []).append(self._mean_nll(val_values))
                # which train epoch this entry belongs to: `history` accumulates
                # across `fit` calls, so an unvalidated fit would shift every later
                # validation entry away from its own epoch
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
        self._recenter_vc(values)
        self.eval()
        return self

    def fit_classical(
        self,
        train_df: pd.DataFrame,
        *,
        max_iter: int = 400,
        history_size: int = 50,
    ) -> dict:
        """Fit an all-``ls`` model the classical way.

        The fit uses full batches, float64, and L-BFGS with a strong-Wolfe
        line search. There are no minibatches, no schedule and no early
        stopping, so the fit is deterministic and lands on the maximum-
        likelihood estimate. It is valid only when every term is a simple
        intercept or an ``LS``, because each node-conditional is then a
        classical transformation model; any other spec raises.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node.
        max_iter : int, optional
            Upper limit on L-BFGS iterations, by default 400.
        history_size : int, optional
            L-BFGS memory, by default 50.

        Returns
        -------
        dict
            A convergence report: ``converged``, ``n_iter``, ``final_nll``,
            ``grad_norm``, ``seconds``, and the fitted ``coefficients``
            from [`ls_coefficients`][tramdag.flow.CausalFlowDAG.ls_coefficients].

        Raises
        ------
        ValueError
            If a term is not classical: anything but the simple intercept and ``LS``.

        Notes
        -----
        float64 is a transient compute mode. The model is upcast for the
        fit, and ``self.double()`` converts the parameters and the range
        buffers of the transforms in one call. Afterwards the model returns
        to float32, so the stored model and ``save``/``load`` stay float32.
        Double precision is what lets the line search resolve the optimum
        cleanly.

        Convergence is torch's own: L-BFGS stops when the NLL or the
        parameters move by less than 1e-9; ``tolerance_grad`` is 0, so the
        gradient never ends the run. ``converged`` says whether a tolerance
        ended the run rather than ``max_iter``; ``|grad|`` and weakly
        identified coefficients do not settle to machine precision.
        """
        other = sorted(
            {t.name for nd in self.spec.values() for t in nd.terms if not t.classical}
        )
        if other:
            raise ValueError(
                "fit_classical requires an all-`ls` spec, that is a simple "
                f"intercept and LS terms only; this spec has {other} terms. Use "
                "fit() for flexible models."
            )
        self.calibrate(train_df)
        self.double()  # parameters + buffers (xmin/xmax) -> float64, one call
        t0 = time.perf_counter()
        try:
            values = self._tensorize(train_df)
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

            def total_nll() -> Tensor:
                per_node = self.node_log_prob(values).values()
                return torch.stack([-lp.mean() for lp in per_node]).sum()

            def closure():
                opt.zero_grad()
                loss = total_nll()
                loss.backward()
                return loss

            opt.step(closure)
            n_iter = next(iter(opt.state.values()))["n_iter"]
            converged = n_iter < max_iter  # torch stopped on a tolerance
            with torch.no_grad():
                final_nll = float(total_nll())
            grad_norm = float(
                torch.nn.utils.get_total_norm(
                    [p.grad for p in self.parameters() if p.grad is not None]
                )
            )
            coefs = self.ls_coefficients()  # read while still float64
        finally:
            self.float()  # restore canonical float32 (lossy ~1e-7, harmless)
        self.eval()

        return {
            "converged": converged,
            "n_iter": n_iter,
            "final_nll": final_nll,
            "grad_norm": grad_norm,
            "seconds": time.perf_counter() - t0,
            "coefficients": coefs,
        }
