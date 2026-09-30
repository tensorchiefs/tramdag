"""Predefined ``fit`` callbacks: ``EarlyStopping`` and ``PerNodeEarlyStopping``.

``fit`` owns validation and progress printing; the callbacks here read the per-node
validation NLL that ``fit`` appends to ``flow.history["val"]`` after every epoch —
computed once, shared by all of them. One ``callbacks=`` list is the whole registration
(the ``fit`` docstring shows it); anything not covered here is a
[`Callback`][tramdag.callbacks.Callback] subclass of your own (docs/fitting.md).
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import copy
import math

import torch

# %% global variables ------------------------------------------------------------------
__all__ = ["Callback", "EarlyStopping", "PerNodeEarlyStopping", "per_node_adam"]


# %% private functions -----------------------------------------------------------------
def _last_val(flow, cb: Callback) -> dict[str, float]:
    """Give the current epoch's per-node validation NLL, or fail loudly.

    A stale entry from an earlier validated fit does not count: the last
    entry must belong to the last train epoch (``history["val_epoch"]``
    records which one it belongs to).
    """
    val_epoch = flow.history.get("val_epoch", [])
    if not val_epoch or val_epoch[-1] != len(flow.history["train"]):
        raise RuntimeError(
            f"{type(cb).__name__} reads flow.history['val'] — pass validation_data= "
            "or validation_split= to fit()"
        )
    return flow.history["val"][-1]


# %% public functions ------------------------------------------------------------------
def per_node_adam(flow, lr: float = 1e-2) -> torch.optim.Adam:
    """Give an Adam with one ``node``-tagged parameter group per node.

    The optimizer that
    [`PerNodeEarlyStopping`][tramdag.callbacks.PerNodeEarlyStopping] needs:
    each group carries its node's name and an ``initial_lr`` stamp.

    Parameters
    ----------
    flow : CausalFlowDAG
        The flow whose nodes' parameters form the groups.
    lr : float, optional
        The rate of every group, by default 1e-2.

    Returns
    -------
    torch.optim.Adam
        One group per node, tagged ``node`` and stamped ``initial_lr``.
    """
    return torch.optim.Adam(
        [
            # initial_lr (torch's scheduler convention) lets PerNodeEarlyStopping
            # restore a frozen group to its start at the next fit begin
            {
                "params": list(flow.nodes[n].parameters()),
                "lr": lr,
                "initial_lr": lr,
                "node": n,
            }
            for n in flow.order
        ]
    )


# %% public classes --------------------------------------------------------------------
class Callback:
    """Base class of ``fit(callbacks=)`` entries — override any of the hooks.

    ``on_fit_begin(flow, optimizer)`` runs once after calibration, before the
    first epoch (the shipped callbacks reset their state here, so one
    instance is safe to reuse across fits). ``on_epoch_end(flow, epoch,
    optimizer)`` runs after every epoch, once the epoch's train NLLs are in
    ``flow.history["train"]``; the fit stops after an epoch in which any
    callback returned ``True``. ``on_fit_end(flow, optimizer)`` runs once
    after the loop and **before** the VC re-centering, so a hook that swaps
    the weights hands them to the re-centering.
    """

    def on_fit_begin(self, flow, optimizer) -> None:
        """Run once before the first epoch."""

    def on_epoch_end(self, flow, epoch: int, optimizer):
        """Run after every epoch; return ``True`` to stop the fit."""

    def on_fit_end(self, flow, optimizer) -> None:
        """Run once after the loop, before the VC re-centering."""


class EarlyStopping(Callback):
    """Keep the best-validation weights; optionally stop once they are old.

    Tracks the summed validation NLL; an epoch improves when it beats the best
    by more than ``min_delta``. With ``restore_best`` (the default)
    the weights of the best epoch are snapshotted and loaded back at fit
    end, before the VC re-centering. With ``patience`` the fit also stops
    once the last improvement is that many epochs old; without it (the
    default) the fit runs its full epoch budget and only the restoration
    happens. Reads ``flow.history["val"]``, so the fit needs
    ``validation_data=`` or ``validation_split=``.

    Parameters
    ----------
    patience : int | None, optional
        Epochs without an improvement before stopping; ``None`` (the
        default) never stops.
    restore_best : bool, optional
        Load the best epoch's weights back at fit end, by default True.
    min_delta : float, optional
        Improvement at or below this is flat, by default 0.0.

    Attributes
    ----------
    best_nll, best_epoch
        The best summed validation NLL seen this fit and its epoch.
    """

    def __init__(
        self,
        *,
        patience: int | None = None,
        restore_best: bool = True,
        min_delta: float = 0.0,
    ):
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience!r}")
        if patience is None and not restore_best:
            raise ValueError(
                "patience=None and restore_best=False is a no-op; pass patience= "
                "or restore_best=True"
            )
        self.patience = patience
        self.restore_best = restore_best
        self.min_delta = min_delta
        self._reset()

    def _reset(self) -> None:
        self.best_nll = math.inf
        self.best_epoch = 0
        self._state = None

    def on_fit_begin(self, flow, optimizer) -> None:
        """Start fresh — neither patience nor the snapshot carries over."""
        self._reset()

    def on_epoch_end(self, flow, epoch: int, optimizer) -> bool:
        """Snapshot on improvement; ``True`` once the best is ``patience`` old."""
        nll = sum(_last_val(flow, self).values())
        if nll < self.best_nll - self.min_delta:
            self.best_nll, self.best_epoch = nll, epoch
            if self.restore_best:
                self._state = copy.deepcopy(flow.state_dict())
        return self.patience is not None and epoch - self.best_epoch >= self.patience

    def on_fit_end(self, flow, optimizer) -> None:
        """Load the best weights back into the flow (``restore_best`` only)."""
        if not self.restore_best:
            return
        if self._state is None:  # a first finite NLL always beats inf
            raise RuntimeError(
                "EarlyStopping has nothing to restore: no epoch reached "
                "on_epoch_end with a finite validation NLL. Either fit() ran no "
                "epoch, or the fit diverged — lower learning_rate, or check the "
                "validation frame for a column the model cannot score."
            )
        flow.load_state_dict(self._state)


class PerNodeEarlyStopping(Callback):
    """Early stopping per node: ``EarlyStopping`` on each node's own NLL.

    Tracks every node's validation NLL; an epoch improves a node when it beats
    that node's best by more than ``min_delta``. With ``patience`` a node
    freezes (rate 0) once its best is that many epochs old, and the fit stops
    when every node has frozen; without it (the default) no node freezes and
    the fit runs its full epoch budget. With ``restore_best`` (the default) a
    node loads the weights of its best epoch back when it freezes, and every
    node loads them in ``on_fit_end``, before the VC re-centering.

    Build the optimizer with [`per_node_adam`][tramdag.callbacks.per_node_adam]
    (one ``node``-tagged group per node), and give ``fit`` a validation set
    (the callback reads ``flow.history["val"]``). Do not attach a torch lr
    scheduler to the same optimizer: it could set a frozen node's rate above
    0 again. With ``restore_best``,
    ``fit`` refuses it beside another restoring callback, such as
    ``EarlyStopping(restore_best=True)``.

    Parameters
    ----------
    patience : int | None, optional
        Epochs without an improvement before a node freezes; ``None`` (the
        default) never freezes.
    restore_best : bool, optional
        Load each node's best-epoch weights back, by default True.
    min_delta : float, optional
        Improvement at or below this is flat, by default 0.0.

    Attributes
    ----------
    best_epoch : dict[str, int]
        ``{node: epoch}`` of each node's best validation NLL.
    frozen : dict[str, int]
        ``{node: epoch}`` of the nodes that left training, so a training
        figure can mark the freezes.
    """

    def __init__(
        self,
        *,
        patience: int | None = None,
        restore_best: bool = True,
        min_delta: float = 0.0,
    ):
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience!r}")
        if patience is None and not restore_best:
            raise ValueError(
                "patience=None and restore_best=False is a no-op; pass patience= "
                "or restore_best=True"
            )
        self.patience = patience
        self.restore_best = restore_best
        self.min_delta = min_delta
        self._reset()

    def _reset(self) -> None:
        self.lr0: dict = {}
        self.best: dict = {}
        self.best_epoch: dict[str, int] = {}
        self.frozen: dict[str, int] = {}
        self._state: dict = {}

    def on_fit_begin(self, flow, optimizer) -> None:
        """Start fresh: every group's rate goes back to its ``initial_lr`` stamp."""
        for g in optimizer.param_groups:
            if "initial_lr" in g:
                g["lr"] = g["initial_lr"]
        self._reset()

    def on_epoch_end(self, flow, epoch: int, optimizer) -> bool:
        """Step on the epoch's validation NLL; ``True`` once every node froze."""
        return self._step(flow, _last_val(flow, self), optimizer, epoch)

    def on_fit_end(self, flow, optimizer) -> None:
        """Load every node's best weights again (``restore_best``).

        A frozen node still runs forward in training mode, so its batch-norm
        buffers move after the freeze; this final restore resets them too.
        """
        if self.restore_best:
            for name in self.lr0:
                self._restore(flow, name)

    def _step(self, flow, nll: dict[str, float], optimizer, epoch: int) -> bool:
        """Step every unfrozen node on its own NLL; ``True`` when all are frozen."""
        for g in optimizer.param_groups:
            if "node" not in g:
                raise ValueError(
                    "PerNodeEarlyStopping needs one 'node'-tagged parameter "
                    "group per node — build the optimizer with per_node_adam(flow, lr)"
                )
            if "initial_lr" not in g:
                raise ValueError(
                    "PerNodeEarlyStopping needs the 'initial_lr' stamp on every "
                    "parameter group — build the optimizer with "
                    "per_node_adam(flow, lr); a bare group's current rate may "
                    "already be zero and would silently become the baseline"
                )
            lr0 = self.lr0.setdefault(g["node"], g["initial_lr"])
            if lr0 == 0.0:
                raise ValueError(
                    f"node {g['node']!r} starts at learning rate 0 — build a "
                    "fresh per_node_adam(flow, lr): its initial_lr stamp lets "
                    "a reused optimizer restore its rates"
                )
            if g["node"] not in self.frozen:
                self._step_node(flow, g, nll[g["node"]], epoch)
        return len(self.frozen) == len(optimizer.param_groups)

    def _step_node(self, flow, g: dict, nll: float, epoch: int) -> None:
        name = g["node"]
        if nll < self.best.get(name, math.inf) - self.min_delta:
            self.best[name], self.best_epoch[name] = nll, epoch
            if self.restore_best:
                self._state[name] = copy.deepcopy(flow.nodes[name].state_dict())
        elif (
            self.patience is not None
            and epoch - self.best_epoch.get(name, 0) >= self.patience
        ):
            self.frozen[name] = epoch
            g["lr"] = 0.0
            if self.restore_best:
                self._restore(flow, name)

    def _restore(self, flow, name: str) -> None:
        if name not in self._state:  # a first finite NLL always beats inf
            raise RuntimeError(
                f"PerNodeEarlyStopping has nothing to restore for node {name!r}: "
                "its validation NLL was never finite. Lower learning_rate, or "
                "check the validation frame for a column the model cannot score."
            )
        flow.nodes[name].load_state_dict(self._state[name])
