"""Predefined ``fit`` callbacks: ``EarlyStopping``.

Every node fits on its own, so a callback hooks one node's fit. ``fit`` owns
validation and progress printing; ``EarlyStopping`` reads the validation NLL that
``fit`` appends to ``node.history["val"]`` after every epoch. One ``callbacks=``
list is the whole registration; anything not covered here is a
[`Callback`][tramdag.callbacks.Callback] subclass of your own (docs/fitting.md).
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import copy
import math

# %% global variables ------------------------------------------------------------------
__all__ = ["Callback", "EarlyStopping"]


# %% private functions -----------------------------------------------------------------
def _last_val(node, cb: Callback) -> float:
    """Give the current epoch's validation NLL of the node, or fail loudly.

    A stale entry from an earlier validated fit does not count: the last
    entry must belong to the last train epoch (``history["val_epoch"]``
    records which one it belongs to).
    """
    val_epoch = node.history.get("val_epoch", [])
    if not val_epoch or val_epoch[-1] != len(node.history["train"]):
        raise RuntimeError(
            f"{type(cb).__name__} reads node.history['val']; pass validation_data= "
            "or validation_split= to fit()"
        )
    return node.history["val"][-1]


# %% public classes --------------------------------------------------------------------
class Callback:
    """Base class of ``fit(callbacks=)`` entries; override any of the hooks.

    ``on_fit_begin(node, optimizer)`` runs once after calibration, before the
    first epoch (the shipped ``EarlyStopping`` resets its state here, so one
    instance is safe to reuse across fits). ``on_epoch_end(node, epoch,
    optimizer)`` runs after every epoch, once the epoch's train NLL is in
    ``node.history["train"]``; the fit stops after an epoch in which any
    callback returned ``True``. ``on_fit_end(node, optimizer)`` runs once
    after the loop and **before** the VC re-centering, so a hook that swaps
    the weights hands them to the re-centering.
    """

    def on_fit_begin(self, node, optimizer) -> None:
        """Run once before the first epoch."""

    def on_epoch_end(self, node, epoch: int, optimizer):
        """Run after every epoch; return ``True`` to stop the fit."""

    def on_fit_end(self, node, optimizer) -> None:
        """Run once after the loop, before the VC re-centering."""


class EarlyStopping(Callback):
    """Keep the best-validation weights; optionally stop once they are old.

    Tracks the node's validation NLL; an epoch improves when it beats the best
    by more than ``min_delta``. With ``smoothing`` the tracked value is the
    exponential moving average of the validation NLL, so the best epoch is the
    one with the best smoothed value, its weights are restored, and
    ``best_nll`` is that smoothed value. The average starts at the first
    validation NLL and also runs through the epochs before ``min_epochs``.
    With ``restore_best`` (the default)
    the weights of the best epoch are snapshotted and loaded back at fit
    end, before the VC re-centering. With ``patience`` the fit also stops
    once the last improvement is that many epochs old; without it (the
    default) the fit runs its full epoch budget and only the restoration
    happens. Reads ``node.history["val"]``, so the fit needs
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
    min_epochs : int, optional
        No improvement counts and no stop happens in an epoch below this
        number, by default 0. Epochs count from 1.
    smoothing : float, optional
        Factor ``a`` of the moving average ``a * old + (1 - a) * nll``; 0, the
        default, uses the raw validation NLL.

    Attributes
    ----------
    best_nll, best_epoch
        The best validation NLL seen this fit and its epoch. The node's
        ``history["best_epoch"]`` holds that epoch on the history axis,
        which counts the epochs of every fit.

    Raises
    ------
    ValueError
        If ``patience`` is below 1, ``min_epochs`` is negative, ``smoothing`` is
        outside [0, 1), or ``patience=None`` comes with
        ``restore_best=False``.
    RuntimeError
        During the fit, if it has no validation data, or at fit end if no
        epoch gave a finite validation NLL to restore.
    """

    def __init__(
        self,
        *,
        patience: int | None = None,
        restore_best: bool = True,
        min_delta: float = 0.0,
        min_epochs: int = 0,
        smoothing: float = 0.0,
    ):
        if min_epochs < 0:
            raise ValueError(f"min_epochs must be at least 0, got {min_epochs!r}")
        if not 0 <= smoothing < 1:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing!r}")
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
        self.min_epochs = min_epochs
        self.smoothing = smoothing
        self._reset()

    def _reset(self) -> None:
        self.best_nll = math.inf
        self.best_epoch = 0
        self._state = None
        self._average = None

    def on_fit_begin(self, node, optimizer) -> None:
        """Start fresh: neither patience nor the snapshot carries over."""
        self._reset()

    def on_epoch_end(self, node, epoch: int, optimizer) -> bool:
        """Snapshot on improvement; ``True`` once the best is ``patience`` old."""
        nll = _last_val(node, self)
        if self.smoothing and self._average is not None:
            nll = self.smoothing * self._average + (1 - self.smoothing) * nll
        self._average = nll
        if epoch < self.min_epochs:
            return False
        if nll < self.best_nll - self.min_delta:
            self.best_nll, self.best_epoch = nll, epoch
            node.history["best_epoch"] = len(node.history["train"])
            if self.restore_best:
                self._state = copy.deepcopy(node.state_dict())
        return self.patience is not None and epoch - self.best_epoch >= self.patience

    def on_fit_end(self, node, optimizer) -> None:
        """Load the best weights back into the node (``restore_best`` only).

        A first finite NLL always beats inf, so no snapshot means that no epoch
        reached a finite validation NLL.
        """
        if not self.restore_best:
            return
        if self._state is None:
            raise RuntimeError(
                "EarlyStopping has nothing to restore: no epoch reached "
                "on_epoch_end with a finite validation NLL. Either fit() ran no "
                "epoch, or fewer epochs than min_epochs, or the fit diverged. "
                "Lower learning_rate, or check the validation frame for a column "
                "the model cannot score."
            )
        node.load_state_dict(self._state)
