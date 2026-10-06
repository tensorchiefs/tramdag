"""Predefined ``fit`` callbacks: ``EarlyStopping``.

Every node fits on its own, so a callback hooks one node's fit. ``fit`` owns
validation and progress printing; the callbacks here read the validation NLL that
``fit`` appends to ``node.history["val"]`` after every epoch — computed once, shared
by all of them. One ``callbacks=`` list is the whole registration
(the ``fit`` docstring shows it); anything not covered here is a
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
            f"{type(cb).__name__} reads node.history['val'] — pass validation_data= "
            "or validation_split= to fit()"
        )
    return node.history["val"][-1]


# %% public classes --------------------------------------------------------------------
class Callback:
    """Base class of ``fit(callbacks=)`` entries — override any of the hooks.

    ``on_fit_begin(node, optimizer)`` runs once after calibration, before the
    first epoch (the shipped callbacks reset their state here, so one
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
    by more than ``min_delta``. With ``restore_best`` (the default)
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

    Attributes
    ----------
    best_nll, best_epoch
        The best validation NLL seen this fit and its epoch.
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

    def on_fit_begin(self, node, optimizer) -> None:
        """Start fresh — neither patience nor the snapshot carries over."""
        self._reset()

    def on_epoch_end(self, node, epoch: int, optimizer) -> bool:
        """Snapshot on improvement; ``True`` once the best is ``patience`` old."""
        nll = _last_val(node, self)
        if nll < self.best_nll - self.min_delta:
            self.best_nll, self.best_epoch = nll, epoch
            if self.restore_best:
                self._state = copy.deepcopy(node.state_dict())
        return self.patience is not None and epoch - self.best_epoch >= self.patience

    def on_fit_end(self, node, optimizer) -> None:
        """Load the best weights back into the node (``restore_best`` only)."""
        if not self.restore_best:
            return
        if self._state is None:  # a first finite NLL always beats inf
            raise RuntimeError(
                "EarlyStopping has nothing to restore: no epoch reached "
                "on_epoch_end with a finite validation NLL. Either fit() ran no "
                "epoch, or the fit diverged — lower learning_rate, or check the "
                "validation frame for a column the model cannot score."
            )
        node.load_state_dict(self._state)
