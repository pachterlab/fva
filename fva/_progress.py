"""Optional tqdm progress bars for the long loops.

Three places in the decomposition are slow enough to be worth reporting, and all
three grow with the number of feature families rather than with the data:

* the ``2^m`` coalitions of the cross-validated Shapley game, each of which is a
  full K-fold ridge fit --- by far the most expensive loop in the package;
* the per-direction loop, which runs the in-sample ``2^m`` game and the
  relative-weights eigendecomposition once per direction;
* the bootstrap resampling, ``B`` refits per direction.

A library should not print unasked, so a bar appears only when the loop is long
enough to be worth watching (:data:`MIN_ITERS`).  Terminal detection is
deliberately *not* used as the trigger: the primary interface is a notebook,
where ``sys.stderr.isatty()`` is False and a bar is exactly what is wanted.
Callers who disagree can pass ``progress=False`` to any public entry point, or
switch it off for the whole session with :func:`fva.set_progress`.
"""

from __future__ import annotations

#: Loops shorter than this run without a bar; below it the bar costs more
#: attention than the wait it describes.  ``2^m`` passes this at ``m = 3``.
MIN_ITERS = 8

#: Session-wide override set by :func:`fva.set_progress`.  ``None`` (the
#: default) leaves the decision to the per-loop heuristic above.
_OVERRIDE: bool | None = None


def set_progress(enabled: bool | None) -> None:
    """Force progress bars on (``True``) or off (``False``) for the session.

    ``None`` restores the default, which shows a bar only for loops of at least
    :data:`MIN_ITERS` iterations.  A ``progress=`` argument passed explicitly to
    a function still wins over this.
    """
    global _OVERRIDE
    _OVERRIDE = enabled


def progress_bar(iterable, *, total: int | None = None, desc: str = "",
                 enabled: bool | None = None):
    """Wrap ``iterable`` in a tqdm bar, or return it unchanged.

    ``enabled=None`` defers to :func:`set_progress` and then to the
    :data:`MIN_ITERS` heuristic; ``True``/``False`` decide outright.  Falls back
    to the bare iterable if tqdm is missing, so the dependency stays soft.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    if enabled is None:
        enabled = _OVERRIDE
    if enabled is None:
        enabled = total is not None and total >= MIN_ITERS
    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
