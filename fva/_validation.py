"""Shared pydantic plumbing for the public API.

Every user-facing function and method is wrapped in :func:`pydantic.validate_call`
using the config and annotated types defined here, so a mistyped argument raises a
:class:`pydantic.ValidationError` naming the offending parameter at the call site
rather than surfacing as an opaque numpy/pandas error deep inside the linear
algebra (or, worse, as a silently wrong number).

Array parameters are annotated with :data:`FloatArray` / :data:`IndexArray` rather
than a bare ``np.ndarray``.  numpy and pandas objects are not pydantic models, so
a bare annotation is validated by ``isinstance`` alone (hence
``arbitrary_types_allowed``) and would reject the lists and DataFrames these
functions have always accepted.  The aliases keep that coercion --- they simply
move the ``np.asarray`` call the functions already made to the boundary, where a
non-numeric input is reported against the parameter that carried it.

Private helpers (``_``-prefixed) are deliberately left undecorated: they are only
ever reached through a validated entry point, and several sit inside the
per-direction loops where the check would be pure overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import BeforeValidator, ConfigDict

# numpy/pandas/matplotlib objects have no pydantic schema; validate by isinstance.
VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _as_index_array(value) -> np.ndarray:
    return np.asarray(value, dtype=int)


#: Any array-like of numbers, coerced to a float ``ndarray`` (no copy if it
#: already is one).
FloatArray = Annotated[np.ndarray, BeforeValidator(_as_float_array)]

#: Any array-like of integers, coerced to an integer ``ndarray``; used for the
#: column-index arrays that select a feature group.
IndexArray = Annotated[np.ndarray, BeforeValidator(_as_index_array)]

#: A matplotlib ``figsize`` --- ``(width, height)`` in inches.
FigSize = tuple[float, float]

#: The three-way figure-saving convention shared by every report method: ``None``
#: (or ``False``) writes nothing, a path writes there, ``True`` writes to the
#: default location.  See :func:`fva.viz.save_figure`.
FigOut = Path | str | bool | None
