"""Encoding non-numeric features into the linear feature bank.

The decomposition is a least-squares fit of every latent direction on the feature
matrix (see :mod:`fva.decomposition`), so the *representation* of a feature is
part of the hypothesis class, not a formatting detail.  Two kinds of column are
meaningless in their raw form:

* **Categorical.**  A string column cannot enter the Gram matrix at all --- it
  fails at the array boundary.  Worse is the integer-coded one (``grade = 1, 2,
  3``), which enters silently and is fitted as if the levels were equally spaced
  on a line, so whatever the fit attributes to it is an artefact of the coding.
* **Circular.**  An angle wraps, and the wrap point is an arbitrary choice of
  origin: 359 degrees and 1 degree are two degrees apart, but a linear fit sees
  them at opposite ends of the range.  No linear function of the raw column can
  represent a dependence on the angle.

:func:`encode_features` re-represents both faithfully --- one-hot dummies for the
categorical, ``sin``/``cos`` for the circular --- and returns the ``feature ->
group`` mapping alongside the matrix.  The mapping is the point: an encoded
column is one concept spread over several columns, which is exactly what a
feature group is, so the dummies of one categorical stay pooled into a single
budget entry rather than appearing as unrelated features.  That also keeps them
from inflating the group count, which the ``2**m`` Shapley game is exponential
in.

What this module deliberately does *not* offer is a general basis expansion
(Fourier terms, splines) over continuous features.  The encodings here are
faithful re-representations: they add no explanatory power a correctly-read
column did not already have.  A basis expansion is a capacity choice --- it
widens the bank, raises ``R^2``, and eats into the noise remainder, which is only
interpretable as "irreducible" relative to a *fixed* hypothesis class.  The
``period`` argument covers the one case where extra harmonics are a matter of
correctness rather than capacity (axial data, below).

Coding conventions
------------------
One-hot expansion drops a reference level by default (``drop_first``).  Features
are centred before the fit, so a full set of ``k`` dummies is exactly linearly
dependent with the intercept: the Gram matrix is rank deficient, and Johnson's
relative weights --- which need it positive definite --- distribute the
per-column split among the collinear columns by numerical noise (see
:func:`fva.decomposition.relative_weights`).  The ``k - 1`` retained dummies
span the same subspace as the full set once centred, so **the group-level budget
is invariant to which level is dropped**; only the per-column Johnson row depends
on the choice, and reads as a contrast against the reference level.

Circular encoding maps a value ``v`` to ``sin(2*pi*v/period)`` and
``cos(2*pi*v/period)``, so ``period`` is given in the units of the column itself
(``360`` for degrees, ``2*pi`` for radians).  This pair captures the first
harmonic --- one full cycle over the period.  If the quantity is an *orientation*
rather than a direction, so that ``v`` and ``v + period/2`` describe the same
object (the major axis of a lesion, a fibre direction), the first harmonic is
identically wrong and the right encoding is the doubled angle: pass half the
turn, ``period=180`` for degrees.  Inference cannot settle this --- it is a fact
about the measurand, not about the numbers --- so it is warned about, never
guessed.

Inference
---------
``infer=True`` (the default) only claims columns whose raw reading is
unambiguously wrong:

* a column holding **string values** (``object``, ``string`` or ``category``
  dtype) is one-hot encoded;
* a **numeric** column whose *name* matches :data:`CIRCULAR_NAME_PATTERN`
  (``angle`` / ``rotation``) and whose values all lie within ``+/-360`` is
  encoded as circular, with degrees or radians inferred from the observed range.

Integer-coded categoricals are never inferred: ``1, 2, 3`` is genuinely
ambiguous (nominal, ordinal, or an already-fine count), and guessing wrong is
silent either way.  Pass them explicitly through ``categorical=``.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from pydantic import InstanceOf, validate_call

from ._validation import VALIDATE_CONFIG

logger = logging.getLogger(__name__)

#: Column names matching this pattern are candidates for circular encoding when
#: ``infer=True``.  Matched case-insensitively anywhere in the name.
CIRCULAR_NAME_PATTERN = re.compile(r"angle|rotation", re.IGNORECASE)

#: A candidate circular column is only claimed by inference if every finite value
#: lies within this bound --- one full turn in degrees.
CIRCULAR_INFER_MAX_ABS = 360.0

#: Largest ``|value|`` that still reads as radians rather than degrees when the
#: unit of an inferred circular column has to be guessed.
RADIAN_MAX_ABS = 2.0 * np.pi

#: Above this many distinct levels a one-hot expansion warns: the bank widens by
#: ``levels - 1`` columns, and a column this wide is often an identifier that was
#: never meant to be a feature.
MAX_LEVELS_WARN = 20

#: Separator between a source column and the level (or ``sin``/``cos``) in an
#: encoded column name.  ``=`` rather than ``_`` so encoded columns are
#: distinguishable from raw ones at a glance in the tables and figures.
ENCODED_SEP = "="

_TOL = 1e-9


class EncodedFeatures(NamedTuple):
    """Return value of :func:`encode_features`.

    Unpack as ``df_feat, feat_to_group = fva.encode_features(df)`` or access by
    name.
    """

    frame: pd.DataFrame       # (n, q') all-numeric feature matrix
    feat_to_group: dict       # encoded feature name -> group name


def _is_string_column(s: pd.Series) -> bool:
    """True if ``s`` holds string values (and so cannot be read as a number)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return any(isinstance(v, str) for v in s.cat.categories)
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    return any(isinstance(v, str) for v in s.dropna())


def _infer_period(values: np.ndarray, name: str) -> tuple[float, bool]:
    """Guess the turn of an inferred circular column from its observed range.

    Returns ``(period, ambiguous)``.  Values that fit inside a single turn of
    radians are read as radians, anything wider as degrees.  The reading is
    reported as ambiguous when the column is narrow enough that both units are
    plausible --- degrees data confined to a few degrees looks exactly like
    radians data --- so the caller can say so rather than silently picking one.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return CIRCULAR_INFER_MAX_ABS, False
    max_abs = float(np.abs(finite).max())
    span = float(finite.max() - finite.min())
    if max_abs <= RADIAN_MAX_ABS + _TOL:
        # A genuine radian column usually sweeps most of a turn; one that covers
        # less than half is equally consistent with a narrow degree range.
        return 2.0 * np.pi, span < np.pi
    return CIRCULAR_INFER_MAX_ABS, False


def _one_hot(s: pd.Series, name: str, *, drop_first: bool,
             inferred: bool) -> pd.DataFrame:
    """One-hot expand ``s``, dropping a reference level unless told otherwise."""
    n_levels = int(s.nunique(dropna=True))
    has_na = bool(s.isna().any())
    if n_levels > MAX_LEVELS_WARN:
        warnings.warn(
            f"categorical column {name!r} has {n_levels} levels and expands to "
            f"{n_levels - (1 if drop_first else 0)} columns"
            + (" (it was inferred from string values --- if it is an identifier "
               "rather than a feature, drop it or pass infer=False)"
               if inferred else "")
            + ". A bank this wide inflates the in-sample R^2; read rho_cv.",
            stacklevel=3,
        )
    if has_na:
        warnings.warn(
            f"categorical column {name!r} has missing values; they are encoded "
            f"as their own level rather than folded into the reference level.",
            stacklevel=3,
        )
    if n_levels < 2:
        warnings.warn(
            f"categorical column {name!r} has {n_levels} level(s) and carries no "
            f"variance; it was dropped.",
            stacklevel=3,
        )
        return pd.DataFrame(index=s.index)

    out = pd.get_dummies(
        s, prefix=name, prefix_sep=ENCODED_SEP, dummy_na=has_na,
        drop_first=drop_first, dtype=float,
    )
    return out


def _sin_cos(s: pd.Series, name: str, period: float) -> pd.DataFrame:
    """Encode ``s`` as one turn of ``sin``/``cos`` over ``period``."""
    theta = 2.0 * np.pi * s.to_numpy(dtype=float) / period
    return pd.DataFrame(
        {f"{name}{ENCODED_SEP}sin": np.sin(theta),
         f"{name}{ENCODED_SEP}cos": np.cos(theta)},
        index=s.index,
    )


def _check_circular(s: pd.Series, name: str, period: float, *,
                    ambiguous: bool) -> None:
    """Warn about the two readings inference cannot settle: unit and half-turn."""
    values = s.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    if not np.isfinite(values).all():
        warnings.warn(
            f"circular column {name!r} has missing values; sin/cos of NaN is NaN "
            f"and propagates through the fit. Impute or drop those rows.",
            stacklevel=3,
        )
    if ambiguous:
        warnings.warn(
            f"circular column {name!r} spans less than half a turn of radians, "
            f"so its unit is ambiguous; it was read as radians (period "
            f"{period:.4g}). Pass circular={{{name!r}: 360}} if it is degrees.",
            stacklevel=3,
        )
        return
    span = float(finite.max() - finite.min())
    if span <= period / 2.0 + _TOL:
        warnings.warn(
            f"circular column {name!r} spans {span:.4g} of its {period:.4g} "
            f"turn --- at most half. If it is an orientation rather than a "
            f"direction (v and v + {period / 2:.4g} describe the same object), "
            f"the first harmonic is the wrong encoding: pass "
            f"circular={{{name!r}: {period / 2:.4g}}} to double the angle.",
            stacklevel=3,
        )


@validate_call(config=VALIDATE_CONFIG)
def encode_features(
    df: InstanceOf[pd.DataFrame],
    *,
    categorical: list[str] | None = None,
    circular: dict[str, float] | list[str] | None = None,
    groups: dict | None = None,
    infer: bool = True,
    drop_first: bool = True,
) -> EncodedFeatures:
    """Re-represent categorical and circular columns for the linear feature bank.

    Parameters
    ----------
    df : DataFrame, shape (n, q)
        Raw features, one column per feature.  Columns not named below and not
        claimed by inference pass through untouched and must already be numeric.
    categorical : list of str, optional
        Columns to one-hot encode, whatever their dtype.  This is where
        integer-coded categoricals go (``grade = 1, 2, 3``): they are never
        inferred, because the coding is genuinely ambiguous.
    circular : list of str or dict, optional
        Columns to encode as ``sin``/``cos``.  A list infers the period; a dict
        gives it per column, **in the units of the column itself** --- ``360``
        for degrees, ``2*pi`` for radians, and *half* of those (``180``, ``pi``)
        for an axial quantity whose value repeats every half turn.  See the
        module docstring.
    groups : dict, optional
        ``source column name -> group name``, keyed by the *pre-encoding* names.
        Every column an encoding produces inherits its source column's group.
        Columns missing from the mapping are left out of the returned one, so
        :class:`~fva.explainer.FeatureGroundedDecomposition` reports them as
        ungrouped in the usual way.  ``None`` (the default) puts each source
        column in a group of its own, which reproduces the per-feature default
        while keeping each encoding's columns pooled.
    infer : bool, default True
        Claim string columns as categorical and ``angle``/``rotation`` columns
        within ``+/-360`` as circular.  See the module docstring for the exact
        rules.  ``False`` encodes only what is named explicitly.
    drop_first : bool, default True
        Drop one level per one-hot expansion.  Keep every level only if the
        per-level Johnson row is what is wanted and the rank deficiency it
        introduces is understood --- the group budget is unaffected either way.

    Returns
    -------
    EncodedFeatures
        ``(frame, feat_to_group)``, ready to hand to
        :class:`~fva.explainer.FeatureGroundedDecomposition`.

    Raises
    ------
    ValueError
        If a named column is missing, is named twice, produces a name that
        collides with another column, or if any column left after encoding is
        not numeric.

    Warns
    -----
    UserWarning
        On the judgement calls encoding cannot make from the data alone: a
        one-hot expansion wide enough to look like an identifier, missing values
        in either kind of column, an inferred circular column whose unit is
        ambiguous, and one narrow enough to be axial rather than directional.

    Examples
    --------
    >>> df_feat, feat_to_group = fva.encode_features(
    ...     df_raw,
    ...     categorical=["grade"],              # integer-coded: not inferred
    ...     circular={"lesion_angle": 180},     # axial: repeats every half turn
    ...     groups={"grade": "clinical", "lesion_angle": "shape"},
    ... )
    >>> decomp = fva.FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)
    """
    cat_cols = list(categorical or [])
    if isinstance(circular, dict):
        circ_cols: dict[str, float | None] = {k: float(v) for k, v in circular.items()}
    else:
        circ_cols = {c: None for c in (circular or [])}

    missing = [c for c in [*cat_cols, *circ_cols] if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) named for encoding are not in df: {missing}. "
            f"Available columns: {list(df.columns)[:10]}"
            + (" ..." if df.shape[1] > 10 else "")
        )
    both = sorted(set(cat_cols) & set(circ_cols))
    if both:
        raise ValueError(
            f"column(s) named as both categorical and circular: {both}."
        )
    for period in circ_cols.values():
        if period is not None and not (np.isfinite(period) and period > 0):
            raise ValueError(
                f"circular periods must be finite and positive; got {period!r}."
            )

    explicit = set(cat_cols) | set(circ_cols)
    if infer:
        for col in df.columns:
            if col in explicit:
                continue
            s = df[col]
            if _is_string_column(s):
                cat_cols.append(col)
            elif (pd.api.types.is_numeric_dtype(s)
                  and not pd.api.types.is_bool_dtype(s)
                  and CIRCULAR_NAME_PATTERN.search(str(col))):
                values = s.to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size and np.abs(finite).max() <= CIRCULAR_INFER_MAX_ABS:
                    circ_cols[col] = None

    # Build the output column by column, in the original order, so an encoding
    # expands in place rather than being appended at the end.
    pieces: list[pd.DataFrame] = []
    expansion: dict[str, list[str]] = {}
    for col in df.columns:
        s = df[col]
        if col in circ_cols:
            period, ambiguous = (
                (circ_cols[col], False) if circ_cols[col] is not None
                else _infer_period(s.to_numpy(dtype=float), str(col))
            )
            _check_circular(s, str(col), period, ambiguous=ambiguous)
            part = _sin_cos(s, str(col), period)
            logger.info("encoded circular column %r as sin/cos (period %.6g).",
                        col, period)
        elif col in cat_cols:
            part = _one_hot(s, str(col), drop_first=drop_first,
                            inferred=col not in explicit)
            logger.info("encoded categorical column %r as %d dummy column(s).",
                        col, part.shape[1])
        else:
            part = s.to_frame()
        if part.shape[1]:
            pieces.append(part)
            expansion[col] = list(part.columns)

    out = (pd.concat(pieces, axis=1) if pieces
           else pd.DataFrame(index=df.index))

    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"encoding produced duplicate column name(s): {sorted(set(duplicated))}. "
            f"Rename the source column(s) --- an encoded name is "
            f"'<column>{ENCODED_SEP}<level>'."
        )
    non_numeric = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    if non_numeric:
        raise ValueError(
            f"column(s) are not numeric and were not encoded: {non_numeric}. "
            f"Pass them as categorical=[...] (or circular=[...]) if they are "
            f"features, or drop them from df."
        )

    if groups is None:
        feat_to_group = {name: str(src) for src, names in expansion.items()
                         for name in names}
    else:
        feat_to_group = {name: groups[src] for src, names in expansion.items()
                         if src in groups for name in names}

    logger.info("encode_features: %d column(s) in, %d out, %d group(s).",
                df.shape[1], out.shape[1], len(set(feat_to_group.values())))
    return EncodedFeatures(frame=out, feat_to_group=feat_to_group)
