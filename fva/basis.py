"""Basis expansion of continuous features, as an explicit capacity choice.

:mod:`fva.encoding` re-represents columns whose raw numbers do not mean what a
line would read into them --- a categorical, an angle.  Those encodings are
*faithful*: they add no explanatory power a correctly-read column did not already
have, so the noise remainder the package reports keeps its meaning.  This module
does something different and it is worth being blunt about it: a basis expansion
**widens the hypothesis class**.  It raises ``R^2``, it eats into the noise
budget, and a ``rho`` computed against an expanded bank is not comparable with a
``rho`` computed against an unexpanded one.

The reason to offer it anyway is that the map from an interpretable feature to a
latent coordinate is routinely smooth but nonlinear --- a sprite's position
produces a localised pixel pattern that the encoder folds nonlinearly into the
code --- and a linear fit on the raw column then reports as "noise" a dependence
that is perfectly well described by the named feature.  That is a statement about
the hypothesis class, not about the feature, and the honest way to tell the two
apart is to *vary* the capacity and look, which is what :func:`capacity_ladder`
does.  A single hidden ``expand=True`` flag would hide exactly the thing worth
seeing; the ladder makes the dependence of ``rho`` on the vocabulary into the
reported quantity.

Choosing the basis
------------------
Which basis is right is decided by the feature, not by taste:

* A **circular** feature (an angle, a phase, a time of day) is genuinely
  periodic, and the Fourier basis ``sin(2 pi k v / period)``,
  ``cos(2 pi k v / period)`` for ``k = 1..harmonics`` is its natural expansion.
  ``harmonics=1`` is the pair :func:`fva.encode_features` already emits, so the
  ladder starts from the faithful encoding and widens from there.
* A **non-periodic continuous** feature gets a cubic B-spline basis with
  ``spline_df`` columns, knots at quantiles of the observed values.  A Fourier
  basis is the wrong choice here even though it is the one hand-rolled feature
  banks usually reach for: rescaling the column to a unit period forces the
  fitted function to take the same value at both ends of the range, an
  endpoint constraint the data has no reason to satisfy, and the raw column then
  has to be carried alongside purely to let the fit break that wrap-around.  The
  spline has no wrap, has local support, is far better conditioned, and
  degenerates *exactly* to the raw column at ``spline_df=1`` (a degree-1 B-spline
  on two boundary knots is the rescaled identity), so again the ladder starts at
  the unexpanded bank.
* A **zero-inflated continuous** feature (a gene's log-normalized expression,
  where most cells sit exactly at zero) is better served by a hurdle spline,
  ``basis="hurdle_spline"``: an indicator of leaving the point mass plus a cubic
  B-spline on the values off it, with knots at quantiles of those values alone.
  A plain spline puts its quantile knots on the point mass, where they coincide
  with the boundary and are dropped, so it silently degrades into one global
  polynomial fitted mostly to the tail.  The hurdle basis spans the on/off jump
  and the dose-response separately, and it contains the raw column, so its
  ``R^2`` is never below the unexpanded bank's.

``basis`` chooses the expansion for non-periodic numeric columns: ``"none"``
(the raw column, no capacity), ``"spline"`` (the default), ``"hurdle_spline"``,
or ``"fourier"``.  A numeric column with no values at the point mass has nothing
for a hurdle to separate and falls back to a plain spline, with a warning.

``mode="auto"`` gives circular columns harmonics and every other numeric column
the ``basis``.  ``mode="circular"`` forces harmonics onto every numeric column,
imposing a period equal to the observed range and carrying the raw column as the
ramp; it reproduces the separable Fourier banks that appear in the literature,
is the same as ``basis="fourier"``, and is provided for comparison, with a
warning.  ``mode="spline"`` applies the ``basis`` to the circular columns too,
which drops the wrap constraint those columns genuinely need.

Interactions
------------
A per-column basis, however wide, stays *separable*: it writes the latent
coordinate as a sum of a function of each feature and can express no joint
dependence between two of them.  Some are structural rather than incidental ---
which pixels a sprite occupies is set jointly by its two position coordinates,
and no sum of a function of ``x`` and a function of ``y`` localises a
two-dimensional bump --- and a separable bank reports that dependence as noise.
``interactions=[("posX", "posY")]`` adds the tensor product of the two features'
bases, which is the smallest basis that spans the joint dependence, as one
further group.  It is a second, larger widening of the hypothesis class on top of
the per-column one and carries the same caveat: name the pairs the features
imply, read the gain against the ladder, and do not go fishing over all pairs.

Because each per-column basis sums to a constant across its own columns, the
tensor product spans both main effects as well as the interaction.  The two
groups are therefore collinear by construction, which the Shapley budget handles
--- it is the situation group attribution exists for --- but the shared mass is
real and the pair's marginal share should be read with that in mind.

Grouping
--------
Every column an expansion produces inherits its source column's group, so the
whole basis for one feature stays pooled into a single budget entry.  This is not
a convenience: the ``2**g`` Shapley game is exponential in the group count, and
an expansion that split into ``g`` unrelated groups would make the budget
exponentially more expensive while attributing fractions of one concept to
several families.  An interaction is one group per pair, for the same reason.

Guards
------
Widening a bank has two failure modes the package can detect, and both are
checked here rather than left to the fit:

* **More columns than the column can support.**  A "continuous" feature with
  ``u`` distinct values spans at most ``u - 1`` dimensions once centred, so a
  basis wider than that is rank deficient by construction --- the group budget
  survives it, but Johnson's per-feature split among the collinear columns is
  then arbitrary.  The requested width is clipped to what the column can carry,
  with a warning naming the column.
* **More columns than the sample can support.**  The expanded width goes through
  :func:`fva._diagnostics.check_sample_size` before anything is fitted, and the
  ladder reports ``null_r2`` alongside every ``rho`` so the in-sample rise that
  is pure feature count is visible next to the rise that is not.

Knots are placed at quantiles of the feature column alone and never see the
latent codes, so the cross-validated ``rho`` the ladder reports is not inflated
by the basis construction.

Examples
--------
::

    df_feat, feat_to_group, spec = fva.expand_features(
        df_raw, mode="auto", spline_df=6, harmonics=3,
        circular={"orientation": 2 * np.pi},
        interactions=[("posX", "posY")],
    )
    print(spec)                      # the hypothesis class, column by column
    decomp = fva.FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)

    table, fig = fva.capacity_ladder(df_latent, df_raw, levels=(1, 2, 4, 6, 8))
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from pydantic import InstanceOf, PositiveInt, validate_call

from ._diagnostics import check_sample_size, null_r2
from ._validation import FigOut, FigSize, FloatArray, VALIDATE_CONFIG
from .encoding import (
    CIRCULAR_INFER_MAX_ABS,
    CIRCULAR_NAME_PATTERN,
    ENCODED_SEP,
    _infer_period,
    _is_string_column,
    _one_hot,
)

logger = logging.getLogger(__name__)

#: Highest B-spline degree used.  Cubic is the standard choice: the second
#: derivative is continuous, so the fitted function looks smooth, and going
#: higher buys wiggle rather than fit.  Narrow bases fall back to a lower degree
#: (see :func:`spline_basis`) rather than refusing to build.
MAX_SPLINE_DEGREE = 3

#: Capacities :func:`capacity_ladder` walks by default.  Starts at 1, which is
#: the *unexpanded* bank for both bases, so the first rung is the baseline every
#: other rung has to beat.
DEFAULT_LEVELS = (1, 2, 3, 4, 6, 8)

#: Expansions available for a non-periodic numeric column (see :func:`expand_features`).
BASES = ("none", "spline", "hurdle_spline", "fourier")

#: Suffix of the point-mass indicator column a hurdle spline emits.
HURDLE_INDICATOR = "nz"

#: Separator joining the two halves of an interaction column or group name.
#: Spaced so that a pair reads as a pair in a legend without needing a lookup,
#: and distinct from :data:`fva.encoding.ENCODED_SEP` so a name can be split
#: back into its parts.
INTERACTION_SEP = " x "


class ExpandedFeatures(NamedTuple):
    """Return value of :func:`expand_features`.

    Unpacks as three values, ``df_feat, feat_to_group, spec = ...``, rather than
    the two of :func:`fva.encode_features`.  The difference is deliberate: an
    expansion is a choice of hypothesis class, and ``spec`` is the record of the
    choice that was made.  A ``rho`` reported without it cannot be compared with
    anything.
    """

    frame: pd.DataFrame        # (n, q') all-numeric expanded feature matrix
    feat_to_group: dict        # expanded feature name -> group name
    spec: pd.DataFrame         # one row per source column: basis, width, reason


class Ladder(NamedTuple):
    """Return value of :func:`capacity_ladder`.

    Unpack as ``table, fig = fva.capacity_ladder(...)``.
    """

    table: pd.DataFrame        # one row per capacity level
    figure: object             # matplotlib Figure, or None if figure=False


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------
def _cox_de_boor(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate every B-spline basis function of ``degree`` on ``knots`` at ``x``.

    The textbook recursion: start from the degree-0 indicators of each knot
    interval and raise the degree one step at a time, each step combining
    adjacent lower-degree functions.  Returns ``(len(x), len(knots) - degree - 1)``
    columns, which sum to 1 at every ``x`` inside the boundary knots.
    """
    t = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    n_int = len(t) - 1

    # Degree 0: the indicator of each half-open knot interval.  Repeated knots
    # give empty intervals, which stay zero and drop out of the recursion.
    B = np.zeros((x.size, n_int))
    for i in range(n_int):
        if t[i + 1] > t[i]:
            B[:, i] = ((x >= t[i]) & (x < t[i + 1])).astype(float)
    # The half-open convention leaves the right boundary unassigned; put it in
    # the last non-empty interval so ``x == max`` is not silently all-zero.
    non_empty = [i for i in range(n_int) if t[i + 1] > t[i]]
    if non_empty:
        B[x >= t[non_empty[-1] + 1], non_empty[-1]] = 1.0

    for d in range(1, degree + 1):
        B_next = np.zeros((x.size, n_int - d))
        for i in range(n_int - d):
            left_den = t[i + d] - t[i]
            right_den = t[i + d + 1] - t[i + 1]
            col = np.zeros(x.size)
            if left_den > 0:
                col += (x - t[i]) / left_den * B[:, i]
            if right_den > 0:
                col += (t[i + d + 1] - x) / right_den * B[:, i + 1]
            B_next[:, i] = col
        B = B_next
    return B


@validate_call(config=VALIDATE_CONFIG)
def spline_basis(x: FloatArray, df: PositiveInt = 4, *,
                 name: str = "x") -> tuple[np.ndarray, list[str]]:
    """Cubic B-spline basis with ``df`` columns and knots at quantiles.

    The basis spans piecewise-cubic functions of ``x`` that are continuous in
    value, slope and curvature at the knots.  One basis function is dropped: the
    full set sums to 1 at every point and so is exactly collinear with the
    intercept once the features are centred, which is the same reason
    :func:`fva.encode_features` drops a one-hot reference level.

    ``df`` is the capacity knob and equals the number of columns returned.
    ``df=1`` is the identity: a degree-1 basis on the two boundary knots reduces
    to the raw column rescaled to ``[0, 1]``, so ``df=1`` reproduces the
    unexpanded bank rather than approximating it.  Above that, the degree rises
    to :data:`MAX_SPLINE_DEGREE` and the remaining budget goes into interior
    knots placed at equally spaced quantiles, so knot density follows the data
    rather than the range and a skewed column is not spent on its empty tail.

    Parameters
    ----------
    x : (n,) array_like
        The feature column.  Non-finite entries propagate as ``NaN`` rows, the
        same behaviour as the circular encoding.
    df : int, default 4
        Number of basis columns, i.e. degrees of freedom of the fitted function.
    name : str
        Prefix for the returned column names.

    Returns
    -------
    (basis, names)
        ``basis`` is ``(n, df)``; ``names`` are ``f"{name}=s1"`` ... in order.

    Notes
    -----
    Ties reduce the number of *distinct* quantile knots.  Duplicated knots are
    dropped rather than stacked, so a heavily tied column silently yields fewer
    interior knots than requested; the returned width is still ``df`` because the
    degree makes up the difference.
    """
    x = np.asarray(x, dtype=float).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError(f"column {name!r} has no finite values to build a basis on.")

    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        raise ValueError(
            f"column {name!r} is constant ({lo:g}); it carries no variance and "
            f"cannot support a basis expansion."
        )

    degree = min(MAX_SPLINE_DEGREE, int(df))
    n_interior = int(df) - degree

    interior: np.ndarray = np.empty(0)
    if n_interior > 0:
        qs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        interior = np.unique(np.quantile(finite, qs))
        interior = interior[(interior > lo) & (interior < hi)]
        if interior.size < n_interior:
            # Ties collapsed some knots.  Raise the degree to keep the promised
            # width instead of returning a narrower basis than the caller sized
            # their bank for.
            degree = int(df) - int(interior.size)

    knots = np.concatenate([
        np.full(degree + 1, lo), interior, np.full(degree + 1, hi),
    ])
    xc = np.clip(x, lo, hi)
    B = _cox_de_boor(xc, knots, degree)
    B[~np.isfinite(x)] = np.nan
    B = B[:, 1:]                     # drop the reference basis function

    if B.shape[1] != int(df):        # pragma: no cover - guarded by construction
        raise RuntimeError(
            f"spline basis for {name!r} produced {B.shape[1]} columns, expected {df}."
        )
    names = [f"{name}{ENCODED_SEP}s{i + 1}" for i in range(B.shape[1])]
    return B, names


@validate_call(config=VALIDATE_CONFIG)
def hurdle_spline_basis(x: FloatArray, df: PositiveInt = 4, *, at: float = 0.0,
                        name: str = "x") -> tuple[np.ndarray, list[str]]:
    """Hurdle basis: a point-mass indicator plus a spline off the point mass.

    Built for zero-inflated features such as log-normalized gene expression.
    The first column is ``1[x != at]``, which carries the jump between the point
    mass and the rest of the distribution.  The remaining ``df`` columns are the
    :func:`spline_basis` of the values off the point mass, with knots at their
    quantiles, and are zero on the point mass.  The spline's dropped reference
    function is what the indicator replaces, so the basis is full rank.

    The span contains ``x - at``, so an expanded column is never less expressive
    than the raw one.  ``df=1`` is the two-part linear model (a jump plus a
    slope), not the raw column, so a capacity ladder over a hurdle bank starts one
    step above the unexpanded bank.

    Parameters
    ----------
    x : (n,) array_like
        The feature column.  Non-finite entries propagate as ``NaN`` rows.
    df : int, default 4
        Columns of the spline part.  The basis has ``df + 1`` columns, or just
        the indicator when the values off the point mass are all equal.
    at : float, default 0.0
        The location of the point mass.
    name : str
        Prefix for the returned column names.

    Returns
    -------
    (basis, names)
        ``names`` are ``f"{name}=nz"`` followed by ``f"{name}=s1"`` ... .

    Raises
    ------
    ValueError
        If no finite value lies at ``at`` or none lies off it; a hurdle has
        nothing to separate.  :func:`expand_features` falls back to a plain
        spline in that case.
    """
    x = np.asarray(x, dtype=float).ravel()
    finite = np.isfinite(x)
    on = finite & (x == at)
    off = finite & (x != at)
    if not on.any() or not off.any():
        raise ValueError(
            f"column {name!r} has {int(on.sum())} value(s) at the point mass "
            f"{at:g} and {int(off.sum())} off it; a hurdle basis needs both."
        )

    ind = off.astype(float)
    ind[~finite] = np.nan
    cols, names = [ind[:, None]], [f"{name}{ENCODED_SEP}{HURDLE_INDICATOR}"]
    if np.unique(x[off]).size > 1:
        S = np.zeros((x.size, int(df)))
        S_off, snames = spline_basis(x[off], int(df), name=name)
        S[off] = S_off
        S[~finite] = np.nan
        cols.append(S)
        names += snames
    return np.column_stack(cols), names


@validate_call(config=VALIDATE_CONFIG)
def harmonic_basis(x: FloatArray, period: float, harmonics: PositiveInt = 1, *,
                   name: str = "x") -> tuple[np.ndarray, list[str]]:
    """Fourier basis ``sin/cos(2 pi k v / period)`` for ``k = 1..harmonics``.

    A truncated Fourier series on the circle: the fitted function is a
    trigonometric polynomial in the angle, bandlimited to ``harmonics``.  It
    wraps by construction, which is the property a periodic feature needs and the
    reason no raw column accompanies it --- a raw angle is discontinuous at the
    origin, and the origin is an arbitrary choice.

    ``harmonics=1`` is the pair :func:`fva.encode_features` emits, so it is the
    faithful encoding rather than an expansion; ``k >= 2`` is capacity.

    Parameters
    ----------
    x : (n,) array_like
    period : float
        The turn, in the units of the column: ``360`` for degrees, ``2*pi`` for
        radians, and *half* of those for an axial quantity that repeats every
        half turn.  See :mod:`fva.encoding`.
    harmonics : int, default 1
    name : str

    Returns
    -------
    ``(basis, names)`` with ``basis`` of shape ``(n, 2 * harmonics)``.
    """
    if not (np.isfinite(period) and period > 0):
        raise ValueError(f"period for {name!r} must be finite and positive; got {period!r}.")
    theta = 2.0 * np.pi * np.asarray(x, dtype=float).ravel() / period
    cols, names = [], []
    for k in range(1, int(harmonics) + 1):
        cols += [np.sin(k * theta), np.cos(k * theta)]
        names += [f"{name}{ENCODED_SEP}sin{k}", f"{name}{ENCODED_SEP}cos{k}"]
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------
def _max_supported_width(values: np.ndarray) -> int:
    """Columns a feature can support before the basis is rank deficient.

    A column taking ``u`` distinct values spans a ``u``-dimensional space of
    functions, one dimension of which is the constant the fit centres away, so
    ``u - 1`` columns is the ceiling.  dSprites scale, with six levels, tops out
    at five however many harmonics are asked for.
    """
    finite = values[np.isfinite(values)]
    return max(1, int(np.unique(finite).size) - 1)


def _clip_width(requested: int, values: np.ndarray, name: str, kind: str) -> int:
    """Clip a requested basis width to what the column can support, warning."""
    ceiling = _max_supported_width(values)
    if requested <= ceiling:
        return requested
    warnings.warn(
        f"{kind} expansion of {name!r} was asked for {requested} column(s) but "
        f"the column takes only {ceiling + 1} distinct values, which support at "
        f"most {ceiling} once centred; the extra columns would be exactly "
        f"collinear and make the per-feature Johnson split among them arbitrary. "
        f"Width clipped to {ceiling}.",
        stacklevel=4,
    )
    return ceiling


def _as_levels(s: pd.Series) -> np.ndarray:
    """The column as float codes, so distinct values can be counted either way.

    Numeric columns are already their own codes; a categorical is factorized,
    which is enough to count combinations even though the codes themselves carry
    no order.
    """
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=float)
    return pd.factorize(s)[0].astype(float)


def _joint_supported_width(a: np.ndarray, b: np.ndarray) -> int:
    """Columns the *pair* ``(a, b)`` can support, by its distinct joint values.

    The one-dimensional ceiling of :func:`_max_supported_width` applied to the
    cells of the joint table: a pair taking ``u`` distinct combinations spans a
    ``u``-dimensional function space, one dimension of which is the constant.
    """
    finite = np.isfinite(a) & np.isfinite(b)
    u = len(np.unique(np.stack([a[finite], b[finite]], axis=1), axis=0))
    return max(1, u - 1)


def _tensor_product(A: pd.DataFrame, B: pd.DataFrame, *,
                    name: str) -> pd.DataFrame:
    """All pairwise products of the columns of ``A`` and ``B``.

    Columns that come out exactly constant are dropped: a product column is zero
    everywhere when no observation falls in the cell its two basis functions
    jointly support, which is a property of the sample rather than of the
    hypothesis class, and leaving it in makes the design matrix exactly rank
    deficient for no gain.
    """
    cols, names = [], []
    for ca in A.columns:
        for cb in B.columns:
            cols.append(A[ca].to_numpy() * B[cb].to_numpy())
            names.append(f"{ca}{INTERACTION_SEP}{cb}")
    M = np.column_stack(cols) if cols else np.empty((len(A), 0))
    keep = M.std(axis=0) > 0
    if not keep.all():
        warnings.warn(
            f"interaction {name!r} produced {int((~keep).sum())} constant "
            f"column(s) of {M.shape[1]}: no observation falls in the cell those "
            f"products support. They are dropped; the remaining basis is "
            f"unaffected.",
            stacklevel=3,
        )
    return pd.DataFrame(M[:, keep], columns=[n for n, k in zip(names, keep) if k],
                        index=A.index)


@validate_call(config=VALIDATE_CONFIG)
def expand_features(
    df: InstanceOf[pd.DataFrame],
    *,
    mode: Literal["auto", "spline", "circular"] = "auto",
    basis: Literal["none", "spline", "hurdle_spline", "fourier"] = "spline",
    spline_df: PositiveInt = 4,
    harmonics: PositiveInt = 1,
    hurdle_at: float = 0.0,
    interactions: list[tuple[str, str]] | None = None,
    categorical: list[str] | None = None,
    circular: dict[str, float] | list[str] | None = None,
    groups: dict | None = None,
    infer: bool = True,
    drop_first: bool = True,
) -> ExpandedFeatures:
    """Expand continuous features onto a basis, widening the hypothesis class.

    This is :func:`fva.encode_features` plus capacity.  Categorical columns are
    one-hot encoded exactly as there; what changes is that a continuous column no
    longer enters as itself but as a basis of ``spline_df`` (or ``2 * harmonics``)
    named functions of it, pooled into one group.

    Read the returned ``rho`` against the ``spec`` and against
    :func:`capacity_ladder`, never on its own: a wider bank explains more of
    every latent space, informative or not, and the noise remainder is only
    "irreducible" relative to a stated hypothesis class.

    Parameters
    ----------
    df : DataFrame, shape (n, q)
        Raw features, one column per feature.
    mode : {"auto", "spline", "circular"}, default "auto"
        ``"auto"`` picks per column: one-hot for categoricals, harmonics for
        circular columns, ``basis`` for everything else numeric.  ``"spline"``
        applies ``basis`` to every numeric column, circular ones included, which
        gives up the wrap those columns need.  ``"circular"`` forces harmonics
        onto every numeric column, imposing a period equal to the observed range
        for the ones that are not periodic and carrying the raw column alongside
        as the ramp that lets the fit break the imposed wrap-around.  Both forced
        modes warn on the columns they mis-handle; they exist so the choice of
        basis can be ablated rather than asserted.
    basis : {"none", "spline", "hurdle_spline", "fourier"}, default "spline"
        Expansion of the non-periodic numeric columns.  ``"none"`` keeps the raw
        column.  ``"spline"`` is a cubic B-spline of ``spline_df`` columns.
        ``"hurdle_spline"`` is an indicator of ``x != hurdle_at`` plus a cubic
        B-spline of ``spline_df`` columns fitted to the values off the point
        mass (see :func:`hurdle_spline_basis`); use it for zero-inflated features
        such as gene expression.  A column with no values at ``hurdle_at`` falls
        back to ``"spline"`` with a warning.  ``"fourier"`` is the same as
        ``mode="circular"``.
    spline_df : int, default 4
        Columns per spline-expanded feature (the spline part of a hurdle
        spline).  ``1`` is the raw column for ``"spline"``.
    harmonics : int, default 1
        Harmonics per circular feature; ``2 * harmonics`` columns.  ``1`` is the
        faithful ``sin``/``cos`` encoding.
    hurdle_at : float, default 0.0
        Location of the point mass for ``basis="hurdle_spline"``.
    interactions : list of (str, str), optional
        Pairs of *source* column names whose bases are multiplied together and
        added as one further group per pair, named ``"a x b"``.  The width is the
        product of the two bases' widths, so it grows quickly; name the pairs the
        features imply rather than enumerating them.  See the module docstring
        for what the tensor product spans and why it is collinear with the two
        main-effect groups.
    categorical, circular, groups, infer, drop_first
        As in :func:`fva.encode_features`.  ``circular`` given as a dict fixes
        the period per column; as a list, the period is inferred.

    Returns
    -------
    ExpandedFeatures
        ``(frame, feat_to_group, spec)``.  ``spec`` has one row per source
        column, giving the basis used, the width requested, the width emitted,
        and the number of distinct values available to support it.

    Raises
    ------
    ValueError
        If a named column is missing, named twice, non-numeric and unencoded, or
        constant; or if an interaction names a column that was not expanded.

    Warns
    -----
    UserWarning
        When a requested width exceeds what a column's distinct values support,
        when a forced mode applies the wrong basis to a column, when an
        interaction is wider than its pair's distinct combinations support or
        produces empty cells, and --- through :mod:`fva._diagnostics` --- when
        the expanded bank is too wide for the sample.

    See Also
    --------
    fva.encode_features : the faithful encoding, with no capacity choice.
    capacity_ladder : sweep the capacity instead of fixing it.
    """
    cat_cols = list(categorical or [])
    if isinstance(circular, dict):
        circ_cols: dict[str, float | None] = {k: float(v) for k, v in circular.items()}
    else:
        circ_cols = {c: None for c in (circular or [])}

    missing = [c for c in [*cat_cols, *circ_cols] if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) named for expansion are not in df: {missing}. "
            f"Available columns: {list(df.columns)[:10]}"
            + (" ..." if df.shape[1] > 10 else "")
        )
    both = sorted(set(cat_cols) & set(circ_cols))
    if both:
        raise ValueError(f"column(s) named as both categorical and circular: {both}.")

    pairs: list[tuple[str, str]] = []
    for pair in interactions or []:
        a, b = (str(c) for c in pair)
        for c in (a, b):
            if c not in df.columns:
                raise ValueError(
                    f"interaction ({a!r}, {b!r}) names {c!r}, which is not a "
                    f"column of df. Available columns: {list(df.columns)[:10]}"
                    + (" ..." if df.shape[1] > 10 else "")
                )
        if a == b:
            raise ValueError(
                f"interaction ({a!r}, {b!r}) pairs a column with itself; the "
                f"square of a basis adds nothing the basis does not already span."
            )
        if (b, a) in pairs or (a, b) in pairs:
            raise ValueError(f"interaction ({a!r}, {b!r}) is named twice.")
        pairs.append((a, b))

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

    pieces: list[pd.DataFrame] = []
    expansion: dict[str, list[str]] = {}
    part_of: dict[str, pd.DataFrame] = {}
    spec_rows: list[dict] = []

    for col in df.columns:
        s = df[col]
        name = str(col)
        if col in cat_cols:
            part = _one_hot(s, name, drop_first=drop_first,
                            inferred=col not in explicit)
            spec_rows.append({"column": name, "basis": "one-hot",
                              "requested": part.shape[1], "emitted": part.shape[1],
                              "distinct": int(s.nunique(dropna=True))})
        else:
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                raise ValueError(
                    f"column {name!r} is not numeric and was not encoded. Pass it "
                    f"as categorical=[{name!r}] if it is a feature, or drop it."
                )
            values = s.to_numpy(dtype=float)
            is_circular = col in circ_cols
            if mode == "circular" or (mode == "auto" and is_circular):
                use = "fourier"
            else:
                use = basis
                if is_circular and use != "fourier":
                    warnings.warn(
                        f"mode='spline' expands the circular column {name!r} on "
                        f"basis={basis!r}, which does not wrap: the fit sees the "
                        f"two ends of the turn as far apart when they are "
                        f"adjacent. Use mode='auto' to give it harmonics.",
                        stacklevel=3,
                    )
            if use == "hurdle_spline" and not np.any(values == hurdle_at):
                warnings.warn(
                    f"basis='hurdle_spline' found no values of {name!r} at the "
                    f"point mass {hurdle_at:g}, so there is no hurdle to separate; "
                    f"it gets a plain spline instead.",
                    stacklevel=3,
                )
                use = "spline"

            if use == "none":
                finite = values[np.isfinite(values)]
                part = pd.DataFrame({name: values}, index=s.index)
                spec_rows.append({"column": name, "basis": "identity",
                                  "requested": 1, "emitted": 1,
                                  "distinct": int(np.unique(finite).size)})
            elif use == "hurdle_spline":
                finite = values[np.isfinite(values)]
                off_vals = finite[finite != hurdle_at]
                n_off = int(np.unique(off_vals).size)
                d = (_clip_width(int(spline_df), off_vals, name, "hurdle spline")
                     if n_off > 1 else int(spline_df))
                B, bnames = hurdle_spline_basis(values, d, at=hurdle_at, name=name)
                part = pd.DataFrame(B, columns=bnames, index=s.index)
                spec_rows.append({"column": name,
                                  "basis": f"hurdle_bspline(at={hurdle_at:g}, df={d}, "
                                           f"zero_frac={np.mean(finite == hurdle_at):.2f})",
                                  "requested": int(spline_df) + 1,
                                  "emitted": part.shape[1],
                                  "distinct": int(np.unique(finite).size)})
            elif use == "fourier":
                if is_circular:
                    period = circ_cols[col]
                    if period is None:
                        period, _ = _infer_period(values, name)
                else:
                    finite = values[np.isfinite(values)]
                    span = float(finite.max() - finite.min())
                    if span <= 0:
                        raise ValueError(
                            f"column {name!r} is constant and cannot be expanded."
                        )
                    # An observed range is not a period.  Say so: the harmonics
                    # will tie the two ends of the range together, and the raw
                    # column below is the only thing that can pull them apart.
                    period = span
                    warnings.warn(
                        f"the Fourier basis expands the non-periodic column {name!r} "
                        f"with the period set to its observed "
                        f"range ({span:.4g}), which forces the fitted function to "
                        f"take the same value at both ends. The raw column is "
                        f"carried alongside to offset that. Use basis='spline' for "
                        f"a spline, which has no such constraint.",
                        stacklevel=3,
                    )
                # Clip on the width actually emitted (two columns per harmonic,
                # plus the ramp when the period was imposed), not on ``harmonics``.
                n_ramp = 0 if is_circular else 1
                want = 2 * int(harmonics) + n_ramp
                allowed = _clip_width(want, values, name, "harmonic")
                k = int(harmonics) if allowed >= want else max(1, (allowed - n_ramp) // 2)
                B, bnames = harmonic_basis(values, period, k, name=name)
                if not is_circular:
                    B = np.column_stack([values, B])
                    bnames = [f"{name}{ENCODED_SEP}raw", *bnames]
                part = pd.DataFrame(B, columns=bnames, index=s.index)
                spec_rows.append({"column": name,
                                  "basis": f"fourier(k={k}, period={period:.4g})",
                                  "requested": int(harmonics), "emitted": part.shape[1],
                                  "distinct": int(np.unique(
                                      values[np.isfinite(values)]).size)})
            else:
                d = _clip_width(int(spline_df), values, name, "spline")
                B, bnames = spline_basis(values, d, name=name)
                part = pd.DataFrame(B, columns=bnames, index=s.index)
                spec_rows.append({"column": name, "basis": f"bspline(df={d})",
                                  "requested": int(spline_df), "emitted": part.shape[1],
                                  "distinct": int(np.unique(
                                      values[np.isfinite(values)]).size)})

        if part.shape[1]:
            pieces.append(part)
            expansion[col] = list(part.columns)
            part_of[str(col)] = part

    # Interactions come after every main effect, so a pair always multiplies the
    # bases the same call emitted rather than whatever the caller expanded last.
    for a, b in pairs:
        missing = [c for c in (a, b) if c not in part_of]
        if missing:
            raise ValueError(
                f"interaction ({a!r}, {b!r}) names {missing}, which produced no "
                f"columns to multiply (a dropped one-hot level, or a column left "
                f"out by `groups`)."
            )
        name = f"{a}{INTERACTION_SEP}{b}"
        A, B = part_of[a], part_of[b]
        want = A.shape[1] * B.shape[1]
        ceiling = _joint_supported_width(_as_levels(df[a]), _as_levels(df[b]))
        if want > ceiling:
            warnings.warn(
                f"interaction {name!r} asks for {want} columns but the pair takes "
                f"only {ceiling + 1} distinct combinations, which support at most "
                f"{ceiling} once centred, so some products are exactly collinear. "
                f"The width is not clipped --- unlike a one-dimensional basis a "
                f"tensor product has no canonical subset to drop --- so narrow the "
                f"per-column bases instead if the Johnson split within the pair "
                f"matters.",
                stacklevel=3,
            )
        part = _tensor_product(A, B, name=name)
        if part.shape[1]:
            pieces.append(part)
            expansion[name] = list(part.columns)
        spec_rows.append({"column": name, "basis": f"tensor({a}, {b})",
                          "requested": want, "emitted": part.shape[1],
                          "distinct": ceiling + 1})

    out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)

    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"expansion produced duplicate column name(s): {sorted(set(duplicated))}. "
            f"Rename the source column(s)."
        )

    if groups is None:
        feat_to_group = {n: str(src) for src, names in expansion.items() for n in names}
    else:
        # A pair keyed explicitly wins; otherwise it takes the names of the two
        # groups it was built from, so an interaction is never silently dropped
        # from the budget just because the caller mapped only source columns.
        groups = dict(groups)
        for a, b in pairs:
            key = f"{a}{INTERACTION_SEP}{b}"
            if key not in groups and a in groups and b in groups:
                groups[key] = f"{groups[a]}{INTERACTION_SEP}{groups[b]}"
        feat_to_group = {n: groups[src] for src, names in expansion.items()
                         if src in groups for n in names}

    check_sample_size(len(out), out.shape[1])
    spec = pd.DataFrame(spec_rows).set_index("column")
    logger.info("expand_features (mode=%s): %d column(s) in, %d out, %d group(s).",
                mode, df.shape[1], out.shape[1], len(set(feat_to_group.values())))
    return ExpandedFeatures(frame=out, feat_to_group=feat_to_group, spec=spec)


# ---------------------------------------------------------------------------
# Capacity ladder
# ---------------------------------------------------------------------------
@validate_call(config=VALIDATE_CONFIG)
def capacity_ladder(
    df_latent: InstanceOf[pd.DataFrame],
    df_raw: InstanceOf[pd.DataFrame],
    *,
    levels: list[int] | tuple[int, ...] = DEFAULT_LEVELS,
    mode: Literal["auto", "spline", "circular"] = "auto",
    basis: Literal["none", "spline", "hurdle_spline", "fourier"] = "spline",
    hurdle_at: float = 0.0,
    interactions: list[tuple[str, str]] | None = None,
    categorical: list[str] | None = None,
    circular: dict[str, float] | list[str] | None = None,
    groups: dict | None = None,
    infer: bool = True,
    n_directions: PositiveInt | None = None,
    cv_folds: int | float = 5,
    seed: int = 0,
    progress: bool | None = None,
    figure: bool = True,
    figsize: FigSize = (7.0, 4.0),
    fig_out: FigOut = None,
) -> Ladder:
    """Report ``rho`` as a function of the capacity of the feature vocabulary.

    Runs the decomposition once per rung of ``levels``, each rung setting both
    ``spline_df`` and ``harmonics`` to that value, and tabulates the explained
    fraction against the width of the bank.  Level 1 is the unexpanded bank, so
    the first row is the baseline and every later row is what widening the
    hypothesis class bought.

    The point of reporting the whole curve rather than one number is that the
    two ways ``rho`` can rise look identical in a single fit and different here.
    A curve that climbs steeply and then flattens says the feature really does
    act nonlinearly on the latent space and the plateau is a property of the
    feature; a curve that keeps climbing in step with ``null_r2`` says the bank
    is just getting wide. The gap between ``rho`` and ``rho_cv`` separates them
    directly, and both are tabulated.

    Parameters
    ----------
    df_latent : DataFrame, shape (n, d)
        Latent codes, as for :class:`~fva.explainer.FeatureGroundedDecomposition`.
    df_raw : DataFrame, shape (n, q)
        Raw (unexpanded) features.
    levels : sequence of int, default :data:`DEFAULT_LEVELS`
        Capacities to walk, in order.
    mode, basis, hurdle_at, interactions, categorical, circular, groups, infer
        Passed to :func:`expand_features` at every rung.  An interaction is held
        fixed while the per-column capacity varies, so each rung is the same
        pairs at the width that rung sets --- the ladder then reads as one
        hypothesis class widening, not two.
    n_directions, cv_folds, seed, progress
        Passed to :class:`~fva.explainer.FeatureGroundedDecomposition`.
    figure : bool, default True
        Draw the ladder.  ``False`` returns ``None`` in its place.
    figsize, fig_out
        Figure size, and the usual three-way saving convention.

    Returns
    -------
    Ladder
        ``(table, figure)``.  ``table`` is indexed by level with columns
        ``n_features``, ``rho``, ``rho_cv``, ``null_r2`` and ``d_rho_cv`` (the
        change in ``rho_cv`` from the previous rung).

    Notes
    -----
    Cost is one full decomposition per rung, and each decomposition is itself
    exponential in the group count, so a long ``levels`` on a many-group bank is
    expensive.  The group count does not change along the ladder --- an expansion
    stays pooled in its source column's group --- so the cost is linear in the
    number of rungs.
    """
    from .explainer import FeatureGroundedDecomposition

    if not len(levels):
        raise ValueError("levels must contain at least one capacity.")
    if any(int(v) < 1 for v in levels):
        raise ValueError(f"levels must all be >= 1; got {list(levels)}.")

    rows = []
    for level in levels:
        with warnings.catch_warnings():
            # Every rung re-raises the same width warnings; they are the ladder's
            # subject matter and are reported in the table, so only surface them
            # once, from the widest rung that triggers them.
            warnings.simplefilter("ignore", UserWarning)
            frame, feat_to_group, _ = expand_features(
                df_raw, mode=mode, basis=basis, hurdle_at=hurdle_at,
                spline_df=int(level), harmonics=int(level), interactions=interactions, categorical=categorical,
                circular=circular, groups=groups, infer=infer,
            )
            decomp = FeatureGroundedDecomposition(
                df_latent, frame, feat_to_group, n_directions=n_directions,
                cv_folds=cv_folds, seed=seed, progress=progress,
            )
            rho, rho_cv = decomp.rho, decomp.rho_cv
        q = frame.shape[1]
        rows.append({"level": int(level), "n_features": q,
                     "rho": rho, "rho_cv": rho_cv,
                     "null_r2": null_r2(len(frame), q)})
        logger.info("capacity level %d: q=%d, rho=%.4f, rho_cv=%.4f",
                    level, q, rho, rho_cv)

    table = pd.DataFrame(rows).set_index("level")
    table["d_rho_cv"] = table["rho_cv"].diff()

    fig = _plot_ladder(table, figsize=figsize) if figure else None
    if fig is not None:
        from .viz import save_figure
        save_figure(fig, fig_out, "capacity_ladder")
    return Ladder(table=table, figure=fig)


def _plot_ladder(table: pd.DataFrame, *, figsize=(7.0, 4.0)):
    """Draw the ladder: both totals against capacity, over the noise floor."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    x = table.index.to_numpy()
    ax.plot(x, table["rho"], "o-", color="#4C72B0", label=r"in-sample $\rho$")
    ax.plot(x, table["rho_cv"], "s-", color="#C44E52",
            label=r"cross-validated $\rho$")
    ax.plot(x, table["null_r2"], "--", color="#333333", linewidth=1.2,
            label=r"noise floor $q/(n-1)$")
    ax.set_xlabel("basis capacity per feature (spline df / harmonics)")
    ax.set_ylabel(r"fraction of latent variance explained ($\rho$)")
    ax.set_ylim(0, 1)
    # The bank width belongs on the tick, not floating in the axes: it collides
    # with the legend wherever the curve happens to leave room.
    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi}\nq={q}" for xi, q in zip(x, table["n_features"])],
                       fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
