"""Interpretable directions in latent spaces via feature-grounded variance decomposition.

Validation
----------
The public entry points --- :class:`FeatureGroundedDecomposition` and its report
methods, and the functions in :mod:`fva.decomposition` and :mod:`fva.viz` ---
are wrapped in :func:`pydantic.validate_call`, so a bad argument raises a
:class:`pydantic.ValidationError` (a subclass of :exc:`ValueError`) naming the
parameter at the call site rather than failing later inside the linear algebra.
Array arguments still accept anything :func:`numpy.asarray` accepts --- lists,
DataFrames, other arrays --- and are coerced at the boundary.

Encoding
--------
The fit is linear in the features, so a column whose raw numbers do not mean
what a line would read into them --- a categorical, an angle --- has to be
re-represented before it can be explained.  :func:`fva.encode_features` does
that, returning the feature matrix *and* the ``feature -> group`` mapping that
keeps each encoding's columns pooled into one budget entry::

    df_feat, feat_to_group = fva.encode_features(df_raw)
    decomp = fva.FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)

String columns and ``angle``/``rotation`` columns are claimed automatically;
integer-coded categoricals and axial (half-turn) angles have to be named, since
neither is decidable from the numbers.  See :mod:`fva.encoding`.

Basis expansion
---------------
Those encodings are faithful: they add no explanatory power a correctly-read
column did not already have.  When a feature acts on the latent space smoothly
but nonlinearly, a linear fit on the raw column reports that dependence as noise,
and widening the hypothesis class is the remedy --- but it is a *choice*, and a
``rho`` from a wider bank is not comparable with one from a narrower bank.
:mod:`fva.basis` keeps that choice explicit and separate from
:func:`encode_features`: splines for non-periodic continuous features, harmonics
for circular ones, and :func:`capacity_ladder` to report ``rho`` as a function of
the capacity rather than at one hidden setting::

    table, fig = fva.capacity_ladder(df_latent, df_raw, levels=(1, 2, 4, 6, 8))

Screening
---------
A budget answers which families explain a direction, which is worth asking only
once something explains it.  :func:`fva.direction_r2` reports just the totals ---
the ``R^2`` of the whole bank on each direction and its variance-weighted sum ---
skipping the ``2**m`` Shapley coalitions and the per-feature relative weights that
dominate the cost of a full fit::

    table = fva.direction_r2(df_latent, df_feat)
    table["r2_cv"]      # per direction, plus a "weighted_sum" entry == rho_cv

Read the ``r2_cv`` column rather than ``r2``: a bank of pure noise scores about
zero out of sample whatever its width, whereas its in-sample ``R^2`` sits at the
``q / (n - 1)`` floor described under Diagnostics below.  The same numbers come
off a fitted :class:`FeatureGroundedDecomposition` as its :attr:`~fva.explainer.
FeatureGroundedDecomposition.r2` and :attr:`~fva.explainer.
FeatureGroundedDecomposition.r2_cv` vectors, so screening first costs nothing
that the full fit would not have spent anyway.

Diagnostics
-----------
The decomposition warns when the data cannot support the feature bank it is
being asked to explain: too few samples for the number of features, an in-sample
total that sits far above the cross-validated one, a rank-deficient feature
correlation matrix, or a fold count too large for the sample.  These are
:class:`UserWarning`\\ s on the standard :mod:`warnings` machinery, so they can
be promoted to errors with ``warnings.simplefilter("error")`` or silenced per
call site in the usual way.  The thresholds live in :mod:`fva._diagnostics`.

One of them is also drawn rather than raised: the budget bars mark the in-sample
``R^2`` a pure-noise bank of the same width would score (``q / (n - 1)``) with a
dashed noise floor, so that a total inflated by feature count is visible in the
figure and not only in a warning.  It appears once that floor clears
``NULL_R2_LINE_MIN``; ``null_line=True``/``False`` on the report methods forces
it on or off.  See :func:`fva.viz.draw_null_floor`.

Progress
--------
The ``2**m`` Shapley coalitions make the cost of a budget exponential in the
number of feature families, so the long loops report progress through ``tqdm``.
A bar appears only for loops long enough to be worth watching; pass
``progress=True``/``False`` to :class:`FeatureGroundedDecomposition` (or to
:func:`fva.decomposition.decompose_latent_space`) to decide per call, or set it
for the whole session::

    fva.set_progress(False)     # never show bars
    fva.set_progress(None)      # restore the default heuristic

Logging
-------
``fva`` logs through the standard :mod:`logging` module under the ``"fva"``
logger (each submodule uses its own ``logging.getLogger(__name__)`` child).
Following library convention, the package attaches a :class:`logging.NullHandler`
so it stays *silent by default* and never configures handlers or levels on the
application's behalf.

To see what the library is doing, configure logging in your own code::

    import logging
    logging.basicConfig(level=logging.INFO)      # INFO: high-level progress
    # logging.getLogger("fva").setLevel(logging.DEBUG)  # DEBUG: per-direction detail

INFO-level messages report coarse progress (shape of the inputs, number of
directions analysed, global R^2). DEBUG-level messages are emitted inside the
hot loops (per-direction R^2, bootstrap resampling) and can be verbose.
"""

import logging

from . import (  # noqa: F401
    basis,
    decomposition,
    encoding,
    explainer,
    viz,
)
from ._progress import set_progress  # noqa: F401
from .basis import capacity_ladder, expand_features  # noqa: F401
from .decomposition import direction_r2, label_ceiling  # noqa: F401
from .encoding import encode_features  # noqa: F401
from .explainer import FeatureGroundedDecomposition  # noqa: F401

__version__ = "0.1.0"

__all__ = ["basis", "decomposition", "encoding", "explainer", "viz",
           "FeatureGroundedDecomposition", "encode_features", "expand_features",
           "capacity_ladder", "direction_r2", "label_ceiling", "set_progress"]

# Silence-by-default: a library must not configure logging output itself.
# Applications opt in via logging.basicConfig() / getLogger("fva").setLevel(...).
logging.getLogger(__name__).addHandler(logging.NullHandler())
