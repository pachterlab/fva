r"""Feature-grounded variance decomposition of latent directions.

This module implements the paper's core method.  Given

* latent codes ``Z`` (n x d) from a trained encoder, and
* a bank of interpretable features ``F`` (n x p),

we ask, for every principal direction of the latent space, *what fraction of
its variance is explained by the interpretable features, and how is that share
distributed across feature families?*  The unexplained remainder is the
"irreducible" part of the direction --- structure the encoder represents that
our vocabulary of radiomic features cannot name.

Method
------
1.  **Directions.**  Eigendecompose the latent covariance
    ``C = (1/n) Z^T Z = V diag(lambda) V^T``.  The k-th direction is the
    eigenvector ``v_k`` with variance ``lambda_k``; the projected score is
    ``s_k = Z v_k`` with ``Var(s_k) = lambda_k``.  (Raw latent axes can be used
    instead via ``mode="axes"``.)

2.  **Explained variance.**  Regress each score on the (standardised) features,
    ``s_k = F beta_k + eps_k``.  The coefficient of determination ``R^2_k`` is
    the fraction of the direction's variance the features explain; ``1 - R^2_k``
    is the noise/unexplained fraction.  We report both the in-sample ``R^2`` and
    an honest K-fold cross-validated ``R^2`` (which cannot be inflated by the
    large feature count).

3.  **Attribution.**  Radiomic features are highly collinear, so raw regression
    weights do not give a meaningful variance split.  We use two principled
    additive decompositions of ``R^2`` that *sum back to* ``R^2``:

    * **Group Shapley** -- treat the feature *families* (shape, curvature,
      intensity, texture, ...) as players in a cooperative game whose value is
      the regression ``R^2``.  The Shapley value of each family is its average
      marginal contribution over all orderings.  With a handful of families the
      game is solved exactly.  This yields a per-direction "budget":
      ``Var = shape% + curvature% + ... + noise%``.

    * **Johnson's relative weights** (Johnson, 2000) -- a per-*feature*
      attribution that is exact, non-negative and collinearity-aware, obtained
      by regressing the score on the orthonormal basis closest to ``F``.

4.  **Global summary.**  Weighting directions by their variance ``lambda_k``
    gives the fraction of the *whole* latent space explained by features,
    ``rho = sum_k lambda_k R^2_k / sum_k lambda_k``, and an analogous per-family
    breakdown of the total latent variance budget.

All regression ``R^2`` values for arbitrary feature subsets are computed from a
single precomputed Gram matrix, so the exact group-Shapley game (``2^G``
coalitions) is cheap even with bootstrap resampling.

References
----------
Lindeman, Merenda & Gold (1980); Gromping (2007), *relaimpo*; Johnson (2000),
*A heuristic method for estimating the relative weight of predictor variables*.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from typing import Literal

import numpy as np
from pydantic import InstanceOf, NonNegativeInt, PositiveInt, validate_call

from ._diagnostics import (
    check_cell_count,
    check_overfit_gap,
    check_sample_size,
    null_r2,
)
from ._progress import progress_bar
from ._validation import VALIDATE_CONFIG, FloatArray, IndexArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linear-algebra core: R^2 of any feature subset from a shared Gram matrix
# ---------------------------------------------------------------------------
@dataclass
class _GramCache:
    """Precomputed quantities for fast subset-R^2 evaluation.

    Features ``X`` are standardised (zero mean, unit variance) and scores are
    centred, so no intercept is needed.
    """

    G: np.ndarray          # X^T X  (p x p)
    ridge: float           # tiny diagonal load for numerical stability
    p: int

    @classmethod
    def build(cls, X: np.ndarray, ridge: float = 1e-6) -> "_GramCache":
        G = X.T @ X
        return cls(G=G, ridge=ridge * np.trace(G) / max(X.shape[1], 1), p=X.shape[1])

    def r2(self, c: np.ndarray, syy: float, idx: np.ndarray) -> float:
        """R^2 of regressing a score on the feature columns in ``idx``.

        ``c = X^T s`` (length p), ``syy = s^T s`` (s centred).
        """
        if len(idx) == 0 or syy <= 0:
            return 0.0
        Gss = self.G[np.ix_(idx, idx)].copy()
        Gss[np.diag_indices_from(Gss)] += self.ridge
        cs = c[idx]
        try:
            beta = np.linalg.solve(Gss, cs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Gss, cs, rcond=None)[0]
        explained = float(cs @ beta)
        return max(0.0, min(1.0, explained / syy))


# ---------------------------------------------------------------------------
# Group (family) Shapley decomposition of R^2
# ---------------------------------------------------------------------------
def _shapley_groups(cache: _GramCache, c: np.ndarray, syy: float,
                    group_cols: list[np.ndarray], *,
                    progress: bool | None = None) -> np.ndarray:
    """Exact Shapley value of each feature group; entries sum to full-model R^2."""
    G = len(group_cols)
    players = list(range(G))
    # The sum below visits every one of the 2^G coalitions, so they are evaluated
    # up front in a single pass -- same work as filling the cache lazily, but it
    # is a countable loop and can therefore carry a progress bar.
    subsets = (S for r in range(G + 1) for S in combinations(players, r))
    r2_cache: dict[frozenset, float] = {}
    for S in progress_bar(subsets, total=2 ** G, desc=f"coalitions (m={G})",
                          enabled=progress):
        if not S:
            r2_cache[frozenset(S)] = 0.0
        else:
            idx = np.concatenate([group_cols[g] for g in S])
            r2_cache[frozenset(S)] = cache.r2(c, syy, idx)

    phi = np.zeros(G)
    for g in players:
        others = [q for q in players if q != g]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(G - r - 1) / factorial(G)
            for S in combinations(others, r):
                phi[g] += w * (r2_cache[frozenset(S + (g,))] - r2_cache[frozenset(S)])
    return phi


def _group_alone(cache: _GramCache, c: np.ndarray, syy: float,
                 group_cols: list[np.ndarray]) -> np.ndarray:
    """R^2 of each feature group fitted on its own, ignoring the other groups.

    The one-player coalition of the Shapley game.  Unlike a Shapley share it is
    not part of an additive budget --- collinear groups double-count the variance
    they share, so these do not sum to the full-model R^2 --- but it is the
    quantity a label ceiling bounds, since the bound is a statement about what
    one family can do by itself.
    """
    return np.array([cache.r2(c, syy, idx) for idx in group_cols])


# ---------------------------------------------------------------------------
# Johnson's relative weights (per-feature attribution)
# ---------------------------------------------------------------------------
@validate_call(config=VALIDATE_CONFIG)
def relative_weights(X: FloatArray, y: FloatArray) -> tuple[np.ndarray, float]:
    """Johnson (2000) relative weights for standardised X and centred y.

    Returns ``(weights, r2)`` where ``weights`` are non-negative, length p, and
    ``weights.sum() == r2`` (up to numerical error).
    """
    n = X.shape[0]
    # Standardise the response to unit variance (X columns are already
    # standardised by the caller).
    yc = y - y.mean()
    ystd = yc.std()
    if ystd < 1e-12:
        return np.zeros(X.shape[1]), 0.0
    ys = yc / ystd
    # Correlation structure of predictors.
    Rxx = (X.T @ X) / n
    w, V = np.linalg.eigh(Rxx)
    # Johnson's construction needs Rxx positive definite: it forms the closest
    # orthonormal basis as X V diag(w^-1/2) V^T, which is only defined when every
    # eigenvalue is strictly positive.  Exact collinearity (a duplicated feature,
    # a family that sums to another) or p > n makes Rxx singular, the clip below
    # binds, and 1/sqrt(w) reaches 1e5 -- the weights still sum to R^2, because
    # that identity is algebraic, but they are distributed by numerical noise
    # among the offending columns rather than by the data.
    tol = float(w.max()) * max(Rxx.shape) * np.finfo(float).eps if w.size else 0.0
    n_deficient = int((w < tol).sum())
    if n_deficient:
        warnings.warn(
            f"the feature correlation matrix is rank deficient ({n_deficient} of "
            f"{w.size} eigenvalues below tolerance; {X.shape[0]} samples for "
            f"{X.shape[1]} features): Johnson's relative weights need it to be "
            f"positive definite, so the per-feature split among the collinear "
            f"columns is numerically arbitrary. The group-Shapley budget is "
            f"unaffected. Drop duplicate features or pool them into a family.",
            stacklevel=2,
        )
    w = np.clip(w, 1e-10, None)
    # Z* = closest set of orthonormal predictors to X; X = Z* Lambda with the
    # symmetric Lambda = V diag(sqrt(w)) V^T.  Columns of Z* have unit variance.
    Lambda = V @ np.diag(np.sqrt(w)) @ V.T
    Zstar = X @ (V @ np.diag(1.0 / np.sqrt(w)) @ V.T)
    # Orthonormal predictors -> standardised regression weights are correlations.
    beta_star = (Zstar.T @ ys) / n
    # Column sums of Lambda^2 are 1, so eps sums to R^2.
    eps = (Lambda ** 2) @ (beta_star ** 2)
    r2 = float(eps.sum())
    return eps, r2


# ---------------------------------------------------------------------------
# Cross-validated R^2 (honest explained variance)
# ---------------------------------------------------------------------------
_RIDGE_GRID = np.logspace(-2, 5, 10)


def _test_sets(n: int, folds: int | float, seed: int) -> list[np.ndarray]:
    """Held-out index arrays for a splitting policy.

    ``folds >= 2`` (an integer) is K-fold: every sample is held out exactly once,
    so the score is computed from n out-of-fold predictions. A float in ``(0, 1)``
    is a single random hold-out of that fraction, which scores only that fraction
    of the data and is correspondingly noisier.
    """
    if n < 2:
        raise ValueError(
            f"cross-validation needs at least 2 samples, got {n}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    if isinstance(folds, float) and not float(folds).is_integer():
        if not 0.0 < folds < 1.0:
            raise ValueError(
                f"a fractional hold-out must lie in (0, 1), got {folds!r}")
        n_test = max(1, int(round(folds * n)))
        if n_test >= n:
            raise ValueError(
                f"a hold-out fraction of {folds} on {n} samples leaves no "
                f"training data; use a smaller fraction or more samples.")
        return [order[:n_test]]
    folds = int(folds)
    if folds < 2:
        raise ValueError(
            f"folds must be an integer >= 2 (K-fold) or a float in (0, 1) "
            f"(single hold-out), got {folds!r}")
    # Beyond n folds np.array_split silently emits empty test sets, which
    # contribute nothing to either the error or the total and so drag the
    # cross-validated R^2 toward 0 without failing.
    if folds > n:
        raise ValueError(
            f"folds={folds} exceeds the {n} available samples; K-fold needs "
            f"K <= n so that every fold holds out at least one sample.")
    if n // folds < 2:
        warnings.warn(
            f"folds={folds} on {n} samples leaves about {n / folds:.1f} "
            f"sample(s) in each held-out fold, so the cross-validated R^2 is "
            f"estimated from very few points per fold and is correspondingly "
            f"unstable. Use fewer folds.",
            stacklevel=3,
        )
    return list(np.array_split(order, folds))


def _cv_r2_multi(X: np.ndarray, S: np.ndarray, *, folds: int | float = 5,
                 alphas: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Cross-validated R^2 of ridge regression for many targets at once.

    ``S`` is (n, k); returns a length-k array of CV R^2, one per column. A single
    SVD of the training design per fold is shared across all targets and all
    ridge penalties, and the best penalty (lowest pooled held-out error) is
    selected per target. Sharing the SVD is what makes wide banks (e.g. raw
    pixels, p~4096) tractable, and tuning the penalty is what stops the CV score
    from collapsing when p is large.

    ``folds`` is an integer K for K-fold, or a float in ``(0, 1)`` for a single
    random hold-out of that fraction; see :func:`_test_sets`.
    """
    if alphas is None:
        alphas = _RIDGE_GRID
    n, p = X.shape
    k = S.shape[1]
    fold_id = _test_sets(n, folds, seed)
    sse = np.zeros((len(alphas), k))
    sst = np.zeros(k)
    for f in range(len(fold_id)):
        te = fold_id[f]
        # the complement of the held-out set, which is the union of the other
        # folds under K-fold and the whole remainder under a single hold-out
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        tr = np.flatnonzero(mask)
        Xtr, Xte = X[tr], X[te]
        Str, Ste = S[tr], S[te]
        mu = Str.mean(0)
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)  # U (ntr,r) s (r) Vt (r,p)
        UtY = U.T @ (Str - mu)        # (r, k)
        XteV = Xte @ Vt.T             # (nte, r)
        for ai, al in enumerate(alphas):
            filt = s / (s ** 2 + al)  # (r,)
            coef = filt[:, None] * UtY      # (r, k)
            pred = XteV @ coef + mu         # (nte, k)
            sse[ai] += np.sum((Ste - pred) ** 2, axis=0)
        sst += np.sum((Ste - Ste.mean(0)) ** 2, axis=0)
    r2 = 1.0 - sse / (sst[None, :] + 1e-12)
    return np.clip(r2.max(axis=0), -1.0, 1.0)  # best penalty per target


@validate_call(config=VALIDATE_CONFIG)
def cv_r2(X: FloatArray, y: FloatArray, *, folds: int | float = 5,
          alphas: FloatArray | None = None, seed: int = 0) -> float:
    """Cross-validated R^2 of penalty-tuned ridge regression of y on X.

    ``folds`` is an integer K for K-fold or a float in ``(0, 1)`` for a single
    random hold-out of that fraction; see :func:`_test_sets`.
    """
    return float(_cv_r2_multi(X, y[:, None],
                              folds=folds, alphas=alphas, seed=seed)[0])


@validate_call(config=VALIDATE_CONFIG)
def shapley_groups_cv(X: FloatArray, S: FloatArray, group_cols: list[IndexArray],
                      *, folds: int | float = 5, seed: int = 0,
                      alphas: FloatArray | None = None,
                      progress: bool | None = None) -> np.ndarray:
    """Group Shapley budget played on the *cross-validated* R^2.

    The budget reported by :meth:`~fva.FeatureGroundedDecomposition.shap` plays
    the Shapley game on the in-sample R^2, which buys two guarantees: the shares
    sum exactly to R^2 (efficiency), and none of them is negative (monotonicity --
    adding regressors to a least-squares fit cannot raise the residual sum of
    squares). The cost is that the total being partitioned is optimistically
    biased.

    Playing the same game on the honest total means taking

        v(S) = cross-validated R^2 using only the groups in S.

    Efficiency holds for any characteristic function with ``v(empty) = 0``, so the
    shares still sum *exactly* to the cross-validated R^2 of the full bank, and
    the stacked-bar reading survives.

    Monotonicity does not survive: a group that carries no signal can lower the
    held-out fit, so its share may come out slightly negative. That is a finding
    about the group -- in-sample attribution cannot express it, and has to hand
    such a group a small positive share instead -- so negative values should be
    reported rather than clipped.

    Cost is ``2**G`` cross-validated fits rather than ``2**G`` closed-form solves,
    so this is meant for the modest number of groups a budget uses; all directions
    are evaluated together in one pass per coalition.

    Parameters
    ----------
    X : (n, p) standardized feature matrix.
    S : (n, k) direction scores, one column per latent direction.
    group_cols : list of index arrays into the columns of ``X``, one per group.
    folds, seed, alphas : passed through to the cross-validated ridge.
    progress : show a progress bar over the ``2**G`` coalitions.  ``None`` (the
        default) shows one whenever there are enough coalitions to be worth
        watching; this is the slowest loop in the package, so it is usually on.

    Returns
    -------
    (k, G) array whose rows sum to the cross-validated R^2 of the full bank.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 6))
    >>> S = (X[:, :3] @ rng.standard_normal(3))[:, None]
    >>> phi = shapley_groups_cv(X, S, [np.arange(3), np.arange(3, 6)])
    >>> bool(phi[0, 0] > phi[0, 1])       # the informative group takes the budget
    True
    """
    G = len(group_cols)
    k = S.shape[1]

    logger.info("cross-validated Shapley over %d groups (%d coalitions)",
                G, 2 ** G)
    # Each coalition is a full K-fold ridge fit, so evaluate all 2^G of them in
    # one countable pass rather than as lazy cache misses scattered through the
    # Shapley sum below -- same total work, but it can report progress.
    subsets = (m for r in range(G + 1) for m in combinations(range(G), r))
    cache: dict[frozenset, np.ndarray] = {}
    for members in progress_bar(subsets, total=2 ** G,
                                desc=f"cv coalitions (m={G})", enabled=progress):
        if not members:
            cache[frozenset(members)] = np.zeros(k)
        else:
            idx = np.concatenate([group_cols[g] for g in members])
            cache[frozenset(members)] = _cv_r2_multi(X[:, idx], S, folds=folds,
                                                     alphas=alphas, seed=seed)

    phi = np.zeros((k, G))
    for g in range(G):
        others = [q for q in range(G) if q != g]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(G - r - 1) / factorial(G)
            for members in combinations(others, r):
                phi[:, g] += w * (cache[frozenset(members + (g,))]
                                  - cache[frozenset(members)])
    return phi


# ---------------------------------------------------------------------------
# Label ceiling: the most a family of discrete labels could ever explain
# ---------------------------------------------------------------------------
#: Distinct values above which a label column is treated as continuous and
#: refused by :func:`label_ceiling`.  Cell means over a continuous column put
#: (almost) every sample in its own cell and return a ceiling of 1, which is true
#: and useless; the number has to be low enough to catch that and high enough to
#: pass a genuinely many-levelled categorical.
MAX_LEVELS = 20


@dataclass
class CeilingResult:
    """What a single family of discrete labels can explain about one score.

    ``ceiling_r2`` is the headline: the largest ``R^2`` any function of these
    labels attains on this sample.  Everything else says why it lands where it
    does -- ``coverage`` and ``unlabeled_var_share`` for the part of the sample
    the labels never speak about, ``n_cells``/``null_r2`` for how much of the
    bound is the fit counting its own degrees of freedom, and ``ceiling_r2_cv``
    for the same bound estimated out of fold.
    """

    n: int
    n_cells: int               # distinct label patterns, incl. the unlabeled one
    n_labeled: int
    coverage: float            # fraction of samples the labels are defined on
    min_cell_size: int
    unlabeled_var_share: float  # part of 1 - ceiling_r2 sitting inside the unlabeled cell
    ceiling_r2: float          # saturated (cell-means) in-sample R^2
    ceiling_r2_cv: float       # same fit scored out of fold
    null_r2: float             # in-sample ceiling reachable by signal-free labels
    r2_labeled_only: float     # saturated R^2 among the labeled samples alone
    max_cell_gap_sd: float     # largest gap between two labeled cell means, in SD
                               # (NaN when fewer than two cells qualify)


def _cell_codes(L: np.ndarray, *, max_levels: int,
                names: list[str] | None = None) -> np.ndarray:
    """Integer code per row naming which distinct label pattern it carries.

    Refuses columns with more than ``max_levels`` distinct values: the ceiling is
    only a bound because every sample sharing a label pattern must share a
    prediction, which stops being a restriction once the patterns are as fine as
    the samples.
    """
    if L.ndim != 2 or L.shape[1] == 0:
        raise ValueError(f"expected a 2-d label matrix with >= 1 column, got "
                         f"shape {L.shape}")
    # NaN never equals itself, so uniquing rows would split every missing entry
    # into its own pattern.  Map it to a sentinel outside the observed range so
    # that "missing" behaves as one more level of the column.
    finite = L[np.isfinite(L)]
    sentinel = (float(finite.max()) + 1.0) if finite.size else 0.0
    L = np.where(np.isfinite(L), L, sentinel)

    wide = []
    for j in range(L.shape[1]):
        n_levels = len(np.unique(L[:, j]))
        if n_levels > max_levels:
            wide.append((names[j] if names else f"column {j}", n_levels))
    if wide:
        listing = ", ".join(f"{nm} ({lv} levels)" for nm, lv in wide[:5])
        raise ValueError(
            f"label_ceiling needs discrete labels, but {listing}"
            + (" ..." if len(wide) > 5 else "")
            + f" exceed max_levels={max_levels}. A continuous column gives every "
            f"sample its own cell and a ceiling of 1, which bounds nothing. Bin "
            f"the column first (the ceiling is then exact for the binning), or "
            f"raise max_levels if the column really is categorical."
        )
    return np.unique(L, axis=0, return_inverse=True)[1].ravel()


def _saturated_r2(s: np.ndarray, codes: np.ndarray) -> float:
    """In-sample ``R^2`` of predicting ``s`` by the mean of its label cell."""
    sst = float(((s - s.mean()) ** 2).sum())
    if sst <= 0:
        return 0.0
    counts = np.bincount(codes)
    means = np.bincount(codes, weights=s) / np.maximum(counts, 1)
    return float(1.0 - ((s - means[codes]) ** 2).sum() / sst)


def _saturated_r2_cv(s: np.ndarray, codes: np.ndarray, *,
                     folds: int | float, seed: int) -> float:
    """Cell means fitted on the training folds and scored on the held-out one.

    Cells absent from a training fold fall back to the training grand mean, which
    is what a saturated model can honestly say about a pattern it has not seen.
    Accumulating the error and the total across folds, as :func:`_cv_r2_multi`
    does, keeps the score comparable to every other cross-validated number here.
    """
    n = len(s)
    k = int(codes.max()) + 1
    sse = sst = 0.0
    for te in _test_sets(n, folds, seed):
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        tr = np.flatnonzero(mask)
        counts = np.bincount(codes[tr], minlength=k)
        sums = np.bincount(codes[tr], weights=s[tr], minlength=k)
        grand = float(s[tr].mean())
        means = np.where(counts > 0, sums / np.maximum(counts, 1), grand)
        sse += float(((s[te] - means[codes[te]]) ** 2).sum())
        sst += float(((s[te] - s[te].mean()) ** 2).sum())
    return float(np.clip(1.0 - sse / (sst + 1e-12), -1.0, 1.0))


@validate_call(config=VALIDATE_CONFIG)
def label_ceiling(s: FloatArray, L: FloatArray, *,
                  names: list[str] | None = None,
                  unlabeled_value: float | None = 0.0,
                  max_levels: PositiveInt = MAX_LEVELS,
                  min_cell: PositiveInt = 10,
                  cv_folds: int | float = 5,
                  seed: int = 0) -> CeilingResult:
    r"""Largest share of a score that a family of discrete labels can explain.

    Samples carrying the same pattern of labels are indistinguishable to the
    family, so any predictor built from those labels alone must give them the
    same value.  Grouping the samples by distinct label pattern and predicting
    each group by its own mean is the best such predictor on this sample --- the
    saturated model in the labels --- and its ``R^2`` is therefore an upper bound
    on what the family contributes to the direction.

    The bound is what makes this different from fitting the labels and reading
    the ``R^2``.  It holds for *any* function of the labels, so it does not
    depend on the hypothesis class: a spline expansion, an interaction bank, or
    an arbitrary nonlinear model of the same labels cannot beat it.  A family
    whose share sits far below its ceiling is under-fitted and more capacity may
    help (see :func:`~fva.basis.capacity_ladder`); a family whose share sits at
    its ceiling has said everything it can say, and the remaining variance is out
    of reach of that vocabulary rather than of the model.

    A ceiling below 1 has two distinguishable causes, and the returned fields
    separate them.  Labels defined on only part of the sample leave the unlabeled
    remainder to a single prediction, so its internal variance is unreachable by
    construction: that is ``coverage`` and ``unlabeled_var_share``.  Labels
    defined everywhere but too coarse to resolve the score leave within-cell
    variance instead, which is the rest of ``1 - ceiling_r2``.

    Parameters
    ----------
    s : (n,) score to be explained --- typically one direction's projection.
    L : (n, m) discrete label matrix for one family, one column per label.
    names : column names for ``L``, used only in error messages.
    unlabeled_value : the value marking "this label does not apply to this
        sample" (``0.0`` by default, matching sparse indicator banks where a
        sample outside the family carries all zeros).  Rows that are entirely
        this value, or entirely missing, are counted as unlabeled.  ``None``
        leaves missingness as the only thing that marks a row unlabeled, which
        is what a carrier-status or one-hot bank wants, since a zero there is an
        observed level rather than an absent label.  This affects only the
        coverage
        diagnostics; the ceiling itself is the same either way, since unlabeled
        rows simply form one more cell.  Note the default reads a one-hot
        expansion with a dropped reference level as unlabeled rather than as the
        reference level, so pass ``None`` for those.
    max_levels : distinct values above which a column is refused as continuous.
    min_cell : smallest cell allowed to enter ``max_cell_gap_sd``, which is a
        descriptive statistic and is noisy when read off near-empty cells.
    cv_folds, seed : passed to the cross-validated ceiling; see
        :func:`_test_sets`.

    Warns
    -----
    UserWarning
        If the labels cut the sample into too many cells for the in-sample
        ceiling to be tight.  See :func:`fva._diagnostics.check_cell_count`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = rng.standard_normal(400)
    >>> lab = np.zeros((400, 1))          # labels defined on a quarter of the sample
    >>> lab[:100, 0] = 1.0
    >>> s[:100] += 3.0                    # ... which they separate perfectly
    >>> c = label_ceiling(s, lab)
    >>> round(c.coverage, 2)
    0.25
    >>> bool(c.ceiling_r2 < 0.8)          # the unlabeled 75% stays out of reach
    True
    """
    s = np.asarray(s, dtype=float).ravel()
    L = np.atleast_2d(np.asarray(L, dtype=float))
    if L.shape[0] != len(s):
        raise ValueError(f"s has {len(s)} samples but L has {L.shape[0]} rows")

    codes = _cell_codes(L, max_levels=max_levels, names=names)
    n, k = len(s), int(codes.max()) + 1
    check_cell_count(n, k)

    # A row with nothing observed is unlabeled whatever the convention; the
    # ``unlabeled_value`` convention is what a sparse indicator bank adds on top.
    missing = ~np.isfinite(L)
    unlabeled = missing if unlabeled_value is None else (
        (L == unlabeled_value) | missing)
    labeled = ~unlabeled.all(axis=1)

    counts = np.bincount(codes)
    means = np.bincount(codes, weights=s) / np.maximum(counts, 1)
    # Cells big enough to read a mean off, and labeled: the gap is a statement
    # about the separation the labels achieve where they are defined.
    big = [c for c in range(k) if counts[c] >= min_cell
           and labeled[codes == c].any()]
    # With fewer than two such cells there is no separation to report, which is
    # not the same as a separation of zero.
    sd = float(s.std())
    gap = ((means[big].max() - means[big].min()) / sd
           if len(big) > 1 and sd > 0 else np.nan)

    var = float(s.var())
    unlabeled_share = (float((~labeled).mean() * s[~labeled].var() / var)
                       if var > 0 and (~labeled).any() else 0.0)

    return CeilingResult(
        n=n,
        n_cells=k,
        n_labeled=int(labeled.sum()),
        coverage=float(labeled.mean()),
        min_cell_size=int(counts.min()),
        unlabeled_var_share=unlabeled_share,
        ceiling_r2=_saturated_r2(s, codes),
        ceiling_r2_cv=_saturated_r2_cv(s, codes, folds=cv_folds, seed=seed),
        null_r2=null_r2(n, k - 1),
        r2_labeled_only=(_saturated_r2(
            s[labeled], _cell_codes(L[labeled], max_levels=max_levels))
            if labeled.any() else 0.0),
        max_cell_gap_sd=float(gap),
    )


# ---------------------------------------------------------------------------
# Shared geometry: directions, feature scaling, variance weights
# ---------------------------------------------------------------------------
#: Index label of the variance-weighted total row carried by the report tables.
WEIGHTED_SUM_LABEL = "weighted_sum"


def _standardise(F: np.ndarray) -> np.ndarray:
    """Centre and scale the feature bank to unit variance, leaving dead columns be.

    A constant column has zero spread and would divide to infinity; its scale is
    set to 1 so it survives as an all-zero column, contributing nothing to any
    fit rather than poisoning every one of them.
    """
    Fmu = F.mean(0)
    Fsd = F.std(0)
    Fsd[Fsd < 1e-9] = 1.0
    return (F - Fmu) / Fsd


def _build_directions(Z: np.ndarray, mode: str, n_directions: int | None):
    """Construct the directions to analyse, ordered by variance.

    Shared by :func:`decompose_latent_space` and :func:`direction_r2` so that the
    screen and the full budget describe the same directions in the same order --
    were they to build them separately, a change to one would silently renumber
    the other's rows.

    Returns ``(scores, eigvals, eigvecs, total_var, k)``, where ``scores`` is
    ``(n, d)`` with ``Var(scores[:, j]) = eigvals[j]``, the columns of ``eigvecs``
    are the directions in latent space, and ``k`` is how many of them to analyse.
    """
    n, d = Z.shape
    Zc = Z - Z.mean(0)
    if mode == "eigen":
        C = (Zc.T @ Zc) / n
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        scores = Zc @ eigvecs            # (n, d), Var(col k) = eigvals[k]
    elif mode == "axes":
        eigvecs = np.eye(d)
        eigvals = Zc.var(0)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        scores = Zc[:, order]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    total_var = float(eigvals.sum())
    k = d if n_directions is None else min(n_directions, d)
    return scores, eigvals, eigvecs, total_var, k


def _variance_weights(eigvals: np.ndarray, total_var: float, k: int) -> np.ndarray:
    """Weights of the analysed directions, renormalised to sum to 1.

    The renormalisation is what makes a truncated analysis readable: with
    ``n_directions`` set, the weighted total is the explained fraction *of the
    variance actually analysed*, not a number silently deflated by the directions
    that were dropped.
    """
    vfrac = eigvals[:k] / total_var
    return vfrac / vfrac.sum()


def _direction_labels(mode: str, eigvecs: np.ndarray, k: int,
                      latent_names: list[str] | None = None) -> list[str]:
    """Row labels for the analysed directions, in the order they are returned.

    Eigen directions are mixtures of every latent column, so they are labelled by
    rank.  Axes directions are single latent columns, but sorting by variance
    means the k-th analysed direction is whichever column ranked k-th, so the
    label has to be recovered from the eigenvector rather than assumed to be the
    k-th column.
    """
    if mode != "axes":
        return [f"PC{i}" for i in range(k)]
    axis_of = np.argmax(np.abs(eigvecs[:, :k]), axis=0)
    if latent_names is None:
        return [str(int(i)) for i in axis_of]
    return [str(latent_names[int(i)]) for i in axis_of]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class DirectionResult:
    index: int
    variance: float            # lambda_k (eigenvalue) or Var(axis)
    variance_frac: float       # lambda_k / sum lambda
    r2: float                  # in-sample R^2
    r2_cv: float               # cross-validated R^2
    group_shares: dict[str, float]   # Shapley share of R^2 per family (sum=r2)
    feature_weights: dict[str, float]  # relative weight per column of F (sum=r2)
    noise_frac: float          # 1 - r2
    r2_ci: tuple[float, float] | None = None  # bootstrap CI on r2


@dataclass
class DecompositionResult:
    directions: list[DirectionResult]
    eigvals: np.ndarray
    eigvecs: np.ndarray        # columns are directions in latent space (d x k)
    feature_names: list[str]
    group_names: list[str]
    mode: str
    # Global summaries
    global_r2: float = 0.0                       # variance-weighted mean R^2
    global_r2_cv: float = 0.0                    # variance-weighted mean CV R^2
    global_group_shares: dict[str, float] = field(default_factory=dict)
    global_noise: float = 0.0

    def summary_table(self):
        """Return a pandas DataFrame, one row per direction."""
        import pandas as pd
        rows = []
        for d in self.directions:
            row = {"direction": d.index, "variance_frac": d.variance_frac,
                   "R2": d.r2, "R2_cv": d.r2_cv, "noise": d.noise_frac}
            row.update({f"grp_{g}": d.group_shares.get(g, 0.0) for g in self.group_names})
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Screen: total explained fraction per direction, without the attribution
# ---------------------------------------------------------------------------
@validate_call(config=VALIDATE_CONFIG)
def direction_r2(
    Z: FloatArray,
    F: FloatArray,
    *,
    mode: Literal["eigen", "axes"] = "eigen",
    n_directions: PositiveInt | None = None,
    cv_folds: int | float = 5,
    latent_names: list[str] | None = None,
    seed: int = 0,
):
    """How much of each latent direction the whole feature bank explains.

    The total ``R^2`` per direction and its variance-weighted sum --- the same
    quantities :func:`decompose_latent_space` reports as ``r2`` and ``rho`` ---
    computed without the attribution step that splits them across families.  That
    step is what costs: the exact Shapley game is ``2^G`` coalitions per direction
    and Johnson's relative weights are a per-feature fit, while the totals here
    are one Gram matrix and one SVD per fold shared across every direction.

    Use it to decide whether a bank is worth decomposing at all.  A budget answers
    "which families explain this direction", which is a question worth asking only
    once something explains it; when nothing does, the budget is a partition of
    approximately zero, and its shares will still be rendered as bars.

    Parameters
    ----------
    Z : (n, d) latent codes (posterior means).
    F : (n, q) raw feature matrix (standardised internally).
    mode : {"eigen", "axes"}
        ``"eigen"`` screens the principal directions of the latent covariance;
        ``"axes"`` screens the raw latent coordinates.
    n_directions : screen only the top-``k`` directions (by variance).
    cv_folds : folds for the cross-validated ``R^2`` -- an integer ``K >= 2`` for
        K-fold, or a float in ``(0, 1)`` for a single hold-out of that fraction.
    latent_names : optional names of the ``d`` latent columns, used to label the
        rows in ``"axes"`` mode.  Ignored for ``"eigen"``, whose directions are
        mixtures of every column and are labelled by rank.
    seed : seed for the cross-validation fold assignment.

    Returns
    -------
    DataFrame
        One row per direction, ordered by variance, plus a final
        ``"weighted_sum"`` row.  Columns:

        ``variance_fraction``
            ``lambda_k / sum lambda``.  Its ``weighted_sum`` entry is the share
            of the latent variance screened, which is ``1.0`` unless
            ``n_directions`` dropped some.
        ``r2``
            In-sample ``R^2`` of the full bank on that direction.  Biased upward
            by roughly ``q / (n - 1)`` (:func:`~fva._diagnostics.null_r2`), so
            read it against that floor rather than against zero.
        ``r2_cv``
            Cross-validated ``R^2``.  This is the one to screen on: a bank with
            no signal scores about zero here whatever its width, and negative
            values are the normal appearance of no signal rather than an error.
        ``noise``
            ``1 - r2``.

        The ``weighted_sum`` row weights each direction by its variance,
        renormalised over the directions screened, so its ``r2`` and ``r2_cv``
        entries reproduce ``rho`` and ``rho_cv`` from a full decomposition of the
        same inputs.  Matching ``rho_cv`` means floor-ing each direction's
        ``r2_cv`` at zero before weighting, so that the total stays a variance
        fraction; the per-direction column keeps its negative values.

    Warns
    -----
    UserWarning
        If the sample is too small for the width of the bank, or if the in-sample
        and cross-validated totals diverge.  See :mod:`fva._diagnostics`.

    Examples
    --------
    >>> table = direction_r2(Z, F)                     # doctest: +SKIP
    >>> table["r2_cv"].drop("weighted_sum").max()      # doctest: +SKIP
    0.83
    """
    n, d = Z.shape
    q = F.shape[1]
    logger.info(
        "direction_r2: n=%d samples, d=%d latent dims, q=%d features, mode=%r.",
        n, d, q, mode,
    )
    check_sample_size(n, q)

    X = _standardise(F)
    scores, eigvals, eigvecs, total_var, k = _build_directions(Z, mode, n_directions)
    logger.info("Screening top %d/%d directions (mode=%r).", k, d, mode)

    cache = _GramCache.build(X)
    all_cols = np.arange(q)
    r2 = np.array([
        cache.r2(X.T @ scores[:, kk], float(scores[:, kk] @ scores[:, kk]), all_cols)
        for kk in range(k)
    ])
    r2_cv = _cv_r2_multi(X, scores[:, :k], folds=cv_folds, seed=seed)

    vfrac = eigvals[:k] / total_var
    w = _variance_weights(eigvals, total_var, k)
    global_r2 = float(w @ r2)
    global_r2_cv = float(w @ np.maximum(r2_cv, 0.0))
    global_noise = float(w @ (1.0 - r2))
    logger.info(
        "Screen complete: rho=%.3f (cv=%.3f) over %d directions.",
        global_r2, global_r2_cv, k,
    )
    check_overfit_gap(global_r2, global_r2_cv)

    import pandas as pd

    labels = _direction_labels(mode, eigvecs, k, latent_names)
    table = pd.DataFrame(
        {"variance_fraction": vfrac, "r2": r2, "r2_cv": r2_cv, "noise": 1.0 - r2},
        index=pd.Index(labels, name="direction"),
    )
    table.loc[WEIGHTED_SUM_LABEL] = [float(vfrac.sum()), global_r2,
                                     global_r2_cv, global_noise]
    return table


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@validate_call(config=VALIDATE_CONFIG)
def decompose_latent_space(
    Z: FloatArray,
    F: FloatArray,
    feature_names: list[str],
    groups: dict[str, list[str]],
    *,
    mode: Literal["eigen", "axes"] = "eigen",
    n_directions: PositiveInt | None = None,
    cv_folds: int | float = 5,
    n_bootstrap: NonNegativeInt = 0,
    compute_feature_weights: bool = True,
    seed: int = 0,
    progress: bool | None = None,
) -> DecompositionResult:
    """Decompose every latent direction into feature-family variance shares.

    Parameters
    ----------
    Z : (n, d) latent codes (posterior means).
    F : (n, p) raw feature matrix (will be standardised internally).
    feature_names : list of p feature column names matching ``F``.
    groups : mapping family name -> list of feature names in that family.
    mode : {"eigen", "axes"}
        ``"eigen"`` analyses eigenvectors of the latent covariance;
        ``"axes"`` analyses the raw latent coordinates.
    n_directions : analyse only the top-``k`` directions (by variance).
    cv_folds : folds for the honest cross-validated R^2 -- an integer ``K >= 2``
        for K-fold, or a float in ``(0, 1)`` for a single hold-out of that
        fraction.
    n_bootstrap : if > 0, bootstrap resamples for a 95% CI on each R^2.
    compute_feature_weights : if False, skip the (per-feature) relative-weights
        attribution -- much faster for very wide feature banks (e.g. raw pixels).
    progress : show progress bars over the per-direction loop, the ``2**m``
        Shapley coalitions and the bootstrap resamples.  ``None`` (the default)
        shows them only for loops long enough to be worth watching.

    Warns
    -----
    UserWarning
        If the sample is too small for the width of the feature bank, or if the
        in-sample and cross-validated totals diverge once the fit is done.  See
        :mod:`fva._diagnostics`.
    """
    n, d = Z.shape
    logger.info(
        "decompose_latent_space: n=%d samples, d=%d latent dims, p=%d features, "
        "mode=%r, cv_folds=%d, n_bootstrap=%d",
        n, d, F.shape[1], mode, cv_folds, n_bootstrap,
    )
    check_sample_size(n, F.shape[1])

    # Standardise features; build the directions to analyse.
    X = _standardise(F)
    scores, eigvals, eigvecs, total_var, k = _build_directions(Z, mode, n_directions)
    logger.info("Analysing top %d/%d directions (mode=%r).", k, d, mode)

    # Group -> column-index arrays into X.
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_names = list(groups.keys())
    group_cols = [
        np.array([name_to_idx[c] for c in groups[g] if c in name_to_idx], dtype=int)
        for g in group_names
    ]

    cache = _GramCache.build(X)
    rng = np.random.default_rng(seed)

    # Cross-validated R^2 for all analysed directions at once (shared SVD).
    r2cv_all = _cv_r2_multi(X, scores[:, :k], folds=cv_folds, seed=seed)

    directions: list[DirectionResult] = []
    for kk in progress_bar(range(k), total=k, desc="directions",
                           enabled=progress):
        s = scores[:, kk]
        syy = float(s @ s)
        c = X.T @ s

        r2_full = cache.r2(c, syy, np.arange(X.shape[1]))
        r2cv = float(r2cv_all[kk])
        phi = _shapley_groups(cache, c, syy, group_cols, progress=progress)
        group_shares = {g: float(v) for g, v in zip(group_names, phi)}

        if compute_feature_weights:
            eps, _ = relative_weights(X, s)
            feat_weights = {feature_names[i]: float(eps[i]) for i in range(len(feature_names))}
        else:
            feat_weights = {}

        logger.debug(
            "direction %d/%d: var_frac=%.4f R^2=%.3f R^2_cv=%.3f",
            kk, k, eigvals[kk] / total_var, r2_full, r2cv,
        )

        r2_ci = None
        if n_bootstrap > 0:
            logger.debug("direction %d: bootstrapping R^2 over %d resamples", kk, n_bootstrap)
            boots = np.empty(n_bootstrap)
            for b in progress_bar(range(n_bootstrap), total=n_bootstrap,
                                  desc=f"bootstrap (dir {kk})", enabled=progress):
                bi = rng.integers(0, n, n)
                Xb = X[bi]
                sb = s[bi] - s[bi].mean()
                cb = Xb.T @ sb
                cacheb = _GramCache.build(Xb)
                boots[b] = cacheb.r2(cb, float(sb @ sb), np.arange(X.shape[1]))
            r2_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

        directions.append(DirectionResult(
            index=kk,
            variance=float(eigvals[kk]),
            variance_frac=float(eigvals[kk] / total_var),
            r2=float(r2_full),
            r2_cv=float(r2cv),
            group_shares=group_shares,
            feature_weights=feat_weights,
            noise_frac=float(1.0 - r2_full),
            r2_ci=r2_ci,
        ))

    # Global, variance-weighted summaries.
    vfrac = _variance_weights(eigvals, total_var, k)
    global_r2 = float(np.sum([vf * dr.r2 for vf, dr in zip(vfrac, directions)]))
    global_r2_cv = float(np.sum([vf * max(dr.r2_cv, 0.0) for vf, dr in zip(vfrac, directions)]))
    global_group = {}
    for g in group_names:
        global_group[g] = float(np.sum([vf * dr.group_shares[g]
                                        for vf, dr in zip(vfrac, directions)]))
    global_noise = float(np.sum([vf * dr.noise_frac for vf, dr in zip(vfrac, directions)]))
    logger.info(
        "Decomposition complete: global rho=%.3f (cv=%.3f), noise=%.3f over %d directions.",
        global_r2, global_r2_cv, global_noise, k,
    )
    check_overfit_gap(global_r2, global_r2_cv)

    return DecompositionResult(
        directions=directions,
        eigvals=eigvals,
        eigvecs=eigvecs,
        feature_names=feature_names,
        group_names=group_names,
        mode=mode,
        global_r2=global_r2,
        global_r2_cv=global_r2_cv,
        global_group_shares=global_group,
        global_noise=global_noise,
    )


# ---------------------------------------------------------------------------
# Feature-grounded latent reconstruction
# ---------------------------------------------------------------------------
@validate_call(config=VALIDATE_CONFIG)
def feature_grounded_codes(Z: FloatArray, F: FloatArray,
                           result: InstanceOf[DecompositionResult],
                           *, ridge: float = 1e-3) -> np.ndarray:
    """Project latent codes onto the feature-explainable subspace.

    For each analysed direction we predict its score from the features and
    reassemble the latent code, ``zhat = sum_k (features -> s_k) v_k``.  Decoding
    ``zhat`` shows what the digit looks like when only the *nameable* latent
    structure is kept.  Returns ``zhat`` with the same shape as ``Z``.
    """
    n, d = Z.shape
    Fmu, Fsd = F.mean(0), F.std(0)
    Fsd[Fsd < 1e-9] = 1.0
    X = (F - Fmu) / Fsd
    zmean = Z.mean(0)
    Zc = Z - zmean

    V = result.eigvecs[:, : len(result.directions)]  # (d, k)
    scores = Zc @ V                                   # (n, k)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    shat = np.empty_like(scores)
    for kk in range(scores.shape[1]):
        beta = np.linalg.solve(A, X.T @ scores[:, kk])
        shat[:, kk] = X @ beta
    Zhat = shat @ V.T + zmean
    return Zhat
