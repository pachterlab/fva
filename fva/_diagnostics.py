"""Data-adequacy guards for the variance budget.

The decomposition is a set of least-squares fits of ``n`` samples on ``q``
features, so it inherits the failure mode of every such fit: as ``q`` approaches
``n`` the in-sample ``R^2`` rises toward 1 whether or not the features carry any
signal, and at ``q >= n`` it *equals* 1 identically because the design spans the
score.  The Shapley budget then still partitions that total exactly --- efficiency
and non-negativity are algebraic identities and hold regardless --- so the report
looks perfectly well-formed while attributing variance that was never explained.

Nothing downstream can detect this: :meth:`_GramCache.r2` clips to ``[0, 1]``, so
the blow-up saturates quietly rather than producing an obviously wrong number.
The guards here are therefore the only place the regime is flagged.  They are
warnings rather than errors --- a wide bank is a legitimate thing to analyse (the
paper's dSprites arm decomposes a ``p = 4096`` pixel bank on purpose) as long as
the cross-validated total is what gets read.

Three checks.  The first two match the two quantities the paper reports side by
side:

``check_sample_size``
    Fires *before* the fit, on the shape of the problem alone.  The reference
    point is the expected in-sample ``R^2`` of a bank of pure noise,
    ``E[R^2] = q / (n - 1)``, which is the bias the fit carries before any signal
    is present.

``check_overfit_gap``
    Fires *after* the fit, on ``rho - rho_cv``.  This is the paper's
    "reporting both exposes any over-fitting gap" made into an actual test: the
    in-sample and cross-validated totals are both computed anyway, and a large
    gap between them is the direct evidence that the bank is too wide for the
    sample, whatever the ``n``/``q`` ratio suggested in advance.

``check_cell_count``
    Guards the label ceiling rather than the fit.  The saturated model over ``k``
    label cells is itself a fit spending ``k - 1`` degrees of freedom, so the
    bound it reports drifts upward for the same reason and is vacuous once
    ``k >= n``.

The thresholds are conventions, not theory, and are exposed as module constants
so a caller can tighten or relax them.
"""

from __future__ import annotations

import warnings

#: Samples per feature below which the in-sample budget is called unreliable.
#: The usual regression rule of thumb; at 10:1 the null-model bias
#: ``q / (n - 1)`` sits near 0.1, i.e. a tenth of the budget is fit noise.
SAMPLES_PER_FEATURE_WARN = 10

#: Gap ``rho - rho_cv`` above which the bank is called overfitted.  Chosen so
#: that a well-specified bank (the paper's arms run gaps of a few points) passes
#: while a bank inflated toward the ``R^2 = 1`` ceiling does not.
OVERFIT_GAP_WARN = 0.20

#: Null-model inflation above which the in-sample label ceiling is called
#: unreliable.  A saturated fit over ``k`` label cells spends ``k - 1`` degrees
#: of freedom, so it carries the same ``q / (n - 1)`` bias as any other fit; at
#: 0.05 a ceiling of, say, 0.20 is still mostly signal, above it the ceiling is
#: increasingly the fit counting its own cells.
CEILING_NULL_R2_WARN = 0.05

#: ``null_r2`` above which the budget figures draw the noise floor by default.
#: Half the level at which :func:`check_sample_size` fires (10:1 puts the floor
#: near 0.1), so the band between the two is the regime where the bias is worth
#: *seeing* on the figure but not worth a second warning on top of the fit-time
#: one -- a dashed line across a tenth of the bar states the case by itself.
#: Below it the line would sit on the axis spine and read as an artifact.
NULL_R2_LINE_MIN = 0.05


def null_r2(n: int, q: int) -> float:
    """Expected in-sample ``R^2`` of ``q`` pure-noise features on ``n`` samples.

    ``E[R^2] = q / (n - 1)`` for a centred response and independent regressors;
    the quantity adjusted-``R^2`` corrects for.  Saturates at 1 once ``q >= n``,
    which is the regime the sample-size check is really about.
    """
    if n <= 1:
        return 1.0
    return min(1.0, q / (n - 1))


def check_sample_size(n: int, q: int, *, stacklevel: int = 3) -> None:
    """Warn when the sample cannot support a bank of ``q`` features.

    Two regimes.  ``q >= n`` is degenerate: the in-sample ``R^2`` is 1 by
    construction and the budget is vacuous.  ``n < SAMPLES_PER_FEATURE_WARN * q``
    is merely biased: the budget is readable but its total is inflated by roughly
    ``null_r2(n, q)``.
    """
    if q == 0 or n == 0:
        return
    if q >= n:
        warnings.warn(
            f"{n} samples for {q} features: with q >= n the in-sample R^2 is 1 "
            f"by construction (the feature bank spans every score), so the "
            f"variance budget partitions a total that carries no information "
            f"and every reported share is an artifact of the fit. Read rho_cv "
            f"and shap(cv=True) instead, or reduce the bank below {n} features.",
            stacklevel=stacklevel,
        )
        return
    per_feature = n / q
    if per_feature < SAMPLES_PER_FEATURE_WARN:
        warnings.warn(
            f"{n} samples for {q} features ({per_feature:.1f} per feature): the "
            f"in-sample R^2 is optimistically biased -- a bank of this width "
            f"made of pure noise would score R^2 ~ {null_r2(n, q):.2f} on this "
            f"sample. Prefer at least {SAMPLES_PER_FEATURE_WARN} samples per "
            f"feature, or read rho_cv and shap(cv=True), which are not inflated "
            f"by feature count.",
            stacklevel=stacklevel,
        )


def check_cell_count(n: int, k: int, *, stacklevel: int = 3) -> None:
    """Warn when ``k`` label cells are too many for ``n`` samples to support.

    The saturated cell-means fit is the tightest bound available on what a family
    of discrete labels can explain, but it is still a fit, and the same arithmetic
    as :func:`check_sample_size` applies with ``q = k - 1``.  Once the labels cut
    the sample into as many cells as there are samples the bound is 1 by
    construction and says nothing; well before that it drifts upward and stops
    being tight.  Either way the cross-validated ceiling is the one to read.
    """
    if k <= 1 or n <= 1:
        return
    if k >= n:
        warnings.warn(
            f"{k} label cells over {n} samples: with one sample per cell the "
            f"saturated fit reproduces the score exactly, so the in-sample "
            f"ceiling is 1 by construction and bounds nothing. This says the "
            f"binding constraint is the sample size rather than the coverage of "
            f"the labels. Read ceiling_r2_cv, which is not inflated by cell "
            f"count, or pool levels into coarser labels.",
            stacklevel=stacklevel,
        )
        return
    bias = null_r2(n, k - 1)
    if bias > CEILING_NULL_R2_WARN:
        warnings.warn(
            f"{k} label cells over {n} samples: the saturated fit spends "
            f"{k - 1} degrees of freedom, so labels carrying no signal at all "
            f"would still reach a ceiling of about {bias:.2f} on this sample. "
            f"The in-sample ceiling is inflated by roughly that much; read "
            f"ceiling_r2_cv alongside it.",
            stacklevel=stacklevel,
        )


def check_overfit_gap(r2: float, r2_cv: float, *, stacklevel: int = 3) -> None:
    """Warn when the in-sample total sits far above the cross-validated one.

    The gap is the direct measurement of what :func:`check_sample_size` can only
    predict from the shape of the problem, so it fires even when the ``n``/``q``
    ratio looked comfortable (heavily collinear or redundant banks reach it at
    ratios that would otherwise pass).
    """
    gap = r2 - r2_cv
    if gap > OVERFIT_GAP_WARN:
        warnings.warn(
            f"in-sample rho={r2:.3f} but cross-validated rho_cv={r2_cv:.3f} "
            f"(gap {gap:.3f}): the feature bank explains far less held-out "
            f"latent variance than the budget tables report. The default shap() "
            f"and johnson() tables partition the in-sample total, so their "
            f"shares are inflated by roughly this much; shap(cv=True) partitions "
            f"the cross-validated total instead.",
            stacklevel=stacklevel,
        )
