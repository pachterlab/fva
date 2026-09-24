"""Dataframe-first API for feature-grounded variance decomposition.

This module wraps the core method in :mod:`decomposition` behind a single class
so that the analysis can be applied to any latent space in a few lines.  All
inputs are pandas objects:

* ``df_latent`` -- (n x d) latent codes, one column per latent dimension.
* ``df_feat``   -- (n x q) interpretable features, one column per feature.
* ``feat_to_group`` -- mapping ``feature name -> group name`` (q -> g).

The class fits the decomposition once and exposes three reports.  Each latent
direction is a principal direction of the latent covariance (``mode="eigen"``)
or a raw latent axis (``mode="axes"``).  For every direction the fraction of its
variance the features explain is ``R^2``; the remainder ``1 - R^2`` is the
irreducible (unexplained) part.  Weighting directions by their variance gives
the headline scalar ``rho``, the fraction of the whole latent space explained.

* :meth:`scree`   -- variance carried by each latent direction (figure only).
* :meth:`shap`    -- group-level Shapley budget: a ``(d+1) x (g+1)`` table, the
  scalar ``rho``, and a stacked-bar figure.  The ``g`` group columns are the
  Shapley shares of ``R^2`` (collinearity-aware, non-negative); the final
  ``noise`` column is the unexplained fraction ``1 - R^2``.  Groups plus noise
  partition each direction's variance, so **every row sums to 1**.  The final row
  is the eigenvalue-weighted sum across directions (the global budget).
* :meth:`johnson` -- per-feature Johnson (2000) relative weights: a
  ``(d+1) x (q+1)`` table, the scalar ``rho``, and a heatmap figure.  Same layout
  as :meth:`shap` but the attribution is refined to individual features.
* :meth:`global_budget` -- the whole latent space's budget as a single stacked
  bar (the ``weighted_sum`` row of the :meth:`shap` or :meth:`johnson` table),
  ``by="group"`` or ``by="feature"`` (figure only).

Both attributions are exact additive decompositions of the same ``R^2``: the
group Shapley shares and the per-feature relative weights each sum to ``R^2`` for
every direction, so the two tables are consistent refinements of one another.

The budget tables carry only quantities on the share scale (group/feature shares
and noise, summing to 1).  The two quantities on a different scale -- the
eigenvalue *weight* of each direction and its total explained fraction ``R^2`` --
are shared across both views and are exposed as object-level vectors,
:attr:`variance_fraction` and :attr:`r2`, rather than duplicated as columns in
each table.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from pydantic import InstanceOf, NonNegativeInt, PositiveInt, validate_call

from . import viz
from ._validation import VALIDATE_CONFIG, FigOut, FigSize
from .decomposition import (
    MAX_LEVELS,
    DecompositionResult,
    _cell_codes,
    _cv_r2_multi,
    _direction_labels,
    _GramCache,
    WEIGHTED_SUM_LABEL,
    decompose_latent_space,
    label_ceiling,
    shapley_groups_cv,
)

logger = logging.getLogger(__name__)


class Report(NamedTuple):
    """Return value of :meth:`FeatureGroundedDecomposition.shap` / ``.johnson``.

    Unpack as ``table, rho, fig = explainer.shap()`` or access by name.
    """

    table: pd.DataFrame   # (d+1) x (g+1) [shap] or (d+1) x (q+1) [johnson]
    rho: float            # fraction of total latent variance explained by features
    figure: object        # matplotlib Figure


class CeilingReport(NamedTuple):
    """Return value of :meth:`FeatureGroundedDecomposition.ceilings`.

    ``table`` and ``alone`` share the layout of the budget tables (one row per
    direction plus a weighted sum, one column per family) and are read together:
    ``alone`` is what a family explains by itself, ``table`` is the most it could
    explain, and ``attainment`` is their ratio.  ``coverage`` carries the label
    geometry behind the ceilings, one row per family.
    """

    table: pd.DataFrame        # (d+1) x g ceilings
    alone: pd.DataFrame        # (d+1) x g family-only R^2
    attainment: pd.DataFrame   # alone / ceiling
    coverage: pd.DataFrame     # g x 5 label geometry


_WEIGHTED_SUM_LABEL = WEIGHTED_SUM_LABEL
_NOISE_COL = "noise"

# Above this group count the exact 2^G Shapley game is slow and the stacked-bar
# figure becomes unreadable; shap() warns and suggests pooling features.
_SHAP_GROUP_WARN = 12

class FeatureGroundedDecomposition:
    """Feature-grounded variance decomposition of a latent space.

    Parameters
    ----------
    df_latent : DataFrame, shape (n, d)
        Latent codes, one row per sample and one column per latent dimension.
    df_feat : DataFrame, shape (n, q)
        Interpretable features, one row per sample and one column per feature.
        Must describe the same samples as ``df_latent`` (aligned by index if both
        carry a shared index, otherwise assumed to be in the same row order).
    feat_to_group : dict, optional
        Maps each feature name (a column of ``df_feat``) to its group name.
        Features absent from this mapping are collected into an ``"ungrouped"``
        group with a warning.  ``None`` (the default) makes each feature its own
        singleton group, reducing the group budget to a per-feature one -- handy
        when groups are not needed or when there are few enough features to treat
        each individually.
    mode : {"eigen", "axes"}, default "eigen"
        ``"eigen"`` analyses the principal directions of the latent covariance;
        ``"axes"`` analyses the raw latent columns of ``df_latent``.
    n_directions : int, optional
        Analyse only the top-``k`` directions by variance.  ``None`` uses all
        ``d`` directions.
    cv_folds : int or float, default 5
        How to hold data out for the cross-validated ``R^2`` (exposed via
        :attr:`rho_cv`, and the total partitioned by ``shap(cv=True)``; the
        default report tables use the in-sample ``R^2`` so that the additive
        budgets are exact).  An integer ``K >= 2`` means K-fold: every sample is
        held out exactly once, so the score uses n out-of-fold predictions.  A
        float in ``(0, 1)`` means a single random hold-out of that fraction, which
        scores only that fraction and is therefore noisier -- markedly so for
        ``shap(cv=True)``, where Shapley values are differences of coalition
        scores and the noise compounds.  Prefer K-fold unless a single split is
        forced on you.
    n_bootstrap : int, default 0
        Resamples for a bootstrap 95% CI on each direction's ``R^2``, exposed via
        :attr:`r2_ci`.  ``0`` (the default) skips the resampling entirely.  Since
        every quantity here is a statistic of a finite sample, an interval is
        what distinguishes a direction the features genuinely explain from one
        whose share is not separable from zero -- worth the cost whenever the
        sample is small enough for that to be in doubt.  Costs ``n_bootstrap``
        extra fits per direction.
    seed : int, default 0
        Random seed for the cross-validation fold assignment and the bootstrap
        resampling.
    progress : bool, optional
        Show progress bars over the long loops.  ``None`` (the default) shows
        them only when a loop is long enough to be worth watching, which in
        practice means the ``2**g`` Shapley coalitions once there are more than a
        few groups.  ``False`` silences them.

    Warns
    -----
    UserWarning
        Raised on the first fit (not at construction, which is lazy) if the
        sample is too small for the width of the feature bank, if the in-sample
        and cross-validated totals diverge, or if the feature bank is rank
        deficient.  See :mod:`fva._diagnostics`.

    Notes
    -----
    The decomposition is computed lazily on first use and cached.  Its quantities
    are exposed directly on the object: the headline scalars :attr:`rho` and
    :attr:`rho_cv`, the per-direction vectors :attr:`variance_fraction` and
    :attr:`r2`, and the raw internals :attr:`directions`, :attr:`eigvals`,
    :attr:`eigvecs`, :attr:`global_group_shares`, and :attr:`global_noise`.
    """

    @validate_call(config=VALIDATE_CONFIG)
    def __init__(
        self,
        df_latent: InstanceOf[pd.DataFrame],
        df_feat: InstanceOf[pd.DataFrame],
        feat_to_group: dict | None = None,
        *,
        mode: Literal["eigen", "axes"] = "eigen",
        n_directions: PositiveInt | None = None,
        cv_folds: int | float = 5,
        n_bootstrap: NonNegativeInt = 0,
        seed: int = 0,
        progress: bool | None = None,
    ):
        df_latent, df_feat = self._align(df_latent, df_feat)
        self._check_numeric(df_feat)

        self.df_latent = df_latent
        self.df_feat = df_feat
        self.feat_to_group = dict(feat_to_group) if feat_to_group is not None else None
        self.mode = mode
        self.n_directions = n_directions
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.progress = progress

        self.feature_names = list(df_feat.columns)
        self.groups = self._invert_mapping(self.feature_names, feat_to_group)
        self.group_names = list(self.groups.keys())

        # Labels for the latent directions (rows of the report tables).  Filled
        # in after fitting for both modes -- see _make_dir_labels.
        self._dir_labels = None

        self._result: DecompositionResult | None = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _align(df_latent: pd.DataFrame, df_feat: pd.DataFrame):
        """Align the two frames by shared index, or by row order as a fallback."""
        if df_latent.index.equals(df_feat.index):
            return df_latent, df_feat
        common = df_latent.index.intersection(df_feat.index)
        if len(common) > 0:
            if len(common) < max(len(df_latent), len(df_feat)):
                warnings.warn(
                    f"df_latent and df_feat share only {len(common)} of "
                    f"{len(df_latent)}/{len(df_feat)} rows; using the intersection.",
                    stacklevel=3,
                )
            return df_latent.loc[common], df_feat.loc[common]
        if len(df_latent) != len(df_feat):
            raise ValueError(
                f"df_latent has {len(df_latent)} rows but df_feat has "
                f"{len(df_feat)}, and their indices do not overlap."
            )
        warnings.warn(
            "df_latent and df_feat have no shared index labels; assuming the rows "
            "are already in the same order.",
            stacklevel=3,
        )
        df_feat = df_feat.copy()
        df_feat.index = df_latent.index
        return df_latent, df_feat

    @staticmethod
    def _check_numeric(df_feat: pd.DataFrame) -> None:
        """Reject non-numeric feature columns at the call site, by name.

        Without this the frame reaches ``np.asarray(..., dtype=float)`` inside
        :func:`~fva.decomposition.decompose_latent_space` and fails on the first
        offending *value* ("could not convert string to float: 'z'"), which does
        not say which column carried it or what to do about it.
        """
        bad = [c for c in df_feat.columns
               if not pd.api.types.is_numeric_dtype(df_feat[c])]
        if bad:
            raise ValueError(
                f"df_feat column(s) are not numeric: {bad[:5]}"
                + (" ..." if len(bad) > 5 else "")
                + ". The decomposition is a linear fit, so categorical and "
                "circular columns have to be re-represented first: "
                "df_feat, feat_to_group = fva.encode_features(df_raw). "
                "See fva.encoding."
            )

    @staticmethod
    def _invert_mapping(feature_names: list[str], feat_to_group: dict | None) -> dict:
        """Turn ``feature -> group`` into ``group -> [features]`` (order-stable).

        ``feat_to_group=None`` makes each feature its own singleton group, so the
        group Shapley budget reduces to a per-feature budget.  This is the natural
        default when groups are not needed (the Johnson view is per-feature
        regardless) or when there are few enough features to treat individually.
        """
        if feat_to_group is None:
            return {f: [f] for f in feature_names}
        groups: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for feat in feature_names:
            g = feat_to_group.get(feat)
            if g is None:
                ungrouped.append(feat)
                g = "ungrouped"
            groups.setdefault(g, []).append(feat)
        if ungrouped:
            warnings.warn(
                f"{len(ungrouped)} feature(s) are not in feat_to_group and were "
                f"placed in an 'ungrouped' group: {ungrouped[:5]}"
                + (" ..." if len(ungrouped) > 5 else ""),
                stacklevel=3,
            )
        return groups

    def _fit(self) -> "FeatureGroundedDecomposition":
        """Run (and cache) the decomposition.  Called automatically when needed."""
        if self._result is None:
            logger.info(
                "Fitting FeatureGroundedDecomposition: %d samples, %d latent dims, "
                "%d features, %d groups.",
                len(self.df_latent), self.df_latent.shape[1],
                len(self.feature_names), len(self.group_names),
            )
            self._result = decompose_latent_space(
                self.df_latent.values,
                self.df_feat.values,
                self.feature_names,
                self.groups,
                mode=self.mode,
                n_directions=self.n_directions,
                cv_folds=self.cv_folds,
                n_bootstrap=self.n_bootstrap,
                compute_feature_weights=True,
                seed=self.seed,
                progress=self.progress,
            )
            if self._dir_labels is None:
                self._dir_labels = self._make_dir_labels()
        return self

    def _make_dir_labels(self) -> list[str]:
        """Row labels for the analysed directions, in the order the fit returns them.

        These have to come from the fitted result rather than from ``df_latent``.
        Both modes sort directions by variance and ``n_directions`` may keep only
        the top ``k``, so in ``"axes"`` mode the k-th analysed direction is
        whichever raw latent column ranked k-th by variance -- which is the k-th
        column only if the columns happened to arrive in descending-variance
        order.  Labelling by position instead attaches each row to the wrong
        column, silently, in the tables and along the figure axis.
        """
        res = self._result
        return _direction_labels(self.mode, res.eigvecs, len(res.directions),
                                 list(self.df_latent.columns))

    @property
    def _res(self) -> DecompositionResult:
        """Internal :class:`~decomposition.DecompositionResult` (fits if needed)."""
        return self._fit()._result

    @property
    def rho(self) -> float:
        """Fraction of total latent variance explained by the features."""
        return self._res.global_r2

    @property
    def rho_cv(self) -> float:
        """Cross-validated estimate of :attr:`rho`."""
        return self._res.global_r2_cv

    @property
    def directions(self) -> list:
        """Per-direction results (variance, ``R^2``, attributions); one per direction."""
        return self._res.directions

    @property
    def eigvals(self) -> np.ndarray:
        """Eigenvalues of the latent covariance (variance carried by each direction)."""
        return self._res.eigvals

    @property
    def eigvecs(self) -> np.ndarray:
        """Eigenvectors of the latent covariance; columns are directions (``d x k``)."""
        return self._res.eigvecs

    @property
    def global_group_shares(self) -> dict:
        """Eigenvalue-weighted Shapley share of each group across all directions."""
        return self._res.global_group_shares

    @property
    def global_noise(self) -> float:
        """Eigenvalue-weighted unexplained fraction across all directions (``1 - rho``)."""
        return self._res.global_noise

    @property
    def variance_fraction(self) -> pd.Series:
        """Eigenvalue weight of each direction (``lambda_k / sum lambda``).

        Indexed by direction label with a final ``weighted_sum`` entry equal to
        the total analysed variance fraction (1.0 when all directions are kept).
        This is the weight used by the eigenvalue-weighted-sum rows of the budget
        tables; it lives on the object rather than as a table column because it is
        on the variance scale, not the share scale, and is shared by both views.
        """
        res = self._res
        vals = [d.variance_frac for d in res.directions]
        idx = list(self._dir_labels) + [_WEIGHTED_SUM_LABEL]
        return pd.Series(vals + [float(sum(vals))], index=idx, name="variance_fraction")

    @property
    def r2(self) -> pd.Series:
        """Total explained fraction ``R^2`` of each direction (``1 - noise``).

        Indexed by direction label with a final ``weighted_sum`` entry equal to
        :attr:`rho`.  Equals the row sum of the attribution columns in either
        budget table; kept as a vector to avoid a column redundant with ``noise``.
        """
        res = self._res
        vals = [d.r2 for d in res.directions]
        idx = list(self._dir_labels) + [_WEIGHTED_SUM_LABEL]
        return pd.Series(vals + [res.global_r2], index=idx, name="r2")

    @property
    def r2_cv(self) -> pd.Series:
        """Cross-validated ``R^2`` of each direction, the counterpart of :attr:`r2`.

        Indexed by direction label with a final ``weighted_sum`` entry equal to
        :attr:`rho_cv`.  A direction whose entry is at or below zero is one the
        bank does not explain out of sample, however much in-sample ``R^2``
        :attr:`r2` credits it with; the gap between the two vectors is the width
        of the bank showing up as fit.

        The ``weighted_sum`` entry floors each direction at zero before weighting
        (so that ``rho_cv`` remains a variance fraction), which the per-direction
        entries do not -- their negative values are the readable signature of a
        direction with nothing in it.

        Screening this without paying for the attribution is
        :func:`fva.decomposition.direction_r2`.
        """
        res = self._res
        vals = [d.r2_cv for d in res.directions]
        idx = list(self._dir_labels) + [_WEIGHTED_SUM_LABEL]
        return pd.Series(vals + [res.global_r2_cv], index=idx, name="r2_cv")

    @property
    def r2_ci(self) -> pd.DataFrame:
        """Bootstrap 95% percentile interval on each direction's ``R^2``.

        A ``(d, 2)`` DataFrame with ``lo``/``hi`` columns, indexed by direction
        label; there is no ``weighted_sum`` row because the interval is
        per-direction.  An interval whose lower end sits at or near zero marks a
        direction whose explained variance is not distinguishable from none, so
        its budget should not be read as structure.

        Requires ``n_bootstrap > 0`` at construction.  Without it the attribute
        raises :exc:`AttributeError` rather than returning an empty table, which
        would read as "no uncertainty" rather than "not computed".
        ``AttributeError`` specifically, so that the intervals are simply absent
        as far as ``hasattr`` and :func:`inspect.getmembers` are concerned rather
        than an error that escapes anything walking the object's attributes.
        """
        if not self.n_bootstrap:
            raise AttributeError(
                "no bootstrap intervals were computed; construct with "
                "FeatureGroundedDecomposition(..., n_bootstrap=200) to get them."
            )
        res = self._res
        return pd.DataFrame(
            [d.r2_ci for d in res.directions],
            index=list(self._dir_labels), columns=["lo", "hi"],
        )

    # ------------------------------------------------------------------ reports
    @validate_call(config=VALIDATE_CONFIG)
    def scree(self, *, figsize: FigSize = (8, 2.8), fig_out: FigOut = None):
        """Plot the variance carried by each latent direction.  Returns a Figure.

        ``fig_out`` controls saving: ``None`` (default) returns the figure
        without writing it; a path saves to that path; ``True`` saves to the
        default location ``figures/scree.png``.
        """
        fig = viz.plot_scree(self._res, figsize=figsize)
        viz.save_figure(fig, fig_out, "scree")
        return fig

    @validate_call(config=VALIDATE_CONFIG)
    def shap(self, *, figure_type: Literal["bar", "heatmap"] = "bar",
             feat_to_group: dict | None = None, top_k: PositiveInt | None = None,
             top_n: PositiveInt = 12, figsize: FigSize = (10, 6),
             cv: bool = False, null_line: bool | None = None,
             scree: bool = False, total: bool = False,
             fig_out: FigOut = None) -> Report:
        """Group-level Shapley variance budget.

        Returns a :class:`Report` ``(table, rho, figure)`` where ``table`` is a
        ``(d+1) x (g+1)`` DataFrame: one row per latent direction plus a final
        eigenvalue-weighted-sum row, one column per group, and a final ``noise``
        column (``1 - R^2``).  The group columns are the Shapley shares of ``R^2``;
        groups plus noise partition each direction's variance, so every row sums
        to 1.  The per-direction ``R^2`` and eigenvalue weight are available as
        :attr:`r2` and :attr:`variance_fraction`.

        ``feat_to_group`` overrides the grouping fixed at construction for this
        call only, recomputing the Shapley group attribution for the supplied
        ``{feature -> group}`` mapping (``None`` uses the constructor's grouping).

        ``cv=True`` plays the Shapley game on the **cross-validated** ``R^2``
        instead of the in-sample one, so the shares partition :attr:`rho_cv` and
        the reported ``rho`` is the cross-validated figure.  Efficiency (the shares
        summing exactly to the total) survives, because that axiom holds for any
        characteristic function vanishing on the empty coalition.  Non-negativity
        does not: it follows from monotonicity of the in-sample fit, and a group
        carrying no signal can *lower* a held-out fit, so its share may come out
        slightly negative.  Such a value is a finding about the group rather than a
        defect in the accounting, so it is reported rather than clipped, with a
        warning; note the stacked bar clips at zero even though the table does not.
        Cost is ``2**g`` cross-validated fits, so keep the group count modest.

        The returned ``rho`` is the variance-weighted mean of the per-direction
        cross-validated ``R^2`` taken as is, which is what the table's
        ``weighted_sum`` row partitions.  It can sit slightly below :attr:`rho_cv`,
        which floors each direction at zero before averaging: a direction the
        features genuinely anti-predict contributes its negative value here and
        zero there.

        ``figure_type`` selects the figure: ``"bar"`` (default) draws one
        stacked bar per direction; ``"heatmap"`` draws a direction-by-group grid
        of the shares.  ``top_k`` limits the directions shown in the bar; ``top_n``
        limits the groups shown in the heatmap (the table always carries all ``g``).

        ``null_line`` draws the pure-noise ``R^2`` floor across the bar, below
        which the explained variance could be fit noise rather than signal:
        ``None`` (the default) draws it only when the floor is high enough to
        matter, ``True`` always, ``False`` never.  See
        :func:`fva.viz.draw_null_floor`.  The heatmap plots shares of ``R^2``
        rather than ``R^2``, so it has no axis for the floor and ignores this;
        ``cv=True`` refuses it, since the floor is the in-sample bias and a
        pure-noise bank scores about zero out of fold.

        ``scree=True`` overlays the scree curve on the bar as a black line: the
        eigenvalue weight of each direction (``lambda_k / sum lambda``, the same
        number printed under each tick), on the bar's own ``[0, 1]`` axis.  It
        puts the budget in proportion -- a direction whose variance is well spent
        may still carry little of the space.  The heatmap has no such axis and
        ignores this.

        ``total=True`` appends the table's ``weighted_sum`` row to the right of
        the bar as a ``total`` bar, set apart by a gap and a dashed separator: the
        whole latent space's budget next to the per-direction ones.  It always
        pools every direction, whatever ``top_k``.  The heatmap ignores this.
        """
        if cv:
            return self._shap_cv(figure_type=figure_type, feat_to_group=feat_to_group,
                                 top_k=top_k, top_n=top_n, figsize=figsize,
                                 null_line=null_line, scree=scree, total=total,
                                 fig_out=fig_out)
        res, group_names = self._group_result(feat_to_group)
        if len(group_names) > _SHAP_GROUP_WARN:
            warnings.warn(
                f"shap() is attributing to {len(group_names)} groups; the exact "
                f"Shapley game is exponential in the number of groups (2^G "
                f"coalitions) and the figure is hard to read with this many. Pass "
                f"feat_to_group to pool features into fewer groups, or use "
                f"johnson() for the per-feature view.",
                stacklevel=2,
            )
        attributions = [d.group_shares for d in res.directions]
        table = self._build_table(group_names, attributions, res.global_group_shares)
        fig = self._budget_figure(
            table, group_names, figure_type=figure_type,
            color_of=viz._color, col_label="group", top_k=top_k, top_n=top_n,
            figsize=figsize, null_line=null_line, scree=scree, total=total,
        )
        viz.save_figure(fig, fig_out, "shap")
        return Report(table=table, rho=res.global_r2, figure=fig)

    # # Capitalised alias matching the requested API.
    # SHAP = shap

    @validate_call(config=VALIDATE_CONFIG)
    def johnson(self, *, figure_type: Literal["heatmap", "bar"] = "heatmap",
                top_n: PositiveInt = 20, top_k: PositiveInt | None = None,
                figsize: FigSize = (10, 6), null_line: bool | None = None,
                total: bool = False, fig_out: FigOut = None) -> Report:
        """Per-feature Johnson relative-weight budget.

        Returns a :class:`Report` ``(table, rho, figure)`` where ``table`` is a
        ``(d+1) x (q+1)`` DataFrame: one row per latent direction plus a final
        eigenvalue-weighted-sum row, one column per feature, and a final
        ``noise`` column.  The feature columns are the Johnson relative weights;
        features plus noise partition each direction's variance, so every row sums
        to 1.

        ``figure_type`` selects the figure: ``"heatmap"`` (default) draws a
        direction-by-feature grid; ``"bar"`` draws one stacked bar per direction.
        ``top_n`` limits the features shown (the bar lumps the rest into an
        ``"other"`` segment; the table always carries all ``q``).  ``top_k`` limits
        the directions shown in the bar.

        ``null_line`` draws the pure-noise ``R^2`` floor across the bar (``None``
        auto, ``True`` always, ``False`` never); the default heatmap has no
        ``R^2`` axis and ignores it.  See :func:`fva.viz.draw_null_floor`.  The
        floor is worth watching here in particular: the per-feature budget is
        always taken over the full bank, which is where ``q`` is largest.

        ``total=True`` appends the ``weighted_sum`` row to the right of the bar as
        a ``total`` bar (see :meth:`shap`); the heatmap ignores it.

        ``fig_out`` controls saving: ``None`` (default) returns the figure
        without writing it; a path saves to that path; ``True`` saves to the
        default location ``figures/johnson.png``.
        """
        res = self._res
        attributions = [d.feature_weights for d in res.directions]
        table = self._build_table(self.feature_names, attributions,
                                  self._global_feature_shares())
        fig = self._budget_figure(
            table, self.feature_names, figure_type=figure_type,
            color_of=self._feature_color(), col_label="feature",
            top_k=top_k, top_n=top_n, figsize=figsize, null_line=null_line,
            total=total,
        )
        viz.save_figure(fig, fig_out, "johnson")
        return Report(table=table, rho=res.global_r2, figure=fig)

    # Johnson = johnson

    @validate_call(config=VALIDATE_CONFIG)
    def global_budget(self, *, by: Literal["group", "feature"] = "group",
                      top_n: PositiveInt = 12, figsize: FigSize = (6, 5),
                      null_line: bool | None = None, fig_out: FigOut = None):
        """Plot the whole latent space's variance budget as a single stacked bar.

        This visualizes the ``weighted_sum`` row of the :meth:`shap` table
        (``by="group"``) or the :meth:`johnson` table (``by="feature"``): the
        eigenvalue-weighted shares across all directions plus the noise remainder,
        so the bar sums to 1.  ``by="feature"`` keeps the ``top_n`` largest
        features and lumps the rest into an ``"other"`` segment.  Returns a Figure.

        ``null_line`` draws the pure-noise ``R^2`` floor across the bar, marking
        how much of ``rho`` a bank of this width would reach on this sample with
        no signal at all: ``None`` (the default) draws it only when that floor is
        high enough to matter, ``True`` always, ``False`` never.  See
        :func:`fva.viz.draw_null_floor`.

        ``fig_out`` controls saving: ``None`` (default) returns the figure
        without writing it; a path saves to that path; ``True`` saves to the
        default location ``figures/global_budget_<by>.png``.
        """
        res = self._res
        rho_pct = res.global_r2 * 100
        if by == "group":
            shares = res.global_group_shares
            title = (f"Total latent variance budget by group\n"
                     f"(features explain {rho_pct:.1f}%)")
            fig = self._global_bar(shares, res.global_noise, top_n=None,
                                   color_of=viz._color, figsize=figsize, title=title,
                                   null_line=null_line)
        elif by == "feature":
            shares = self._global_feature_shares()
            title = (f"Total latent variance budget by feature\n"
                     f"(features explain {rho_pct:.1f}%)")
            fig = self._global_bar(shares, res.global_noise, top_n=top_n,
                                   color_of=self._feature_color(), figsize=figsize,
                                   title=title, null_line=null_line)
        else:
            raise ValueError(f"by must be 'group' or 'feature', got {by!r}")
        viz.save_figure(fig, fig_out, f"global_budget_{by}")
        return fig

    @validate_call(config=VALIDATE_CONFIG)
    def ceilings(self, feat_to_group: dict | None = None, *, cv: bool = False,
                 unlabeled_value: float | None = 0.0,
                 max_levels: PositiveInt = MAX_LEVELS,
                 min_cell: PositiveInt = 10) -> CeilingReport:
        r"""How much of each direction every discrete family could ever explain.

        A small share in the :meth:`shap` budget has two readings that the budget
        itself cannot tell apart: the family is defined on the whole sample and
        says little about the direction, or the family is barely defined at all
        and could not have said much whatever it encoded.  Grouping samples by
        their distinct pattern of labels within a family and predicting each group
        by its own mean is the best any function of those labels can do, so its
        ``R^2`` bounds the family from above.  Comparing the two separates the
        readings: see :func:`~fva.decomposition.label_ceiling` for what the bound
        rests on.

        Only families whose columns are all discrete can be bounded this way, so
        continuous families are skipped with a warning naming them.  Binning a
        continuous column and re-running gives a ceiling that is exact for the
        binning.

        Parameters
        ----------
        feat_to_group : an alternative grouping, as in :meth:`shap`.  ``None``
            uses the one given at construction.
        cv : score both the ceilings and the family fits out of fold rather than
            in sample, as :meth:`shap` does with ``cv=True``.  Worth reading
            whenever a family cuts the sample into many cells: the in-sample
            ceiling then inflates toward 1 and stops being tight, while the
            cross-validated one does not.  It is an estimate rather than a bound,
            so it can come out below the family's own held-out fit by noise.
        unlabeled_value : the value marking a label as not applying to a sample;
            see :func:`~fva.decomposition.label_ceiling`.  Affects only the
            coverage columns, not the ceilings.
        max_levels : distinct values above which a column counts as continuous
            and its family is skipped.
        min_cell : smallest cell entering the ``max_cell_gap_sd`` diagnostic.

        Returns
        -------
        CeilingReport
            ``table`` -- ``(d+1) x g`` ceilings, one row per direction plus the
            eigenvalue-weighted ``weighted_sum`` row, same layout as the budget
            tables.  ``alone`` -- the same shape, each family's ``R^2`` fitted on
            its own (the quantity the ceiling bounds; not a Shapley share, which
            can exceed it when families suppress one another).  ``attainment`` --
            ``alone / table``, the fraction of the reachable share the family
            actually reaches.  ``coverage`` -- one row per family with the
            direction-independent label geometry: how many samples carry a label,
            how many distinct cells they form, the smallest cell, and the
            in-sample ceiling that signal-free labels of the same shape would
            reach.

        Examples
        --------
        >>> report = decomp.ceilings()                     # doctest: +SKIP
        >>> report.attainment.loc["weighted_sum"]          # doctest: +SKIP
        """
        self._fit()
        res = self._result
        groups = (self.groups
                  if feat_to_group is None or feat_to_group == self.feat_to_group
                  else self._invert_mapping(self.feature_names, feat_to_group))

        k = len(res.directions)
        Z = self.df_latent.values
        scores = (Z - Z.mean(0)) @ res.eigvecs[:, :k]
        F = self.df_feat.values
        Fsd = F.std(0)
        Fsd[Fsd < 1e-9] = 1.0
        X = (F - F.mean(0)) / Fsd
        cache = _GramCache.build(X)
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}

        # Which families are discrete enough to be bounded at all.
        kept, skipped = {}, []
        for g, cols in groups.items():
            idx = np.array([name_to_idx[c] for c in cols if c in name_to_idx],
                           dtype=int)
            if len(idx) == 0:
                continue
            try:
                _cell_codes(F[:, idx], max_levels=max_levels,
                            names=[self.feature_names[i] for i in idx])
            except ValueError as exc:
                skipped.append((g, exc))
                continue
            kept[g] = idx
        if skipped:
            warnings.warn(
                f"no ceiling for {len(skipped)} of {len(groups)} families, whose "
                f"columns are not discrete: "
                + ", ".join(g for g, _ in skipped[:5])
                + (" ..." if len(skipped) > 5 else "")
                + f". First reason: {skipped[0][1]}",
                stacklevel=2,
            )
        if not kept:
            raise ValueError(
                "no family has discrete labels, so nothing can be bounded. The "
                "ceiling is a statement about samples that share a label pattern "
                "and therefore share a prediction; bin the continuous columns "
                "first if a bound over the binning is what is wanted."
            )

        names = list(kept)
        ceil = np.zeros((k, len(names)))
        alone = np.zeros((k, len(names)))
        cov_rows = []
        for j, g in enumerate(names):
            idx = kept[g]
            L = F[:, idx]
            per_dir = [label_ceiling(scores[:, kk], L,
                                     names=[self.feature_names[i] for i in idx],
                                     unlabeled_value=unlabeled_value,
                                     max_levels=max_levels, min_cell=min_cell,
                                     cv_folds=self.cv_folds, seed=self.seed)
                       for kk in range(k)]
            ceil[:, j] = [c.ceiling_r2_cv if cv else c.ceiling_r2 for c in per_dir]
            if cv:
                alone[:, j] = _cv_r2_multi(X[:, idx], scores,
                                           folds=self.cv_folds, seed=self.seed)
            else:
                alone[:, j] = [cache.r2(X.T @ scores[:, kk],
                                        float(scores[:, kk] @ scores[:, kk]), idx)
                               for kk in range(k)]
            # Label geometry is a property of the family, not of the direction.
            c0 = per_dir[0]
            cov_rows.append({"n_labeled": c0.n_labeled, "coverage": c0.coverage,
                             "n_cells": c0.n_cells, "min_cell_size": c0.min_cell_size,
                             "null_r2": c0.null_r2})

        vfrac = np.array([d.variance_frac for d in res.directions])
        w = vfrac / vfrac.sum()
        index = list(self._dir_labels) + [_WEIGHTED_SUM_LABEL]
        ceil_t = pd.DataFrame(np.vstack([ceil, w @ ceil]), index=index, columns=names)
        alone_t = pd.DataFrame(np.vstack([alone, w @ alone]), index=index, columns=names)
        for t in (ceil_t, alone_t):
            t.index.name = "direction"
        attain = alone_t / ceil_t.where(ceil_t > 0)
        coverage = pd.DataFrame(cov_rows, index=names)
        coverage.index.name = "group"
        return CeilingReport(table=ceil_t, alone=alone_t, attainment=attain,
                             coverage=coverage)

    # ------------------------------------------------------------------ helpers
    def _shap_cv(self, *, figure_type, feat_to_group, top_k, top_n, figsize,
                 null_line, scree, total, fig_out) -> Report:
        """:meth:`shap` with the Shapley game played on the cross-validated R^2."""
        self._fit()
        if null_line is True:
            warnings.warn(
                "null_line=True has no meaning for a cross-validated budget: "
                "q/(n-1) is the *in-sample* bias of a pure-noise bank, whereas "
                "out of fold such a bank scores about zero (slightly negative). "
                "Drawing it here would mark a floor several times too high. The "
                "line is suppressed; the in-sample shap() is where it belongs.",
                stacklevel=3,
            )
        groups = (self.groups if feat_to_group is None
                  else self._invert_mapping(self.feature_names, feat_to_group))
        group_names = list(groups.keys())
        if len(group_names) > _SHAP_GROUP_WARN:
            warnings.warn(
                f"shap(cv=True) is attributing to {len(group_names)} groups; the "
                f"exact Shapley game costs 2^G cross-validated fits, which is much "
                f"more expensive than the in-sample game. Pool features into fewer "
                f"groups via feat_to_group.",
                stacklevel=3,
            )

        if isinstance(self.cv_folds, float) and not float(self.cv_folds).is_integer():
            warnings.warn(
                f"cv_folds={self.cv_folds} holds out a single split, so each "
                f"coalition is scored on that fraction of the data only. Shapley "
                f"values are differences of coalition scores, so the noise "
                f"compounds: shares are markedly less stable across seeds and "
                f"spurious negative shares become more common. Prefer an integer "
                f"number of folds for the cross-validated budget.",
                stacklevel=3,
            )

        res = self._result
        F = self.df_feat.values.astype(float)
        Fsd = F.std(0)
        Fsd[Fsd < 1e-9] = 1.0                      # as in decompose_latent_space
        X = (F - F.mean(0)) / Fsd
        Z = self.df_latent.values.astype(float)
        # eigvecs carries every direction; keep only the analysed ones so that the
        # scores line up with res.directions when n_directions is set
        k = len(res.directions)
        scores = (Z - Z.mean(0)) @ res.eigvecs[:, :k]   # matches both modes
        name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        group_cols = [
            np.array([name_to_idx[c] for c in groups[g] if c in name_to_idx], dtype=int)
            for g in group_names
        ]

        phi = shapley_groups_cv(X, scores, group_cols, folds=self.cv_folds,
                                seed=self.seed, progress=self.progress)
        r2_cv = _cv_r2_multi(X, scores, folds=self.cv_folds, seed=self.seed)

        if (phi < 0).any():
            neg = phi[phi < 0]
            warnings.warn(
                f"cross-validated Shapley produced {neg.size} negative share(s) "
                f"(most negative {phi.min():+.4g}, {np.abs(neg).sum()/np.abs(phi).sum():.2%} "
                f"of the attributed mass). A negative share means the group costs "
                f"held-out fit; the table keeps it, the stacked bar clips at zero.",
                stacklevel=3,
            )

        vfrac = np.array([d.variance_frac for d in res.directions])
        w = vfrac / vfrac.sum()
        attributions = [dict(zip(group_names, row)) for row in phi]
        global_attr = dict(zip(group_names, phi.T @ w))
        table = self._build_table(group_names, attributions, global_attr,
                                  global_noise=float((1.0 - r2_cv) @ w))
        fig = self._budget_figure(
            table, group_names, figure_type=figure_type, color_of=viz._color,
            col_label="group", top_k=top_k, top_n=top_n, figsize=figsize,
            null_line=False, scree=scree, total=total,
        )
        viz.save_figure(fig, fig_out, "shap_cv")
        return Report(table=table, rho=float(r2_cv @ w), figure=fig)

    def _group_result(self, feat_to_group: dict | None):
        """Result and group names for a grouping, recomputing if overridden.

        ``feat_to_group=None`` (or equal to the constructor value) returns the
        cached fit and its groups.  Any other ``{feature -> group}`` mapping
        triggers a fresh Shapley group attribution for that grouping; the
        eigendecomposition and per-direction ``R^2`` are unchanged, so only the
        group-level shares (and ``global_group_shares``) differ.
        """
        self._fit()  # populate direction labels and the grouping-invariant summaries
        if feat_to_group is None or feat_to_group == self.feat_to_group:
            return self._result, self.group_names
        groups = self._invert_mapping(self.feature_names, feat_to_group)
        result = decompose_latent_space(
            self.df_latent.values, self.df_feat.values,
            self.feature_names, groups,
            mode=self.mode, n_directions=self.n_directions,
            cv_folds=self.cv_folds, compute_feature_weights=False, seed=self.seed,
        )
        return result, list(groups.keys())

    def _build_table(
        self,
        columns: list[str],
        attributions: list[dict],
        global_attr: dict,
        *,
        global_noise: float | None = None,
    ) -> pd.DataFrame:
        """Assemble a ``(d+1) x (len(columns)+1)`` budget table.

        ``attributions`` is one ``{column: share}`` dict per direction (shares sum
        to that direction's ``R^2``); the appended ``noise`` column is the
        unexplained remainder, so every row sums to 1.  ``global_attr`` is the
        eigenvalue-weighted aggregate used for the final ``weighted_sum`` row, whose
        noise entry defaults to the in-sample ``global_noise`` of the fit and can be
        overridden when the shares partition a different total.
        """
        res = self._result
        rows = []
        index = []
        for label, attr in zip(self._dir_labels, attributions):
            shares = [float(attr.get(c, 0.0)) for c in columns]
            rows.append(shares + [1.0 - float(sum(shares))])  # ... + noise
            index.append(label)

        # Eigenvalue-weighted sum across directions (the global budget).
        global_shares = [float(global_attr.get(c, 0.0)) for c in columns]
        rows.append(global_shares + [res.global_noise if global_noise is None
                                     else float(global_noise)])
        index.append(_WEIGHTED_SUM_LABEL)

        table = pd.DataFrame(rows, index=index, columns=list(columns) + [_NOISE_COL])
        table.index.name = "direction"
        return table

    def _global_feature_shares(self) -> dict:
        """Variance-weighted global relative weight of each feature.

        Mirrors :attr:`DecompositionResult.global_group_shares` at the feature
        level: ``sum_k (lambda_k / sum lambda) * eps_{k,j}``, summing over features
        to the global ``R^2``.
        """
        res = self._res
        vfrac = np.array([d.variance_frac for d in res.directions])
        vfrac = vfrac / vfrac.sum()
        return {
            f: float(sum(w * d.feature_weights.get(f, 0.0)
                         for w, d in zip(vfrac, res.directions)))
            for f in self.feature_names
        }

    def _null_floor_shape(self):
        """``(n, q)`` for the noise-floor line: the shape the fit actually saw.

        The same pair :func:`~fva.decomposition.decompose_latent_space` passes to
        ``check_sample_size``, so the line on the figure and the warning on the
        fit are guaranteed to quote the same floor.  ``df_latent`` is the aligned
        frame, so ``n`` already accounts for any rows dropped by :meth:`_align`.
        """
        return len(self.df_latent), self.df_feat.shape[1]

    def _global_bar(self, shares: dict, noise: float, *, top_n, color_of,
                    figsize, title, null_line=None):
        """Draw one stacked bar: ``shares`` (largest first) + ``other`` + noise."""
        import matplotlib.pyplot as plt

        items = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        other = 0.0
        if top_n is not None and len(items) > top_n:
            other = float(sum(v for _, v in items[top_n:]))
            items = items[:top_n]

        fig, ax = plt.subplots(figsize=figsize)
        bottom = 0.0
        for i, (name, h) in enumerate(items):
            ax.bar(0, h, bottom=bottom, width=0.6, color=color_of(name, i),
                   edgecolor="white", label=f"{name} ({h*100:.1f}%)")
            bottom += h
        if other > 0:
            ax.bar(0, other, bottom=bottom, width=0.6, color="#bbbbbb",
                   edgecolor="white", label=f"other ({other*100:.1f}%)")
            bottom += other
        ax.bar(0, noise, bottom=bottom, width=0.6, color="#dddddd",
               edgecolor="white", label=f"noise ({noise*100:.1f}%)")
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of total latent variance")
        if title:
            ax.set_title(title)
        viz.draw_null_floor(ax, *self._null_floor_shape(), null_line=null_line)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig

    def _feature_color(self):
        """Color function keying each feature to its group's stable color."""
        feat_group = {f: g for g, fs in self.groups.items() for f in fs}
        group_idx = {g: i for i, g in enumerate(self.group_names)}
        def color_of(name, i):
            g = feat_group.get(name, name)
            return viz._color(g, group_idx.get(g, i))
        return color_of

    @staticmethod
    def _rank_columns(table: pd.DataFrame, columns: list[str], top_n: int | None):
        """Split ``columns`` into (top-``n``, rest) by their ``weighted_sum`` share."""
        global_row = table.loc[_WEIGHTED_SUM_LABEL, columns].astype(float)
        ordered = list(global_row.sort_values(ascending=False).index)
        if top_n is None or top_n >= len(ordered):
            return ordered, []
        return ordered[:top_n], ordered[top_n:]

    def _budget_figure(self, table, columns, *, figure_type, color_of, col_label,
                       top_k, top_n, figsize, null_line=None, scree=False,
                       total=False):
        """Dispatch to the stacked-bar or heatmap renderer for a budget table.

        ``null_line``, ``scree`` and ``total`` reach the bar only: the heatmap's cells are
        shares *of* ``R^2``, a different scale from ``R^2`` itself, so there is no
        height on it for either annotation to mark.
        """
        if figure_type == "bar":
            return self._budget_bar(table, columns, color_of=color_of, top_k=top_k,
                                    top_n=top_n, figsize=figsize,
                                    null_line=null_line, scree=scree, total=total)
        if figure_type == "heatmap":
            return self._budget_heatmap(table, columns, col_label=col_label,
                                        top_n=top_n, figsize=figsize)
        raise ValueError(
            f"figure_type must be 'bar' or 'heatmap', got {figure_type!r}")

    def _budget_bar(self, table, columns, *, color_of, top_k, top_n, figsize,
                    null_line=None, scree=False, total=False):
        """Stacked bar per direction: attribution shares (+ ``other``) + noise.

        ``scree=True`` overlays the directions' eigenvalue weights as a black
        line on the same axis (see :func:`fva.viz.draw_scree_line`).
        ``total=True`` adds the ``weighted_sum`` row as a final bar on the right,
        after a gap and a dashed separator; the scree line skips it.
        """
        import matplotlib.pyplot as plt

        top, rest = self._rank_columns(table, columns, top_n)
        dir_rows = [lbl for lbl in table.index if lbl != _WEIGHTED_SUM_LABEL]
        if top_k:
            dir_rows = dir_rows[:top_k]
        vfrac = self.variance_fraction
        x_dirs = np.arange(len(dir_rows))
        rows, x = list(dir_rows), x_dirs.astype(float)
        if total:
            rows.append(_WEIGHTED_SUM_LABEL)
            x = np.append(x, len(dir_rows) + 0.5)
        bottom = np.zeros(len(rows))

        fig, ax = plt.subplots(figsize=figsize)
        for i, c in enumerate(top):
            h = table.loc[rows, c].astype(float).values
            ax.bar(x, h, bottom=bottom, label=c, color=color_of(c, i), width=0.85)
            bottom += h
        if rest:
            other = table.loc[rows, rest].astype(float).sum(axis=1).values
            ax.bar(x, other, bottom=bottom, label="other", color="#bbbbbb",
                   width=0.85, edgecolor="white")
            bottom += other
        noise = table.loc[rows, _NOISE_COL].astype(float).values
        ax.bar(x, noise, bottom=bottom, label="noise (unexplained)",
               color="#dddddd", width=0.85, edgecolor="white")
        ax.set_xlabel("latent direction (ordered by variance)")
        ax.set_ylabel("fraction of direction variance")
        ax.set_title("Feature-grounded variance budget per latent direction")
        ax.set_xticks(x)
        ticks = [f"{lbl}\n{vfrac[lbl]*100:.0f}%" for lbl in dir_rows]
        if total:
            ticks.append("total\n100%")
            ax.axvline(len(dir_rows) - 0.25, color="#888888", linestyle="--",
                       linewidth=0.8)
        ax.set_xticklabels(ticks, fontsize=8)
        viz.draw_null_floor(ax, *self._null_floor_shape(), null_line=null_line)
        if scree:
            viz.draw_scree_line(ax, x_dirs, [vfrac[lbl] for lbl in dir_rows])
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig

    def _budget_heatmap(self, table, columns, *, col_label, top_n, figsize):
        """Heatmap of per-direction shares for the top-``n`` ``columns``."""
        import matplotlib.pyplot as plt

        top, _ = self._rank_columns(table, columns, top_n)
        dir_rows = [lbl for lbl in table.index if lbl != _WEIGHTED_SUM_LABEL]
        mat = table.loc[dir_rows, top].astype(float).values

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([str(t).replace("_", " ") for t in top],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(dir_rows)))
        ax.set_yticklabels(dir_rows, fontsize=8)
        ax.set_xlabel(col_label)
        ax.set_ylabel("latent direction")
        ax.set_title(f"Variance budget by {col_label} (share of $R^2$) per direction")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of $R^2$")
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        n, d = self.df_latent.shape
        q = self.df_feat.shape[1]
        g = len(self.group_names)
        fitted = self._result is not None
        return (f"FeatureGroundedDecomposition(n={n}, d={d}, q={q}, groups={g}, "
                f"mode={self.mode!r}, fitted={fitted})")
