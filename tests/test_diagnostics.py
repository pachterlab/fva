"""Tests for the data-adequacy guards in :mod:`fva._diagnostics`.

These cover the regime the rest of the suite deliberately stays out of: too few
samples for the width of the feature bank, a rank-deficient bank, a fold count
the sample cannot support, and the in-sample/cross-validated divergence those
produce.  The point of each guard is that the arithmetic downstream *succeeds* --
the Shapley identities are algebraic and hold regardless -- so the warning is the
only signal that the numbers are not worth reading.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fva import FeatureGroundedDecomposition
from fva._diagnostics import (
    NULL_R2_LINE_MIN,
    OVERFIT_GAP_WARN,
    SAMPLES_PER_FEATURE_WARN,
    check_overfit_gap,
    check_sample_size,
    null_r2,
)
from fva.decomposition import _test_sets, decompose_latent_space, relative_weights
from fva.viz import draw_null_floor


def _noise_problem(n, q, d=4, seed=0):
    """Latent codes and features that are independent of each other by construction."""
    rng = np.random.default_rng(seed)
    names = [f"f{i}" for i in range(q)]
    df_latent = pd.DataFrame(rng.standard_normal((n, d)))
    df_feat = pd.DataFrame(rng.standard_normal((n, q)), columns=names)
    groups = {name: f"fam{i % 3}" for i, name in enumerate(names)}
    return df_latent, df_feat, groups


def _wide_but_signalled(n=200, q=20, d=3, seed=7):
    """A bank wide enough for the null floor to matter, carrying real signal.

    ``n / q`` lands exactly on ``SAMPLES_PER_FEATURE_WARN``, so ``check_sample_size``
    stays quiet while ``null_r2`` clears ``NULL_R2_LINE_MIN`` -- the band between
    the two thresholds that the line exists to cover, where the figure is the only
    thing that says anything.
    """
    rng = np.random.default_rng(seed)
    names = [f"f{i}" for i in range(q)]
    F = rng.normal(size=(n, q))
    Z = F @ rng.normal(size=(q, d)) + 0.5 * rng.normal(size=(n, d))
    return (pd.DataFrame(Z, columns=[f"z{i}" for i in range(d)]),
            pd.DataFrame(F, columns=names),
            {name: f"fam{i % 4}" for i, name in enumerate(names)})


def _floor_ys(fig):
    """The ``y`` of every horizontal line on a figure's primary axes.

    The budget figures are drawn entirely out of patches (bars, heatmap cells),
    so any Line2D on the axes is the noise floor and nothing else.
    """
    return [float(np.unique(line.get_ydata())[0]) for line in fig.axes[0].lines]


# ------------------------------------------------------------------- null_r2
def test_null_r2_matches_empirical_inflation():
    """q/(n-1) predicts the in-sample R^2 that a pure-noise bank actually scores."""
    n, q = 120, 40
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, q))
    observed = []
    for _ in range(40):
        y = rng.standard_normal(n)
        y = y - y.mean()
        beta, *_ = np.linalg.lstsq(X - X.mean(0), y, rcond=None)
        resid = y - (X - X.mean(0)) @ beta
        observed.append(1.0 - (resid @ resid) / (y @ y))
    assert np.mean(observed) == pytest.approx(null_r2(n, q), abs=0.05)


def test_null_r2_saturates_at_one():
    """Once q >= n the bank spans every score, so the predicted bias is total."""
    assert null_r2(50, 50) == 1.0
    assert null_r2(50, 200) == 1.0
    assert null_r2(1, 5) == 1.0


# --------------------------------------------------------------- sample size
def test_sample_size_quiet_when_ample():
    """A comfortable n/q ratio must not warn; these guards have to stay quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_sample_size(n=10_000, q=20)


def test_sample_size_warns_below_ratio():
    """Fewer than SAMPLES_PER_FEATURE_WARN samples per feature is flagged."""
    q = 40
    n = int(SAMPLES_PER_FEATURE_WARN * q) - 40
    with pytest.warns(UserWarning, match="per feature"):
        check_sample_size(n=n, q=q)


def test_sample_size_warns_hard_when_saturated():
    """q >= n gets the stronger message: the budget is vacuous, not just biased."""
    with pytest.warns(UserWarning, match="in-sample R\\^2 is 1 by construction"):
        check_sample_size(n=30, q=40)


def test_sample_size_ignores_empty_problems():
    """Degenerate shapes are someone else's error to raise, not ours to warn about."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_sample_size(n=0, q=0)
        check_sample_size(n=10, q=0)


# ------------------------------------------------------------- overfitting gap
def test_overfit_gap_quiet_when_totals_agree():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_overfit_gap(0.62, 0.61)


def test_overfit_gap_warns_when_totals_diverge():
    with pytest.warns(UserWarning, match="cross-validated rho_cv"):
        check_overfit_gap(0.9, 0.9 - OVERFIT_GAP_WARN - 0.05)


def test_overfit_gap_ignores_cv_above_in_sample():
    """A negative gap is noise in the CV estimate, not evidence of overfitting."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_overfit_gap(0.40, 0.45)


# ----------------------------------------------------------- end-to-end wiring
def test_saturated_bank_warns_but_still_partitions():
    """q > n: the budget stays exactly additive, and the warning is the only tell."""
    df_latent, df_feat, groups = _noise_problem(n=40, q=60)
    with pytest.warns(UserWarning) as record:
        expl = FeatureGroundedDecomposition(df_latent, df_feat, groups,
                                            progress=False)
        table, rho, _ = expl.shap()

    messages = " ".join(str(w.message) for w in record)
    assert "in-sample R^2 is 1 by construction" in messages
    assert "cross-validated rho_cv" in messages

    # The failure is silent in the numbers: R^2 saturates (up to the Gram cache's
    # tiny ridge load, which shrinks a perfect fit by a hair) and the partition holds.
    assert rho == pytest.approx(1.0, abs=1e-4)
    assert np.allclose(table.sum(axis=1).values, 1.0, atol=1e-6)


def test_healthy_problem_raises_no_diagnostics(dataset):
    """The shared fixture must stay clean, or every other test drowns in warnings."""
    df_latent, df_feat, feat_to_group = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        expl = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                            progress=False)
        expl.shap()


def test_functional_entry_point_warns_too():
    """decompose_latent_space is public, so it carries the guards on its own."""
    df_latent, df_feat, groups = _noise_problem(n=50, q=80)
    inverted = {}
    for feat, grp in groups.items():
        inverted.setdefault(grp, []).append(feat)
    with pytest.warns(UserWarning) as record:
        decompose_latent_space(df_latent.values, df_feat.values,
                               list(df_feat.columns), inverted, progress=False)
    assert any("in-sample R^2 is 1 by construction" in str(w.message) for w in record)


# ----------------------------------------------------------- null-floor line
def test_null_floor_auto_draws_when_the_bias_is_material():
    """The band the line is for: too small a ratio to warn, big enough to see."""
    df_latent, df_feat, groups = _wide_but_signalled()
    expl = FeatureGroundedDecomposition(df_latent, df_feat, groups, progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # nothing else may fire here
        _, _, fig = expl.shap()
    assert _floor_ys(fig) == [pytest.approx(null_r2(200, 20))]


def test_null_floor_auto_stays_off_when_the_bank_is_healthy(explainer):
    """A floor at 3.5% would sit on the axis spine and read as an artifact."""
    assert null_r2(200, 7) < NULL_R2_LINE_MIN          # the fixture's shape
    _, _, fig = explainer.shap()
    assert _floor_ys(fig) == []


def test_null_floor_can_be_forced_on_below_the_threshold(explainer):
    """Panels compared side by side need the line present in every one of them."""
    _, _, fig = explainer.shap(null_line=True)
    assert _floor_ys(fig) == [pytest.approx(null_r2(200, 7))]


def test_null_floor_can_be_forced_off_above_the_threshold():
    df_latent, df_feat, groups = _wide_but_signalled()
    expl = FeatureGroundedDecomposition(df_latent, df_feat, groups, progress=False)
    _, _, fig = expl.shap(null_line=False)
    assert _floor_ys(fig) == []


def test_null_floor_is_labelled_for_the_legend(explainer):
    _, _, fig = explainer.shap(null_line=True)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("noise floor" in lbl for lbl in labels)


def test_null_floor_reaches_the_johnson_bar_and_the_global_budget(explainer):
    """Every figure with an R^2 height on it, not just shap()'s."""
    _, _, fig = explainer.johnson(figure_type="bar", null_line=True)
    assert _floor_ys(fig) == [pytest.approx(null_r2(200, 7))]
    for by in ("group", "feature"):
        fig = explainer.global_budget(by=by, null_line=True)
        assert _floor_ys(fig) == [pytest.approx(null_r2(200, 7))]


def test_heatmap_has_no_height_for_the_floor(explainer):
    """Cells are shares *of* R^2, a different scale; the request is simply moot."""
    _, _, fig = explainer.shap(figure_type="heatmap", null_line=True)
    assert _floor_ys(fig) == []


def test_null_floor_is_never_drawn_on_a_cross_validated_budget(explainer):
    """q/(n-1) is the in-sample bias; out of fold a noise bank scores about zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, _, fig = explainer.shap(cv=True)
    assert _floor_ys(fig) == []


def test_forcing_the_floor_onto_a_cv_budget_warns_and_is_refused(explainer):
    """Silently ignoring it would leave the caller believing a floor was checked."""
    with pytest.warns(UserWarning, match="cross-validated budget"):
        _, _, fig = explainer.shap(cv=True, null_line=True)
    assert _floor_ys(fig) == []


def test_draw_null_floor_reports_what_it_drew():
    """The return value is the contract the renderers are tested through."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    assert draw_null_floor(ax, 200, 20) == pytest.approx(null_r2(200, 20))
    assert draw_null_floor(ax, 200, 7) is None                    # below threshold
    assert draw_null_floor(ax, 200, 7, null_line=True) == pytest.approx(7 / 199)
    assert draw_null_floor(ax, 200, 20, null_line=False) is None
    assert draw_null_floor(ax, 200, 0, null_line=True) is None    # no features, no floor
    plt.close(fig)


def test_null_floor_threshold_is_inclusive():
    """NULL_R2_LINE_MIN is a floor to clear, not to exceed; keeps the band closed."""
    import matplotlib.pyplot as plt

    n = 201                                    # q / 200 == NULL_R2_LINE_MIN exactly
    q = round(NULL_R2_LINE_MIN * (n - 1))
    fig, ax = plt.subplots()
    assert draw_null_floor(ax, n, q) == pytest.approx(NULL_R2_LINE_MIN)
    plt.close(fig)


# ------------------------------------------------------------ rank deficiency
def test_relative_weights_warns_on_duplicated_feature():
    """An exactly duplicated column makes R_ff singular; Johnson's basis is then ill-defined."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    X = np.column_stack([X, X[:, 0]])            # exact collinearity
    X = (X - X.mean(0)) / X.std(0)
    y = rng.standard_normal(200)
    with pytest.warns(UserWarning, match="rank deficient"):
        relative_weights(X, y)


def test_relative_weights_quiet_on_full_rank_bank():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 5))
    X = (X - X.mean(0)) / X.std(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        relative_weights(X, rng.standard_normal(200))


def test_relative_weights_still_sums_to_r2_when_deficient():
    """The additivity identity is algebraic, which is exactly why it cannot warn us."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 3))
    X = np.column_stack([X, X[:, 1]])
    X = (X - X.mean(0)) / X.std(0)
    y = X @ np.array([1.0, 0.5, -1.0, 0.5]) + 0.3 * rng.standard_normal(100)
    with pytest.warns(UserWarning, match="rank deficient"):
        eps, r2 = relative_weights(X, y - y.mean())
    assert eps.sum() == pytest.approx(r2, rel=1e-6)


# ------------------------------------------------------------------- CV folds
def test_folds_exceeding_samples_is_an_error():
    """Beyond n folds np.array_split emits empty test sets that silently score 0."""
    with pytest.raises(ValueError, match="exceeds the 4 available samples"):
        _test_sets(4, 10, seed=0)


def test_tiny_folds_warn():
    with pytest.warns(UserWarning, match="held-out fold"):
        _test_sets(9, 5, seed=0)


def test_holdout_fraction_leaving_no_training_data_is_an_error():
    with pytest.raises(ValueError, match="no training data"):
        _test_sets(50, 0.99, seed=0)


def test_cross_validation_needs_two_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        _test_sets(1, 5, seed=0)


def test_usable_fold_settings_stay_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        folds = _test_sets(100, 5, seed=0)
    assert sum(len(f) for f in folds) == 100


# ----------------------------------------------------------------- r2_ci wiring
def test_bootstrap_intervals_are_reachable_from_the_explainer(dataset):
    """n_bootstrap was previously unreachable from the public class."""
    df_latent, df_feat, feat_to_group = dataset
    expl = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                        n_bootstrap=25, seed=0, progress=False)
    ci = expl.r2_ci
    assert list(ci.columns) == ["lo", "hi"]
    assert len(ci) == len(expl.directions)
    assert (ci["lo"] <= ci["hi"]).all()
    # The point estimate should sit inside its own interval.
    point = expl.r2.drop("weighted_sum")
    assert ((ci["lo"] <= point + 1e-9) & (point <= ci["hi"] + 1e-9)).all()


def test_r2_ci_indexes_only_the_analysed_directions(dataset):
    """axes mode + n_directions: the labels must match the k rows, not all d columns."""
    df_latent, df_feat, feat_to_group = dataset
    expl = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                        mode="axes", n_directions=2,
                                        n_bootstrap=15, seed=0, progress=False)
    ci = expl.r2_ci
    assert len(ci) == 2
    assert list(ci.index) == list(df_latent.var().sort_values(ascending=False).index[:2])


def test_r2_ci_without_bootstrap_raises(explainer):
    """An empty table would read as 'no uncertainty' rather than 'not computed'."""
    with pytest.raises(AttributeError, match="n_bootstrap"):
        explainer.r2_ci


def test_r2_ci_absence_does_not_break_introspection(explainer):
    """AttributeError, not ValueError: the tutorial tours the API with getmembers."""
    import inspect

    assert not hasattr(explainer, "r2_ci")
    names = [name for name, _ in inspect.getmembers(explainer)]
    assert "rho" in names and "r2_ci" not in names
