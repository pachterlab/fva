"""Unit tests for the dataframe-first API :class:`FeatureGroundedDecomposition`.

These cover the public surface used in the tutorial: the headline scalars, the
SHAP and Johnson budget tables (and the additive identities that every row sums
to 1), the object-level vectors, the override hooks, input alignment, and the
figure-returning report methods.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fva import FeatureGroundedDecomposition
from fva.explainer import _NOISE_COL, _WEIGHTED_SUM_LABEL


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- scalars
def test_rho_in_unit_interval(explainer):
    assert 0.0 <= explainer.rho <= 1.0
    assert explainer.rho > 0.8  # mixture data is mostly explained
    assert explainer.rho_cv <= explainer.rho + 1e-9


def test_repr_reports_shapes_and_fit_state(dataset):
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)
    assert "fitted=False" in repr(exp)
    _ = exp.rho  # triggers the lazy fit
    r = repr(exp)
    assert "fitted=True" in r
    assert f"n={len(df_latent)}" in r


# ---------------------------------------------------------------- SHAP table
def test_shap_table_shape_and_rows_sum_to_one(explainer, dataset):
    df_latent, df_feat, _ = dataset
    table, rho, fig = explainer.shap()
    d = df_latent.shape[1]
    g = len(explainer.group_names)
    assert table.shape == (d + 1, g + 1)
    assert _NOISE_COL in table.columns
    assert _WEIGHTED_SUM_LABEL in table.index
    assert np.allclose(table.sum(axis=1), 1.0)
    assert rho == pytest.approx(explainer.rho)
    assert fig is not None


def test_shap_noise_matches_r2(explainer):
    table, _, _ = explainer.shap()
    assert np.allclose(explainer.r2.values, 1.0 - table[_NOISE_COL].values)


def test_shap_override_grouping_recomputes(explainer):
    """Passing ``feat_to_group`` regroups for that call only."""
    base, _, _ = explainer.shap()
    # Pool everything into a single group: that column should carry the whole R^2.
    one = {f: "all" for f in explainer.feature_names}
    table, _, _ = explainer.shap(feat_to_group=one)
    assert list(table.columns) == ["all", _NOISE_COL]
    assert np.allclose(table.sum(axis=1), 1.0)
    # The grouping fixed at construction is unchanged afterwards.
    again, _, _ = explainer.shap()
    assert list(again.columns) == list(base.columns)


def test_shap_heatmap_figure_type(explainer):
    _, _, fig = explainer.shap(figure_type="heatmap")
    assert fig is not None


def test_shap_scree_overlay(explainer):
    """scree=True draws the eigenvalue weights as a black line on the bar."""
    vfrac = explainer.variance_fraction

    _, _, plain = explainer.shap()
    assert not [ln for ln in plain.axes[0].get_lines()
                if str(ln.get_label()).startswith("scree")]

    _, _, fig = explainer.shap(scree=True)
    lines = [ln for ln in fig.axes[0].get_lines()
             if str(ln.get_label()).startswith("scree")]
    assert len(lines) == 1
    assert lines[0].get_color() == "black"
    labels = [lbl for lbl in explainer.shap().table.index if lbl != "weighted_sum"]
    assert np.allclose(lines[0].get_ydata(), [vfrac[lbl] for lbl in labels])


def test_shap_scree_overlay_follows_top_k(explainer):
    """The overlay covers exactly the directions the bar draws."""
    _, _, fig = explainer.shap(scree=True, top_k=2)
    line, = [ln for ln in fig.axes[0].get_lines()
             if str(ln.get_label()).startswith("scree")]
    assert len(line.get_ydata()) == 2


def test_shap_invalid_figure_type_raises(explainer):
    with pytest.raises(ValueError, match="figure_type"):
        explainer.shap(figure_type="bogus")


def test_shap_many_groups_warns(dataset):
    """Attributing to more than the warn threshold of groups warns the user."""
    df_latent, df_feat, _ = dataset
    # Each feature its own group is < threshold; build a wide one to exceed it.
    rng = np.random.default_rng(0)
    F = rng.normal(size=(150, 15))
    names = [f"f{i}" for i in range(15)]
    df_f = pd.DataFrame(F, columns=names)
    Z = F @ rng.normal(size=(15, 3)) + rng.normal(size=(150, 3))
    df_l = pd.DataFrame(Z, columns=[f"z{i}" for i in range(3)])
    exp = FeatureGroundedDecomposition(df_l, df_f)  # 15 singleton groups
    with pytest.warns(UserWarning, match="exponential"):
        exp.shap()


# ---------------------------------------------------------------- Johnson table
def test_johnson_table_shape_and_rows_sum_to_one(explainer, dataset):
    df_latent, df_feat, _ = dataset
    table, rho, fig = explainer.johnson()
    d = df_latent.shape[1]
    q = df_feat.shape[1]
    assert table.shape == (d + 1, q + 1)
    assert np.allclose(table.sum(axis=1), 1.0)
    assert np.allclose(explainer.r2.values, 1.0 - table[_NOISE_COL].values)
    assert rho == pytest.approx(explainer.rho)
    assert fig is not None


def test_johnson_bar_figure_type(explainer):
    _, _, fig = explainer.johnson(figure_type="bar", top_n=3)
    assert fig is not None


# ---------------------------------------------------------------- vectors
def test_variance_fraction_and_r2_vectors(explainer, dataset):
    d = dataset[0].shape[1]
    vf = explainer.variance_fraction
    r2 = explainer.r2
    assert len(vf) == d + 1
    assert len(r2) == d + 1
    assert vf[_WEIGHTED_SUM_LABEL] == pytest.approx(1.0)  # all directions kept
    assert r2[_WEIGHTED_SUM_LABEL] == pytest.approx(explainer.rho)


def test_r2_cv_vector_matches_directions_and_rho_cv(explainer, dataset):
    """``r2_cv`` is the held-out counterpart of ``r2``, totalling to ``rho_cv``."""
    d = dataset[0].shape[1]
    r2_cv = explainer.r2_cv
    assert len(r2_cv) == d + 1
    assert list(r2_cv.index) == list(explainer.r2.index)
    assert r2_cv[_WEIGHTED_SUM_LABEL] == pytest.approx(explainer.rho_cv)
    for label, direction in zip(explainer.r2.index, explainer.directions):
        assert r2_cv[label] == pytest.approx(direction.r2_cv, abs=1e-12)
    # Held-out scores cannot exceed the in-sample ones they are estimating.
    assert (r2_cv.drop(_WEIGHTED_SUM_LABEL)
            <= explainer.r2.drop(_WEIGHTED_SUM_LABEL) + 1e-9).all()


def test_r2_cv_matches_the_standalone_screen(dataset):
    """``direction_r2`` and a fitted explainer agree on the totals."""
    from fva import FeatureGroundedDecomposition, direction_r2

    df_latent, df_feat, feat_to_group = dataset
    expl = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group, seed=0)
    table = direction_r2(df_latent, df_feat, seed=0)
    assert table.loc[_WEIGHTED_SUM_LABEL, "r2_cv"] == pytest.approx(expl.rho_cv, abs=1e-10)
    assert list(table.index) == list(expl.r2_cv.index)
    np.testing.assert_allclose(table["r2_cv"].values, expl.r2_cv.values, atol=1e-10)


def test_directions_and_eigh_internals(explainer, dataset):
    d = dataset[0].shape[1]
    assert len(explainer.directions) == d
    assert explainer.eigvals.shape == (d,)
    assert explainer.eigvecs.shape == (d, d)
    # Eigenvalues come out in descending order.
    assert np.all(np.diff(explainer.eigvals) <= 1e-9)


# ---------------------------------------------------------------- global budget / scree
def test_global_budget_group_and_feature(explainer):
    assert explainer.global_budget(by="group") is not None
    assert explainer.global_budget(by="feature", top_n=4) is not None


def test_global_budget_invalid_by_raises(explainer):
    with pytest.raises(ValueError, match="group.*feature|by must"):
        explainer.global_budget(by="bogus")


def test_scree_returns_figure(explainer):
    assert explainer.scree() is not None


# ---------------------------------------------------------------- modes / n_directions
def test_axes_mode_uses_column_labels(dataset):
    """Rows carry the latent column names, ordered as the fit orders them.

    Directions are sorted by variance in both modes, so the labels follow that
    order rather than the column order of ``df_latent``.  Labelling by position
    instead silently attaches each row to the wrong column whenever the columns
    do not already arrive sorted -- which this fixture does not.
    """
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(
        df_latent, df_feat, feat_to_group, mode="axes",
    )
    table, _, _ = exp.shap()
    dir_rows = [i for i in table.index if i != _WEIGHTED_SUM_LABEL]
    assert sorted(dir_rows) == sorted(str(c) for c in df_latent.columns)
    assert dir_rows == list(df_latent.var().sort_values(ascending=False).index)


def test_axes_mode_labels_match_reported_variance(dataset):
    """Each row's reported variance fraction is really that named column's."""
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(
        df_latent, df_feat, feat_to_group, mode="axes",
    )
    reported = exp.variance_fraction.drop(_WEIGHTED_SUM_LABEL)
    truth = df_latent.var() / df_latent.var().sum()
    for label, value in reported.items():
        assert value == pytest.approx(truth[label], rel=1e-6)


def test_axes_mode_labels_survive_n_directions(dataset):
    """Keeping the top k must not misalign labels against the k analysed rows."""
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(
        df_latent, df_feat, feat_to_group, mode="axes", n_directions=2,
    )
    expected = list(df_latent.var().sort_values(ascending=False).index[:2])
    assert list(exp.r2.drop(_WEIGHTED_SUM_LABEL).index) == expected
    assert list(exp.variance_fraction.drop(_WEIGHTED_SUM_LABEL).index) == expected
    table, _, _ = exp.shap()
    assert [i for i in table.index if i != _WEIGHTED_SUM_LABEL] == expected


def test_n_directions_limits_rows(dataset):
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(
        df_latent, df_feat, feat_to_group, n_directions=2,
    )
    table, _, _ = exp.shap()
    assert table.shape[0] == 2 + 1  # 2 directions + weighted_sum


# ---------------------------------------------------------------- grouping defaults
def test_default_grouping_is_per_feature(dataset):
    """``feat_to_group=None`` makes each feature its own singleton group."""
    df_latent, df_feat, _ = dataset
    exp = FeatureGroundedDecomposition(df_latent, df_feat)
    assert set(exp.group_names) == set(df_feat.columns)


def test_ungrouped_features_warn(dataset):
    """Features missing from the mapping land in an 'ungrouped' group with a warning."""
    df_latent, df_feat, feat_to_group = dataset
    partial = dict(feat_to_group)
    dropped = df_feat.columns[0]
    del partial[dropped]
    with pytest.warns(UserWarning, match="ungrouped"):
        exp = FeatureGroundedDecomposition(df_latent, df_feat, partial)
    assert "ungrouped" in exp.group_names


# ---------------------------------------------------------------- alignment
def test_align_by_shared_index():
    rng = np.random.default_rng(0)
    idx = [f"s{i}" for i in range(40)]
    df_l = pd.DataFrame(rng.normal(size=(40, 3)), index=idx)
    df_f = pd.DataFrame(rng.normal(size=(40, 2)), index=idx[::-1])  # same labels, shuffled
    exp = FeatureGroundedDecomposition(df_l, df_f)
    # Rows are aligned by label, so the two frames share an index after construction.
    assert exp.df_latent.index.equals(exp.df_feat.index)


def test_align_partial_overlap_warns():
    rng = np.random.default_rng(1)
    df_l = pd.DataFrame(rng.normal(size=(30, 2)), index=[f"s{i}" for i in range(30)])
    df_f = pd.DataFrame(rng.normal(size=(30, 2)), index=[f"s{i}" for i in range(10, 40)])
    with pytest.warns(UserWarning, match="share only"):
        exp = FeatureGroundedDecomposition(df_l, df_f)
    assert len(exp.df_latent) == 20  # the intersection


def test_align_no_shared_index_assumes_order():
    rng = np.random.default_rng(2)
    df_l = pd.DataFrame(rng.normal(size=(25, 2)), index=range(25))
    df_f = pd.DataFrame(rng.normal(size=(25, 2)), index=range(100, 125))
    with pytest.warns(UserWarning, match="same order"):
        exp = FeatureGroundedDecomposition(df_l, df_f)
    assert len(exp.df_latent) == 25


def test_align_mismatched_length_raises():
    rng = np.random.default_rng(3)
    df_l = pd.DataFrame(rng.normal(size=(20, 2)), index=range(20))
    df_f = pd.DataFrame(rng.normal(size=(15, 2)), index=range(100, 115))
    with pytest.raises(ValueError, match="rows"):
        FeatureGroundedDecomposition(df_l, df_f)


# ------------------------------------------------- SHAP on the cross-validated R^2
def test_shap_cv_rows_sum_to_one(explainer, dataset):
    """Efficiency survives the switch: shares + noise still partition each row."""
    df_latent, _, _ = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table, rho, fig = explainer.shap(cv=True)
    d = df_latent.shape[1]
    g = len(explainer.group_names)
    assert table.shape == (d + 1, g + 1)
    assert np.allclose(table.sum(axis=1), 1.0)
    assert fig is not None
    # rho is the variance-weighted CV R^2, at or below the in-sample headline
    assert rho <= explainer.rho + 1e-9


def test_shap_cv_noise_is_cross_validated(explainer):
    """The noise column reports 1 - CV R^2, so it exceeds the in-sample noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_table, _, _ = explainer.shap(cv=True)
    ins_table, _, _ = explainer.shap()
    dirs = [i for i in ins_table.index if i != _WEIGHTED_SUM_LABEL]
    assert (cv_table.loc[dirs, _NOISE_COL].values
            >= ins_table.loc[dirs, _NOISE_COL].values - 1e-9).all()


def test_shap_cv_warns_on_negative_share():
    """A group of pure noise costs held-out fit, and that is reported."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 6))
    df_feat = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    signal = X[:, :3] @ rng.standard_normal(3)
    df_latent = pd.DataFrame({"z0": signal, "z1": rng.standard_normal(n)})
    feat_to_group = {f"f{i}": ("signal" if i < 3 else "junk") for i in range(6)}
    exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group, seed=0)

    with pytest.warns(UserWarning, match="negative share"):
        table, _, _ = exp.shap(cv=True)
    assert np.allclose(table.sum(axis=1), 1.0)
    # the informative group takes the budget on the direction built from it
    assert table.loc["PC0", "signal"] > table.loc["PC0", "junk"]


def test_shap_cv_respects_grouping_override(explainer):
    """Pooling every feature into one group hands it the whole CV R^2."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table, rho, _ = explainer.shap(
            cv=True, feat_to_group={f: "all" for f in explainer.feature_names})
    assert list(table.columns) == ["all", _NOISE_COL]
    assert np.allclose(table.sum(axis=1), 1.0)
    assert table.loc[_WEIGHTED_SUM_LABEL, "all"] == pytest.approx(rho, abs=1e-9)


def test_shap_cv_respects_n_directions(dataset):
    """With n_directions set, the scores must be truncated to match."""
    df_latent, df_feat, feat_to_group = dataset
    k = 2
    exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                       n_directions=k, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table, _, _ = exp.shap(cv=True)
    assert table.shape[0] == k + 1                 # k directions + weighted_sum
    assert np.allclose(table.sum(axis=1), 1.0)


# ------------------------------------------------------- cv_folds splitting policy
def test_cv_folds_accepts_holdout_fraction(dataset):
    """A float cv_folds is a single hold-out; the budget still partitions."""
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                       cv_folds=0.25, seed=0)
    assert exp.rho_cv <= exp.rho + 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table, _, _ = exp.shap(cv=True)
    assert np.allclose(table.sum(axis=1), 1.0)


def test_shap_cv_warns_on_single_holdout(dataset):
    """Shapley on one split is noisy, and says so."""
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                       cv_folds=0.25, seed=0)
    with pytest.warns(UserWarning, match="single split"):
        exp.shap(cv=True)


@pytest.mark.parametrize("bad", [1, 0, -3, 0.0, 1.0])
def test_invalid_cv_folds_rejected(dataset, bad):
    """Fold counts below 2 and out-of-range fractions are errors, not silent."""
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                       cv_folds=bad, seed=0)
    with pytest.raises(ValueError, match="folds|hold-out"):
        _ = exp.rho_cv


def test_kfold_is_more_stable_than_single_holdout(dataset):
    """K-fold scores all n samples, so it varies less across seeds than one split."""
    df_latent, df_feat, feat_to_group = dataset

    def spread(cv_folds):
        vals = []
        for seed in range(6):
            exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                               cv_folds=cv_folds, seed=seed)
            vals.append(exp.rho_cv)
        return float(np.std(vals))

    assert spread(5) < spread(0.2)


# ------------------------------------------------------------ argument validation
def test_constructor_rejects_non_dataframe(dataset):
    """The dataframe-first API says what it wants when handed a bare array."""
    df_latent, df_feat, _ = dataset
    with pytest.raises(ValidationError, match="df_feat|DataFrame"):
        FeatureGroundedDecomposition(df_latent, df_feat.values)


@pytest.mark.parametrize("kwargs", [
    {"mode": "bogus"},          # not one of the two supported modes
    {"n_directions": 0},        # fewer than one direction is meaningless
    {"seed": None},             # not an int
])
def test_constructor_rejects_bad_arguments(dataset, kwargs):
    df_latent, df_feat, feat_to_group = dataset
    with pytest.raises(ValidationError):
        FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group, **kwargs)


@pytest.mark.parametrize("method,kwargs", [
    ("shap", {"figure_type": "pie"}),
    ("shap", {"top_k": 0}),
    ("johnson", {"top_n": -5}),
    ("global_budget", {"by": "direction"}),
    ("scree", {"figsize": "big"}),
])
def test_report_methods_reject_bad_arguments(explainer, method, kwargs):
    """The report methods validate their own options before doing any work."""
    with pytest.raises(ValidationError):
        getattr(explainer, method)(**kwargs)


def test_shap_total_bar(explainer):
    """total=True appends the weighted_sum row as a final bar on the right."""
    table, _, plain = explainer.shap(top_k=2)
    _, _, fig = explainer.shap(top_k=2, total=True)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert len(labels) == len(plain.axes[0].get_xticklabels()) + 1
    assert labels[-1].startswith("total")
    # the noise segment of the last bar is the weighted_sum row's noise
    noise = [c for c in ax.containers if c.get_label() == "noise (unexplained)"][0]
    assert noise.patches[-1].get_height() == pytest.approx(
        table.loc["weighted_sum", "noise"])
    assert noise.patches[-1].get_x() > noise.patches[-2].get_x() + 1


def test_johnson_total_bar(explainer):
    _, _, fig = explainer.johnson(figure_type="bar", total=True)
    assert fig.axes[0].get_xticklabels()[-1].get_text().startswith("total")


# ------------------------------------------------ spline-basis aggregation
@pytest.fixture
def spline_dataset():
    import fva
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame({"a": rng.uniform(-2, 2, n), "b": rng.normal(size=n),
                        "c": rng.normal(size=n)})
    z1 = np.sin(2 * raw["a"]) + 0.1 * rng.normal(size=n)
    z2 = raw["b"] ** 2 + raw["c"] + 0.1 * rng.normal(size=n)
    df_latent = pd.DataFrame({"z1": z1, "z2": z2, "z3": 0.1 * rng.normal(size=n)})
    df_feat, feat_to_group, _ = fva.expand_features(
        raw, spline_df=5, groups={"a": "A", "b": "BC", "c": "BC"},
        interactions=[("a", "b")])
    return df_latent, df_feat, feat_to_group


def test_johnson_sums_spline_columns_into_source(spline_dataset):
    df_latent, df_feat, feat_to_group = spline_dataset
    ex = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)
    table = ex.johnson().table
    assert list(table.columns) == ["a", "b", "c", "a x b", _NOISE_COL]
    for d, label in zip(ex.directions, table.index):
        for src in ("a", "b", "c", "a x b"):
            cols = [f for f in ex.feature_names if ex.feature_sources[f] == src]
            assert len(cols) > 1
            assert table.loc[label, src] == pytest.approx(
                sum(d.feature_weights[f] for f in cols))
    # The tensor product spans both main effects, so the bank is nearly
    # collinear and Johnson's total drifts from the Gram-cache R^2 at ~1e-5.
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-3)
    assert ex.global_budget(by="feature") is not None


def test_feature_sources_empty_disables_aggregation(spline_dataset):
    df_latent, df_feat, feat_to_group = spline_dataset
    ex = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                      feature_sources={})
    table = ex.johnson().table
    assert list(table.columns) == list(df_feat.columns) + [_NOISE_COL]


def test_feature_sources_split_across_groups_raises(spline_dataset):
    df_latent, df_feat, feat_to_group = spline_dataset
    bad = dict(feat_to_group)
    bad[df_feat.columns[0]] = "other"
    with pytest.raises(ValueError, match="more than one group"):
        FeatureGroundedDecomposition(df_latent, df_feat, bad)


def test_non_spline_columns_are_not_aggregated(dataset):
    df_latent, df_feat, feat_to_group = dataset
    ex = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)
    assert ex.source_names == list(df_feat.columns)
