"""Unit tests for the label ceiling in :mod:`fva.decomposition`.

The ceiling claims to be an upper bound on what a family of discrete labels can
explain about a direction, so most of these tests check the bound rather than a
particular number: that it is attained when the labels determine the score, that
no fit of the same labels beats it however the labels are expanded, and that it
falls exactly as far as the unlabeled part of the sample carries variance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fva import FeatureGroundedDecomposition, label_ceiling
from fva.decomposition import _cell_codes, _saturated_r2


def _eta_squared(s, codes):
    """Between-cell variance over total variance, computed the long way."""
    grand = s.mean()
    between = sum(len(s[codes == c]) * (s[codes == c].mean() - grand) ** 2
                  for c in np.unique(codes))
    return between / ((s - grand) ** 2).sum()


# ------------------------------------------------------------------- the bound
def test_ceiling_is_one_when_labels_determine_the_score():
    """A score that is constant within every cell is fully reachable."""
    rng = np.random.default_rng(0)
    L = rng.integers(0, 3, size=(300, 2)).astype(float)
    codes = _cell_codes(L, max_levels=20)
    s = np.array([1.5, -2.0, 0.25, 4.0, -1.0, 3.0, 0.0, 2.5, -3.5])[codes]
    c = label_ceiling(s, L, unlabeled_value=None)
    assert c.ceiling_r2 == pytest.approx(1.0)


def test_ceiling_equals_between_cell_variance_share():
    """The saturated fit is the one-way ANOVA, so the ceiling is eta squared."""
    rng = np.random.default_rng(1)
    L = rng.integers(0, 4, size=(500, 1)).astype(float)
    s = 2.0 * L[:, 0] + rng.normal(size=500)
    c = label_ceiling(s, L, unlabeled_value=None)
    assert c.ceiling_r2 == pytest.approx(_eta_squared(s, _cell_codes(L, max_levels=20)))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_no_expansion_of_the_labels_beats_the_ceiling(degree):
    """The bound is over every function of the labels, not one hypothesis class.

    Polynomial and interaction expansions of the same labels are all measurable
    with respect to them, so none of them can explain more than the cell means.
    """
    rng = np.random.default_rng(2)
    L = rng.integers(0, 4, size=(400, 2)).astype(float)
    s = L[:, 0] * L[:, 1] + rng.normal(size=400)  # deliberately non-additive

    cols = [np.ones(len(s))]
    for a in range(degree + 1):
        for b in range(degree + 1):
            cols.append(L[:, 0] ** a * L[:, 1] ** b)
    X = np.column_stack(cols)
    pred = X @ np.linalg.lstsq(X, s, rcond=None)[0]
    fitted_r2 = 1 - ((s - pred) ** 2).sum() / ((s - s.mean()) ** 2).sum()

    c = label_ceiling(s, L, unlabeled_value=None)
    assert fitted_r2 <= c.ceiling_r2 + 1e-9


def test_ceiling_bounds_the_family_fit_on_random_data():
    """Holds for labels that carry no signal at all, where both numbers are small."""
    rng = np.random.default_rng(3)
    L = rng.integers(0, 2, size=(600, 3)).astype(float)
    s = rng.normal(size=600)
    X = (L - L.mean(0)) / L.std(0)
    pred = X @ np.linalg.lstsq(X, s - s.mean(), rcond=None)[0]
    linear_r2 = 1 - ((s - s.mean() - pred) ** 2).sum() / ((s - s.mean()) ** 2).sum()
    assert linear_r2 <= label_ceiling(s, L, unlabeled_value=None).ceiling_r2 + 1e-9


# ---------------------------------------------------------------------- coverage
def test_unlabeled_variance_is_out_of_reach():
    """Variance inside the unlabeled cell is subtracted from the ceiling exactly."""
    rng = np.random.default_rng(4)
    n = 800
    L = np.zeros((n, 1))
    L[:200, 0] = 1.0                      # a quarter of the sample is labeled
    s = rng.normal(size=n)
    s[:200] += 5.0                        # ... and perfectly separated by the label
    c = label_ceiling(s, L)
    assert c.coverage == pytest.approx(0.25)
    assert c.n_labeled == 200
    assert c.ceiling_r2 <= 1.0 - c.unlabeled_var_share + 1e-12
    # Within the labeled part there is only one cell, so nothing is explained
    # there and no gap between labeled cells exists to report.
    assert c.r2_labeled_only == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(c.max_cell_gap_sd)


def test_full_coverage_leaves_only_within_cell_variance():
    """With labels everywhere the shortfall is resolution, not coverage."""
    rng = np.random.default_rng(5)
    L = rng.integers(1, 4, size=(500, 1)).astype(float)   # no zeros: all labeled
    s = L[:, 0] + rng.normal(size=500)
    c = label_ceiling(s, L)
    assert c.coverage == pytest.approx(1.0)
    assert c.unlabeled_var_share == 0.0
    assert 0.0 < c.ceiling_r2 < 1.0


def test_missing_rows_are_unlabeled_even_without_a_sentinel_value():
    """A one-hot bank codes its reference level as zeros, but NaN is still absent."""
    rng = np.random.default_rng(15)
    L = np.zeros((300, 2))
    L[:100, 0] = 1.0                      # level A; rows 100-199 are the reference
    L[200:] = np.nan                      # ... and the rest were never measured
    c = label_ceiling(rng.normal(size=300), L, unlabeled_value=None)
    assert c.n_labeled == 200
    assert c.coverage == pytest.approx(2 / 3)


def test_unlabeled_value_none_counts_every_observed_row_as_labeled():
    """Coverage is a reading of the labels; the ceiling does not depend on it."""
    rng = np.random.default_rng(6)
    L = rng.integers(0, 2, size=(300, 2)).astype(float)
    s = L[:, 0] + rng.normal(size=300)
    with_zero = label_ceiling(s, L)
    without = label_ceiling(s, L, unlabeled_value=None)
    assert with_zero.coverage < 1.0
    assert without.coverage == 1.0
    assert without.ceiling_r2 == pytest.approx(with_zero.ceiling_r2)


def test_labeled_cell_gap_is_reported_in_standard_deviations():
    rng = np.random.default_rng(7)
    n = 600
    L = np.zeros((n, 2))
    L[:200, 0] = 1.0                      # group A
    L[200:400, 1] = 1.0                   # group B
    s = rng.normal(size=n)
    s[:200] += 1.0
    s[200:400] -= 1.0
    c = label_ceiling(s, L)
    assert c.n_cells == 3                 # A, B and the unlabeled remainder
    assert c.max_cell_gap_sd == pytest.approx(2.0 / s.std(), rel=0.15)


# --------------------------------------------------------------- discreteness
def test_continuous_column_is_refused_by_name():
    rng = np.random.default_rng(8)
    L = np.column_stack([rng.integers(0, 2, 200), rng.normal(size=200)])
    with pytest.raises(ValueError, match="tumor_volume"):
        label_ceiling(rng.normal(size=200), L, names=["mutated", "tumor_volume"])


def test_max_levels_can_be_raised_for_a_wide_categorical():
    rng = np.random.default_rng(9)
    L = rng.integers(0, 30, size=(600, 1)).astype(float)
    s = rng.normal(size=600)
    with pytest.raises(ValueError, match="max_levels"):
        label_ceiling(s, L)
    assert label_ceiling(s, L, max_levels=40).n_cells == 30


def test_missing_values_form_one_level_not_one_cell_per_row():
    """NaN never equals itself, so it has to be mapped to a level explicitly."""
    rng = np.random.default_rng(14)
    L = np.tile([[0.0], [1.0], [np.nan]], (40, 1))
    c = label_ceiling(rng.normal(size=len(L)), L, unlabeled_value=None)
    assert c.n_cells == 3


# ------------------------------------------------------------ small-sample guards
def test_one_cell_per_sample_is_degenerate_and_warns():
    """The bound is vacuous once the labels cut the sample as finely as it is."""
    rng = np.random.default_rng(10)
    L = np.arange(40, dtype=float)[:, None]
    s = rng.normal(size=40)
    with pytest.warns(UserWarning, match="bounds nothing"):
        c = label_ceiling(s, L, max_levels=100, unlabeled_value=None)
    assert c.ceiling_r2 == pytest.approx(1.0)
    assert c.null_r2 == pytest.approx(1.0)


def test_many_cells_warn_and_the_cross_validated_ceiling_stays_honest():
    """In-sample the cells buy R^2 they cannot reproduce out of fold."""
    rng = np.random.default_rng(11)
    L = rng.integers(0, 2, size=(120, 6)).astype(float)   # up to 64 cells on n=120
    s = rng.normal(size=120)
    with pytest.warns(UserWarning, match="degrees of freedom"):
        c = label_ceiling(s, L, unlabeled_value=None)
    assert c.ceiling_r2 > 0.3          # inflated by the cell count alone
    assert c.ceiling_r2_cv < 0.1       # and the inflation does not survive
    assert c.null_r2 > 0.4


def test_cross_validated_ceiling_recovers_real_structure():
    """The honest ceiling is not simply pessimistic: real separation survives it."""
    rng = np.random.default_rng(12)
    L = rng.integers(0, 3, size=(600, 1)).astype(float)
    s = 3.0 * L[:, 0] + rng.normal(size=600)
    c = label_ceiling(s, L, unlabeled_value=None)
    assert c.ceiling_r2_cv == pytest.approx(c.ceiling_r2, abs=0.05)


def test_saturated_r2_is_zero_for_a_constant_score():
    codes = np.array([0, 0, 1, 1])
    assert _saturated_r2(np.zeros(4), codes) == 0.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="samples"):
        label_ceiling(np.zeros(10), np.zeros((9, 1)))


# ------------------------------------------------------------------- explainer
def _mixed_dataset(n: int = 500, seed: int = 0):
    """Latents driven by a discrete clinical family and a continuous one."""
    rng = np.random.default_rng(seed)
    stage = rng.integers(1, 4, n).astype(float)
    mutated = rng.integers(0, 2, n).astype(float)
    volume = rng.normal(size=n)
    df_feat = pd.DataFrame({"stage": stage, "mutated": mutated, "volume": volume})
    Z = np.column_stack([
        2.0 * stage + 0.5 * rng.normal(size=n),
        volume + 0.2 * rng.normal(size=n),
        rng.normal(size=n),
    ])
    df_latent = pd.DataFrame(Z, columns=["z0", "z1", "z2"])
    feat_to_group = {"stage": "clinical", "mutated": "clinical", "volume": "shape"}
    return df_latent, df_feat, feat_to_group


def test_ceilings_report_shape_and_bound():
    df_latent, df_feat, feat_to_group = _mixed_dataset()
    decomp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                          mode="axes", seed=0)
    with pytest.warns(UserWarning, match="not discrete"):
        rep = decomp.ceilings()

    assert list(rep.table.columns) == ["clinical"]      # the continuous family is skipped
    # "axes" mode orders the rows by variance, so only the last row is fixed.
    assert sorted(rep.table.index[:-1]) == ["z0", "z1", "z2"]
    assert rep.table.index[-1] == "weighted_sum"
    assert rep.alone.shape == rep.table.shape
    # The bound, direction by direction, including the weighted sum.
    assert (rep.alone.values <= rep.table.values + 1e-9).all()
    assert (rep.attainment.values <= 1.0 + 1e-9).all()
    assert rep.coverage.loc["clinical", "n_cells"] == 6
    assert 0.0 < rep.coverage.loc["clinical", "coverage"] <= 1.0


def test_ceilings_weighted_sum_matches_the_variance_weights():
    df_latent, df_feat, feat_to_group = _mixed_dataset()
    decomp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                          mode="axes", seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = decomp.ceilings()
    w = decomp.variance_fraction.drop("weighted_sum").values
    w = w / w.sum()
    expected = float(w @ rep.table.drop("weighted_sum").values[:, 0])
    assert rep.table.loc["weighted_sum", "clinical"] == pytest.approx(expected)


def test_ceilings_cross_validated_switch():
    df_latent, df_feat, feat_to_group = _mixed_dataset()
    decomp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                          mode="axes", seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        in_sample = decomp.ceilings()
        held_out = decomp.ceilings(cv=True)
    # z0 is a linear function of stage, so both agree there; the cross-validated
    # ceiling cannot exceed the in-sample one by more than sampling noise.
    assert held_out.table.loc["z0", "clinical"] == pytest.approx(
        in_sample.table.loc["z0", "clinical"], abs=0.05)
    assert held_out.table.loc["z2", "clinical"] < in_sample.table.loc["z2", "clinical"]


def test_ceilings_alternative_grouping():
    df_latent, df_feat, feat_to_group = _mixed_dataset()
    decomp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group,
                                          mode="axes", seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = decomp.ceilings({"stage": "stage", "mutated": "mutation",
                               "volume": "shape"})
    assert sorted(rep.table.columns) == ["mutation", "stage"]
    assert rep.coverage.loc["mutation", "n_cells"] == 2


def test_ceilings_raises_when_no_family_is_discrete():
    rng = np.random.default_rng(13)
    df_feat = pd.DataFrame(rng.normal(size=(200, 3)),
                           columns=["a", "b", "c"])
    df_latent = pd.DataFrame(rng.normal(size=(200, 2)), columns=["z0", "z1"])
    decomp = FeatureGroundedDecomposition(df_latent, df_feat, seed=0)
    with pytest.warns(UserWarning, match="not discrete"):
        with pytest.raises(ValueError, match="no family has discrete labels"):
            decomp.ceilings()
