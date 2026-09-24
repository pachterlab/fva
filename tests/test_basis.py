"""Tests for :mod:`fva.basis`.

Three things are being checked.  The mechanical one is that each basis has the
promised width, names and grouping.  The substantive one is the pair of claims
the module docstring makes about *which* basis belongs to which feature: a
spline recovers a smooth nonlinear dependence a raw column cannot, and does not
impose the wrap a Fourier basis on a non-periodic column does; harmonics recover
a periodic dependence a spline handles badly.  The third is that the guards fire
--- a basis wider than the column's distinct values can support is clipped, and
the ladder reports the capacity it was run at rather than hiding it.

The identity at ``level=1`` matters more than it looks: both bases reduce to the
unexpanded bank there, so the first rung of every ladder is the baseline and any
later rise is attributable to capacity rather than to a changed encoding.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fva import FeatureGroundedDecomposition, capacity_ladder, expand_features
from fva.basis import (
    DEFAULT_LEVELS,
    _max_supported_width,
    harmonic_basis,
    hurdle_spline_basis,
    spline_basis,
)


@pytest.fixture
def raw():
    """A non-periodic continuous column, an angle, and a string categorical."""
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "area": rng.uniform(-2, 2, size=n),
        "rotation_angle": rng.uniform(0, 360, size=n),
        "grade": rng.choice(["low", "mid", "high"], size=n),
    })


# ------------------------------------------------------------------- bases
@pytest.mark.parametrize("df", [1, 2, 3, 4, 6, 9])
def test_spline_basis_has_the_requested_width(df):
    x = np.linspace(0, 1, 200)
    B, names = spline_basis(x, df, name="x")
    assert B.shape == (200, df)
    assert len(names) == df


def test_spline_df_one_is_the_raw_column():
    """The identity that makes level 1 of the ladder the unexpanded bank."""
    x = np.random.default_rng(0).uniform(-3, 5, size=200)
    B, _ = spline_basis(x, 1)
    # A degree-1 basis on two boundary knots is x rescaled to [0, 1]; an affine
    # map of the column is the same regressor once the fit centres and scales.
    assert np.corrcoef(B[:, 0], x)[0, 1] == pytest.approx(1.0)


def test_spline_basis_partitions_unity_before_the_reference_is_dropped():
    from fva.basis import _cox_de_boor

    lo, hi = 0.0, 1.0
    knots = np.concatenate([np.full(4, lo), np.array([0.3, 0.7]), np.full(4, hi)])
    B = _cox_de_boor(np.linspace(lo, hi, 101), knots, 3)
    assert np.allclose(B.sum(axis=1), 1.0)


def test_spline_basis_rejects_a_constant_column():
    with pytest.raises(ValueError, match="constant"):
        spline_basis(np.ones(50), 4, name="flat")


def test_harmonic_basis_width_and_unit_circle():
    x = np.linspace(0, 360, 200)
    B, names = harmonic_basis(x, 360.0, 3, name="a")
    assert B.shape == (200, 6)
    assert names[0] == "a=sin1" and names[-1] == "a=cos3"
    for k in range(3):
        assert np.allclose(B[:, 2 * k] ** 2 + B[:, 2 * k + 1] ** 2, 1.0)


def test_harmonic_basis_wraps_and_the_spline_does_not():
    """The property that decides which basis a column gets."""
    x = np.array([0.0, 360.0])
    B, _ = harmonic_basis(x, 360.0, 2)
    assert np.allclose(B[0], B[1])                 # a full turn is the identity
    S, _ = spline_basis(np.linspace(0, 360, 100), 6)
    assert not np.allclose(S[0], S[-1])            # the spline keeps them apart


# --------------------------------------------------------------- expansion
def test_auto_mode_picks_a_basis_per_column(raw):
    frame, mapping, spec = expand_features(raw, mode="auto", spline_df=4, harmonics=2)

    assert spec.loc["area", "basis"].startswith("bspline")
    assert spec.loc["rotation_angle", "basis"].startswith("fourier")
    assert spec.loc["grade", "basis"] == "one-hot"
    # 4 spline + 2*2 harmonic + 2 dummies
    assert frame.shape[1] == 4 + 4 + 2
    assert {mapping[c] for c in frame.columns} == {"area", "rotation_angle", "grade"}


def test_expansion_stays_pooled_in_one_group(raw):
    """The whole basis for one concept is one budget entry, not several."""
    frame, mapping, _ = expand_features(raw, mode="auto", spline_df=6)
    area_cols = [c for c in frame.columns if mapping[c] == "area"]
    assert len(area_cols) == 6


def test_level_one_reproduces_the_unencoded_bank(raw):
    """Both bases collapse to the faithful encoding at capacity 1."""
    from fva import encode_features

    expanded, _, _ = expand_features(raw, mode="auto", spline_df=1, harmonics=1)
    encoded, _ = encode_features(raw)
    assert expanded.shape[1] == encoded.shape[1]
    assert np.corrcoef(expanded["area=s1"], encoded["area"])[0, 1] == pytest.approx(1.0)


def test_forced_circular_mode_warns_and_carries_the_ramp(raw):
    with pytest.warns(UserWarning, match="non-periodic"):
        frame, _, spec = expand_features(raw[["area"]], mode="circular", harmonics=4)
    assert "area=raw" in frame.columns
    assert frame.shape[1] == 1 + 2 * 4
    assert spec.loc["area", "basis"].startswith("fourier")


def test_forced_spline_mode_warns_on_a_circular_column(raw):
    with pytest.warns(UserWarning, match="does not wrap"):
        expand_features(raw[["rotation_angle"]], mode="spline", spline_df=4)


def test_width_is_clipped_to_what_the_column_supports():
    """A six-level 'continuous' factor cannot support nine columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"scale": rng.choice(np.linspace(0.5, 1.0, 6), size=300)})
    assert _max_supported_width(df["scale"].to_numpy()) == 5
    with pytest.warns(UserWarning, match="distinct values"):
        frame, _, spec = expand_features(df, mode="auto", spline_df=9)
    assert frame.shape[1] == 5
    assert spec.loc["scale", "requested"] == 9 and spec.loc["scale", "emitted"] == 5


def test_non_numeric_unencoded_column_is_rejected():
    df = pd.DataFrame({"tag": ["a", "b", "c"] * 10})
    with pytest.raises(ValueError, match="not numeric"):
        expand_features(df, infer=False)


# ---------------------------------------------------------- does it explain
def test_spline_recovers_a_smooth_nonlinearity_the_raw_column_misses():
    """The substantive claim: capacity, not the feature, was the limit."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.uniform(-3, 3, size=n)
    # A latent axis that is a smooth, non-monotone function of one named feature.
    z = np.exp(-x ** 2) + 0.05 * rng.normal(size=n)
    df_latent = pd.DataFrame({"z0": z, "z1": 0.01 * rng.normal(size=n)})
    df_raw = pd.DataFrame({"x": x})

    raw_rho = FeatureGroundedDecomposition(
        df_latent, df_raw, {"x": "x"}, n_directions=1).rho_cv
    frame, mapping, _ = expand_features(df_raw, mode="auto", spline_df=8)
    spline_rho = FeatureGroundedDecomposition(
        df_latent, frame, mapping, n_directions=1).rho_cv

    assert raw_rho < 0.2          # a line through a bump explains nothing
    assert spline_rho > 0.9       # the named feature explained it all along


def test_harmonics_recover_a_periodic_dependence():
    rng = np.random.default_rng(1)
    n = 600
    a = rng.uniform(0, 360, size=n)
    z = np.sin(np.deg2rad(3 * a)) + 0.05 * rng.normal(size=n)
    df_latent = pd.DataFrame({"z0": z, "z1": 0.01 * rng.normal(size=n)})
    df_raw = pd.DataFrame({"rotation_angle": a})

    one, mapping, _ = expand_features(df_raw, harmonics=1)
    three, mapping3, _ = expand_features(df_raw, harmonics=3)
    rho1 = FeatureGroundedDecomposition(df_latent, one, mapping, n_directions=1).rho_cv
    rho3 = FeatureGroundedDecomposition(df_latent, three, mapping3, n_directions=1).rho_cv

    assert rho1 < 0.2             # the first harmonic is orthogonal to the third
    assert rho3 > 0.9


# ------------------------------------------------------------------ ladder
def test_ladder_reports_one_row_per_level_and_a_widening_bank(raw):
    rng = np.random.default_rng(0)
    z = np.exp(-raw["area"].to_numpy() ** 2) + 0.1 * rng.normal(size=len(raw))
    df_latent = pd.DataFrame({"z0": z, "z1": rng.normal(size=len(raw)) * 0.01})

    levels = (1, 2, 4)
    table, fig = capacity_ladder(df_latent, raw[["area"]], levels=levels,
                                 n_directions=1, figure=True)

    assert list(table.index) == list(levels)
    assert table["n_features"].is_monotonic_increasing
    assert np.isnan(table["d_rho_cv"].iloc[0])
    # The curve is the deliverable: capacity, not the feature, was the limit.
    assert table["rho_cv"].iloc[-1] > table["rho_cv"].iloc[0]
    assert fig is not None


def test_ladder_null_floor_tracks_bank_width(raw):
    df_latent = pd.DataFrame(
        np.random.default_rng(0).normal(size=(len(raw), 2)),
        columns=["z0", "z1"], index=raw.index,
    )
    table, _ = capacity_ladder(df_latent, raw, levels=(1, 4), figure=False,
                               n_directions=2)
    n = len(raw)
    assert table["null_r2"].iloc[1] == pytest.approx(
        table["n_features"].iloc[1] / (n - 1))
    # Random latents: the cross-validated total stays near zero however wide the
    # bank gets, which is what separates a real plateau from a wide one.
    assert table["rho_cv"].max() < 0.1


def test_ladder_rejects_a_zero_level(raw):
    df_latent = pd.DataFrame({"z0": np.arange(len(raw), dtype=float)}, index=raw.index)
    with pytest.raises(ValueError, match=">= 1"):
        capacity_ladder(df_latent, raw[["area"]], levels=(0, 1), figure=False)


def test_default_levels_start_at_the_unexpanded_bank():
    assert DEFAULT_LEVELS[0] == 1


def test_expansion_is_silent_on_a_well_sized_bank(raw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expand_features(raw, mode="auto", spline_df=4, harmonics=2)


# ------------------------------------------------------------- interactions
@pytest.fixture
def grid():
    """Two coordinates on a coarse grid, and a latent that depends on both jointly.

    ``z`` agrees on whether the two coordinates sit on the same side of the grid,
    which is an exclusive-or: every additive function of ``x`` and ``y`` has zero
    correlation with it however wide the per-column basis, so the tensor product
    is the whole of what a separable bank cannot reach.
    """
    rng = np.random.default_rng(0)
    n = 900
    x = rng.integers(0, 10, size=n).astype(float)
    y = rng.integers(0, 10, size=n).astype(float)
    z = ((x < 5) == (y < 5)).astype(float)
    return (pd.DataFrame({"posX": x, "posY": y}),
            pd.DataFrame({"z0": z}))


def test_interaction_adds_one_group_of_the_product_width(grid):
    raw_, _ = grid
    frame, feat_to_group, spec = expand_features(
        raw_, spline_df=3, interactions=[("posX", "posY")])
    assert "posX x posY" in spec.index
    assert spec.loc["posX x posY", "requested"] == 9
    inter = [c for c, g in feat_to_group.items() if g == "posX x posY"]
    assert len(inter) == 9
    assert frame.shape[1] == 3 + 3 + 9


def test_interaction_recovers_a_joint_dependence_the_separable_bank_misses(grid):
    raw_, df_latent = grid
    sep, g_sep, _ = expand_features(raw_, spline_df=4)
    joint, g_joint, _ = expand_features(raw_, spline_df=4,
                                        interactions=[("posX", "posY")])
    rho_sep = FeatureGroundedDecomposition(df_latent, sep, g_sep,
                                           cv_folds=3).rho_cv
    rho_joint = FeatureGroundedDecomposition(df_latent, joint, g_joint,
                                             cv_folds=3).rho_cv
    assert rho_sep < 0.1
    assert rho_joint > 0.7


def test_interaction_group_is_named_from_the_source_groups(grid):
    raw_, _ = grid
    _, feat_to_group, _ = expand_features(
        raw_, spline_df=2, groups={"posX": "position", "posY": "position"},
        interactions=[("posX", "posY")])
    assert set(feat_to_group.values()) == {"position", "position x position"}


def test_interaction_rejects_an_unknown_or_repeated_pair(grid):
    raw_, _ = grid
    with pytest.raises(ValueError, match="not a column"):
        expand_features(raw_, interactions=[("posX", "posZ")])
    with pytest.raises(ValueError, match="with itself"):
        expand_features(raw_, interactions=[("posX", "posX")])
    with pytest.raises(ValueError, match="named twice"):
        expand_features(raw_, interactions=[("posX", "posY"), ("posY", "posX")])


def test_interaction_warns_when_wider_than_the_pair_supports():
    # Perfectly dependent columns: each supports a width-4 basis on its own, but
    # the pair visits only 5 of the 25 cells, so the 16 products are collinear.
    a = np.tile(np.arange(5.0), 40)
    df = pd.DataFrame({"a": a, "b": 2 * a})
    with pytest.warns(UserWarning, match="distinct combinations"):
        expand_features(df, spline_df=4, interactions=[("a", "b")])


def test_ladder_carries_the_interaction_at_every_rung(grid):
    raw_, df_latent = grid
    table, _ = capacity_ladder(df_latent, raw_, levels=(1, 2, 3),
                               interactions=[("posX", "posY")], cv_folds=3,
                               figure=False)
    # 2 main effects of width L, plus their L^2 product, at every rung.
    assert list(table["n_features"]) == [1 + 1 + 1, 2 + 2 + 4, 3 + 3 + 9]


# ------------------------------------------------------------ basis options
def _zero_inflated(n=2000, zero_frac=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x = np.where(rng.uniform(size=n) < zero_frac, 0.0, rng.lognormal(0.5, 0.5, n))
    return x, rng


def test_hurdle_basis_width_names_and_zero_rows():
    x, _ = _zero_inflated()
    B, names = hurdle_spline_basis(x, 4, name="g")
    assert B.shape == (x.size, 5)
    assert names == ["g=nz", "g=s1", "g=s2", "g=s3", "g=s4"]
    assert np.array_equal(B[:, 0], (x != 0).astype(float))
    assert np.allclose(B[x == 0, 1:], 0.0)
    assert np.linalg.matrix_rank(np.column_stack([np.ones(x.size), B])) == 6


def test_hurdle_basis_contains_the_raw_column():
    x, _ = _zero_inflated()
    B, _ = hurdle_spline_basis(x, 4)
    X = np.column_stack([np.ones(x.size), B])
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    assert np.allclose(X @ coef, x, atol=1e-8)


def test_hurdle_basis_places_knots_off_the_point_mass():
    # A plain spline on a 90%-zero column loses its quantile knots to the point
    # mass and becomes one global polynomial.  At equal width, the hurdle spline
    # should fit a response among the nonzero values better and stay better
    # conditioned.
    x, rng = _zero_inflated(n=4000, zero_frac=0.9)
    off = x > 0
    y = np.where(off, 1 + np.tanh(4 * (x - np.median(x[off]))), 0.0)
    y = y + 0.05 * rng.normal(size=x.size)

    def r2_off(B):
        X = np.column_stack([np.ones(x.size), B])
        res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - res[off].var() / y[off].var()

    def cond(B):
        return np.linalg.cond((B - B.mean(0)) / B.std(0))

    plain, _ = spline_basis(x, 8)
    hurdle, _ = hurdle_spline_basis(x, 7)
    assert r2_off(hurdle) > r2_off(plain) + 0.01
    assert r2_off(hurdle) > 0.98
    assert cond(hurdle) < cond(plain) / 2


def test_hurdle_basis_on_a_binary_column_is_the_indicator():
    B, names = hurdle_spline_basis(np.array([0.0, 1.0, 0.0, 1.0]), 4, name="b")
    assert names == ["b=nz"] and B.shape == (4, 1)


def test_hurdle_basis_needs_a_point_mass():
    with pytest.raises(ValueError, match="needs both"):
        hurdle_spline_basis(np.linspace(1, 2, 20), 3)


def test_hurdle_basis_propagates_nan():
    x, _ = _zero_inflated(n=200)
    x[3] = np.nan
    B, _ = hurdle_spline_basis(x, 3)
    assert np.isnan(B[3]).all() and np.isfinite(np.delete(B, 3, axis=0)).all()


def test_expand_features_basis_options():
    x, rng = _zero_inflated(n=500)
    df = pd.DataFrame({"g": x, "c": rng.normal(size=500)})
    groups = {"g": "G", "c": "C"}

    none, m, spec = expand_features(df, basis="none", groups=groups)
    assert list(none.columns) == ["g", "c"] and spec.loc["g", "basis"] == "identity"

    spl, _, _ = expand_features(df, basis="spline", spline_df=4, groups=groups)
    assert spl.shape[1] == 8

    with pytest.warns(UserWarning, match="plain spline"):
        hur, m, spec = expand_features(df, basis="hurdle_spline", spline_df=4,
                                       groups=groups)
    assert [c for c in hur.columns if c.startswith("g=")] == [
        "g=nz", "g=s1", "g=s2", "g=s3", "g=s4"]
    assert spec.loc["c", "basis"].startswith("bspline")   # no zeros: fell back
    assert spec.loc["g", "basis"].startswith("hurdle_bspline")
    assert {m[c] for c in hur.columns if c.startswith("g=")} == {"G"}

    with pytest.warns(UserWarning, match="non-periodic"):
        four, _, _ = expand_features(df, basis="fourier", harmonics=2, groups=groups)
    with pytest.warns(UserWarning, match="non-periodic"):
        circ, _, _ = expand_features(df, mode="circular", harmonics=2, groups=groups)
    pd.testing.assert_frame_equal(four, circ)


def test_expand_features_rejects_unknown_basis(raw):
    with pytest.raises(Exception):
        expand_features(raw, basis="wavelet")


def test_hurdle_columns_aggregate_to_the_gene_in_johnson():
    x, rng = _zero_inflated(n=800)
    df = pd.DataFrame({"g": x, "h": _zero_inflated(n=800, seed=1)[0]})
    z = pd.DataFrame({"z0": np.sin(2 * df["g"]) + 0.1 * rng.normal(size=800),
                      "z1": (df["h"] > 0) + 0.1 * rng.normal(size=800)})
    frame, m, _ = expand_features(df, basis="hurdle_spline", spline_df=3)
    ex = FeatureGroundedDecomposition(z, frame, m)
    table = ex.johnson().table
    assert list(table.columns[:-1]) == ["g", "h"]
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-6)


def test_capacity_ladder_passes_basis_through():
    x, rng = _zero_inflated(n=400)
    df = pd.DataFrame({"g": x})
    z = pd.DataFrame({"z0": np.sqrt(df["g"]) + 0.1 * rng.normal(size=400)})
    table, _ = capacity_ladder(z, df, levels=(1, 3), basis="hurdle_spline",
                               figure=False)
    assert list(table["n_features"]) == [2, 4]
