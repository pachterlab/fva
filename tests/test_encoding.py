"""Tests for :mod:`fva.encoding`.

Two things are being checked, and they are different in kind.  The mechanical
one is that the expansions have the right shape, names and grouping.  The
substantive one is that they are *faithful*: a latent direction driven by a
categorical or by an angle is unexplainable in the raw coding and explained once
encoded, which is the whole reason the module exists.  The invariance test is the
claim the module docstring makes about ``drop_first`` --- the group budget does
not depend on which level is dropped --- since that is what licenses reading a
categorical's share off the group table at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fva import FeatureGroundedDecomposition, encode_features
from fva.encoding import (
    CIRCULAR_NAME_PATTERN,
    MAX_LEVELS_WARN,
    _infer_period,
    _is_string_column,
)


@pytest.fixture
def raw():
    """Raw features with one string categorical and one angle, plus numerics."""
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "area": rng.normal(size=n),
        "grade": rng.choice(["low", "mid", "high"], size=n),
        "rotation_angle": rng.uniform(0, 360, size=n),
    })


# ---------------------------------------------------------------- inference
def test_infers_string_categorical_and_named_angle(raw):
    out, mapping = encode_features(raw)

    assert list(out.columns) == [
        "area", "grade=low", "grade=mid",
        "rotation_angle=sin", "rotation_angle=cos",
    ]
    # Encoded columns are pooled under their source column, untouched ones are
    # their own singleton group -- the per-feature default, expansions aside.
    assert mapping["grade=low"] == mapping["grade=mid"] == "grade"
    assert mapping["rotation_angle=sin"] == "rotation_angle"
    assert mapping["area"] == "area"
    assert out.dtypes.map(pd.api.types.is_numeric_dtype).all()


def test_sin_cos_lies_on_the_unit_circle(raw):
    out, _ = encode_features(raw)
    r = out["rotation_angle=sin"] ** 2 + out["rotation_angle=cos"] ** 2
    assert np.allclose(r, 1.0)
    # Degrees, so a full turn maps to a full cycle: 0 and 360 coincide.
    deg = pd.DataFrame({"angle": [0.0, 90.0, 180.0, 359.999]})
    enc, _ = encode_features(deg)
    assert enc["angle=cos"].iloc[0] == pytest.approx(1.0)
    assert enc["angle=sin"].iloc[1] == pytest.approx(1.0)
    assert enc["angle=cos"].iloc[3] == pytest.approx(1.0, abs=1e-6)


def test_integer_codes_are_never_inferred():
    """``1, 2, 3`` is ambiguous, so inference must leave it alone."""
    df = pd.DataFrame({"grade": [1, 2, 3, 1, 2, 3]})
    out, _ = encode_features(df)
    assert list(out.columns) == ["grade"]

    out, mapping = encode_features(df, categorical=["grade"])
    assert list(out.columns) == ["grade=2", "grade=3"]
    assert set(mapping.values()) == {"grade"}


def test_angle_outside_one_turn_is_not_claimed():
    """A column named ``angle`` holding something else keeps its raw reading."""
    df = pd.DataFrame({"angle": [0.0, 400.0, 1200.0]})
    out, _ = encode_features(df)
    assert list(out.columns) == ["angle"]


def test_name_pattern_and_string_detection():
    assert CIRCULAR_NAME_PATTERN.search("lesion_Angle")
    assert CIRCULAR_NAME_PATTERN.search("ROTATION_deg")
    assert not CIRCULAR_NAME_PATTERN.search("area")
    assert _is_string_column(pd.Series(["a", "b"]))
    assert _is_string_column(pd.Series(["a", "b"], dtype="category"))
    assert not _is_string_column(pd.Series([1.0, 2.0]))


def test_infer_false_encodes_nothing_and_then_errors(raw):
    with pytest.raises(ValueError, match="not numeric"):
        encode_features(raw, infer=False)


def test_radians_inferred_from_range():
    period, ambiguous = _infer_period(np.linspace(-np.pi, np.pi, 50), "a")
    assert period == pytest.approx(2 * np.pi)
    assert not ambiguous
    period, ambiguous = _infer_period(np.linspace(0, 300, 50), "a")
    assert period == pytest.approx(360.0)
    assert not ambiguous


# ----------------------------------------------------------------- warnings
def test_narrow_range_warns_about_ambiguous_unit():
    df = pd.DataFrame({"angle": np.linspace(0.0, 2.0, 40)})
    with pytest.warns(UserWarning, match="unit is ambiguous"):
        encode_features(df)


def test_half_turn_range_warns_about_axial_data():
    """Orientation data (v ~ v + half turn) needs the doubled angle, not this."""
    df = pd.DataFrame({"angle": np.linspace(0.0, 179.0, 60)})
    with pytest.warns(UserWarning, match="at most half"):
        encode_features(df)
    # Saying so explicitly silences it and doubles the angle.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, _ = encode_features(df, circular={"angle": 180.0})
    theta = 2 * np.pi * df["angle"] / 180.0
    assert np.allclose(out["angle=sin"], np.sin(theta))


def test_wide_categorical_warns():
    n = MAX_LEVELS_WARN + 5
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)]})
    with pytest.warns(UserWarning, match="levels"):
        encode_features(df)


def test_missing_values_warn_and_get_their_own_level():
    df = pd.DataFrame({"grade": ["a", "b", None, "a", "b", "c"]})
    with pytest.warns(UserWarning, match="missing values"):
        out, _ = encode_features(df)
    assert any("nan" in c.lower() for c in out.columns)


def test_constant_categorical_is_dropped():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "site": ["a", "a", "a"]})
    with pytest.warns(UserWarning, match="carries no variance"):
        out, mapping = encode_features(df)
    assert list(out.columns) == ["x"]
    assert "site" not in mapping.values()


# -------------------------------------------------------------------- errors
def test_unknown_or_doubly_named_columns_raise(raw):
    with pytest.raises(ValueError, match="not in df"):
        encode_features(raw, categorical=["nope"])
    with pytest.raises(ValueError, match="both categorical and circular"):
        encode_features(raw, categorical=["grade"], circular=["grade"])
    with pytest.raises(ValueError, match="finite and positive"):
        encode_features(raw, circular={"rotation_angle": 0.0})


def test_name_collision_raises():
    df = pd.DataFrame({"g": ["a", "b", "a", "b"], "g=b": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="duplicate column name"):
        encode_features(df)


def test_explainer_names_the_offending_column(raw):
    """The promise the error message makes must actually point somewhere."""
    df_latent = pd.DataFrame(np.random.default_rng(0).normal(size=(len(raw), 3)))
    with pytest.raises(ValueError, match="grade"):
        FeatureGroundedDecomposition(df_latent, raw)


# -------------------------------------------------------------------- groups
def test_groups_are_inherited_by_expanded_columns(raw):
    _, mapping = encode_features(
        raw, groups={"grade": "clinical", "rotation_angle": "shape", "area": "shape"},
    )
    assert mapping["grade=low"] == "clinical"
    assert mapping["rotation_angle=sin"] == mapping["rotation_angle=cos"] == "shape"
    assert mapping["area"] == "shape"


def test_columns_missing_from_groups_are_left_ungrouped(raw):
    """Left out of the mapping, so the explainer's own 'ungrouped' path fires."""
    _, mapping = encode_features(raw, groups={"grade": "clinical"})
    assert set(mapping) == {"grade=low", "grade=mid"}


# ---------------------------------------------------- faithfulness of the fit
def _latent_from(signal: np.ndarray, seed: int = 1) -> pd.DataFrame:
    """A latent space driven by ``signal`` plus a little noise."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(signal.shape[1], 3))
    Z = signal @ mixing + 0.1 * rng.normal(size=(len(signal), 3))
    return pd.DataFrame(Z, columns=["z0", "z1", "z2"])


def test_categorical_is_unexplainable_raw_and_explained_encoded():
    """The point of one-hot: level means are not a line through the codes."""
    rng = np.random.default_rng(0)
    n = 400
    codes = rng.integers(0, 3, size=n)
    # Level means are non-monotone in the code, so no linear function of the raw
    # column can fit them; the one-hot expansion fits them exactly.
    effect = np.array([1.0, -1.0, 1.0])[codes]
    df_latent = _latent_from(effect[:, None])
    df_raw = pd.DataFrame({"grade": codes})

    raw_rho = FeatureGroundedDecomposition(df_latent, df_raw).rho
    df_feat, mapping = encode_features(df_raw, categorical=["grade"])
    enc_rho = FeatureGroundedDecomposition(df_latent, df_feat, mapping).rho

    assert raw_rho < 0.05
    assert enc_rho > 0.9


def test_angle_is_unexplainable_raw_and_explained_encoded():
    rng = np.random.default_rng(0)
    n = 400
    deg = rng.uniform(0, 360, size=n)
    signal = np.cos(np.deg2rad(deg))[:, None]
    df_latent = _latent_from(signal)
    df_raw = pd.DataFrame({"rotation_angle": deg})

    raw_rho = FeatureGroundedDecomposition(df_latent, df_raw).rho
    df_feat, mapping = encode_features(df_raw)
    enc_rho = FeatureGroundedDecomposition(df_latent, df_feat, mapping).rho

    assert raw_rho < 0.05
    assert enc_rho > 0.9


def _axial_case(low: float, high: float, n: int = 400):
    """A latent space driven by a half-turn-periodic signal, measured in [low, high)."""
    deg = np.random.default_rng(0).uniform(low, high, size=n)
    df_latent = _latent_from(np.cos(2 * np.deg2rad(deg))[:, None])
    return pd.DataFrame({"angle": deg}), df_latent


def test_axial_angle_needs_the_doubled_encoding():
    """An orientation is invariant to a half turn; the first harmonic is not.

    Measured over a full turn the two harmonics are orthogonal, so the default
    encoding is not merely worse than the doubled one -- it is blind to the
    signal outright.
    """
    df_raw, df_latent = _axial_case(0.0, 360.0)
    first, m1 = encode_features(df_raw)
    doubled, m2 = encode_features(df_raw, circular={"angle": 180.0})

    assert FeatureGroundedDecomposition(df_latent, first, m1).rho < 0.05
    assert FeatureGroundedDecomposition(df_latent, doubled, m2).rho > 0.9


def test_axial_warning_fires_on_a_half_turn_range_and_is_worth_heeding():
    """Axial data folded into [0, 180) is what the warning can actually detect.

    Over half a turn the harmonics are no longer orthogonal, so the first one
    recovers much of the signal by accident -- the encoding is understated
    rather than blind, which is why this is a warning and not an error.
    """
    df_raw, df_latent = _axial_case(0.0, 180.0)
    with pytest.warns(UserWarning, match="at most half"):
        first, m1 = encode_features(df_raw)
    doubled, m2 = encode_features(df_raw, circular={"angle": 180.0})

    rho_first = FeatureGroundedDecomposition(df_latent, first, m1).rho
    rho_doubled = FeatureGroundedDecomposition(df_latent, doubled, m2).rho
    assert rho_first < rho_doubled - 0.02


def test_group_budget_is_invariant_to_the_dropped_level():
    """The docstring's claim: only the per-column Johnson row sees the coding."""
    rng = np.random.default_rng(0)
    n = 400
    codes = rng.choice(["a", "b", "c"], size=n)
    effect = pd.Series(codes).map({"a": 1.0, "b": -1.0, "c": 0.3}).to_numpy()
    df_latent = _latent_from(np.column_stack([effect, rng.normal(size=n)]))
    df_raw = pd.DataFrame({"grade": codes, "area": rng.normal(size=n)})

    dropped, m_drop = encode_features(df_raw)
    # Reordering the levels changes which one drop_first removes.
    reordered = df_raw.assign(
        grade=pd.Categorical(df_raw["grade"], categories=["c", "b", "a"])
    )
    other, m_other = encode_features(reordered)
    assert list(dropped.columns) != list(other.columns)

    # Exact for OLS; the tiny diagonal load in _GramCache makes it hold only to
    # the size of that ridge, which is what the tolerance here is measuring.
    rho_a = FeatureGroundedDecomposition(df_latent, dropped, m_drop).rho
    rho_b = FeatureGroundedDecomposition(df_latent, other, m_other).rho
    assert rho_a == pytest.approx(rho_b, rel=1e-5)

    shap_a = FeatureGroundedDecomposition(df_latent, dropped, m_drop).shap().table
    shap_b = FeatureGroundedDecomposition(df_latent, other, m_other).shap().table
    assert np.allclose(shap_a["grade"].to_numpy(), shap_b["grade"].to_numpy(),
                       atol=1e-5)


def test_keeping_every_level_is_collinear_but_spans_the_same_subspace():
    """Why drop_first defaults to True, and why the group budget survives it.

    The full dummy set is exactly linearly dependent once centred, so the feature
    correlation matrix is singular and Johnson's per-column split among those
    columns is numerically arbitrary.  The subspace they span is unchanged,
    though, which is why the group-level share is the same either way.
    """
    rng = np.random.default_rng(0)
    n = 300
    codes = rng.choice(["a", "b", "c"], size=n)
    effect = pd.Series(codes).map({"a": 1.0, "b": -1.0, "c": 0.3}).to_numpy()
    df_latent = _latent_from(effect[:, None])
    df_raw = pd.DataFrame({"grade": codes})

    full, m_full = encode_features(df_raw, drop_first=False)
    dropped, m_drop = encode_features(df_raw)
    assert (full.shape[1], dropped.shape[1]) == (3, 2)

    X = full.to_numpy()
    Rxx = np.corrcoef((X - X.mean(0)).T)
    assert np.linalg.eigvalsh(Rxx).min() < 1e-12   # exactly collinear

    rho_full = FeatureGroundedDecomposition(df_latent, full, m_full).rho
    rho_drop = FeatureGroundedDecomposition(df_latent, dropped, m_drop).rho
    assert rho_full == pytest.approx(rho_drop, rel=1e-5)
