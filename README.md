# Naming the Directions of a Latent Space

A small Python package for **feature-grounded variance decomposition**: given the
latent codes from any encoder (a VAE, an autoencoder, CLIP, ...) and a bank of
interpretable features (radiomic descriptors, hand-crafted measurements,
metadata), it answers a simple question for every latent direction —

> *What fraction of this direction's variance do the interpretable features
> explain, and how does that explained variance split across feature groups and
> individual features?*

Whatever the features cannot account for is reported as an explicit
**irreducible-noise** budget, so each direction is summarized by a set of named
shares plus a noise remainder that together sum to one. The result is a latent
space whose axes carry interpretable labels instead of opaque indices.

<table>
<tr>
<td width="50%"><img src="notebooks/figures/global_budget.png" width="100%"></td>
<td width="50%"><img src="notebooks/figures/shap_budget.png" width="100%"></td>
</tr>
<tr>
<td><em>The whole latent space in one bar: the variance-weighted share of total
latent variance attributed to each feature group, with the gray segment the
unexplained (noise) remainder. They sum to one, so this is the <code>rho</code>
headline broken out by group.</em></td>
<td><em>The same budget resolved per latent direction. Each bar is one direction
(the percentage below it is the share of total latent variance it carries);
colored segments are the per-group shares and the gray segment is the noise
remainder. High-variance directions are almost entirely explained; the
lowest-variance ones are increasingly noise.</em></td>
</tr>
</table>

## Installation

```bash
pip install fva
```

## Quickstart

```python
import fva
decomp = fva.FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)
scree_fig = decomp.scree()
global_table, rho, global_fig = decomp.global_budget()
shap_table, rho, shap_fig = decomp.shap()
johnson_table, rho, johnson_fig = decomp.johnson()
```

See **[`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb)** for a walkthrough of the whole workflow on a small synthetic dataset and the API details.

## Screening before you decompose

A budget answers *which families* explain a direction, which is worth asking only
once something explains it. `fva.direction_r2` reports just the totals — the
$R^2$ of the whole bank on each direction, and its variance-weighted sum:

```python
table = fva.direction_r2(df_latent, df_feat)   # no feat_to_group needed
```

```
              variance_fraction     r2  r2_cv  noise
direction
PC0                       0.401  0.978  0.975  0.022
PC1                       0.256  0.964  0.960  0.036
...
PC9                       0.008  0.028 -0.009  0.972
weighted_sum              1.000  0.907  0.902  0.093
```

The `weighted_sum` row reproduces `rho` and `rho_cv` from a full decomposition of
the same inputs exactly, because both go through the same eigendecomposition and
the same Gram matrix. What it skips is the attribution — the $2^g$ Shapley
coalitions and the per-feature Johnson fit — which is what the run actually costs:
on $n = 800$, $d = 32$, $q = 240$ features in 8 groups, the screen takes 0.45 s
against 18.7 s for the full fit.

Read the `r2_cv` column rather than `r2`. A bank with no signal scores about zero
out of sample whatever its width, so negative entries are the ordinary appearance
of an unexplained direction rather than an error; the in-sample `r2` for the same
direction sits at the $q/(n-1)$ noise floor described under
[Diagnostics](#diagnostics). Read it per direction, too — the total is variance
weighted, so a direction the bank explains well while carrying little variance is
visible in the column and invisible in the last row.

The same vectors come off a fitted decomposition as `decomp.r2` and
`decomp.r2_cv`, so screening first costs nothing the full fit would not have
spent anyway.

## Non-numeric features

The decomposition is a linear fit, so a categorical column and a wrapped angle
mean nothing in their raw form — `grade = 1, 2, 3` is fitted as if the levels sat
equally spaced on a line, and 359° and 1° land at opposite ends of the range
despite being two degrees apart. `encode_features` re-represents both and returns
the grouping that keeps each expansion pooled into one budget entry:

```python
df_feat, feat_to_group = fva.encode_features(df_raw)
decomp = fva.FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)
```

String columns and `angle`/`rotation` columns within ±360 are claimed
automatically. The two things it will not guess have to be named:

```python
df_feat, feat_to_group = fva.encode_features(
    df_raw,
    categorical=["grade"],            # integer codes: nominal or ordinal is ambiguous
    circular={"lesion_angle": 180},   # axial: θ and θ+180° are the same orientation
    groups={"grade": "clinical", "lesion_angle": "shape"},
)
```

One-hot expansion drops a reference level, since a full dummy set is collinear
with the intercept once centred. The **group** budget is invariant to which level
is dropped; only the per-feature Johnson row reads as a contrast against the
reference.

There is deliberately no *automatic* basis expansion. The encodings above are
faithful re-representations that add no explanatory power a correctly-read column
did not already have, whereas a basis expansion widens the bank and raises `R²` —
and the noise remainder this package reports is only meaningful relative to a
*fixed* hypothesis class. Expansion is available, but as an explicit choice you
make and report: see below.

## Basis expansion and the capacity ladder

A feature often acts on the latent space smoothly but nonlinearly — a sprite's
position produces a localized pixel pattern the encoder folds nonlinearly into
the code — and a linear fit on the raw column then reports as noise a dependence
the named feature describes perfectly well. `fva.basis` widens the hypothesis
class to cover that case, and keeps the widening visible:

```python
df_feat, feat_to_group, spec = fva.expand_features(
    df_raw, mode="auto", spline_df=6, harmonics=3,
    circular={"orientation": 2 * np.pi},
)
print(spec)      # per column: which basis, how wide, how wide it asked to be
```

Note the three-value unpack. `encode_features` returns two, because it made no
choice; `expand_features` returns the `spec` as well, because a `rho` from an
expanded bank cannot be compared with anything without it.

`mode="auto"` picks the basis per column, and which basis is right is decided by
the feature rather than by taste:

- a **circular** feature is genuinely periodic, so it gets the Fourier basis
  `sin/cos(2πkv/period)` for `k = 1..harmonics`. `harmonics=1` is the pair
  `encode_features` already emits.
- a **non-periodic continuous** feature gets a cubic B-spline with `spline_df`
  columns, knots at quantiles. A Fourier basis is the wrong choice here even
  though hand-rolled feature banks usually reach for it: rescaling the column to
  a unit period forces the fitted function to take the same value at both ends of
  the range, and the raw column then has to be carried alongside purely to break
  that wrap-around. The spline has no wrap, has local support, is better
  conditioned, and at `spline_df=1` degenerates *exactly* to the raw column.

`mode="spline"` and `mode="circular"` force one basis onto everything, warning on
the columns they mis-handle, so the choice of basis can be ablated rather than
asserted. Every expansion stays pooled in its source column's group, so one
concept is one budget entry and the `2^g` Shapley game does not grow.

A per-column basis, however wide, stays **separable**: it writes the latent
coordinate as a sum of a function of each feature and expresses no joint
dependence between two. Some joint dependences are structural — which pixels a
sprite occupies is set jointly by its two position coordinates, and no sum of a
function of `x` and a function of `y` localizes a two-dimensional bump — and a
separable bank reports them as noise. Name the pairs the features imply:

```python
df_feat, feat_to_group, spec = fva.expand_features(
    df_raw, spline_df=4, harmonics=2,
    interactions=[("posX", "posY")],
)
```

Each pair adds the tensor product of the two features' bases as one further
group, `"posX x posY"`. It is a second widening on top of the per-column one, so
read the gain against the ladder rather than quoting it alone, and do not fish
over all pairs. Because each basis sums to a constant across its own columns, the
tensor product spans both main effects as well as the interaction; the two groups
are collinear by construction, which the Shapley budget handles, but the shared
mass is real.

Rather than fixing the capacity at one hidden setting, sweep it:

```python
table, fig = fva.capacity_ladder(df_latent, df_raw, levels=(1, 2, 4, 6, 8))
```

Level 1 is the unexpanded bank, so the first row is the baseline and every later
row is what widening bought. The curve separates the two ways `rho` can rise,
which look identical in a single fit: one that climbs and then flattens says the
feature really does act nonlinearly and the plateau is a property of the feature;
one that keeps climbing in step with the tabulated `null_r2` says the bank is
just getting wide. Both `rho` and `rho_cv` are reported at every rung.

Two guards fire before anything is fitted. A "continuous" column with `u`
distinct values spans at most `u - 1` dimensions once centered, so a wider basis
is rank deficient by construction and the requested width is clipped with a
warning naming the column. And the expanded width goes through the usual
sample-size check, so a bank that outgrew its sample says so.

## What a family could ever explain

The ladder asks whether a wider hypothesis class buys anything. The ceiling asks
the prior question: whether anything *could*. Samples carrying the same pattern
of labels within a family are indistinguishable to it and must receive the same
prediction, so grouping them by label pattern and predicting each group by its
own mean is the best any function of those labels can do. Its $R^2$ bounds the
family from above, for every hypothesis class at once — no spline, interaction,
or nonlinear model of the same labels can beat it.

```python
ceil, alone, attainment, coverage = decomp.ceilings()
```

`alone` is what each family explains fitted on its own, `ceil` is the most it
could, and `attainment` is the ratio. That ratio is what a small share needs in
order to be read: a family at 30% of its ceiling is under-fitted and the ladder
is the next move, while a family at 95% of a ceiling of 0.2 has said everything
it can and the rest of the direction is out of reach of that vocabulary rather
than of the model. `coverage` carries the label geometry behind each bound —
how many samples carry a label at all, how many distinct cells they form, the
smallest cell, and the ceiling that signal-free labels of the same shape would
reach anyway.

A ceiling below 1 has two causes and the report separates them. Labels defined
on only part of the sample leave the unlabeled remainder to a single prediction,
so its internal variance is unreachable by construction; labels defined
everywhere but too coarse leave within-cell variance instead. The first is a
property of the vocabulary, the second of its resolution.

The bound only exists because samples sharing a pattern share a prediction, so
continuous families are skipped with a warning naming them (bin a column first
for a bound that is exact over the binning). And the saturated fit is still a
fit: $k$ cells spend $k-1$ degrees of freedom, so it inflates on small samples
and is vacuous once $k \ge n$ — which is itself the finding that the binding
constraint is the sample size rather than the coverage. `ceilings(cv=True)`
scores both the ceilings and the family fits out of fold, and warnings fire on
the cell count either way.

Note that the ceiling bounds a family fitted **alone**, which is why `alone` is
reported rather than the Shapley share from `shap()`. The two differ when
families suppress one another, and a Shapley value can legitimately exceed what
its family reaches by itself.

## Output

For each latent direction the method produces an additive variance budget:

- a **group-level (SHAP) budget** — collinearity-aware Shapley shares of the
  explained variance per feature family (shape, texture, intensity, ...), plus
  the noise remainder;
- a **per-feature (Johnson) budget** — the same explained variance refined down
  to individual features;
- a single **global budget** that collapses all directions into one stacked bar,
  weighted by how much variance each direction carries.

A cross-validated $R^2$ keeps the explained fraction from being inflated by
overfitting, and the group shares, per-feature shares, and noise all partition
each direction's variance exactly.

## Diagnostics

The shares always partition the total exactly — that is an algebraic property of
the Shapley game, not evidence that the total means anything. With too few
samples for the width of the feature bank the in-sample $R^2$ saturates toward 1
regardless of signal, and the budget stays perfectly well-formed while
apportioning nothing. The package therefore warns when the data cannot support
what it is being asked to explain:

- fewer than ~10 samples per feature, or outright $q \ge n$, where the in-sample
  budget is vacuous;
- a large gap between the in-sample `rho` and the cross-validated `rho_cv`;
- a rank-deficient feature bank, which makes the *per-feature* Johnson split
  arbitrary among collinear columns (the group budget is unaffected);
- more feature groups than the exponential $2^g$ Shapley game can afford, or more
  CV folds than the sample supports;
- more label cells than the sample can support in `ceilings()`, where the bound
  stops being tight and, at $k \ge n$, stops being a bound.

These are standard `UserWarning`s, so `warnings.simplefilter("error")` promotes
them to failures in a pipeline.

The same quantity that drives the first of those warnings is also drawn on the
budget figures. A bank of $q$ pure-noise features scores $R^2 \approx q/(n-1)$
in sample before any signal is present, so the stacked bars mark that height
with a dashed **noise floor**: whatever sits below the line could be fit noise
rather than structure. It appears automatically once the floor clears 5% of the
budget, and `null_line=True` / `False` on `shap()`, `johnson()`, and
`global_budget()` forces it on or off — force it on when panels will be compared
side by side, so that a missing line means "no floor" rather than "a small one".
The floor is an in-sample bias, so `shap(cv=True)` declines to draw it.

`shap(scree=True)` adds the other half of the picture: the scree curve drawn
over the stacked bars as a black line, on the bars' own $[0, 1]$ axis. The bars
say how each direction's variance is spent; the line says how much of the latent
space that direction is worth, so a well-explained direction carrying almost
none of the variance is visible as such.

Pass `n_bootstrap=200` for percentile intervals
on each direction's $R^2$ via `decomp.r2_ci`, which is how you tell a direction
the features genuinely explain from one whose share is not separable from zero.

Because the Shapley game costs $2^g$ fits in the number of feature groups, the
long loops show a `tqdm` progress bar; pass `progress=False` (or call
`fva.set_progress(False)`) to silence them.

## Reference

This package accompanies the paper *Naming the Directions of a Latent Space:
Feature-Grounded Variance Decomposition with an Irreducible-Noise Budget* by Joseph Rich, Raphi Kang, Pietro Perona, and Lior Pachter
([DOI: forthcoming](https://doi.org/)).