"""Plotting helpers for feature-grounded variance decomposition.

Kept deliberately small: the budget figures themselves live on
:class:`~fva.explainer.FeatureGroundedDecomposition`; this module owns the
stand-alone scree plot, the shared colour function so that a group keeps the
same colour across the scree, bar, and heatmap views, and the noise-floor
annotation the budget figures draw on themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import InstanceOf, NonNegativeInt, PositiveInt, validate_call

from ._diagnostics import NULL_R2_LINE_MIN, null_r2
from ._validation import VALIDATE_CONFIG, FigOut, FigSize
from .decomposition import DecompositionResult

logger = logging.getLogger(__name__)

# A categorical colour wheel; indices beyond its length wrap around.
_PALETTE = list(colormaps["tab20"].colors)

# Where ``fig_out=True`` writes figures, named after the report method.
_DEFAULT_FIG_DIR = Path("figures")


@validate_call(config=VALIDATE_CONFIG)
def save_figure(fig: InstanceOf[Figure], fig_out: FigOut, default_name: str, *,
                dpi: PositiveInt = 150):
    """Optionally save ``fig`` according to a ``fig_out`` argument.

    ``fig_out`` follows a three-way convention shared by all report methods:

    * ``None`` (or ``False``) -- do nothing (the default; figures are returned,
      not written).
    * a path (``str`` / :class:`~pathlib.Path`) -- save to exactly that path.
    * ``True`` -- save to the default location ``figures/<default_name>.png``.

    Parent directories are created as needed.  Returns the path written, or
    ``None`` if nothing was saved.
    """
    if fig_out is None or fig_out is False:
        return None
    if fig_out is True:
        path = _DEFAULT_FIG_DIR / f"{default_name}.png"
    else:
        path = Path(fig_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
    return path


def _color(name: str, index: int):
    """Stable colour for a group/feature, keyed by its position ``index``.

    Callers pass a consistent index per name (the group's position in
    ``group_names``), so the same group keeps the same colour across figures.
    """
    return _PALETTE[index % len(_PALETTE)]


@validate_call(config=VALIDATE_CONFIG)
def draw_null_floor(ax: InstanceOf[Axes], n: NonNegativeInt, q: NonNegativeInt, *,
                    null_line: bool | None = None) -> float | None:
    """Mark the pure-noise ``R^2`` floor on a budget figure.

    Draws a dashed horizontal line at :func:`~fva._diagnostics.null_r2`, the
    in-sample ``R^2`` a bank of ``q`` features made of pure noise would score on
    ``n`` samples.  Both budget bars stack the explained shares from the bottom
    and the noise remainder on top, so the line reads directly: everything below
    it could be fit noise rather than signal.  The height is the same for every
    direction, since each direction is a fit of the same ``n`` on the same ``q``.

    ``null_line`` is three-way, following the ``fig_out`` convention:

    * ``None`` (the default) -- draw only when the floor is high enough to matter
      (``>= NULL_R2_LINE_MIN``).  Keeps the line out of healthy figures, where it
      would sit on the axis spine and read as a rendering artifact.
    * ``True`` -- always draw.  Use this for figure panels that will be compared
      side by side, so that an absent line means "no floor" rather than "a floor
      below the threshold".
    * ``False`` -- never draw.

    The caller is responsible for the applicability of the floor: it is the
    *in-sample* bias, so it must not be drawn on a cross-validated budget, where
    a pure-noise bank scores about zero (see
    :meth:`~fva.explainer.FeatureGroundedDecomposition.shap` with ``cv=True``).

    Returns the ``y`` value drawn, or ``None`` if nothing was drawn.  The line
    carries a legend label, so call this before ``ax.legend()``.
    """
    if null_line is False:
        return None
    floor = null_r2(n, q)
    if floor <= 0.0:
        return None
    if null_line is None and floor < NULL_R2_LINE_MIN:
        return None
    ax.axhline(floor, color="#333333", linestyle="--", linewidth=1.2, zorder=5,
               label=f"noise floor ($R^2 \\approx {floor:.2f}$ from a bank\n"
                     f"of {q} pure-noise features on {n} samples)")
    logger.info("null R^2 floor %.3f (n=%d, q=%d) marked on the budget figure",
                floor, n, q)
    return floor


def draw_scree_line(ax: Axes, x, fracs) -> None:
    """Overlay the scree curve on a per-direction budget figure.

    ``fracs`` is the eigenvalue weight of each direction shown (``lambda_k / sum
    lambda``), drawn as a black line with markers at ``x``.  Both the bar heights
    and the scree are fractions in ``[0, 1]``, so the line shares the budget
    axis: it reads as "how much of the total latent variance this direction is",
    against the bar's "how that direction's variance is spent".  Directions are
    ordered by variance, so the line falls monotonically and marks where the
    budget stops describing much of the space.

    The line carries a legend label, so call this before ``ax.legend()``.
    """
    ax.plot(list(x), list(fracs), color="black", marker="o", markersize=4,
            linewidth=1.5, zorder=6, label="scree (share of total variance)")


@validate_call(config=VALIDATE_CONFIG)
def plot_scree(result: InstanceOf[DecompositionResult], *,
               figsize: FigSize = (8, 2.8)):
    """Bar plot of the variance fraction carried by each latent direction.

    Directions are ordered by variance (as produced by the decomposition); the
    height of each bar is ``lambda_k / sum lambda``.  Returns the Figure.
    """
    fracs = [d.variance_frac for d in result.directions]
    labels = [
        f"PC{d.index}" if result.mode == "eigen" else str(d.index)
        for d in result.directions
    ]
    x = range(len(fracs))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, fracs, color="#4c72b0", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_xlabel("latent direction (ordered by variance)")
    ax.set_ylabel("fraction of total\nlatent variance")
    ax.set_title("Scree: variance carried by each latent direction")
    for xi, h in zip(x, fracs):
        ax.text(xi, h, f"{h*100:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(fracs) * 1.15 if fracs else 1)
    fig.tight_layout()
    return fig
