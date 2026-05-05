"""Force-based label placement.

Iterative repulsion solver implemented on top of :mod:`matplotlib`.
The algorithm draws each label at its initial point, then iteratively:

1. Pushes labels away from anchor points (data points).
2. Pushes labels away from one another using axis-aligned bounding-box
   overlap.
3. Pulls each label gently back toward its anchor (so labels don't drift
   off the canvas).

When a label's final position differs from its anchor a thin line is
drawn between them.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib.axes import Axes

__all__ = ["repel_text"]


_R_GREY_MAP = {
    f"grey{int(p * 100)}": f"{p:.2f}" for p in [i / 10 for i in range(0, 11)]
}
_R_GREY_MAP.update({
    f"gray{int(p * 100)}": f"{p:.2f}" for p in [i / 10 for i in range(0, 11)]
})


def _safe_color(c) -> str:
    """Translate R-style ``"grey90"`` to a matplotlib-friendly form."""
    if isinstance(c, str) and c.lower() in _R_GREY_MAP:
        return _R_GREY_MAP[c.lower()]
    return c


def _bbox_in_data(ax: Axes, text) -> Tuple[float, float, float, float]:
    """Return ``(x0, y0, x1, y1)`` of ``text``'s bbox in data coordinates."""
    fig = ax.figure
    fig.canvas.draw()  # force layout so get_window_extent is accurate
    bb_disp = text.get_window_extent(renderer=fig.canvas.get_renderer())
    bb_data = bb_disp.transformed(ax.transData.inverted())
    return float(bb_data.x0), float(bb_data.y0), float(bb_data.x1), float(bb_data.y1)


def repel_text(
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    *,
    fontsize: float = 8.0,
    color: Optional[Sequence] = None,
    max_iter: int = 60,
    push_anchor: float = 0.04,
    push_label: float = 0.06,
    pull_strength: float = 0.005,
    segment_color: str = "0.5",
    segment_alpha: float = 0.5,
    segment_lw: float = 0.2,
    ha: str = "center",
    va: str = "center",
) -> List:
    """Place text labels on ``ax`` with simple force-based de-overlapping.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw onto.
    x, y : sequence of float
        Anchor coordinates (in data units).
    labels : sequence of str
        Strings to draw.
    fontsize : float, default ``8.0``
        Font size in points.
    color : sequence, optional
        Per-label colour. Defaults to black.
    max_iter : int, default ``60``
        Maximum number of relaxation iterations.
    push_anchor, push_label : float
        Repulsion-force magnitudes (data-units per iteration).
    pull_strength : float
        Strength of the spring pulling labels back to anchors.
    segment_color, segment_alpha, segment_lw : visual kwargs for the
        connector lines.
    ha, va : alignment kwargs for the matplotlib text.

    Returns
    -------
    list
        The :class:`matplotlib.text.Text` artists that were placed.
    """
    n = len(labels)
    if n == 0:
        return []
    pts_x = np.asarray(x, dtype=float)
    pts_y = np.asarray(y, dtype=float)

    if color is None:
        color = ["black"] * n
    color = [_safe_color(c) for c in list(color)]
    if len(color) != n:
        color = (color * ((n // len(color)) + 1))[:n]
    segment_color = _safe_color(segment_color)

    # Use anchor positions as starting label positions.
    lab_x = pts_x.copy()
    lab_y = pts_y.copy()

    # First-pass create texts to measure bbox sizes
    texts = [
        ax.text(
            float(lab_x[i]), float(lab_y[i]), str(labels[i]),
            fontsize=fontsize, color=color[i], ha=ha, va=va,
        )
        for i in range(n)
    ]

    # Precompute axis ranges for sane default step sizes.
    ax.figure.canvas.draw()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = max(abs(xlim[1] - xlim[0]), 1e-9)
    span_y = max(abs(ylim[1] - ylim[0]), 1e-9)
    step_x = push_anchor * span_x
    step_y = push_anchor * span_y
    step_lab_x = push_label * span_x
    step_lab_y = push_label * span_y

    for _ in range(max_iter):
        # Get bboxes of all labels in data units
        boxes = [_bbox_in_data(ax, t) for t in texts]
        moved = False

        # Force from anchors (data points)
        for i in range(n):
            x0, y0, x1, y1 = boxes[i]
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            for j in range(n):
                # Repel from anchor j (including own anchor if overlapping)
                ax_pt = pts_x[j]
                ay_pt = pts_y[j]
                if x0 <= ax_pt <= x1 and y0 <= ay_pt <= y1:
                    dx = cx - ax_pt
                    dy = cy - ay_pt
                    if dx == 0.0 and dy == 0.0:
                        # Choose deterministic upward push for tied case.
                        dy = step_y
                    norm = np.hypot(dx / span_x, dy / span_y) or 1.0
                    lab_x[i] += step_x * (dx / span_x) / norm
                    lab_y[i] += step_y * (dy / span_y) / norm
                    moved = True

        # Force from other labels (bbox overlap)
        for i in range(n):
            x0i, y0i, x1i, y1i = boxes[i]
            for j in range(i + 1, n):
                x0j, y0j, x1j, y1j = boxes[j]
                # axis-aligned overlap?
                ox = min(x1i, x1j) - max(x0i, x0j)
                oy = min(y1i, y1j) - max(y0i, y0j)
                if ox > 0.0 and oy > 0.0:
                    cxi = 0.5 * (x0i + x1i)
                    cyi = 0.5 * (y0i + y1i)
                    cxj = 0.5 * (x0j + x1j)
                    cyj = 0.5 * (y0j + y1j)
                    dx = cxi - cxj
                    dy = cyi - cyj
                    if dx == 0.0 and dy == 0.0:
                        dy = step_lab_y
                    norm = np.hypot(dx / span_x, dy / span_y) or 1.0
                    lab_x[i] += step_lab_x * (dx / span_x) / norm
                    lab_y[i] += step_lab_y * (dy / span_y) / norm
                    lab_x[j] -= step_lab_x * (dx / span_x) / norm
                    lab_y[j] -= step_lab_y * (dy / span_y) / norm
                    moved = True

        # Pull-back spring toward anchors (small magnitude)
        for i in range(n):
            lab_x[i] += pull_strength * (pts_x[i] - lab_x[i])
            lab_y[i] += pull_strength * (pts_y[i] - lab_y[i])

        # Keep labels within axis limits.
        lab_x = np.clip(lab_x, xlim[0], xlim[1])
        lab_y = np.clip(lab_y, ylim[0], ylim[1])

        # Apply new positions.
        for i, t in enumerate(texts):
            t.set_position((float(lab_x[i]), float(lab_y[i])))

        if not moved:
            break

    # Draw connector segments where the label moved away from its anchor.
    for i in range(n):
        if not (np.isclose(lab_x[i], pts_x[i]) and np.isclose(lab_y[i], pts_y[i])):
            ax.plot(
                [pts_x[i], lab_x[i]], [pts_y[i], lab_y[i]],
                color=segment_color, alpha=segment_alpha, linewidth=segment_lw,
                solid_capstyle="butt", zorder=1,
            )

    return texts
