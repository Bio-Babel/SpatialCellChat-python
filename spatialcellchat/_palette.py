"""Color palette helpers ported from ``visualization.R`` and ``utilities.R``.

Implements:
- :func:`colorRamp3` — modified ``circlize::colorRamp3``
  (default ``space="RGB"`` interpolates in *linear* RGB and then converts to
  sRGB hex, exactly as ``colorspace::sRGB`` / ``colorspace::RGB`` do in R).
- :func:`ggPalette` — equivalent of ``scales::hue_pal`` /
  ``grDevices::hcl(h, l=65, c=100)`` returning evenly spaced HCL colors.
- :func:`scPalette` — the literal 26-color palette baked into
  ``visualization.R``; for ``n > 26`` the palette is stretched with R's
  ``colorRampPalette`` algorithm (linear interpolation in sRGB).
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
from matplotlib import colors as mcolors

__all__ = [
    "colorRamp3",
    "ggPalette",
    "scPalette",
]


# ---------------------------------------------------------------------------
# sRGB <-> linear RGB conversion (matches colorspace package)
# ---------------------------------------------------------------------------
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Vectorised sRGB → linear-RGB transform (IEC 61966-2-1)."""
    out = np.where(
        c > 0.04045,
        np.power((c + 0.055) / 1.055, 2.4),
        c / 12.92,
    )
    return out


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Vectorised linear-RGB → sRGB transform (IEC 61966-2-1)."""
    out = np.where(
        c > 0.0031308,
        1.055 * np.power(np.clip(c, 0, None), 1.0 / 2.4) - 0.055,
        12.92 * c,
    )
    return out


def _hex_from_rgb(rgb: np.ndarray, alpha: str) -> np.ndarray:
    """Convert an ``(N, 3)`` array of sRGB floats to ``#RRGGBBAA`` strings.

    Mirrors R's :func:`colorspace::hex` rounding (``round(x*255)``) and uses
    upper-case hex output to match :func:`circlize::colorRamp3`.
    """
    clipped = np.clip(rgb, 0.0, 1.0)
    ints = np.round(clipped * 255).astype(int)
    out = np.empty(ints.shape[0], dtype=object)
    for k in range(ints.shape[0]):
        out[k] = "#%02X%02X%02X%s" % (ints[k, 0], ints[k, 1], ints[k, 2], alpha)
    return out


def _hex_from_rgb_trunc(rgb: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3)`` sRGB floats to ``"#RRGGBB"`` using R's :func:`rgb`
    *truncating* convention (mirrors ``maxColorValue = 255``)."""
    clipped = np.clip(rgb, 0.0, 1.0)
    ints = (clipped * 255).astype(int)  # floor towards zero
    out = np.empty(ints.shape[0], dtype=object)
    for k in range(ints.shape[0]):
        out[k] = "#%02X%02X%02X" % (ints[k, 0], ints[k, 1], ints[k, 2])
    return out


_R_COLOR_OVERRIDES: dict = {
    # R's `colors()` (X11 names) differ from CSS4 for these aliases.
    "green": "#00FF00",
    "gray": "#BEBEBE",
    "grey": "#BEBEBE",
    "purple": "#A020F0",
    "maroon": "#B03060",
}


def _hex_to_rgb01(colors: Sequence[str]) -> np.ndarray:
    """Convert R-style colour names / hex codes to an ``(N, 3)`` sRGB array.

    Falls back to matplotlib for hex codes; remaps a small set of CSS↔X11
    aliases so that named colours match R's :func:`grDevices::colors`.
    """
    out = np.empty((len(colors), 3), dtype=float)
    for k, c in enumerate(colors):
        key = c.lower() if isinstance(c, str) else c
        resolved = _R_COLOR_OVERRIDES.get(key, c)
        rgba = mcolors.to_rgb(resolved)
        out[k, 0], out[k, 1], out[k, 2] = rgba
    return out


# ---------------------------------------------------------------------------
# colorRamp3 (R: utilities.R:1881)
# ---------------------------------------------------------------------------
def colorRamp3(
    breaks: Sequence[float],
    colors: Sequence[str],
    transparency: float = 0.0,
    space: str = "RGB",
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """Linear-interpolation colour ramp (port of ``circlize::colorRamp3``).

    Parameters
    ----------
    breaks : sequence of float
        Numeric break points; need not be sorted.
    colors : sequence of str
        Colour at each break (matplotlib-recognised names or ``"#RRGGBB"``).
    transparency : float, default ``0.0``
        Single value in ``[0, 1]``; ``0`` means fully opaque.
    space : str, default ``"RGB"``
        Currently only ``"RGB"`` is supported — that is the value used
        throughout the R package. Other values raise ``NotImplementedError``.

    Returns
    -------
    Callable[[ndarray], ndarray]
        A function mapping numeric values to ``"#RRGGBBAA"`` strings.
    """
    breaks_arr = np.asarray(breaks, dtype=float)
    color_rgb = _hex_to_rgb01(list(colors))

    if breaks_arr.shape[0] != color_rgb.shape[0]:
        raise ValueError("Length of `breaks` should be equal to `colors`.\n")

    # 1. Sort by break value
    order = np.argsort(breaks_arr, kind="stable")
    breaks_arr = breaks_arr[order]
    color_rgb = color_rgb[order]

    # 2. Drop duplicated break values (mirrors R's ``duplicated``)
    keep = np.ones(breaks_arr.shape[0], dtype=bool)
    seen: set = set()
    for k, v in enumerate(breaks_arr):
        if v in seen:
            keep[k] = False
        else:
            seen.add(v)
    breaks_arr = breaks_arr[keep]
    color_rgb = color_rgb[keep]

    if breaks_arr.shape[0] == 1:
        raise ValueError("You should have at least two distinct break values.")

    if space != "RGB":
        # All callers in SpatialCellChat use the default `RGB` space.
        raise NotImplementedError(
            "Only space='RGB' is supported in this port (matches all callers)."
        )

    # Convert sRGB anchors to linear RGB for interpolation in `RGB` space.
    color_lin = _srgb_to_linear(color_rgb)

    # Pre-compute alpha hex (mirrors R: 1 - clip(transparency, 0, 1))
    t = max(0.0, min(1.0, float(transparency)))
    alpha = 1.0 - t
    alpha_int = int(round(alpha * 255))
    alpha_hex = f"{alpha_int:02X}"

    def fun(x: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        l_na = np.isnan(x_arr)
        if np.all(l_na):
            return np.array([None] * x_arr.size, dtype=object)

        x2 = x_arr[~l_na]
        # Clip to break range
        x2 = np.clip(x2, breaks_arr[0], breaks_arr[-1])

        # ``.bincode(x, breaks, right=TRUE, include.lowest=TRUE)`` returns the
        # 1-based bin index ``i`` such that breaks[i-1] < x <= breaks[i],
        # with the leftmost edge inclusive. Replicate via searchsorted.
        ibin = np.searchsorted(breaks_arr, x2, side="left")
        # Values exactly equal to breaks[0] should fall into bin 1 (Python 0)
        ibin = np.where(ibin == 0, 1, ibin)
        ibin = ibin - 1  # zero-based index of left anchor

        res_lin = np.empty((x2.shape[0], 3), dtype=float)
        for i in np.unique(ibin):
            sel = ibin == i
            x_sel = x2[sel]
            b1, b2 = breaks_arr[i], breaks_arr[i + 1]
            c1, c2 = color_lin[i], color_lin[i + 1]
            # ``(x - break2)*(col2 - col1)/(break2 - break1) + col2``
            for j in range(3):
                res_lin[sel, j] = (
                    (x_sel - b2) * (c2[j] - c1[j]) / (b2 - b1) + c2[j]
                )

        # Clip linear values, then convert back to sRGB and hex
        res_srgb = _linear_to_srgb(np.clip(res_lin, 0.0, 1.0))
        hex_no_na = _hex_from_rgb(res_srgb, alpha_hex)

        out = np.empty(x_arr.size, dtype=object)
        out[l_na] = None
        out[~l_na] = hex_no_na
        return out

    return fun


# ---------------------------------------------------------------------------
# ggPalette (R: visualization.R:778)
# ---------------------------------------------------------------------------
def ggPalette(n: int) -> np.ndarray:
    """Generate ``n`` evenly spaced HCL colours like R's ``hcl(h, l=65, c=100)``.

    Returns
    -------
    numpy.ndarray
        Array of length ``n`` containing ``"#RRGGBB"`` strings.
    """
    hues = np.linspace(15.0, 375.0, n + 1)[:n]
    return _hcl_to_hex(hues, l=65.0, c=100.0)


def _hcl_to_hex(h: np.ndarray, l: float, c: float) -> np.ndarray:
    """Convert HCL (= polarLUV) colours to ``#RRGGBB`` strings.

    Implements ``grDevices::hcl`` precisely: first to CIE-LUV, then to XYZ
    (D65), then to sRGB, with R's default fixup behaviour (clip channels to
    [0, 1] before hex conversion).
    """
    h_rad = np.deg2rad(h)
    # polarLUV -> LUV
    L = np.full_like(h, l, dtype=float)
    U = c * np.cos(h_rad)
    V = c * np.sin(h_rad)

    # LUV -> XYZ (D65 reference white from R / colorspace)
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    un_p = 4.0 * Xn / (Xn + 15.0 * Yn + 3.0 * Zn)
    vn_p = 9.0 * Yn / (Xn + 15.0 * Yn + 3.0 * Zn)

    # When L == 0 -> XYZ = 0
    Y = np.where(
        L > 8.0,
        Yn * np.power((L + 16.0) / 116.0, 3.0),
        Yn * L / 903.3,
    )
    a = (52.0 * L / np.where(U + 13.0 * L * un_p == 0, 1, U + 13.0 * L * un_p) - 1.0) / 3.0
    b = -5.0 * Y
    c_ = -1.0 / 3.0
    d = Y * (39.0 * L / np.where(V + 13.0 * L * vn_p == 0, 1, V + 13.0 * L * vn_p) - 5.0)
    X = np.where(L == 0, 0.0, (d - b) / (a - c_))
    Z = np.where(L == 0, 0.0, X * a + b)

    # XYZ (D65) -> linear sRGB (matrix from IEC 61966-2-1 / colorspace)
    M = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])
    XYZ = np.stack([X / 100.0, Y / 100.0, Z / 100.0], axis=1)
    linear = XYZ @ M.T

    # linear sRGB -> sRGB and hex (with R's default fixup → clip)
    srgb = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    return _hex_from_rgb(srgb, alpha="").astype(str)


# ---------------------------------------------------------------------------
# scPalette (R: visualization.R:792)
# ---------------------------------------------------------------------------
_SC_PALETTE: tuple = (
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#F29403", "#F781BF",
    "#BC9DCC", "#A65628", "#54B0E4", "#222F75", "#1B9E77", "#B2DF8A",
    "#E3BE00", "#FB9A99", "#E7298A", "#910241", "#00CDD1", "#A6CEE3",
    "#CE1261", "#5E4FA2", "#8CA77B", "#00441B", "#DEDC00", "#B3DE69",
    "#8DD3C7", "#999999",
)


def scPalette(n: int) -> np.ndarray:
    """Single-cell visualization palette.

    Returns the literal 26-colour palette baked into ``visualization.R``.
    For ``n > 26`` the palette is stretched with R's ``colorRampPalette``
    algorithm — linear interpolation in sRGB across the 26 anchors.

    Parameters
    ----------
    n : int
        Number of colours.

    Returns
    -------
    numpy.ndarray
        Array of length ``n`` containing uppercase ``"#RRGGBB"`` strings.
    """
    if n <= len(_SC_PALETTE):
        return np.array(_SC_PALETTE[:n], dtype=object)
    return _r_color_ramp_palette(_SC_PALETTE, n)


def _r_color_ramp_palette(colors: Sequence[str], n: int) -> np.ndarray:
    """Mirror ``grDevices::colorRampPalette(colors)(n)``.

    ``colorRampPalette`` defaults to ``space="rgb"`` and
    ``interpolate="linear"``: anchor colours are converted to sRGB
    ``[0, 1]`` floats (via :func:`col2rgb` / 255), each channel is linearly
    interpolated with R's ``approxfun`` formula
    ``y1 + (y2 - y1) * ((v - x1) / (x2 - x1))`` (operator order matters
    bit-for-bit), and the final ``rgb(..., maxColorValue = 255)`` truncates
    toward zero rather than rounding.
    """
    rgb01 = _hex_to_rgb01(list(colors))
    if rgb01.shape[0] == 1:
        rgb01 = np.vstack([rgb01, rgb01])
    anchors = np.linspace(0.0, 1.0, rgb01.shape[0])
    samples = np.linspace(0.0, 1.0, n)
    out = _r_approx(samples, anchors, rgb01)
    return _hex_from_rgb_trunc(out).astype(str)


def _r_approx(v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation matching R's ``approxfun`` operator order.

    Parameters
    ----------
    v : (m,) ndarray
        Sample positions.
    x : (k,) ndarray
        Sorted anchor positions.
    y : (k, c) ndarray
        Anchor values per channel.

    Returns
    -------
    (m, c) ndarray
        Interpolated values.
    """
    out = np.empty((v.shape[0], y.shape[1]), dtype=float)
    # Find the bin index for each sample (right-side, last bin includes endpoint).
    idx = np.searchsorted(x, v, side="right") - 1
    idx = np.clip(idx, 0, x.shape[0] - 2)
    x1 = x[idx]
    x2 = x[idx + 1]
    for c in range(y.shape[1]):
        y1 = y[idx, c]
        y2 = y[idx + 1, c]
        out[:, c] = y1 + (y2 - y1) * ((v - x1) / (x2 - x1))
    return out
