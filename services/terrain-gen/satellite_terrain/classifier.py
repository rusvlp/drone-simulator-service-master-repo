import numpy as np


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB [0,1] → HSV [0,1] conversion."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.where(maxc > 1e-9, delta / maxc, 0.0)

    h = np.zeros_like(v)
    mask_r = (maxc == r) & (delta > 1e-9)
    mask_g = (maxc == g) & (delta > 1e-9)
    mask_b = (maxc == b) & (delta > 1e-9)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0
    h = h / 6.0

    return np.stack([h, s, v], axis=-1)


# Terrain class definitions in HSV space.
# Each entry: (hue_center, hue_width, sat_center, sat_width, val_center, val_width, elevation)
# Elevation is the target height value in [0, 1].
# hue_width=0.5 effectively makes hue irrelevant (used for achromatic classes like snow/rock).
_TERRAIN_CLASSES = [
    # (hc,   hw,   sc,   sw,   vc,   vw,   elevation)
    (0.615, 0.04, 0.70, 0.20, 0.35, 0.20, 0.02),  # deep water    — dark blue
    (0.575, 0.05, 0.55, 0.20, 0.62, 0.20, 0.07),  # shallow water — cyan-blue
    (0.130, 0.05, 0.30, 0.15, 0.88, 0.10, 0.12),  # beach / sand  — pale yellow
    (0.085, 0.06, 0.38, 0.20, 0.65, 0.20, 0.24),  # desert / arid — orange-brown
    (0.215, 0.05, 0.45, 0.20, 0.58, 0.20, 0.31),  # grassland     — yellow-green
    (0.300, 0.06, 0.58, 0.20, 0.45, 0.20, 0.46),  # forest        — mid green
    (0.325, 0.04, 0.65, 0.15, 0.28, 0.15, 0.55),  # dense forest  — dark green
    (0.180, 0.05, 0.38, 0.18, 0.48, 0.18, 0.38),  # shrubland     — olive
    (0.065, 0.07, 0.22, 0.15, 0.52, 0.20, 0.67),  # bare highland — brown-gray
    (0.000, 0.50, 0.10, 0.10, 0.55, 0.22, 0.78),  # mountain rock — gray (hue-agnostic)
    (0.000, 0.50, 0.06, 0.06, 0.92, 0.08, 0.95),  # snow / ice    — white (hue-agnostic)
]


def classify(image_rgb: np.ndarray) -> np.ndarray:
    """
    Map a true-color satellite image to a raw heightmap via soft terrain classification.

    Each pixel is assigned a height by computing weighted membership across all terrain
    classes (Gaussian soft assignment in HSV space), then taking the weighted mean elevation.
    Pixels that don't match any class well fall back to a brightness proxy.

    Args:
        image_rgb: (H, W, 3) float32 array with values in [0, 1].

    Returns:
        heightmap: (H, W) float32 array with values in [0, 1].
    """
    hsv = _rgb_to_hsv(image_rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    height = np.zeros(h.shape, dtype=np.float64)
    total_weight = np.zeros(h.shape, dtype=np.float64)

    for hc, hw, sc, sw, vc, vw, elevation in _TERRAIN_CLASSES:
        # Hue distance respects circular topology of the hue wheel.
        hue_dist = np.minimum(np.abs(h - hc), 1.0 - np.abs(h - hc))
        w = (
            np.exp(-0.5 * (hue_dist / hw) ** 2)
            * np.exp(-0.5 * ((s - sc) / sw) ** 2)
            * np.exp(-0.5 * ((v - vc) / vw) ** 2)
        )
        height += w * elevation
        total_weight += w

    classified = total_weight > 1e-6
    height[classified] /= total_weight[classified]

    # Unclassified pixels (urban, industrial, mixed): use brightness as rough proxy.
    height[~classified] = v[~classified] * 0.5

    return height.astype(np.float32)
