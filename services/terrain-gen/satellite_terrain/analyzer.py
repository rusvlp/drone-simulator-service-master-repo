"""
Multi-feature terrain elevation estimator for true-color satellite imagery.

Two analysis modes:

  analyze_neural  (neural, recommended)
    Step 1 — Semantic segmentation (SegFormer-ADE20K or DeepLabV3-VOC) assigns
             a base elevation tier to each pixel based on land-cover class:
             water ≈ 0.04, beach ≈ 0.11, grass ≈ 0.28, forest ≈ 0.46,
             hill ≈ 0.58, rock ≈ 0.74, mountain ≈ 0.82.
    Step 2 — Multi-scale texture roughness modulates elevation within each tier:
             flat plains → near-zero roughness, mountains → high roughness.
    Step 3 — Illumination relief captures slope-driven shading invisible to the
             classifier (slow brightness gradients on gentle slopes).
    Step 4 — Hard masks clamp water to floor and snow to ceiling.

  analyze  (classic / rule-based)
    Same texture + relief pipeline but uses the hand-crafted HSV spectral
    classifier instead of a neural network.  No torch dependency.

Fusion weights (neural mode):
  0.55 × neural segment + 0.28 × roughness + 0.17 × relief

Fusion weights (classic mode):
  0.55 × spectral       + 0.30 × roughness + 0.15 × relief
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .classifier import classify as _spectral


# ---------------------------------------------------------------------------
# Shared feature extractors
# ---------------------------------------------------------------------------

def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]).astype(np.float64)


def _local_roughness(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Local standard deviation via E[x²] − E[x]²."""
    mu  = gaussian_filter(gray, sigma=sigma)
    mu2 = gaussian_filter(gray ** 2, sigma=sigma)
    return np.sqrt(np.maximum(mu2 - mu ** 2, 0.0))


def _multiscale_roughness(gray: np.ndarray) -> np.ndarray:
    """
    Weighted sum of local roughness at three spatial scales.

    Scale σ is relative to image size so the algorithm works on any resolution.
      fine   (D/80)  — rock texture, tree canopy gaps, shadows
      medium (D/30)  — ridgelines, individual hills
      coarse (D/10)  — mountain massifs vs. flat plains
    """
    D = max(gray.shape)
    r = (0.50 * _local_roughness(gray, D / 80)
       + 0.35 * _local_roughness(gray, D / 30)
       + 0.15 * _local_roughness(gray, D / 10))
    r_max = r.max()
    return r / r_max if r_max > 1e-9 else r


def _illumination_relief(gray: np.ndarray) -> np.ndarray:
    """
    Band-pass luminance gradient as a proxy for slope-driven shading.

    Removes very fine texture (σ_lo) and very broad albedo variation (σ_hi),
    keeps the mid-frequency component that correlates with terrain slope.
    Output is shifted to [0, 1] so that mean slope ≈ 0.5.
    """
    D = max(gray.shape)
    lo   = gaussian_filter(gray, sigma=D / 120)
    hi   = gaussian_filter(gray, sigma=D / 8)
    band = lo - hi                              # positive → locally bright ridge
    std  = band.std()
    if std > 1e-9:
        band = band / (3.0 * std)              # 3σ clip
    return np.clip(band + 0.5, 0.0, 1.0)


def _water_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b > r + 0.04) & (b > g + 0.02) & ((r + g + b) / 3 < 0.70)


def _snow_mask(rgb: np.ndarray) -> np.ndarray:
    brightness = rgb.mean(axis=-1)
    spread     = rgb.max(axis=-1) - rgb.min(axis=-1)
    return (brightness > 0.82) & (spread < 0.13)


# ---------------------------------------------------------------------------
# Neural analysis (SegFormer / DeepLabV3)
# ---------------------------------------------------------------------------

def analyze_neural(
    image_rgb: np.ndarray,
    backbone: str = "segformer",
    device: str = "auto",
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Estimate a terrain heightmap using a neural segmentation backbone.

    Args:
        image_rgb: (H, W, 3) float32 array, values in [0, 1].
        backbone:  'segformer' (ADE20K, default) or 'deeplabv3' (Pascal VOC).
        device:    'auto', 'cpu', 'cuda', or 'mps'.
        tile_size: Tile size for large-image tiling (default: 512).
        overlap:   Overlap between tiles in pixels (default: 64).

    Returns:
        raw_heights: (H, W) float32 array, values in [0, 1].
    """
    from .segmenter import segment as _neural

    gray = _luminance(image_rgb)

    h_neural    = _neural(image_rgb, backbone=backbone, device=device,
                          tile_size=tile_size, overlap=overlap).astype(np.float64)
    h_roughness = _multiscale_roughness(gray)
    h_relief    = _illumination_relief(gray)

    height = (
        0.55 * h_neural
        + 0.28 * h_roughness
        + 0.17 * h_relief
    )

    water = _water_mask(image_rgb)
    snow  = _snow_mask(image_rgb)

    height[water] *= 0.06

    snow_brightness = image_rgb[snow].mean(axis=-1)
    height[snow] = height[snow] * (1.0 - snow_brightness) + snow_brightness

    h_min, h_max = height.min(), height.max()
    if h_max > h_min:
        height = (height - h_min) / (h_max - h_min)

    return height.astype(np.float32)


# ---------------------------------------------------------------------------
# Classic analysis (HSV rule-based)
# ---------------------------------------------------------------------------

def analyze(image_rgb: np.ndarray) -> np.ndarray:
    """
    Estimate a terrain heightmap using the rule-based spectral classifier.

    Fuses HSV color classification with multi-scale texture roughness and
    illumination relief.  No neural dependency.

    Args:
        image_rgb: (H, W, 3) float32 array, values in [0, 1].

    Returns:
        raw_heights: (H, W) float32 array, values in [0, 1].
    """
    gray = _luminance(image_rgb)

    h_spectral  = _spectral(image_rgb).astype(np.float64)
    h_roughness = _multiscale_roughness(gray)
    h_relief    = _illumination_relief(gray)

    height = (
        0.55 * h_spectral
        + 0.30 * h_roughness
        + 0.15 * h_relief
    )

    water = _water_mask(image_rgb)
    snow  = _snow_mask(image_rgb)

    height[water] *= 0.06

    snow_brightness = image_rgb[snow].mean(axis=-1)
    height[snow] = height[snow] * (1.0 - snow_brightness) + snow_brightness

    h_min, h_max = height.min(), height.max()
    if h_max > h_min:
        height = (height - h_min) / (h_max - h_min)

    return height.astype(np.float32)
