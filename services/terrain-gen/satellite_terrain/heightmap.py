import numpy as np
from scipy.ndimage import gaussian_filter


def build(
    raw_heights: np.ndarray,
    smooth_sigma: float = 3.0,
    sea_level: float = 0.08,
) -> np.ndarray:
    """
    Apply Gaussian smoothing, sea-level clamping, and contrast normalization.

    The pipeline:
      1. Smooth to remove classification noise and produce natural-looking terrain.
      2. Clamp everything below `sea_level` to 0 (ocean floor stays flat).
      3. Stretch the remaining range to fill [0, 1].

    Args:
        raw_heights:  (H, W) float32 classification output, values in [0, 1].
        smooth_sigma: Standard deviation for Gaussian kernel (pixels).
                      Larger values create smoother, more gradual terrain.
        sea_level:    Fraction of the pre-smoothing height range treated as ocean.

    Returns:
        heightmap: (H, W) float32 array, values in [0, 1].
    """
    smoothed = gaussian_filter(raw_heights.astype(np.float64), sigma=smooth_sigma)

    below_sea = smoothed < sea_level
    smoothed[below_sea] = 0.0
    land = smoothed[~below_sea]
    if land.size > 0:
        smoothed[~below_sea] = (land - sea_level) / (1.0 - sea_level)

    h_min, h_max = smoothed.min(), smoothed.max()
    if h_max > h_min:
        smoothed = (smoothed - h_min) / (h_max - h_min)

    return smoothed.astype(np.float32)
