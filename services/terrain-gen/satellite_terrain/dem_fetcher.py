"""
Fetch SRTM elevation data from the OpenTopoData public API.

Dataset : SRTM30m (~30 arc-second, ~1 km at equator)
API     : https://www.opentopodata.org/
No API key required. Rate limit: 1 request/s, 100 locations/request.

Strategy: query at a coarse grid (_SAMPLE_GRID x _SAMPLE_GRID), then
bicubically interpolate to the desired output resolution.  A 64x64
query needs 41 HTTP requests (~45 s); 32x32 needs 11 requests (~12 s).
"""

import json
import time
import urllib.parse
import urllib.request
from typing import NamedTuple

import numpy as np
from scipy.ndimage import zoom


class BBox(NamedTuple):
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float


_API_URL = "https://api.opentopodata.org/v1/srtm30m"
_BATCH_SIZE = 100
_REQUEST_DELAY = 1.1  # seconds between batches (respect rate limit)
_SAMPLE_GRID = 64     # coarse query grid; interpolated up to target_shape


def fetch(bbox: BBox, target_shape: tuple) -> np.ndarray:
    """
    Download SRTM elevation for a bounding box and interpolate to target_shape.

    Args:
        bbox:         (lat_min, lon_min, lat_max, lon_max)
        target_shape: (H, W) output resolution in pixels.

    Returns:
        heightmap: (H, W) float32 in [0, 1], where 1 = highest point in the scene.
    """
    lats = np.linspace(bbox.lat_max, bbox.lat_min, _SAMPLE_GRID)
    lons = np.linspace(bbox.lon_min, bbox.lon_max, _SAMPLE_GRID)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    coords = list(zip(lat_grid.ravel().tolist(), lon_grid.ravel().tolist()))

    elevations = _query_api(coords)

    raw = np.array(elevations, dtype=np.float32).reshape(_SAMPLE_GRID, _SAMPLE_GRID)
    raw = np.nan_to_num(raw, nan=0.0)
    raw = np.maximum(raw, 0.0)  # clamp below-sea-level to 0

    H, W = target_shape
    upsampled = zoom(raw, (H / _SAMPLE_GRID, W / _SAMPLE_GRID), order=3)
    upsampled = upsampled[:H, :W]

    h_max = upsampled.max()
    if h_max > 0:
        upsampled = upsampled / h_max

    return upsampled.astype(np.float32)


def _query_api(coords: list) -> list:
    elevations = []
    n_batches = (len(coords) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(coords), _BATCH_SIZE)):
        batch = coords[start : start + _BATCH_SIZE]
        print(f"    DEM batch {batch_idx + 1}/{n_batches} ({len(batch)} points)...", end="\r")

        locations = "|".join(f"{lat:.5f},{lon:.5f}" for lat, lon in batch)
        url = f"{_API_URL}?locations={urllib.parse.quote(locations)}"
        req = urllib.request.Request(url, headers={"User-Agent": "terrain-gen/1.0"})

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        for result in data["results"]:
            elev = result.get("elevation")
            elevations.append(float(elev) if elev is not None else 0.0)

        if start + _BATCH_SIZE < len(coords):
            time.sleep(_REQUEST_DELAY)

    print(f"    DEM fetch complete: {len(elevations)} points.    ")
    return elevations
