"""
Satellite Image -> Terrain Heightmap Converter
===============================================
Generates a terrain heightmap from a true-color satellite image.

Methods (--method):

  segment  [DEFAULT]
    Neural segmentation: SegFormer-B2 (ADE20K) or DeepLabV3-ResNet101 (VOC)
    maps land-cover classes to elevation tiers, then fuses with multi-scale
    texture roughness and illumination relief for sub-class detail.
    First run downloads the model weights (~200-240 MB, cached afterward).

    python main.py photo.jpg output/
    python main.py photo.jpg output/ --method segment --backbone segformer
    python main.py photo.jpg output/ --method segment --backbone deeplabv3
    python main.py photo.jpg output/ --method segment --device cpu

  analyze
    Rule-based: HSV spectral classification + texture roughness + illumination
    relief.  No neural dependency, no internet required.

    python main.py photo.jpg output/ --method analyze

  color
    Spectral-only: infers elevation from HSV color alone.  Fastest, least
    accurate on complex terrain.

    python main.py photo.jpg output/ --method color

  dem
    Downloads real SRTM30m elevation data from OpenTopoData for the
    geographic bounding box of the image.  Most accurate; requires internet
    and known coordinates.

    python main.py photo.jpg output/ --bbox 55.50 37.30 56.00 38.00
    python main.py photo.jpg output/ --bbox 55.50 37.30 56.00 38.00 --dem-grid 32

DEMO MODE  (no input image needed):
    python main.py --demo output/
    python main.py --demo output/ --method segment
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import satellite_terrain as st


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def _subsample(heightmap: np.ndarray, max_dim: int) -> np.ndarray:
    H, W = heightmap.shape
    if max(H, W) <= max_dim:
        return heightmap
    scale = max_dim / max(H, W)
    new_w, new_h = int(W * scale), int(H * scale)
    img = Image.fromarray((heightmap * 255).astype(np.uint8), mode="L")
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Demo: synthetic satellite-like image via fractal noise
# ---------------------------------------------------------------------------

def _fractal_noise(H: int, W: int, octaves: int = 8, seed: int = 42) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    noise = np.zeros((H, W), dtype=np.float64)
    amplitude, frequency = 1.0, 1.0
    for _ in range(octaves):
        layer = rng.standard_normal((H, W))
        layer = gaussian_filter(layer, sigma=max(H, W) / (4.0 * frequency))
        layer = (layer - layer.min()) / (layer.max() - layer.min() + 1e-9)
        noise += amplitude * layer
        amplitude *= 0.5
        frequency *= 2.0
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    return noise.astype(np.float32)


def make_demo_image(H: int = 512, W: int = 512) -> np.ndarray:
    elev = _fractal_noise(H, W)
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    def zone(lo, hi, r, g, b, r2=None, g2=None, b2=None):
        mask = (elev >= lo) & (elev < hi)
        t = (elev[mask] - lo) / (hi - lo)
        if r2 is None:
            rgb[mask] = [r, g, b]
        else:
            rgb[mask, 0] = r + t * (r2 - r)
            rgb[mask, 1] = g + t * (g2 - g)
            rgb[mask, 2] = b + t * (b2 - b)

    zone(0.00, 0.12, 0.05, 0.12, 0.55,  0.07, 0.20, 0.72)
    zone(0.12, 0.20, 0.10, 0.55, 0.75,  0.75, 0.80, 0.50)
    zone(0.20, 0.25, 0.75, 0.80, 0.50,  0.55, 0.70, 0.30)
    zone(0.25, 0.45, 0.25, 0.58, 0.20,  0.18, 0.50, 0.15)
    zone(0.45, 0.65, 0.15, 0.42, 0.12,  0.35, 0.30, 0.15)
    zone(0.65, 0.80, 0.48, 0.40, 0.28,  0.72, 0.72, 0.70)
    zone(0.80, 1.01, 0.80, 0.80, 0.80,  1.00, 1.00, 1.00)

    rng = np.random.default_rng(0)
    rgb += rng.uniform(-0.015, 0.015, rgb.shape).astype(np.float32)
    return rgb.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process(
    image_rgb: np.ndarray,
    output_dir: Path,
    stem: str,
    smooth_sigma: float,
    sea_level: float,
    export_obj: bool,
    bbox: list | None,
    dem_grid: int,
    method: str,
    backbone: str,
    device: str,
    tile_size: int,
    overlap: int,
) -> None:
    H, W = image_rgb.shape[:2]
    print(f"  Input size : {W}x{H} px")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Elevation source ---------------------------------------------------
    if bbox:
        from satellite_terrain.dem_fetcher import BBox, fetch as fetch_dem
        lat_min, lon_min, lat_max, lon_max = bbox
        print(f"  [dem] Fetching SRTM elevation  ({lat_min},{lon_min}) -> ({lat_max},{lon_max})")
        print(f"  Query grid: {dem_grid}x{dem_grid}  "
              f"({(dem_grid*dem_grid + 99)//100} HTTP requests)")

        import satellite_terrain.dem_fetcher as _dem
        _dem._SAMPLE_GRID = dem_grid
        hmap = fetch_dem(BBox(lat_min, lon_min, lat_max, lon_max), (H, W))

        tex_filename = f"{stem}_satellite.jpg"
        tex_path = output_dir / tex_filename
        Image.fromarray((image_rgb * 255).astype(np.uint8), mode="RGB").save(str(tex_path))
        obj_texture = tex_filename
        print(f"  -> {tex_path}  (satellite texture)")

    elif method == "segment":
        from satellite_terrain.segmenter import resolve_device, cuda_info
        device = resolve_device(device)
        device_label = cuda_info() if device.startswith("cuda") else device.upper()
        print(f"  [segment] Neural segmentation  backbone={backbone}  device={device_label}  "
              f"tile={tile_size}px  overlap={overlap}px")
        raw = st.analyze_neural(image_rgb, backbone=backbone, device=device,
                                tile_size=tile_size, overlap=overlap)
        print(f"  Smoothing  (sigma={smooth_sigma}, sea_level={sea_level})...")
        hmap = st.build(raw, smooth_sigma=smooth_sigma, sea_level=sea_level)
        obj_texture = f"{stem}_terrain.png"

    elif method == "color":
        print("  [color] Spectral classification...")
        raw = st.classify(image_rgb)
        print(f"  Smoothing  (sigma={smooth_sigma}, sea_level={sea_level})...")
        hmap = st.build(raw, smooth_sigma=smooth_sigma, sea_level=sea_level)
        obj_texture = f"{stem}_terrain.png"

    else:  # method == "analyze"
        print("  [analyze] Spectral + texture roughness + illumination relief...")
        raw = st.analyze(image_rgb)
        print(f"  Smoothing  (sigma={smooth_sigma}, sea_level={sea_level})...")
        hmap = st.build(raw, smooth_sigma=smooth_sigma, sea_level=sea_level)
        obj_texture = f"{stem}_terrain.png"

    # --- Export -------------------------------------------------------------
    hm_path = output_dir / f"{stem}_heightmap.png"
    st.save_heightmap(hmap, hm_path)
    print(f"  -> {hm_path}")

    color_path = output_dir / f"{stem}_terrain.png"
    st.save_colored_terrain(hmap, color_path)
    print(f"  -> {color_path}")

    if export_obj:
        mesh_hmap = _subsample(hmap, max_dim=512)
        mH, mW = mesh_hmap.shape
        obj_path = output_dir / f"{stem}_terrain.obj"
        st.save_obj(mesh_hmap, obj_path, texture_name=obj_texture)
        print(f"  -> {obj_path}  ({mW}x{mH} mesh, "
              f"{mW * mH:,} vertices, {2*(mW-1)*(mH-1):,} triangles)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a satellite image to terrain heightmap and 3D mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", type=Path,
        help="Input satellite image (JPG, PNG, TIFF, ...). Omit when using --demo.",
    )
    parser.add_argument(
        "output", type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--method", choices=["segment", "analyze", "color"], default="segment",
        help=(
            "Heightmap extraction method (ignored when --bbox is used). "
            "'segment' (default): neural semantic segmentation + texture + relief. "
            "'analyze': rule-based spectral + texture + illumination relief. "
            "'color': spectral colour classification only."
        ),
    )
    parser.add_argument(
        "--backbone", choices=["segformer", "deeplabv3"], default="segformer",
        help=(
            "Neural backbone for --method segment. "
            "'segformer' (default): SegFormer-B2 on ADE20K — best terrain classes. "
            "'deeplabv3': DeepLabV3-ResNet101 on Pascal VOC — lighter, fewer terrain classes."
        ),
    )
    parser.add_argument(
        "--device", default="auto",
        help="Compute device for neural inference: 'auto' (default), 'cpu', 'cuda', 'mps'.",
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        help=(
            "Geographic bounding box of the image. "
            "When provided, downloads real SRTM elevation instead of classifying colors. "
            "Example: --bbox 55.50 37.30 56.00 38.00  (Moscow area)"
        ),
    )
    parser.add_argument(
        "--dem-grid", type=int, default=64, metavar="N",
        help="Grid resolution for DEM queries (NxN points). "
             "Smaller = faster, less detail. Default: 64 (~45 s). Try 32 for ~12 s.",
    )
    parser.add_argument(
        "--smooth", type=float, default=3.0, metavar="SIGMA",
        help="Gaussian smoothing sigma, pixels (default: 3.0).",
    )
    parser.add_argument(
        "--sea-level", type=float, default=0.08, metavar="LEVEL",
        help="Height fraction [0-1] treated as ocean (default: 0.08).",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512, metavar="PX",
        help=(
            "Tile size for large-image tiling (default: 512). "
            "Images wider/taller than 1024 px are split automatically."
        ),
    )
    parser.add_argument(
        "--overlap", type=int, default=64, metavar="PX",
        help="Overlap between adjacent tiles in pixels (default: 64).",
    )
    parser.add_argument(
        "--no-obj", action="store_true",
        help="Skip OBJ mesh export.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate a synthetic fractal satellite image for testing.",
    )
    parser.add_argument(
        "--demo-size", type=int, default=512, metavar="PX",
        help="Side length of the synthetic demo image (default: 512).",
    )
    args = parser.parse_args()

    if not args.demo and args.input is None:
        parser.error("Provide an input image or use --demo.")

    if args.demo:
        print(f"Generating {args.demo_size}x{args.demo_size} synthetic satellite image...")
        image = make_demo_image(args.demo_size, args.demo_size)
        stem = "demo"
        args.output.mkdir(parents=True, exist_ok=True)
        demo_path = args.output / "demo_satellite.png"
        Image.fromarray((image * 255).astype(np.uint8), mode="RGB").save(str(demo_path))
        print(f"  -> {demo_path}")
    else:
        if not args.input.exists():
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {args.input.name}...")
        image = load_image(args.input)
        stem = args.input.stem

    process(
        image_rgb=image,
        output_dir=args.output,
        stem=stem,
        smooth_sigma=args.smooth,
        sea_level=args.sea_level,
        export_obj=not args.no_obj,
        bbox=args.bbox,
        method=args.method,
        backbone=args.backbone,
        device=args.device,
        dem_grid=args.dem_grid,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    print("Done.")


if __name__ == "__main__":
    main()
