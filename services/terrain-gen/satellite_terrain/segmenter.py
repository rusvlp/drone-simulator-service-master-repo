"""
Neural terrain height estimator via semantic segmentation.

Two backends are available:

  segformer  (default, recommended)
    SegFormer-B2 fine-tuned on ADE20K-150 via HuggingFace Transformers.
    ADE20K has rich outdoor / landscape vocabulary: mountain, hill, rock,
    sand, grass, field, forest, water, sea, river, lake, snow, etc.
    Requires: transformers, torch

  deeplabv3
    DeepLabV3-ResNet101 pretrained on COCO + Pascal VOC via torchvision.
    Fewer terrain classes (21 VOC labels); most pixels fall back to spectral
    classification. Useful when transformers is unavailable.
    Requires: torchvision, torch

Both backends support automatic tiling for large images: when max(H, W)
exceeds TILE_THRESHOLD, the image is split into overlapping tiles, each
processed independently, and results are blended back with a linear-ramp
weight that ensures smooth seams at tile boundaries.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tiling defaults
# ---------------------------------------------------------------------------

TILE_SIZE      = 512   # px — matches SegFormer's training resolution
TILE_OVERLAP   = 64    # px — overlap between adjacent tiles
TILE_THRESHOLD = 1024  # px — auto-tile when max(H, W) exceeds this


# ---------------------------------------------------------------------------
# ADE20K-150 class index (0-based) → base elevation [0, 1]
# ---------------------------------------------------------------------------
_ADE20K_ELEV: dict[int, float] = {
    4:   0.46,   # tree / forest
    9:   0.28,   # grass
    13:  0.22,   # earth / ground
    16:  0.82,   # mountain
    17:  0.32,   # plant / shrub
    21:  0.04,   # water (inland)
    26:  0.04,   # sea
    29:  0.18,   # field
    34:  0.74,   # rock / stone
    46:  0.11,   # sand / beach
    60:  0.04,   # river
    68:  0.58,   # hill
    72:  0.32,   # palm tree
    94:  0.22,   # land / soil
    113: 0.50,   # waterfall
    128: 0.04,   # lake
}

# Sky is atmospheric — not terrain; falls back to spectral classifier.
_ADE20K_SKY_CLS = 2

# ADE20K classes that indicate tree/shrub/plant vegetation (used for tree placement)
_ADE20K_VEG_CLASSES: frozenset[int] = frozenset([
    4,   # tree / forest
    17,  # plant / shrub
    72,  # palm tree
    9,   # grass (may include tree-covered slopes at satellite scale)
])

# ADE20K-150 class index → RGB color for segmentation texture
_ADE20K_COLORS: dict[int, tuple[int, int, int]] = {
    4:   ( 34, 139,  34),  # tree
    9:   (124, 185,  93),  # grass
    13:  (139, 119, 101),  # earth / ground
    16:  (128, 128, 128),  # mountain
    17:  ( 60, 179, 113),  # plant / shrub
    21:  ( 30, 100, 180),  # water
    26:  ( 30, 100, 180),  # sea
    29:  (144, 238, 144),  # field
    34:  (105, 105, 105),  # rock / stone
    46:  (210, 180, 140),  # sand / beach
    60:  ( 30, 100, 180),  # river
    68:  (107, 142,  35),  # hill
    72:  ( 34, 139,  34),  # palm tree
    94:  (139, 119, 101),  # land / soil
    113: ( 30, 100, 180),  # waterfall
    128: ( 30, 100, 180),  # lake
    2:   (135, 206, 235),  # sky (fallback)
}
_ADE20K_DEFAULT_COLOR: tuple[int, int, int] = (160, 130, 100)  # unclassified → earth


# ---------------------------------------------------------------------------
# Pascal VOC-21 class index (0-based) → base elevation [0, 1]
# ---------------------------------------------------------------------------
_VOC_ELEV: dict[int, float] = {
    0:  0.25,   # background → generic lowland
    16: 0.38,   # pottedplant → treat as vegetation
}

_VOC_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (160, 130, 100),  # background → earth
    16: ( 60, 179, 113),  # potted plant → green
}
_VOC_DEFAULT_COLOR: tuple[int, int, int] = (160, 130, 100)


# ---------------------------------------------------------------------------
# SegFormer backend (HuggingFace Transformers)
# ---------------------------------------------------------------------------

_SF_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
_sf_cache: dict = {}


def _load_segformer(device: str):
    if "model" not in _sf_cache:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        print(f"  Loading SegFormer ({_SF_MODEL_ID}) — first run downloads ~200 MB...")
        proc  = SegformerImageProcessor.from_pretrained(_SF_MODEL_ID)
        model = SegformerForSemanticSegmentation.from_pretrained(_SF_MODEL_ID)
        model.eval().to(device)
        _sf_cache.update(proc=proc, model=model, device=device)
    return _sf_cache["proc"], _sf_cache["model"]


def _run_segformer(image_rgb: np.ndarray, device: str) -> np.ndarray:
    """Return ADE20K segmentation map (H, W) int64."""
    import torch
    import torch.nn.functional as F

    H, W = image_rgb.shape[:2]
    pil = Image.fromarray((image_rgb * 255).astype(np.uint8))
    proc, model = _load_segformer(device)

    inputs = proc(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits          # (1, 150, H/4, W/4)

    up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# DeepLabV3 backend (torchvision)
# ---------------------------------------------------------------------------

_dl_cache: dict = {}


def _load_deeplabv3(device: str):
    if "model" not in _dl_cache:
        from torchvision.models.segmentation import (
            deeplabv3_resnet101,
            DeepLabV3_ResNet101_Weights,
        )
        print("  Loading DeepLabV3-ResNet101 — first run downloads ~240 MB...")
        weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        model   = deeplabv3_resnet101(weights=weights)
        model.eval().to(device)
        _dl_cache.update(model=model, transform=weights.transforms(), device=device)
    return _dl_cache["model"], _dl_cache["transform"]


def _run_deeplabv3(image_rgb: np.ndarray, device: str) -> np.ndarray:
    """Return Pascal-VOC segmentation map (H, W) int64."""
    import torch
    import torch.nn.functional as F

    H, W = image_rgb.shape[:2]
    pil = Image.fromarray((image_rgb * 255).astype(np.uint8))
    model, transform = _load_deeplabv3(device)

    inp = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)["out"]                  # (1, 21, H', W')

    up = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _classes_to_heights(
    seg_map: np.ndarray,
    image_rgb: np.ndarray,
    elev_map: dict[int, float],
    sky_cls: int | None,
) -> np.ndarray:
    """Map a segmentation class map to a float64 height array with spectral fallback."""
    H, W = image_rgb.shape[:2]
    height = np.full((H, W), np.nan, dtype=np.float64)

    for cls_id, elev in elev_map.items():
        height[seg_map == cls_id] = elev

    if sky_cls is not None:
        height[seg_map == sky_cls] = np.nan   # sky → unknown

    unknown = np.isnan(height)
    if unknown.any():
        from .classifier import classify as _spectral
        height[unknown] = _spectral(image_rgb).astype(np.float64)[unknown]

    return height


def _classes_to_texture(
    seg_map: np.ndarray,
    color_map: dict[int, tuple[int, int, int]],
    default_color: tuple[int, int, int] = (160, 130, 100),
) -> np.ndarray:
    """Map segmentation class IDs to float32 RGB texture (values in [0, 1])."""
    H, W = seg_map.shape
    tex = np.full((H, W, 3), default_color, dtype=np.float64)
    for cls_id, color in color_map.items():
        tex[seg_map == cls_id] = color
    return (tex / 255.0).astype(np.float32)


def _tile_weight(
    tile_h: int, tile_w: int,
    y0: int, x0: int,
    H: int, W: int,
    overlap: int,
) -> np.ndarray:
    """
    Linear-ramp blending weight for one tile.

    At borders shared with an adjacent tile the weight ramps from 0 → 1
    over `overlap` pixels so that overlapping predictions blend smoothly.
    At actual image edges (no neighbour) the weight stays at 1.0.
    """
    wy = np.ones(tile_h, dtype=np.float64)
    wx = np.ones(tile_w, dtype=np.float64)
    r  = max(1, overlap)

    if y0 > 0:             wy[:r]  = np.linspace(0.0, 1.0, r)   # top shared border
    if y0 + tile_h < H:   wy[-r:] = np.linspace(1.0, 0.0, r)   # bottom shared border
    if x0 > 0:             wx[:r]  = np.linspace(0.0, 1.0, r)   # left shared border
    if x0 + tile_w < W:   wx[-r:] = np.linspace(1.0, 0.0, r)   # right shared border

    return np.outer(wy, wx)


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def _segment_tiled(
    image_rgb: np.ndarray,
    run_fn,
    elev_map: dict[int, float],
    sky_cls: int | None,
    tile_size: int,
    overlap: int,
    color_map: dict[int, tuple[int, int, int]] | None = None,
    default_color: tuple[int, int, int] = (160, 130, 100),
    veg_classes: frozenset[int] | None = None,
) -> np.ndarray | tuple:
    """
    Process a large image tile-by-tile and blend results with linear-ramp weights.

    Tiles are placed on a regular grid with `overlap` pixels between neighbours.
    The final heightmap is the weighted average of all tiles at each pixel.
    When `color_map` is provided, also returns a blended RGB texture.
    When `veg_classes` is provided, also returns a boolean vegetation mask.
    """
    H, W = image_rgb.shape[:2]
    stride = tile_size - overlap

    # Build tile top-left coordinates; ensure the last tile reaches the edge.
    def positions(length: int) -> list[int]:
        max_start = max(0, length - tile_size)
        pts = list(range(0, max_start, stride))
        pts.append(max_start)
        return sorted(set(pts))

    y_starts = positions(H)
    x_starts = positions(W)
    n_tiles  = len(y_starts) * len(x_starts)
    print(f"    Tiling {len(x_starts)} x {len(y_starts)} = {n_tiles} tiles  "
          f"(tile={tile_size}px  overlap={overlap}px)")

    height_acc  = np.zeros((H, W), dtype=np.float64)
    weight_acc  = np.zeros((H, W), dtype=np.float64)
    texture_acc = np.zeros((H, W, 3), dtype=np.float64) if color_map is not None else None
    veg_acc     = np.zeros((H, W), dtype=np.float64) if veg_classes is not None else None

    for idx, (y0, x0) in enumerate(
        (y, x) for y in y_starts for x in x_starts
    ):
        y1 = min(y0 + tile_size, H)
        x1 = min(x0 + tile_size, W)
        th, tw = y1 - y0, x1 - x0

        tile    = image_rgb[y0:y1, x0:x1]
        seg_map = run_fn(tile)
        tile_h  = _classes_to_heights(seg_map, tile, elev_map, sky_cls)
        weight  = _tile_weight(th, tw, y0, x0, H, W, overlap)

        height_acc[y0:y1, x0:x1] += tile_h * weight
        weight_acc[y0:y1, x0:x1] += weight

        if texture_acc is not None:
            tile_tex = _classes_to_texture(seg_map, color_map, default_color)
            texture_acc[y0:y1, x0:x1] += tile_tex * weight[:, :, np.newaxis]

        if veg_acc is not None:
            tile_veg = np.isin(seg_map, list(veg_classes)).astype(np.float64)
            veg_acc[y0:y1, x0:x1] += tile_veg * weight

        print(f"    [{idx + 1}/{n_tiles}]  y={y0}:{y1}  x={x0}:{x1}")

    valid  = weight_acc > 1e-9
    result = np.zeros((H, W), dtype=np.float32)
    result[valid] = (height_acc[valid] / weight_acc[valid]).astype(np.float32)

    extras = []
    if texture_acc is not None:
        texture = np.zeros((H, W, 3), dtype=np.float32)
        texture[valid] = (texture_acc[valid] / weight_acc[valid, np.newaxis]).astype(np.float32)
        extras.append(texture)
    if veg_acc is not None:
        veg_score = np.zeros((H, W), dtype=np.float32)
        veg_score[valid] = (veg_acc[valid] / weight_acc[valid]).astype(np.float32)
        extras.append(veg_score > 0.4)  # majority-vote threshold

    return (result, *extras) if extras else result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to the best available device string."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Check your CUDA / driver installation."
        )
    return device


def cuda_info() -> str:
    """Return a human-readable CUDA device description, or 'CPU' if unavailable."""
    import torch
    if not torch.cuda.is_available():
        return "CPU"
    idx      = torch.cuda.current_device()
    name     = torch.cuda.get_device_name(idx)
    total_mb = torch.cuda.get_device_properties(idx).total_memory // (1024 ** 2)
    return f"CUDA:{idx}  {name}  ({total_mb} MB)"


def segment(
    image_rgb: np.ndarray,
    backbone: str = "segformer",
    device: str = "auto",
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    return_texture: bool = False,
    return_veg_mask: bool = False,
) -> np.ndarray | tuple:
    """
    Estimate terrain elevation from a satellite image via semantic segmentation.

    For images larger than TILE_THRESHOLD px on any side, the image is
    automatically split into overlapping tiles of `tile_size` px with `overlap`
    px blending margin. Smaller images are processed in one pass.

    Args:
        image_rgb:       (H, W, 3) float32 array, values in [0, 1].
        backbone:        'segformer' (ADE20K, default) or 'deeplabv3' (Pascal VOC).
        device:          'auto', 'cpu', 'cuda', or 'mps'.
        tile_size:       Side length of each tile in pixels (default: 512).
        overlap:         Overlap between adjacent tiles in pixels (default: 64).
        return_texture:  When True, also return an (H, W, 3) float32 RGB texture.
        return_veg_mask: When True, also return a (H, W) bool vegetation mask
                         based on tree/plant/shrub class indices directly from
                         the segmentation model (no color approximation).

    Returns:
        raw_heights (H, W) float32.
        Followed by texture_rgb (H, W, 3) if return_texture.
        Followed by veg_mask (H, W) bool if return_veg_mask.
    """
    device = resolve_device(device)

    if backbone == "segformer":
        run_fn        = lambda img: _run_segformer(img, device)
        elev_map      = _ADE20K_ELEV
        sky_cls       = _ADE20K_SKY_CLS
        color_map     = _ADE20K_COLORS
        default_color = _ADE20K_DEFAULT_COLOR
        veg_classes   = _ADE20K_VEG_CLASSES
    elif backbone == "deeplabv3":
        run_fn        = lambda img: _run_deeplabv3(img, device)
        elev_map      = _VOC_ELEV
        sky_cls       = None
        color_map     = _VOC_COLORS
        default_color = _VOC_DEFAULT_COLOR
        veg_classes   = None  # VOC doesn't have reliable vegetation classes
    else:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose 'segformer' or 'deeplabv3'.")

    cm  = color_map if return_texture else None
    vc  = veg_classes if return_veg_mask else None
    H, W = image_rgb.shape[:2]

    if max(H, W) > TILE_THRESHOLD:
        return _segment_tiled(image_rgb, run_fn, elev_map, sky_cls,
                              tile_size, overlap, cm, default_color, vc)

    # Image fits in one pass.
    seg_map = run_fn(image_rgb)
    heights = _classes_to_heights(seg_map, image_rgb, elev_map, sky_cls).astype(np.float32)

    extras = []
    if return_texture:
        extras.append(_classes_to_texture(seg_map, color_map, default_color))
    if return_veg_mask and vc is not None:
        extras.append(np.isin(seg_map, list(vc)))

    return (heights, *extras) if extras else heights
