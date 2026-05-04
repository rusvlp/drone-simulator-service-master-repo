from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def enhance_texture(
    image_rgb: np.ndarray,
    sharpen: float = 1.5,
    saturation: float = 1.2,
) -> np.ndarray:
    """
    Sharpen and boost saturation of a satellite texture.

    Args:
        image_rgb:  (H, W, 3) float32 array, values in [0, 1].
        sharpen:    Sharpening strength — radius multiplier for UnsharpMask
                    (0 = off, 1.0 = moderate, 2.0 = aggressive).
        saturation: Color saturation multiplier (1.0 = original, 1.2 = slight boost).

    Returns:
        Enhanced (H, W, 3) float32 array, values in [0, 1].
    """
    pil = Image.fromarray((image_rgb * 255).clip(0, 255).astype(np.uint8), mode="RGB")

    if sharpen > 0:
        pil = pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen * 1.5,
            percent=int(80 + sharpen * 60),
            threshold=2,
        ))

    if saturation != 1.0:
        pil = ImageEnhance.Color(pil).enhance(saturation)

    return np.array(pil, dtype=np.float32) / 255.0


def save_texture(image_rgb: np.ndarray, path: Path,
                 sharpen: float = 1.5, saturation: float = 1.2) -> None:
    """Save satellite texture as lossless PNG with sharpening and saturation boost."""
    enhanced = enhance_texture(image_rgb, sharpen=sharpen, saturation=saturation)
    arr = (enhanced * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path), format="PNG")


def save_heightmap(heightmap: np.ndarray, path: Path) -> None:
    """Save heightmap as 8-bit grayscale PNG (0 = sea, 255 = peak)."""
    arr = (heightmap * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(str(path))


def save_colored_terrain(heightmap: np.ndarray, path: Path) -> None:
    """
    Save a false-color terrain map using the matplotlib 'terrain' colormap.

    Color scale (approx):
      0.0  → deep blue  (water)
      0.25 → teal       (shallow / coast)
      0.35 → sandy      (beach / lowland)
      0.50 → green      (vegetation)
      0.75 → brown      (highland / bare rock)
      1.0  → white      (snow / peak)
    """
    import matplotlib.cm as cm

    rgba = cm.terrain(heightmap)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(str(path))


def save_obj(
    heightmap: np.ndarray,
    path: Path,
    texture_name: str | None = None,
    scale_z: float = 0.3,
    y_up: bool = False,
) -> None:
    """
    Export terrain as a Wavefront OBJ triangle mesh.

    The XY plane spans [0, W-1] x [0, H-1].  Z is scaled by `scale_z` relative
    to the longer XY dimension so that terrain relief looks natural at 1:1 view.

    UV coordinates are emitted so that a satellite image or colored terrain PNG
    can be applied directly as a texture in any 3D viewer.

    Args:
        heightmap:    (H, W) float32, values in [0, 1].
        path:         Output .obj file path.
        texture_name: If given, a companion .mtl file is written and referenced.
        scale_z:      Vertical exaggeration factor relative to XY extent.
        y_up:         If True, emit Y-up axes (X/Z ground plane, Y = height)
                      for Unity / Godot / most game engines.
                      If False (default), emit Z-up axes for Blender.
    """
    H, W = heightmap.shape
    z_scale = max(H, W) * scale_z
    path = Path(path)

    mtl_path = path.with_suffix(".mtl")
    if texture_name:
        mtl_path.write_text(
            f"newmtl terrain\n"
            f"map_Kd {texture_name}\n"
        )

    with open(path, "w") as f:
        if texture_name:
            f.write(f"mtllib {mtl_path.name}\n")
            f.write("usemtl terrain\n\n")

        f.write(f"# Terrain mesh  --  {W}x{H} grid\n\n")

        # Vertices  (v x y z)
        for row in range(H):
            for col in range(W):
                h = float(heightmap[row, col]) * z_scale
                if y_up:
                    # Y-up: ground plane is XZ, Y = elevation
                    x = float(col)
                    y = h
                    z = float(H - 1 - row)
                else:
                    # Z-up (Blender default): ground plane is XY, Z = elevation
                    x = float(col)
                    y = float(H - 1 - row)
                    z = h
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")

        f.write("\n")

        # UV coordinates  (vt u v)
        for row in range(H):
            for col in range(W):
                u = col / (W - 1)
                v = 1.0 - row / (H - 1)
                f.write(f"vt {u:.6f} {v:.6f}\n")

        f.write("\n")

        # Faces  (f v/vt v/vt v/vt)
        def vi(r: int, c: int) -> int:
            return r * W + c + 1  # 1-indexed

        for row in range(H - 1):
            for col in range(W - 1):
                tl, tr = vi(row, col),     vi(row,     col + 1)
                bl, br = vi(row + 1, col), vi(row + 1, col + 1)
                # Each quad → two CCW triangles
                f.write(f"f {tl}/{tl} {bl}/{bl} {tr}/{tr}\n")
                f.write(f"f {tr}/{tr} {bl}/{bl} {br}/{br}\n")
