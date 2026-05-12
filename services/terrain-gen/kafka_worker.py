"""
Kafka worker for terrain_gen.

Consumes terrain.requests, runs the generation pipeline,
uploads results to MinIO, publishes to terrain.results.

Usage:
  python kafka_worker.py
  python kafka_worker.py --brokers localhost:9093 --minio-endpoint localhost:9000

Environment variables (or pass via CLI flags):
  KAFKA_BROKERS          default: localhost:9093
  KAFKA_REQUESTS_TOPIC   default: terrain.requests
  KAFKA_RESULTS_TOPIC    default: terrain.results
  KAFKA_GROUP_ID         default: terrain-gen-worker
  MINIO_ENDPOINT         default: localhost:9000
  MINIO_ACCESS_KEY       default: minio_user
  MINIO_SECRET_KEY       default: superStrongPassword
  MINIO_BUCKET           default: diploma
  MINIO_USE_SSL          default: false
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError
from minio import Minio
from PIL import Image

# terrain-gen pipeline lives in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from main import process


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def _make_minio(endpoint: str, access_key: str, secret_key: str, use_ssl: bool) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=use_ssl)


def _ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    import json
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket}/*"],
        }],
    })
    client.set_bucket_policy(bucket, policy)

    try:
        from minio.corsconfig import CORSConfig, CORSRule
        cors = CORSConfig([
            CORSRule(
                allowed_origins=["*"],
                allowed_methods=["GET", "HEAD"],
                allowed_headers=["*"],
            )
        ])
        client.set_bucket_cors(bucket, cors)
    except Exception as e:
        print(f"[worker] CORS setup skipped: {e}")


def _upload(client: Minio, bucket: str, local_path: Path, object_name: str, public_url: str) -> str:
    client.fput_object(bucket, object_name, str(local_path))
    return f"{public_url.rstrip('/')}/{bucket}/{object_name}"


# ---------------------------------------------------------------------------
# Tree placement
# ---------------------------------------------------------------------------

def _generate_trees(
    image_rgb: np.ndarray,
    heightmap: np.ndarray,
    scale_z: float,
    y_up: bool = False,
    seg_veg_mask: "np.ndarray | None" = None,
) -> dict:
    """
    Generates tree positions from semantic segmentation class indices (primary)
    with RGB ExG as fallback/supplement. seg_veg_mask is a (H, W) bool array
    derived directly from ADE20K class IDs (tree=4, plant=17, etc.) — not from
    color approximation.
    y_up is stored in JSON so the viewer knows the OBJ axis convention.
    """
    from scipy.ndimage import binary_erosion, binary_dilation

    sea_level = 0.08

    # Resize heightmap to image resolution if needed
    if heightmap.shape == image_rgb.shape[:2]:
        hmap_r = heightmap
    else:
        from PIL import Image as _Img
        hmap_r = np.array(
            _Img.fromarray((heightmap * 255).astype(np.uint8)).resize(
                (image_rgb.shape[1], image_rgb.shape[0]), _Img.BILINEAR
            ), dtype=np.float32
        ) / 255.0

    H, W = image_rgb.shape[:2]

    # --- RGB ExG: direct pixel-level green detection ---
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    exg = 2.0 * G - R - B
    ngrdi = (G - R) / (G + R + 1e-8)
    brightness = (R + G + B) / 3.0
    rgb_veg = (exg > 0.04) & (ngrdi > 0.005) & (G > 0.10) & (brightness > 0.07) & (brightness < 0.88)

    if seg_veg_mask is not None:
        # Resize segmentation mask to image resolution if needed
        if seg_veg_mask.shape[:2] != (H, W):
            from PIL import Image as _Img
            seg_pil = _Img.fromarray(seg_veg_mask.astype(np.uint8) * 255)
            seg_veg_mask = np.array(
                seg_pil.resize((W, H), _Img.NEAREST), dtype=bool
            )
        # Union: tree wherever EITHER the neural model OR RGB ExG detects vegetation
        veg_mask = seg_veg_mask | rgb_veg
    else:
        veg_mask = rgb_veg

    # Exclude water / sea-level areas
    veg_mask = veg_mask & (hmap_r > sea_level + 0.04)

    # Light morphological cleanup
    veg_mask = binary_erosion(veg_mask, iterations=1)
    veg_mask = binary_dilation(veg_mask, iterations=1)

    rng = np.random.default_rng(42)

    # Grid sampling — one tree per cell, random position within vegetation pixels
    cell_size = 3
    trees = []
    for cy in range(0, H, cell_size):
        for cx in range(0, W, cell_size):
            cell = veg_mask[cy:cy + cell_size, cx:cx + cell_size]
            ys, xs = np.where(cell)
            if len(xs) == 0:
                continue
            idx = rng.integers(len(xs))
            px, py = cx + xs[idx], cy + ys[idx]

            nx = px / max(W - 1, 1)
            nz = py / max(H - 1, 1)

            hx = min(int(nx * (hmap_r.shape[1] - 1)), hmap_r.shape[1] - 1)
            hz = min(int(nz * (hmap_r.shape[0] - 1)), hmap_r.shape[0] - 1)
            ny = float(hmap_r[hz, hx])

            trees.append({
                "x": round(nx, 4),
                "z": round(nz, 4),
                "y": round(ny, 4),
                "s": round(float(rng.uniform(0.8, 1.4)), 2),
            })

    print(f"[worker] trees generated: {len(trees)} positions  "
          f"({'seg+rgb' if seg_veg_mask is not None else 'rgb'} mode)")
    return {"count": len(trees), "scale_z": scale_z, "y_up": y_up, "trees": trees}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _handle(msg: dict, minio: Minio, bucket: str, public_url: str) -> dict:
    job_id      = msg["job_id"]
    image_b64   = msg["image_base64"]
    scale_z     = float(msg.get("scale_z", 0.3))
    y_up        = bool(msg.get("y_up", False))
    texture_mode = msg.get("texture_mode", "photo")

    image_bytes = base64.b64decode(image_b64)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img_path = tmp / "input.png"
        img_path.write_bytes(image_bytes)

        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size
        print(f"[worker] job={job_id} received image {orig_w}x{orig_h} px  ({len(image_bytes):,} bytes)")

        # Upscale tiny images — anything smaller than 256px won't produce a useful mesh
        MIN_DIM = 256
        if max(orig_w, orig_h) < MIN_DIM:
            scale = MIN_DIM / max(orig_w, orig_h)
            new_w, new_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            print(f"[worker] job={job_id} upscaled {orig_w}x{orig_h} → {new_w}x{new_h}")

        image_rgb = np.array(pil_img, dtype=np.float32) / 255.0
        out_dir = tmp / "output"

        import satellite_terrain as st

        process(
            image_rgb=image_rgb,
            output_dir=out_dir,
            stem=job_id,
            smooth_sigma=3.0,
            sea_level=0.08,
            export_obj=True,
            bbox=None,
            dem_grid=64,
            method="segment",
            backbone="segformer",
            device="auto",
            tile_size=512,
            overlap=64,
            scale_z=scale_z,
            y_up=y_up,
            texture_mode=texture_mode,
        )

        # Rebuild heightmap + get vegetation mask for tree placement
        # return_veg_mask uses ADE20K class indices directly (tree=4, plant=17, etc.)
        # — no color approximation, same inference pass as heightmap generation
        import satellite_terrain as st
        raw, seg_veg_mask = st.analyze_neural(image_rgb, backbone="segformer", device="auto",
                                              tile_size=512, overlap=64, return_veg_mask=True)
        heightmap = st.build(raw, smooth_sigma=3.0, sea_level=0.08)

        trees_data = _generate_trees(image_rgb, heightmap, scale_z, y_up, seg_veg_mask=seg_veg_mask)
        trees_path = out_dir / f"{job_id}_trees.json"
        trees_path.write_text(json.dumps(trees_data, separators=(",", ":")))

        prefix = f"terrain/{job_id}"

        model_path = out_dir / f"{job_id}_terrain.obj"
        model_url = _upload(minio, bucket, model_path, f"{prefix}/terrain.obj", public_url)

        mtl_path = out_dir / f"{job_id}_terrain.mtl"
        if mtl_path.exists():
            _upload(minio, bucket, mtl_path, f"{prefix}/terrain.mtl", public_url)

        for candidate in [
            out_dir / f"{job_id}_satellite.png",
            out_dir / f"{job_id}_classified.png",
            out_dir / f"{job_id}_terrain.png",
        ]:
            if candidate.exists():
                texture_url = _upload(minio, bucket, candidate, f"{prefix}/texture.png", public_url)
                break
        else:
            texture_url = ""

        trees_url = _upload(minio, bucket, trees_path, f"{prefix}/trees.json", public_url)

    return {
        "job_id":      job_id,
        "status":      "done",
        "model_url":   model_url,
        "texture_url": texture_url,
        "trees_url":   trees_url,
        "error":       "",
    }


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def run(cfg: argparse.Namespace) -> None:
    minio = _make_minio(cfg.minio_endpoint, cfg.minio_access_key, cfg.minio_secret_key, cfg.minio_ssl)
    _ensure_bucket(minio, cfg.bucket)
    public_url = cfg.minio_public_url

    consumer = Consumer({
        "bootstrap.servers": cfg.brokers,
        "group.id": cfg.group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "fetch.max.bytes": 20971520,
        "message.max.bytes": 20971520,
    })
    consumer.subscribe([cfg.requests_topic])

    producer = Producer({
        "bootstrap.servers": cfg.brokers,
    })

    print(f"[worker] Listening on {cfg.brokers} → {cfg.requests_topic}")
    print(f"[worker] Results  → {cfg.results_topic}")
    print(f"[worker] MinIO    → {cfg.minio_endpoint}/{cfg.bucket}")

    try:
        while True:
            kafka_msg = consumer.poll(timeout=1.0)
            if kafka_msg is None:
                continue
            if kafka_msg.error():
                if kafka_msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"[worker] Kafka error: {kafka_msg.error()}")
                continue

            msg = json.loads(kafka_msg.value().decode("utf-8"))
            job_id = msg.get("job_id", "?")
            print(f"[worker] job={job_id} received")

            try:
                result = _handle(msg, minio, cfg.bucket, public_url)
                print(f"[worker] job={job_id} done")
            except Exception as exc:
                traceback.print_exc()
                result = {
                    "job_id":      job_id,
                    "status":      "failed",
                    "model_url":   "",
                    "texture_url": "",
                    "error":       str(exc),
                }
                print(f"[worker] job={job_id} failed: {exc}")

            producer.produce(cfg.results_topic, value=json.dumps(result).encode("utf-8"))
            producer.flush()
    finally:
        consumer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def main() -> None:
    p = argparse.ArgumentParser(description="Terrain-gen Kafka worker")
    p.add_argument("--brokers",          default=_env("KAFKA_BROKERS",        "localhost:9093"))
    p.add_argument("--requests-topic",   default=_env("KAFKA_REQUESTS_TOPIC", "terrain.requests"))
    p.add_argument("--results-topic",    default=_env("KAFKA_RESULTS_TOPIC",  "terrain.results"))
    p.add_argument("--group-id",         default=_env("KAFKA_GROUP_ID",       "terrain-gen-worker"))
    p.add_argument("--minio-endpoint",   default=_env("MINIO_ENDPOINT",       "localhost:9000"))
    p.add_argument("--minio-access-key", default=_env("MINIO_ACCESS_KEY",     "minio_user"))
    p.add_argument("--minio-secret-key", default=_env("MINIO_SECRET_KEY",     "superStrongPassword"))
    p.add_argument("--minio-ssl",        action="store_true",
                   default=_env("MINIO_USE_SSL", "false").lower() == "true")
    p.add_argument("--bucket",           default=_env("MINIO_BUCKET",         "diploma"))
    p.add_argument("--minio-public-url", default=_env("MINIO_PUBLIC_URL",     "http://localhost:9000"))
    cfg = p.parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
