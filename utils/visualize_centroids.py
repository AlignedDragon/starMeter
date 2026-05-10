"""Render polygon outlines plus distance-transform core points to data/viz.

Drops one overlay per sampled image into data/viz/centroids/ so you can sanity
check that the core points actually sit on each nanostar's dense body (rather
than the bbox center, which drifts onto thin spikes).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]


def _polygons_to_mask(polygons, h: int, w: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in polygons:
        if len(poly) >= 6:
            d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
    return np.asarray(m, dtype=np.uint8)


def core_point(polygons, bbox: list[float]) -> tuple[float, float]:
    """Pole of inaccessibility via distance transform."""
    x, y, w, h = bbox
    pad = 4
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    cw = int(w) + 2 * pad
    ch = int(h) + 2 * pad
    shifted = [[c - (x0 if i % 2 == 0 else y0) for i, c in enumerate(p)] for p in polygons]
    mask = _polygons_to_mask(shifted, ch, cw)
    if mask.sum() == 0:
        return x + w / 2, y + h / 2
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    iy, ix = np.unravel_index(int(dt.argmax()), dt.shape)
    return float(ix + x0), float(iy + y0)


def render(img_path: Path, anns: list[dict], out_path: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 22
    for a in anns:
        polys = a.get("segmentation", [])
        for p in polys:
            if len(p) >= 6:
                draw.line(
                    [(p[i], p[i + 1]) for i in range(0, len(p), 2)] + [(p[0], p[1])],
                    fill=(255, 220, 0), width=4,
                )
        cx, cy = core_point(polys, a["bbox"])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json")
    ap.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    ap.add_argument("--out", type=Path, default=REPO / "data/viz/centroids")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with args.coco.open() as f:
        coco = json.load(f)
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        if a.get("segmentation"):
            anns_by_img.setdefault(a["image_id"], []).append(a)

    random.seed(args.seed)
    images = [im for im in coco["images"] if im["id"] in anns_by_img]
    targets = random.sample(images, min(args.num, len(images)))
    for info in targets:
        out_path = args.out / info["file_name"]
        render(args.images_dir / info["file_name"], anns_by_img[info["id"]], out_path)
        print(f"{info['file_name']}: {len(anns_by_img[info['id']])} instances -> {out_path}")


if __name__ == "__main__":
    main()
