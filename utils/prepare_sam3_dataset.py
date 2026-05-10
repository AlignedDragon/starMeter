"""Convert the two VIA project JSONs into a SAM3-ready COCO-style dataset.

SAM3's training pipeline ingests COCO instance-segmentation JSON (the same
schema SAM2 used). We emit:

    data/annotations/train.json   -- 300 annotated images (one nanostar category)
    data/annotations/eval.json    -- remaining ~100 unannotated images (images only)

Polygons come from VIA's `xy` field which is laid out as
``[shape_id, x1, y1, x2, y2, ...]`` where shape_id == 7 means polygon.
Anything else (rect, ellipse, circle) is converted to a polygon approximation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
VIA_JSONS = (
    DATA_DIR / "first_half_quasi_ready.json",
    DATA_DIR / "second_half_ready.json",
)
CATEGORY = {"id": 1, "name": "nanostar", "supercategory": "particle"}


def via_xy_to_polygons(xy: list[float]) -> list[list[float]]:
    """Return a list of flat [x1,y1,x2,y2,...] polygons for a VIA region."""
    if not xy:
        return []
    shape = int(xy[0])
    coords = xy[1:]
    if shape in (7, 4):  # polygon, polyline
        if len(coords) < 6:
            return []
        return [[float(c) for c in coords]]
    if shape == 2 and len(coords) >= 4:  # rect: x,y,w,h
        x, y, w, h = coords[:4]
        return [[x, y, x + w, y, x + w, y + h, x, y + h]]
    if shape == 5 and len(coords) >= 3:  # circle: cx,cy,r
        cx, cy, r = coords[:3]
        n = 32
        return [[
            v
            for i in range(n)
            for v in (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        ]]
    if shape == 3 and len(coords) >= 4:  # ellipse: cx,cy,rx,ry
        cx, cy, rx, ry = coords[:4]
        n = 32
        return [[
            v
            for i in range(n)
            for v in (cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n))
        ]]
    return []


def polygon_bbox_area(poly: list[float]) -> tuple[list[float], float]:
    xs = poly[0::2]
    ys = poly[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bbox = [x0, y0, x1 - x0, y1 - y0]
    # Shoelace
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return bbox, abs(area) / 2.0


def load_via(path: Path) -> tuple[dict[str, str], dict[str, list[list[float]]]]:
    """Returns (vid -> fname, vid -> [region xy lists])."""
    with path.open() as f:
        d = json.load(f)
    vid_to_fname = {fid: rec["fname"] for fid, rec in d["file"].items()}
    regions: dict[str, list[list[float]]] = {}
    for entry in d["metadata"].values():
        regions.setdefault(entry["vid"], []).append(entry["xy"])
    return vid_to_fname, regions


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)


def build_coco(
    image_records: Iterable[tuple[str, list[list[float]]]],
    images_dir: Path,
    with_annotations: bool,
) -> dict:
    coco = {"images": [], "annotations": [], "categories": [CATEGORY]}
    ann_id = 1
    for img_id, (fname, regions) in enumerate(image_records, start=1):
        path = images_dir / fname
        w, h = image_size(path)
        coco["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })
        if not with_annotations:
            continue
        for xy in regions:
            for poly in via_xy_to_polygons(xy):
                bbox, area = polygon_bbox_area(poly)
                if area <= 0:
                    continue
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORY["id"],
                    "segmentation": [poly],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1
    return coco


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "annotations")
    args = p.parse_args()

    if not args.images_dir.is_dir():
        raise SystemExit(f"images dir not found: {args.images_dir} (run merge_images.py first)")

    annotated: dict[str, list[list[float]]] = {}  # fname -> regions
    all_fnames: set[str] = set()

    for j in VIA_JSONS:
        vid_to_fname, regions = load_via(j)
        for fid, fname in vid_to_fname.items():
            all_fnames.add(fname)
            if fid in regions:
                annotated[fname] = regions[fid]

    # Cross-check against actual files on disk
    on_disk = {p.name for p in args.images_dir.glob("*.jpg")}
    annotated = {k: v for k, v in annotated.items() if k in on_disk}
    missing_disk = all_fnames - on_disk
    if missing_disk:
        print(f"warning: {len(missing_disk)} VIA-listed images not found on disk")

    eval_fnames = sorted(on_disk - set(annotated), key=lambda s: int(s.split(".")[0]))
    train_records = [(f, annotated[f]) for f in sorted(annotated, key=lambda s: int(s.split(".")[0]))]
    eval_records = [(f, []) for f in eval_fnames]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_coco = build_coco(train_records, args.images_dir, with_annotations=True)
    eval_coco = build_coco(eval_records, args.images_dir, with_annotations=False)

    train_path = args.out_dir / "train.json"
    eval_path = args.out_dir / "eval.json"
    with train_path.open("w") as f:
        json.dump(train_coco, f)
    with eval_path.open("w") as f:
        json.dump(eval_coco, f)

    print(
        f"train: {len(train_coco['images'])} images, "
        f"{len(train_coco['annotations'])} annotations -> {train_path}"
    )
    print(f"eval:  {len(eval_coco['images'])} images (no GT) -> {eval_path}")


if __name__ == "__main__":
    main()
