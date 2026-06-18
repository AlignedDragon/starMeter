"""Measure nanostar branch lengths by removing the star center, then linearizing
the remaining arm skeletons -- the method in final_processing_layer.py on `main`.

Per star:
  1. skeletonize the mask.
  2. line_eraser: keep only pixels with >=3 skeleton neighbours (the central hub
     where arms meet) and SUBTRACT them from the skeleton. What remains is the set
     of arm segments with the center cut out.
  3. each arm becomes its own contour; approxPolyDP LINEARIZES it into straight
     segments and arcLength gives the branch length.

Lines are drawn per star in distinct colors. Branch length is reported as
arcLength/2 (the contour wraps the thin arm, so its perimeter is ~2x the arm).

Crop matches src/dataset.py::crop_scale_bar so polygons line up after scaling.

Examples:
    python utils/branch_length.py --all --csv lengths.csv
    python utils/branch_length.py --image 5.jpg --labels
    python utils/branch_length.py --coco data/annotations/predictions_overlap.json --all
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from branch_lines import color_for, instance_mask   # reuse helpers

REPO = Path(__file__).resolve().parents[1]


def arm_segments(mask: np.ndarray, min_branch: float, core_radius: int):
    """Subtract the star center from its skeleton, then linearize each arm.

    Two skeleton operations (the idea from final_processing_layer.py on `main`):
      * arms   = the COMPLETE skeleton of the mask (thin 1-px lines for every arm).
      * center = the mask shrunk until only the core blob is left (morphological
                 opening with a `core_radius` disk removes the thin arms but keeps
                 the thick hub).
    arms - center splits the skeleton into one component per arm; each is fit to a
    straight line clipped to its extent. Returns [((x1,y1),(x2,y2), length_px), ...]
    where length is the arm centerline length (core edge -> tip)."""
    full = skeletonize(mask > 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_radius + 1, 2 * core_radius + 1))
    core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k) > 0
    core = binary_dilation(core, iterations=1)
    arms = (full & ~core).astype(np.uint8)
    n, labels = cv2.connectedComponents(arms, connectivity=8)
    out = []
    for c in range(1, n):
        ys, xs = np.where(labels == c)
        length = float(len(xs))                             # centerline length
        if length < min_branch:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        proj = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        lo, hi = float(proj.min()), float(proj.max())
        out.append(((x0 + lo * vx, y0 + lo * vy), (x0 + hi * vx, y0 + hi * vy), length))
    return out


def render(img_path: Path, anns: list[dict], size: int, frac: float, thickness: int,
           min_area: int, dim: float, min_branch: float, core_radius: int, labels: bool):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    side = min(int(H * (1 - frac)), W)
    sc = size / side
    base = np.array(img.crop((0, 0, side, side)).resize((size, size), Image.BILINEAR))
    if dim < 1.0:
        base = (base * dim).astype(np.uint8)

    rows = []
    for i, ann in enumerate(anns):
        m = instance_mask(ann.get("segmentation", []), size, sc)
        if int(m.sum()) < min_area:
            continue
        col = color_for(i)
        for (x1, y1), (x2, y2), length in arm_segments(m, min_branch, core_radius):
            cv2.line(base, (round(x1), round(y1)), (round(x2), round(y2)), col, thickness)
            rows.append((i, round(length, 1)))
            if labels:
                cv2.putText(base, str(round(length)), (round(x2), round(y2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return Image.fromarray(base), rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json")
    p.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    p.add_argument("--out", type=Path, default=REPO / "data/branch_length")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--frac", type=float, default=0.10)
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--min-area", type=int, default=16)
    p.add_argument("--dim", type=float, default=0.85)
    p.add_argument("--min-branch", type=float, default=12, help="drop arms shorter than this (px)")
    p.add_argument("--core-radius", type=int, default=12,
                   help="opening radius to isolate the star center (larger = bigger hub removed)")
    p.add_argument("--labels", action="store_true", help="print each branch length")
    p.add_argument("--csv", type=Path, help="also write per-branch lengths to this CSV")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--image", type=str)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    coco = json.loads(args.coco.read_text())
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    images = [im for im in coco["images"] if im["id"] in by_img]
    if args.image:
        images = [im for im in coco["images"] if im["file_name"] == args.image]
    elif not args.all:
        images = images[: args.num]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_rows, n = [], 0
    for im in images:
        src = args.images_dir / im["file_name"]
        if not src.exists():
            continue
        out, rows = render(src, by_img.get(im["id"], []), args.size, args.frac, args.thickness,
                           args.min_area, args.dim, args.min_branch, args.core_radius, args.labels)
        out.save(args.out / im["file_name"], quality=88)
        for star_i, length in rows:
            csv_rows.append((im["file_name"], star_i, length))
        n += 1

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["file_name", "star_index", "branch_length_px"])
            w.writerows(csv_rows)
        print(f"wrote {len(csv_rows)} branch measurements -> {args.csv}")
    print(f"wrote {n} branch-length overlays -> {args.out}")


if __name__ == "__main__":
    main()
