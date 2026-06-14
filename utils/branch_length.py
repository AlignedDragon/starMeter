"""Draw nanostar branch lines from the star CENTER to each arm tip.

Unlike branch_lines.py (which fits a line along each arm), here every line starts
at the star's center so its length measures the branch length (center -> tip).

Per star:
  1. center  = pole of inaccessibility (distance-transform maximum of the mask) =
               the dense hub of the star, matching src/dataset.py::core_point.
  2. tips    = skeleton endpoints (1-neighbour pixels), keeping only those at least
               `--min-branch` px from the center (drops core spurs / noise).
  3. a straight line center -> tip is drawn per branch; its length IS the branch
     length. Each star gets its own color.

Crop matches src/dataset.py::crop_scale_bar so polygon coords line up after scaling.

Examples:
    python utils/branch_length.py --all                          # GT train.json
    python utils/branch_length.py --image 5.jpg --labels         # show lengths
    python utils/branch_length.py --coco preds.json --all --csv out.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from branch_lines import _NB8, color_for, instance_mask   # reuse helpers

REPO = Path(__file__).resolve().parents[1]


def star_center(mask: np.ndarray) -> tuple[float, float]:
    """Pole of inaccessibility: the interior pixel farthest from any edge."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, maxloc = cv2.minMaxLoc(dt)        # maxloc = (x, y)
    return float(maxloc[0]), float(maxloc[1])


def branch_tips(mask: np.ndarray, center: tuple[float, float],
                min_branch: float, merge: float = 8.0) -> list[tuple[float, float, float]]:
    """Skeleton endpoints kept >= min_branch from center; near-duplicate tips
    merged (keep the farther). Returns [(x, y, length_from_center), ...]."""
    skel = skeletonize(mask > 0).astype(np.uint8)
    nb = convolve(skel, _NB8, mode="constant", cval=0)
    ys, xs = np.where((skel > 0) & (nb == 1))
    cx, cy = center
    cand = []
    for x, y in zip(xs, ys):
        d = float(np.hypot(x - cx, y - cy))
        if d >= min_branch:
            cand.append((float(x), float(y), d))
    cand.sort(key=lambda t: -t[2])             # farthest first
    kept: list[tuple[float, float, float]] = []
    for x, y, d in cand:
        if all(np.hypot(x - kx, y - ky) > merge for kx, ky, _ in kept):
            kept.append((x, y, d))
    return kept


def render(img_path: Path, anns: list[dict], size: int, frac: float, thickness: int,
           min_area: int, dim: float, min_branch: float, labels: bool):
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
        cx, cy = star_center(m)
        c = (round(cx), round(cy))
        for x, y, length in branch_tips(m, (cx, cy), min_branch):
            cv2.line(base, c, (round(x), round(y)), col, thickness)
            if labels:
                cv2.putText(base, str(round(length)), (round(x), round(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            rows.append((i, round(length, 1)))           # branch length in px (display scale)
        cv2.circle(base, c, max(2, thickness + 1), col, -1)
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
    p.add_argument("--min-branch", type=float, default=18,
                   help="ignore tips closer than this to the center (px, display scale)")
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
    csv_rows = []
    n = 0
    for im in images:
        src = args.images_dir / im["file_name"]
        if not src.exists():
            continue
        out, rows = render(src, by_img.get(im["id"], []), args.size, args.frac,
                           args.thickness, args.min_area, args.dim, args.min_branch, args.labels)
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
    print(f"wrote {n} center-to-tip overlays -> {args.out}")


if __name__ == "__main__":
    main()
