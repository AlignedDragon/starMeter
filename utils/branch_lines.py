"""Draw nanostar branch centerlines (skeletons) from instance segmentations.

For each instance we rasterize its mask, skeletonize it (skimage) to a 1-px medial
axis, and draw that centerline — so the star's arms/branches show as lines. Each
star gets its own color so overlapping stars stay distinguishable.

The crop matches src/dataset.py::crop_scale_bar (top-left bar-free square) so the
polygons (full-image coordinates) line up after scaling.

Examples:
    python utils/branch_lines.py                                   # train.json, 8 samples
    python utils/branch_lines.py --all --out data/branch_lines     # all GT images
    python utils/branch_lines.py --coco preds.json --image 100.jpg
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, label
from skimage.morphology import skeletonize

REPO = Path(__file__).resolve().parents[1]

_NB8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])      # 8-neighbour kernel


def straight_branches(mask: np.ndarray, min_branch: int) -> list[tuple]:
    """Split a star's skeleton at its junctions and fit one straight line per arm.

    Mirrors the idea in final_processing_layer.py on `main`: skeletonize, drop the
    junction/core pixels (those with >=3 skeleton neighbours) so each arm becomes a
    separate connected component, then fit a straight segment to each. Returns a
    list of ((x1, y1), (x2, y2)) endpoints."""
    skel = skeletonize(mask > 0).astype(np.uint8)
    nb = convolve(skel, _NB8, mode="constant", cval=0)
    arms = skel.copy()
    arms[(skel > 0) & (nb >= 3)] = 0                    # cut at junctions
    lab, n = label(arms, structure=np.ones((3, 3)))     # 8-connected components
    segs = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if len(xs) < min_branch:                        # drop short spurs
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        proj = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        lo, hi = float(proj.min()), float(proj.max())
        segs.append(((x0 + lo * vx, y0 + lo * vy), (x0 + hi * vx, y0 + hi * vy)))
    return segs


def color_for(idx: int) -> tuple[int, int, int]:
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.95, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def instance_mask(segmentation: list[list[float]], size: int, sc: float) -> np.ndarray:
    """Rasterize one instance's polygon(s) into a (size, size) uint8 mask."""
    m = np.zeros((size, size), np.uint8)
    for poly in segmentation:
        pts = (np.array(poly, np.float32).reshape(-1, 2) * sc).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def render(img_path: Path, anns: list[dict], size: int, frac: float,
           thickness: int, min_area: int, dim: float, mode: str, min_branch: int) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    side = min(int(H * (1 - frac)), W)                 # same square crop as training
    sc = size / side
    base = np.array(img.crop((0, 0, side, side)).resize((size, size), Image.BILINEAR))
    if dim < 1.0:                                      # optionally darken so lines pop
        base = (base * dim).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i, ann in enumerate(anns):
        m = instance_mask(ann.get("segmentation", []), size, sc)
        if int(m.sum()) < min_area:
            continue
        col = color_for(i)                             # per-star color
        if mode == "straight":
            for (x1, y1), (x2, y2) in straight_branches(m, min_branch):
                cv2.line(base, (round(x1), round(y1)), (round(x2), round(y2)), col, thickness)
        else:                                          # raw skeleton centerline
            skel = skeletonize(m > 0).astype(np.uint8)
            if thickness > 1:
                skel = cv2.dilate(skel, kernel, iterations=thickness - 1)
            base[skel > 0] = col
    return Image.fromarray(base)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json",
                   help="COCO json with instance segmentations (default: GT train.json)")
    p.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    p.add_argument("--out", type=Path, default=REPO / "data/branch_lines")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--frac", type=float, default=0.10)
    p.add_argument("--mode", choices=["straight", "skeleton"], default="straight",
                   help="straight: one fitted line per arm (default); skeleton: raw centerline")
    p.add_argument("--min-branch", type=int, default=14,
                   help="straight mode: drop arm segments shorter than this (px) — prunes spurs")
    p.add_argument("--thickness", type=int, default=2, help="line width in px")
    p.add_argument("--min-area", type=int, default=16, help="skip instances smaller than this")
    p.add_argument("--dim", type=float, default=0.85, help="background brightness (1.0 = none)")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--image", type=str, help="render this filename only")
    p.add_argument("--all", action="store_true", help="render every image")
    args = p.parse_args()

    coco = json.loads(args.coco.read_text())
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    images = [im for im in coco["images"] if im["id"] in by_img]  # GT-labeled images only
    if args.image:
        images = [im for im in coco["images"] if im["file_name"] == args.image]
    elif not args.all:
        images = images[: args.num]

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in images:
        src = args.images_dir / im["file_name"]
        if not src.exists():
            continue
        out = render(src, by_img.get(im["id"], []), args.size, args.frac,
                     args.thickness, args.min_area, args.dim, args.mode, args.min_branch)
        out.save(args.out / im["file_name"], quality=88)
        n += 1
    print(f"wrote {n} branch-line overlays -> {args.out}")


if __name__ == "__main__":
    main()
