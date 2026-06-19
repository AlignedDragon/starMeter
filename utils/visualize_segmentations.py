"""Render polygon overlays from a SAM3/COCO annotation file for sanity checks.

Tailored for OVERLAPPING nanostars: each instance is alpha-blended on its own
layer (so where two stars cross you see both colors mix, instead of the later
one carving out the earlier) and then a crisp, darker per-instance border is
drawn on top so overlapping boundaries stay readable. Works for ground-truth
COCO (train.json) and for full-frame predictions (src/m2f_pipeline.py output)
alike — both carry polygons in the source image's coordinates.

Examples:
    python utils/visualize_segmentations.py                          # 8 random train images
    python utils/visualize_segmentations.py --image 42.jpg           # one specific image
    python utils/visualize_segmentations.py --num 20 --out vis/      # 20 images, custom out dir
    python utils/visualize_segmentations.py \
        --coco results/predictions_valid_m2f.json --all --score-thresh 0.5
"""
from __future__ import annotations

import argparse
import colorsys
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def color_for(idx: int) -> tuple[int, int, int]:
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def render(img_path: Path, anns: list[dict], size: int | None = None,
           fill_alpha: int = 95, border_w: int = 2) -> Image.Image:
    """Overlay `anns` polygons on the image, blending overlaps. Returns RGB.

    If `size` is given, the image (and polygons) are scaled so the longest edge
    is `size` px — 4096² frames are large, so this keeps rendering cheap.
    """
    img = Image.open(img_path).convert("RGBA")
    W, H = img.size
    sc = 1.0 if size is None else size / max(W, H)
    if sc != 1.0:
        img = img.resize((round(W * sc), round(H * sc)), Image.BILINEAR)
    base = img

    # Pass 1: alpha-blend each instance's fill one at a time so overlaps mix.
    for i, ann in enumerate(anns):
        r, g, b = color_for(i)
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for poly in ann.get("segmentation", []):
            pts = [(poly[k] * sc, poly[k + 1] * sc) for k in range(0, len(poly), 2)]
            if len(pts) >= 3:
                d.polygon(pts, fill=(r, g, b, fill_alpha))
        base = Image.alpha_composite(base, layer)

    # Pass 2: crisp darker borders on top, so overlapping edges stay distinct.
    borders = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(borders)
    for i, ann in enumerate(anns):
        r, g, b = color_for(i)
        dark = (int(r * 0.45), int(g * 0.45), int(b * 0.45), 255)
        for poly in ann.get("segmentation", []):
            pts = [(poly[k] * sc, poly[k + 1] * sc) for k in range(0, len(poly), 2)]
            if len(pts) >= 3:
                d.line(pts + [pts[0]], fill=dark, width=border_w, joint="curve")
    base = Image.alpha_composite(base, borders)
    return base.convert("RGB")


def draw_image(img_path: Path, anns: list[dict], out_path: Path,
               size: int | None = None, fill_alpha: int = 95, border_w: int = 2) -> None:
    out = render(img_path, anns, size, fill_alpha, border_w)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=88)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", type=Path, default=DATA_DIR / "annotations" / "train.json")
    p.add_argument("--images-dir", type=Path, default=DATA_DIR / "images")
    p.add_argument("--out", type=Path, default=DATA_DIR / "viz")
    p.add_argument("--num", type=int, default=8, help="number of random samples")
    p.add_argument("--image", type=str, help="render this filename only")
    p.add_argument("--all", action="store_true", help="render every image")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=2048,
                   help="scale longest edge to this many px (0 = native resolution)")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="drop annotations with score below this (GT has no score, so kept)")
    p.add_argument("--fill-alpha", type=int, default=95)
    p.add_argument("--border-w", type=int, default=2)
    args = p.parse_args()

    with args.coco.open() as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        if a.get("score", 1.0) >= args.score_thresh:
            anns_by_image.setdefault(a["image_id"], []).append(a)

    if args.image:
        targets = [im for im in coco["images"] if im["file_name"] == args.image]
        if not targets:
            raise SystemExit(f"{args.image} not in {args.coco}")
    elif args.all:
        targets = coco["images"]
    else:
        random.seed(args.seed)
        targets = random.sample(coco["images"], min(args.num, len(coco["images"])))

    size = args.size or None
    for im in targets:
        anns = anns_by_image.get(im["id"], [])
        out_path = args.out / im["file_name"]
        draw_image(args.images_dir / im["file_name"], anns, out_path,
                   size, args.fill_alpha, args.border_w)
        print(f"{im['file_name']}: {len(anns)} polygons -> {out_path}")


if __name__ == "__main__":
    main()
