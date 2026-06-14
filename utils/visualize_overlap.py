"""Overlay COCO instance predictions on the cropped frame the model sees.

Tailored for OVERLAPPING nanostars: each instance is alpha-blended (so where two
stars overlap you see both colors mixing) and then a crisp, darker per-instance
border is drawn on top, so overlapping boundaries stay readable.

The crop matches src/dataset.py::crop_scale_bar (top-left square, bottom/right
`frac` trimmed) so polygons in cropped-frame coordinates line up after scaling.

Examples:
    python utils/visualize_overlap.py --pred data/annotations/predictions_m2f.json --all
    python utils/visualize_overlap.py --pred preds.json --image 100.jpg --score-thresh 0.7
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]


def color_for(idx: int) -> tuple[int, int, int]:
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def render(img_path: Path, anns: list[dict], size: int, frac: float,
           fill_alpha: int, border_w: int) -> Image.Image:
    img = Image.open(img_path).convert("RGBA")
    W, H = img.size
    side = min(int(H * (1 - frac)), W)               # same square crop as training
    base = img.crop((0, 0, side, side)).resize((size, size), Image.BILINEAR).convert("RGBA")
    sc = size / side

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, required=True, help="COCO predictions json")
    p.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    p.add_argument("--out", type=Path, default=REPO / "data/viz_overlap")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--frac", type=float, default=0.10, help="scale-bar band fraction (match training)")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--fill-alpha", type=int, default=95)
    p.add_argument("--border-w", type=int, default=2)
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--image", type=str, help="render this filename only")
    p.add_argument("--all", action="store_true", help="render every image")
    args = p.parse_args()

    coco = json.loads(args.pred.read_text())
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        if a.get("score", 1.0) >= args.score_thresh:
            by_img.setdefault(a["image_id"], []).append(a)

    images = coco["images"]
    if args.image:
        images = [im for im in images if im["file_name"] == args.image]
    elif not args.all:
        images = images[: args.num]

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in images:
        src = args.images_dir / im["file_name"]
        if not src.exists():
            continue
        out = render(src, by_img.get(im["id"], []), args.size, args.frac,
                     args.fill_alpha, args.border_w)
        out.save(args.out / im["file_name"], quality=88)
        n += 1
    print(f"wrote {n} overlays -> {args.out}")


if __name__ == "__main__":
    main()
