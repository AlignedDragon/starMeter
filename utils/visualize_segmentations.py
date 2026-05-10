"""Render polygon overlays from a SAM3/COCO annotation file for sanity checks.

Examples:
    python utils/visualize_segmentations.py                       # 8 random train images
    python utils/visualize_segmentations.py --image 42.jpg        # one specific image
    python utils/visualize_segmentations.py --num 20 --out vis/   # 20 images, custom out dir
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


def color_for(idx: int) -> tuple[int, int, int, int]:
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255), 110


def draw_image(img_path: Path, anns: list[dict], out_path: Path) -> None:
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, ann in enumerate(anns):
        col = color_for(i)
        outline = (col[0], col[1], col[2], 255)
        for poly in ann.get("segmentation", []):
            pts = [(poly[k], poly[k + 1]) for k in range(0, len(poly), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=col, outline=outline)
    out = Image.alpha_composite(img, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=85)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=Path, default=DATA_DIR / "annotations" / "train.json")
    p.add_argument("--images-dir", type=Path, default=DATA_DIR / "images")
    p.add_argument("--out", type=Path, default=DATA_DIR / "viz")
    p.add_argument("--num", type=int, default=8, help="number of random samples")
    p.add_argument("--image", type=str, help="render this filename only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with args.coco.open() as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    if args.image:
        targets = [im for im in coco["images"] if im["file_name"] == args.image]
        if not targets:
            raise SystemExit(f"{args.image} not in {args.coco}")
    else:
        random.seed(args.seed)
        targets = random.sample(coco["images"], min(args.num, len(coco["images"])))

    for im in targets:
        anns = anns_by_image.get(im["id"], [])
        out_path = args.out / im["file_name"]
        draw_image(args.images_dir / im["file_name"], anns, out_path)
        print(f"{im['file_name']}: {len(anns)} polygons -> {out_path}")


if __name__ == "__main__":
    main()
