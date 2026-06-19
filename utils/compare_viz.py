"""Stitch two prediction sets side by side (left | right) for the same images.

Reuses utils/visualize_segmentations.render so each panel gets the proper overlapping
nanostar treatment (per-instance alpha-blended fills + crisp darker borders),
then pastes the two panels together under labelled header bars so one checkpoint
can be eyeballed against another on the validation set:

    python utils/compare_viz.py \
        --left  results/predictions_valid_m2f.old.json --left-label  raw \
        --right results/predictions_valid_m2f.json     --right-label denoiser \
        --out results/viz_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from visualize_segmentations import REPO_ROOT as REPO, render

BAR_H = 56


def _by_image(coco_path: Path, score_thresh: float) -> tuple[list[dict], dict]:
    coco = json.loads(coco_path.read_text())
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        if a.get("score", 1.0) >= score_thresh:
            by_img.setdefault(a["image_id"], []).append(a)
    return coco["images"], by_img


def _label_bar(width: int, text: str) -> Image.Image:
    bar = Image.new("RGB", (width, BAR_H), (20, 20, 20))
    d = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    d.text((12, 10), text, fill=(255, 255, 255), font=font)
    return bar


def _panel(img_path: Path, anns: list[dict], label: str, args) -> Image.Image:
    body = render(img_path, anns, args.size, args.fill_alpha, args.border_w)
    bar = _label_bar(body.width, f"{label}  ({len(anns)} inst)")
    out = Image.new("RGB", (body.width, body.height + BAR_H))
    out.paste(bar, (0, 0))
    out.paste(body, (0, BAR_H))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--left-label", default="left")
    p.add_argument("--right-label", default="right")
    p.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    p.add_argument("--out", type=Path, default=REPO / "data/viz_compare")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--fill-alpha", type=int, default=95)
    p.add_argument("--border-w", type=int, default=2)
    args = p.parse_args()

    images, lanns = _by_image(args.left, args.score_thresh)
    _, ranns = _by_image(args.right, args.score_thresh)
    args.out.mkdir(parents=True, exist_ok=True)

    for im in images:
        src = args.images_dir / im["file_name"]
        if not src.exists():
            continue
        left = _panel(src, lanns.get(im["id"], []), args.left_label, args)
        right = _panel(src, ranns.get(im["id"], []), args.right_label, args)
        combo = Image.new("RGB", (left.width + right.width, max(left.height, right.height)),
                          (20, 20, 20))
        combo.paste(left, (0, 0))
        combo.paste(right, (left.width, 0))
        out_path = args.out / im["file_name"]
        combo.save(out_path, quality=88)
        print(f"{im['file_name']}: {len(lanns.get(im['id'], []))} | "
              f"{len(ranns.get(im['id'], []))} -> {out_path}")


if __name__ == "__main__":
    main()
