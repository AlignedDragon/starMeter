"""Dump the exact preprocessing the Mask2Former pipeline feeds the model, with
ground-truth masks overlaid, for visual inspection.

For every image we:
  1. crop to a bar-free SQUARE (drops the bottom scale-bar band + a right strip),
  2. downscale to `--size`,
  3. overlay the COCO ground-truth polygons (scaled into the crop),
and save it to ``--out``. The crop/size match src/dataset.py::crop_scale_bar and
the m2f processor config, so what you see here is what the model sees (minus the
random flip/rot90, which is augmentation) — and the overlay confirms the
annotations still line up after cropping.

We also write a small augmentation grid (all 8 flip×rot90 variants) for the first
few images. The masks are composited BEFORE the dihedral transform, so they
rotate/flip with the image and stay aligned by construction.

  python utils/preprocess_images.py                 # all images -> data/preprocessed
  python utils/preprocess_images.py --limit 20      # just the first 20
  python utils/preprocess_images.py --no-overlay    # plain previews, no masks
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
SCALE_BAR_FRAC = 0.10  # keep in sync with src/dataset.py


def crop_scale_bar(img: Image.Image, frac: float = SCALE_BAR_FRAC) -> int:
    """Crop ``img`` to the bar-free square in place-of-return; return its side.

    Same logic as src/dataset.py::crop_scale_bar. Returns the square side so the
    caller can scale full-res polygon coordinates into the crop.
    """
    W, H = img.size
    return min(int(H * (1 - frac)), W)


def color_for(idx: int) -> tuple[int, int, int, int]:
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255), 110


def annotated_square(img: Image.Image, anns: list[dict], size: int) -> Image.Image:
    """Crop to the bar-free square, downscale to `size`, overlay GT polygons.

    Polygons are in full-image coordinates; the crop shares the top-left origin,
    so we only scale by ``size / side`` (no offset). Points outside the square
    are clipped naturally by the canvas.
    """
    side = crop_scale_bar(img)
    square = img.crop((0, 0, side, side)).resize((size, size), Image.BILINEAR).convert("RGBA")
    if anns:
        scale = size / side
        overlay = Image.new("RGBA", square.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for i, ann in enumerate(anns):
            col = color_for(i)
            outline = (col[0], col[1], col[2], 255)
            for poly in ann.get("segmentation", []):
                pts = [(poly[k] * scale, poly[k + 1] * scale) for k in range(0, len(poly), 2)]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=col, outline=outline)
        square = Image.alpha_composite(square, overlay)
    return square.convert("RGB")


def dihedral(img: Image.Image, flip: bool, k: int) -> Image.Image:
    """Optional h-flip then k*90° clockwise rotation (matches _flip_rot)."""
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if k:
        img = img.rotate(-90 * k, expand=True)
    return img


def aug_grid(annotated: Image.Image, cell: int = 256) -> Image.Image:
    """2 rows (no-flip / flip) x 4 cols (rot 0/90/180/270) contact sheet.

    Operates on the already-annotated square, so masks transform with the image.
    """
    pad = 8
    grid = Image.new("RGB", (4 * cell + 5 * pad, 2 * cell + 3 * pad), (32, 32, 32))
    for r, flip in enumerate((False, True)):
        for k in range(4):
            c = dihedral(annotated, flip, k).resize((cell, cell), Image.BILINEAR)
            grid.paste(c, (pad + k * (cell + pad), pad + r * (cell + pad)))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    ap.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json")
    ap.add_argument("--out", type=Path, default=REPO / "data/preprocessed")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--aug-demo", type=int, default=6,
                    help="how many images to also dump an augmentation grid for")
    ap.add_argument("--no-overlay", action="store_true", help="skip GT masks")
    args = ap.parse_args()

    anns_by_file: dict[str, list[dict]] = {}
    if not args.no_overlay and args.coco.exists():
        with args.coco.open() as f:
            coco = json.load(f)
        id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
        for a in coco["annotations"]:
            anns_by_file.setdefault(id_to_file[a["image_id"]], []).append(a)

    files = sorted(args.images_dir.glob("*.jpg"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if args.limit:
        files = files[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    demo_dir = args.out / "_aug_demo"
    demo_dir.mkdir(exist_ok=True)

    n_overlaid = 0
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        anns = anns_by_file.get(f.name, [])
        n_overlaid += bool(anns)
        annotated = annotated_square(img, anns, args.size)
        annotated.save(args.out / f"{f.stem}.jpg", quality=90)
        if i < args.aug_demo:
            aug_grid(annotated).save(demo_dir / f"{f.stem}_grid.jpg", quality=90)
        print(f"[{i + 1}/{len(files)}] {f.name}: {len(anns)} polygons")

    print(f"\nwrote {len(files)} previews ({n_overlaid} with GT masks) -> {args.out}")
    print(f"augmentation grids ({min(args.aug_demo, len(files))}) -> {demo_dir}")


if __name__ == "__main__":
    main()
