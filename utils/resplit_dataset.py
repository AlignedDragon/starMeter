"""Derive a reproducible train/valid COCO split from the master VIA annotations.

The repo keeps annotations in a single editable VIA3 project
(``data/all_annotations_via.json``) and treats the COCO jsons as *derived*
products -- same philosophy as utils/prepare_sam3_dataset.py. This script reads
that master project (every annotated image, one polygon per instance) and emits:

    data/annotations/train.json   -- COCO, training split (with annotations)
    data/annotations/valid.json   -- COCO, validation split (with annotations)

The split is a seeded shuffle over all images, so it is deterministic and
re-runnable: same master + same seed/fraction => byte-identical splits.

How the master was assembled
----------------------------
``data/all_annotations_via.json`` was built by merging the 300 originally
labelled images (old ``train.json``) with images 100-206, which were labelled
later in VIA and folded back in. See ``--merge`` below to rebuild the master
from an existing COCO file plus a fresh VIA export.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

# Reuse the canonical VIA<->COCO helpers so geometry handling stays consistent.
from prepare_sam3_dataset import CATEGORY, polygon_bbox_area, via_xy_to_polygons
from predictions_to_via import build_via3

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ANN_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"
MASTER_VIA = DATA_DIR / "all_annotations_via.json"


def _img_num(fname: str) -> int:
    return int(fname.split(".")[0])


def via_polygons_by_fname(via_path: Path) -> dict[str, list[list[float]]]:
    """fname -> list of flat [x1,y1,...] polygons (one per VIA region)."""
    d = json.loads(via_path.read_text())
    vid_to_fname = {fid: rec["fname"] for fid, rec in d["file"].items()}
    by_vid: dict[str, list] = {}
    for entry in d["metadata"].values():
        by_vid.setdefault(entry["vid"], []).append(entry["xy"])
    out: dict[str, list[list[float]]] = {}
    for vid, fname in vid_to_fname.items():
        polys: list[list[float]] = []
        for xy in by_vid.get(vid, []):
            polys.extend(via_xy_to_polygons(xy))
        out[fname] = polys
    return out


def build_coco(fnames: list[str], polys_by_fname: dict[str, list[list[float]]],
               sizes: dict[str, tuple[int, int]]) -> dict:
    coco = {"images": [], "annotations": [], "categories": [CATEGORY]}
    ann_id = 1
    for img_id, fname in enumerate(fnames, start=1):
        w, h = sizes[fname]
        coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for poly in polys_by_fname[fname]:
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


def merge_master(coco_path: Path, via_path: Path, out: Path) -> None:
    """One-time migration: fold an existing COCO json and a fresh VIA export into
    a single master VIA3 project over all images, brush-ready."""
    coco = json.loads(coco_path.read_text())
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    polys: dict[str, list[list[float]]] = {im["file_name"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        polys[by_id[a["image_id"]]].extend(a["segmentation"])

    for fname, ps in via_polygons_by_fname(via_path).items():
        if fname in polys and polys[fname]:
            raise SystemExit(f"{fname} is in both COCO and VIA; refusing to merge")
        polys.setdefault(fname, []).extend(ps)

    fnames = sorted(polys, key=_img_num)
    images = [{"id": i, "file_name": f} for i, f in enumerate(fnames, start=1)]
    anns_by_img = {
        i: [{"segmentation": [p]} for p in polys[f]]
        for i, f in enumerate(fnames, start=1)
    }
    # No loc_prefix / absolute paths: files stay local (loc=1) so the project is
    # portable -- open it in VIA anywhere and point it at the image folder.
    via, _ = build_via3(images, anns_by_img, score_thresh=0.0, label=CATEGORY["name"],
                        pname="starMeter all annotations (brush-ready)")
    out.write_text(json.dumps(via))
    print(f"merged {len(fnames)} images, "
          f"{sum(len(v) for v in polys.values())} regions -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--via", type=Path, default=MASTER_VIA,
                   help="master VIA3 project to split (default: data/all_annotations_via.json)")
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    p.add_argument("--out-dir", type=Path, default=ANN_DIR,
                   help="where train.json / valid.json are written")
    p.add_argument("--valid-frac", type=float, default=0.10,
                   help="fraction of images held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge", nargs=2, metavar=("COCO", "VIA"), type=Path,
                   help="rebuild the master VIA (--via) by merging an existing COCO json "
                        "with a fresh VIA export, then exit")
    args = p.parse_args()

    if args.merge:
        merge_master(args.merge[0], args.merge[1], args.via)
        return

    polys_by_fname = via_polygons_by_fname(args.via)

    on_disk = {p.name for p in args.images_dir.glob("*.jpg")}
    missing = set(polys_by_fname) - on_disk
    if missing:
        raise SystemExit(f"{len(missing)} master images missing on disk, e.g. {sorted(missing)[:5]}")
    sizes = {f: Image.open(args.images_dir / f).size for f in polys_by_fname}

    fnames = sorted(polys_by_fname, key=_img_num)
    rng = random.Random(args.seed)
    rng.shuffle(fnames)
    n_valid = round(len(fnames) * args.valid_frac)
    valid_fnames = sorted(fnames[:n_valid], key=_img_num)
    train_fnames = sorted(fnames[n_valid:], key=_img_num)

    train_coco = build_coco(train_fnames, polys_by_fname, sizes)
    valid_coco = build_coco(valid_fnames, polys_by_fname, sizes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train.json").write_text(json.dumps(train_coco))
    (args.out_dir / "valid.json").write_text(json.dumps(valid_coco))

    print(f"master: {len(polys_by_fname)} images, "
          f"{sum(len(v) for v in polys_by_fname.values())} regions")
    print(f"train:  {len(train_coco['images'])} images, "
          f"{len(train_coco['annotations'])} annotations -> {args.out_dir / 'train.json'}")
    print(f"valid:  {len(valid_coco['images'])} images, "
          f"{len(valid_coco['annotations'])} annotations -> {args.out_dir / 'valid.json'}")


if __name__ == "__main__":
    main()
