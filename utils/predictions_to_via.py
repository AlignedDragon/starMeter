"""Turn model predictions (COCO json) into a VIA3 project for model-assisted labeling.

Workflow to grow the training set cheaply (correct, don't draw from scratch):

    1. run inference  ->  predictions json (src/m2f_pipeline.py)
    2. this script     ->  a VIA3 project json
    3. open it in the VGG Image Annotator (VIA3), add the image folder, CORRECT the
       predicted polygons (fix branches, delete false positives, add misses)
    4. export the VIA project, then prepare_sam3_dataset.py converts it back to COCO
       and you retrain with the larger train.json.

Output schema matches the VIA3 projects this repo already ingests
(`file` / `view` / `metadata`, polygons as `xy = [7, x1, y1, x2, y2, ...]`), so
utils/prepare_sam3_dataset.py::load_via reads it unchanged.

Coordinates: predictions are in the bar-free cropped frame, which shares the
source image's top-left origin (see crop_scale_bar), so the polygons drop straight
onto the full 4096² image in VIA with no transform — they just don't extend into
the trimmed bottom/right band (which is correct; that band is ignored in training).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _poly_area(poly: list[float]) -> float:
    """Shoelace area of a flat [x1,y1,x2,y2,...] polygon."""
    xs, ys = poly[0::2], poly[1::2]
    a = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def _keep_polys(segmentation: list[list[float]]) -> list[list[float]]:
    """Keep only the single largest polygon of an instance. Disconnected fragments
    are dropped downstream anyway, so an instance is reduced to its main blob."""
    polys = [p for p in segmentation if len(p) >= 6]
    if not polys:
        return []
    return [max(polys, key=_poly_area)]


def build_via3(images: list[dict], anns_by_img: dict[int, list[dict]],
               score_thresh: float, label: str, pname: str) -> tuple[dict, int]:
    project = {
        "pid": "__VIA_PROJECT_ID__",
        "rev": "__VIA_PROJECT_REV_ID__",
        "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
        "pname": pname,
        "creator": "starMeter utils/predictions_to_via.py",
        "created": 0,
        "vid_list": [],
    }
    config = {
        "file": {"loc_prefix": {"1": "", "2": "", "3": "", "4": ""}},
        "ui": {
            "file_content_align": "center",
            "file_metadata_editor_visible": True,
            "spatial_metadata_editor_visible": True,
            "spatial_region_label_attribute_id": "",
            "gtimeline_visible_row_count": "4",
        },
    }
    # One per-region attribute so corrected regions carry the class in VIA.
    attribute = {
        "1": {
            "aname": "class", "anchor_id": "FILE1_Z0_XY1", "type": 3,
            "desc": "", "options": {label: label}, "default_option_id": "",
        }
    }

    file: dict[str, dict] = {}
    view: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    kept = 0
    for i, im in enumerate(images, start=1):
        fid = str(i)
        file[fid] = {"fid": fid, "fname": im["file_name"], "type": 2, "loc": 1,
                     "src": im["file_name"]}
        view[fid] = {"fid_list": [fid]}
        project["vid_list"].append(fid)
        ri = 0
        for ann in anns_by_img.get(im["id"], []):
            if ann.get("score", 1.0) < score_thresh:
                continue
            for poly in _keep_polys(ann.get("segmentation", [])):
                xy = [7] + [round(float(c), 1) for c in poly]  # 7 == polygon
                metadata[f"{fid}_{ri}"] = {
                    "vid": fid, "flg": 0, "z": [], "xy": xy, "av": {"1": label},
                }
                ri += 1
                kept += 1
    return {"project": project, "config": config, "attribute": attribute,
            "file": file, "view": view, "metadata": metadata}, kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, default=REPO / "data/annotations/predictions_m2f.json",
                   help="COCO predictions json from src/m2f_pipeline.py")
    p.add_argument("--out", type=Path, default=Path.home() / "Downloads/predictions_via.json",
                   help="VIA3 project json to open for correction")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="drop predictions below this score before exporting")
    p.add_argument("--label", default="nanostar")
    p.add_argument("--pname", default="starmeter model-assisted")
    args = p.parse_args()

    coco = json.loads(args.pred.read_text())
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    via, kept = build_via3(coco["images"], anns_by_img, args.score_thresh,
                           args.label, args.pname)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(via))
    print(f"{len(coco['images'])} images, {kept} predicted regions (score>={args.score_thresh}) "
          f"-> {args.out}")
    print("Open in VIA3 (https://www.robots.ox.ac.uk/~vgg/software/via/), add the image "
          "folder, correct the polygons, then re-export and run prepare_sam3_dataset.py.")


if __name__ == "__main__":
    main()
