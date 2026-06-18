"""Export nanostar branch lines (from branch_length.py) as a VIA3 project.

Each arm becomes a VIA POLYLINE region (xy = [4, x1, y1, x2, y2]) so you can open
the project in the VGG Image Annotator and edit the lines -- drag endpoints, add or
delete branches -- then re-export. Same VIA3 schema this repo already ingests
(`file` / `view` / `metadata`); shape id 4 == polyline.

Branch lines are recomputed from the segmentation with branch_length.arm_segments
(complete skeleton for arms, shrunk mask for the center, subtract, fit a line per
arm). They are computed at `--size` and mapped back to ORIGINAL image coordinates
(top-left bar-free square shares the source origin), so they sit correctly on the
full image in VIA.

Examples:
    python utils/branch_lines_to_via.py                                   # GT train.json
    python utils/branch_lines_to_via.py --coco data/annotations/predictions_overlap.json
    python utils/branch_lines_to_via.py --core-radius 16 --min-branch 18
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from branch_length import arm_segments
from branch_lines import instance_mask

REPO = Path(__file__).resolve().parents[1]


def build_via3(images: list[dict], by_img: dict[int, list[dict]], size: int, frac: float,
               min_branch: float, core_radius: int, score_thresh: float,
               label: str, pname: str) -> tuple[dict, int]:
    project = {
        "pid": "__VIA_PROJECT_ID__", "rev": "__VIA_PROJECT_REV_ID__",
        "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__", "pname": pname,
        "creator": "starMeter utils/branch_lines_to_via.py", "created": 0, "vid_list": [],
    }
    config = {
        "file": {"loc_prefix": {"1": "", "2": "", "3": "", "4": ""}},
        "ui": {"file_content_align": "center", "file_metadata_editor_visible": True,
               "spatial_metadata_editor_visible": True,
               "spatial_region_label_attribute_id": "", "gtimeline_visible_row_count": "4"},
    }
    attribute = {"1": {"aname": "class", "anchor_id": "FILE1_Z0_XY1", "type": 3,
                       "desc": "", "options": {label: label}, "default_option_id": ""}}

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

        W, H = im["width"], im["height"]
        side = min(int(H * (1 - frac)), W)         # bar-free square crop (shared origin)
        sc = size / side                            # polygon coords -> compute-size pixels
        ri = 0
        for ann in by_img.get(im["id"], []):
            if ann.get("score", 1.0) < score_thresh:
                continue
            m = instance_mask(ann.get("segmentation", []), size, sc)
            for (x1, y1), (x2, y2), _ in arm_segments(m, min_branch, core_radius):
                # map compute-size coords back to ORIGINAL image coordinates
                xy = [4,
                      round(x1 / sc, 1), round(y1 / sc, 1),
                      round(x2 / sc, 1), round(y2 / sc, 1)]
                metadata[f"{fid}_{ri}"] = {"vid": fid, "flg": 0, "z": [], "xy": xy,
                                           "av": {"1": label}}
                ri += 1
                kept += 1
    return {"project": project, "config": config, "attribute": attribute,
            "file": file, "view": view, "metadata": metadata}, kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json")
    p.add_argument("--out", type=Path, default=REPO / "data/annotations/branch_lines_via.json")
    p.add_argument("--size", type=int, default=1024, help="resolution the lines are computed at")
    p.add_argument("--frac", type=float, default=0.10)
    p.add_argument("--min-branch", type=float, default=12)
    p.add_argument("--core-radius", type=int, default=12)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--label", default="branch")
    p.add_argument("--pname", default="starmeter branch lines")
    args = p.parse_args()

    coco = json.loads(args.coco.read_text())
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    images = [im for im in coco["images"] if im["id"] in by_img]

    via, kept = build_via3(images, by_img, args.size, args.frac, args.min_branch,
                           args.core_radius, args.score_thresh, args.label, args.pname)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(via))
    print(f"{len(images)} images, {kept} branch polylines -> {args.out}")
    print("Open in VIA3 (https://www.robots.ox.ac.uk/~vgg/software/via/), add the image "
          "folder, edit the lines, then re-export.")


if __name__ == "__main__":
    main()
