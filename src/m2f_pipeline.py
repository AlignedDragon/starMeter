"""End-to-end Mask2Former inference: image → downscaled M2F → COCO predictions.

The full image is cropped to a bar-free square (crop_scale_bar) and run through
Mask2Former in one pass (nanostars are few and large, so no tiling is needed);
the processor downscales the square to ~`size`. Predicted masks come back in the
cropped frame's coordinates, which share the source image's top-left origin.
This matches how m2f_dataset.py builds training samples. Output is a COCO-format
JSON readable by utils/visualize_segmentations.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from dataset import crop_scale_bar

REPO = Path(__file__).resolve().parents[1]


def mask_to_polygons(mask: np.ndarray) -> tuple[list[list[float]], list[float], float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [], [0, 0, 0, 0], 0.0
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
    area = float(mask.sum())
    try:
        import cv2  # type: ignore

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        polys = [[float(v) for xy in c.reshape(-1, 2) for v in xy] for c in contours if len(c) >= 3]
        if polys:
            return polys, bbox, area
    except ImportError:
        pass
    x0, y0, w, h = bbox
    return [[x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]], bbox, area


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=REPO / "checkpoints/mask2former-nanostar")
    ap.add_argument("--input-json", type=Path, default=REPO / "data/annotations/eval.json")
    ap.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    ap.add_argument("--out", type=Path, default=REPO / "data/annotations/predictions_m2f.json")
    ap.add_argument("--size", type=int, default=1024,
                    help="processor resizes the image so its longest edge is size (no tiling)")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=256,
                    help="min mask area in native (cropped) pixels")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Mask2FormerImageProcessor.from_pretrained(args.ckpt)
    processor.size = {"shortest_edge": args.size, "longest_edge": args.size}
    processor.do_resize = True
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.ckpt).to(device).eval()

    with args.input_json.open() as f:
        coco_in = json.load(f)

    predictions: list[dict] = []
    ann_id = 1
    for info in tqdm(coco_in["images"], desc="m2f"):
        img = Image.open(args.images_dir / info["file_name"]).convert("RGB")
        img = crop_scale_bar(img)  # match training: drop the scale-bar band
        W, H = img.size  # cropped dims; top-left origin matches the source image

        enc = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        # target_sizes in cropped native coords → masks land in the source frame.
        res = processor.post_process_instance_segmentation(
            out, target_sizes=[(H, W)], threshold=args.score_thresh,
        )[0]
        seg = res["segmentation"].cpu().numpy()  # (H, W) instance ids, -1 = none

        for sinfo in res["segments_info"]:
            m = (seg == sinfo["id"]).astype(np.uint8)
            if int(m.sum()) < args.min_area:
                continue
            polys, bbox_p, area = mask_to_polygons(m)
            if area <= 0:
                continue
            predictions.append({
                "id": ann_id,
                "image_id": info["id"],
                "category_id": 1,
                "segmentation": polys,
                "bbox": bbox_p,
                "area": area,
                "iscrowd": 0,
                "score": float(sinfo["score"]),
            })
            ann_id += 1

    out_coco = {
        "images": coco_in["images"],
        "annotations": predictions,
        "categories": coco_in.get("categories", [{"id": 1, "name": "nanostar"}]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out_coco, f)
    print(f"wrote {len(predictions)} predictions over {len(coco_in['images'])} images -> {args.out}")


if __name__ == "__main__":
    main()
