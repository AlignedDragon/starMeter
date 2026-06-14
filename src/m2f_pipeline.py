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


def overlapping_instances(out, size, score_thresh, mask_thresh=0.5):
    """Per-query thresholded masks, allowing instances to OVERLAP.

    `post_process_instance_segmentation` argmaxes every pixel to a single
    instance, so where two nanostar branches cross the shared pixels are awarded
    to one star and carved out of the other. For overlapping nanostars we instead
    threshold each kept query's own mask, so every star keeps its full branches
    through the crossings. Returns a list of (mask uint8 (H,W), score).
    """
    H, W = size
    class_logits = out.class_queries_logits[0]            # (Q, num_labels + 1)
    mask_logits = out.masks_queries_logits[0]             # (Q, h, w)
    scores = class_logits.softmax(-1)
    num_labels = scores.shape[-1] - 1                     # last column = "no object"
    fg, _ = scores[:, :num_labels].max(-1)                # foreground prob per query
    keep = fg >= score_thresh
    if int(keep.sum()) == 0:
        return []
    masks = torch.nn.functional.interpolate(
        mask_logits[keep].unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
    )[0].sigmoid()                                        # (k, H, W) in native coords
    out_list = []
    for prob, s in zip(masks, fg[keep]):
        binm = prob > mask_thresh
        n = int(binm.sum())
        if n == 0:
            continue
        # mask-aware score: class prob weighted by mask confidence (as in detectron2)
        mask_score = float((prob[binm]).mean())
        out_list.append((binm.cpu().numpy().astype(np.uint8), float(s) * mask_score))
    return out_list


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
    ap.add_argument("--no-overlap", action="store_true",
                    help="use argmax label map (mutually exclusive masks) instead of "
                         "overlapping per-query masks; overlaps are kept by default")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Mask2FormerImageProcessor.from_pretrained(args.ckpt)
    # Exact square resize (see m2f_train.py: shortest_edge==longest_edge is
    # rejected for square inputs).
    processor.size = {"height": args.size, "width": args.size}
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
        if args.no_overlap:
            res = processor.post_process_instance_segmentation(
                out, target_sizes=[(H, W)], threshold=args.score_thresh,
            )[0]
            seg = res["segmentation"].cpu().numpy()  # (H, W) instance ids, -1 = none
            instances = [((seg == s["id"]).astype(np.uint8), float(s["score"]))
                         for s in res["segments_info"]]
        else:
            instances = overlapping_instances(out, (H, W), args.score_thresh)

        for m, score in instances:
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
                "score": score,
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
