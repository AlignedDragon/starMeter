"""End-to-end inference: image → centroid heatmap → SAM (point prompts) → COCO.

For each input image we run the stage-1 UNet to get nanostar centers, then for
every center query SAM with that point as positive and the K nearest neighbors
as negatives. Output is a COCO-format JSON readable by
utils/visualize_segmentations.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from centroid import UNet, detect

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


def neighbors(points: np.ndarray, idx: int, k: int) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros((0, 2), dtype=np.float32)
    d = np.linalg.norm(points - points[idx], axis=1)
    d[idx] = np.inf
    order = np.argsort(d)[:k]
    return points[order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--centroid-ckpt", type=Path, default=REPO / "checkpoints/centroid.pt")
    ap.add_argument("--sam-ckpt", type=Path, default=REPO / "checkpoints/sam3-nanostar")
    ap.add_argument("--input-json", type=Path, default=REPO / "data/annotations/eval.json")
    ap.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    ap.add_argument("--out", type=Path, default=REPO / "data/annotations/predictions.json")
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--min-dist", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.centroid_ckpt, map_location=device)
    centroid_model = UNet().to(device)
    centroid_model.load_state_dict(ckpt["state_dict"])

    processor = SamProcessor.from_pretrained(args.sam_ckpt)
    sam = SamModel.from_pretrained(args.sam_ckpt).to(device).eval()

    with args.input_json.open() as f:
        coco_in = json.load(f)

    predictions: list[dict] = []
    ann_id = 1
    for info in tqdm(coco_in["images"], desc="pipeline"):
        img = Image.open(args.images_dir / info["file_name"]).convert("RGB")
        pts = detect(centroid_model, img, device,
                     out_size=ckpt.get("out_size", 512),
                     thresh=args.thresh, min_dist=args.min_dist)
        if len(pts) == 0:
            continue
        for i, p in enumerate(pts):
            negs = neighbors(pts, i, args.negatives)
            prompt = np.concatenate([p[None], negs], axis=0).tolist()
            labels = [1] + [0] * len(negs)
            enc = processor(
                img,
                input_points=[[prompt]],
                input_labels=[[labels]],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = sam(**enc, multimask_output=False)
            mask = processor.image_processor.post_process_masks(
                out.pred_masks.cpu(), enc["original_sizes"].cpu(), enc["reshaped_input_sizes"].cpu()
            )[0][0, 0].numpy().astype(np.uint8)
            polys, bbox, area = mask_to_polygons(mask)
            if area <= 0:
                continue
            predictions.append({
                "id": ann_id,
                "image_id": info["id"],
                "category_id": 1,
                "segmentation": polys,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "score": 1.0,
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
