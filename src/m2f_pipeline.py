"""End-to-end Mask2Former inference: image → tiled M2F → COCO predictions.

Mask2Former is run on overlapping 1024² tiles of the full 4096² image; per-tile
instance predictions are stitched into a global frame and de-duplicated by IoU
to suppress boundary double-counts. Output is a COCO-format JSON readable by
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
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

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


def iou_bbox(a: list[float], b: list[float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    return inter / (union + 1e-6)


def run_tile(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    tile: Image.Image,
    device: str,
    score_thresh: float,
) -> list[tuple[float, np.ndarray]]:
    """Returns list of (score, binary mask H×W) for one tile."""
    enc = processor(images=tile, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    res = processor.post_process_instance_segmentation(
        out, target_sizes=[tile.size[::-1]], threshold=score_thresh,
    )[0]
    seg = res["segmentation"].cpu().numpy()  # (H, W) instance ids, -1 = none
    instances: list[tuple[float, np.ndarray]] = []
    for info in res["segments_info"]:
        m = seg == info["id"]
        if not m.any():
            continue
        instances.append((float(info["score"]), m.astype(np.uint8)))
    return instances


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=REPO / "checkpoints/mask2former-nanostar")
    ap.add_argument("--input-json", type=Path, default=REPO / "data/annotations/eval.json")
    ap.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    ap.add_argument("--out", type=Path, default=REPO / "data/annotations/predictions_m2f.json")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--dedup-iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Mask2FormerImageProcessor.from_pretrained(args.ckpt)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.ckpt).to(device).eval()

    with args.input_json.open() as f:
        coco_in = json.load(f)

    predictions: list[dict] = []
    ann_id = 1
    for info in tqdm(coco_in["images"], desc="m2f"):
        img = Image.open(args.images_dir / info["file_name"]).convert("RGB")
        W, H = img.size

        # Generate tile origins covering the image, including the bottom/right edges.
        xs = list(range(0, max(1, W - args.tile + 1), args.stride))
        if xs[-1] + args.tile < W:
            xs.append(W - args.tile)
        ys = list(range(0, max(1, H - args.tile + 1), args.stride))
        if ys[-1] + args.tile < H:
            ys.append(H - args.tile)

        # Collect all candidate instances on the global canvas.
        candidates: list[tuple[float, np.ndarray, list[float]]] = []  # (score, mask, bbox)
        for y0 in ys:
            for x0 in xs:
                tile = img.crop((x0, y0, x0 + args.tile, y0 + args.tile))
                for score, m_local in run_tile(model, processor, tile, device, args.score_thresh):
                    if int(m_local.sum()) < args.min_area:
                        continue
                    m_global = np.zeros((H, W), dtype=np.uint8)
                    m_global[y0:y0 + args.tile, x0:x0 + args.tile] = m_local
                    ys_, xs_ = np.where(m_global > 0)
                    if xs_.size == 0:
                        continue
                    bbox = [float(xs_.min()), float(ys_.min()),
                            float(xs_.max() - xs_.min()), float(ys_.max() - ys_.min())]
                    candidates.append((score, m_global, bbox))

        # Greedy NMS by score over global masks (cheap bbox prefilter, then mask IoU).
        candidates.sort(key=lambda t: -t[0])
        kept: list[tuple[float, np.ndarray, list[float]]] = []
        for score, m, bbox in candidates:
            dup = False
            for _, mk, bk in kept:
                if iou_bbox(bbox, bk) < args.dedup_iou * 0.5:
                    continue
                if mask_iou(m, mk) > args.dedup_iou:
                    dup = True
                    break
            if not dup:
                kept.append((score, m, bbox))

        for score, m, bbox in kept:
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
