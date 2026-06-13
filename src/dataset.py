"""Datasets and augmentations for the two-stage nanostar pipeline.

Augmentations are intentionally minimal: random crop (mandatory at 4096²),
4-way 90° rotation, horizontal flip. These exploit symmetries the data
genuinely has; we skip photometric jitter on the assumption that the TEM
distribution is fixed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


# --- geometry helpers --------------------------------------------------------

def polygons_to_mask(polygons: Iterable[Iterable[float]], h: int, w: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in polygons:
        if len(poly) >= 6:
            d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
    return np.asarray(m, dtype=np.uint8)


SCALE_BAR_FRAC = 0.10  # bottom band that always contains the burned-in scale bar


def crop_scale_bar(img: Image.Image, frac: float = SCALE_BAR_FRAC) -> Image.Image:
    """Remove the burned-in "100 nm" scale bar by cropping to a bar-free square.

    Left in place the bar (a) looks like a dark nanostar branch and (b) becomes a
    spatial/orientation landmark once flip/rot90 augmentation moves it around.
    A flat fill would be just as learnable — and worse where a star overlaps the
    bar — so we physically crop it out instead, accepting the loss of any object
    in the discarded region.

    We crop to a SQUARE of side ``H*(1-frac)``, trimming the bottom ``frac`` band
    (which holds the bar) and an equal strip off the right to keep it square. A
    square is essential: if we kept the image rectangular, flip/rot90 would change
    its aspect ratio, forcing orientation-dependent padding downstream — and that
    padding's position would itself leak the rotation to the model. With a square,
    rot90 introduces no padding at all.

    The crop keeps the top-left origin, so every polygon/centroid (x, y) stays
    valid; clipped instances simply fall outside the canvas. Apply BEFORE
    augmentation, identically at train and inference time. Annotations in the
    discarded region are dropped, so annotators should ignore it.
    """
    W, H = img.size
    side = min(int(H * (1 - frac)), W)
    return img.crop((0, 0, side, side))


def core_point(polygons: Iterable[Iterable[float]], bbox: list[float]) -> tuple[float, float]:
    """Pole of inaccessibility: pixel inside the polygon farthest from any
    boundary, found via distance transform on a tight crop. This sits in the
    dense core of the nanostar, not at the geometric/bbox center which can be
    far off-shape for asymmetric stars."""
    x, y, w, h = bbox
    pad = 4
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    cw = int(w) + 2 * pad
    ch = int(h) + 2 * pad
    shifted = [[c - (x0 if i % 2 == 0 else y0) for i, c in enumerate(p)] for p in polygons]
    mask = polygons_to_mask(shifted, ch, cw)
    if mask.sum() == 0:
        return x + w / 2, y + h / 2
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    iy, ix = np.unravel_index(int(dt.argmax()), dt.shape)
    return float(ix + x0), float(iy + y0)


def polygon_centroids(coco: dict) -> dict[int, list[tuple[float, float]]]:
    """For each image_id, list of (x, y) core points (distance-transform max)."""
    out: dict[int, list[tuple[float, float]]] = {}
    for a in coco["annotations"]:
        if not a.get("segmentation"):
            continue
        out.setdefault(a["image_id"], []).append(core_point(a["segmentation"], a["bbox"]))
    return out


# --- augmentation ------------------------------------------------------------

class CropFlipRot:
    """Random square crop + flip + rot90. Operates on (image, points, masks)."""

    def __init__(self, crop: int, flip: bool = True, rot90: bool = True):
        self.crop = crop
        self.flip = flip
        self.rot90 = rot90

    def __call__(
        self,
        img: Image.Image,
        points: np.ndarray | None = None,        # (N, 2) xy
        masks: np.ndarray | None = None,         # (N, H, W)
    ) -> tuple[Image.Image, np.ndarray | None, np.ndarray | None]:
        W, H = img.size
        c = self.crop
        x0 = random.randint(0, max(0, W - c))
        y0 = random.randint(0, max(0, H - c))
        img = img.crop((x0, y0, x0 + c, y0 + c))
        if points is not None:
            points = points - np.array([x0, y0], dtype=np.float32)
        if masks is not None:
            masks = masks[:, y0:y0 + c, x0:x0 + c]

        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if points is not None:
                points[:, 0] = c - 1 - points[:, 0]
            if masks is not None:
                masks = masks[:, :, ::-1].copy()

        if self.rot90:
            k = random.randint(0, 3)
            if k:
                img = img.rotate(-90 * k, expand=False)
                if points is not None:
                    for _ in range(k):
                        x, y = points[:, 0].copy(), points[:, 1].copy()
                        points[:, 0] = y
                        points[:, 1] = c - 1 - x
                if masks is not None:
                    masks = np.rot90(masks, k=-k, axes=(1, 2)).copy()
        return img, points, masks


# --- centroid heatmap dataset (stage 1) --------------------------------------

def gaussian_heatmap(points: np.ndarray, h: int, w: int, sigma: float = 4.0) -> np.ndarray:
    hm = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return hm
    yy, xx = np.mgrid[0:h, 0:w]
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            hm = np.maximum(hm, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))
    return hm


class CentroidHeatmapDataset(Dataset):
    """Random crops of full images → input image + centroid Gaussian heatmap."""

    def __init__(
        self,
        coco_json: Path,
        images_dir: Path,
        crop: int = 1024,
        out_size: int = 512,
        sigma: float = 4.0,
        samples_per_epoch: int = 600,
    ):
        with Path(coco_json).open() as f:
            coco = json.load(f)
        self.images = list(coco["images"])
        self.points_by_id = polygon_centroids(coco)
        self.images_dir = Path(images_dir)
        self.aug = CropFlipRot(crop)
        self.out_size = out_size
        self.sigma = sigma
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> dict:
        info = random.choice(self.images)
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        pts = np.asarray(self.points_by_id.get(info["id"], []), dtype=np.float32).reshape(-1, 2)
        img, pts, _ = self.aug(img, points=pts)
        c = self.aug.crop
        s = self.out_size
        img_r = img.resize((s, s), Image.BILINEAR)
        if pts is not None and len(pts):
            pts = pts * (s / c)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < s) & (pts[:, 1] >= 0) & (pts[:, 1] < s)]
        hm = gaussian_heatmap(pts if pts is not None else np.zeros((0, 2)), s, s, self.sigma)
        x = torch.from_numpy(np.asarray(img_r, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {"image": x, "heatmap": torch.from_numpy(hm)}


# --- SAM point-prompt dataset (stage 2) --------------------------------------

class SamPointDataset(Dataset):
    """One sample = one instance, prompted by a positive point inside its mask
    plus negative points at neighboring instances' centroids."""

    def __init__(
        self,
        coco_json: Path,
        images_dir: Path,
        processor,
        crop: int = 1024,
        max_negatives: int = 4,
        neighbor_radius: float = 400.0,
    ):
        with Path(coco_json).open() as f:
            coco = json.load(f)
        self.images = {im["id"]: im for im in coco["images"]}
        self.anns_by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            if a.get("segmentation"):
                self.anns_by_img.setdefault(a["image_id"], []).append(a)
        self.index = [(img_id, i) for img_id, anns in self.anns_by_img.items() for i, _ in enumerate(anns)]
        self.centers_by_img: dict[int, np.ndarray] = {
            img_id: np.array([core_point(a["segmentation"], a["bbox"]) for a in anns], dtype=np.float32)
            for img_id, anns in self.anns_by_img.items()
        }
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.aug = CropFlipRot(crop)
        self.crop = crop
        self.max_negatives = max_negatives
        self.neighbor_radius = neighbor_radius

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        img_id, ann_idx = self.index[idx]
        info = self.images[img_id]
        anns = self.anns_by_img[img_id]
        ann = anns[ann_idx]

        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        W, H = img.size
        target = polygons_to_mask(ann["segmentation"], H, W)

        centers = self.centers_by_img[img_id].copy()
        # Crop window centered (with jitter) on target so it stays in view.
        cx, cy = centers[ann_idx]
        c = self.crop
        jitter = c // 4
        x0 = int(np.clip(cx - c / 2 + random.uniform(-jitter, jitter), 0, max(0, W - c)))
        y0 = int(np.clip(cy - c / 2 + random.uniform(-jitter, jitter), 0, max(0, H - c)))
        img_c = img.crop((x0, y0, x0 + c, y0 + c))
        target_c = target[y0:y0 + c, x0:x0 + c]
        centers_c = centers - np.array([x0, y0], dtype=np.float32)

        # Inline flip/rot (we already cropped manually above to control center).
        if random.random() < 0.5:
            img_c = img_c.transpose(Image.FLIP_LEFT_RIGHT)
            target_c = target_c[:, ::-1].copy()
            centers_c[:, 0] = c - 1 - centers_c[:, 0]
        k = random.randint(0, 3)
        if k:
            img_c = img_c.rotate(-90 * k, expand=False)
            target_c = np.rot90(target_c, k=-k).copy()
            for _ in range(k):
                x, y = centers_c[:, 0].copy(), centers_c[:, 1].copy()
                centers_c[:, 0] = y
                centers_c[:, 1] = c - 1 - x

        # Positive point: a random True pixel inside the target mask.
        ys, xs = np.where(target_c > 0)
        if len(xs) == 0:
            # Crop dropped the instance — fall back to any non-zero index later.
            return self.__getitem__((idx + 1) % len(self))
        j = random.randint(0, len(xs) - 1)
        pos = np.array([xs[j], ys[j]], dtype=np.float32)

        # Negatives: neighbor centroids inside the crop, except this one.
        in_view = (
            (centers_c[:, 0] >= 0) & (centers_c[:, 0] < c)
            & (centers_c[:, 1] >= 0) & (centers_c[:, 1] < c)
        )
        in_view[ann_idx] = False
        neighbors = centers_c[in_view]
        if len(neighbors):
            d = np.linalg.norm(neighbors - pos, axis=1)
            neighbors = neighbors[d < self.neighbor_radius]
        if len(neighbors) > self.max_negatives:
            idxs = np.random.choice(len(neighbors), self.max_negatives, replace=False)
            neighbors = neighbors[idxs]

        points = np.concatenate([pos[None], neighbors], axis=0) if len(neighbors) else pos[None]
        labels = np.concatenate([[1], np.zeros(len(neighbors), dtype=np.int64)])

        # Pad to fixed slots so we can collate.
        max_pts = 1 + self.max_negatives
        pad = max_pts - len(points)
        if pad > 0:
            points = np.concatenate([points, np.zeros((pad, 2), dtype=np.float32)], axis=0)
            labels = np.concatenate([labels, -np.ones(pad, dtype=np.int64)])  # -1 = ignore in SAM

        enc = self.processor(
            img_c,
            input_points=[[points.tolist()]],
            input_labels=[[labels.tolist()]],
            return_tensors="pt",
        )
        return {
            "pixel_values": enc["pixel_values"][0],
            "input_points": enc["input_points"][0],
            "input_labels": enc["input_labels"][0],
            "labels": torch.from_numpy(target_c.astype(np.float32)),
        }


def sam_collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_points": torch.stack([b["input_points"] for b in batch]),
        "input_labels": torch.stack([b["input_labels"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
