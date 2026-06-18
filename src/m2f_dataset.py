"""Mask2Former dataset for the nanostar COCO data.

One sample = one of two script-free square crops of a source frame (no tiling),
every instance kept as a separate instance ID. Nanostars are few (≤20) and
large, so each crop fits in one pass. Augmentations are flip + rotation at every
30° (see ROT_ANGLES); we skip photometric jitter.

Both crops are SQUARE so rot90 keeps them square and the processor never has to
add orientation-dependent padding — which would leak the rotation. Multiples of
90° are exact transposes; the 30/60/120/... angles are done as a rotate-then-
inscribe (rotate about the centre, crop the largest upright square that still
fits) so the patch never leaves the original boundary — no black corners. The
image is resampled (bilinear) for those angles; masks rotate nearest-neighbour
so they stay binary and registered. The two crops (see
BAR_FREE_CROPS) both avoid the burned-in "100 nm" scale bar in the bottom-left
(bbox ≈ (46, 3804, 900, 4049) on 4096² frames) while between them recovering
almost the whole frame: A is the top-left square above the bar, B the
bottom-right square to its right; they overlap in the center, so a centered star
is seen in both. Every source image is enumerated as BOTH crops.

The square is downscaled isotropically by the processor (no anisotropic squash).
The processor is fed the image plus an instance segmentation map (uint16, pixel
value == instance id, 0 = background) and an `instance_id_to_semantic_id`
mapping every instance to class 0 (`nanostar`), and returns pixel_values,
pixel_mask, mask_labels, class_labels.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import polygons_to_mask

# Two script-free square crops for 4096² frames. The scale-bar bbox is
# ≈ (46, 3804, 900, 4049); A's bottom edge sits above its top, B's left edge at
# its right, so neither contains any burned-in text. (x0, y0, x1, y1).
BAR_FREE_CROPS = ((0, 0, 3800, 3800), (900, 900, 4096, 4096))


def _crop_polys(seg: list[list[float]], x0: int, y0: int) -> list[list[float]]:
    """Shift COCO polygons into a crop's local frame (origin -> x0, y0). Points
    outside the crop are kept; polygons_to_mask rasterizes them clipped."""
    return [[c - (x0 if i % 2 == 0 else y0) for i, c in enumerate(poly)] for poly in seg]


# Rotation angles sampled per crop. Multiples of 90° are exact transposes; the
# 30/60/120/... offsets are done as a rotate-then-inscribe so the patch never
# leaves the original square (no black padding, no edge fill).
ROT_ANGLES = tuple(range(0, 360, 30))  # 0,30,...,330


def _inscribed_side(angle: float, side: int) -> int:
    """Largest upright square that fits inside ``side``×``side`` after rotating by
    ``angle``: side / (|cos|+|sin|). The centered crop of this size is filled
    entirely by original pixels, so no padding is ever introduced."""
    t = math.radians(angle)
    return int(round(side / (abs(math.cos(t)) + abs(math.sin(t)))))


def _flip_rot(img: Image.Image, masks: np.ndarray, side: int) -> tuple[Image.Image, np.ndarray]:
    """Random h-flip + rotation on a square ``side``×``side`` crop and its
    (N, side, side) masks.

    Rotation angle is drawn from ROT_ANGLES. Multiples of 90° are exact
    transposes (lossless). For 30/60/120/... we rotate about the centre and crop
    the largest inscribed upright square, so the patch stays inside the original
    boundary — no black corners. Image and masks share one affine matrix so they
    stay registered; masks rotate with nearest-neighbour to stay binary.
    """
    arr = np.ascontiguousarray(np.asarray(img))

    if random.random() < 0.5:
        arr = np.ascontiguousarray(arr[:, ::-1])
        if len(masks):
            masks = np.ascontiguousarray(masks[:, :, ::-1])

    angle = random.choice(ROT_ANGLES)
    if angle % 90 == 0:
        k = angle // 90                                   # exact quarter turns
        if k:
            arr = np.ascontiguousarray(np.rot90(arr, k=k, axes=(0, 1)))
            if len(masks):
                masks = np.ascontiguousarray(np.rot90(masks, k=k, axes=(1, 2)))
    else:
        c = side / 2.0
        M = cv2.getRotationMatrix2D((c, c), angle, 1.0)
        arr = cv2.warpAffine(arr, M, (side, side), flags=cv2.INTER_LINEAR)
        if len(masks):
            masks = np.stack(
                [cv2.warpAffine(m, M, (side, side), flags=cv2.INTER_NEAREST) for m in masks],
                axis=0,
            )
        s = _inscribed_side(angle, side)
        o = (side - s) // 2                               # centred inscribed crop
        arr = np.ascontiguousarray(arr[o:o + s, o:o + s])
        if len(masks):
            masks = np.ascontiguousarray(masks[:, o:o + s, o:o + s])

    return Image.fromarray(arr), masks


class Mask2FormerDataset(Dataset):
    def __init__(
        self,
        coco_json: Path,
        images_dir: Path,
        processor,
        size: int = 1024,
        samples_per_epoch: int | None = None,
        min_pixels: int = 256,
        dn_lambda: float = 0.0,
    ):
        with Path(coco_json).open() as f:
            coco = json.load(f)
        self.images = list(coco["images"])
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.size = size
        self.min_pixels = min_pixels
        self.samples_per_epoch = samples_per_epoch or len(self.images)
        # >0 enables mask-denoising targets: a noised copy of every GT mask with
        # `dn_lambda` of its pixels flipped (see docs/mask_denoising.md). Aligned
        # 1:1 with the processor's mask_labels / class_labels.
        self.dn_lambda = dn_lambda

        self.anns_by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            if a.get("segmentation"):
                self.anns_by_img.setdefault(a["image_id"], []).append(a)

        # Enumerate every source image as BOTH script-free crops.
        self.samples = [(info, box) for info in self.images for box in BAR_FREE_CROPS]
        if samples_per_epoch is None:
            self.samples_per_epoch = len(self.samples)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        info, (x0, y0, x1, y1) = self.samples[idx % len(self.samples)]
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        img = img.crop((x0, y0, x1, y1))  # script-free square; rot90 stays exact
        W, H = img.size  # square: W == H == crop side
        anns = self.anns_by_img.get(info["id"], [])
        # Rasterize in the crop's local frame; instances outside it clip away.
        if anns:
            masks = np.stack(
                [polygons_to_mask(_crop_polys(a["segmentation"], x0, y0), H, W) for a in anns],
                axis=0,
            )
        else:
            masks = np.zeros((0, H, W), dtype=np.uint8)

        img, masks = _flip_rot(img, masks, side=W)

        # Drop instances too small to learn from (e.g. clipped by the crop edge).
        if len(masks):
            keep = masks.reshape(len(masks), -1).sum(axis=1) >= self.min_pixels
            masks = masks[keep]
        if len(masks) == 0:
            # This crop had no usable instances; draw another at random (a fixed
            # idx would recurse forever on an empty crop).
            return self.__getitem__(random.randrange(len(self.samples)))

        # Build instance-segmentation map: 0 = background, 1..N = instance ids.
        Hc, Wc = masks.shape[1:]
        seg_map = np.zeros((Hc, Wc), dtype=np.uint16)
        instance_id_to_semantic_id: dict[int, int] = {}
        for i, m in enumerate(masks, start=1):
            seg_map[m > 0] = i
            instance_id_to_semantic_id[i] = 0  # single class: nanostar

        enc = self.processor(
            images=[img],
            segmentation_maps=[seg_map],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=0,  # 0 = background; without this the processor looks up
            return_tensors="pt",  # instance_id_to_semantic_id[0] and raises KeyError
        )
        out = {
            "pixel_values": enc["pixel_values"][0],
            "pixel_mask": enc["pixel_mask"][0],
            "mask_labels": enc["mask_labels"][0],
            "class_labels": enc["class_labels"][0],
        }
        if self.dn_lambda > 0:
            # Noise the SAME masks the loss uses (processor output), so the noised
            # copy stays aligned with mask_labels / class_labels in order and
            # resolution. XOR flips ~dn_lambda of pixels in both directions.
            gt = out["mask_labels"] > 0.5
            flip = torch.rand_like(out["mask_labels"]) < self.dn_lambda
            out["dn_masks"] = (gt ^ flip).float()
        return out


def m2f_collate(batch: list[dict]) -> dict:
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "pixel_mask": torch.stack([b["pixel_mask"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }
    if "dn_masks" in batch[0]:
        out["dn_masks"] = [b["dn_masks"] for b in batch]
    return out
