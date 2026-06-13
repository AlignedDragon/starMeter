"""Mask2Former dataset for the nanostar COCO data.

One sample = a full image cropped to a bar-free square (no tiling), every
instance kept as a separate instance ID. Nanostars are few (≤20) and large, so
the whole frame fits in one pass; this matches how m2f_pipeline.py runs
inference. Augmentations are flip + 4-way rot90; we skip photometric jitter.

The crop is square (see crop_scale_bar) precisely so rot90 keeps it square and
the processor never has to add orientation-dependent padding — which would leak
the rotation. The square is downscaled isotropically by the processor (no
anisotropic squash). The processor is fed the image plus an instance
segmentation map (uint16, pixel value == instance id, 0 = background) and an
`instance_id_to_semantic_id` mapping every instance to class 0 (`nanostar`), and
returns pixel_values, pixel_mask, mask_labels, class_labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import crop_scale_bar, polygons_to_mask


def _flip_rot(img: Image.Image, masks: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Random h-flip + 4-way rot90 on a (possibly non-square) image and its
    (N, H, W) masks. Uses exact transpose ops so 90° rotation expands cleanly."""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if len(masks):
            masks = masks[:, :, ::-1].copy()
    k = random.randint(0, 3)
    if k:
        img = img.rotate(-90 * k, expand=True)            # clockwise k * 90°
        if len(masks):
            masks = np.rot90(masks, k=-k, axes=(1, 2)).copy()
    return img, masks


class Mask2FormerDataset(Dataset):
    def __init__(
        self,
        coco_json: Path,
        images_dir: Path,
        processor,
        size: int = 1024,
        samples_per_epoch: int | None = None,
        min_pixels: int = 256,
    ):
        with Path(coco_json).open() as f:
            coco = json.load(f)
        self.images = list(coco["images"])
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.size = size
        self.min_pixels = min_pixels
        self.samples_per_epoch = samples_per_epoch or len(self.images)

        self.anns_by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            if a.get("segmentation"):
                self.anns_by_img.setdefault(a["image_id"], []).append(a)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> dict:
        info = random.choice(self.images)
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        img = crop_scale_bar(img)  # drop the scale-bar band before augmentation
        W, H = img.size
        anns = self.anns_by_img.get(info["id"], [])
        # Rasterize at the cropped resolution; polygons in the dropped band clip away.
        if anns:
            masks = np.stack([polygons_to_mask(a["segmentation"], H, W) for a in anns], axis=0)
        else:
            masks = np.zeros((0, H, W), dtype=np.uint8)

        img, masks = _flip_rot(img, masks)

        # Drop instances too small to learn from (e.g. clipped by the band crop).
        if len(masks):
            keep = masks.reshape(len(masks), -1).sum(axis=1) >= self.min_pixels
            masks = masks[keep]
        if len(masks) == 0:
            # Resample if the image had no usable instances.
            return self.__getitem__(_idx)

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
            return_tensors="pt",
        )
        return {
            "pixel_values": enc["pixel_values"][0],
            "pixel_mask": enc["pixel_mask"][0],
            "mask_labels": enc["mask_labels"][0],
            "class_labels": enc["class_labels"][0],
        }


def m2f_collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "pixel_mask": torch.stack([b["pixel_mask"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }
