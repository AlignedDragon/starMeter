"""Mask2Former dataset for the nanostar COCO data.

One sample = one random 1024² crop of a full image, with all instances whose
mask intersects the crop kept as separate instance IDs. Augmentations match
the SAM stage (flip + 4-way rot90); we still skip photometric jitter.

The Mask2FormerImageProcessor is fed the cropped image plus an instance
segmentation map (uint16) where pixel value == instance id (0 = background),
and an `instance_id_to_semantic_id` mapping every instance to class 0
(`nanostar`). The processor returns pixel_values, pixel_mask, mask_labels,
class_labels in the format the model expects.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import CropFlipRot, polygons_to_mask


class Mask2FormerDataset(Dataset):
    def __init__(
        self,
        coco_json: Path,
        images_dir: Path,
        processor,
        crop: int = 1024,
        samples_per_epoch: int | None = None,
        min_pixels: int = 32,
    ):
        with Path(coco_json).open() as f:
            coco = json.load(f)
        self.images = list(coco["images"])
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.aug = CropFlipRot(crop)
        self.crop = crop
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
        W, H = img.size
        anns = self.anns_by_img.get(info["id"], [])
        # (N, H, W) per-instance binary masks at full resolution.
        if anns:
            masks = np.stack(
                [polygons_to_mask(a["segmentation"], H, W) for a in anns], axis=0
            )
        else:
            masks = np.zeros((0, H, W), dtype=np.uint8)

        img_c, _, masks_c = self.aug(img, points=None, masks=masks)

        # Drop instances that no longer have enough pixels in the crop.
        if masks_c is None or len(masks_c) == 0:
            keep = np.zeros((0,), dtype=bool)
        else:
            keep = masks_c.reshape(len(masks_c), -1).sum(axis=1) >= self.min_pixels
        masks_c = masks_c[keep] if len(keep) else np.zeros((0, self.crop, self.crop), dtype=np.uint8)

        if len(masks_c) == 0:
            # Resample if the crop landed on background only.
            return self.__getitem__(_idx)

        # Build instance-segmentation map: 0 = background, 1..N = instance ids.
        seg_map = np.zeros((self.crop, self.crop), dtype=np.uint16)
        instance_id_to_semantic_id: dict[int, int] = {}
        for i, m in enumerate(masks_c, start=1):
            seg_map[m > 0] = i
            instance_id_to_semantic_id[i] = 0  # single class: nanostar

        enc = self.processor(
            images=[img_c],
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
