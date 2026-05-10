"""Stage-2: fine-tune SAM's mask decoder with point prompts.

Each instance is prompted by a positive point inside its mask and up to N
negative points at neighboring centroids — matching what the stage-1 detector
will supply at inference time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from dataset import SamPointDataset, sam_collate

REPO = Path(__file__).resolve().parents[1]


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(-1, -2))
    den = p.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) + 1e-6
    return (1 - num / den).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="facebook/sam-vit-base")
    ap.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json")
    ap.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    ap.add_argument("--out", type=Path, default=REPO / "checkpoints/sam3-nanostar")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--crop", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device)
    for n, p in model.named_parameters():
        p.requires_grad = "mask_decoder" in n

    ds = SamPointDataset(args.coco, args.images_dir, processor, crop=args.crop)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=sam_collate,
        pin_memory=device == "cuda",
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                input_points=batch["input_points"].to(device),
                input_labels=batch["input_labels"].to(device),
                multimask_output=False,
            )
            logits = out.pred_masks.squeeze(1).squeeze(1)
            labels = batch["labels"].to(device)
            logits = F.interpolate(
                logits.unsqueeze(1), size=labels.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, labels) + dice_loss(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch {epoch + 1}: avg loss {total / len(dl):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
