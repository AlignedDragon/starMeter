"""Fine-tune Mask2Former on the nanostar COCO dataset (single class).

Mask2Former predicts instance masks end-to-end, so unlike the two-stage SAM
pipeline there is no separate centroid model. We start from a COCO-instance
checkpoint and reset the classification head to a single class (`nanostar`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from m2f_dataset import Mask2FormerDataset, m2f_collate

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="facebook/mask2former-swin-tiny-coco-instance")
    ap.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json")
    ap.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    ap.add_argument("--out", type=Path, default=REPO / "checkpoints/mask2former-nanostar")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024,
                    help="downscale the full image to size² (no tiling)")
    ap.add_argument("--samples-per-epoch", type=int, default=600)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Mask2FormerImageProcessor.from_pretrained(args.model)
    # Force the processor to keep images square at the chosen resolution.
    processor.size = {"shortest_edge": args.size, "longest_edge": args.size}
    processor.do_resize = True

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model,
        id2label={0: "nanostar"},
        label2id={"nanostar": 0},
        ignore_mismatched_sizes=True,
    ).to(device)

    ds = Mask2FormerDataset(
        args.coco, args.images_dir, processor,
        size=args.size, samples_per_epoch=args.samples_per_epoch,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=m2f_collate,
        pin_memory=device == "cuda",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                pixel_mask=batch["pixel_mask"].to(device),
                mask_labels=[m.to(device) for m in batch["mask_labels"]],
                class_labels=[c.to(device) for c in batch["class_labels"]],
            )
            loss = out.loss
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
