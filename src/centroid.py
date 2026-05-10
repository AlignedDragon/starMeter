"""Stage-1 centroid detector: small UNet → Gaussian heatmap → peak detection.

  python src/centroid.py train               # writes checkpoints/centroid.pt
  python src/centroid.py infer image.jpg     # prints (x, y) peaks at native res
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CentroidHeatmapDataset

REPO = Path(__file__).resolve().parents[1]


def conv_block(ic: int, oc: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.d1 = conv_block(3, base)
        self.d2 = conv_block(base, base * 2)
        self.d3 = conv_block(base * 2, base * 4)
        self.d4 = conv_block(base * 4, base * 8)
        self.bot = conv_block(base * 8, base * 16)
        self.u4 = conv_block(base * 24, base * 8)
        self.u3 = conv_block(base * 12, base * 4)
        self.u2 = conv_block(base * 6, base * 2)
        self.u1 = conv_block(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.bot(self.pool(d4))
        u4 = self.u4(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), d4], 1))
        u3 = self.u3(torch.cat([F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False), d3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False), d2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False), d1], 1))
        return self.head(u1).squeeze(1)  # (B, H, W)


def peaks_from_heatmap(hm: np.ndarray, thresh: float = 0.3, min_dist: int = 6) -> np.ndarray:
    """Non-max suppression via max-pooling. Returns (N, 2) xy pixel coords."""
    t = torch.from_numpy(hm)[None, None]
    pooled = F.max_pool2d(t, kernel_size=2 * min_dist + 1, stride=1, padding=min_dist)
    keep = (t == pooled) & (t > thresh)
    ys, xs = torch.where(keep[0, 0])
    return np.stack([xs.numpy(), ys.numpy()], axis=1).astype(np.float32)


# --- training ----------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CentroidHeatmapDataset(args.coco, args.images_dir, crop=args.crop, out_size=args.out_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in tqdm(dl, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x = batch["image"].to(device)
            y = batch["heatmap"].to(device)
            pred = model(x)
            # Focal-MSE: weight near-peak pixels.
            w = 1 + 9 * y
            loss = (w * (pred.sigmoid() - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / len(dl):.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "out_size": args.out_size}, args.out)
    print(f"saved -> {args.out}")


# --- inference ---------------------------------------------------------------

def detect(
    model: UNet,
    img: Image.Image,
    device: str,
    tile: int = 1024,
    out_size: int = 512,
    thresh: float = 0.3,
    min_dist: int = 6,
) -> np.ndarray:
    """Tile the full-res image, predict heatmaps, NMS in native coordinates."""
    W, H = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    pad_w = (tile - W % tile) % tile
    pad_h = (tile - H % tile) % tile
    if pad_w or pad_h:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)))
    ph, pw = arr.shape[:2]
    full_hm = np.zeros((ph, pw), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for y0 in range(0, ph, tile):
            for x0 in range(0, pw, tile):
                tile_arr = arr[y0:y0 + tile, x0:x0 + tile]
                t = torch.from_numpy(tile_arr).permute(2, 0, 1)[None].to(device)
                t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
                hm = model(t).sigmoid()[0].cpu().numpy()
                hm = np.asarray(Image.fromarray(hm).resize((tile, tile), Image.BILINEAR))
                full_hm[y0:y0 + tile, x0:x0 + tile] = np.maximum(full_hm[y0:y0 + tile, x0:x0 + tile], hm)
    full_hm = full_hm[:H, :W]
    return peaks_from_heatmap(full_hm, thresh=thresh, min_dist=min_dist)


def cmd_infer(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    img = Image.open(args.image).convert("RGB")
    pts = detect(model, img, device, out_size=ckpt.get("out_size", 512),
                 thresh=args.thresh, min_dist=args.min_dist)
    for x, y in pts:
        print(f"{x:.1f},{y:.1f}")
    print(f"# {len(pts)} peaks", file=sys.stderr)


# --- cli ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--coco", type=Path, default=REPO / "data/annotations/train.json")
    t.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    t.add_argument("--out", type=Path, default=REPO / "checkpoints/centroid.pt")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--crop", type=int, default=1024)
    t.add_argument("--out-size", type=int, default=512)
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer")
    i.add_argument("image", type=Path)
    i.add_argument("--checkpoint", type=Path, default=REPO / "checkpoints/centroid.pt")
    i.add_argument("--thresh", type=float, default=0.3)
    i.add_argument("--min-dist", type=int, default=6)
    i.set_defaults(func=cmd_infer)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
