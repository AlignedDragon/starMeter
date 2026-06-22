# Training Mask2Former on Kaggle

`kaggle_train.ipynb` is a **self-contained** notebook — all the dataset,
augmentation and training code is inlined, so the only thing you upload is the
data. It trains a plain Mask2Former (swin-tiny) on the two script-free crops with
flip + every-30° rotation augmentation, and reports **val loss + COCO mask AP**
each epoch.

## 1. Upload the data as a Kaggle Dataset

The images are large (407 × 4096² JPEGs, ≈4 GB). Upload them as a Kaggle Dataset
(**Datasets → New Dataset**). The notebook auto-detects the layout under
`/kaggle/input`, but the cleanest structure is:

```
<your-dataset>/
  images/        0.jpg, 1.jpg, ...          # the 407 frames
  annotations/
    train.json                              # COCO instances (366 imgs)
    valid.json                              # COCO instances (41 imgs)
```

`train.json` / `valid.json` live in the repo at `data/annotations/`. The images
are the files in your local dataset folder (the repo's `data/images/` are just
symlinks to them). You can put images and annotations in one dataset or two — the
notebook globs for `train.json`, `valid.json`/`eval.json` and the image folder
wherever they land.

## 2. Create the notebook

1. Kaggle → **Code → New Notebook**.
2. **Add Data** → attach the dataset(s) from step 1.
3. **Settings**:
   - **Accelerator: GPU** — T4 16 GB or P100.
   - **Internet: On** — required so `from_pretrained` can download
     `facebook/mask2former-swin-tiny-coco-instance` the first time.
4. **File → Upload Notebook** → `kaggle/kaggle_train.ipynb` (or paste its cells).
5. **Run All**.

Keep `SMOKE_TEST = True` for the first run (1 epoch, 8 samples, 4 eval frames) to
prove it works end-to-end. Then set it `False` and Run All again for the real run.

## 3. Configure the run

Edit the **config cell** near the top:

| Variable | Default | Notes |
|---|---|---|
| `MODEL` | `facebook/mask2former-swin-tiny-coco-instance` | backbone checkpoint |
| `IMG_SIZE` | `1024` | full crop is downscaled to this square; drop to `768`/`512` on OOM |
| `BATCH_SIZE` / `GRAD_ACCUM` | `1` / `2` | effective batch = product; raise accum (not batch) on 16 GB |
| `EPOCHS` | `20` | |
| `LR` | `5e-5` | AdamW |
| `SAMPLES_PER_EPOCH` | `600` | random crops drawn per epoch (dataset has 2×N distinct) |
| `SCORE_THRESH` | `0.5` | confidence cutoff for AP eval |
| `EVAL_EVERY` | `1` | run validation every N epochs |
| `SMOKE_TEST` | `True` | set `False` for the real run |

## 4. What it does

- **Two crops per image:** `A = [0:3800, 0:3800]`, `B = [900:4096, 900:4096]`.
  Both are square and avoid the burned-in *"100 nm"* scale-bar in the bottom-left
  (bbox ≈ `(46, 3804, 900, 4049)`). Every image is enumerated as **both** crops.
- **Augmentation:** random h-flip + rotation drawn from every 30°. 90° multiples
  are exact transposes; 30/60/120/... rotate about the centre and crop the largest
  inscribed upright square, so the patch never leaves the crop (no black padding).
  Masks rotate with the image and stay binary.
- **Train on crops, validate on full frames.** Validation runs the model on the
  whole 4096² image (matching real inference) and scores COCO segmentation AP; it
  also reports the val loss on the deterministic (un-augmented) crops.

## 5. Outputs

Everything lands in `/kaggle/working` (downloadable from the **Output** tab):

- `checkpoints/mask2former-nanostar/` — a plain Mask2Former checkpoint (model +
  processor), saved **every epoch**. Load it with
  `Mask2FormerForUniversalSegmentation.from_pretrained(...)`, or with the repo's
  `src/m2f_pipeline.py` for full-frame inference.

## Notes / troubleshooting

- **CUDA OOM:** lower `IMG_SIZE` first (1024 → 768 → 512), keep `BATCH_SIZE = 1`
  and raise `GRAD_ACCUM`.
- **Internet Off:** add the backbone as a Kaggle model/dataset and set `MODEL` to
  its local `/kaggle/input/...` path.
- **Denoising** (`src/m2f_denoise.py`) is **not** in `kaggle_train.ipynb` — it hooks
  Mask2Former decoder internals against a pinned `transformers` version. To train with
  denoising on Kaggle use **`kaggle_train_dn.ipynb`** instead (see below).

---

# Training WITH mask-denoising — `kaggle_train_dn.ipynb`

`kaggle_train_dn.ipynb` trains `Mask2FormerDN` (the `--dn` path) and keeps the same
per-epoch **val loss + COCO mask AP** reporting. Two differences from the plain notebook:

1. **Pinned `transformers==4.41.2`.** The denoising decoder reimplements the Mask2Former
   decoder loop against `transformers` 4.41 internals (`decoder.mask_predictor`,
   `tm.queries_features`, `self.criterion`, …); newer releases move/rename those. Cell 2
   installs the exact version and asserts it.
2. **You also upload the repo's `src/` folder** as a second (tiny) Kaggle Dataset. The
   notebook imports `Mask2FormerDN` (`m2f_denoise.py`) and `Mask2FormerDataset`
   (`m2f_dataset.py`, which needs `dataset.py` for `polygons_to_mask`) from it rather than
   inlining — the dn decoder is too entangled with HF internals to inline safely.

So the uploads are **two datasets**: the data (images + annotations, as above) **and** `src/`.
Attach both, set **Accelerator = GPU**, **Internet = On**, upload the notebook, Run All.
Knobs are the same plus `DN_LAMBDA` (default `0.2`, the fraction of GT mask pixels flipped to
build the noised dn masks — matches `m2f_train.py --lambda-p`). The checkpoint saved each epoch
is a **plain** Mask2Former (`export_base()` strips the dn-only params), so inference via
`src/m2f_pipeline.py` is unchanged.
