# Training Mask2Former on Kaggle

This folder makes the repo runnable on a Kaggle GPU notebook with no source edits.
`kaggle_train.ipynb` resolves the read-only `/kaggle/input` data, copies the code
to the writable `/kaggle/working`, installs the pinned dependencies, and runs
training + inference.

## 1. Upload the data as a Kaggle Dataset

`data/` (images + annotations) is **gitignored**, so it is not in the GitHub repo —
you must upload it yourself.

1. Zip the local `data/` folder so the archive contains:
   ```
   images/0.jpg, 1.jpg, ...
   annotations/train.json
   annotations/eval.json
   ```
2. Kaggle → **Datasets → New Dataset** → upload the zip. Name it e.g.
   `starmeter-data`. (≈6 MB, uploads in seconds.)

You also need the **code**. Two options:

- **Simplest (no GitHub):** add the whole repo folder to the *same* dataset (so it
  also contains `src/`, `kaggle/`, …). The notebook auto-detects both code and data
  from one dataset.
- **From GitHub:** if the repo is public, replace the "copy code" cell with
  `!git clone https://github.com/AlignedDragon/starMeter` — but you still need the
  data dataset above, since `data/` isn't tracked.

## 2. Create the notebook

1. Kaggle → **Code → New Notebook**.
2. **Add Data** → attach the dataset(s) from step 1.
3. Notebook **Settings**:
   - **Accelerator: GPU** (T4 is enough).
   - **Internet: On** — required so `from_pretrained` can download
     `facebook/mask2former-swin-tiny-coco-instance` the first time.
4. **File → Upload Notebook** → `kaggle/kaggle_train.ipynb` (or paste its cells).
5. **Run All**.

## 3. Configure the run

Edit the **config cell** at the top of the notebook:

| Variable | Default | Notes |
|---|---|---|
| `USE_DENOISING` | `True` | the overlap-handling branch (see `docs/mask_denoising.md`); set `False` for plain Mask2Former |
| `IMG_SIZE` | `512` | fits T4 comfortably; raise to `1024` for accuracy if memory allows |
| `BATCH_SIZE` | `2` | drop to `1` if you hit CUDA OOM |
| `EPOCHS` | `20` | |
| `SMOKE_TEST` | `True` | first runs 1 epoch / 8 samples to prove it works end-to-end |

Keep `SMOKE_TEST = True` for the first run. Once it completes, set it to `False`
and Run All again for the real training.

## 4. Outputs

Everything lands in `/kaggle/working` (downloadable / saved as notebook output):
- `checkpoints/mask2former-nanostar/` — a **plain** Mask2Former checkpoint (the
  training-only denoising params are stripped, so it loads with stock
  `Mask2FormerForUniversalSegmentation`).
- `predictions_m2f.json` — COCO predictions over `eval.json`.

## Notes / troubleshooting

- **transformers is pinned to 4.41.0.** The denoising code (`src/m2f_denoise.py`)
  hooks into the Mask2Former decoder internals validated against that version.
  Plain Mask2Former (`USE_DENOISING = False`) works on newer versions too.
- **CUDA OOM:** lower `IMG_SIZE` (512 → 384) and/or `BATCH_SIZE` (2 → 1).
- **Offline (Internet Off):** add `facebook/mask2former-swin-tiny-coco-instance` as
  a Kaggle dataset/model and set `MODEL = "/kaggle/input/<that>/..."` in the config
  cell.
