"""Upload trained Mask2Former checkpoints to one Hugging Face *model* repo, each
in its own subfolder.

Companion to utils/hf_upload.py (which pushes the *dataset*). Each checkpoint is
the config.json / preprocessor_config.json / model.safetensors trio that
src/m2f_pipeline.py loads with `from_pretrained`. They land under per-variant
subfolders so both live in a single repo:

    kalandarX/starMeter-model
    ├── raw/        (results/checkpoints/mask2former-raw)
    └── denoiser/   (results/checkpoints/mask2former-denoiser)

Load a variant back with:
    Mask2FormerForUniversalSegmentation.from_pretrained(
        "kalandarX/starMeter-model", subfolder="denoiser")

By default the repo is WIPED first (deleted and recreated) so the result is
exactly what is on disk now — no leftovers from earlier botched uploads. Pass
--no-clean to upload on top of whatever is already there.

Auth: set MY_HF_TOKEN (same env var the dataset upload uses), e.g.
    export MY_HF_TOKEN=hf_...

Examples:
    python utils/hf_upload_model.py                       # wipe + upload raw/ and denoiser/
    python utils/hf_upload_model.py --no-clean            # upload without wiping
    python utils/hf_upload_model.py \
        --variant raw=results/checkpoints/mask2former-raw \
        --variant denoiser=results/checkpoints/mask2former-denoiser
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[1]
CKPTS = REPO / "results" / "checkpoints"
DEFAULT_VARIANTS = {
    "raw": CKPTS / "mask2former-raw",
    "denoiser": CKPTS / "mask2former-denoiser",
}


def _parse_variant(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--variant must be name=path, got {s!r}")
    name, path = s.split("=", 1)
    return name.strip(), Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="kalandarX/starMeter-model",
                    help="target Hub model repo (namespace/name)")
    ap.add_argument("--variant", type=_parse_variant, action="append",
                    metavar="NAME=PATH",
                    help="subfolder=local-checkpoint-dir; repeatable. "
                         "Defaults to raw + denoiser.")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="do NOT wipe the repo first (default wipes it)")
    ap.add_argument("--commit-message", default="upload checkpoints")
    args = ap.parse_args()

    variants = dict(args.variant) if args.variant else DEFAULT_VARIANTS

    # Validate every local folder BEFORE touching the remote, so a wipe is never
    # followed by a failed upload that leaves the repo empty.
    for name, path in variants.items():
        if not path.is_dir():
            raise SystemExit(f"checkpoint folder not found: {name} -> {path}")
        missing = [f for f in ("config.json", "model.safetensors")
                   if not (path / f).exists()]
        if missing:
            raise SystemExit(f"{path} is missing {missing}")

    token = os.environ.get("MY_HF_TOKEN")
    if not token:
        raise SystemExit("set MY_HF_TOKEN in the environment first")
    api = HfApi(token=token)

    if args.clean:
        # Wipe = delete the whole repo and recreate it, for a guaranteed clean slate.
        api.delete_repo(repo_id=args.repo_id, repo_type="model", missing_ok=True)
        print(f"wiped {args.repo_id}")
    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    for name, path in variants.items():
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(path),
            path_in_repo=name,
            commit_message=f"{args.commit_message}: {name}",
        )
        print(f"uploaded {path} -> {args.repo_id}/{name}")

    print(f"done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
