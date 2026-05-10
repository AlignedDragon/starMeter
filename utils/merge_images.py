"""Merge data/first_half and data/second_half into a single data/images/ folder.

Defaults to creating symlinks (4096x4096 JPEGs are large); pass --copy to
duplicate the bytes instead.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_SOURCES = (DATA_DIR / "first_half", DATA_DIR / "second_half")


def merge(sources: list[Path], out_dir: Path, copy: bool, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_dir in sources:
        if not src_dir.is_dir():
            raise SystemExit(f"missing {src_dir}")
        for src in sorted(src_dir.glob("*.jpg")):
            dst = out_dir / src.name
            if dst.exists() or dst.is_symlink():
                if not overwrite:
                    continue
                dst.unlink()
            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
            n += 1
    print(f"merged {n} images into {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, nargs="+", default=list(DEFAULT_SOURCES),
                   help="source dirs to merge (defaults to data/first_half data/second_half)")
    p.add_argument("--out", type=Path, default=DATA_DIR / "images")
    p.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    p.add_argument("--overwrite", action="store_true", help="replace existing entries")
    args = p.parse_args()
    merge(args.sources, args.out, args.copy, args.overwrite)


if __name__ == "__main__":
    main()
