"""Check FER2013 and fetch the FER+ labels.

FER2013 cannot be redistributed here. Download ``fer2013.csv`` from the Kaggle
challenge "Challenges in Representation Learning: Facial Expression
Recognition Challenge", place it in ``data/fer2013/`` and run

    python scripts/get_fer_data.py

The script verifies the checksum of ``fer2013.csv`` (the value torchvision
uses) and downloads ``fer2013new.csv`` (FER+ votes) from microsoft/FERPlus.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path

FER_MD5 = "f8428a1edbd21e88f42c73edd2a14f95"
FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
FERPLUS_MD5 = "413eba86d6e454536b99705b8c7fc5c5"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("data/fer2013"))
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    fer = args.dir / "fer2013.csv"
    gz = args.dir / "fer2013.csv.gz"
    if not fer.exists() and gz.exists():
        with gzip.open(gz, "rb") as src, open(fer, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if not fer.exists():
        raise SystemExit(f"{fer} not found; download it from Kaggle first (see the docstring)")
    digest = md5(fer)
    print(f"fer2013.csv md5 {digest}: {'OK' if digest == FER_MD5 else 'MISMATCH (not the original file)'}")
    plus = args.dir / "fer2013new.csv"
    if not plus.exists():
        print(f"downloading {FERPLUS_URL}")
        urllib.request.urlretrieve(FERPLUS_URL, plus)
    digest = md5(plus)
    print(f"fer2013new.csv md5 {digest}: {'OK' if digest == FERPLUS_MD5 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
