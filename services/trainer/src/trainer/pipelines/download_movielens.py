from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dst, "wb") as out:
        shutil.copyfileobj(response, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        help="MovieLens zip URL",
    )
    parser.add_argument("--target-dir", default="data/raw", help="Directory where CSV files will be copied")
    parser.add_argument("--tmp-dir", default="/tmp/openrecommender", help="Temporary working directory")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    tmp_dir = Path(args.tmp_dir)
    zip_path = tmp_dir / "movielens.zip"
    extract_dir = tmp_dir / "extract"

    print(f"Downloading dataset from {args.url}")
    download(args.url, zip_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    candidates = list(extract_dir.glob("**/movies.csv"))
    if not candidates:
        raise FileNotFoundError("movies.csv not found in downloaded archive")

    dataset_root = candidates[0].parent
    required = ["movies.csv", "ratings.csv", "tags.csv"]

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = dataset_root / name
        if not src.exists():
            raise FileNotFoundError(f"{name} not found in downloaded archive")
        dst = target_dir / name
        shutil.copy2(src, dst)
        print(f"Wrote {dst}")

    print("MovieLens dataset is ready.")


if __name__ == "__main__":
    main()
