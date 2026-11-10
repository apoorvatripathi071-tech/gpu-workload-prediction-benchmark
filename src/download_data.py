"""
Downloads the two public GPU cluster traces used in this study.

Philly trace (Microsoft):        https://github.com/msr-fiddle/philly-traces
Alibaba PAI GPU trace (v2020):    https://github.com/alibaba/clusterdata

Both are public research datasets, free to download, no login required.
Run this once before running preprocess.py.
"""

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PHILLY_DIR = DATA_DIR / "philly-traces"
ALIBABA_DIR = DATA_DIR / "clusterdata"


def clone_if_missing(url: str, dest: Path, sparse_subdir: str | None = None):
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {dest} already exists")
        return

    if sparse_subdir is None:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
        return

    # Sparse checkout so we don't pull unrelated trace directories from
    # alibaba/clusterdata (the repo hosts many unrelated trace releases).
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                     "--depth", "1", url, str(dest)], check=True)
    subprocess.run(["git", "sparse-checkout", "set", sparse_subdir], check=True, cwd=dest)
    subprocess.run(["git", "checkout"], check=True, cwd=dest)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Downloading Microsoft Philly trace ...")
    clone_if_missing(
        "https://github.com/msr-fiddle/philly-traces.git",
        PHILLY_DIR,
    )

    print("Downloading Alibaba GPU cluster trace (v2020 subdirectory only) ...")
    clone_if_missing(
        "https://github.com/alibaba/clusterdata.git",
        ALIBABA_DIR,
        sparse_subdir="cluster-trace-gpu-v2020",
    )

    print("\nDone. Verify contents:")
    print(f"  {PHILLY_DIR}")
    print(f"  {ALIBABA_DIR / 'cluster-trace-gpu-v2020'}")


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9+ required")
    main()
