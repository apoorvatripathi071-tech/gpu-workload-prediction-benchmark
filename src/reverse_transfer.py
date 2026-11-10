"""
Transfer-direction analysis (Section 9, external-review requirement): the
main results train on Philly and zero-shot-evaluate on Alibaba. This script
reverses the direction -- train on Alibaba, zero-shot-evaluate on Philly --
to test whether transfer is symmetric or whether one direction is
structurally easier (e.g. because one trace has a narrower or more
representative distribution than the other).

train.py's run() takes philly_path/alibaba_path as plain paths; passing them
swapped makes it train on "alibaba_path" data and transfer-evaluate on
"philly_path" data, reusing all the same leakage-safe splitting, scaling,
and log-transform logic without duplicating it.

Usage:
    python src/reverse_transfer.py --target util
    python src/reverse_transfer.py --target completion
"""

import argparse
import json
from pathlib import Path

from train import run, DATA_DIR, RESULTS_DIR


def main(target: str, epochs: int, batch_size: int, device: str, seed: int):
    print(f"Training on ALIBABA, zero-shot transfer to PHILLY (target={target}) ...")
    # Swapped on purpose: run()'s "philly_path" argument is just "the
    # dataset trained/split on"; passing Alibaba's file there makes Alibaba
    # the source domain and Philly the zero-shot transfer target.
    results = run(
        target, epochs, batch_size, device, seed=seed, save=False,
        philly_path=DATA_DIR / "alibaba_processed.npz",
        alibaba_path=DATA_DIR / "philly_processed.npz",
    )

    for name, r in results.items():
        print(f"\n=== {name} (source=Alibaba, transfer-target=Philly) ===")
        print(f"  in-distribution (Alibaba): {r['in_distribution']}")
        print(f"  transfer (Philly):         {r['transfer']}")

    out_path = RESULTS_DIR / f"reverse_transfer_{target}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["completion", "util"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.target, args.epochs, args.batch_size, args.device, args.seed)
