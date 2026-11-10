"""
Window-length ablation (Section 8/9, external-review requirement): tests
whether window_len=10 is an arbitrary pragmatic choice or a robust one, by
re-preprocessing both traces at several window lengths and comparing the
Hybrid model's performance.

Uses a representative SAMPLE (not the full 62k/521k-row trace) for each
window length, since running full-scale preprocessing + training four times
over is not affordable within this timeline -- this is a standard,
defensible practice for an ablation study and is stated explicitly rather
than silently substituted for the main result (which does use full-scale
data at window_len=10).

Usage:
    python src/window_ablation.py --philly-sample 8000 --alibaba-sample 15000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train import run

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data" / "processed"
RESULTS_DIR = SRC_DIR.parent / "results"

WINDOW_LENS = [5, 10, 15, 20]


def preprocess(window_len: int, philly_sample: int, alibaba_sample: int):
    philly_out = DATA_DIR / f"philly_w{window_len}.npz"
    alibaba_out = DATA_DIR / f"alibaba_w{window_len}.npz"

    if not philly_out.exists():
        subprocess.run([sys.executable, str(SRC_DIR / "preprocess_philly.py"), str(philly_sample),
                         "--window-len", str(window_len), "--out", str(philly_out)], check=True)
    if not alibaba_out.exists():
        subprocess.run([sys.executable, str(SRC_DIR / "preprocess_alibaba.py"), str(alibaba_sample),
                         "--window-len", str(window_len), "--out", str(alibaba_out)], check=True)
    return philly_out, alibaba_out


def main(philly_sample: int, alibaba_sample: int, epochs: int):
    report = {}
    for w in WINDOW_LENS:
        print(f"\n########## window_len={w} ##########")
        philly_path, alibaba_path = preprocess(w, philly_sample, alibaba_sample)
        results = run("util", epochs, batch_size=128, device="cpu", seed=42,
                       philly_path=philly_path, alibaba_path=alibaba_path,
                       save=False, verbose=False, models_to_run=("hybrid",))
        report[w] = results["hybrid"]
        print(f"  in-distribution RMSE={report[w]['in_distribution']['rmse']:.3f}  "
              f"transfer RMSE={report[w]['transfer']['rmse']:.3f}  "
              f"transfer within-2x={report[w]['transfer']['within_2x_pct']:.1f}%")

    out_path = RESULTS_DIR / "window_ablation_util.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--philly-sample", type=int, default=8000)
    parser.add_argument("--alibaba-sample", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    main(args.philly_sample, args.alibaba_sample, args.epochs)
