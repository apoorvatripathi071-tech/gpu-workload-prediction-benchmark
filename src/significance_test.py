"""
Statistical significance testing between models (Section 9, journal-tier
rigor requirement -- point-estimate RMSE/MAE comparisons alone are not
sufficient evidence that one model beats another).

Trains each model ONCE per target (same train/val/test split, same seed),
then compares per-sample absolute errors between the Hybrid model and each
baseline using a paired Wilcoxon signed-rank test. Paired because all models
are evaluated on the EXACT SAME test samples (and the same Alibaba transfer
samples), so each model's error on sample i can be directly paired against
another model's error on that same sample i -- this is more statistically
informative than an unpaired test on summary statistics across seeds, and
doesn't require the added training cost of multiple seeds per model.
Wilcoxon (not a paired t-test) is used because per-sample errors on this
data are not normally distributed (heavy-tailed for completion_time,
bounded/skewed for utilization) -- Wilcoxon does not assume normality.

Usage:
    python src/significance_test.py --target util
    python src/significance_test.py --target completion
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from train import run

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def paired_test(errors_a: np.ndarray, errors_b: np.ndarray, label_a: str, label_b: str) -> dict:
    """Wilcoxon signed-rank test on paired per-sample absolute errors.
    Null hypothesis: the two models' error distributions are identical.
    Reports which model has the lower median error alongside the p-value,
    since a significant result alone doesn't say which direction it points."""
    diff = errors_a - errors_b
    diff = diff[diff != 0]  # wilcoxon requires at least one non-zero difference
    if len(diff) < 10:
        return {"comparison": f"{label_a} vs {label_b}", "note": "too few non-tied samples for a reliable test"}
    stat, p = wilcoxon(diff)
    median_a, median_b = float(np.median(errors_a)), float(np.median(errors_b))
    return {
        "comparison": f"{label_a} vs {label_b}",
        "median_abs_error_a": median_a,
        "median_abs_error_b": median_b,
        "lower_median_error": label_a if median_a < median_b else label_b,
        "wilcoxon_statistic": float(stat),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
        "n_paired_samples": int(len(diff)),
    }


def main(target: str, epochs: int, batch_size: int, device: str, seed: int):
    print(f"Training all three models once (target={target}, seed={seed}) to collect paired per-sample errors ...")
    results = run(target, epochs, batch_size, device, seed=seed, save=False, return_arrays=True)

    report = {"target": target, "seed": seed, "in_distribution": [], "transfer": []}
    for split in ("in_distribution", "transfer"):
        cnn_err = np.abs(results["cnn"][split]["_preds"] - results["cnn"][split]["_trues"])
        lstm_err = np.abs(results["lstm"][split]["_preds"] - results["lstm"][split]["_trues"])
        hybrid_err = np.abs(results["hybrid"][split]["_preds"] - results["hybrid"][split]["_trues"])

        report[split].append(paired_test(hybrid_err, cnn_err, "hybrid", "cnn"))
        report[split].append(paired_test(hybrid_err, lstm_err, "hybrid", "lstm"))
        print(f"\n[{split}]")
        for r in report[split]:
            print(f"  {r}")

    out_path = RESULTS_DIR / f"significance_{target}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
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
