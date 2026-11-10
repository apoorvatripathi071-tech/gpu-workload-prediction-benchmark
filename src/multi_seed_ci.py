"""
Multi-seed runs + bootstrap 95% confidence intervals (Section 9, journal-tier
rigor requirement -- point-estimate RMSE comparisons alone don't show
whether a difference is real or noise, and this matters especially because
in-distribution utilization RMSE is nearly tied across CNN/LSTM/Hybrid).

Trains each model N_SEEDS times (different weight initialization/data
shuffling, same architecture and data), collects the RMSE distribution
across seeds, and reports mean +/- 95% bootstrap CI for both in-distribution
and Alibaba transfer.

Usage:
    python src/multi_seed_ci.py --target util --seeds 5
    python src/multi_seed_ci.py --target completion --seeds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np

from train import run

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lower = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return {"mean": float(values.mean()), "std": float(values.std()),
            "ci_95_lower": lower, "ci_95_upper": upper, "n_seeds": len(values)}


def main(target: str, n_seeds: int, epochs: int, batch_size: int, device: str):
    seeds = list(range(42, 42 + n_seeds))
    per_model_metrics = {name: {"in_distribution": {"rmse": [], "within_2x_pct": []},
                                 "transfer": {"rmse": [], "within_2x_pct": []}}
                          for name in ("cnn", "lstm", "hybrid")}

    for seed in seeds:
        print(f"\n########## seed={seed} ##########")
        results = run(target, epochs, batch_size, device, seed=seed, save=False, verbose=False)
        for name in ("cnn", "lstm", "hybrid"):
            for split in ("in_distribution", "transfer"):
                per_model_metrics[name][split]["rmse"].append(results[name][split]["rmse"])
                per_model_metrics[name][split]["within_2x_pct"].append(results[name][split]["within_2x_pct"])
        print(f"  done: " + ", ".join(
            f"{name}={per_model_metrics[name]['transfer']['rmse'][-1]:.2f}" for name in ("cnn", "lstm", "hybrid")))

    report = {"target": target, "seeds": seeds}
    for name in ("cnn", "lstm", "hybrid"):
        report[name] = {}
        for split in ("in_distribution", "transfer"):
            report[name][split] = {
                "rmse": bootstrap_ci(np.array(per_model_metrics[name][split]["rmse"])),
                "within_2x_pct": bootstrap_ci(np.array(per_model_metrics[name][split]["within_2x_pct"])),
            }

    print(f"\n=== Summary (target={target}, {n_seeds} seeds) ===")
    for name in ("cnn", "lstm", "hybrid"):
        for split in ("in_distribution", "transfer"):
            r = report[name][split]["rmse"]
            print(f"  {name:8s} {split:16s} RMSE = {r['mean']:.3f} "
                  f"(95% CI [{r['ci_95_lower']:.3f}, {r['ci_95_upper']:.3f}], n={r['n_seeds']})")

    out_path = RESULTS_DIR / f"multiseed_ci_{target}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["completion", "util"], required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.target, args.seeds, args.epochs, args.batch_size, args.device)
