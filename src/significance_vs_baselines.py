"""
Paired significance testing targeted at the paper's ACTUAL central claim
(external-review requirement): the original significance_test.py only
compared deep models against each other (Hybrid vs CNN, Hybrid vs LSTM) on
a single seed. That doesn't test the paper's headline claim -- "deep models
do not reliably beat simple baselines" -- and seed 42 was independently
shown to be unrepresentative for architecture ranking (Section 7.1.1).

This script tests each deep model against naive persistence and against
XGBoost (both deterministic/near-deterministic, so a single fit each),
across ALL FIVE seeds used for the confidence-interval analysis, and
reports whether the direction and significance of "deep vs. baseline" is
consistent across seeds -- which is what actually matters for a claim that
is supposed to be robust to seed choice.

Usage:
    python src/significance_vs_baselines.py --target util
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from train import run
from baseline_classical import (DATA_DIR, load_npz, chronological_split, Scaler,
                                 flatten, compute_metrics)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEEDS = [42, 43, 44, 45, 46]


def paired_test(errors_deep: np.ndarray, errors_baseline: np.ndarray, deep_name: str, baseline_name: str) -> dict:
    diff = errors_deep - errors_baseline
    diff = diff[diff != 0]
    if len(diff) < 10:
        return {"note": "too few non-tied samples"}
    stat, p = wilcoxon(diff)
    med_deep, med_base = float(np.median(errors_deep)), float(np.median(errors_baseline))
    return {
        "median_abs_error_deep": med_deep,
        "median_abs_error_baseline": med_base,
        "deep_wins": bool(med_deep < med_base),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
        "n": int(len(diff)),
    }


def get_baseline_predictions(target: str, seed: int = 42):
    """Naive and XGBoost predictions on the same test/transfer split used
    throughout -- deterministic (naive) or near-deterministic (XGBoost with
    a fixed seed), so computed once rather than per-seed."""
    y_key = "y_completion" if target == "completion" else "y_util"
    log_transform = (target == "completion")

    philly = load_npz(DATA_DIR / "philly_processed.npz")
    alibaba = load_npz(DATA_DIR / "alibaba_processed.npz")
    n = len(philly["job_id"])
    train_idx, val_idx, test_idx = chronological_split(n)
    fit_idx = np.concatenate([train_idx, val_idx])

    def target_values(data):
        y = data[y_key]
        return np.log1p(y) if log_transform else y

    X_fit = flatten(philly, fit_idx)
    X_test = flatten(philly, test_idx)
    X_ali = flatten(alibaba, np.arange(len(alibaba["job_id"])))
    y_fit_raw = target_values(philly)[fit_idx]
    y_test_raw = philly[y_key][test_idx]
    y_ali_raw = alibaba[y_key]

    x_scaler = Scaler().fit(X_fit)
    X_fit_s, X_test_s, X_ali_s = x_scaler.transform(X_fit), x_scaler.transform(X_test), x_scaler.transform(X_ali)

    def invert(p):
        if not log_transform:
            return p
        log_cap = np.log1p(90 * 24 * 60 * 60)
        return np.expm1(np.clip(p, 0, log_cap))

    # Naive
    if target == "util":
        naive_test = philly["X_window"][test_idx][:, -1]
        naive_ali = alibaba["X_window"][:, -1]
    else:
        median_completion = float(np.median(philly[y_key][fit_idx]))
        naive_test = np.full(len(test_idx), median_completion)
        naive_ali = np.full(len(alibaba["job_id"]), median_completion)

    # XGBoost
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    xgb.fit(X_fit_s, y_fit_raw)
    xgb_test = invert(xgb.predict(X_test_s))
    xgb_ali = invert(xgb.predict(X_ali_s))

    return {
        "in_distribution": {"y_true": y_test_raw, "naive": naive_test, "xgboost": xgb_test},
        "transfer": {"y_true": y_ali_raw, "naive": naive_ali, "xgboost": xgb_ali},
    }


def main(target: str, epochs: int, batch_size: int, device: str):
    print(f"Computing baseline predictions (target={target}) ...")
    baselines = get_baseline_predictions(target)

    report = {"target": target, "seeds": SEEDS, "per_seed": {}}
    for seed in SEEDS:
        print(f"\n########## seed={seed} ##########")
        results = run(target, epochs, batch_size, device, seed=seed, save=False,
                       verbose=False, return_arrays=True)
        seed_report = {}
        for model_name in ("cnn", "lstm", "hybrid"):
            seed_report[model_name] = {}
            for split in ("in_distribution", "transfer"):
                deep_preds = results[model_name][split]["_preds"]
                deep_trues = results[model_name][split]["_trues"]
                deep_err = np.abs(deep_preds - deep_trues)

                split_baselines = baselines[split]
                for baseline_name in ("naive", "xgboost"):
                    base_err = np.abs(split_baselines[baseline_name] - split_baselines["y_true"])
                    key = f"{baseline_name}_{split}"
                    seed_report[model_name][key] = paired_test(deep_err, base_err, model_name, baseline_name)
        report["per_seed"][seed] = seed_report
        for model_name in ("cnn", "lstm", "hybrid"):
            transfer_vs_naive = seed_report[model_name]["naive_transfer"]
            print(f"  {model_name} vs naive (transfer): deep_wins={transfer_vs_naive.get('deep_wins')} "
                  f"p={transfer_vs_naive.get('p_value'):.2e}" if 'p_value' in transfer_vs_naive else
                  f"  {model_name} vs naive (transfer): {transfer_vs_naive}")

    # Cross-seed summary: for each model/baseline/split combo, how many of
    # the 5 seeds show the deep model significantly winning vs. losing.
    summary = {}
    for model_name in ("cnn", "lstm", "hybrid"):
        summary[model_name] = {}
        for baseline_name in ("naive", "xgboost"):
            for split in ("in_distribution", "transfer"):
                key = f"{baseline_name}_{split}"
                per_seed_results = [report["per_seed"][s][model_name][key] for s in SEEDS]
                sig_deep_wins = sum(1 for r in per_seed_results if r.get("significant_at_0.05") and r.get("deep_wins"))
                sig_base_wins = sum(1 for r in per_seed_results if r.get("significant_at_0.05") and not r.get("deep_wins"))
                not_sig = sum(1 for r in per_seed_results if not r.get("significant_at_0.05"))
                summary[model_name][key] = {
                    "seeds_where_deep_significantly_better": sig_deep_wins,
                    "seeds_where_baseline_significantly_better": sig_base_wins,
                    "seeds_not_significant": not_sig,
                }

    report["cross_seed_summary"] = summary
    print("\n=== Cross-seed summary ===")
    print(json.dumps(summary, indent=2))

    out_path = RESULTS_DIR / f"significance_vs_baselines_{target}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["completion", "util"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.target, args.epochs, args.batch_size, args.device)
