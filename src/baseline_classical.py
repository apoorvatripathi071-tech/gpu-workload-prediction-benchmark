"""
Classical/tabular baselines (Section 7, expanded following external review):
naive persistence, Ridge regression, and XGBoost -- added specifically to
answer the reviewer question "does the deep hybrid model actually earn its
complexity, or would a much simpler/cheaper model do just as well?" This is
the standard way to justify a deep-learning contribution in an applied
systems paper: show it beats not just weaker deep baselines, but a strong,
cheap, well-understood tabular learner.

All three operate on the SAME flattened (X_window, X_static) features the
neural models use, with the same log1p treatment for completion_time and
the same train/val/test chronological split and Alibaba zero-shot transfer
protocol, so results are directly comparable to results_{target}.json.

Usage:
    python src/baseline_classical.py --target util
    python src/baseline_classical.py --target completion
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from train import DATA_DIR, RESULTS_DIR, load_npz, chronological_split, Scaler


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    """Same metric stack as train.py's evaluate(), duplicated rather than
    imported since evaluate() is coupled to a PyTorch DataLoader and these
    baselines don't use one."""
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae = float(np.mean(np.abs(preds - trues)))
    median_ae = float(np.median(np.abs(preds - trues)))
    nonzero = trues != 0
    mape = float(np.mean(np.abs((preds[nonzero] - trues[nonzero]) / trues[nonzero])) * 100) if nonzero.any() else float("nan")
    preds_nonneg = np.clip(preds, 0, None)
    rmsle = float(np.sqrt(np.mean((np.log1p(preds_nonneg) - np.log1p(trues)) ** 2)))
    ratio = np.maximum(preds, 1e-6) / np.maximum(trues, 1e-6)
    within_2x = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100)
    return {"rmse": rmse, "mae": mae, "median_ae": median_ae, "rmsle": rmsle,
            "mape": mape, "within_2x_pct": within_2x}


def flatten(data: dict, idx: np.ndarray) -> np.ndarray:
    return np.concatenate([data["X_window"][idx], data["X_static"][idx]], axis=1)


def run(target: str, seed: int = 42):
    y_key = "y_completion" if target == "completion" else "y_util"
    log_transform = (target == "completion")

    philly = load_npz(DATA_DIR / "philly_processed.npz")
    alibaba = load_npz(DATA_DIR / "alibaba_processed.npz")
    n = len(philly["job_id"])
    train_idx, val_idx, test_idx = chronological_split(n)
    # Train+val combined for these baselines -- unlike the neural models,
    # none of naive/Ridge/XGBoost use a validation set for early stopping,
    # so folding val into train uses the data that would otherwise sit idle.
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

    # Feature scaling for Ridge (XGBoost is scale-invariant, but scaling
    # doesn't hurt it and keeps the pipeline uniform across both).
    x_scaler = Scaler().fit(X_fit)
    X_fit_s, X_test_s, X_ali_s = x_scaler.transform(X_fit), x_scaler.transform(X_test), x_scaler.transform(X_ali)

    def invert(preds_transformed):
        # Same fix as train.py's evaluate(): Ridge's raw output is not
        # bounded, and under the Alibaba distribution shift a modest
        # overshoot in log-space explodes catastrophically through expm1
        # (observed: RMSE ~1.75e111 before this clip was added). Clip to
        # log1p(90 days), the same physically generous cap used for the
        # neural models, before exponentiating.
        if not log_transform:
            return preds_transformed
        log_cap = np.log1p(90 * 24 * 60 * 60)
        clipped = np.clip(preds_transformed, 0, log_cap)
        return np.expm1(clipped)

    results = {}

    # --- Naive baseline ---
    # For utilization: the last reading in the machine-history window is a
    # natural "persistence" prediction (predict the most recently observed
    # load continues). For completion time, no comparable "last value" of
    # the SAME target exists in the window (the window is a utilization
    # history, not a completion-time history), so the naive baseline here
    # is a constant predictor: the training-set median completion time.
    # This distinction is stated explicitly rather than glossed over.
    if target == "util":
        naive_test = philly["X_window"][test_idx][:, -1]
        naive_ali = alibaba["X_window"][:, -1]
        naive_desc = "last-observed-value persistence"
    else:
        median_completion = float(np.median(philly[y_key][fit_idx]))
        naive_test = np.full(len(test_idx), median_completion)
        naive_ali = np.full(len(alibaba["job_id"]), median_completion)
        naive_desc = "constant (training-set median) predictor"
    results["naive"] = {
        "description": naive_desc,
        "in_distribution": compute_metrics(naive_test, y_test_raw),
        "transfer": compute_metrics(naive_ali, y_ali_raw),
    }

    # --- Ridge regression ---
    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(X_fit_s, y_fit_raw)
    ridge_test = invert(ridge.predict(X_test_s))
    ridge_ali = invert(ridge.predict(X_ali_s))
    results["ridge"] = {
        "in_distribution": compute_metrics(ridge_test, y_test_raw),
        "transfer": compute_metrics(ridge_ali, y_ali_raw),
    }

    # --- XGBoost ---
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=seed,
                        n_jobs=-1)
    xgb.fit(X_fit_s, y_fit_raw)
    xgb_test = invert(xgb.predict(X_test_s))
    xgb_ali = invert(xgb.predict(X_ali_s))
    results["xgboost"] = {
        "in_distribution": compute_metrics(xgb_test, y_test_raw),
        "transfer": compute_metrics(xgb_ali, y_ali_raw),
    }

    for name, r in results.items():
        print(f"\n=== {name} (target={target}) ===")
        print(f"  in-distribution: {r['in_distribution']}")
        print(f"  Alibaba transfer: {r['transfer']}")

    out_path = RESULTS_DIR / f"results_classical_{target}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["completion", "util"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.target, args.seed)
