"""
Trains Ridge and XGBoost on leakage-safe completion-time lag features
(lag_features.py) -- the genuinely zero-shot-compatible target-history
experiment (external-review requirement, second round). Unlike ARIMA
(Section 5.2), these models' PARAMETERS are fit only on Philly; Alibaba
is used only as input at inference time (its own leakage-safe lag
features), never for fitting. This is "source-trained + online
target-history features" -- operationally valid, not label-fitting, but
also not label-free in the sense that Ridge/XGBoost trained on
machine-history features are (those never see ANY Alibaba-derived value,
including as input; these see Alibaba's own observed prior completions
as input features).

Usage:
    python src/train_lag_features.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from preprocess_philly import load_job_log
from preprocess_alibaba import load_task_table
from train import DATA_DIR as PROCESSED_DIR, chronological_split, Scaler

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
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


def load_lag_features_aligned_to_targets(npz_path: Path, target_df, id_col="job_id", duration_col="completion_time_sec"):
    """Aligns the lag-feature matrix (built over ALL valid instances, keyed
    by job_id) to the SAME retained job_id order as the main pipeline's
    processed .npz, so results are directly comparable to Table 5."""
    d = np.load(npz_path, allow_pickle=True)
    lag_ids, X, feature_names = d["job_id"], d["X"], d["feature_names"]

    target_lookup = target_df.set_index(id_col)[duration_col]
    y = target_lookup.loc[lag_ids].to_numpy()
    return lag_ids, X, y, list(feature_names)


def main():
    print("Loading Philly job log and Alibaba task table (for target lookup) ...")
    philly_jobs = load_job_log()
    alibaba_tasks = load_task_table()
    alibaba_tasks["job_id"] = alibaba_tasks["job_name"] + "_" + alibaba_tasks["task_name"]

    philly_ids, X_philly, y_philly, feat_names = load_lag_features_aligned_to_targets(
        PROCESSED_DIR / "philly_lag_features.npz", philly_jobs)
    alibaba_ids, X_alibaba, y_alibaba, _ = load_lag_features_aligned_to_targets(
        PROCESSED_DIR / "alibaba_lag_features.npz", alibaba_tasks)

    print(f"Philly: {len(philly_ids)} instances with lag features, {X_philly.shape[1]} features")
    print(f"Alibaba: {len(alibaba_ids)} instances with lag features")

    # Same chronological split logic as the main pipeline (job_ids are
    # already in end_time-sorted-then-filtered order from lag_features.py's
    # construction, which follows the underlying trace's chronological
    # structure since we iterate the end_time-sorted frame in order).
    n = len(philly_ids)
    train_idx, val_idx, test_idx = chronological_split(n)
    fit_idx = np.concatenate([train_idx, val_idx])

    y_key_log = np.log1p(y_philly)  # same log1p treatment as the main completion-time target
    X_fit, X_test = X_philly[fit_idx], X_philly[test_idx]
    y_fit_log, y_test_raw = y_key_log[fit_idx], y_philly[test_idx]
    y_ali_raw = y_alibaba

    x_scaler = Scaler().fit(X_fit)
    X_fit_s, X_test_s, X_ali_s = x_scaler.transform(X_fit), x_scaler.transform(X_test), x_scaler.transform(X_alibaba)

    def invert(p):
        log_cap = np.log1p(90 * 24 * 60 * 60)
        return np.expm1(np.clip(p, 0, log_cap))

    results = {}

    print("\nTraining Ridge on lag features ...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_fit_s, y_fit_log)
    results["ridge_lag_features"] = {
        "in_distribution": compute_metrics(invert(ridge.predict(X_test_s)), y_test_raw),
        "transfer": compute_metrics(invert(ridge.predict(X_ali_s)), y_ali_raw),
    }

    print("Training XGBoost on lag features ...")
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb.fit(X_fit_s, y_fit_log)
    results["xgboost_lag_features"] = {
        "in_distribution": compute_metrics(invert(xgb.predict(X_test_s)), y_test_raw),
        "transfer": compute_metrics(invert(xgb.predict(X_ali_s)), y_ali_raw),
    }

    for name, r in results.items():
        print(f"\n=== {name} ===")
        print(f"  in-distribution: {r['in_distribution']}")
        print(f"  transfer:        {r['transfer']}")

    out_path = RESULTS_DIR / "results_lag_features_completion.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
