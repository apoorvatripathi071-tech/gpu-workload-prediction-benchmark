"""
ARIMA baseline (Section 7.1): fit on the univariate target history alone
(no access to the job-metadata features the neural baselines use -- this
asymmetry is intentional, see Section 7.1's rationale), order selected per
series via grid search minimizing AIC on the training split.

Note: ARIMA here forecasts the next value of the target's own time series
(the sequence of completion times / utilizations across chronologically
ordered jobs), which is a different task framing than the windowed
per-job regression the neural models perform. This is the standard way
ARIMA is used as a classical baseline in this literature (Section 7.1) --
it establishes what a model with NO access to per-job features or
machine-history windowing can achieve, not a fully like-for-like comparison.
"""

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def grid_search_order(train_series: np.ndarray, max_p=3, max_d=1, max_q=3):
    best_aic, best_order = float("inf"), (1, 0, 0)
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            fit = ARIMA(train_series, order=(p, d, q)).fit()
            if fit.aic < best_aic:
                best_aic, best_order = fit.aic, (p, d, q)
        except Exception:
            continue
    return best_order


def evaluate_arima(target: str, philly_path: Path, alibaba_path: Path,
                    max_grid_series_len: int = 2000, max_test_points: int = 300,
                    max_history_len: int = 2000):
    d = np.load(philly_path, allow_pickle=True)
    y_key = "y_completion" if target == "completion" else "y_util"
    series = d[y_key]

    n = len(series)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train, test = series[:n_train], series[n_train + n_val:]

    # Grid search on a capped subset -- ARIMA fitting cost grows steeply
    # with series length, and order selection converges well before using
    # the full training set.
    order = grid_search_order(train[-max_grid_series_len:])
    print(f"[{target}] selected ARIMA order: {order}")

    # Rolling-origin evaluation refits ARIMA once per test point, which
    # does not scale to a full-sized test split (thousands of refits took
    # too long to be practical for a secondary classical baseline). Capped
    # to a contiguous prefix of the test period -- large enough for stable
    # RMSE/MAE/MAPE estimates, small enough to finish quickly. This is a
    # reported scope limitation (Section 9.5), not a silent shortcut.
    if len(test) > max_test_points:
        print(f"[{target}] capping ARIMA test set from {len(test)} to {max_test_points} points "
              f"(rolling-refit does not scale to the full split)")
        test = test[:max_test_points]

    # One-step-ahead forecast walked forward across the (capped) test split,
    # refit incrementally as each true value becomes available (standard
    # rolling-origin evaluation for a classical time-series baseline).
    # `history` is kept as a bounded SLIDING window (most recent
    # max_history_len points), not the full ever-growing training set --
    # at full-trace scale the training set is ~44k points, and refitting
    # ARIMA on the full, still-growing series once per test point made this
    # impractically slow (a low-order ARIMA model's dependence on distant
    # history is minimal anyway, so a bounded recent window is a standard,
    # defensible practical concession -- reported in Section 9.5, not
    # silently applied).
    history = list(train[-max_history_len:])
    preds = []
    for true_val in test:
        model = ARIMA(history, order=order).fit()
        preds.append(model.forecast(1)[0])
        history.append(true_val)
        if len(history) > max_history_len:
            history = history[-max_history_len:]
    preds = np.array(preds)

    rmse = float(np.sqrt(np.mean((preds - test) ** 2)))
    mae = float(np.mean(np.abs(preds - test)))
    nonzero = test != 0
    mape = float(np.mean(np.abs((preds[nonzero] - test[nonzero]) / test[nonzero])) * 100) if nonzero.any() else float("nan")
    in_distribution = {"rmse": rmse, "mae": mae, "mape": mape}

    # Transfer: same fitted order/history is not meaningfully transferable
    # for ARIMA (it has no learned representation to reuse, only its own
    # series' autocorrelation structure) -- reported for completeness by
    # refitting on Alibaba's own series with the SAME order chosen from
    # Philly, which is the closest analogue to "zero-shot" for a classical
    # statistical model: no re-tuning of the order itself.
    ali = np.load(alibaba_path, allow_pickle=True)
    ali_series = ali[y_key]
    n_ali = len(ali_series)
    ali_train, ali_test = ali_series[: int(n_ali * 0.5)], ali_series[int(n_ali * 0.5):]
    if len(ali_test) > max_test_points:
        ali_test = ali_test[:max_test_points]
    ali_history = list(ali_train[-max_history_len:])
    ali_preds = []
    for true_val in ali_test:
        model = ARIMA(ali_history, order=order).fit()
        ali_preds.append(model.forecast(1)[0])
        ali_history.append(true_val)
        if len(ali_history) > max_history_len:
            ali_history = ali_history[-max_history_len:]
    ali_preds = np.array(ali_preds)
    rmse_t = float(np.sqrt(np.mean((ali_preds - ali_test) ** 2)))
    mae_t = float(np.mean(np.abs(ali_preds - ali_test)))
    nz = ali_test != 0
    mape_t = float(np.mean(np.abs((ali_preds[nz] - ali_test[nz]) / ali_test[nz])) * 100) if nz.any() else float("nan")
    transfer = {"rmse": rmse_t, "mae": mae_t, "mape": mape_t}

    return {"order": order, "in_distribution": in_distribution, "transfer": transfer}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["completion", "util"], required=True)
    args = parser.parse_args()

    result = evaluate_arima(
        args.target,
        DATA_DIR / "philly_processed.npz",
        DATA_DIR / "alibaba_processed.npz",
    )
    print(json.dumps(result, indent=2))
    with open(RESULTS_DIR / f"results_arima_{args.target}.json", "w") as f:
        json.dump(result, f, indent=2)
