"""
Leakage-safe completion-time lag features (external-review requirement,
second round): tests whether the completion-time bottleneck genuinely is
feature representation, using a fully zero-shot-compatible experiment
(unlike ARIMA, Section 5.2, which is deliberately excluded from the
zero-shot claim because it is fit on real target-domain history).

CRITICAL DISTINCTION FROM A NAIVE LAG-FEATURE IMPLEMENTATION: for a focal
instance i with start time s_i, a "prior completed job's duration" is only
OPERATIONALLY KNOWABLE if that prior job j actually FINISHED before i
started -- i.e. end_time_j < s_i -- not merely if j STARTED before i
(start_time_j < s_i) or appears earlier by index. A job that started
before i but is still RUNNING when i starts has an unknown duration at
that moment, even though it is "in the past" by submission order. Using
start-time or index ordering instead of end_time_j < s_i would silently
leak not-yet-observed durations into the feature set. This is a distinct,
easy-to-miss leakage mode from the ones already handled in Section 4 (which
guards against using FUTURE machine telemetry, not against using a prior
job's duration before that job has actually finished).

For each retained instance i (the SAME instance set used throughout this
paper, from philly_processed.npz / alibaba_processed.npz, so results are
directly comparable to Table 5), computes:
  - last K completed durations (log1p) among jobs/tasks with
    end_time_j < start_time_i, most-recent-first, imputed with the
    training-set median log1p(duration) where fewer than K are available
  - rolling mean/median/std/min/max of the AVAILABLE (non-imputed) lags
  - a count of how many genuine priors were available (0..K), so the model
    can learn to trust the lag features less when count is low

Usage:
    python src/lag_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_philly import load_job_log
from preprocess_alibaba import load_task_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
K = 10  # matches the main window length for consistency, not a magic number


def compute_lag_features(df: pd.DataFrame, start_col: str, end_col: str,
                          duration_col: str, retained_ids: set, id_col: str,
                          k: int = K) -> pd.DataFrame:
    """df must contain every valid instance (not just retained ones) --
    prior completions can come from instances that were themselves dropped
    from the main windowed dataset, since all we need from them is a valid
    (start_time, end_time, duration), not a full machine-history window.

    Fully vectorized (no per-instance Python loop): for m retained instances
    we build one (m, k) matrix of lag values via a single gather, using
    n_prior[idx] (rank by START time among all END times) as the boundary --
    NOT idx itself (the instance's own rank by END time). These two differ
    whenever any instance finishes DURING the focal instance's own execution
    window (end_time in (start_time_i, end_time_i)) -- such instances rank
    below idx by end_time but are not yet completed at start_time_i, so
    slicing up to idx (an earlier, buggy version of this function did this)
    silently leaks their durations. Slicing up to n_prior[idx] is the fix."""
    df = df.dropna(subset=[start_col, end_col, duration_col]).copy()
    df = df.sort_values(end_col).reset_index(drop=True)

    end_times = df[end_col].to_numpy()
    log_durations = np.log1p(df[duration_col].to_numpy().astype(np.float64))

    # For each instance, find how many OTHER instances have end_time strictly
    # before this instance's start_time -- searchsorted on the array sorted
    # by end_time, exactly analogous to the searchsorted optimization used
    # for the machine-history windows in Section 4, but keyed on completion
    # (end_time), not on raw telemetry timestamps.
    start_times = df[start_col].to_numpy()
    n_prior = np.searchsorted(end_times, start_times, side="left")

    train_median_log_duration = float(np.median(log_durations[: int(len(log_durations) * 0.70)]))

    ids = df[id_col].to_numpy()
    # Hash-based membership via pandas. np.isin must NOT be used here: on
    # object/string arrays numpy falls back to a brute-force O(n_all x
    # n_retained) scan (911K x 521K comparisons for Alibaba), measured at
    # multiple HOURS at this scale versus under a second for a hash lookup.
    is_retained = df[id_col].isin(retained_ids).to_numpy()
    retained_idx = np.where(is_retained)[0]

    base = n_prior[retained_idx]                       # (m,) -- correct boundary (by start_time rank)
    k_avail = np.minimum(base, k)                       # (m,)

    j = np.arange(k)[None, :]                            # (1, k)
    positions = base[:, None] - 1 - j                     # (m, k); col 0 = most recent prior
    valid = j < k_avail[:, None]                          # (m, k)
    safe_positions = np.clip(positions, 0, len(log_durations) - 1)
    gathered = log_durations[safe_positions]              # (m, k)
    lags = np.where(valid, gathered, train_median_log_duration)  # (m, k)

    # Rolling stats without np.nan* functions: nanmedian on a (m, k) array
    # with NaNs takes numpy's apply_along_axis path -- a per-row Python loop.
    # Sum/sort-based equivalents below are fully vectorized.
    has_any = k_avail > 0
    counts = np.maximum(k_avail, 1).astype(np.float64)
    sums = np.where(valid, gathered, 0.0).sum(axis=1)
    roll_mean = sums / counts
    sq_sums = np.where(valid, gathered ** 2, 0.0).sum(axis=1)
    roll_std = np.sqrt(np.maximum(sq_sums / counts - roll_mean ** 2, 0.0))
    sorted_vals = np.sort(np.where(valid, gathered, np.inf), axis=1)
    rows_i = np.arange(len(retained_idx))
    lo = np.clip((k_avail - 1) // 2, 0, k - 1)
    hi = np.clip(k_avail // 2, 0, k - 1)
    roll_median = (sorted_vals[rows_i, lo] + sorted_vals[rows_i, hi]) / 2
    roll_min = np.where(valid, gathered, np.inf).min(axis=1)
    roll_max = np.where(valid, gathered, -np.inf).max(axis=1)

    roll_mean = np.where(has_any, roll_mean, train_median_log_duration)
    roll_median = np.where(has_any, roll_median, train_median_log_duration)
    roll_std = np.where(has_any & (k_avail > 1), roll_std, 0.0)
    roll_min = np.where(has_any, roll_min, train_median_log_duration)
    roll_max = np.where(has_any, roll_max, train_median_log_duration)

    out = {id_col: ids[retained_idx]}
    for col in range(k):
        out[f"lag_{col}"] = lags[:, col]
    out["roll_mean"] = roll_mean
    out["roll_median"] = roll_median
    out["roll_std"] = roll_std
    out["roll_min"] = roll_min
    out["roll_max"] = roll_max
    out["n_prior_available"] = k_avail
    return pd.DataFrame(out)


def build_philly_lag_features() -> pd.DataFrame:
    jobs = load_job_log()
    retained_ids = set(np.load(DATA_DIR / "philly_processed.npz", allow_pickle=True)["job_id"].tolist())
    return compute_lag_features(jobs, "start_time", "end_time", "completion_time_sec",
                                 retained_ids, "job_id")


def build_alibaba_lag_features() -> pd.DataFrame:
    tasks = load_task_table()
    tasks["job_id"] = tasks["job_name"] + "_" + tasks["task_name"]
    retained_ids = set(np.load(DATA_DIR / "alibaba_processed.npz", allow_pickle=True)["job_id"].tolist())
    # Alibaba start/end are already relative-second floats (Section 3.2 caveat), which
    # is exactly what searchsorted needs -- no unit conversion required.
    tasks["end_time"] = tasks["start_time"] + tasks["completion_time_sec"]
    return compute_lag_features(tasks, "start_time", "end_time", "completion_time_sec",
                                 retained_ids, "job_id")


def save_npz(df: pd.DataFrame, path: Path):
    feature_cols = [c for c in df.columns if c != "job_id"]
    np.savez(path, job_id=df["job_id"].to_numpy(),
             X=df[feature_cols].to_numpy(dtype=np.float64),
             feature_names=np.array(feature_cols))


if __name__ == "__main__":
    print("Building Philly lag features ...")
    philly_lags = build_philly_lag_features()
    print(f"  {len(philly_lags)} retained instances with lag features")
    print(f"  n_prior_available distribution: min={philly_lags['n_prior_available'].min()}, "
          f"median={philly_lags['n_prior_available'].median()}, max={philly_lags['n_prior_available'].max()}")
    save_npz(philly_lags, DATA_DIR / "philly_lag_features.npz")

    print("\nBuilding Alibaba lag features ...")
    alibaba_lags = build_alibaba_lag_features()
    print(f"  {len(alibaba_lags)} retained instances with lag features")
    print(f"  n_prior_available distribution: min={alibaba_lags['n_prior_available'].min()}, "
          f"median={alibaba_lags['n_prior_available'].median()}, max={alibaba_lags['n_prior_available'].max()}")
    save_npz(alibaba_lags, DATA_DIR / "alibaba_lag_features.npz")

    print(f"\nWrote {DATA_DIR / 'philly_lag_features.npz'} and {DATA_DIR / 'alibaba_lag_features.npz'}")
