"""
Preprocesses the Microsoft Philly trace into per-job records with:
  - static request-time features (num_gpus requested, cyclical submit time)
  - completion_time_sec target (derived from cluster_job_log)
  - avg_gpu_util target (derived by joining a job's assigned machine/GPU
    against cluster_gpu_util over its active time window)

NOTE ON SCOPE: Philly's public trace has NO per-job GPU memory usage field
(only per-server system RAM and static per-GPU capacity). Memory consumption
is therefore intentionally NOT produced here -- see Section 9.5 (Threats to
Validity) in the paper draft. Only completion_time_sec and avg_gpu_util are
cross-dataset-comparable targets.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "philly-traces" / "trace-data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_job_log() -> pd.DataFrame:
    with open(DATA_DIR / "cluster_job_log", encoding="utf-8") as f:
        jobs = json.load(f)

    records = []
    for job in jobs:
        attempts = job.get("attempts") or []
        if not attempts:
            continue
        # Use the last attempt as the "real" run (earlier attempts may have
        # been preempted/failed reschedules -- matches how Jeon et al. treat
        # multi-attempt jobs in the original ATC'19 analysis).
        last = attempts[-1]
        start_time = last.get("start_time")
        end_time = last.get("end_time")
        if start_time is None or end_time is None:
            continue
        detail = last.get("detail") or []
        if not detail:
            continue
        machine = detail[0].get("ip")
        gpus = detail[0].get("gpus") or []
        if machine is None or not gpus:
            continue

        submitted = job.get("submitted_time")
        if submitted is None:
            continue

        records.append({
            "job_id": job["jobid"],
            "status": job.get("status"),
            "submitted_time": submitted,
            "start_time": start_time,
            "end_time": end_time,
            "machine": machine,
            "gpu_indices": [int(g.replace("gpu", "")) for g in gpus],
            "num_gpus": len(gpus),
        })

    df = pd.DataFrame.from_records(records)
    for col in ("submitted_time", "start_time", "end_time"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["submitted_time", "start_time", "end_time"])

    df["completion_time_sec"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df["queue_delay_sec"] = (df["start_time"] - df["submitted_time"]).dt.total_seconds()
    # Drop clearly corrupt records (negative duration, zero-length, or absurd
    # multi-year durations that indicate a logging error per the trace README).
    df = df[(df["completion_time_sec"] > 0) & (df["completion_time_sec"] < 60 * 60 * 24 * 30)]
    df = df[df["queue_delay_sec"] >= 0]
    return df.reset_index(drop=True)


GPU_UTIL_COLS = ["time", "machineId"] + [f"gpu{i}_util" for i in range(8)]


def load_gpu_util(machines_filter: set[str] | None = None) -> pd.DataFrame:
    """Streams cluster_gpu_util (3GB / 44.7M lines) line by line rather than
    via pandas.read_csv directly.

    Two real problems in the raw file forced this approach:
    1. A subset of lines have more comma-separated fields than the 10-column
       header declares (observed e.g. around line 46983 -- looks like a rare
       logging glitch in the source telemetry, not something documented in
       the trace README). str.split(',', maxsplit=10) sidesteps this by
       simply discarding anything past the first 10 fields, rather than
       failing tokenization like the pandas C parser does.
    2. At 44.7M rows, loading the full file is unnecessary and memory-heavy
       when we only need the machines referenced by the job sample being
       processed -- so filtering happens per line during the read, not after.
    """
    path = DATA_DIR / "cluster_gpu_util"
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split(",", maxsplit=10)
            if len(parts) < 10:
                continue
            machine = parts[1]
            if machines_filter is not None and machine not in machines_filter:
                continue
            rows.append(parts[:10])

    df = pd.DataFrame(rows, columns=GPU_UTIL_COLS)
    # Timestamps carry a trailing tz abbreviation (e.g. "PDT"/"PST") that
    # pandas can't parse directly; both zones are US Pacific (observed only
    # ~1hr apart across DST), and job windows are short relative to that, so
    # the tz suffix is stripped and times are treated as naive/local for the
    # purposes of the completion-time and utilization-window joins.
    df["time"] = pd.to_datetime(df["time"].str.replace(r" P[SD]T$", "", regex=True), errors="coerce")
    for col in [c for c in GPU_UTIL_COLS if c.startswith("gpu")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time"])
    return df


WINDOW_LEN = 10  # matches preprocess_alibaba.py -- required for cross-dataset transfer


def attach_gpu_utilization_windowed(jobs: pd.DataFrame, gpu_util: pd.DataFrame, window_len: int = WINDOW_LEN):
    """For each job, the input window is the last `window_len` per-minute
    utilization readings on the job's assigned MACHINE (averaged across all
    8 GPU slots, i.e. general recent machine load -- not the job's own
    specific GPU(s), since that's what's actually comparable to Alibaba's
    per-machine, not per-GPU-slot, telemetry) observed strictly BEFORE the
    job's start_time. The target is this job's own average utilization
    (on its OWN assigned GPU indices) during its OWN [start_time, end_time]
    window.

    This mirrors preprocess_alibaba.py's "recent machine history predicts
    this new job's outcome" framing exactly, rather than an earlier version
    of this function that used "first N minutes of this same job" as the
    window -- which would have been a DIFFERENT signal definition than
    Alibaba's (Alibaba's pai_machine_metric data doesn't support a
    within-job early-window split at all; see preprocess_alibaba.py's
    module docstring). Since the whole point of Section 9 is testing
    whether a model trained on one cluster generalizes to another, both
    datasets must define "the window" identically -- otherwise a drop in
    transfer accuracy would be confounded with a change in what the input
    even means, not just a change in cluster.
    """
    # Precompute per-machine sorted numpy arrays (times as int64 ns, and both
    # the all-GPU mean used for the history window and the raw 8-column
    # matrix used for the job's own target) ONCE, so each job's lookup is a
    # pair of np.searchsorted calls (O(log n)) instead of re-filtering the
    # full per-machine DataFrame with a boolean mask (O(n)) 111k times --
    # the latter does not scale to Alibaba's 911k-task full run.
    all_gpu_cols = [f"gpu{i}_util" for i in range(8)]
    machine_arrays = {}
    for m, g in gpu_util.groupby("machineId"):
        g = g.sort_values("time")
        machine_arrays[m] = {
            "times": g["time"].to_numpy(dtype="datetime64[ns]"),
            "gpu_matrix": g[all_gpu_cols].to_numpy(dtype=np.float32),
            "mean_all": g[all_gpu_cols].mean(axis=1).to_numpy(dtype=np.float32),
        }

    job_ids, windows, y_util, y_completion, static_rows = [], [], [], [], []
    for row in jobs.itertuples(index=False):
        arrs = machine_arrays.get(row.machine)
        if arrs is None:
            continue

        start_ns = np.datetime64(row.start_time, "ns")
        end_ns = np.datetime64(row.end_time, "ns")
        idx_start = np.searchsorted(arrs["times"], start_ns, side="left")
        if idx_start < window_len:
            continue
        window = arrs["mean_all"][idx_start - window_len: idx_start]
        if np.isnan(window).all():
            continue

        idx_end = np.searchsorted(arrs["times"], end_ns, side="right")
        if idx_end <= idx_start:
            continue
        own_col_idx = [i for i in row.gpu_indices if 0 <= i < 8]
        if not own_col_idx:
            continue
        own_slice = arrs["gpu_matrix"][idx_start:idx_end][:, own_col_idx]
        target_util = np.nanmean(own_slice)
        if np.isnan(target_util):
            continue

        job_ids.append(row.job_id)
        windows.append(np.nan_to_num(window, nan=0.0))
        y_util.append(target_util)
        y_completion.append(row.completion_time_sec)
        static_rows.append([
            row.num_gpus, row.submit_hour_sin, row.submit_hour_cos,
            row.submit_dow_sin, row.submit_dow_cos,
        ])

    return {
        "job_id": np.array(job_ids),
        "X_window": np.nan_to_num(np.vstack(windows), nan=0.0) if windows else np.empty((0, window_len)),
        "X_static": np.array(static_rows, dtype=np.float64),
        "y_completion": np.array(y_completion, dtype=np.float64),
        "y_util": np.array(y_util, dtype=np.float64),
    }


def add_cyclical_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    hour = df[time_col].dt.hour + df[time_col].dt.minute / 60.0
    dow = df[time_col].dt.dayofweek
    df["submit_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["submit_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["submit_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["submit_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def main(sample_for_util: int | None = None, window_len: int = WINDOW_LEN, out_path: Path | None = None):
    print("Loading Philly job log ...")
    jobs = load_job_log()
    print(f"  {len(jobs)} valid jobs after cleaning")

    if sample_for_util:
        # Utilization join is O(jobs) linear scans; useful to sanity-check
        # on a subset first before running the full 111k-job join.
        jobs_for_util = jobs.sample(n=min(sample_for_util, len(jobs)), random_state=42)
    else:
        jobs_for_util = jobs

    jobs_for_util = add_cyclical_time_features(jobs_for_util, "submitted_time")

    machines_needed = set(jobs_for_util["machine"].unique())
    print(f"Loading Philly per-minute GPU utilization for {len(machines_needed)} machines ...")
    gpu_util = load_gpu_util(machines_filter=machines_needed)
    print(f"  {len(gpu_util)} utilization records read")

    print(f"Joining GPU utilization (windowed, window_len={window_len}) for {len(jobs_for_util)} jobs ...")
    data = attach_gpu_utilization_windowed(jobs_for_util, gpu_util, window_len=window_len)
    print(f"  {len(data['job_id'])}/{len(jobs_for_util)} jobs retained "
          f"(dropped: machine has fewer than {window_len} history points before job start, "
          f"or no utilization reading during the job's own window)")

    # Sort by submitted_time so the caller can make a chronological split
    # (Section 6.3) rather than a random one.
    order = jobs_for_util.set_index("job_id").loc[data["job_id"]]["submitted_time"].argsort().to_numpy()
    for key in ("job_id", "X_window", "X_static", "y_completion", "y_util"):
        data[key] = data[key][order]

    out_path = out_path or (OUT_DIR / "philly_processed.npz")
    np.savez(out_path, **data)
    print(f"Wrote {out_path} ({len(data['job_id'])} rows, window_len={window_len})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", nargs="?", type=int, default=None,
                         help="optional job-count cap for quick testing")
    parser.add_argument("--window-len", type=int, default=WINDOW_LEN)
    parser.add_argument("--out", type=str, default=None, help="override output .npz path")
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    main(sample_for_util=args.sample, window_len=args.window_len, out_path=out)
