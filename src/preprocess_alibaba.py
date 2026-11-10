"""
Preprocesses the Alibaba PAI GPU cluster trace (v2020) into per-task records
harmonized to the SAME schema as preprocess_philly.py's output:
  job_id, X_window (window_len,), X_static (5,), y_completion, y_util

WINDOWING DESIGN (revised from an earlier version -- see note below):
For each task, the input window is the last `window_len` GPU-utilization
readings observed on that task's MACHINE from OTHER tasks/workers that ran
on it before this task started; the target is this task's OWN completion
time and average utilization. This is a "recent machine load history
predicts this new job's outcome" framing, and is applied identically in
preprocess_philly.py (there, using the job's own machine's per-minute
history) so the two datasets are structurally comparable for the
cross-dataset transfer test in Section 9.

WHY NOT AN "EARLY MINUTES OF THIS TASK" WINDOW (what an earlier version of
this script attempted, mirroring a first draft of the Philly script):
pai_machine_metric turned out to log roughly ONE aggregated metric snapshot
per worker's entire lifetime, not continuous per-minute telemetry the way
Philly's cluster_gpu_util does -- confirmed empirically (2,834 metric rows
across 5,000 sampled workers, i.e. much less than one row per worker on
average). There is no "first 10 minutes of this task" sequence to extract.
Recent-machine-history is the framing the real data actually supports.

CAVEATS (see paper Section 9.5, Threats to Validity):
- Alibaba timestamps are relative seconds since trace start, not absolute
  wall-clock time, so submit_hour/dow cyclical features are only an
  APPROXIMATION (relative_seconds % 86400 / % (86400*7)), assuming trace
  start aligns with a day/week boundary.
- plan_mem/plan_gpu in pai_task_table are REQUESTED/PLANNED allocations,
  not measured usage -- not used as the memory-consumption target (Philly
  cannot supply a comparable label at all; see preprocess_philly.py).
- machine_gpu in pai_machine_metric is a SUM across all physical GPUs on
  that machine (observed max ~787, consistent with up to 8 GPUs near 100%
  each). It is divided by the machine's actual physical GPU count (from
  pai_machine_spec's cap_gpu field; verified machines in this trace have
  either 2 or 8 GPUs) to bring it onto the same 0-100% per-GPU scale as
  Philly's gpu{i}_util columns. An earlier version normalized by the
  task's requested GPU count instead of the machine's physical capacity,
  which produced values above 100% -- caught by sanity-checking output
  ranges before trusting the pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "clusterdata" / "cluster-trace-gpu-v2020" / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_COLS = ["job_name", "task_name", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
INSTANCE_COLS = ["job_name", "task_name", "inst_name", "worker_name", "inst_id",
                  "status", "start_time", "end_time", "machine"]
METRIC_COLS = ["worker_name", "machine", "start_time", "end_time",
               "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
               "machine_gpu", "machine_load_1", "machine_net_receive",
               "machine_num_worker", "machine_cpu"]

WINDOW_LEN = 10  # matches preprocess_philly.py -- required for cross-dataset transfer


def load_task_table() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "pai_task_table.csv", names=TASK_COLS, header=None,
                      na_values=[""])
    df = df.dropna(subset=["start_time", "end_time"])
    df["completion_time_sec"] = df["end_time"] - df["start_time"]
    df = df[(df["completion_time_sec"] > 0) & (df["completion_time_sec"] < 60 * 60 * 24 * 30)]
    df["num_gpus"] = pd.to_numeric(df["inst_num"], errors="coerce").fillna(1).clip(lower=1)
    return df.reset_index(drop=True)


def load_instances_for_machines(machines_needed: set[str] | None, task_keys: pd.DataFrame | None) -> pd.DataFrame:
    """Reads instance_table rows either filtered by a specific set of
    machines (broad scan, used to build machine-level history) or by
    specific (job_name, task_name) keys (narrow scan, used to find each
    sampled task's own machine). Exactly one of the two filters should be
    provided. Chunked to avoid materializing the ~2GB instance table."""
    key_df = task_keys[["job_name", "task_name"]].drop_duplicates() if task_keys is not None else None
    chunks = []
    for chunk in pd.read_csv(DATA_DIR / "pai_instance_table.csv", names=INSTANCE_COLS,
                              header=None, usecols=[0, 1, 3, 6, 8], chunksize=500_000):
        if key_df is not None:
            filtered = chunk.merge(key_df, on=["job_name", "task_name"], how="inner")
        else:
            filtered = chunk[chunk["machine"].isin(machines_needed)]
        if not filtered.empty:
            chunks.append(filtered)
    cols = ["job_name", "task_name", "worker_name", "start_time", "machine"]
    if not chunks:
        return pd.DataFrame(columns=cols)
    return pd.concat(chunks, ignore_index=True)


def load_machine_metrics(workers_needed: set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(DATA_DIR / "pai_machine_metric.csv", names=METRIC_COLS,
                              header=None, usecols=[0, 1, 2, 3, 7], chunksize=500_000):
        filtered = chunk[chunk["worker_name"].isin(workers_needed)]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=METRIC_COLS)
    return pd.concat(chunks, ignore_index=True)


def load_machine_gpu_capacity() -> dict:
    """machine -> physical GPU count (cap_gpu): the correct normalization
    divisor for machine_gpu, verified against actual machine specs rather
    than assumed from a task's requested GPU count."""
    spec = pd.read_csv(DATA_DIR / "pai_machine_spec.csv",
                        names=["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"],
                        header=None)
    return spec.set_index("machine")["cap_gpu"].to_dict()


def add_approx_cyclical_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """See module docstring: Alibaba timestamps are relative seconds since
    trace start, not wall-clock time, so this is an approximation."""
    df = df.copy()
    sec_of_day = df[time_col] % 86400
    sec_of_week = df[time_col] % (86400 * 7)
    df["submit_hour_sin"] = np.sin(2 * np.pi * sec_of_day / 86400)
    df["submit_hour_cos"] = np.cos(2 * np.pi * sec_of_day / 86400)
    df["submit_dow_sin"] = np.sin(2 * np.pi * sec_of_week / (86400 * 7))
    df["submit_dow_cos"] = np.cos(2 * np.pi * sec_of_week / (86400 * 7))
    return df


def build_machine_history(instances_broad: pd.DataFrame, metrics_broad: pd.DataFrame,
                           gpu_capacity: dict) -> dict:
    """Builds, per machine, a chronologically sorted array of (start_time,
    normalized_utilization) from every worker instance that ran on it,
    using each worker's own single metric snapshot as one history point."""
    worker_util = (
        metrics_broad.dropna(subset=["machine_gpu"])
        .groupby("worker_name")["machine_gpu"].mean()
        .reset_index()  # merging a bare Series against "worker_name" silently
    )                    # mis-joins; this must be a 2-column frame to merge on.
    merged = instances_broad.merge(worker_util, on="worker_name", how="inner")
    merged["n_gpus"] = merged["machine"].map(gpu_capacity)
    merged = merged.dropna(subset=["n_gpus"])
    merged = merged[merged["n_gpus"] > 0]
    merged["util_norm"] = merged["machine_gpu"] / merged["n_gpus"]

    history = {}
    for machine, g in merged.groupby("machine"):
        g = g.sort_values("start_time")
        history[machine] = (g["start_time"].to_numpy(), g["util_norm"].to_numpy())
    return history


def build_dataset(tasks: pd.DataFrame, own_machine: pd.DataFrame, machine_history: dict,
                   gpu_capacity: dict, window_len: int = WINDOW_LEN) -> dict:
    tasks = tasks.merge(own_machine[["job_name", "task_name", "machine"]],
                         on=["job_name", "task_name"], how="inner")

    # searchsorted (O(log n) per task) instead of a fresh boolean mask over
    # the full per-machine history for every one of up to ~911k tasks
    # (O(n) each) -- the same fix applied in preprocess_philly.py, needed
    # here even more given Alibaba's much larger full-scale row count.
    job_ids, windows, y_util, y_completion, static_rows = [], [], [], [], []
    for row in tasks.itertuples(index=False):
        hist = machine_history.get(row.machine)
        if hist is None:
            continue
        times, utils = hist
        idx_start = np.searchsorted(times, row.start_time, side="left")
        if idx_start < window_len:
            continue
        window = utils[idx_start - window_len: idx_start]

        # This task's own outcome: mean utilization across its own
        # worker(s) on this machine during [start_time, end_time], drawn
        # from the SAME machine_history array but restricted to entries
        # that fall inside the task's own execution window (not the
        # "before" history used as input, so there is no overlap/leakage).
        idx_end = np.searchsorted(times, row.end_time, side="right")
        if idx_end <= idx_start:
            continue

        job_ids.append(f"{row.job_name}_{row.task_name}")
        windows.append(window)
        y_util.append(np.nanmean(utils[idx_start:idx_end]))
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


def main(sample: int | None = None, window_len: int = WINDOW_LEN, out_path: Path | None = None):
    print("Loading Alibaba task table ...")
    tasks = load_task_table()
    print(f"  {len(tasks)} valid tasks after cleaning")

    if sample:
        tasks = tasks.sample(n=min(sample, len(tasks)), random_state=42)
    tasks = add_approx_cyclical_time_features(tasks, "start_time")

    print(f"Finding each of {len(tasks)} sampled tasks' own machine ...")
    own_machine = load_instances_for_machines(machines_needed=None, task_keys=tasks[["job_name", "task_name"]])
    own_machine = own_machine.drop_duplicates(subset=["job_name", "task_name"], keep="first")
    print(f"  {len(own_machine)} tasks mapped to a machine")

    machines_needed = set(own_machine["machine"].unique())
    print(f"Loading ALL instances that ever ran on these {len(machines_needed)} machines "
          f"(needed to build machine-level history, not just the sampled tasks' own workers) ...")
    instances_broad = load_instances_for_machines(machines_needed=machines_needed, task_keys=None)
    print(f"  {len(instances_broad)} instance records read")

    workers_needed = set(instances_broad["worker_name"].unique())
    print(f"Loading machine metrics for {len(workers_needed)} workers ...")
    metrics_broad = load_machine_metrics(workers_needed)
    print(f"  {len(metrics_broad)} metric records read")

    gpu_capacity = load_machine_gpu_capacity()
    print("Building per-machine chronological utilization history ...")
    machine_history = build_machine_history(instances_broad, metrics_broad, gpu_capacity)
    print(f"  history built for {len(machine_history)} machines")

    print(f"Assembling windowed dataset (window_len={window_len}) ...")
    data = build_dataset(tasks, own_machine, machine_history, gpu_capacity, window_len=window_len)
    print(f"  {len(data['job_id'])}/{len(tasks)} tasks retained "
          f"(dropped: machine has fewer than {window_len} prior history points, "
          f"or no reading during the task's own window)")

    if len(data["job_id"]) > 0:
        task_start = tasks.set_index(tasks["job_name"] + "_" + tasks["task_name"])["start_time"]
        order = task_start.loc[data["job_id"]].argsort().to_numpy()
        for key in ("job_id", "X_window", "X_static", "y_completion", "y_util"):
            data[key] = data[key][order]

    out_path = out_path or (OUT_DIR / "alibaba_processed.npz")
    np.savez(out_path, **data)
    print(f"Wrote {out_path} ({len(data['job_id'])} rows, window_len={window_len})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", nargs="?", type=int, default=None,
                         help="optional task-count cap for quick testing")
    parser.add_argument("--window-len", type=int, default=WINDOW_LEN)
    parser.add_argument("--out", type=str, default=None, help="override output .npz path")
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    main(sample=args.sample, window_len=args.window_len, out_path=out)
