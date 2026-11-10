"""
Retained-vs-dropped instance comparison (Section 4.4, external-review
requirement). The preprocessing pipeline keeps only ~56-57% of valid
jobs/tasks after applying the window-availability and target-availability
criteria (Section 4.2). Dropped instances are not random -- they are
disproportionately jobs with insufficient prior machine history (i.e.
scheduled early in their machine's observed lifetime) or with no telemetry
during their own execution window (very short jobs). This script checks
whether that selection materially shifts the benchmark population's
duration and GPU-count distribution, which bears directly on how the
paper's results should be read as "jobs with sufficient prior machine
history," not "all jobs."

Usage:
    python src/retention_analysis.py
"""

import json
from pathlib import Path

import numpy as np

from preprocess_philly import load_job_log
from preprocess_alibaba import load_task_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def summarize(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else float("nan"),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "p90": float(np.percentile(values, 90)) if len(values) else float("nan"),
    }


def analyze_philly():
    jobs = load_job_log()
    retained_ids = set(np.load(DATA_DIR / "philly_processed.npz", allow_pickle=True)["job_id"].tolist())

    jobs["retained"] = jobs["job_id"].isin(retained_ids)
    jobs["trace_position_pct"] = jobs["submitted_time"].rank(pct=True) * 100

    retained = jobs[jobs["retained"]]
    dropped = jobs[~jobs["retained"]]

    return {
        "total_valid": int(len(jobs)),
        "retained": {
            "duration_sec": summarize(retained["completion_time_sec"].to_numpy()),
            "num_gpus": summarize(retained["num_gpus"].to_numpy()),
            "trace_position_pct": summarize(retained["trace_position_pct"].to_numpy()),
        },
        "dropped": {
            "duration_sec": summarize(dropped["completion_time_sec"].to_numpy()),
            "num_gpus": summarize(dropped["num_gpus"].to_numpy()),
            "trace_position_pct": summarize(dropped["trace_position_pct"].to_numpy()),
        },
    }


def analyze_alibaba():
    tasks = load_task_table()
    tasks["job_id"] = tasks["job_name"] + "_" + tasks["task_name"]
    retained_ids = set(np.load(DATA_DIR / "alibaba_processed.npz", allow_pickle=True)["job_id"].tolist())

    tasks["retained"] = tasks["job_id"].isin(retained_ids)
    tasks["trace_position_pct"] = tasks["start_time"].rank(pct=True) * 100

    retained = tasks[tasks["retained"]]
    dropped = tasks[~tasks["retained"]]

    return {
        "total_valid": int(len(tasks)),
        "retained": {
            "duration_sec": summarize(retained["completion_time_sec"].to_numpy()),
            "num_gpus": summarize(retained["num_gpus"].to_numpy()),
            "trace_position_pct": summarize(retained["trace_position_pct"].to_numpy()),
        },
        "dropped": {
            "duration_sec": summarize(dropped["completion_time_sec"].to_numpy()),
            "num_gpus": summarize(dropped["num_gpus"].to_numpy()),
            "trace_position_pct": summarize(dropped["trace_position_pct"].to_numpy()),
        },
    }


def main():
    print("Analyzing Philly retention ...")
    philly = analyze_philly()
    print(json.dumps(philly, indent=2))

    print("\nAnalyzing Alibaba retention ...")
    alibaba = analyze_alibaba()
    print(json.dumps(alibaba, indent=2))

    out = {"philly": philly, "alibaba": alibaba}
    out_path = RESULTS_DIR / "retention_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
