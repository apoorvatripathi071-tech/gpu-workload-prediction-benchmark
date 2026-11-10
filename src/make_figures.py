"""
Core figures for the paper (Section 8, external-review requirement):
distribution plots straight from the harmonized data (no model needed),
plus a predicted-vs-actual scatter for the Hybrid model.

Usage:
    python src/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import (DATA_DIR, RESULTS_DIR, load_npz, chronological_split,
                    Scaler, evaluate, train_one_model, make_loader, build_model)

FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def completion_time_ccdf():
    philly = load_npz(DATA_DIR / "philly_processed.npz")
    alibaba = load_npz(DATA_DIR / "alibaba_processed.npz")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for data, label, color in [(philly, "Philly", "tab:blue"), (alibaba, "Alibaba", "tab:orange")]:
        y = np.sort(data["y_completion"])
        ccdf = 1.0 - np.arange(len(y)) / len(y)
        ax.plot(y, ccdf, label=label, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Completion time (seconds, log scale)")
    ax.set_ylabel("P(X > x) (log scale)")
    ax.set_title("Completion-time complementary CDF\n(heavy-tailed: note the log-log axes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "completion_time_ccdf.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'completion_time_ccdf.png'}")


def utilization_distribution():
    philly = load_npz(DATA_DIR / "philly_processed.npz")
    alibaba = load_npz(DATA_DIR / "alibaba_processed.npz")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(0, 100, 51)
    ax.hist(philly["y_util"], bins=bins, alpha=0.55, label="Philly", color="tab:blue", density=True)
    ax.hist(alibaba["y_util"], bins=bins, alpha=0.55, label="Alibaba", color="tab:orange", density=True)
    ax.set_xlabel("GPU utilization (%)")
    ax.set_ylabel("Density")
    ax.set_title("Utilization target distribution by dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "utilization_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'utilization_distribution.png'}")


def predicted_vs_actual(target: str = "util", epochs: int = 50, device: str = "cpu"):
    """Trains a fresh Hybrid model (same protocol as train.py) just to get
    predictions for the scatter plot -- kept self-contained rather than
    threading plotting through train.py's run(), which returns aggregated
    metrics dicts across all three models rather than one model's arrays."""
    philly = load_npz(DATA_DIR / "philly_processed.npz")
    alibaba = load_npz(DATA_DIR / "alibaba_processed.npz")
    y_key = "y_completion" if target == "completion" else "y_util"
    log_transform = (target == "completion")

    n = len(philly["job_id"])
    train_idx, val_idx, test_idx = chronological_split(n)

    def target_values(data):
        y = data[y_key]
        return np.log1p(y) if log_transform else y

    window_scaler = Scaler().fit(philly["X_window"][train_idx])
    static_scaler = Scaler().fit(philly["X_static"][train_idx])
    y_scaler = Scaler().fit(target_values(philly)[train_idx].reshape(-1, 1))

    def prep(idx, data=philly):
        return (window_scaler.transform(data["X_window"][idx]),
                static_scaler.transform(data["X_static"][idx]),
                y_scaler.transform(target_values(data)[idx].reshape(-1, 1)).ravel())

    xw_tr, xs_tr, y_tr = prep(train_idx)
    xw_val, xs_val, y_val = prep(val_idx)
    xw_test, xs_test, y_test = prep(test_idx)
    train_loader = make_loader(xw_tr, xs_tr, y_tr, 256, shuffle=True)
    val_loader = make_loader(xw_val, xs_val, y_val, 256)
    test_loader = make_loader(xw_test, xs_test, y_test, 256)

    xw_ali = window_scaler.transform(alibaba["X_window"])
    xs_ali = static_scaler.transform(alibaba["X_static"])
    y_ali = y_scaler.transform(target_values(alibaba).reshape(-1, 1)).ravel()
    alibaba_loader = make_loader(xw_ali, xs_ali, y_ali, 256)

    model = build_model("hybrid", window_len=philly["X_window"].shape[1], n_static=philly["X_static"].shape[1])
    model = train_one_model(model, train_loader, val_loader, epochs=epochs, device=device)

    in_dist = evaluate(model, test_loader, y_scaler, log_transform, device, return_arrays=True)
    transfer = evaluate(model, alibaba_loader, y_scaler, log_transform, device, return_arrays=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, res, title in [(axes[0], in_dist, "In-distribution (Philly test)"),
                            (axes[1], transfer, "Zero-shot transfer (Alibaba)")]:
        p, t = res["_preds"], res["_trues"]
        lo, hi = float(min(p.min(), t.min())), float(max(p.max(), t.max()))
        ax.scatter(t, p, s=4, alpha=0.25, color="tab:blue")
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="Perfect prediction")
        if log_transform:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.legend()
    fig.suptitle(f"Hybrid CNN-LSTM: predicted vs. actual ({target})")
    fig.tight_layout()
    out_path = FIG_DIR / f"predicted_vs_actual_{target}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    completion_time_ccdf()
    utilization_distribution()
    predicted_vs_actual("util")
