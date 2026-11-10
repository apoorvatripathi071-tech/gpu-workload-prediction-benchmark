"""
Trains CNN, LSTM, and Hybrid CNN-LSTM (Section 5/7) on the Philly-processed
dataset, evaluates in-distribution (Section 9.1) and zero-shot cross-dataset
transfer to Alibaba (Section 9.2), for both prediction targets
(completion_time_sec, avg_gpu_util).

Usage:
    python src/train.py --target completion --epochs 50
    python src/train.py --target util --epochs 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def chronological_split(n: int, train_frac=0.70, val_frac=0.15):
    """Data is pre-sorted chronologically by preprocess_*.py, so indices
    are simply sliced in order (Section 6.3) -- no shuffling, since a
    random split would leak future information into training via
    temporally adjacent, non-independent jobs."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        np.arange(0, n_train),
        np.arange(n_train, n_train + n_val),
        np.arange(n_train + n_val, n),
    )


class Scaler:
    """Min-max scaler fit ONLY on training data (Section 6.3) -- the same
    fitted parameters are reused, unchanged, for validation/test splits and
    for the Alibaba zero-shot transfer evaluation, since re-fitting on the
    target trace would leak distributional information a real deployment
    would not have access to."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray):
        self.min_ = x.min(axis=0, keepdims=True)
        self.max_ = x.max(axis=0, keepdims=True)
        self.max_[self.max_ == self.min_] += 1e-8  # avoid div-by-zero on constant columns
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.max_ - self.min_) + self.min_


def make_loader(x_window, x_static, y, batch_size=64, shuffle=False):
    ds = TensorDataset(
        torch.tensor(x_window, dtype=torch.float32),
        torch.tensor(x_static, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_model(model, train_loader, val_loader, epochs=50, patience=10, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xw, xs, y in train_loader:
            xw, xs, y = xw.to(device), xs.to(device), y.to(device)
            opt.zero_grad()
            pred = model(xw, xs)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xw, xs, y in val_loader:
                xw, xs, y = xw.to(device), xs.to(device), y.to(device)
                val_losses.append(loss_fn(model(xw, xs), y).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, loader, y_scaler, log_transform: bool, device="cpu", return_arrays: bool = False) -> dict:
    """Inverse-transforms predictions and targets fully back to native units
    (undoing both min-max scaling and, for completion_time, the log1p
    transform) BEFORE computing RMSE/MAE/MAPE. This matters because log1p
    is nonlinear: an RMSE computed in log-space and then linearly rescaled
    is not the same quantity as RMSE computed in native seconds -- the two
    only coincide for linear transforms. Metrics reported here are always
    in native units (Section 8's definitions), matching what's written in
    the manuscript; only the TRAINING target was log-transformed, as a
    modeling choice to stop small values (a large fraction of completion
    times) from being numerically swamped by the heavy right tail during
    optimization -- see Section 9's discussion of why raw MAPE on
    completion_time is otherwise dominated by near-zero denominators.
    """
    model.eval()
    preds, trues = [], []
    for xw, xs, y in loader:
        xw, xs = xw.to(device), xs.to(device)
        preds.append(model(xw, xs).cpu().numpy())
        trues.append(y.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    trues = y_scaler.inverse_transform(trues.reshape(-1, 1)).ravel()
    if log_transform:
        # expm1 grows explosively, so clipping must happen HERE, in
        # log-space, not on the normalized-[0,1]-ish value beforehand: a
        # loose margin there (e.g. +0.5) still explodes after expm1
        # (observed: RMSE ~1e8s on a target that maxes out at 2.1e6s).
        # Clip to log1p(90 days), a physically generous upper bound for
        # any real training job, regardless of how far the model
        # extrapolates under the Alibaba distribution shift. As before,
        # heavy clipping on the transfer set is itself a reportable
        # finding about the size of the generalization gap (Section 9.3),
        # not something to hide by clipping quietly.
        log_cap = np.log1p(90 * 24 * 60 * 60)
        n_clipped = int(np.sum(preds > log_cap))
        preds = np.clip(preds, 0, log_cap)
        preds, trues = np.expm1(preds), np.expm1(trues)
        if n_clipped:
            print(f"    [note] clipped {n_clipped}/{len(preds)} predictions "
                  f"exceeding a 90-day physical cap before expm1")

    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae = float(np.mean(np.abs(preds - trues)))
    median_ae = float(np.median(np.abs(preds - trues)))
    nonzero = trues != 0
    mape = float(np.mean(np.abs((preds[nonzero] - trues[nonzero]) / trues[nonzero])) * 100) if nonzero.any() else float("nan")
    # RMSLE: the primary metric for completion_time (Section 8, revised
    # following external review) -- raw MAPE is unstable for heavy-tailed
    # targets with near-zero true values (a large fraction of jobs finish
    # in seconds), and penalizes proportional errors on small values far
    # more harshly than on large ones. RMSLE instead measures error in
    # log-ratio terms, which matches how "off by a small vs. large job"
    # actually feels proportionally, and is far less sensitive to the
    # near-zero-denominator pathology than raw MAPE. Predictions are
    # clipped to >= 0 first since RMSLE (like the targets themselves) is
    # only defined for non-negative values -- a model output dipping
    # slightly negative should not crash the metric.
    preds_nonneg = np.clip(preds, 0, None)
    rmsle = float(np.sqrt(np.mean((np.log1p(preds_nonneg) - np.log1p(trues)) ** 2)))
    # Standard companion metric for heavy-tailed duration data (used
    # throughout the GPU-scheduling literature cited in Section 2):
    # the fraction of predictions within a factor of 2x of the true value,
    # which stays meaningful even when MAPE is distorted by near-zero
    # true values.
    ratio = np.maximum(preds, 1e-6) / np.maximum(trues, 1e-6)
    within_2x = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100)
    out = {"rmse": rmse, "mae": mae, "median_ae": median_ae, "rmsle": rmsle,
           "mape": mape, "within_2x_pct": within_2x}
    if return_arrays:
        # Per-sample predictions/truths, needed for paired significance
        # testing between models on the SAME test points (significance_test.py)
        # -- not included by default since run()'s results dict is serialized
        # straight to JSON and numpy arrays aren't JSON-serializable.
        out["_preds"] = preds
        out["_trues"] = trues
    return out


def run(target: str, epochs: int, batch_size: int, device: str, seed: int = 42,
        philly_path: Path | None = None, alibaba_path: Path | None = None,
        save: bool = True, verbose: bool = True, models_to_run=("cnn", "lstm", "hybrid"),
        return_arrays: bool = False):
    assert target in ("completion", "util")
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_key = "y_completion" if target == "completion" else "y_util"
    # completion_time_sec spans 2 to 2.1M seconds (6 orders of magnitude,
    # median 1524s, 99th percentile 441,015s -- see Section 9 discussion).
    # Training directly on raw seconds lets the huge right tail dominate
    # the loss and starves the model of signal for the ~24% of jobs under
    # 5 minutes; log1p keeps both ends of the distribution comparably
    # weighted during optimization. Utilization is already bounded [0,100]
    # and doesn't have this pathology, so it's left untransformed.
    log_transform = (target == "completion")

    philly = load_npz(philly_path or (DATA_DIR / "philly_processed.npz"))
    alibaba = load_npz(alibaba_path or (DATA_DIR / "alibaba_processed.npz"))
    if verbose:
        print(f"Philly: {len(philly['job_id'])} rows | Alibaba: {len(alibaba['job_id'])} rows")

    n = len(philly["job_id"])
    train_idx, val_idx, test_idx = chronological_split(n)

    def target_values(data):
        y = data[y_key]
        return np.log1p(y) if log_transform else y

    window_scaler = Scaler().fit(philly["X_window"][train_idx])
    static_scaler = Scaler().fit(philly["X_static"][train_idx])
    y_scaler = Scaler().fit(target_values(philly)[train_idx].reshape(-1, 1))

    def prep(split_idx, data=philly):
        xw = window_scaler.transform(data["X_window"][split_idx])
        xs = static_scaler.transform(data["X_static"][split_idx])
        y = y_scaler.transform(target_values(data)[split_idx].reshape(-1, 1)).ravel()
        return xw, xs, y

    xw_tr, xs_tr, y_tr = prep(train_idx)
    xw_val, xs_val, y_val = prep(val_idx)
    xw_test, xs_test, y_test = prep(test_idx)

    train_loader = make_loader(xw_tr, xs_tr, y_tr, batch_size, shuffle=True)
    val_loader = make_loader(xw_val, xs_val, y_val, batch_size)
    test_loader = make_loader(xw_test, xs_test, y_test, batch_size)

    # Alibaba transfer set: same scalers (fit on Philly train only), no retraining.
    xw_ali = window_scaler.transform(alibaba["X_window"])
    xs_ali = static_scaler.transform(alibaba["X_static"])
    y_ali = y_scaler.transform(target_values(alibaba).reshape(-1, 1)).ravel()
    alibaba_loader = make_loader(xw_ali, xs_ali, y_ali, batch_size)

    results = {}
    for name in models_to_run:
        if verbose:
            print(f"\n=== Training {name} (target={target}, seed={seed}) ===")
        model = build_model(name, window_len=philly["X_window"].shape[1], n_static=philly["X_static"].shape[1])
        t0 = time.time()
        model = train_one_model(model, train_loader, val_loader, epochs=epochs, device=device)
        train_time = time.time() - t0

        t0 = time.time()
        in_dist = evaluate(model, test_loader, y_scaler, log_transform, device, return_arrays=return_arrays)
        inference_time_ms = (time.time() - t0) / max(len(test_idx), 1) * 1000

        transfer = evaluate(model, alibaba_loader, y_scaler, log_transform, device, return_arrays=return_arrays)

        results[name] = {"in_distribution": in_dist, "transfer": transfer,
                          "train_time_sec": train_time, "inference_ms_per_sample": inference_time_ms}
        if verbose:
            log_msg = {k: v for k, v in in_dist.items() if not k.startswith("_")}
            transfer_msg = {k: v for k, v in transfer.items() if not k.startswith("_")}
            print(f"  in-distribution: {log_msg}")
            print(f"  Alibaba transfer: {transfer_msg}")
            print(f"  train_time={train_time:.1f}s  inference={inference_time_ms:.3f}ms/sample")

    if save:
        # Strip non-JSON-serializable per-sample arrays (only present when
        # return_arrays=True, used by significance_test.py) before writing.
        serializable = {
            name: {
                split: {k: v for k, v in metrics.items() if not k.startswith("_")}
                for split, metrics in r.items() if split in ("in_distribution", "transfer")
            } | {k: v for k, v in r.items() if k not in ("in_distribution", "transfer")}
            for name, r in results.items()
        }
        out_path = RESULTS_DIR / f"results_{target}.json"
        import json
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        if verbose:
            print(f"\nWrote {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["completion", "util"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.target, args.epochs, args.batch_size, args.device, seed=args.seed)
