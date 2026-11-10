# Leakage-Aware Cross-Dataset Evaluation of GPU Workload Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24096961.svg)](https://doi.org/10.5281/zenodo.24096961)

Code artifact for the paper *"Leakage-Aware Cross-Dataset Evaluation of GPU Workload Prediction: When Do Deep Sequence Models Actually Beat Simple Baselines?"* (Apoorva Tripathi).

Benchmarks CNN / LSTM / Hybrid CNN-LSTM models against naive persistence, Ridge, XGBoost, and an ARIMA target-history reference on two public production GPU traces (Microsoft Philly, Alibaba PAI GPU v2020), under a leakage-safe, harmonized machine-history representation and zero-shot cross-dataset transfer.

## Layout

- `src/` — the full pipeline: data download, preprocessing, models, baselines, multi-seed/significance/ablation analyses, figures (16 scripts; each has a usage docstring).
- `results/` — machine-generated metric JSONs and figures. Every number in every table of the paper comes from one of these files; nothing is hand-transcribed.
- `data/` — **not in the repo** (~13 GB). Both traces are public and free:

```
python src/download_data.py
python src/preprocess_philly.py
python src/preprocess_alibaba.py
```

File-level MD5 checksums of the exact trace files used, plus the Philly repo commit, are listed in the paper's Reproducibility section so inputs can be verified byte-for-byte.

## Environment

Python 3.13.3, PyTorch 2.12.1 (CPU-only), scikit-learn 1.9.0, XGBoost 3.3.0, statsmodels 0.14.6, NumPy 2.5.1, Pandas 3.0.3, SciPy 1.18.0. Seeds 42–46 for the multi-seed analyses.

## Reproducing the paper's tables

Run order and the table each output feeds:

```
python src/baseline_classical.py --target util            # Table 3
python src/baseline_classical.py --target completion      # Table 5
python src/baseline_arima.py --target util                # Table 3 (ARIMA row)
python src/baseline_arima.py --target completion          # Table 5 (ARIMA row)
python src/train.py --target util --epochs 50             # Table 3 (deep rows)
python src/train.py --target completion --epochs 50       # Table 5 (deep rows)
python src/multi_seed_ci.py --target util                 # Table 4
python src/multi_seed_ci.py --target completion           # Table 5 (CI column)
python src/significance_test.py --target util             # Section 7.1.2
python src/significance_vs_baselines.py --target util     # Table 4b
python src/retention_analysis.py                          # Table 2b
python src/window_ablation.py                             # Table 6
python src/reverse_transfer.py                            # Table 7
python src/lag_features.py                                # Section 7.4 features
python src/train_lag_features.py                          # Table 5b
python src/make_figures.py                                # Figures 1-4
```
