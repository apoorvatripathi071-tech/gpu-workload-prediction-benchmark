"""
CNN, LSTM, and Hybrid CNN-LSTM models for GPU resource demand prediction,
per paper Section 5 (Proposed Hybrid CNN-LSTM Architecture) and Section 7
(Baselines).

Each model takes:
  - x_window: (batch, window_len) recent machine-utilization history
  - x_static: (batch, n_static) job request-time features
and outputs a single scalar regression target (completion time OR
utilization -- separate model instances are trained per target, per
Section 5.2's rationale for keeping baseline comparisons fair).
"""

import torch
import torch.nn as nn


class StandaloneCNN(nn.Module):
    """Baseline (Section 7.2): convolutional stage only, no recurrent stage."""

    def __init__(self, window_len: int, n_static: int, conv_channels=(32, 64)):
        super().__init__()
        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        conv_out_len = window_len // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * conv_out_len + n_static, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_window, x_static):
        x = x_window.unsqueeze(1)          # (batch, 1, window_len)
        x = self.conv(x)                    # (batch, c2, window_len // 4)
        x = torch.cat([x.flatten(1), x_static], dim=1)
        return self.fc(x).squeeze(-1)


class StandaloneLSTM(nn.Module):
    """Baseline (Section 7.3): recurrent stage only, raw window as input,
    no convolutional feature extraction beforehand."""

    def __init__(self, window_len: int, n_static: int, hidden_sizes=(128, 64)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=h1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=h1, hidden_size=h2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(h2 + n_static, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_window, x_static):
        x = x_window.unsqueeze(-1)          # (batch, window_len, 1)
        x, _ = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)
        x = torch.cat([h_n[-1], x_static], dim=1)
        return self.fc(x).squeeze(-1)


class HybridCNNLSTM(nn.Module):
    """Proposed model (Section 5.2): convolutional stage for local pattern
    extraction, feeding a recurrent stage for temporal dependency modelling,
    then a dense output head with the static job-metadata features
    concatenated in."""

    def __init__(self, window_len: int, n_static: int,
                 conv_channels=(32, 64), lstm_hidden=(128, 64)):
        super().__init__()
        c1, c2 = conv_channels
        h1, h2 = lstm_hidden
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm1 = nn.LSTM(input_size=c2, hidden_size=h1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=h1, hidden_size=h2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(h2 + n_static, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_window, x_static):
        x = x_window.unsqueeze(1)           # (batch, 1, window_len)
        x = self.conv(x)                     # (batch, c2, window_len // 4)
        x = x.transpose(1, 2)                # (batch, seq_len, c2) for LSTM
        x, _ = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)
        x = torch.cat([h_n[-1], x_static], dim=1)
        return self.fc(x).squeeze(-1)


def build_model(name: str, window_len: int, n_static: int) -> nn.Module:
    registry = {
        "cnn": StandaloneCNN,
        "lstm": StandaloneLSTM,
        "hybrid": HybridCNNLSTM,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}', expected one of {list(registry)}")
    return registry[name](window_len=window_len, n_static=n_static)
