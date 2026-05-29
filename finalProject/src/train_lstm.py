"""Train an LSTM model for stock closing-price prediction.

The script is intentionally self-contained for the SEP740 final project:
it can download Yahoo Finance data, reuse a cached CSV, train a PyTorch LSTM,
evaluate against a naive baseline, and save figures/metrics for the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Log_Return",
    "MA5_Ratio",
    "MA20_Ratio",
    "Volatility20",
    "Volume_Change",
]


@dataclass
class Metrics:
    mae: float
    rmse: float
    mse: float
    mape_percent: float
    r2: float
    directional_accuracy: float


class StockLSTM(nn.Module):
    """Compact LSTM regressor for next-day closing price."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_stock_csv(path: Path) -> pd.DataFrame:
    """Read CSVs produced by yfinance with either flat or multi-level headers."""
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            if len(set(df.columns.get_level_values(1))) == 1:
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = [
                    first if first != "Price" else second
                    for first, second in df.columns.to_list()
                ]
    except Exception:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    df.index.name = "Date"
    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[RAW_COLUMNS].sort_index()
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()


def download_yahoo(ticker: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance returned an empty dataset.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return normalize_columns(data)


def make_supervised(
    df: pd.DataFrame,
    sequence_length: int,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    df = add_features(df)
    train_rows = int(len(df) * train_ratio)
    train_df = df.iloc[:train_rows]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(train_df[FEATURE_COLUMNS])
    target_scaler.fit(train_df[["Target_Return"]])

    scaled_features = feature_scaler.transform(df[FEATURE_COLUMNS])
    scaled_target = target_scaler.transform(df[["Target_Return"]]).ravel()

    x_values, y_values, target_dates, previous_close = [], [], [], []
    for i in range(sequence_length - 1, len(df) - 1):
        x_values.append(scaled_features[i - sequence_length + 1 : i + 1])
        y_values.append(scaled_target[i])
        target_dates.append(df.index[i + 1])
        previous_close.append(df["Close"].iloc[i])

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        np.asarray(target_dates),
        np.asarray(previous_close, dtype=np.float32),
        feature_scaler,
        target_scaler,
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["Log_Return"] = np.log(featured["Close"] / featured["Close"].shift(1))
    featured["MA5_Ratio"] = featured["Close"] / featured["Close"].rolling(5).mean() - 1.0
    featured["MA20_Ratio"] = featured["Close"] / featured["Close"].rolling(20).mean() - 1.0
    featured["Volatility20"] = featured["Log_Return"].rolling(20).std()
    featured["Volume_Change"] = featured["Volume"].pct_change()
    featured["Target_Return"] = np.log(featured["Close"].shift(-1) / featured["Close"])
    return featured.replace([np.inf, -np.inf], np.nan).dropna()


def inverse_target(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).ravel()


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    previous_close: np.ndarray,
) -> Metrics:
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100.0
    actual_direction = np.sign(actual - previous_close)
    predicted_direction = np.sign(predicted - previous_close)
    directional_accuracy = float(np.mean(actual_direction == predicted_direction))
    return Metrics(
        mae=float(mae),
        rmse=float(rmse),
        mse=float(mse),
        mape_percent=float(mape),
        r2=float(r2_score(actual, predicted)),
        directional_accuracy=directional_accuracy,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    patience: int = 5,
) -> tuple[list[float], list[float], int]:
    """Train with Huber loss, early stopping, and best-model restoration."""
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        model.eval()
        val_batch_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_loss = criterion(model(batch_x), batch_y)
                val_batch_losses.append(val_loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss = float(np.mean(val_batch_losses))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Huber loss: {train_loss:.4f} | "
            f"Validation Huber loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"Restored best model from epoch {best_epoch} "
        f"(validation Huber loss: {best_val_loss:.4f})."
    )

    return train_losses, val_losses, best_epoch


def predict(model: nn.Module, x_values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    loader = DataLoader(
        TensorDataset(torch.tensor(x_values, dtype=torch.float32)),
        batch_size=256,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch_x,) in loader:
            outputs.append(model(batch_x.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def save_plots(
    output_dir: Path,
    dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4.8))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title("LSTM training and validation Huber loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5.2))
    plt.plot(dates, actual, label="Actual close", linewidth=2)
    plt.plot(dates, predicted, label="LSTM prediction", linewidth=1.8)
    plt.plot(dates, baseline, label="Naive baseline", linewidth=1.2, alpha=0.75)
    plt.xlabel("Date")
    plt.ylabel("Close price (USD)")
    plt.title("AAPL next-day close prediction on test period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "predictions.png", dpi=180)
    plt.close()

    residuals = actual - predicted
    plt.figure(figsize=(8, 4.8))
    plt.hist(residuals, bins=35, color="#356A8A", edgecolor="white")
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Actual - predicted close price (USD)")
    plt.ylabel("Frequency")
    plt.title("Distribution of LSTM test residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=180)
    plt.close()

    # Zoomed prediction chart for the most recent 120 test days.
    recent_days = min(120, len(dates))
    plt.figure(figsize=(12, 5.2))
    plt.plot(
        dates[-recent_days:],
        actual[-recent_days:],
        label="Actual close",
        linewidth=2.2,
    )
    plt.plot(
        dates[-recent_days:],
        predicted[-recent_days:],
        label="LSTM prediction",
        linewidth=1.8,
    )
    plt.plot(
        dates[-recent_days:],
        baseline[-recent_days:],
        label="Naive baseline",
        linewidth=1.2,
        alpha=0.75,
    )
    plt.xlabel("Date")
    plt.ylabel("Close price (USD)")
    plt.title(f"AAPL next-day prediction: last {recent_days} test days")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "predictions_zoomed.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=740)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.csv is not None:
        df = read_stock_csv(args.csv)
        data_source = str(args.csv)
    else:
        df = download_yahoo(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            cache_dir=Path(".yf_cache"),
        )
        data_source = f"Yahoo Finance download for {args.ticker}"

    x_all, y_all, dates, previous_close_all, _, target_scaler = make_supervised(
        df=df,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
    )

    test_start = int(len(x_all) * args.train_ratio)
    val_start = int(test_start * (1.0 - args.validation_ratio))

    x_train, y_train = x_all[:val_start], y_all[:val_start]
    x_val, y_val = x_all[val_start:test_start], y_all[val_start:test_start]
    x_test, y_test = x_all[test_start:], y_all[test_start:]
    test_dates = dates[test_start:]
    previous_close = previous_close_all[test_start:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = StockLSTM(
        input_size=x_all.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    train_losses, val_losses, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        patience=5,
    )

    pred_scaled = predict(model, x_test, device=device)
    actual_returns = inverse_target(y_test, target_scaler)
    predicted_returns = inverse_target(pred_scaled, target_scaler)
    actual = previous_close * np.exp(actual_returns)
    predicted = previous_close * np.exp(predicted_returns)
    baseline = previous_close.copy()

    lstm_metrics = compute_metrics(actual, predicted, previous_close)
    baseline_metrics = compute_metrics(actual, baseline, previous_close)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(
        output_dir=args.output_dir,
        dates=test_dates,
        actual=actual,
        predicted=predicted,
        baseline=baseline,
        train_losses=train_losses,
        val_losses=val_losses,
    )

    metrics_payload = {
        "data_source": data_source,
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "sequence_length": args.sequence_length,
        "train_windows": int(len(x_train)),
        "validation_windows": int(len(x_val)),
        "test_windows": int(len(x_test)),
        "device": str(device),
        "loss_function": "HuberLoss(delta=1.0)",
        "learning_rate": args.learning_rate,
        "epochs_requested": args.epochs,
        "epochs_trained": len(train_losses),
        "best_epoch": best_epoch,
        "early_stopping_patience": 5,
        "seed": args.seed,
        "lstm": asdict(lstm_metrics),
        "naive_baseline": asdict(baseline_metrics),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()