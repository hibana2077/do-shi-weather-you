import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = Path(__file__).resolve().parent
DATA_PATH = LAB_DIR / "data" / "agg_data.csv"
RESULTS_DIR = LAB_DIR / "results"

if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from model import LSTMForecaster, LTSFForecaster, WeatherFusionForecaster


TIME_COLUMNS = {
    "StnPresMaxTime",
    "StnPresMinTime",
    "T Max Time",
    "T Min Time",
    "RHMinTime",
    "WGustTime",
    "PrecpMax10Time",
    "PrecpMax60Time",
    "UVI Max Time",
}

MISSING_MARKERS = {"/": np.nan, "T": 0.0, "X": np.nan, "--": np.nan, "...": np.nan}


@dataclass
class SplitData:
    train: Dataset
    val: Dataset
    test: Dataset
    feature_columns: list[str]
    target_mean: float
    target_std: float
    test_dates: list[str]


class WeatherWindowDataset(Dataset):
    def __init__(self, values, targets, dates, window_size, indices):
        self.values = values.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.dates = dates
        self.window_size = window_size
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        start = self.indices[item]
        end = start + self.window_size
        x = self.values[start:end]
        y = self.targets[end]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)

    def target_dates(self):
        return [self.dates[i + self.window_size] for i in self.indices]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_weather_frame(path=DATA_PATH):
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        df["Date"] = pd.RangeIndex(len(df))

    drop_columns = [c for c in TIME_COLUMNS if c in df.columns]
    drop_columns.extend(c for c in ["ObsTime"] if c in df.columns)
    df = df.drop(columns=drop_columns)

    for column in df.columns:
        if column == "Date":
            continue
        df[column] = df[column].replace(MISSING_MARKERS)
        df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_columns = [c for c in df.columns if c != "Date"]
    df[numeric_columns] = df[numeric_columns].interpolate(limit_direction="both")
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    all_missing = [c for c in numeric_columns if df[c].isna().all()]
    if all_missing:
        df = df.drop(columns=all_missing)
    return df


def build_splits(df, target, window_size, train_ratio, val_ratio):
    feature_columns = [c for c in df.columns if c != "Date"]
    if target not in feature_columns:
        raise ValueError(f"Target column '{target}' was not found in {DATA_PATH}")

    n_sequences = len(df) - window_size
    train_end = int(n_sequences * train_ratio)
    val_end = int(n_sequences * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n_sequences:
        raise ValueError("Split ratios leave an empty train, validation, or test split.")

    train_raw_end = train_end + window_size
    train_values = df.loc[: train_raw_end - 1, feature_columns]
    mean = train_values.mean()
    std = train_values.std().replace(0, 1.0)

    scaled = ((df[feature_columns] - mean) / std).to_numpy()
    target_mean = float(mean[target])
    target_std = float(std[target])
    target_values = ((df[target] - target_mean) / target_std).to_numpy()
    date_values = df["Date"].dt.strftime("%Y-%m-%d").tolist() if hasattr(df["Date"], "dt") else df["Date"].astype(str).tolist()

    train_indices = range(0, train_end)
    val_indices = range(train_end, val_end)
    test_indices = range(val_end, n_sequences)
    return SplitData(
        train=WeatherWindowDataset(scaled, target_values, date_values, window_size, train_indices),
        val=WeatherWindowDataset(scaled, target_values, date_values, window_size, val_indices),
        test=WeatherWindowDataset(scaled, target_values, date_values, window_size, test_indices),
        feature_columns=feature_columns,
        target_mean=target_mean,
        target_std=target_std,
        test_dates=WeatherWindowDataset(scaled, target_values, date_values, window_size, test_indices).target_dates(),
    )


def make_model(model_name, window_size, n_features, hidden_size):
    if model_name == "lstm":
        return LSTMForecaster(n_features, hidden_size=hidden_size)
    if model_name == "ltsf_linear":
        return LTSFForecaster(window_size, 1, n_features)
    if model_name == "fusion":
        return WeatherFusionForecaster(window_size, 1, n_features, hidden_size=hidden_size, use_disagreement=True)
    if model_name == "fusion_no_disagreement":
        return WeatherFusionForecaster(window_size, 1, n_features, hidden_size=hidden_size, use_disagreement=False)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, dataloader, device, target_mean, target_std):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for features, target in dataloader:
            features = features.to(device)
            output = model(features).detach().cpu().numpy().reshape(-1)
            target_np = target.numpy().reshape(-1)
            predictions.extend(output * target_std + target_mean)
            actuals.extend(target_np * target_std + target_mean)
    return np.asarray(actuals), np.asarray(predictions)


def metric_dict(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(actual, predicted)
    denom = np.maximum(np.abs(actual), 1e-6)
    mape = float(np.mean(np.abs((actual - predicted) / denom)) * 100.0)
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "mape": mape,
        "r2": float(r2_score(actual, predicted)),
    }


def train_neural_model(args, df, model_name, window_size, seed):
    set_seed(seed)
    split = build_splits(df, args.target, window_size, args.train_ratio, args.val_ratio)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(split.train, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(split.val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(split.test, batch_size=args.batch_size, shuffle=False)

    model = make_model(model_name, window_size, len(split.feature_columns), args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), target)
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)

        val_actual, val_pred = evaluate_model(model, val_loader, device, split.target_mean, split.target_std)
        val_mse = mean_squared_error(val_actual, val_pred)
        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_mse))

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    actual, predicted = evaluate_model(model, test_loader, device, split.target_mean, split.target_std)
    metrics = metric_dict(actual, predicted)
    return {
        "model": model_name,
        "window_size": window_size,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mse": float(best_val),
        "metrics": metrics,
        "train_loss": train_loss_history,
        "validation_mse": val_loss_history,
        "target": args.target,
        "feature_columns": split.feature_columns,
        "device": str(device),
        "test_dates": split.test_dates,
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
    }


def evaluate_baseline(df, target, window_size, train_ratio, val_ratio, model_name, seed):
    split = build_splits(df, target, window_size, train_ratio, val_ratio)
    target_index = split.feature_columns.index(target)
    actual = []
    predicted = []
    raw_values = df[split.feature_columns].to_numpy()
    n_sequences = len(df) - window_size
    val_end = int(n_sequences * (train_ratio + val_ratio))
    for start in range(val_end, n_sequences):
        end = start + window_size
        actual.append(raw_values[end, target_index])
        if model_name == "persistence":
            predicted.append(raw_values[end - 1, target_index])
        elif model_name == "weekly_naive":
            lag = 7 if window_size >= 7 else 1
            predicted.append(raw_values[end - lag, target_index])
        elif model_name == "rolling_mean_7":
            lag = min(7, window_size)
            predicted.append(raw_values[end - lag : end, target_index].mean())
        else:
            raise ValueError(f"Unknown baseline: {model_name}")

    return {
        "model": model_name,
        "window_size": window_size,
        "seed": seed,
        "best_epoch": 0,
        "best_validation_mse": None,
        "metrics": metric_dict(np.asarray(actual), np.asarray(predicted)),
        "train_loss": [],
        "validation_mse": [],
        "target": target,
        "feature_columns": split.feature_columns,
        "device": "none",
        "test_dates": split.test_dates,
        "actual": list(map(float, actual)),
        "predicted": list(map(float, predicted)),
    }


def write_run_json(result):
    run_dir = RESULTS_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{result['model']}_wd{result['window_size']}_seed{result['seed']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path


def write_predictions_csv(result):
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{result['model']}_wd{result['window_size']}_seed{result['seed']}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "actual_temperature_c", "predicted_temperature_c", "absolute_error_c"])
        for date, actual, predicted in zip(result["test_dates"], result["actual"], result["predicted"]):
            writer.writerow([date, actual, predicted, abs(actual - predicted)])
    return path


def aggregate_summary(results):
    rows = []
    for result in results:
        row = {
            "model": result["model"],
            "window_size": result["window_size"],
            "seed": result["seed"],
            "best_epoch": result["best_epoch"],
            **result["metrics"],
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = (
        summary.groupby(["model", "window_size"])
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mape_mean=("mape", "mean"),
            r2_mean=("r2", "mean"),
            seeds=("seed", "count"),
        )
        .reset_index()
        .sort_values(["mae_mean", "rmse_mean"])
    )
    aggregate_path = RESULTS_DIR / "experiment_summary_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)
    return summary_path, aggregate_path, aggregate


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run full weather forecasting experiments.")
    parser.add_argument("--target", default="Temperature")
    parser.add_argument("--models", default="persistence,weekly_naive,rolling_mean_7,ltsf_linear,lstm,fusion_no_disagreement,fusion")
    parser.add_argument("--windows", default="7,14,30,60,90")
    parser.add_argument("--seeds", default="13,29,47")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_weather_frame()
    models = parse_str_list(args.models)
    windows = parse_int_list(args.windows)
    seeds = parse_int_list(args.seeds)
    neural_models = {"lstm", "ltsf_linear", "fusion_no_disagreement", "fusion"}
    baselines = {"persistence", "weekly_naive", "rolling_mean_7"}

    results = []
    total = len(models) * len(windows) * len(seeds)
    completed = 0
    for window_size in windows:
        for seed in seeds:
            for model_name in models:
                completed += 1
                print(f"[{completed}/{total}] model={model_name} window={window_size} seed={seed}", flush=True)
                if model_name in baselines:
                    result = evaluate_baseline(df, args.target, window_size, args.train_ratio, args.val_ratio, model_name, seed)
                elif model_name in neural_models:
                    result = train_neural_model(args, df, model_name, window_size, seed)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                write_run_json(result)
                write_predictions_csv(result)
                results.append(result)

    summary_path, aggregate_path, aggregate = aggregate_summary(results)
    best = aggregate.iloc[0].to_dict()
    print(f"Summary written to {summary_path}")
    print(f"Aggregate summary written to {aggregate_path}")
    print("Best configuration:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
