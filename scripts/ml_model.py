from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_R2 = 0.97


class ModelTrainingError(RuntimeError):
    """Raised when the time series is insufficient for model training."""


@dataclass
class ForecastResult:
    metrics: dict[str, float | int | str | bool]
    past_predictions: pd.DataFrame
    future_predictions: pd.DataFrame


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total == 0:
        return 0.0
    residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (residual / total)


def _build_features(day_index: np.ndarray, closes: np.ndarray | None = None, scale: float | None = None) -> np.ndarray:
    x = day_index.astype(float)
    x_max = max(float(scale if scale is not None else np.max(x)), 1.0)
    scaled = x / x_max
    features = [np.ones_like(scaled), scaled, scaled**2]

    if closes is not None:
        close_series = pd.Series(closes)
        rolling_7 = close_series.rolling(7, min_periods=1).mean().to_numpy()
        rolling_30 = close_series.rolling(30, min_periods=1).mean().to_numpy()
        features.extend([rolling_7, rolling_30])

    return np.column_stack(features)


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(features.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(features.T @ features + penalty) @ features.T @ target


def _predict_trend(coefficients: np.ndarray, day_index: np.ndarray, scale: float) -> np.ndarray:
    trend_features = _build_features(day_index, scale=scale)
    if len(coefficients) > trend_features.shape[1]:
        last_columns = len(coefficients) - trend_features.shape[1]
        padding = np.zeros((len(day_index), last_columns))
        trend_features = np.column_stack([trend_features, padding])
    return trend_features @ coefficients


def train_predict_evaluate(dataframe: pd.DataFrame, future_days: int = 365, test_fraction: float = 0.2) -> ForecastResult:
    df = dataframe[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    if len(df) < 10:
        raise ModelTrainingError("Sao necessarias pelo menos 10 linhas validas para treinar o modelo.")

    first_date = df["Date"].min()
    df["day_index"] = (df["Date"] - first_date).dt.days.astype(float)
    split_index = max(1, min(int(len(df) * (1 - test_fraction)), len(df) - 1))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    if train_df.empty or test_df.empty:
        raise ModelTrainingError("Split temporal invalido: treino ou teste ficou vazio.")

    feature_scale = max(float(df["day_index"].max()), 1.0)
    train_x = _build_features(train_df["day_index"].to_numpy(), train_df["Close"].to_numpy(), scale=feature_scale)
    train_y = train_df["Close"].to_numpy(dtype=float)
    coefficients = _fit_ridge(train_x, train_y, alpha=0.1)

    test_x = _build_features(test_df["day_index"].to_numpy(), test_df["Close"].to_numpy(), scale=feature_scale)
    y_true = test_df["Close"].to_numpy(dtype=float)
    y_pred = test_x @ coefficients

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(_r2_score(y_true, y_pred))

    future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=future_days, freq="D")
    future_index = (future_dates - first_date).days.to_numpy(dtype=float)
    future_pred = _predict_trend(coefficients, future_index, scale=feature_scale)
    future_pred = np.maximum(future_pred, 0)

    past_predictions = test_df[["Date", "Close"]].copy()
    past_predictions["Predicted"] = y_pred
    future_predictions = pd.DataFrame({"Date": future_dates, "Predicted": future_pred})

    metrics: dict[str, float | int | str | bool] = {
        "iterations": 1,
        "epochs": 1,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "r2": r2,
        "target_r2": TARGET_R2,
        "target_reached": bool(r2 >= TARGET_R2),
        "rmse": rmse,
        "mae": mae,
        "model": "ridge_polynomial_temporal",
    }
    return ForecastResult(metrics=metrics, past_predictions=past_predictions, future_predictions=future_predictions)
