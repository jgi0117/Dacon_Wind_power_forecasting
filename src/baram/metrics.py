"""대회 공식 평가 산식과 발전량 정규화 유틸리티."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}


def _validate_frames(answer: pd.DataFrame, prediction: pd.DataFrame) -> None:
    missing_answer = set(TARGET_COLS) - set(answer.columns)
    missing_prediction = set(TARGET_COLS) - set(prediction.columns)
    if missing_answer:
        raise ValueError(f"Missing answer columns: {sorted(missing_answer)}")
    if missing_prediction:
        raise ValueError(f"Missing prediction columns: {sorted(missing_prediction)}")
    if len(answer) != len(prediction):
        raise ValueError(
            f"Answer/prediction row mismatch: {len(answer)} != {len(prediction)}"
        )


def align_by_time(
    answer: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    answer_time_col: str = "kst_dtm",
    prediction_time_col: str = "forecast_kst_dtm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly align answer and prediction frames by timestamp."""
    answer = answer.copy()
    prediction = prediction.copy()
    answer[answer_time_col] = pd.to_datetime(answer[answer_time_col])
    prediction[prediction_time_col] = pd.to_datetime(
        prediction[prediction_time_col]
    )
    if answer[answer_time_col].duplicated().any():
        raise ValueError("Answer timestamps contain duplicates.")
    if prediction[prediction_time_col].duplicated().any():
        raise ValueError("Prediction timestamps contain duplicates.")
    answer = answer.set_index(answer_time_col).sort_index()
    prediction = prediction.set_index(prediction_time_col).sort_index()
    if not answer.index.equals(prediction.index):
        raise ValueError(
            "Answer/prediction timestamps differ: "
            f"answer_only={len(answer.index.difference(prediction.index))}, "
            f"prediction_only={len(prediction.index.difference(answer.index))}"
        )
    return answer[TARGET_COLS], prediction[TARGET_COLS]


def capacity_factor(y: pd.Series, target: str) -> pd.Series:
    return y.astype(float) / CAPACITY_KWH[target]


def restore_generation(prediction: np.ndarray, target: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    return np.clip(prediction, 0.0, 1.0) * CAPACITY_KWH[target]


def target_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    '''Competition score for one capacity-factor target (higher is better).'''
    actual = np.asarray(actual, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual >= 0.10)
    if not valid.any():
        raise ValueError('No rows are eligible for the target score.')
    actual_valid = actual[valid]
    error = np.abs(np.clip(prediction[valid], 0.0, 1.0) - actual_valid)
    nmae = float(error.mean())
    unit_price = np.select(
        [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
    )
    denominator = float(np.sum(actual_valid * 4.0))
    ficr = 0.0 if denominator <= 0.0 else float(
        np.sum(actual_valid * unit_price) / denominator
    )
    return 0.5 * (1.0 - nmae) + 0.5 * ficr


def metric_report(answer: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    _validate_frames(answer, prediction)
    groups: dict[str, dict[str, float | int]] = {}
    nmaes: list[float] = []
    ficrs: list[float] = []
    for target in TARGET_COLS:
        actual = answer[target].to_numpy(dtype=float)
        forecast = prediction[target].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[target]
        valid = np.isfinite(actual) & (actual >= capacity * 0.10)
        if not valid.any():
            raise ValueError(f"{target}: 평가 가능한 정답 행이 없습니다.")
        if not np.isfinite(forecast[valid]).all():
            raise ValueError(f"{target}: 평가 대상 예측에 NaN/inf가 있습니다.")
        actual_valid = actual[valid]
        error_rate = np.abs(forecast[valid] - actual_valid) / capacity
        nmae = float(error_rate.mean())
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0
        )
        ficr = float(np.sum(actual_valid * unit_price) / np.sum(actual_valid * 4.0))
        nmaes.append(nmae)
        ficrs.append(ficr)
        groups[target] = {
            "n_valid": int(valid.sum()),
            "nmae": nmae,
            "one_minus_nmae": 1.0 - nmae,
            "ficr": ficr,
            "within_6pct_rate": float(np.mean(error_rate <= 0.06)),
            "between_6_and_8pct_rate": float(
                np.mean((error_rate > 0.06) & (error_rate <= 0.08))
            ),
            "over_8pct_rate": float(np.mean(error_rate > 0.08)),
        }
    one_minus_nmae = 1.0 - float(np.mean(nmaes))
    ficr = float(np.mean(ficrs))
    return {
        "score": 0.5 * one_minus_nmae + 0.5 * ficr,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "groups": groups,
    }


def metric(
    answer: pd.DataFrame, prediction: pd.DataFrame
) -> tuple[float, float, float]:
    """Return ``(score, one_minus_nmae, ficr)``."""
    report = metric_report(answer, prediction)
    return report["score"], report["one_minus_nmae"], report["ficr"]


def report_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Convert the per-target section of a metric report to a DataFrame."""
    frame = pd.DataFrame.from_dict(report["groups"], orient="index")
    frame.index.name = "group"
    return frame


def evaluate_complete_rows(
    answer: pd.DataFrame, prediction: pd.DataFrame
) -> dict[str, Any]:
    usable = answer[TARGET_COLS].notna().all(axis=1)
    if not usable.any():
        raise ValueError("세 그룹 정답이 모두 있는 평가 행이 없습니다.")
    return metric_report(answer.loc[usable, TARGET_COLS], prediction.loc[usable, TARGET_COLS])


def flatten_report(model_name: str, report: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "score": report["score"],
        "one_minus_nmae": report["one_minus_nmae"],
        "ficr": report["ficr"],
    }
    for target, group in report["groups"].items():
        row[f"{target}__nmae"] = group["nmae"]
        row[f"{target}__ficr"] = group["ficr"]
    return row
