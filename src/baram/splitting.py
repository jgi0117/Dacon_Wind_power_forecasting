"""예측기준시점과 01시~익일 00시 배치를 지키는 시간 분할."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig


@dataclass(frozen=True)
class SplitPlan:
    validation_start: pd.Timestamp
    iteration_selection_end: pd.Timestamp
    comparison_start: pd.Timestamp
    validation_fit_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    final_fit_cutoff: pd.Timestamp


def forecast_cutoff(batch_start: pd.Timestamp) -> pd.Timestamp:
    """01시에 시작하는 예측 배치의 전일 14시 예측기준시점을 반환한다."""
    batch_start = pd.Timestamp(batch_start)
    return batch_start.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=14)


def _validate_batch_boundary(
    X: pd.DataFrame, timestamp: pd.Timestamp, boundary_name: str
) -> None:
    if timestamp not in X.index:
        raise ValueError(f"{boundary_name}={timestamp}가 특성 인덱스에 없습니다.")
    if "time__interval_hour" not in X.columns:
        raise ValueError("배치 경계 검증에 필요한 time__interval_hour가 없습니다.")
    interval_hour = float(X.loc[timestamp, "time__interval_hour"])
    if not np.isclose(interval_hour, 0.0):
        raise ValueError(
            f"{boundary_name}={timestamp}는 예측 배치 시작이 아닙니다. "
            f"time__interval_hour={interval_hour}; 01:00 경계를 사용하세요."
        )


def build_split_plan(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: PipelineConfig,
) -> SplitPlan:
    validation_start = pd.Timestamp(config.validation_start)
    iteration_selection_end = pd.Timestamp(config.iteration_selection_end)
    comparison_start = pd.Timestamp(config.comparison_start)
    if not validation_start < iteration_selection_end < comparison_start:
        raise ValueError(
            "validation_start < iteration_selection_end < comparison_start 순서여야 합니다."
        )
    for name, timestamp in (
        ("validation_start", validation_start),
        ("iteration_selection_end", iteration_selection_end),
        ("comparison_start", comparison_start),
    ):
        _validate_batch_boundary(X_train, timestamp, name)

    test_start = pd.Timestamp(X_test.index.min())
    _validate_batch_boundary(X_test, test_start, "test_start")
    return SplitPlan(
        validation_start=validation_start,
        iteration_selection_end=iteration_selection_end,
        comparison_start=comparison_start,
        validation_fit_cutoff=forecast_cutoff(validation_start),
        test_start=test_start,
        final_fit_cutoff=forecast_cutoff(test_start),
    )


def delivery_month(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    """익일 00시를 전날 배치에 귀속시킨 월을 반환한다."""
    return (index - pd.Timedelta(hours=1)).to_period("M")
