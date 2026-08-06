"""Multi-task RealMLP 학습·평가·제출 생성을 실행하는 진입점."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import PipelineConfig, parse_args
from ..constants import SUPPORTED_MODEL_NAMES
from ..data import load_artifacts, write_submission
from ..metrics import (
    CAPACITY_KWH,
    TARGET_COLS,
    evaluate_complete_rows,
    flatten_report,
)
from ..models import build_model
from ..splitting import SplitPlan, build_split_plan, delivery_month


LOGGER = logging.getLogger("baram.pipeline")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _capacity_factor_frame(targets: pd.DataFrame) -> pd.DataFrame:
    result = targets.loc[:, TARGET_COLS].astype(float).copy()
    for target in TARGET_COLS:
        result[target] /= CAPACITY_KWH[target]
    return result


def _restore_prediction_frame(
    prediction: np.ndarray, index: pd.Index
) -> pd.DataFrame:
    values = np.asarray(prediction, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    expected_shape = (len(index), len(TARGET_COLS))
    if values.shape != expected_shape:
        raise ValueError(
            f'Multi-task prediction shape mismatch: {values.shape} != {expected_shape}'
        )
    if not np.isfinite(values).all():
        raise ValueError('Multi-task prediction contains NaN or infinite values.')
    frame = pd.DataFrame(
        np.clip(values, 0.0, 1.0), index=index, columns=TARGET_COLS
    )
    for target in TARGET_COLS:
        frame[target] *= CAPACITY_KWH[target]
    return frame


def _fit_validation_models(
    config: PipelineConfig,
    X_fit: pd.DataFrame,
    y_fit: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    early_stopping_mask: np.ndarray,
    iteration_schedule: dict[str, dict[str, int]],
) -> tuple[
    dict[str, pd.DataFrame], dict[str, dict[str, Any]],
    dict[str, dict[str, int]],
]:
    predictions: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {name: {} for name in config.models}
    fit_available = y_fit[TARGET_COLS].notna().any(axis=1)
    tune_available = (
        early_stopping_mask
        & y_validation[TARGET_COLS].notna().any(axis=1).to_numpy()
    )
    X_joint = X_fit.loc[fit_available]
    y_joint = _capacity_factor_frame(y_fit.loc[fit_available])
    X_tune = X_validation.loc[tune_available]
    y_tune = _capacity_factor_frame(y_validation.loc[tune_available])
    for model_name in config.models:
        iterations = iteration_schedule.get(model_name, {}).get('multitask')
        LOGGER.info(
            "검증 multi-task 모델 학습: %s / rows=%d / iterations=%s",
            model_name,
            len(X_joint),
            iterations,
        )
        model = build_model(model_name, config, iterations=iterations)
        model.fit(X_joint, y_joint, X_tune, y_tune)
        model_metadata = model.metadata()
        best_iteration = int(model_metadata['best_iteration'])
        iteration_schedule.setdefault(model_name, {})['multitask'] = best_iteration
        predictions[model_name] = _restore_prediction_frame(
            model.predict(X_validation), X_validation.index
        )
        model_metadata['n_fit_rows'] = int(len(X_joint))
        model_metadata['n_fit_rows_by_target'] = {
            target: int(y_joint[target].notna().sum())
            for target in TARGET_COLS
        }
        if iterations is not None:
            model_metadata['scheduled_iteration'] = iterations
        metadata[model_name]['multitask'] = model_metadata
    return predictions, metadata, iteration_schedule


def _fit_final_models(
    config: PipelineConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    iteration_schedule: dict[str, dict[str, int]],
    final_fit_cutoff: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    time_available = X_train.index < final_fit_cutoff
    available = time_available & y_train[TARGET_COLS].notna().any(axis=1).to_numpy()
    X_joint = X_train.loc[available]
    y_joint = _capacity_factor_frame(y_train.loc[available])
    if X_joint.index.max() >= final_fit_cutoff:
        raise RuntimeError("최종 학습 정답에 예측기준시점 이후 행이 포함됐습니다.")
    for model_name in config.models:
        iterations = iteration_schedule.get(model_name, {}).get('multitask')
        LOGGER.info(
            "최종 multi-task 모델 학습 및 추론: %s / rows=%d / iterations=%s",
            model_name,
            len(X_joint),
            iterations,
        )
        model = build_model(model_name, config, iterations=iterations)
        model.fit(X_joint, y_joint)
        predictions[model_name] = _restore_prediction_frame(
            model.predict(X_test), X_test.index
        )
    return predictions


def _monthly_evaluation(
    answer: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    periods = delivery_month(pd.DatetimeIndex(answer.index))
    rows: list[dict[str, Any]] = []
    for period in periods.unique().sort_values():
        month_mask = np.asarray(periods == period)
        for model_name, prediction in predictions.items():
            report = evaluate_complete_rows(
                answer.loc[month_mask], prediction.loc[month_mask]
            )
            row = flatten_report(model_name, report)
            row["delivery_month"] = str(period)
            rows.append(row)
    columns = ["delivery_month", "model"]
    result = pd.DataFrame(rows)
    return result[columns + [column for column in result if column not in columns]]


def _build_masks(
    X_train: pd.DataFrame, plan: SplitPlan
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit_mask = np.asarray(X_train.index < plan.validation_fit_cutoff)
    validation_mask = np.asarray(X_train.index >= plan.validation_start)
    validation_index = X_train.index[validation_mask]
    early_stopping_mask = np.asarray(
        validation_index < plan.iteration_selection_end
    )
    comparison_mask = np.asarray(validation_index >= plan.comparison_start)
    if not all(mask.any() for mask in (fit_mask, early_stopping_mask, comparison_mask)):
        raise ValueError("학습·앙상블 보정·최종 비교 중 빈 구간이 있습니다.")
    return fit_mask, validation_mask, early_stopping_mask, comparison_mask


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """누수 없는 평가부터 multi-task 제출 파일 생성까지 실행한다."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.models:
        raise ValueError("At least one model must be selected.")
    unknown_models = set(config.models) - set(SUPPORTED_MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unsupported models: {sorted(unknown_models)}")
    if len(set(config.models)) != len(config.models):
        raise ValueError("Duplicate model names are not allowed.")
    X_train, y_train, X_test = load_artifacts(config.artifacts_dir)
    plan = build_split_plan(X_train, X_test, config)
    iteration_folds: list[Any] = []
    iteration_schedule: dict[str, dict[str, int]] = {}
    iteration_audit: dict[str, Any] = {}
    fit_mask, validation_mask, early_stopping_mask, comparison_mask = _build_masks(
        X_train, plan
    )

    X_fit, y_fit = X_train.loc[fit_mask], y_train.loc[fit_mask]
    X_validation = X_train.loc[validation_mask]
    y_validation = y_train.loc[validation_mask]
    purged_validation_rows = int(
        np.sum(
            (X_train.index >= plan.validation_fit_cutoff)
            & (X_train.index < plan.validation_start)
        )
    )
    LOGGER.info(
        "학습=%d, purge=%d, iteration folds=%d, 최종 비교=%d, 특성=%d",
        len(X_fit),
        purged_validation_rows,
        len(iteration_folds),
        comparison_mask.sum(),
        X_train.shape[1],
    )

    validation_predictions, metadata, iteration_schedule = _fit_validation_models(
        config,
        X_fit,
        y_fit,
        X_validation,
        y_validation,
        early_stopping_mask,
        iteration_schedule,
    )
    comparison_predictions = {
        name: frame.loc[comparison_mask]
        for name, frame in validation_predictions.items()
    }
    comparison_answer = y_validation.loc[comparison_mask]
    reports = {
        name: evaluate_complete_rows(comparison_answer, prediction)
        for name, prediction in comparison_predictions.items()
    }
    results = pd.DataFrame(
        [flatten_report(name, report) for name, report in reports.items()]
    ).sort_values("score", ascending=False, ignore_index=True)
    results.to_csv(
        config.output_dir / "evaluation_results.csv", index=False, encoding="utf-8"
    )
    _monthly_evaluation(comparison_answer, comparison_predictions).to_csv(
        config.output_dir / "evaluation_results_by_month.csv",
        index=False,
        encoding="utf-8",
    )
    pd.concat(
        validation_predictions, names=["model", "forecast_kst_dtm"]
    ).reset_index().to_csv(
        config.output_dir / "validation_predictions.csv", index=False, encoding="utf-8"
    )

    report_payload = {
        'early_stopping_window': {
            'start': X_validation.index[early_stopping_mask].min(),
            'end': X_validation.index[early_stopping_mask].max(),
            'metric': 'ficr-aware-loss',
        },
        "config": asdict(config),
        "split": {
            "validation_fit_start": X_fit.index.min(),
            "validation_fit_end": X_fit.index.max(),
            "validation_fit_cutoff_exclusive": plan.validation_fit_cutoff,
            "purged_rows_before_validation": purged_validation_rows,
            "iteration_selection_folds": [asdict(fold) for fold in iteration_folds],
            "comparison_start": X_validation.index[comparison_mask].min(),
            "comparison_end": X_validation.index[comparison_mask].max(),
            "test_start": plan.test_start,
            "final_fit_end": X_train.index[
                X_train.index < plan.final_fit_cutoff
            ].max(),
            "final_fit_cutoff_exclusive": plan.final_fit_cutoff,
            "purged_rows_before_test": int(
                np.sum(X_train.index >= plan.final_fit_cutoff)
            ),
            "final_fit_rows_by_target": {
                target: int(
                    np.sum(
                        (X_train.index < plan.final_fit_cutoff)
                        & y_train[target].notna().to_numpy()
                    )
                )
                for target in TARGET_COLS
            },
        },
        "iteration_selection": iteration_audit,
        "iteration_schedule": iteration_schedule,
        "model_metadata": metadata,
        "evaluation_reports": reports,
    }
    (config.output_dir / "run_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    LOGGER.info("평가 결과\n%s", results.to_string(index=False))

    if config.evaluation_only:
        return results

    test_predictions = _fit_final_models(
        config,
        X_train,
        y_train,
        X_test,
        iteration_schedule,
        plan.final_fit_cutoff,
    )
    for model_name, prediction in test_predictions.items():
        write_submission(
            config.data_dir / "sample_submission.csv",
            prediction,
            config.output_dir / f"submission_{model_name}.csv",
        )
    LOGGER.info(
        "submission files created: count=%d, output=%s",
        len(test_predictions),
        config.output_dir,
    )
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
