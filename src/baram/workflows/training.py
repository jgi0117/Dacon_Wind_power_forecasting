"""세 모델 학습·추론과 앙상블을 한 번에 실행하는 진입점."""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import PipelineConfig, parse_args
from ..constants import EPOCH_MODEL_NAMES, ITERATION_MODEL_NAMES, SUPPORTED_MODEL_NAMES
from ..data import load_artifacts, write_submission
from .iteration_cache import load_iteration_cache, save_iteration_cache
from ..metrics import (
    TARGET_COLS,
    capacity_factor,
    evaluate_complete_rows,
    flatten_report,
    restore_generation,
)
from ..models import build_model
from ..splitting import (
    IterationFold,
    SplitPlan,
    build_iteration_folds,
    build_split_plan,
    delivery_month,
)


LOGGER = logging.getLogger("baram.pipeline")


def _prediction_frames(
    arrays: dict[str, np.ndarray], index: pd.Index
) -> dict[str, pd.DataFrame]:
    return {
        name: pd.DataFrame(values, index=index, columns=TARGET_COLS)
        for name, values in arrays.items()
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _select_iteration_schedule(
    config: PipelineConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    folds: list[IterationFold],
    model_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    """여러 시계열 fold의 best iteration 중앙값을 그룹별로 선택한다."""
    schedule: dict[str, dict[str, int]] = {
        model_name: {} for model_name in model_names
    }
    audit: dict[str, dict[str, Any]] = {
        model_name: {} for model_name in model_names
    }
    for target in TARGET_COLS:
        for model_name in model_names:
            fold_iterations: list[int] = []
            fold_audit: list[dict[str, Any]] = []
            for fold in folds:
                train_mask = np.asarray(X_train.index < fold.train_cutoff)
                train_mask &= y_train[target].notna().to_numpy()
                valid_mask = np.asarray(
                    (X_train.index >= fold.validation_start)
                    & (X_train.index < fold.validation_end)
                )
                valid_mask &= y_train[target].notna().to_numpy()
                X_fold_train = X_train.loc[train_mask]
                X_fold_valid = X_train.loc[valid_mask]
                if X_fold_train.empty or X_fold_valid.empty:
                    raise ValueError(
                        f"{model_name}/{target}/{fold.name}: 학습 또는 검증 행이 없습니다."
                    )
                if X_fold_train.index.max() >= fold.train_cutoff:
                    raise RuntimeError("iteration fold 학습 정답 cutoff 위반입니다.")
                y_fold_train = capacity_factor(
                    y_train.loc[train_mask, target], target
                )
                y_fold_valid = capacity_factor(
                    y_train.loc[valid_mask, target], target
                )
                LOGGER.info(
                    "반복 수 탐색: %s / %s / %s", model_name, target, fold.name
                )
                model = build_model(model_name, config)
                model.fit(X_fold_train, y_fold_train, X_fold_valid, y_fold_valid)
                best_iteration = int(model.metadata()["best_iteration"])
                fold_iterations.append(best_iteration)
                fold_audit.append(
                    {
                        "fold": fold.name,
                        "train_end": X_fold_train.index.max(),
                        "train_cutoff_exclusive": fold.train_cutoff,
                        "validation_start": X_fold_valid.index.min(),
                        "validation_end": X_fold_valid.index.max(),
                        "n_train_rows": len(X_fold_train),
                        "n_validation_rows": len(X_fold_valid),
                        "best_iteration": best_iteration,
                    }
                )
            median_iteration = max(1, int(round(float(np.median(fold_iterations)))))
            schedule[model_name][target] = median_iteration
            audit[model_name][target] = {
                "folds": fold_audit,
                "fold_best_iterations": fold_iterations,
                "median_best_iteration": median_iteration,
            }
            LOGGER.info(
                "반복 수 중앙값: %s / %s = %d (%s)",
                model_name,
                target,
                median_iteration,
                fold_iterations,
            )
    return schedule, audit


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
    arrays = {
        name: np.zeros((len(X_validation), len(TARGET_COLS)), dtype=float)
        for name in config.models
    }
    metadata: dict[str, dict[str, Any]] = {name: {} for name in config.models}
    for target_index, target in enumerate(TARGET_COLS):
        available = y_fit[target].notna()
        X_target = X_fit.loc[available]
        y_target = capacity_factor(y_fit.loc[available, target], target)
        tune_available = early_stopping_mask & y_validation[target].notna().to_numpy()
        X_tune = X_validation.loc[tune_available]
        y_tune = capacity_factor(y_validation.loc[tune_available, target], target)
        for model_name in config.models:
            iterations = iteration_schedule.get(model_name, {}).get(target)
            LOGGER.info(
                "검증 모델 고정 반복 학습: %s / %s / iterations=%s",
                model_name,
                target,
                iterations,
            )
            model = build_model(model_name, config, iterations=iterations)
            if model_name in EPOCH_MODEL_NAMES:
                model.fit(X_target, y_target, X_tune, y_tune)
                best_iteration = int(model.metadata()['best_iteration'])
                iteration_schedule.setdefault(model_name, {})[target] = best_iteration
            else:
                model.fit(X_target, y_target)
            arrays[model_name][:, target_index] = restore_generation(
                model.predict(X_validation), target
            )
            metadata[model_name][target] = model.metadata()
            metadata[model_name][target]["n_fit_rows"] = int(len(X_target))
            if iterations is not None:
                metadata[model_name][target]["median_best_iteration"] = iterations
            del model
            gc.collect()
    return _prediction_frames(arrays, X_validation.index), metadata, iteration_schedule


def _fit_final_models(
    config: PipelineConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    iteration_schedule: dict[str, dict[str, int]],
    final_fit_cutoff: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    arrays = {
        name: np.zeros((len(X_test), len(TARGET_COLS)), dtype=float)
        for name in config.models
    }
    time_available = X_train.index < final_fit_cutoff
    for target_index, target in enumerate(TARGET_COLS):
        available = time_available & y_train[target].notna().to_numpy()
        X_target = X_train.loc[available]
        y_target = capacity_factor(y_train.loc[available, target], target)
        if X_target.index.max() >= final_fit_cutoff:
            raise RuntimeError("최종 학습 정답에 예측기준시점 이후 행이 포함됐습니다.")
        for model_name in config.models:
            iterations = iteration_schedule.get(model_name, {}).get(target)
            LOGGER.info(
                "최종 모델 고정 반복 학습 및 추론: %s / %s / iterations=%s",
                model_name,
                target,
                iterations,
            )
            model = build_model(model_name, config, iterations=iterations)
            model.fit(X_target, y_target)
            arrays[model_name][:, target_index] = restore_generation(
                model.predict(X_test), target
            )
            del model
            gc.collect()
    return _prediction_frames(arrays, X_test.index)


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
    """누수 없는 평가부터 네 제출 파일 생성까지 실행한다."""
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
    iteration_models = tuple(
        name for name in config.models if name in ITERATION_MODEL_NAMES
    )
    iteration_folds = build_iteration_folds(X_train, config) if iteration_models else []
    cached_iterations = (
        load_iteration_cache(
            config,
            X_train,
            iteration_folds,
            required_models=iteration_models,
        )
        if iteration_models
        else ({}, {})
    )
    if cached_iterations is None:
        iteration_schedule, iteration_audit = _select_iteration_schedule(
            config, X_train, y_train, iteration_folds, iteration_models
        )
        cache_path = save_iteration_cache(
            config, X_train, iteration_folds, iteration_schedule, iteration_audit
        )
        LOGGER.info("반복 수 탐색 결과 저장: %s", cache_path)
    else:
        iteration_schedule, iteration_audit = cached_iterations
        if iteration_models:
            LOGGER.info(
                "저장된 반복 수 탐색 결과를 재사용합니다: %s",
                config.iteration_cache_dir / "iteration_selection_cache.json",
            )
    iteration_schedule = {
        name: iteration_schedule[name] for name in iteration_models
    }
    iteration_audit = {name: iteration_audit[name] for name in iteration_models}
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
            'metric': 'competition-score',
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
