"""Teacher-student RealMLP training plus optional G3 residual correction."""

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
from ..models.g3_residual_model import Group3ResidualRealMLP, G3_TARGET
from ..splitting import SplitPlan, build_split_plan, delivery_month


LOGGER = logging.getLogger("baram.pipeline")

MULTITASK_STRATEGY = (
    "v6-temporal-x-5h-student-privileged-same-group-y-teacher-distillation"
)
G3_RESIDUAL_NAME = "realmlp_g3_residual"


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
    prediction: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    values = np.asarray(prediction, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    expected_shape = (len(index), len(TARGET_COLS))
    if values.shape != expected_shape:
        raise ValueError(
            f"Multi-task prediction shape mismatch: "
            f"{values.shape} != {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Multi-task prediction contains NaN or infinite values.")

    frame = pd.DataFrame(
        np.clip(values, 0.0, 1.0),
        index=index,
        columns=TARGET_COLS,
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
    dict[str, pd.DataFrame],
    dict[str, dict[str, Any]],
    dict[str, dict[str, int]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    predictions: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {
        name: {} for name in config.models
    }

    splits = [
        {
            "label": "student-epoch-selection",
            "start": X_validation.index.min(),
            "end_exclusive": pd.Timestamp(config.iteration_selection_end),
        },
        {
            "label": "outer-comparison",
            "start": pd.Timestamp(config.comparison_start),
            "end_exclusive": None,
        },
    ]

    audit: dict[str, Any] = {
        "method": MULTITASK_STRATEGY,
        "student_history_hours": int(config.student_history_hours),
        "teacher_history_hours": int(config.teacher_history_hours),
        "teacher_oof_folds": int(config.teacher_oof_folds),
        "teacher_weight": float(config.distillation_teacher_weight),
        "student_epoch_window": splits[0],
        "outer_evaluation": "2024Q4",
        "runs": {},
    }

    for model_name in config.models:
        X_joint = X_fit
        y_joint = _capacity_factor_frame(y_fit)
        X_tune = X_validation
        y_tune = _capacity_factor_frame(y_validation)
        iterations = iteration_schedule.get(model_name) or None

        LOGGER.info(
            "검증 teacher-student 모델 학습: %s / rows=%d / iterations=%s",
            model_name,
            len(X_joint),
            iterations,
        )

        model = build_model(
            model_name,
            config,
            iterations=iterations,
        )
        model.fit(
            X_joint,
            y_joint,
            X_tune,
            y_tune,
        )
        model_metadata = model.metadata()

        model_schedule = iteration_schedule.setdefault(
            model_name,
            {},
        )
        model_schedule["student"] = int(
            model_metadata.get(
                "student_best_iteration",
                model_metadata.get(
                    "best_iteration",
                    config.max_epochs,
                ),
            )
        )

        predictions[model_name] = _restore_prediction_frame(
            model.predict(X_validation),
            X_validation.index,
        )

        model_metadata["selection_metric"] = (
            "chronological-ficr-aware-student-selection"
        )
        model_metadata["n_fit_rows"] = int(len(X_joint))
        model_metadata["multitask_strategy"] = MULTITASK_STRATEGY
        model_metadata["fit_start"] = X_joint.index.min()
        model_metadata["fit_end"] = X_joint.index.max()
        model_metadata["n_fit_rows_by_target"] = {
            target: int(y_joint[target].notna().sum())
            for target in TARGET_COLS
        }
        model_metadata["scheduled_iteration"] = dict(model_schedule)
        metadata[model_name]["multitask"] = model_metadata

        audit["runs"][model_name] = {
            "student_epoch": model_schedule["student"],
            "teacher_oof": model_metadata.get("teacher_oof", {}),
        }
        audit.setdefault("selected_epochs", {})[model_name] = dict(
            model_schedule
        )

    return (
        predictions,
        metadata,
        iteration_schedule,
        splits,
        audit,
    )


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
    X_joint = X_train
    y_joint = _capacity_factor_frame(y_train)
    y_joint.loc[~time_available, TARGET_COLS] = np.nan

    observed_rows = y_joint[TARGET_COLS].notna().any(axis=1)
    if (
        observed_rows.any()
        and y_joint.index[observed_rows].max() >= final_fit_cutoff
    ):
        raise RuntimeError(
            "최종 학습 정답에 예측기준시점 이후 행이 포함됐습니다."
        )

    for model_name in config.models:
        iterations = iteration_schedule.get(model_name) or None

        LOGGER.info(
            "최종 teacher-student 모델 학습 및 추론: "
            "%s / rows=%d / iterations=%s",
            model_name,
            len(X_joint),
            iterations,
        )

        model = build_model(
            model_name,
            config,
            iterations=iterations,
        )
        model.fit(
            X_joint,
            y_joint,
        )

        predictions[model_name] = _restore_prediction_frame(
            model.predict(X_test),
            X_test.index,
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
                answer.loc[month_mask],
                prediction.loc[month_mask],
            )
            row = flatten_report(model_name, report)
            row["delivery_month"] = str(period)
            rows.append(row)

    columns = ["delivery_month", "model"]
    result = pd.DataFrame(rows)

    return result[
        columns
        + [
            column
            for column in result
            if column not in columns
        ]
    ]


def _build_masks(
    X_train: pd.DataFrame,
    plan: SplitPlan,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    fit_mask = np.asarray(
        X_train.index < plan.validation_fit_cutoff
    )
    validation_mask = np.asarray(
        X_train.index >= plan.validation_start
    )

    validation_index = X_train.index[validation_mask]

    early_stopping_mask = np.asarray(
        validation_index < plan.iteration_selection_end
    )
    comparison_mask = np.asarray(
        validation_index >= plan.comparison_start
    )

    if not all(
        mask.any()
        for mask in (
            fit_mask,
            validation_mask,
            early_stopping_mask,
            comparison_mask,
        )
    ):
        raise ValueError(
            "학습·학생 epoch 선택·최종 비교 중 빈 구간이 있습니다."
        )

    return (
        fit_mask,
        validation_mask,
        early_stopping_mask,
        comparison_mask,
    )


def _fit_g3_residual_validation(
    config: PipelineConfig,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    base_prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit residual on Jan-Jun, inspect Jul-Sep, evaluate on untouched Q4."""
    residual_validation_start = pd.Timestamp(
        config.group3_residual_validation_start
    )
    comparison_start = pd.Timestamp(
        config.comparison_start
    )

    observed = y_validation[G3_TARGET].notna()

    train_mask = (
        observed
        & (X_validation.index < residual_validation_start)
    )
    tuning_mask = (
        observed
        & (X_validation.index >= residual_validation_start)
        & (X_validation.index < comparison_start)
    )

    if not train_mask.any():
        raise ValueError(
            "No rows are available for G3 residual training."
        )
    if not tuning_mask.any():
        raise ValueError(
            "No rows are available for G3 residual validation."
        )

    model = Group3ResidualRealMLP(config)

    LOGGER.info(
        "G3 residual validation model: train=%d, tuning=%d, epochs=%d",
        int(train_mask.sum()),
        int(tuning_mask.sum()),
        config.group3_residual_epochs,
    )

    model.fit(
        X_validation.loc[train_mask],
        base_prediction.loc[train_mask, G3_TARGET],
        y_validation.loc[train_mask, G3_TARGET],
        X_valid=X_validation.loc[tuning_mask],
        base_prediction_valid_kw=base_prediction.loc[
            tuning_mask,
            G3_TARGET,
        ],
        actual_valid_kw=y_validation.loc[
            tuning_mask,
            G3_TARGET,
        ],
    )

    corrected = base_prediction.copy()
    corrected[G3_TARGET] = model.correct_prediction_kw(
        X_validation,
        base_prediction[G3_TARGET],
    )

    metadata = model.metadata()
    metadata.update(
        {
            "residual_train_start": (
                X_validation.index[train_mask].min()
            ),
            "residual_train_end": (
                X_validation.index[train_mask].max()
            ),
            "residual_validation_start": (
                X_validation.index[tuning_mask].min()
            ),
            "residual_validation_end": (
                X_validation.index[tuning_mask].max()
            ),
            "outer_comparison_start": comparison_start,
            "outer_comparison_used_for_fit": False,
        }
    )

    return corrected, metadata


def _fit_g3_residual_final(
    config: PipelineConfig,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    base_validation_prediction: pd.DataFrame,
    X_test: pd.DataFrame,
    base_test_prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit residual on all available 2024 OOS base predictions."""
    observed = y_validation[G3_TARGET].notna()

    if not observed.any():
        raise ValueError(
            "No observed group 3 labels for final residual fit."
        )

    model = Group3ResidualRealMLP(config)

    LOGGER.info(
        "G3 residual final refit: rows=%d / epochs=%d",
        int(observed.sum()),
        config.group3_residual_epochs,
    )

    model.fit(
        X_validation.loc[observed],
        base_validation_prediction.loc[
            observed,
            G3_TARGET,
        ],
        y_validation.loc[
            observed,
            G3_TARGET,
        ],
    )

    corrected_test = base_test_prediction.copy()
    corrected_test[G3_TARGET] = model.correct_prediction_kw(
        X_test,
        base_test_prediction[G3_TARGET],
    )

    metadata = model.metadata()
    metadata.update(
        {
            "final_residual_fit_start": (
                X_validation.index[observed].min()
            ),
            "final_residual_fit_end": (
                X_validation.index[observed].max()
            ),
            "final_residual_fit_rows": int(observed.sum()),
            "test_rows": int(len(X_test)),
        }
    )

    return corrected_test, metadata


def run_pipeline(
    config: PipelineConfig,
) -> pd.DataFrame:
    """Run Version 6 base model and optional G3 residual experiment."""
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not config.models:
        raise ValueError(
            "At least one model must be selected."
        )

    unknown_models = (
        set(config.models)
        - set(SUPPORTED_MODEL_NAMES)
    )
    if unknown_models:
        raise ValueError(
            f"Unsupported models: {sorted(unknown_models)}"
        )

    if len(set(config.models)) != len(config.models):
        raise ValueError(
            "Duplicate model names are not allowed."
        )

    if config.group3_residual and tuple(config.models) != ("realmlp",):
        raise ValueError(
            "G3 residual experiment currently requires --models realmlp."
        )

    X_train, y_train, X_test = load_artifacts(
        config.artifacts_dir
    )

    plan = build_split_plan(
        X_train,
        X_test,
        config,
    )

    iteration_schedule: dict[
        str,
        dict[str, int],
    ] = {}

    (
        fit_mask,
        validation_mask,
        early_stopping_mask,
        comparison_mask,
    ) = _build_masks(
        X_train,
        plan,
    )

    fit_feature_mask = np.asarray(
        X_train.index < plan.validation_start
    )

    X_fit = X_train.loc[
        fit_feature_mask
    ]
    y_fit = y_train.loc[
        fit_feature_mask
    ].copy()

    y_fit.loc[
        y_fit.index >= plan.validation_fit_cutoff,
        TARGET_COLS,
    ] = np.nan

    X_validation = X_train.loc[
        validation_mask
    ]
    y_validation = y_train.loc[
        validation_mask
    ]

    purged_validation_rows = int(
        np.sum(
            (
                X_train.index
                >= plan.validation_fit_cutoff
            )
            & (
                X_train.index
                < plan.validation_start
            )
        )
    )

    LOGGER.info(
        "학습=%d, outer purge=%d, validation splits=%d, "
        "최종 비교=%d, 특성=%d",
        len(X_fit),
        purged_validation_rows,
        2,
        comparison_mask.sum(),
        X_train.shape[1],
    )

    (
        validation_predictions,
        metadata,
        iteration_schedule,
        iteration_splits,
        iteration_audit,
    ) = _fit_validation_models(
        config,
        X_fit,
        y_fit,
        X_validation,
        y_validation,
        early_stopping_mask,
        iteration_schedule,
    )

    residual_validation_metadata: dict[str, Any] | None = None

    if config.group3_residual:
        base_validation_prediction = validation_predictions["realmlp"]

        corrected_validation, residual_validation_metadata = (
            _fit_g3_residual_validation(
                config,
                X_validation,
                y_validation,
                base_validation_prediction,
            )
        )

        validation_predictions[
            G3_RESIDUAL_NAME
        ] = corrected_validation

    comparison_predictions = {
        name: frame.loc[
            comparison_mask
        ]
        for name, frame
        in validation_predictions.items()
    }

    comparison_answer = y_validation.loc[
        comparison_mask
    ]

    reports = {
        name: evaluate_complete_rows(
            comparison_answer,
            prediction,
        )
        for name, prediction
        in comparison_predictions.items()
    }

    results = pd.DataFrame(
        [
            flatten_report(
                name,
                report,
            )
            for name, report
            in reports.items()
        ]
    ).sort_values(
        "score",
        ascending=False,
        ignore_index=True,
    )

    results.to_csv(
        config.output_dir
        / "evaluation_results.csv",
        index=False,
        encoding="utf-8",
    )

    _monthly_evaluation(
        comparison_answer,
        comparison_predictions,
    ).to_csv(
        config.output_dir
        / "evaluation_results_by_month.csv",
        index=False,
        encoding="utf-8",
    )

    pd.concat(
        validation_predictions,
        names=[
            "model",
            "forecast_kst_dtm",
        ],
    ).reset_index().to_csv(
        config.output_dir
        / "validation_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    report_payload: dict[str, Any] = {
        "version": "6",
        "config": asdict(config),
        "split": {
            "validation_fit_start": X_fit.index.min(),
            "validation_fit_end": X_fit.index.max(),
            "validation_fit_cutoff_exclusive": (
                plan.validation_fit_cutoff
            ),
            "purged_rows_before_validation": (
                purged_validation_rows
            ),
            "iteration_selection_splits": (
                iteration_splits
            ),
            "comparison_start": (
                X_validation.index[
                    comparison_mask
                ].min()
            ),
            "comparison_end": (
                X_validation.index[
                    comparison_mask
                ].max()
            ),
            "test_start": plan.test_start,
            "final_fit_end": (
                X_train.index[
                    X_train.index
                    < plan.final_fit_cutoff
                ].max()
            ),
            "final_fit_cutoff_exclusive": (
                plan.final_fit_cutoff
            ),
            "purged_rows_before_test": int(
                np.sum(
                    X_train.index
                    >= plan.final_fit_cutoff
                )
            ),
            "final_fit_rows_by_target": {
                target: int(
                    np.sum(
                        (
                            X_train.index
                            < plan.final_fit_cutoff
                        )
                        & y_train[
                            target
                        ].notna().to_numpy()
                    )
                )
                for target in TARGET_COLS
            },
        },
        "iteration_selection": iteration_audit,
        "iteration_schedule": iteration_schedule,
        "model_metadata": metadata,
        "g3_residual_validation": (
            residual_validation_metadata
        ),
        "evaluation_reports": reports,
    }

    (
        config.output_dir
        / "run_report.json"
    ).write_text(
        json.dumps(
            report_payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "평가 결과\n%s",
        results.to_string(index=False),
    )

    if config.evaluation_only:
        return results

    base_test_predictions = _fit_final_models(
        config,
        X_train,
        y_train,
        X_test,
        iteration_schedule,
        plan.final_fit_cutoff,
    )

    for model_name, prediction in (
        base_test_predictions.items()
    ):
        write_submission(
            config.data_dir
            / "sample_submission.csv",
            prediction,
            config.output_dir
            / f"submission_{model_name}.csv",
        )

    if config.group3_residual:
        base_validation_prediction = (
            validation_predictions["realmlp"]
        )
        base_test_prediction = (
            base_test_predictions["realmlp"]
        )

        (
            corrected_test_prediction,
            final_residual_metadata,
        ) = _fit_g3_residual_final(
            config,
            X_validation,
            y_validation,
            base_validation_prediction,
            X_test,
            base_test_prediction,
        )

        write_submission(
            config.data_dir
            / "sample_submission.csv",
            corrected_test_prediction,
            config.output_dir
            / f"submission_{G3_RESIDUAL_NAME}.csv",
        )

        report_payload[
            "g3_residual_final"
        ] = final_residual_metadata

        (
            config.output_dir
            / "run_report.json"
        ).write_text(
            json.dumps(
                report_payload,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

    LOGGER.info(
        "submission files created: output=%s",
        config.output_dir,
    )

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    run_pipeline(
        parse_args()
    )


if __name__ == "__main__":
    main()
