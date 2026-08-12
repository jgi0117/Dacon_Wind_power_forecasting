"""Temporal-X Student + privileged-y Teacher RealMLP distillation.

Student input during training and inference
-------------------------------------------
    current weather/time
    + previous N hours of weather/time
    + recency-weighted X-history summary
    + group/turbine metadata

Teacher input during training
-----------------------------
    exactly the same Student input
    + previous M hours of the SAME group's true target

The Teacher therefore differs from the Student only by privileged target
history. True target history is never used by the Student at inference.

Teacher epoch selection is performed once on the earliest leakage-free
chronological history. The selected epoch is then reused for every expanding
OOF Teacher fold and for the final full-history distillation fit. Each OOF fold
therefore trains the Teacher only once.

The Student selector uses the external chronological validation window. The
selected Student epoch is reused for the final refit.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import TARGET_COLS
import baram.metrics as metrics_module

from .base import RegressionModel
from .group_conditioned_realmlp_model import GroupConditionedRealMLPModel


LOGGER = logging.getLogger("baram.pipeline")

_X_LAG_PREFIX = "student_feature__lag_"
_X_WEIGHTED_PREFIX = "student_feature__weighted_recent__"
_TEACHER_TARGET_PREFIX = "teacher_target__"


@contextmanager
def _worst_group_ficr_scope(weight: float):
    """Temporarily set FICR worst-group regularization for one model role."""
    previous = metrics_module.WORST_GROUP_FICR_REG_WEIGHT
    metrics_module.WORST_GROUP_FICR_REG_WEIGHT = float(weight)
    try:
        yield
    finally:
        metrics_module.WORST_GROUP_FICR_REG_WEIGHT = previous



def _prediction_frame(
    model: GroupConditionedRealMLPModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    values = np.asarray(model.predict(features), dtype=np.float32)
    expected = (len(features), len(TARGET_COLS))
    if values.shape != expected:
        raise ValueError(
            f"Teacher prediction shape mismatch: {values.shape} != {expected}"
        )
    return pd.DataFrame(
        values,
        index=features.index,
        columns=TARGET_COLS,
    )


def _combine_context(
    features: pd.DataFrame,
    history_hours: int,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepend only the last needed context rows before building lags."""
    if context is None or context.empty:
        return features.copy()

    context_tail = context.tail(history_hours)
    combined = pd.concat([context_tail, features], axis=0)

    # A duplicated boundary timestamp should belong to `features`.
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _append_feature_history(
    features: pd.DataFrame,
    history_hours: int,
    decay: float,
    *,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build inference-safe X history shared by Student and Teacher.

    Keeps raw lag1..lagN and adds one exponentially weighted summary per
    numeric feature. The summary gives more influence to recent observations
    without replacing the raw lag features.
    """
    if history_hours < 1:
        raise ValueError("history_hours must be at least 1.")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1].")

    combined = _combine_context(
        features,
        history_hours,
        context,
    )

    result = combined.copy()
    lag_frames: list[pd.DataFrame] = []

    for lag in range(1, history_hours + 1):
        shifted = combined.shift(lag).add_prefix(
            f"{_X_LAG_PREFIX}{lag:02d}__"
        )
        lag_frames.append(shifted)

    if lag_frames:
        result = pd.concat([result, *lag_frames], axis=1)

    # Weighted summary only for numeric source columns.
    numeric_columns = list(
        combined.select_dtypes(
            include=[np.number, "bool"],
        ).columns
    )

    if numeric_columns:
        weights = np.asarray(
            [decay ** i for i in range(history_hours)],
            dtype=np.float64,
        )
        weights /= weights.sum()

        weighted = pd.DataFrame(
            0.0,
            index=combined.index,
            columns=numeric_columns,
            dtype=np.float64,
        )

        valid = pd.DataFrame(
            True,
            index=combined.index,
            columns=numeric_columns,
        )

        for lag, weight in enumerate(weights, start=1):
            shifted = combined[numeric_columns].shift(lag)
            weighted += shifted.fillna(0.0) * float(weight)
            valid &= shifted.notna()

        weighted = weighted.where(valid)
        weighted = weighted.add_prefix(_X_WEIGHTED_PREFIX)
        result = pd.concat([result, weighted], axis=1)

    return result.loc[features.index]


def _student_history_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(_X_LAG_PREFIX)
        or column.startswith(_X_WEIGHTED_PREFIX)
    ]


def _student_history_ready(frame: pd.DataFrame) -> np.ndarray:
    """Rows where every generated Student temporal-X feature is available."""
    columns = _student_history_columns(frame)
    if not columns:
        return np.ones(len(frame), dtype=bool)

    return (
        frame.loc[:, columns]
        .notna()
        .all(axis=1)
        .to_numpy()
    )


def _append_teacher_target_history(
    student_features: pd.DataFrame,
    targets: pd.DataFrame,
    history_hours: int,
) -> pd.DataFrame:
    """Append only privileged y lag1..lagN to the shared Student features."""
    if history_hours < 1:
        raise ValueError("history_hours must be at least 1.")

    if not student_features.index.equals(targets.index):
        targets = targets.reindex(student_features.index)

    missing_targets = set(TARGET_COLS) - set(targets.columns)
    if missing_targets:
        raise ValueError(
            f"Missing teacher target columns: {sorted(missing_targets)}"
        )

    blocks = [student_features.copy()]

    for lag in range(1, history_hours + 1):
        shifted_y = (
            targets.loc[:, TARGET_COLS]
            .shift(lag)
            .rename(
                columns={
                    target: f"teacher_target__{target}__lag_{lag:02d}"
                    for target in TARGET_COLS
                }
            )
        )
        blocks.append(shifted_y)

    return pd.concat(blocks, axis=1)


def _group_teacher_eligibility(
    privileged_X: pd.DataFrame,
    y: pd.DataFrame,
    history_hours: int,
) -> dict[str, np.ndarray]:
    """Teacher eligibility is evaluated independently for each target group."""
    eligibility: dict[str, np.ndarray] = {}

    student_ready = _student_history_ready(privileged_X)

    for target in TARGET_COLS:
        lag_columns = [
            f"teacher_target__{target}__lag_{lag:02d}"
            for lag in range(1, history_hours + 1)
        ]

        missing_columns = [
            column
            for column in lag_columns
            if column not in privileged_X.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{target}: missing privileged history columns: "
                f"{missing_columns}"
            )

        target_history_ready = (
            privileged_X.loc[:, lag_columns]
            .notna()
            .all(axis=1)
            .to_numpy()
        )

        observed_current = y[target].notna().to_numpy()

        eligibility[target] = (
            student_ready
            & target_history_ready
            & observed_current
        )

    return eligibility


def _any_group_eligible(
    eligibility: dict[str, np.ndarray],
) -> np.ndarray:
    return np.column_stack(
        [eligibility[target] for target in TARGET_COLS]
    ).any(axis=1)


def _prior_group_counts(
    eligibility: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Count eligible Teacher rows strictly before each timestamp."""
    counts: dict[str, np.ndarray] = {}

    for target in TARGET_COLS:
        values = eligibility[target].astype(np.int64)
        cumulative = np.cumsum(values)

        prior = np.empty_like(cumulative)
        prior[0] = 0
        prior[1:] = cumulative[:-1]
        counts[target] = prior

    return counts


def _prediction_candidate_mask(
    eligibility: dict[str, np.ndarray],
    prior_counts: dict[str, np.ndarray],
    min_group_rows: int,
) -> np.ndarray:
    ready = np.zeros(
        len(next(iter(eligibility.values()))),
        dtype=bool,
    )

    for target in TARGET_COLS:
        ready |= (
            eligibility[target]
            & (prior_counts[target] >= min_group_rows)
        )

    return ready


def _group_train_counts_before(
    eligibility: dict[str, np.ndarray],
    cutoff_position: int,
) -> dict[str, int]:
    return {
        target: int(
            np.count_nonzero(
                eligibility[target][:cutoff_position]
            )
        )
        for target in TARGET_COLS
    }


def _mask_fold_predictions_by_group_readiness(
    prediction: pd.DataFrame,
    *,
    fold_positions: np.ndarray,
    eligibility: dict[str, np.ndarray],
    group_train_counts: dict[str, int],
    min_group_rows: int,
) -> pd.DataFrame:
    masked = prediction.copy()

    for target in TARGET_COLS:
        current_eligible = eligibility[target][fold_positions]
        enough_group_training = (
            group_train_counts[target] >= min_group_rows
        )

        if not enough_group_training:
            masked.loc[:, target] = np.nan
            continue

        masked.loc[~current_eligible, target] = np.nan

    return masked


def _blend_distillation_targets(
    hard_targets: pd.DataFrame,
    teacher_oof: pd.DataFrame,
    teacher_weight: float,
) -> tuple[pd.DataFrame, int]:
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError("teacher_weight must be between 0 and 1.")

    result = (
        hard_targets.loc[:, TARGET_COLS]
        .astype(float)
        .copy()
    )
    aligned_teacher = teacher_oof.reindex(result.index)

    n_distilled = 0

    for target in TARGET_COLS:
        hard = result[target]
        soft = aligned_teacher[target]

        use = (
            hard.notna()
            & soft.notna()
            & np.isfinite(soft.to_numpy(dtype=float))
        )

        result.loc[use, target] = (
            (1.0 - teacher_weight) * hard.loc[use]
            + teacher_weight * soft.loc[use]
        )

        n_distilled += int(use.sum())

    return result, n_distilled


def _best_epoch_from_metadata(
    metadata: dict[str, Any],
    fallback: int,
) -> tuple[int, float | None, int]:
    """Select epoch from minimum finite recorded validation loss."""
    history = metadata.get("training_history", [])
    candidates: list[tuple[int, float]] = []

    if isinstance(history, list):
        for row_number, row in enumerate(history, start=1):
            if not isinstance(row, dict):
                continue

            value = row.get("validation_loss")
            if value is None:
                continue

            try:
                loss = float(value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(loss):
                continue

            raw_step = row.get("step", row_number)
            try:
                epoch = int(raw_step)
            except (TypeError, ValueError):
                epoch = row_number

            epoch = max(1, min(int(fallback), epoch))
            candidates.append((epoch, loss))

    if not candidates:
        LOGGER.warning(
            "No finite validation_loss in training_history; "
            "falling back to epochs=%d.",
            fallback,
        )
        return int(fallback), None, 0

    best_epoch, best_loss = min(
        candidates,
        key=lambda item: item[1],
    )

    return int(best_epoch), float(best_loss), len(candidates)


def _chronological_inner_split(
    train_positions: np.ndarray,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError(
            "teacher_inner_validation_fraction must be in (0, 0.5)."
        )

    n_rows = int(len(train_positions))
    if n_rows < 2:
        raise ValueError(
            "Teacher inner split requires at least two timestamp rows."
        )

    n_valid = max(
        1,
        int(np.ceil(n_rows * validation_fraction)),
    )
    n_valid = min(n_valid, n_rows - 1)
    split_at = n_rows - n_valid

    return (
        train_positions[:split_at],
        train_positions[split_at:],
    )


class DistilledTemporalRealMLPModel(RegressionModel):
    """Temporal-X Student distilled from same-input + privileged-y Teacher."""

    def __init__(
        self,
        config: PipelineConfig,
        iterations: dict[str, int] | int | None = None,
    ) -> None:
        self.config = replace(
            config,
            group3_stacking=False,
            temporal_prediction_correction=False,
            group3_reliability_weighting=False,
        )

        self.teacher_epoch_preselected = False
        if isinstance(iterations, dict):
            self.student_epochs = int(
                iterations.get("student", config.max_epochs)
            )
            if "teacher" in iterations:
                self.teacher_epochs = int(iterations["teacher"])
                self.teacher_epoch_preselected = True
            else:
                self.teacher_epochs = int(config.teacher_epochs)
        elif iterations is None:
            self.student_epochs = int(config.max_epochs)
            self.teacher_epochs = int(config.teacher_epochs)
        else:
            self.student_epochs = int(iterations)
            self.teacher_epochs = int(config.teacher_epochs)

        self.teacher_selection_metadata: dict[str, Any] = {}

        self.student_model: GroupConditionedRealMLPModel | None = None
        self.student_selection_metadata: dict[str, Any] = {}
        self.teacher_oof_metadata: dict[str, Any] = {}

        # Raw X is kept only as temporal context for future inference.
        self._raw_fit_X: pd.DataFrame | None = None

        self.fit_mode = ""
        self.elapsed_seconds = 0.0

    @staticmethod
    def _require_targets(
        y: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame:
        if not isinstance(y, pd.DataFrame):
            raise TypeError(
                "Teacher-student distillation requires a target DataFrame."
            )

        missing = set(TARGET_COLS) - set(y.columns)
        if missing:
            raise ValueError(
                f"Missing distillation targets: {sorted(missing)}"
            )

        return y.loc[:, TARGET_COLS]

    def _student_features(
        self,
        X: pd.DataFrame,
        *,
        context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _append_feature_history(
            X,
            history_hours=int(self.config.student_history_hours),
            decay=float(self.config.history_decay),
            context=context,
        )

    def _teacher_oof_predictions(
        self,
        student_X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        history_hours = int(self.config.teacher_history_hours)
        min_group_rows = max(
            int(self.config.teacher_min_train_rows),
            history_hours + 1,
        )
        inner_validation_fraction = float(
            self.config.teacher_inner_validation_fraction
        )

        privileged_X = _append_teacher_target_history(
            student_X, y, history_hours
        )
        eligibility = _group_teacher_eligibility(
            privileged_X, y, history_hours
        )
        any_eligible = _any_group_eligible(eligibility)
        prior_counts = _prior_group_counts(eligibility)
        candidate_mask = _prediction_candidate_mask(
            eligibility, prior_counts, min_group_rows
        )
        candidate_positions = np.flatnonzero(candidate_mask)

        oof = pd.DataFrame(
            np.nan, index=student_X.index, columns=TARGET_COLS, dtype=np.float32
        )
        eligible_rows_by_target = {
            target: int(np.count_nonzero(eligibility[target]))
            for target in TARGET_COLS
        }
        first_ready_timestamp_by_target: dict[str, pd.Timestamp | None] = {}
        for target in TARGET_COLS:
            ready_mask = eligibility[target] & (prior_counts[target] >= min_group_rows)
            ready_positions = np.flatnonzero(ready_mask)
            first_ready_timestamp_by_target[target] = (
                student_X.index[int(ready_positions[0])]
                if len(ready_positions) else None
            )

        if len(candidate_positions) == 0:
            self.teacher_oof_metadata = {
                "enabled": False,
                "reason": "no-group-reached-minimum-teacher-training-rows",
                "student_history_hours": int(self.config.student_history_hours),
                "teacher_target_history_hours": history_hours,
                "history_decay": float(self.config.history_decay),
                "min_train_rows_semantics": "per-group",
                "min_train_rows_per_group": min_group_rows,
                "eligible_rows_by_target": eligible_rows_by_target,
                "first_ready_timestamp_by_target": first_ready_timestamp_by_target,
            }
            return oof

        # Select the Teacher epoch only once during the validation-stage fit.
        # Final fit receives the selected Teacher epoch through iteration_schedule.
        if not self.teacher_epoch_preselected:
            selection_cutoff = int(candidate_positions[0])
            selection_mask = any_eligible.copy()
            selection_mask[selection_cutoff:] = False
            selection_positions = np.flatnonzero(selection_mask)
            if len(selection_positions) < 2:
                raise ValueError(
                    "Teacher global epoch selection requires at least two "
                    "chronological history rows before the first OOF block."
                )
            inner_train_positions, inner_valid_positions = _chronological_inner_split(
                selection_positions, inner_validation_fraction
            )
            teacher_selector = GroupConditionedRealMLPModel(
                self.config, epochs=int(self.config.teacher_epochs)
            )
            with _worst_group_ficr_scope(0.0):
                teacher_selector.fit(
                    privileged_X.iloc[inner_train_positions],
                    y.iloc[inner_train_positions],
                    privileged_X.iloc[inner_valid_positions],
                    y.iloc[inner_valid_positions],
                )
            selector_metadata = teacher_selector.metadata()
            selected_teacher_epoch, selected_validation_loss, history_rows = (
                _best_epoch_from_metadata(
                    selector_metadata, int(self.config.teacher_epochs)
                )
            )
            self.teacher_epochs = int(selected_teacher_epoch)
            self.teacher_epoch_preselected = True
            self.teacher_selection_metadata = {
                "source": "single-global-chronological-inner-validation",
                "selected_epoch": int(selected_teacher_epoch),
                "max_epochs": int(self.config.teacher_epochs),
                "validation_loss": selected_validation_loss,
                "selection_history_rows": int(history_rows),
                "inner_train_rows": int(len(inner_train_positions)),
                "inner_validation_rows": int(len(inner_valid_positions)),
                "inner_train_start": student_X.index[int(inner_train_positions[0])],
                "inner_train_end": student_X.index[int(inner_train_positions[-1])],
                "inner_validation_start": student_X.index[int(inner_valid_positions[0])],
                "inner_validation_end": student_X.index[int(inner_valid_positions[-1])],
            }
            LOGGER.info(
                "Teacher global epoch selection: selected_epoch=%d/%d, "
                "validation_loss=%s",
                self.teacher_epochs,
                int(self.config.teacher_epochs),
                (f"{selected_validation_loss:.8f}"
                 if selected_validation_loss is not None else "N/A"),
            )
        else:
            self.teacher_selection_metadata = {
                "source": "reused-from-validation-iteration-schedule",
                "selected_epoch": int(self.teacher_epochs),
            }
            LOGGER.info(
                "Teacher epoch reused from validation stage: epochs=%d",
                self.teacher_epochs,
            )

        n_folds = int(self.config.teacher_oof_folds)
        folds = [
            np.asarray(fold, dtype=int)
            for fold in np.array_split(candidate_positions, n_folds)
            if len(fold) > 0
        ]
        fold_metadata: list[dict[str, Any]] = []

        for fold_number, fold_positions in enumerate(folds, start=1):
            fold_start_position = int(fold_positions[0])
            fold_end_position = int(fold_positions[-1])
            train_mask = any_eligible.copy()
            train_mask[fold_start_position:] = False
            train_positions = np.flatnonzero(train_mask)
            if len(train_positions) < 1:
                LOGGER.warning(
                    "Teacher fold %d skipped: no chronological history.",
                    fold_number,
                )
                continue

            group_train_counts = _group_train_counts_before(
                eligibility, fold_start_position
            )
            ready_groups = [
                target for target in TARGET_COLS
                if group_train_counts[target] >= min_group_rows
            ]
            if not ready_groups:
                continue

            # Exactly one Teacher fit per OOF fold. No per-fold selector/refit pair.
            teacher = GroupConditionedRealMLPModel(
                self.config, epochs=self.teacher_epochs
            )
            with _worst_group_ficr_scope(0.0):
                teacher.fit(
                    privileged_X.iloc[train_positions],
                    y.iloc[train_positions],
                )

            fold_X = privileged_X.iloc[fold_positions]
            fold_prediction = _prediction_frame(teacher, fold_X)
            fold_prediction = _mask_fold_predictions_by_group_readiness(
                fold_prediction,
                fold_positions=fold_positions,
                eligibility=eligibility,
                group_train_counts=group_train_counts,
                min_group_rows=min_group_rows,
            )
            oof.loc[fold_prediction.index, TARGET_COLS] = fold_prediction.to_numpy()
            per_group_predictions = {
                target: int(fold_prediction[target].notna().sum())
                for target in TARGET_COLS
            }
            fold_metadata.append({
                "fold": fold_number,
                "train_timestamp_rows": int(len(train_positions)),
                "prediction_timestamp_rows": int(len(fold_positions)),
                "group_train_rows": group_train_counts,
                "ready_groups": ready_groups,
                "predicted_cells_by_target": per_group_predictions,
                "train_start": student_X.index[int(train_positions[0])],
                "train_end": student_X.index[int(train_positions[-1])],
                "prediction_start": student_X.index[fold_start_position],
                "prediction_end": student_X.index[fold_end_position],
                "teacher_epochs": int(self.teacher_epochs),
                "teacher_fit_count": 1,
            })

        predicted_rows = int(oof.notna().any(axis=1).sum())
        predicted_cells_by_target = {
            target: int(oof[target].notna().sum()) for target in TARGET_COLS
        }
        self.teacher_oof_metadata = {
            "enabled": predicted_rows > 0,
            "student_history_hours": int(self.config.student_history_hours),
            "teacher_target_history_hours": history_hours,
            "history_decay": float(self.config.history_decay),
            "history_scope": "shared-temporal-X + same-group-target-only",
            "teacher_selected_epoch": int(self.teacher_epochs),
            "teacher_epoch_selection": self.teacher_selection_metadata,
            "teacher_oof_fit_policy": "one-fit-per-fold-fixed-epoch",
            "teacher_inner_validation_fraction": inner_validation_fraction,
            "oof_folds_requested": n_folds,
            "oof_folds_completed": len(fold_metadata),
            "min_train_rows_semantics": "per-group",
            "min_train_rows_per_group": min_group_rows,
            "eligible_timestamp_rows_any_group": int(np.count_nonzero(any_eligible)),
            "eligible_rows_by_target": eligible_rows_by_target,
            "first_ready_timestamp_by_target": first_ready_timestamp_by_target,
            "oof_prediction_rows": predicted_rows,
            "oof_prediction_cells_by_target": predicted_cells_by_target,
            "folds": fold_metadata,
        }
        return oof

    def _distilled_targets(
        self,
        student_X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        teacher_oof = self._teacher_oof_predictions(
            student_X,
            y,
        )

        distilled, n_distilled = _blend_distillation_targets(
            y,
            teacher_oof,
            float(self.config.distillation_teacher_weight),
        )

        self.teacher_oof_metadata[
            "distilled_target_cells"
        ] = n_distilled
        self.teacher_oof_metadata[
            "teacher_weight"
        ] = float(
            self.config.distillation_teacher_weight
        )

        return distilled

    @staticmethod
    def _drop_unready_student_rows(
        student_X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ready = _student_history_ready(student_X)
        return (
            student_X.loc[ready],
            y.loc[ready],
        )

    def _fit_with_validation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
    ) -> None:
        student_X = self._student_features(X)
        student_valid_X = self._student_features(
            X_valid,
            context=X,
        )

        distilled_y = self._distilled_targets(
            student_X,
            y,
        )

        student_X_fit, distilled_y_fit = (
            self._drop_unready_student_rows(
                student_X,
                distilled_y,
            )
        )

        selection_end = pd.Timestamp(
            self.config.iteration_selection_end
        )

        tune_mask = np.asarray(
            X_valid.index < selection_end
        )
        tune_observed = (
            y_valid.notna()
            .any(axis=1)
            .to_numpy()
        )
        tune_ready = _student_history_ready(
            student_valid_X
        )

        tune_mask = (
            tune_mask
            & tune_observed
            & tune_ready
        )

        if not tune_mask.any():
            raise ValueError(
                "Student epoch-selection validation window is empty."
            )

        student_selector = GroupConditionedRealMLPModel(
            self.config,
            epochs=self.config.max_epochs,
        )
        with _worst_group_ficr_scope(0.20):
            student_selector.fit(
                student_X_fit,
                distilled_y_fit,
                student_valid_X.loc[tune_mask],
                y_valid.loc[tune_mask],
            )

        selector_metadata = student_selector.metadata()
        (
            selected_student_epoch,
            selected_validation_loss,
            selection_history_rows,
        ) = _best_epoch_from_metadata(
            selector_metadata,
            int(self.config.max_epochs),
        )

        self.student_selection_metadata = {
            **selector_metadata,
            "reported_best_iteration": selector_metadata.get(
                "best_iteration"
            ),
            "best_iteration": int(selected_student_epoch),
            "best_iteration_source": (
                "minimum training_history.validation_loss"
            ),
            "best_validation_loss": selected_validation_loss,
            "selection_history_rows": int(selection_history_rows),
        }

        self.student_epochs = int(
            selected_student_epoch
        )

        LOGGER.info(
            "Student epoch selection: selected_epoch=%d/%d, "
            "validation_loss=%s",
            self.student_epochs,
            int(self.config.max_epochs),
            (
                f"{selected_validation_loss:.8f}"
                if selected_validation_loss is not None
                else "N/A"
            ),
        )

        self.student_model = GroupConditionedRealMLPModel(
            self.config,
            epochs=self.student_epochs,
        )
        with _worst_group_ficr_scope(0.20):
            self.student_model.fit(
                student_X_fit,
                distilled_y_fit,
            )

        # Keep raw history for subsequent validation prediction.
        self._raw_fit_X = X.copy()

        self.fit_mode = (
            "group-conditioned-temporal-x-"
            "chronological-oof-distillation-validation"
        )

    def _fit_final(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> None:
        student_X = self._student_features(X)
        distilled_y = self._distilled_targets(
            student_X,
            y,
        )

        student_X_fit, distilled_y_fit = (
            self._drop_unready_student_rows(
                student_X,
                distilled_y,
            )
        )

        self.student_model = GroupConditionedRealMLPModel(
            self.config,
            epochs=self.student_epochs,
        )
        with _worst_group_ficr_scope(0.20):
            self.student_model.fit(
                student_X_fit,
                distilled_y_fit,
            )

        # X may contain target-masked rows close to test. They are still valid
        # X-history context because no true future target is used.
        self._raw_fit_X = X.copy()

        self.fit_mode = (
            "group-conditioned-temporal-x-"
            "full-history-oof-distillation-refit"
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> "DistilledTemporalRealMLPModel":
        targets = self._require_targets(y)
        started = time.perf_counter()

        if X_valid is not None and y_valid is not None:
            valid_targets = self._require_targets(y_valid)
            self._fit_with_validation(
                X,
                targets,
                X_valid,
                valid_targets,
            )
        elif X_valid is None and y_valid is None:
            self._fit_final(
                X,
                targets,
            )
        else:
            raise ValueError(
                "X_valid and y_valid must be provided together."
            )

        self.elapsed_seconds = (
            time.perf_counter() - started
        )

        LOGGER.info(
            "Group-conditioned teacher-student fit complete: "
            "mode=%s, student_epochs=%d, student_history=%dh, "
            "teacher_target_history=%dh",
            self.fit_mode,
            self.student_epochs,
            int(self.config.student_history_hours),
            int(self.config.teacher_history_hours),
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if self.student_model is None:
            raise RuntimeError(
                "Distilled student model is not fitted."
            )
        if self._raw_fit_X is None:
            raise RuntimeError(
                "Raw fit history is unavailable for temporal prediction."
            )

        student_X = self._student_features(
            X,
            context=self._raw_fit_X,
        )

        ready = _student_history_ready(student_X)
        if not ready.all():
            missing = int((~ready).sum())
            raise ValueError(
                f"{missing} prediction rows do not have complete "
                "Student X history. Check temporal continuity/context."
            )

        return self.student_model.predict(
            student_X
        )

    def metadata(
        self,
    ) -> dict[str, Any]:
        if self.student_model is None:
            raise RuntimeError(
                "Distillation metadata requires fit()."
            )

        student = self.student_model.metadata()
        selection = (
            self.student_selection_metadata
            or student
        )

        return {
            **student,
            "model": (
                "GroupConditioned-RealMLP-TD-"
                "temporal-X-teacher-student"
            ),
            "architecture": (
                "Student=current-X+X-lag1..N+recency-summary; "
                "Teacher=Student-input+same-group-y-lag1..M"
            ),
            "fit_mode": self.fit_mode,
            "student_history_hours": int(
                self.config.student_history_hours
            ),
            "teacher_history_hours": int(
                self.config.teacher_history_hours
            ),
            "history_decay": float(
                self.config.history_decay
            ),
            "teacher_max_epochs": int(self.config.teacher_epochs),
            "teacher_selected_epoch": int(self.teacher_epochs),
            "teacher_epoch_preselected": bool(self.teacher_epoch_preselected),
            "teacher_inner_validation_fraction": float(
                self.config.teacher_inner_validation_fraction
            ),
            "teacher_epoch_selection": (
                "single-global-chronological-validation-then-reuse"
            ),
            "teacher_selection_metadata": self.teacher_selection_metadata,
            "teacher_min_train_rows_semantics": "per-group",
            "teacher_min_train_rows_per_group": int(
                self.config.teacher_min_train_rows
            ),
            "teacher_oof_folds": int(
                self.config.teacher_oof_folds
            ),
            "teacher_oof": self.teacher_oof_metadata,
            "distillation_teacher_weight": float(
                self.config.distillation_teacher_weight
            ),
            "student_worst_group_ficr_reg_weight": 0.20,
            "teacher_worst_group_ficr_reg_weight": 0.0,
            "student_best_iteration": int(
                self.student_epochs
            ),
            "best_iteration": int(
                self.student_epochs
            ),
            "student_selection": selection,
            "elapsed_seconds": float(
                self.elapsed_seconds
            ),
        }