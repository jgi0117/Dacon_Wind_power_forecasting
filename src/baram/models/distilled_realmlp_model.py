"""Teacher-student group-conditioned RealMLP distillation.

Teacher input during training:
    current weather/time
    + previous N hours of weather/time
    + previous N hours of the *same group's* true target
    + group/turbine metadata

Student input during training and inference:
    current weather/time
    + group/turbine metadata

Both Teacher and Student use GroupConditionedRealMLPModel, which internally
uses the existing PyTabKit RealMLPModel and therefore preserves n_ens=8.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import TARGET_COLS

from .base import RegressionModel
from .group_conditioned_realmlp_model import GroupConditionedRealMLPModel


LOGGER = logging.getLogger("baram.pipeline")


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
    return pd.DataFrame(values, index=features.index, columns=TARGET_COLS)


def _append_privileged_history(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    history_hours: int,
) -> pd.DataFrame:
    """Append lag 1..N X and group-specific y source columns.

    The wrapper later converts:
        teacher_target__kpx_group_1__lag_01
    into:
        teacher_target__self__lag_01
    only for the Group 1 long row, and drops the other groups' target lags.
    """
    if history_hours < 1:
        raise ValueError("history_hours must be at least 1.")
    if not features.index.equals(targets.index):
        targets = targets.reindex(features.index)

    missing_targets = set(TARGET_COLS) - set(targets.columns)
    if missing_targets:
        raise ValueError(
            f"Missing teacher target columns: {sorted(missing_targets)}"
        )

    blocks = [features.copy()]
    for lag in range(1, history_hours + 1):
        shifted_X = features.shift(lag).add_prefix(
            f"teacher_feature__lag_{lag:02d}__"
        )
        shifted_y = targets.loc[:, TARGET_COLS].shift(lag).rename(
            columns={
                target: f"teacher_target__{target}__lag_{lag:02d}"
                for target in TARGET_COLS
            }
        )
        blocks.extend([shifted_X, shifted_y])

    return pd.concat(blocks, axis=1)


def _complete_teacher_history_mask(
    privileged_X: pd.DataFrame,
    y: pd.DataFrame,
    history_hours: int,
) -> np.ndarray:
    """Timestamp eligibility when at least one group has complete own history.

    This is intentionally NOT "all three groups must have history".  Therefore
    Group 1/2 can use 2022 teacher supervision even though Group 3 labels are
    absent in 2022.
    """
    eligible_by_group: list[np.ndarray] = []

    for target in TARGET_COLS:
        lag_columns = [
            f"teacher_target__{target}__lag_{lag:02d}"
            for lag in range(1, history_hours + 1)
        ]
        complete_history = (
            privileged_X.loc[:, lag_columns].notna().all(axis=1).to_numpy()
        )
        observed_current = y[target].notna().to_numpy()
        eligible_by_group.append(complete_history & observed_current)

    return np.column_stack(eligible_by_group).any(axis=1)


def _blend_distillation_targets(
    hard_targets: pd.DataFrame,
    teacher_oof: pd.DataFrame,
    teacher_weight: float,
) -> tuple[pd.DataFrame, int]:
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError("teacher_weight must be between 0 and 1.")

    result = hard_targets.loc[:, TARGET_COLS].astype(float).copy()
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


class DistilledTemporalRealMLPModel(RegressionModel):
    """Privileged-history teacher -> metadata-conditioned direct student."""

    def __init__(
        self,
        config: PipelineConfig,
        iterations: dict[str, int] | int | None = None,
    ) -> None:
        self.config = replace(
            config,
            group3_stacking=False,
            temporal_prediction_correction=False,
            # Reliability was designed for the old 3-output loss.  The long
            # wrapper solves Group 3 sharing explicitly, so keep it disabled.
            group3_reliability_weighting=False,
        )

        if isinstance(iterations, dict):
            self.student_epochs = int(
                iterations.get("student", config.max_epochs)
            )
        elif iterations is None:
            self.student_epochs = config.max_epochs
        else:
            self.student_epochs = int(iterations)

        self.teacher_epochs = int(config.teacher_epochs)
        self.student_model: GroupConditionedRealMLPModel | None = None
        self.student_selection_metadata: dict[str, Any] = {}
        self.teacher_oof_metadata: dict[str, Any] = {}
        self.fit_mode = ""
        self.elapsed_seconds = 0.0

    @staticmethod
    def _require_targets(
        y: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame:
        if not isinstance(y, pd.DataFrame):
            raise TypeError(
                "Teacher-student distillation requires target DataFrame."
            )
        missing = set(TARGET_COLS) - set(y.columns)
        if missing:
            raise ValueError(
                f"Missing distillation targets: {sorted(missing)}"
            )
        return y.loc[:, TARGET_COLS]

    def _teacher_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        history_hours = int(self.config.teacher_history_hours)
        privileged_X = _append_privileged_history(
            X, y, history_hours
        )
        eligible = _complete_teacher_history_mask(
            privileged_X, y, history_hours
        )
        eligible_positions = np.flatnonzero(eligible)

        oof = pd.DataFrame(
            np.nan,
            index=X.index,
            columns=TARGET_COLS,
            dtype=np.float32,
        )

        min_train_rows = max(
            int(self.config.teacher_min_train_rows),
            history_hours + 1,
        )
        if len(eligible_positions) <= min_train_rows:
            self.teacher_oof_metadata = {
                "enabled": False,
                "reason": "not-enough-eligible-rows",
                "eligible_rows": int(len(eligible_positions)),
                "min_train_rows": int(min_train_rows),
            }
            return oof

        n_folds = int(self.config.teacher_oof_folds)
        prediction_positions = eligible_positions[min_train_rows:]
        folds = [
            fold
            for fold in np.array_split(prediction_positions, n_folds)
            if len(fold) > 0
        ]

        fold_metadata: list[dict[str, Any]] = []

        for fold_number, fold_positions in enumerate(folds, start=1):
            fold_start_position = int(fold_positions[0])
            fold_end_position = int(fold_positions[-1])

            train_mask = eligible.copy()
            train_mask[fold_start_position:] = False
            train_positions = np.flatnonzero(train_mask)
            if len(train_positions) < min_train_rows:
                continue

            teacher = GroupConditionedRealMLPModel(
                self.config,
                epochs=self.teacher_epochs,
            )
            teacher.fit(
                privileged_X.iloc[train_positions],
                y.iloc[train_positions],
            )

            fold_X = privileged_X.iloc[fold_positions]
            fold_prediction = _prediction_frame(teacher, fold_X)
            oof.loc[
                fold_prediction.index,
                TARGET_COLS,
            ] = fold_prediction.to_numpy()

            per_group_predictions = {
                target: int(fold_prediction[target].notna().sum())
                for target in TARGET_COLS
            }
            fold_metadata.append(
                {
                    "fold": fold_number,
                    "train_rows": int(len(train_positions)),
                    "prediction_rows": int(len(fold_positions)),
                    "predicted_cells_by_target": per_group_predictions,
                    "train_start": X.index[train_positions[0]],
                    "train_end": X.index[train_positions[-1]],
                    "prediction_start": X.index[fold_start_position],
                    "prediction_end": X.index[fold_end_position],
                    "teacher_epochs": self.teacher_epochs,
                }
            )

        predicted_rows = int(oof.notna().any(axis=1).sum())
        self.teacher_oof_metadata = {
            "enabled": predicted_rows > 0,
            "history_hours": history_hours,
            "history_scope": "same-group-target-only",
            "teacher_epochs": self.teacher_epochs,
            "oof_folds_requested": n_folds,
            "oof_folds_completed": len(fold_metadata),
            "eligible_timestamp_rows": int(len(eligible_positions)),
            "oof_prediction_rows": predicted_rows,
            "oof_prediction_cells_by_target": {
                target: int(oof[target].notna().sum())
                for target in TARGET_COLS
            },
            "folds": fold_metadata,
        }
        return oof

    def _distilled_targets(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        teacher_oof = self._teacher_oof_predictions(X, y)
        distilled, n_distilled = _blend_distillation_targets(
            y,
            teacher_oof,
            float(self.config.distillation_teacher_weight),
        )
        self.teacher_oof_metadata["distilled_target_cells"] = n_distilled
        self.teacher_oof_metadata["teacher_weight"] = float(
            self.config.distillation_teacher_weight
        )
        return distilled

    def _fit_with_validation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
    ) -> None:
        distilled_y = self._distilled_targets(X, y)

        selection_end = pd.Timestamp(
            self.config.iteration_selection_end
        )
        tune_mask = np.asarray(X_valid.index < selection_end)
        tune_observed = y_valid.notna().any(axis=1).to_numpy()
        tune_mask = tune_mask & tune_observed
        if not tune_mask.any():
            raise ValueError(
                "Student epoch-selection validation window is empty."
            )

        student_selector = GroupConditionedRealMLPModel(
            self.config,
            epochs=self.config.max_epochs,
        )
        student_selector.fit(
            X,
            distilled_y,
            X_valid.loc[tune_mask],
            y_valid.loc[tune_mask],
        )
        self.student_selection_metadata = (
            student_selector.metadata()
        )
        self.student_epochs = int(
            self.student_selection_metadata.get(
                "best_iteration",
                self.config.max_epochs,
            )
        )

        self.student_model = GroupConditionedRealMLPModel(
            self.config,
            epochs=self.student_epochs,
        )
        self.student_model.fit(X, distilled_y)
        self.fit_mode = (
            "group-conditioned-chronological-oof-distillation-validation"
        )

    def _fit_final(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> None:
        distilled_y = self._distilled_targets(X, y)
        self.student_model = GroupConditionedRealMLPModel(
            self.config,
            epochs=self.student_epochs,
        )
        self.student_model.fit(X, distilled_y)
        self.fit_mode = (
            "group-conditioned-full-history-oof-distillation-refit"
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
                X, targets, X_valid, valid_targets
            )
        elif X_valid is None and y_valid is None:
            self._fit_final(X, targets)
        else:
            raise ValueError(
                "X_valid and y_valid must be provided together."
            )

        self.elapsed_seconds = time.perf_counter() - started
        LOGGER.info(
            "Group-conditioned teacher-student fit complete: "
            "mode=%s, student_epochs=%d",
            self.fit_mode,
            self.student_epochs,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.student_model is None:
            raise RuntimeError(
                "Distilled student model is not fitted."
            )
        return self.student_model.predict(X)

    def metadata(self) -> dict[str, Any]:
        if self.student_model is None:
            raise RuntimeError(
                "Distillation metadata requires fit()."
            )

        student = self.student_model.metadata()
        selection = self.student_selection_metadata or student
        return {
            **student,
            "model": "GroupConditioned-RealMLP-TD-teacher-student",
            "architecture": (
                "same-group-12h privileged teacher -> "
                "metadata-conditioned direct student"
            ),
            "fit_mode": self.fit_mode,
            "teacher_history_hours": int(
                self.config.teacher_history_hours
            ),
            "teacher_epochs": self.teacher_epochs,
            "teacher_oof": self.teacher_oof_metadata,
            "distillation_teacher_weight": float(
                self.config.distillation_teacher_weight
            ),
            "student_best_iteration": int(self.student_epochs),
            "best_iteration": int(self.student_epochs),
            "student_selection": selection,
            "elapsed_seconds": float(self.elapsed_seconds),
        }