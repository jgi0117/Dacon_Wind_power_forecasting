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

Important:
``teacher_min_train_rows`` is interpreted PER GROUP.

For example, with ``teacher_min_train_rows=720``:
- Group 1 soft labels start only after Group 1 has at least 720 eligible
  historical training rows.
- Group 2 follows the same rule independently.
- Group 3 starts only after Group 3 itself has at least 720 eligible rows.
  Group 1/2 cannot satisfy Group 3's minimum on its behalf.

The Teacher itself remains shared across groups.  Thus, when Group 3 begins
producing teacher soft labels, the shared RealMLP can already benefit from
Group 1/2 weather-to-power learning while Group 3 has also accumulated the
required amount of its own turbine-specific supervision.
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

    The group-conditioned wrapper later converts, for example,

        teacher_target__kpx_group_1__lag_01

    into

        teacher_target__self__lag_01

    only for the Group 1 long row.  Other groups' target-history columns are
    dropped before that long row reaches RealMLP.
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


def _group_teacher_eligibility(
    privileged_X: pd.DataFrame,
    y: pd.DataFrame,
    history_hours: int,
) -> dict[str, np.ndarray]:
    """Return Teacher-eligible timestamps independently for each group.

    A group is eligible at timestamp ``t`` only when:

    1. the current target for that group is observed, and
    2. every lag-1 .. lag-N target value for that SAME group is observed.

    This deliberately does not require Group 3 to be present for Group 1/2,
    and it never lets Group 1/2 history stand in for Group 3 history.
    """
    eligibility: dict[str, np.ndarray] = {}

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

        complete_history = (
            privileged_X.loc[:, lag_columns]
            .notna()
            .all(axis=1)
            .to_numpy()
        )
        observed_current = y[target].notna().to_numpy()

        eligibility[target] = complete_history & observed_current

    return eligibility


def _any_group_eligible(
    eligibility: dict[str, np.ndarray],
) -> np.ndarray:
    stacked = np.column_stack(
        [eligibility[target] for target in TARGET_COLS]
    )
    return stacked.any(axis=1)


def _prior_group_counts(
    eligibility: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Count eligible training rows strictly before each timestamp."""
    counts: dict[str, np.ndarray] = {}

    for target in TARGET_COLS:
        values = eligibility[target].astype(np.int64)
        cumulative = np.cumsum(values)

        # Number of eligible rows before position i, not including i itself.
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
    """Timestamp can enter OOF when at least one group is ready.

    "Ready" means the group is eligible at that timestamp and has at least
    ``min_group_rows`` eligible historical rows strictly before it.
    """
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
    """Eligible rows available to each group strictly before a fold."""
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
    """Remove soft labels for groups that are not independently ready."""
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
        raise ValueError(
            "teacher_weight must be between 0 and 1."
        )

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
            # Legacy reliability weighting was designed around the old
            # simultaneous three-output objective.  Long-format group
            # conditioning handles sharing explicitly.
            group3_reliability_weighting=False,
        )

        if isinstance(iterations, dict):
            self.student_epochs = int(
                iterations.get(
                    "student",
                    config.max_epochs,
                )
            )
        elif iterations is None:
            self.student_epochs = int(config.max_epochs)
        else:
            self.student_epochs = int(iterations)

        self.teacher_epochs = int(config.teacher_epochs)

        self.student_model: (
            GroupConditionedRealMLPModel | None
        ) = None

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
                "Teacher-student distillation requires "
                "a target DataFrame."
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
        history_hours = int(
            self.config.teacher_history_hours
        )

        # This configuration value is now interpreted per group.
        min_group_rows = max(
            int(self.config.teacher_min_train_rows),
            history_hours + 1,
        )

        privileged_X = _append_privileged_history(
            X,
            y,
            history_hours,
        )

        eligibility = _group_teacher_eligibility(
            privileged_X,
            y,
            history_hours,
        )

        any_eligible = _any_group_eligible(eligibility)
        prior_counts = _prior_group_counts(eligibility)

        candidate_mask = _prediction_candidate_mask(
            eligibility,
            prior_counts,
            min_group_rows,
        )
        candidate_positions = np.flatnonzero(candidate_mask)

        oof = pd.DataFrame(
            np.nan,
            index=X.index,
            columns=TARGET_COLS,
            dtype=np.float32,
        )

        eligible_rows_by_target = {
            target: int(
                np.count_nonzero(eligibility[target])
            )
            for target in TARGET_COLS
        }

        first_ready_timestamp_by_target: dict[
            str, pd.Timestamp | None
        ] = {}

        for target in TARGET_COLS:
            ready_mask = (
                eligibility[target]
                & (prior_counts[target] >= min_group_rows)
            )
            ready_positions = np.flatnonzero(ready_mask)

            first_ready_timestamp_by_target[target] = (
                X.index[int(ready_positions[0])]
                if len(ready_positions)
                else None
            )

        if len(candidate_positions) == 0:
            self.teacher_oof_metadata = {
                "enabled": False,
                "reason": (
                    "no-group-reached-minimum-teacher-training-rows"
                ),
                "history_hours": history_hours,
                "min_train_rows_semantics": "per-group",
                "min_train_rows_per_group": min_group_rows,
                "eligible_rows_by_target": eligible_rows_by_target,
                "first_ready_timestamp_by_target": (
                    first_ready_timestamp_by_target
                ),
            }
            return oof

        n_folds = int(self.config.teacher_oof_folds)

        folds = [
            np.asarray(fold, dtype=int)
            for fold in np.array_split(
                candidate_positions,
                n_folds,
            )
            if len(fold) > 0
        ]

        fold_metadata: list[dict[str, Any]] = []

        for fold_number, fold_positions in enumerate(
            folds,
            start=1,
        ):
            fold_start_position = int(fold_positions[0])
            fold_end_position = int(fold_positions[-1])

            # Strict chronology: Teacher sees only timestamps before
            # the fold's first prediction timestamp.
            train_mask = any_eligible.copy()
            train_mask[fold_start_position:] = False
            train_positions = np.flatnonzero(train_mask)

            if len(train_positions) == 0:
                continue

            group_train_counts = _group_train_counts_before(
                eligibility,
                fold_start_position,
            )

            ready_groups = [
                target
                for target in TARGET_COLS
                if group_train_counts[target] >= min_group_rows
            ]

            if not ready_groups:
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

            fold_prediction = _prediction_frame(
                teacher,
                fold_X,
            )

            fold_prediction = (
                _mask_fold_predictions_by_group_readiness(
                    fold_prediction,
                    fold_positions=fold_positions,
                    eligibility=eligibility,
                    group_train_counts=group_train_counts,
                    min_group_rows=min_group_rows,
                )
            )

            oof.loc[
                fold_prediction.index,
                TARGET_COLS,
            ] = fold_prediction.to_numpy()

            per_group_predictions = {
                target: int(
                    fold_prediction[target].notna().sum()
                )
                for target in TARGET_COLS
            }

            fold_metadata.append(
                {
                    "fold": fold_number,
                    "train_timestamp_rows": int(
                        len(train_positions)
                    ),
                    "prediction_timestamp_rows": int(
                        len(fold_positions)
                    ),
                    "group_train_rows": group_train_counts,
                    "ready_groups": ready_groups,
                    "predicted_cells_by_target": (
                        per_group_predictions
                    ),
                    "train_start": X.index[
                        int(train_positions[0])
                    ],
                    "train_end": X.index[
                        int(train_positions[-1])
                    ],
                    "prediction_start": X.index[
                        fold_start_position
                    ],
                    "prediction_end": X.index[
                        fold_end_position
                    ],
                    "teacher_epochs": self.teacher_epochs,
                }
            )

        predicted_rows = int(
            oof.notna().any(axis=1).sum()
        )

        predicted_cells_by_target = {
            target: int(oof[target].notna().sum())
            for target in TARGET_COLS
        }

        self.teacher_oof_metadata = {
            "enabled": predicted_rows > 0,
            "history_hours": history_hours,
            "history_scope": "same-group-target-only",
            "teacher_epochs": self.teacher_epochs,
            "oof_folds_requested": n_folds,
            "oof_folds_completed": len(fold_metadata),
            "min_train_rows_semantics": "per-group",
            "min_train_rows_per_group": min_group_rows,
            "eligible_timestamp_rows_any_group": int(
                np.count_nonzero(any_eligible)
            ),
            "eligible_rows_by_target": (
                eligible_rows_by_target
            ),
            "first_ready_timestamp_by_target": (
                first_ready_timestamp_by_target
            ),
            "oof_prediction_rows": predicted_rows,
            "oof_prediction_cells_by_target": (
                predicted_cells_by_target
            ),
            "folds": fold_metadata,
        }

        return oof

    def _distilled_targets(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        teacher_oof = self._teacher_oof_predictions(
            X,
            y,
        )

        distilled, n_distilled = (
            _blend_distillation_targets(
                y,
                teacher_oof,
                float(
                    self.config.distillation_teacher_weight
                ),
            )
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

    def _fit_with_validation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
    ) -> None:
        distilled_y = self._distilled_targets(
            X,
            y,
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

        tune_mask = tune_mask & tune_observed

        if not tune_mask.any():
            raise ValueError(
                "Student epoch-selection validation "
                "window is empty."
            )

        student_selector = (
            GroupConditionedRealMLPModel(
                self.config,
                epochs=self.config.max_epochs,
            )
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

        self.student_model = (
            GroupConditionedRealMLPModel(
                self.config,
                epochs=self.student_epochs,
            )
        )

        self.student_model.fit(
            X,
            distilled_y,
        )

        self.fit_mode = (
            "group-conditioned-"
            "chronological-oof-distillation-validation"
        )

    def _fit_final(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> None:
        distilled_y = self._distilled_targets(
            X,
            y,
        )

        self.student_model = (
            GroupConditionedRealMLPModel(
                self.config,
                epochs=self.student_epochs,
            )
        )

        self.student_model.fit(
            X,
            distilled_y,
        )

        self.fit_mode = (
            "group-conditioned-"
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
            valid_targets = self._require_targets(
                y_valid
            )

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
                "X_valid and y_valid must be "
                "provided together."
            )

        self.elapsed_seconds = (
            time.perf_counter() - started
        )

        LOGGER.info(
            "Group-conditioned teacher-student "
            "fit complete: mode=%s, student_epochs=%d",
            self.fit_mode,
            self.student_epochs,
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

        return self.student_model.predict(X)

    def metadata(self) -> dict[str, Any]:
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
                "teacher-student"
            ),
            "architecture": (
                "same-group-12h privileged teacher -> "
                "metadata-conditioned direct student"
            ),
            "fit_mode": self.fit_mode,
            "teacher_history_hours": int(
                self.config.teacher_history_hours
            ),
            "teacher_epochs": self.teacher_epochs,
            "teacher_min_train_rows_semantics": (
                "per-group"
            ),
            "teacher_min_train_rows_per_group": int(
                self.config.teacher_min_train_rows
            ),
            "teacher_oof": self.teacher_oof_metadata,
            "distillation_teacher_weight": float(
                self.config.distillation_teacher_weight
            ),
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