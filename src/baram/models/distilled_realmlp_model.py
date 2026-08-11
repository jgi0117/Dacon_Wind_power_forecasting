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

Important
---------
``teacher_min_train_rows`` is interpreted PER GROUP.

Teacher OOF training uses expanding chronological folds.  Inside every OOF
fold, the available past is split chronologically into:

    inner train -> inner validation

The inner validation loss recorded by RealMLPModel is used directly to select
the best epoch.  The selected epoch is then used to refit the Teacher on ALL
history available before the OOF fold, and that refitted Teacher predicts the
future OOF block.

This deliberately avoids relying on PyTabKit's ``stop_epoch`` metadata, which
can report 1 even when the model actually trained for all requested epochs.

The same validation-history based epoch selection is also used for the Student
selector.
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
    values = np.asarray(
        model.predict(features),
        dtype=np.float32,
    )
    expected = (
        len(features),
        len(TARGET_COLS),
    )
    if values.shape != expected:
        raise ValueError(
            "Teacher prediction shape mismatch: "
            f"{values.shape} != {expected}"
        )
    return pd.DataFrame(
        values,
        index=features.index,
        columns=TARGET_COLS,
    )


def _append_privileged_history(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    history_hours: int,
) -> pd.DataFrame:
    """Append lag 1..N X and group-specific y source columns."""
    if history_hours < 1:
        raise ValueError(
            "history_hours must be at least 1."
        )

    if not features.index.equals(
        targets.index
    ):
        targets = targets.reindex(
            features.index
        )

    missing_targets = (
        set(TARGET_COLS)
        - set(targets.columns)
    )
    if missing_targets:
        raise ValueError(
            "Missing teacher target columns: "
            f"{sorted(missing_targets)}"
        )

    blocks = [
        features.copy()
    ]

    for lag in range(
        1,
        history_hours + 1,
    ):
        shifted_X = (
            features.shift(lag)
            .add_prefix(
                f"teacher_feature__lag_{lag:02d}__"
            )
        )

        shifted_y = (
            targets.loc[
                :,
                TARGET_COLS,
            ]
            .shift(lag)
            .rename(
                columns={
                    target: (
                        f"teacher_target__{target}"
                        f"__lag_{lag:02d}"
                    )
                    for target in TARGET_COLS
                }
            )
        )

        blocks.extend(
            [
                shifted_X,
                shifted_y,
            ]
        )

    return pd.concat(
        blocks,
        axis=1,
    )


def _group_teacher_eligibility(
    privileged_X: pd.DataFrame,
    y: pd.DataFrame,
    history_hours: int,
) -> dict[str, np.ndarray]:
    """Return Teacher-eligible timestamps independently for each group."""
    eligibility: dict[
        str,
        np.ndarray,
    ] = {}

    for target in TARGET_COLS:
        lag_columns = [
            (
                f"teacher_target__{target}"
                f"__lag_{lag:02d}"
            )
            for lag in range(
                1,
                history_hours + 1,
            )
        ]

        missing_columns = [
            column
            for column in lag_columns
            if column
            not in privileged_X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{target}: missing privileged "
                "history columns: "
                f"{missing_columns}"
            )

        complete_history = (
            privileged_X.loc[
                :,
                lag_columns,
            ]
            .notna()
            .all(axis=1)
            .to_numpy()
        )

        observed_current = (
            y[target]
            .notna()
            .to_numpy()
        )

        eligibility[target] = (
            complete_history
            & observed_current
        )

    return eligibility


def _any_group_eligible(
    eligibility: dict[
        str,
        np.ndarray,
    ],
) -> np.ndarray:
    stacked = np.column_stack(
        [
            eligibility[target]
            for target in TARGET_COLS
        ]
    )
    return stacked.any(axis=1)


def _prior_group_counts(
    eligibility: dict[
        str,
        np.ndarray,
    ],
) -> dict[str, np.ndarray]:
    """Count eligible training rows strictly before each timestamp."""
    counts: dict[
        str,
        np.ndarray,
    ] = {}

    for target in TARGET_COLS:
        values = (
            eligibility[target]
            .astype(np.int64)
        )

        cumulative = np.cumsum(
            values
        )

        prior = np.empty_like(
            cumulative
        )
        prior[0] = 0
        prior[1:] = cumulative[:-1]

        counts[target] = prior

    return counts


def _prediction_candidate_mask(
    eligibility: dict[
        str,
        np.ndarray,
    ],
    prior_counts: dict[
        str,
        np.ndarray,
    ],
    min_group_rows: int,
) -> np.ndarray:
    """Timestamp enters OOF when at least one group is independently ready."""
    ready = np.zeros(
        len(
            next(
                iter(
                    eligibility.values()
                )
            )
        ),
        dtype=bool,
    )

    for target in TARGET_COLS:
        ready |= (
            eligibility[target]
            & (
                prior_counts[target]
                >= min_group_rows
            )
        )

    return ready


def _group_train_counts_before(
    eligibility: dict[
        str,
        np.ndarray,
    ],
    cutoff_position: int,
) -> dict[str, int]:
    """Eligible rows available to each group strictly before a fold."""
    return {
        target: int(
            np.count_nonzero(
                eligibility[target][
                    :cutoff_position
                ]
            )
        )
        for target in TARGET_COLS
    }


def _mask_fold_predictions_by_group_readiness(
    prediction: pd.DataFrame,
    *,
    fold_positions: np.ndarray,
    eligibility: dict[
        str,
        np.ndarray,
    ],
    group_train_counts: dict[
        str,
        int,
    ],
    min_group_rows: int,
) -> pd.DataFrame:
    """Remove soft labels for groups that are not independently ready."""
    masked = prediction.copy()

    for target in TARGET_COLS:
        current_eligible = (
            eligibility[target][
                fold_positions
            ]
        )

        enough_group_training = (
            group_train_counts[target]
            >= min_group_rows
        )

        if not enough_group_training:
            masked.loc[
                :,
                target,
            ] = np.nan
            continue

        masked.loc[
            ~current_eligible,
            target,
        ] = np.nan

    return masked


def _blend_distillation_targets(
    hard_targets: pd.DataFrame,
    teacher_oof: pd.DataFrame,
    teacher_weight: float,
) -> tuple[pd.DataFrame, int]:
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError(
            "teacher_weight must be "
            "between 0 and 1."
        )

    result = (
        hard_targets.loc[
            :,
            TARGET_COLS,
        ]
        .astype(float)
        .copy()
    )

    aligned_teacher = (
        teacher_oof.reindex(
            result.index
        )
    )

    n_distilled = 0

    for target in TARGET_COLS:
        hard = result[target]
        soft = aligned_teacher[target]

        use = (
            hard.notna()
            & soft.notna()
            & np.isfinite(
                soft.to_numpy(
                    dtype=float
                )
            )
        )

        result.loc[
            use,
            target,
        ] = (
            (
                1.0
                - teacher_weight
            )
            * hard.loc[use]
            + teacher_weight
            * soft.loc[use]
        )

        n_distilled += int(
            use.sum()
        )

    return (
        result,
        n_distilled,
    )


def _best_epoch_from_metadata(
    metadata: dict[str, Any],
    fallback: int,
) -> tuple[int, float | None, int]:
    """Select epoch from recorded validation history, not PyTabKit stop_epoch.

    Returns
    -------
    best_epoch:
        1-based epoch selected by minimum finite validation_loss.

    best_validation_loss:
        Validation loss at the selected epoch, if available.

    history_rows:
        Number of finite validation-history rows used for selection.
    """
    history = metadata.get(
        "training_history",
        [],
    )

    candidates: list[
        tuple[int, float]
    ] = []

    if isinstance(
        history,
        list,
    ):
        for row_number, row in enumerate(
            history,
            start=1,
        ):
            if not isinstance(
                row,
                dict,
            ):
                continue

            value = row.get(
                "validation_loss"
            )

            if value is None:
                continue

            try:
                loss = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(
                loss
            ):
                continue

            raw_step = row.get(
                "step",
                row_number,
            )

            try:
                epoch = int(
                    raw_step
                )
            except (
                TypeError,
                ValueError,
            ):
                epoch = row_number

            epoch = max(
                1,
                min(
                    int(fallback),
                    epoch,
                ),
            )

            candidates.append(
                (
                    epoch,
                    loss,
                )
            )

    if not candidates:
        LOGGER.warning(
            "No finite validation_loss was "
            "found in training_history; "
            "falling back to epochs=%d.",
            fallback,
        )
        return (
            int(fallback),
            None,
            0,
        )

    best_epoch, best_loss = min(
        candidates,
        key=lambda item: item[1],
    )

    return (
        int(best_epoch),
        float(best_loss),
        len(candidates),
    )


def _chronological_inner_split(
    train_positions: np.ndarray,
    validation_fraction: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Split available past into earlier train and later validation rows."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError(
            "teacher_inner_validation_fraction "
            "must be in (0, 0.5)."
        )

    n_rows = int(
        len(
            train_positions
        )
    )

    if n_rows < 2:
        raise ValueError(
            "Teacher inner split requires "
            "at least two timestamp rows."
        )

    n_valid = max(
        1,
        int(
            np.ceil(
                n_rows
                * validation_fraction
            )
        ),
    )

    n_valid = min(
        n_valid,
        n_rows - 1,
    )

    split_at = (
        n_rows
        - n_valid
    )

    inner_train = (
        train_positions[
            :split_at
        ]
    )

    inner_valid = (
        train_positions[
            split_at:
        ]
    )

    return (
        inner_train,
        inner_valid,
    )


class DistilledTemporalRealMLPModel(
    RegressionModel
):
    """Privileged-history teacher -> metadata-conditioned direct student."""

    def __init__(
        self,
        config: PipelineConfig,
        iterations: (
            dict[str, int]
            | int
            | None
        ) = None,
    ) -> None:
        self.config = replace(
            config,
            group3_stacking=False,
            temporal_prediction_correction=False,
            group3_reliability_weighting=False,
        )

        if isinstance(
            iterations,
            dict,
        ):
            self.student_epochs = int(
                iterations.get(
                    "student",
                    config.max_epochs,
                )
            )
        elif iterations is None:
            self.student_epochs = int(
                config.max_epochs
            )
        else:
            self.student_epochs = int(
                iterations
            )

        self.teacher_epochs = int(
            config.teacher_epochs
        )

        self.student_model: (
            GroupConditionedRealMLPModel
            | None
        ) = None

        self.student_selection_metadata: dict[
            str,
            Any,
        ] = {}

        self.teacher_oof_metadata: dict[
            str,
            Any,
        ] = {}

        self.fit_mode = ""
        self.elapsed_seconds = 0.0

    @staticmethod
    def _require_targets(
        y: (
            pd.DataFrame
            | pd.Series
        ),
    ) -> pd.DataFrame:
        if not isinstance(
            y,
            pd.DataFrame,
        ):
            raise TypeError(
                "Teacher-student distillation "
                "requires a target DataFrame."
            )

        missing = (
            set(TARGET_COLS)
            - set(y.columns)
        )

        if missing:
            raise ValueError(
                "Missing distillation targets: "
                f"{sorted(missing)}"
            )

        return y.loc[
            :,
            TARGET_COLS,
        ]

    def _teacher_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        history_hours = int(
            self.config.teacher_history_hours
        )

        min_group_rows = max(
            int(
                self.config.teacher_min_train_rows
            ),
            history_hours + 1,
        )

        inner_validation_fraction = float(
            self.config.teacher_inner_validation_fraction
        )

        privileged_X = (
            _append_privileged_history(
                X,
                y,
                history_hours,
            )
        )

        eligibility = (
            _group_teacher_eligibility(
                privileged_X,
                y,
                history_hours,
            )
        )

        any_eligible = (
            _any_group_eligible(
                eligibility
            )
        )

        prior_counts = (
            _prior_group_counts(
                eligibility
            )
        )

        candidate_mask = (
            _prediction_candidate_mask(
                eligibility,
                prior_counts,
                min_group_rows,
            )
        )

        candidate_positions = (
            np.flatnonzero(
                candidate_mask
            )
        )

        oof = pd.DataFrame(
            np.nan,
            index=X.index,
            columns=TARGET_COLS,
            dtype=np.float32,
        )

        eligible_rows_by_target = {
            target: int(
                np.count_nonzero(
                    eligibility[target]
                )
            )
            for target in TARGET_COLS
        }

        first_ready_timestamp_by_target: dict[
            str,
            pd.Timestamp | None,
        ] = {}

        for target in TARGET_COLS:
            ready_mask = (
                eligibility[target]
                & (
                    prior_counts[target]
                    >= min_group_rows
                )
            )

            ready_positions = (
                np.flatnonzero(
                    ready_mask
                )
            )

            first_ready_timestamp_by_target[
                target
            ] = (
                X.index[
                    int(
                        ready_positions[0]
                    )
                ]
                if len(
                    ready_positions
                )
                else None
            )

        if len(
            candidate_positions
        ) == 0:
            self.teacher_oof_metadata = {
                "enabled": False,
                "reason": (
                    "no-group-reached-minimum-"
                    "teacher-training-rows"
                ),
                "history_hours": (
                    history_hours
                ),
                "min_train_rows_semantics": (
                    "per-group"
                ),
                "min_train_rows_per_group": (
                    min_group_rows
                ),
                "eligible_rows_by_target": (
                    eligible_rows_by_target
                ),
                "first_ready_timestamp_by_target": (
                    first_ready_timestamp_by_target
                ),
            }
            return oof

        n_folds = int(
            self.config.teacher_oof_folds
        )

        folds = [
            np.asarray(
                fold,
                dtype=int,
            )
            for fold in np.array_split(
                candidate_positions,
                n_folds,
            )
            if len(fold) > 0
        ]

        fold_metadata: list[
            dict[str, Any]
        ] = []

        for fold_number, fold_positions in enumerate(
            folds,
            start=1,
        ):
            fold_start_position = int(
                fold_positions[0]
            )

            fold_end_position = int(
                fold_positions[-1]
            )

            # Strict chronology:
            # Teacher can only use timestamps before
            # the first timestamp of the OOF fold.
            train_mask = (
                any_eligible.copy()
            )

            train_mask[
                fold_start_position:
            ] = False

            train_positions = (
                np.flatnonzero(
                    train_mask
                )
            )

            if len(
                train_positions
            ) < 2:
                LOGGER.warning(
                    "Teacher fold %d skipped: "
                    "not enough chronological "
                    "history for inner validation.",
                    fold_number,
                )
                continue

            group_train_counts = (
                _group_train_counts_before(
                    eligibility,
                    fold_start_position,
                )
            )

            ready_groups = [
                target
                for target
                in TARGET_COLS
                if (
                    group_train_counts[
                        target
                    ]
                    >= min_group_rows
                )
            ]

            if not ready_groups:
                continue

            (
                inner_train_positions,
                inner_valid_positions,
            ) = _chronological_inner_split(
                train_positions,
                inner_validation_fraction,
            )

            # ---------------------------------------------------------
            # 1) Teacher epoch selector
            # ---------------------------------------------------------
            teacher_selector = (
                GroupConditionedRealMLPModel(
                    self.config,
                    epochs=self.teacher_epochs,
                )
            )

            teacher_selector.fit(
                privileged_X.iloc[
                    inner_train_positions
                ],
                y.iloc[
                    inner_train_positions
                ],
                privileged_X.iloc[
                    inner_valid_positions
                ],
                y.iloc[
                    inner_valid_positions
                ],
            )

            selector_metadata = (
                teacher_selector.metadata()
            )

            (
                selected_teacher_epoch,
                selected_validation_loss,
                selection_history_rows,
            ) = _best_epoch_from_metadata(
                selector_metadata,
                self.teacher_epochs,
            )

            LOGGER.info(
                "Teacher fold %d epoch selection: "
                "inner_train=%d, inner_valid=%d, "
                "selected_epoch=%d/%d, "
                "validation_loss=%s",
                fold_number,
                len(
                    inner_train_positions
                ),
                len(
                    inner_valid_positions
                ),
                selected_teacher_epoch,
                self.teacher_epochs,
                (
                    f"{selected_validation_loss:.8f}"
                    if (
                        selected_validation_loss
                        is not None
                    )
                    else "N/A"
                ),
            )

            # ---------------------------------------------------------
            # 2) Refit Teacher on ALL available history using selected
            #    epoch.  This model produces the actual OOF prediction.
            # ---------------------------------------------------------
            teacher = (
                GroupConditionedRealMLPModel(
                    self.config,
                    epochs=selected_teacher_epoch,
                )
            )

            teacher.fit(
                privileged_X.iloc[
                    train_positions
                ],
                y.iloc[
                    train_positions
                ],
            )

            fold_X = (
                privileged_X.iloc[
                    fold_positions
                ]
            )

            fold_prediction = (
                _prediction_frame(
                    teacher,
                    fold_X,
                )
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
            ] = (
                fold_prediction
                .to_numpy()
            )

            per_group_predictions = {
                target: int(
                    fold_prediction[
                        target
                    ]
                    .notna()
                    .sum()
                )
                for target
                in TARGET_COLS
            }

            fold_metadata.append(
                {
                    "fold": (
                        fold_number
                    ),
                    "train_timestamp_rows": int(
                        len(
                            train_positions
                        )
                    ),
                    "inner_train_timestamp_rows": int(
                        len(
                            inner_train_positions
                        )
                    ),
                    "inner_validation_timestamp_rows": int(
                        len(
                            inner_valid_positions
                        )
                    ),
                    "inner_validation_fraction": (
                        inner_validation_fraction
                    ),
                    "prediction_timestamp_rows": int(
                        len(
                            fold_positions
                        )
                    ),
                    "group_train_rows": (
                        group_train_counts
                    ),
                    "ready_groups": (
                        ready_groups
                    ),
                    "predicted_cells_by_target": (
                        per_group_predictions
                    ),
                    "train_start": (
                        X.index[
                            int(
                                train_positions[0]
                            )
                        ]
                    ),
                    "train_end": (
                        X.index[
                            int(
                                train_positions[-1]
                            )
                        ]
                    ),
                    "inner_train_start": (
                        X.index[
                            int(
                                inner_train_positions[0]
                            )
                        ]
                    ),
                    "inner_train_end": (
                        X.index[
                            int(
                                inner_train_positions[-1]
                            )
                        ]
                    ),
                    "inner_validation_start": (
                        X.index[
                            int(
                                inner_valid_positions[0]
                            )
                        ]
                    ),
                    "inner_validation_end": (
                        X.index[
                            int(
                                inner_valid_positions[-1]
                            )
                        ]
                    ),
                    "prediction_start": (
                        X.index[
                            fold_start_position
                        ]
                    ),
                    "prediction_end": (
                        X.index[
                            fold_end_position
                        ]
                    ),
                    "teacher_max_epochs": (
                        self.teacher_epochs
                    ),
                    "teacher_selected_epoch": int(
                        selected_teacher_epoch
                    ),
                    "teacher_selected_validation_loss": (
                        selected_validation_loss
                    ),
                    "teacher_selection_history_rows": int(
                        selection_history_rows
                    ),
                    "teacher_epoch_selection_source": (
                        "minimum training_history.validation_loss"
                    ),
                    "teacher_refit_epochs": int(
                        selected_teacher_epoch
                    ),
                }
            )

        predicted_rows = int(
            oof.notna()
            .any(axis=1)
            .sum()
        )

        predicted_cells_by_target = {
            target: int(
                oof[target]
                .notna()
                .sum()
            )
            for target
            in TARGET_COLS
        }

        self.teacher_oof_metadata = {
            "enabled": (
                predicted_rows > 0
            ),
            "history_hours": (
                history_hours
            ),
            "history_scope": (
                "same-group-target-only"
            ),
            "teacher_max_epochs": (
                self.teacher_epochs
            ),
            "teacher_epoch_selection": (
                "chronological-inner-validation"
            ),
            "teacher_epoch_selection_metric": (
                "validation_loss"
            ),
            "teacher_inner_validation_fraction": (
                inner_validation_fraction
            ),
            "oof_folds_requested": (
                n_folds
            ),
            "oof_folds_completed": (
                len(
                    fold_metadata
                )
            ),
            "min_train_rows_semantics": (
                "per-group"
            ),
            "min_train_rows_per_group": (
                min_group_rows
            ),
            "eligible_timestamp_rows_any_group": int(
                np.count_nonzero(
                    any_eligible
                )
            ),
            "eligible_rows_by_target": (
                eligible_rows_by_target
            ),
            "first_ready_timestamp_by_target": (
                first_ready_timestamp_by_target
            ),
            "oof_prediction_rows": (
                predicted_rows
            ),
            "oof_prediction_cells_by_target": (
                predicted_cells_by_target
            ),
            "folds": (
                fold_metadata
            ),
        }

        return oof

    def _distilled_targets(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        teacher_oof = (
            self._teacher_oof_predictions(
                X,
                y,
            )
        )

        (
            distilled,
            n_distilled,
        ) = (
            _blend_distillation_targets(
                y,
                teacher_oof,
                float(
                    self.config
                    .distillation_teacher_weight
                ),
            )
        )

        self.teacher_oof_metadata[
            "distilled_target_cells"
        ] = n_distilled

        self.teacher_oof_metadata[
            "teacher_weight"
        ] = float(
            self.config
            .distillation_teacher_weight
        )

        return distilled

    def _fit_with_validation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
    ) -> None:
        distilled_y = (
            self._distilled_targets(
                X,
                y,
            )
        )

        selection_end = pd.Timestamp(
            self.config
            .iteration_selection_end
        )

        tune_mask = np.asarray(
            X_valid.index
            < selection_end
        )

        tune_observed = (
            y_valid.notna()
            .any(axis=1)
            .to_numpy()
        )

        tune_mask = (
            tune_mask
            & tune_observed
        )

        if not tune_mask.any():
            raise ValueError(
                "Student epoch-selection "
                "validation window is empty."
            )

        student_selector = (
            GroupConditionedRealMLPModel(
                self.config,
                epochs=(
                    self.config
                    .max_epochs
                ),
            )
        )

        student_selector.fit(
            X,
            distilled_y,
            X_valid.loc[
                tune_mask
            ],
            y_valid.loc[
                tune_mask
            ],
        )

        selector_metadata = (
            student_selector.metadata()
        )

        (
            selected_student_epoch,
            selected_validation_loss,
            selection_history_rows,
        ) = _best_epoch_from_metadata(
            selector_metadata,
            int(
                self.config.max_epochs
            ),
        )

        # Preserve the original selector metadata but overwrite the
        # unreliable PyTabKit-derived best_iteration with our selection.
        self.student_selection_metadata = {
            **selector_metadata,
            "reported_best_iteration": (
                selector_metadata.get(
                    "best_iteration"
                )
            ),
            "best_iteration": int(
                selected_student_epoch
            ),
            "best_iteration_source": (
                "minimum training_history.validation_loss"
            ),
            "best_validation_loss": (
                selected_validation_loss
            ),
            "selection_history_rows": int(
                selection_history_rows
            ),
        }

        self.student_epochs = int(
            selected_student_epoch
        )

        LOGGER.info(
            "Student epoch selection: "
            "selected_epoch=%d/%d, "
            "validation_loss=%s",
            self.student_epochs,
            int(
                self.config.max_epochs
            ),
            (
                f"{selected_validation_loss:.8f}"
                if (
                    selected_validation_loss
                    is not None
                )
                else "N/A"
            ),
        )

        self.student_model = (
            GroupConditionedRealMLPModel(
                self.config,
                epochs=(
                    self.student_epochs
                ),
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
        distilled_y = (
            self._distilled_targets(
                X,
                y,
            )
        )

        self.student_model = (
            GroupConditionedRealMLPModel(
                self.config,
                epochs=(
                    self.student_epochs
                ),
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
        y: (
            pd.DataFrame
            | pd.Series
        ),
        X_valid: (
            pd.DataFrame
            | None
        ) = None,
        y_valid: (
            pd.DataFrame
            | pd.Series
            | None
        ) = None,
    ) -> "DistilledTemporalRealMLPModel":
        targets = (
            self._require_targets(
                y
            )
        )

        started = (
            time.perf_counter()
        )

        if (
            X_valid is not None
            and y_valid is not None
        ):
            valid_targets = (
                self._require_targets(
                    y_valid
                )
            )

            self._fit_with_validation(
                X,
                targets,
                X_valid,
                valid_targets,
            )

        elif (
            X_valid is None
            and y_valid is None
        ):
            self._fit_final(
                X,
                targets,
            )

        else:
            raise ValueError(
                "X_valid and y_valid must "
                "be provided together."
            )

        self.elapsed_seconds = (
            time.perf_counter()
            - started
        )

        LOGGER.info(
            "Group-conditioned teacher-student "
            "fit complete: mode=%s, "
            "student_epochs=%d",
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
                "Distilled student model "
                "is not fitted."
            )

        return (
            self.student_model
            .predict(X)
        )

    def metadata(
        self,
    ) -> dict[str, Any]:
        if self.student_model is None:
            raise RuntimeError(
                "Distillation metadata "
                "requires fit()."
            )

        student = (
            self.student_model
            .metadata()
        )

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
            "fit_mode": (
                self.fit_mode
            ),
            "teacher_history_hours": int(
                self.config
                .teacher_history_hours
            ),
            "teacher_max_epochs": int(
                self.teacher_epochs
            ),
            "teacher_inner_validation_fraction": float(
                self.config
                .teacher_inner_validation_fraction
            ),
            "teacher_epoch_selection": (
                "chronological-inner-validation-"
                "minimum-validation-loss"
            ),
            "teacher_min_train_rows_semantics": (
                "per-group"
            ),
            "teacher_min_train_rows_per_group": int(
                self.config
                .teacher_min_train_rows
            ),
            "teacher_oof_folds": int(
                self.config
                .teacher_oof_folds
            ),
            "teacher_oof": (
                self.teacher_oof_metadata
            ),
            "distillation_teacher_weight": float(
                self.config
                .distillation_teacher_weight
            ),
            "student_best_iteration": int(
                self.student_epochs
            ),
            "best_iteration": int(
                self.student_epochs
            ),
            "student_selection": (
                selection
            ),
            "elapsed_seconds": float(
                self.elapsed_seconds
            ),
        }