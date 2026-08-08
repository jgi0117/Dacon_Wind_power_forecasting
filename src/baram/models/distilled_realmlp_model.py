'''Teacher-student RealMLP with leakage-safe privileged 12-hour history distillation.'''

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
from .realmlp_model import RealMLPModel


LOGGER = logging.getLogger('baram.pipeline')


def _prediction_frame(
    model: RealMLPModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    values = np.asarray(model.predict(features), dtype=np.float32)
    expected = (len(features), len(TARGET_COLS))
    if values.shape != expected:
        raise ValueError(
            f'Teacher prediction shape mismatch: {values.shape} != {expected}'
        )
    return pd.DataFrame(values, index=features.index, columns=TARGET_COLS)


def _append_privileged_history(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    history_hours: int,
) -> pd.DataFrame:
    '''Append previous feature and target values for teacher-only training.

    The current row target is never included. Only lag 1..history_hours is used.
    These privileged columns are never required by the deployed student.
    '''
    if history_hours < 1:
        raise ValueError('history_hours must be at least 1.')
    if not features.index.equals(targets.index):
        targets = targets.reindex(features.index)
    missing_targets = set(TARGET_COLS) - set(targets.columns)
    if missing_targets:
        raise ValueError(
            f'Missing teacher target columns: {sorted(missing_targets)}'
        )

    result = features.copy()

    for lag in range(1, history_hours + 1):
        shifted_X = features.shift(lag)
        shifted_y = targets.loc[:, TARGET_COLS].shift(lag)

        for column in features.columns:
            result[f'teacher_feature__{column}__lag_{lag:02d}'] = shifted_X[column]

        for target in TARGET_COLS:
            result[f'teacher_target__{target}__lag_{lag:02d}'] = shifted_y[target]

    return result


def _complete_teacher_history_mask(
    privileged_X: pd.DataFrame,
    y: pd.DataFrame,
    history_hours: int,
) -> np.ndarray:
    '''Rows eligible for teacher training/prediction.

    Current y must contain at least one target, and every privileged target lag
    must be observed. Feature NaNs are allowed because RealMLP can preprocess
    them, but missing teacher target history is not allowed.
    '''
    observed_current = y.loc[:, TARGET_COLS].notna().any(axis=1)
    lag_columns = [
        f'teacher_target__{target}__lag_{lag:02d}'
        for lag in range(1, history_hours + 1)
        for target in TARGET_COLS
    ]
    complete_history = privileged_X.loc[:, lag_columns].notna().all(axis=1)
    return (observed_current & complete_history).to_numpy(dtype=bool)


def _blend_distillation_targets(
    hard_targets: pd.DataFrame,
    teacher_oof: pd.DataFrame,
    teacher_weight: float,
) -> tuple[pd.DataFrame, int]:
    '''Blend hard labels with teacher OOF predictions only where both exist.'''
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError('teacher_weight must be between 0 and 1.')

    result = hard_targets.loc[:, TARGET_COLS].astype(float).copy()
    aligned_teacher = teacher_oof.reindex(result.index)
    n_distilled = 0

    for target in TARGET_COLS:
        hard = result[target]
        soft = aligned_teacher[target]
        use = hard.notna() & soft.notna() & np.isfinite(soft.to_numpy(dtype=float))
        result.loc[use, target] = (
            (1.0 - teacher_weight) * hard.loc[use]
            + teacher_weight * soft.loc[use]
        )
        n_distilled += int(use.sum())

    return result, n_distilled


class DistilledTemporalRealMLPModel(RegressionModel):
    '''Student RealMLP distilled from a privileged temporal teacher.

    Teacher input during training:
        current X + previous N hours of X + previous N hours of true y

    Student input during training and inference:
        current X only

    The teacher generates chronological out-of-fold soft predictions. Those
    predictions are blended with the hard labels, so the deployed student can
    absorb part of the temporal target-history signal without requiring target
    history at test time.
    '''

    def __init__(
        self,
        config: PipelineConfig,
        iterations: dict[str, int] | int | None = None,
    ) -> None:
        self.config = replace(
            config,
            group3_stacking=False,
            temporal_prediction_correction=False,
        )
        if isinstance(iterations, dict):
            self.student_epochs = int(
                iterations.get('student', config.max_epochs)
            )
        elif iterations is None:
            self.student_epochs = config.max_epochs
        else:
            self.student_epochs = int(iterations)

        self.teacher_epochs = int(config.teacher_epochs)
        self.student_model: RealMLPModel | None = None
        self.student_selection_metadata: dict[str, Any] = {}
        self.teacher_oof_metadata: dict[str, Any] = {}
        self.fit_mode = ''
        self.elapsed_seconds = 0.0

    @staticmethod
    def _require_targets(y: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if not isinstance(y, pd.DataFrame):
            raise TypeError('Teacher-student distillation requires target DataFrame.')
        missing = set(TARGET_COLS) - set(y.columns)
        if missing:
            raise ValueError(f'Missing distillation targets: {sorted(missing)}')
        return y.loc[:, TARGET_COLS]

    def _teacher_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        '''Generate expanding-window chronological teacher OOF predictions.'''
        history_hours = int(self.config.teacher_history_hours)
        privileged_X = _append_privileged_history(X, y, history_hours)
        eligible = _complete_teacher_history_mask(privileged_X, y, history_hours)
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
                'enabled': False,
                'reason': 'not-enough-eligible-rows',
                'eligible_rows': int(len(eligible_positions)),
                'min_train_rows': int(min_train_rows),
            }
            return oof

        n_folds = int(self.config.teacher_oof_folds)
        prediction_positions = eligible_positions[min_train_rows:]
        folds = [
            fold for fold in np.array_split(prediction_positions, n_folds)
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

            teacher = RealMLPModel(self.config, epochs=self.teacher_epochs)
            teacher.fit(
                privileged_X.iloc[train_positions],
                y.iloc[train_positions],
            )
            fold_X = privileged_X.iloc[fold_positions]
            fold_prediction = _prediction_frame(teacher, fold_X)
            oof.loc[fold_prediction.index, TARGET_COLS] = fold_prediction.to_numpy()

            fold_metadata.append(
                {
                    'fold': fold_number,
                    'train_rows': int(len(train_positions)),
                    'prediction_rows': int(len(fold_positions)),
                    'train_start': X.index[train_positions[0]],
                    'train_end': X.index[train_positions[-1]],
                    'prediction_start': X.index[fold_start_position],
                    'prediction_end': X.index[fold_end_position],
                    'teacher_epochs': self.teacher_epochs,
                }
            )

        predicted_rows = int(oof.notna().any(axis=1).sum())
        self.teacher_oof_metadata = {
            'enabled': predicted_rows > 0,
            'history_hours': history_hours,
            'teacher_epochs': self.teacher_epochs,
            'oof_folds_requested': n_folds,
            'oof_folds_completed': len(fold_metadata),
            'eligible_rows': int(len(eligible_positions)),
            'oof_prediction_rows': predicted_rows,
            'folds': fold_metadata,
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
        self.teacher_oof_metadata['distilled_target_cells'] = n_distilled
        self.teacher_oof_metadata['teacher_weight'] = float(
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

        selection_end = pd.Timestamp(self.config.iteration_selection_end)
        tune_mask = X_valid.index < selection_end
        tune_observed = y_valid.notna().any(axis=1).to_numpy()
        tune_mask = np.asarray(tune_mask) & tune_observed
        if not tune_mask.any():
            raise ValueError('Student epoch-selection validation window is empty.')

        student_selector = RealMLPModel(
            self.config,
            epochs=self.config.max_epochs,
        )
        student_selector.fit(
            X,
            distilled_y,
            X_valid.loc[tune_mask],
            y_valid.loc[tune_mask],
        )
        self.student_selection_metadata = student_selector.metadata()
        self.student_epochs = int(
            self.student_selection_metadata.get(
                'best_iteration', self.config.max_epochs
            )
        )

        self.student_model = RealMLPModel(
            self.config,
            epochs=self.student_epochs,
        )
        self.student_model.fit(X, distilled_y)
        self.fit_mode = 'chronological-oof-distillation-validation'

    def _fit_final(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        distilled_y = self._distilled_targets(X, y)
        self.student_model = RealMLPModel(
            self.config,
            epochs=self.student_epochs,
        )
        self.student_model.fit(X, distilled_y)
        self.fit_mode = 'full-history-oof-distillation-refit'

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> 'DistilledTemporalRealMLPModel':
        targets = self._require_targets(y)
        started = time.perf_counter()

        if X_valid is not None and y_valid is not None:
            valid_targets = self._require_targets(y_valid)
            self._fit_with_validation(X, targets, X_valid, valid_targets)
        elif X_valid is None and y_valid is None:
            self._fit_final(X, targets)
        else:
            raise ValueError('X_valid and y_valid must be provided together.')

        self.elapsed_seconds = time.perf_counter() - started
        LOGGER.info(
            'Teacher-student distillation fit complete: mode=%s, student_epochs=%d',
            self.fit_mode,
            self.student_epochs,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.student_model is None:
            raise RuntimeError('Distilled student model is not fitted.')
        return self.student_model.predict(X)

    def metadata(self) -> dict[str, Any]:
        if self.student_model is None:
            raise RuntimeError('Distillation metadata requires fit().')

        student = self.student_model.metadata()
        selection = self.student_selection_metadata or student
        return {
            **student,
            'model': 'RealMLP-TD-teacher-student-distillation',
            'architecture': (
                'privileged-12h-feature-target-teacher-to-direct-student'
            ),
            'targets': list(TARGET_COLS),
            'teacher_student_distillation': True,
            'teacher_history_hours': int(self.config.teacher_history_hours),
            'distillation_teacher_weight': float(
                self.config.distillation_teacher_weight
            ),
            'teacher_epochs': self.teacher_epochs,
            'teacher_oof_folds': int(self.config.teacher_oof_folds),
            'teacher_min_train_rows': int(self.config.teacher_min_train_rows),
            'teacher_input': 'current-X-plus-lagged-X-plus-lagged-true-y',
            'student_input': 'current-X-only',
            'teacher_used_at_inference': False,
            'fit_mode': self.fit_mode,
            'student_best_iteration': self.student_epochs,
            'best_iteration': self.student_epochs,
            'training_history': selection.get('training_history', []),
            'student_training_history': student.get('training_history', []),
            'teacher_oof': self.teacher_oof_metadata,
            'elapsed_seconds': self.elapsed_seconds,
        }


__all__ = [
    'DistilledTemporalRealMLPModel',
    '_append_privileged_history',
    '_blend_distillation_targets',
]