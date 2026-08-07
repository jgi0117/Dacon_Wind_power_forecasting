'''Two-stage RealMLP using leakage-safe temporal base-prediction context.'''

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import TARGET_COLS

from .base import RegressionModel
from .realmlp_model import RealMLPModel


N_PREDICTION_LAGS = 12


def _append_prediction_context(
    features: pd.DataFrame,
    prediction_history: pd.DataFrame,
) -> pd.DataFrame:
    '''Append current and previous 12 direct predictions without recursion.'''
    missing = set(TARGET_COLS) - set(prediction_history.columns)
    if missing:
        raise ValueError(f'Missing base prediction columns: {sorted(missing)}')
    history = prediction_history.loc[:, TARGET_COLS].sort_index()
    if history.index.has_duplicates:
        raise ValueError('Base prediction history contains duplicate timestamps.')
    result = features.copy()
    for target in TARGET_COLS:
        series = history[target]
        for lag in range(N_PREDICTION_LAGS + 1):
            name = f'base_prediction__{target}__lag_{lag:02d}'
            values = series.shift(lag).reindex(features.index)
            if values.isna().any():
                raise ValueError(
                    f'Incomplete base prediction context for {name}.'
                )
            result[name] = values.to_numpy(dtype=np.float32)
    return result


def _prediction_frame(
    model: RealMLPModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    values = np.asarray(model.predict(features), dtype=np.float32)
    expected = (len(features), len(TARGET_COLS))
    if values.shape != expected:
        raise ValueError(f'Base prediction shape mismatch: {values.shape} != {expected}')
    return pd.DataFrame(values, index=features.index, columns=TARGET_COLS)


class TemporalCorrectionRealMLPModel(RegressionModel):
    '''Correct direct predictions using their non-recursive 12-hour history.'''

    def __init__(
        self,
        config: PipelineConfig,
        iterations: dict[str, int] | int | None = None,
    ) -> None:
        self.config = replace(config, group3_stacking=False)
        if isinstance(iterations, dict):
            self.base_epochs = int(iterations.get('base', config.max_epochs))
            self.correction_epochs = int(
                iterations.get('correction', config.max_epochs)
            )
        elif iterations is None:
            self.base_epochs = config.max_epochs
            self.correction_epochs = config.max_epochs
        else:
            self.base_epochs = int(iterations)
            self.correction_epochs = int(iterations)
        self.base_model: RealMLPModel | None = None
        self.correction_model: RealMLPModel | None = None
        self.base_selection_metadata: dict[str, Any] = {}
        self.correction_selection_metadata: dict[str, Any] = {}
        self.fit_mode = ''
        self.fit_tail: pd.DataFrame | None = None
        self.elapsed_seconds = 0.0

    @staticmethod
    def _require_targets(y: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if not isinstance(y, pd.DataFrame):
            raise TypeError('Temporal correction requires three target columns.')
        missing = set(TARGET_COLS) - set(y.columns)
        if missing:
            raise ValueError(f'Missing temporal correction targets: {sorted(missing)}')
        return y.loc[:, TARGET_COLS]

    def _base_context(
        self,
        model: RealMLPModel,
        history_X: pd.DataFrame,
        current_X: pd.DataFrame,
    ) -> pd.DataFrame:
        combined = pd.concat([
            history_X.tail(N_PREDICTION_LAGS), current_X
        ]).sort_index()
        prediction = _prediction_frame(model, combined)
        return _append_prediction_context(current_X, prediction)

    def _fit_with_validation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
    ) -> None:
        base_tune_end = pd.Timestamp(self.config.iteration_selection_end)
        base_tune = X_valid.index < base_tune_end
        observed = y.notna().any(axis=1).to_numpy()
        base_selector = RealMLPModel(self.config, epochs=self.config.max_epochs)
        base_selector.fit(
            X.loc[observed], y.loc[observed],
            X_valid.loc[base_tune], y_valid.loc[base_tune]
        )
        self.base_selection_metadata = base_selector.metadata()
        self.base_epochs = int(self.base_selection_metadata['best_iteration'])

        base_oof = RealMLPModel(self.config, epochs=self.base_epochs)
        base_oof.fit(X.loc[observed], y.loc[observed])
        correction_X = self._base_context(base_oof, X, X_valid)

        correction_train_end = pd.Timestamp(
            self.config.correction_validation_start
        )
        correction_valid_end = pd.Timestamp(self.config.comparison_start)
        correction_train = X_valid.index < correction_train_end
        correction_valid = (
            (X_valid.index >= correction_train_end)
            & (X_valid.index < correction_valid_end)
        )
        if not correction_train.any() or not correction_valid.any():
            raise ValueError('Correction train or validation window is empty.')
        correction_observed = y_valid.notna().any(axis=1).to_numpy()
        correction_train &= correction_observed
        correction_valid &= correction_observed
        correction_selector = RealMLPModel(
            self.config, epochs=self.config.max_epochs
        )
        correction_selector.fit(
            correction_X.loc[correction_train],
            y_valid.loc[correction_train],
            correction_X.loc[correction_valid],
            y_valid.loc[correction_valid],
        )
        self.correction_selection_metadata = correction_selector.metadata()
        self.correction_epochs = int(
            self.correction_selection_metadata['best_iteration']
        )

        correction_refit = (
            (X_valid.index < correction_valid_end)
            & correction_observed
        )
        self.correction_model = RealMLPModel(
            self.config, epochs=self.correction_epochs
        )
        self.correction_model.fit(
            correction_X.loc[correction_refit],
            y_valid.loc[correction_refit],
        )
        self.base_model = base_oof
        self.fit_tail = X.tail(N_PREDICTION_LAGS)
        self.fit_mode = 'chronological-validation'

    def _fit_final(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        oof_start = pd.Timestamp(
            year=self.config.temporal_oof_year, month=1, day=1
        )
        observed = y.notna().any(axis=1).to_numpy()
        prior = X.index < oof_start
        oof = X.index >= oof_start
        if not prior.any() or not oof.any():
            raise ValueError('Final temporal OOF split is empty.')

        base_oof = RealMLPModel(self.config, epochs=self.base_epochs)
        base_oof.fit(X.loc[prior & observed], y.loc[prior & observed])
        correction_X = self._base_context(
            base_oof, X.loc[prior], X.loc[oof]
        )
        self.correction_model = RealMLPModel(
            self.config, epochs=self.correction_epochs
        )
        self.correction_model.fit(
            correction_X.loc[observed[oof]],
            y.loc[oof].loc[observed[oof]],
        )

        self.base_model = RealMLPModel(
            self.config, epochs=self.base_epochs
        )
        self.base_model.fit(X.loc[observed], y.loc[observed])
        self.fit_tail = X.tail(N_PREDICTION_LAGS)
        self.fit_mode = 'full-history-refit'

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> 'TemporalCorrectionRealMLPModel':
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
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if (
            self.base_model is None
            or self.correction_model is None
            or self.fit_tail is None
        ):
            raise RuntimeError('Temporal correction model is not fitted.')
        correction_X = self._base_context(self.base_model, self.fit_tail, X)
        return self.correction_model.predict(correction_X)

    def metadata(self) -> dict[str, Any]:
        if self.base_model is None or self.correction_model is None:
            raise RuntimeError('Temporal correction metadata requires fit().')
        base = self.base_model.metadata()
        correction = self.correction_model.metadata()
        base_selection = self.base_selection_metadata or base
        correction_selection = self.correction_selection_metadata or correction
        return {
            **correction,
            'model': 'RealMLP-TD-temporal-correction',
            'architecture': 'direct-base-plus-12h-oof-prediction-correction',
            'targets': list(TARGET_COLS),
            'temporal_prediction_correction': True,
            'prediction_context_hours': N_PREDICTION_LAGS,
            'prediction_context_type': 'base-prediction-non-recursive',
            'fit_mode': self.fit_mode,
            'base_best_iteration': self.base_epochs,
            'correction_best_iteration': self.correction_epochs,
            'best_iteration': self.correction_epochs,
            'base_training_history': base_selection.get('training_history', []),
            'training_history': correction_selection.get('training_history', []),
            'base_elapsed_seconds': base.get('elapsed_seconds'),
            'correction_elapsed_seconds': correction.get('elapsed_seconds'),
            'elapsed_seconds': self.elapsed_seconds,
        }


__all__ = [
    'N_PREDICTION_LAGS',
    'TemporalCorrectionRealMLPModel',
    '_append_prediction_context',
]
