'''PyTabKit RealMLP with chronological competition-score early stopping.'''

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from .base import RegressionModel


LOGGER = logging.getLogger('baram.pipeline')
_SCORE_LOSS = 'baram_score_loss'


def _competition_score_loss(y_pred: Any, y: Any) -> Any:
    import torch
    prediction = y_pred.squeeze(-1)
    actual = y.squeeze(-1)
    valid = torch.isfinite(actual) & torch.isfinite(prediction) & (actual >= 0.10)
    if not torch.any(valid):
        return prediction.new_tensor(float('inf'))
    prediction = prediction[..., valid].clamp(0.0, 1.0)
    actual = actual[..., valid]
    error = (prediction - actual).abs()
    nmae = error.mean(dim=-1)
    unit_price = torch.where(
        error <= 0.06, 4.0, torch.where(error <= 0.08, 3.0, 0.0)
    )
    ficr = (actual * unit_price).sum(dim=-1) / (actual * 4.0).sum(dim=-1).clamp_min(1e-12)
    return -(0.5 * (1.0 - nmae) + 0.5 * ficr)


class RealMLPModel(RegressionModel):
    def __init__(self, config: PipelineConfig, epochs: int | None = None) -> None:
        self.config = config
        self.epochs = int(epochs or config.max_epochs)
        self.best_iteration = self.epochs
        self.model: Any | None = None
        self.elapsed_seconds = 0.0
        self.device = 'cpu'
        self.n_threads = 1
        self.training_history: list[dict[str, float | int | None]] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> 'RealMLPModel':
        from pytabkit import RealMLP_TD_Regressor
        from pytabkit.models.training.metrics import Metrics

        started = time.perf_counter()
        self.device = self.config.device or 'cpu'
        self.n_threads = (
            os.cpu_count() or 1
            if self.config.n_jobs < 1
            else self.config.n_jobs
        )
        original_apply = Metrics.apply

        def score_aware_apply(y_pred: Any, actual: Any, metric_name: str) -> Any:
            if metric_name == _SCORE_LOSS:
                value = _competition_score_loss(y_pred, actual)
                mean_value = float(value.detach().mean().cpu())
                self.training_history.append({
                    'step': len(self.training_history) + 1,
                    'train_loss': None,
                    'validation_loss': mean_value,
                    'validation_score': -mean_value,
                })
                return value
            return original_apply(y_pred, actual, metric_name)

        Metrics.apply = staticmethod(score_aware_apply)
        has_validation = X_valid is not None and y_valid is not None
        fit_X_val = X_valid if has_validation else X.iloc[:min(512, len(X))]
        fit_y_val = y_valid if has_validation else y.iloc[:min(512, len(y))]
        fit_y = np.array(y.to_numpy(dtype=np.float32), copy=True)
        fit_y_val_array = np.array(
            fit_y_val.to_numpy(dtype=np.float32), copy=True
        )
        self.model = RealMLP_TD_Regressor(
            device=self.device, random_state=self.config.seed,
            n_threads=self.n_threads, n_epochs=self.epochs,
            batch_size=self.config.batch_size, n_cv=1, n_refit=0,
            n_ens=8, normalize_output=False,
            train_metric_name='mae', val_metric_name=_SCORE_LOSS,
            use_early_stopping=has_validation,
            early_stopping_additive_patience=self.config.early_stopping_patience,
            early_stopping_multiplicative_patience=1.0,
            stop_epoch=None if has_validation else self.epochs,
            verbosity=2,
        )
        try:
            self.model.fit(
                X, fit_y, X_val=fit_X_val, y_val=fit_y_val_array
            )
        finally:
            Metrics.apply = staticmethod(original_apply)
        self.best_iteration = self._read_best_iteration()
        self.elapsed_seconds = time.perf_counter() - started
        LOGGER.info('RealMLP best epoch=%d', self.best_iteration)
        return self

    def _read_best_iteration(self) -> int:
        candidates = [
            getattr(self.model, 'fit_params_', None),
            getattr(getattr(self.model, 'alg_interface_', None), 'fit_params', None),
        ]
        for candidate in candidates:
            if isinstance(candidate, list) and candidate:
                candidate = candidate[0]
            if isinstance(candidate, dict):
                value = candidate.get('stop_epoch', candidate.get('best_epoch'))
                if isinstance(value, dict):
                    value = value.get(_SCORE_LOSS)
                if value is not None:
                    return max(1, int(value))
        return self.epochs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('RealMLP must be fitted before predict().')
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'RealMLP-TD', 'training': 'supervised-gradient',
            'max_epochs': self.epochs, 'best_iteration': self.best_iteration,
            'validation_metric': 'competition-score', 'n_ens': 8,
            'loss_name': 'mae', 'selection_metric': 'competition-score',
            'training_history': self.training_history,
            'device': self.device, 'n_threads': self.n_threads,
            'elapsed_seconds': self.elapsed_seconds,
        }
