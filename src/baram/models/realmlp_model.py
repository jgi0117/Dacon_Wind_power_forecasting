'''PyTabKit shared-trunk multi-output RealMLP for Version 4.'''

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import TARGET_COLS, ficr_aware_loss_torch
from .base import RegressionModel


LOGGER = logging.getLogger('baram.pipeline')
_TRAIN_FICR_LOSS = 'baram_train_ficr_aware_loss'
_VAL_FICR_LOSS = 'baram_val_ficr_aware_loss'
_REALMLP_TD_REG_PARAMS = {
    'hidden_sizes': [256] * 3,
    'max_one_hot_cat_size': 9,
    'embedding_size': 8,
    'weight_param': 'ntk',
    'weight_init_mode': 'std',
    'bias_init_mode': 'he+5',
    'bias_lr_factor': 0.1,
    'act': 'mish',
    'use_parametric_act': True,
    'act_lr_factor': 0.1,
    'wd': 2e-2,
    'wd_sched': 'flat_cos',
    'bias_wd_factor': 0.0,
    'block_str': 'w-b-a-d',
    'p_drop': 0.15,
    'p_drop_sched': 'flat_cos',
    'add_front_scale': True,
    'scale_lr_factor': 6.0,
    'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
    'num_emb_type': 'pbld',
    'plr_sigma': 0.1,
    'plr_hidden_1': 16,
    'plr_hidden_2': 4,
    'plr_lr_factor': 0.1,
    # NaN-masked multi-task targets make data-derived clamp bounds invalid.
    # Predictions are clipped to [0, 1] when restored to generation units.
    'clamp_output': False,
    'opt': 'adam',
    'sq_mom': 0.95,
}


def _competition_score_loss(y_pred: Any, y: Any) -> Any:
    import torch

    prediction = y_pred
    actual = y
    if actual.ndim < prediction.ndim:
        actual = actual.unsqueeze(0).expand_as(prediction)
    elif prediction.ndim < actual.ndim:
        prediction = prediction.unsqueeze(0).expand_as(actual)
    if actual.ndim < 2:
        actual = actual.reshape(-1, 1)
        prediction = prediction.reshape(-1, 1)
    valid = torch.isfinite(actual) & torch.isfinite(prediction) & (actual >= 0.10)
    safe_actual = torch.where(valid, actual, torch.zeros_like(actual))
    safe_prediction = torch.where(
        valid, prediction.clamp(0.0, 1.0), torch.zeros_like(prediction)
    )
    error = (safe_prediction - safe_actual).abs()
    valid_float = valid.to(error.dtype)
    count = valid_float.sum(dim=-2)
    safe_count = count.clamp_min(1.0)
    nmae = (error * valid_float).sum(dim=-2) / safe_count
    unit_price = torch.where(
        error <= 0.06, 4.0, torch.where(error <= 0.08, 3.0, 0.0)
    )
    numerator = (safe_actual * unit_price * valid_float).sum(dim=-2)
    denominator = (safe_actual * 4.0 * valid_float).sum(dim=-2)
    ficr = numerator / denominator.clamp_min(1e-12)
    group_score = 0.5 * (1.0 - nmae) + 0.5 * ficr
    valid_groups = count > 0
    valid_group_count = valid_groups.sum(dim=-1).clamp_min(1)
    score = (
        torch.where(valid_groups, group_score, torch.zeros_like(group_score))
        .sum(dim=-1)
        / valid_group_count
    )
    return -score


class RealMLPModel(RegressionModel):
    def __init__(self, config: PipelineConfig, epochs: int | None = None) -> None:
        self.config = config
        self.epochs = int(epochs or config.max_epochs)
        self.best_iteration = self.epochs
        self.model: Any | None = None
        self.elapsed_seconds = 0.0
        self.device = 'cpu'
        self.n_threads = 1
        self.early_stopping_enabled = False
        self.training_history: list[dict[str, Any]] = []
        self.target_names = list(TARGET_COLS)

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
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
        epoch_train_losses: list[float] = []
        epoch_group_train_losses: dict[str, list[float]] = {}
        if isinstance(y, pd.DataFrame):
            self.target_names = [str(column) for column in y.columns]
        else:
            self.target_names = [str(y.name or TARGET_COLS[0])]
        epoch_group_train_losses = {target: [] for target in self.target_names}

        def score_aware_apply(y_pred: Any, actual: Any, metric_name: str) -> Any:
            if metric_name in {_TRAIN_FICR_LOSS, _VAL_FICR_LOSS}:
                value = ficr_aware_loss_torch(
                    actual, y_pred,
                    ficr_weight=self.config.ficr_weight,
                    temperature=self.config.ficr_temperature,
                )
                mean_loss = float(value.detach().mean().cpu())
                group_losses = {
                    target: float(
                        ficr_aware_loss_torch(
                            actual[..., index:index + 1],
                            y_pred[..., index:index + 1],
                            ficr_weight=self.config.ficr_weight,
                            temperature=self.config.ficr_temperature,
                        ).detach().mean().cpu()
                    )
                    for index, target in enumerate(self.target_names)
                }
                if metric_name == _TRAIN_FICR_LOSS:
                    epoch_train_losses.append(mean_loss)
                    for target, loss in group_losses.items():
                        epoch_group_train_losses[target].append(loss)
                    return value
                score_value = _competition_score_loss(y_pred, actual)
                history: dict[str, Any] = {
                    'step': len(self.training_history) + 1,
                    'train_loss': (
                        float(np.mean(epoch_train_losses))
                        if epoch_train_losses else None
                    ),
                    'validation_loss': mean_loss,
                    'validation_score': -float(
                        score_value.detach().mean().cpu()
                    ),
                }
                for index, target in enumerate(self.target_names):
                    train_losses = epoch_group_train_losses[target]
                    history[f'{target}__train_loss'] = (
                        float(np.mean(train_losses)) if train_losses else None
                    )
                    history[f'{target}__validation_loss'] = group_losses[target]
                    group_score = _competition_score_loss(
                        y_pred[..., index:index + 1],
                        actual[..., index:index + 1],
                    )
                    history[f'{target}__validation_score'] = -float(
                        group_score.detach().mean().cpu()
                    )
                self.training_history.append(history)
                epoch_train_losses.clear()
                for losses in epoch_group_train_losses.values():
                    losses.clear()
                return value
            return original_apply(y_pred, actual, metric_name)

        Metrics.apply = staticmethod(score_aware_apply)
        has_validation = X_valid is not None and y_valid is not None
        self.early_stopping_enabled = False
        fallback_rows = min(512, len(X))
        fit_X_val = X_valid if has_validation else X.iloc[-fallback_rows:]
        fit_y_val = y_valid if has_validation else y.iloc[-fallback_rows:]
        fit_y = np.array(y.to_numpy(dtype=np.float32), copy=True)
        fit_y_val_array = np.array(
            fit_y_val.to_numpy(dtype=np.float32), copy=True
        )
        # sklearn validation rejects NaN targets before PyTabKit reaches the
        # custom loss. Zero is below the 10% competition eligibility threshold,
        # so the masked loss gives missing targets exactly zero gradient.
        fit_y[~np.isfinite(fit_y)] = 0.0
        fit_y_val_array[~np.isfinite(fit_y_val_array)] = 0.0
        self.model = RealMLP_TD_Regressor(
            device=self.device, random_state=self.config.seed,
            n_threads=self.n_threads, n_epochs=self.epochs,
            batch_size=self.config.batch_size, n_cv=1, n_refit=0,
            n_ens=8, normalize_output=False,
            lr=self.config.learning_rate, lr_sched='coslog4',
            **_REALMLP_TD_REG_PARAMS,
            train_metric_name=_TRAIN_FICR_LOSS,
            val_metric_name=_VAL_FICR_LOSS,
            use_early_stopping=False,
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
                    value = value.get(_VAL_FICR_LOSS)
                if value is not None:
                    return max(1, int(value))
        return self.epochs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('RealMLP must be fitted before predict().')
        prediction = np.asarray(self.model.predict(X), dtype=float)
        return prediction.reshape(len(X), len(self.target_names))

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'RealMLP-TD', 'training': 'supervised-gradient',
            'architecture': 'shared-trunk-multi-head',
            'targets': self.target_names,
            'max_epochs': self.epochs, 'best_iteration': self.best_iteration,
            'validation_metric': 'ficr-aware-loss', 'n_ens': 8,
            'loss_name': 'ficr-aware', 'selection_metric': 'ficr-aware-loss',
            'ficr_weight': self.config.ficr_weight,
            'ficr_temperature': self.config.ficr_temperature,
            'learning_rate': self.config.learning_rate,
            'lr_schedule': 'coslog4',
            'dropout': 0.15,
            'dropout_schedule': 'flat_cos',
            'weight_decay': 2e-2,
            'weight_decay_schedule': 'flat_cos',
            'optimizer': 'adam',
            'squared_momentum': 0.95,
            'weight_parameterization': 'ntk',
            'target_normalization': False,
            'target_masking': 'finite-target-and-capacity-factor>=0.10',
            'early_stopping': self.early_stopping_enabled,
            'training_history': self.training_history,
            'device': self.device, 'n_threads': self.n_threads,
            'elapsed_seconds': self.elapsed_seconds,
        }
