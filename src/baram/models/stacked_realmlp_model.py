'''Leakage-safe two-stage RealMLP for a group 3 private capacity model.'''

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
_STAGE1_TARGETS = list(TARGET_COLS[:2])
_GROUP3_TARGET = TARGET_COLS[2]
STACK_FEATURE_COLUMNS = (
    'stack__group_1_cf',
    'stack__group_2_cf',
    'stack__proxy_mean_cf',
    'stack__proxy_difference_cf',
)


def _append_stack_features(
    features: pd.DataFrame, stage1_prediction: np.ndarray
) -> pd.DataFrame:
    prediction = np.asarray(stage1_prediction, dtype=np.float32)
    if prediction.shape != (len(features), 2):
        raise ValueError(
            'Stage-1 prediction shape mismatch: '
            f'{prediction.shape} != {(len(features), 2)}'
        )
    prediction = np.clip(prediction, 0.0, 1.0)
    result = features.copy()
    result[STACK_FEATURE_COLUMNS[0]] = prediction[:, 0]
    result[STACK_FEATURE_COLUMNS[1]] = prediction[:, 1]
    result[STACK_FEATURE_COLUMNS[2]] = prediction.mean(axis=1)
    result[STACK_FEATURE_COLUMNS[3]] = prediction[:, 0] - prediction[:, 1]
    return result


def _merge_training_histories(
    stage1: list[dict[str, Any]], group3: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    '''Keep stage-1 curves and attach the private group 3 curves by epoch.'''
    merged: dict[int, dict[str, Any]] = {
        int(item['step']): dict(item) for item in stage1
    }
    for item in group3:
        step = int(item['step'])
        row = merged.setdefault(step, {'step': step})
        for key, value in item.items():
            if key.startswith(f'{_GROUP3_TARGET}__'):
                row[key] = value
            elif key != 'step':
                row[f'group3_stage__{key}'] = value
    return [merged[step] for step in sorted(merged)]


class StackedGroup3RealMLPModel(RegressionModel):
    '''Predict groups 1/2 jointly, then group 3 with temporal OOF proxies.'''

    def __init__(
        self,
        config: PipelineConfig,
        iterations: dict[str, int] | int | None = None,
    ) -> None:
        self.config = replace(
            config,
            ficr_loss='sigmoid',
            group3_reliability_weighting=False,
            temporal_group_dro=False,
            ficr_boundary_consistency_weight=0.0,
        )
        if isinstance(iterations, dict):
            self.stage1_epochs = int(
                iterations.get('stage1', config.max_epochs)
            )
            self.group3_epochs = int(
                iterations.get('group3', config.max_epochs)
            )
        elif iterations is not None:
            self.stage1_epochs = int(iterations)
            self.group3_epochs = int(iterations)
        else:
            self.stage1_epochs = config.max_epochs
            self.group3_epochs = config.max_epochs
        self.stage1_model: RealMLPModel | None = None
        self.group3_model: RealMLPModel | None = None
        self.oof_audit: list[dict[str, Any]] = []
        self.elapsed_seconds = 0.0

    def _temporal_oof_stage1(
        self, X: pd.DataFrame, y: pd.DataFrame, epochs: int
    ) -> pd.DataFrame:
        observed_group3 = y[_GROUP3_TARGET].notna().to_numpy()
        years = sorted(
            np.unique(pd.DatetimeIndex(X.index[observed_group3]).year).tolist()
        )
        oof = pd.DataFrame(
            np.nan, index=X.index, columns=_STAGE1_TARGETS, dtype=np.float32
        )
        self.oof_audit = []
        for year in years:
            cutoff = pd.Timestamp(year=int(year), month=1, day=1)
            holdout = observed_group3 & (X.index.year == int(year))
            train = (
                (X.index < cutoff)
                & y.loc[:, _STAGE1_TARGETS].notna().any(axis=1).to_numpy()
            )
            if int(train.sum()) < 256:
                LOGGER.warning(
                    'Skip group 3 stack OOF year=%d: only %d prior rows',
                    year,
                    int(train.sum()),
                )
                continue
            LOGGER.info(
                'Stage-1 temporal OOF: train before %d (%d rows), predict %d rows',
                year,
                int(train.sum()),
                int(holdout.sum()),
            )
            fold_model = RealMLPModel(
                self.config, epochs=epochs
            )
            fold_model.fit(
                X.loc[train], y.loc[train, _STAGE1_TARGETS]
            )
            oof.loc[holdout, _STAGE1_TARGETS] = fold_model.predict(
                X.loc[holdout]
            )
            self.oof_audit.append({
                'holdout_year': int(year),
                'train_start': X.index[train].min(),
                'train_end': X.index[train].max(),
                'n_train': int(train.sum()),
                'n_holdout': int(holdout.sum()),
                'epochs': epochs,
            })
        return oof

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> 'StackedGroup3RealMLPModel':
        if not isinstance(y, pd.DataFrame):
            raise TypeError('Stacked group 3 training requires three targets.')
        missing_targets = set(TARGET_COLS) - set(y.columns)
        if missing_targets:
            raise ValueError(f'Missing stacking targets: {sorted(missing_targets)}')
        has_validation = X_valid is not None and y_valid is not None
        if has_validation and not isinstance(y_valid, pd.DataFrame):
            raise TypeError('Stacked validation requires three targets.')

        started = time.perf_counter()
        stage1_train = y.loc[:, _STAGE1_TARGETS].notna().any(axis=1).to_numpy()
        self.stage1_model = RealMLPModel(
            self.config, epochs=self.stage1_epochs
        )
        stage1_valid_y = (
            y_valid.loc[:, _STAGE1_TARGETS] if has_validation else None
        )
        self.stage1_model.fit(
            X.loc[stage1_train],
            y.loc[stage1_train, _STAGE1_TARGETS],
            X_valid,
            stage1_valid_y,
        )
        oof_epochs = (
            self.stage1_model.best_iteration
            if has_validation else self.stage1_epochs
        )
        oof = self._temporal_oof_stage1(X, y, epochs=oof_epochs)
        group3_train = (
            y[_GROUP3_TARGET].notna()
            & oof.loc[:, _STAGE1_TARGETS].notna().all(axis=1)
        ).to_numpy()
        if int(group3_train.sum()) < 256:
            raise ValueError(
                'Insufficient leakage-free group 3 stack rows: '
                f'{int(group3_train.sum())}'
            )
        group3_X = _append_stack_features(
            X.loc[group3_train],
            oof.loc[group3_train, _STAGE1_TARGETS].to_numpy(),
        )
        group3_y = y.loc[group3_train, [_GROUP3_TARGET]]

        group3_valid_X = None
        group3_valid_y = None
        if has_validation:
            valid_group3 = y_valid[_GROUP3_TARGET].notna().to_numpy()
            valid_stage1 = self.stage1_model.predict(X_valid.loc[valid_group3])
            group3_valid_X = _append_stack_features(
                X_valid.loc[valid_group3], valid_stage1
            )
            group3_valid_y = y_valid.loc[valid_group3, [_GROUP3_TARGET]]

        self.group3_model = RealMLPModel(
            self.config, epochs=self.group3_epochs
        )
        self.group3_model.fit(
            group3_X, group3_y, group3_valid_X, group3_valid_y
        )
        self.elapsed_seconds = time.perf_counter() - started
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.stage1_model is None or self.group3_model is None:
            raise RuntimeError('Stacked RealMLP must be fitted before predict().')
        stage1 = self.stage1_model.predict(X)
        group3_X = _append_stack_features(X, stage1)
        group3 = self.group3_model.predict(group3_X)
        return np.concatenate([stage1, group3], axis=1)

    def metadata(self) -> dict[str, Any]:
        if self.stage1_model is None or self.group3_model is None:
            raise RuntimeError('Stacked RealMLP metadata requires a fitted model.')
        stage1 = self.stage1_model.metadata()
        group3 = self.group3_model.metadata()
        return {
            **stage1,
            'architecture': 'two-stage-realmlp-with-private-group3-model',
            'targets': list(TARGET_COLS),
            'loss_name': 'sigmoid-ficr-aware',
            'ficr_loss': 'sigmoid',
            'group3_reliability_weighting': False,
            'stacking': True,
            'stacking_method': 'expanding-year-temporal-oof',
            'stack_features': list(STACK_FEATURE_COLUMNS),
            'stage1_targets': list(_STAGE1_TARGETS),
            'stage1_best_iteration': stage1['best_iteration'],
            'group3_best_iteration': group3['best_iteration'],
            'best_iteration': group3['best_iteration'],
            'stage1_epochs': self.stage1_epochs,
            'group3_epochs': self.group3_epochs,
            'oof_stage1_epochs': sorted({
                int(item['epochs']) for item in self.oof_audit
            }),
            'group3_fit_rows': int(
                sum(item['n_holdout'] for item in self.oof_audit)
            ),
            'oof_audit': self.oof_audit,
            'training_history': _merge_training_histories(
                stage1['training_history'], group3['training_history']
            ),
            'stage1_elapsed_seconds': stage1['elapsed_seconds'],
            'group3_elapsed_seconds': group3['elapsed_seconds'],
            'elapsed_seconds': self.elapsed_seconds,
        }
