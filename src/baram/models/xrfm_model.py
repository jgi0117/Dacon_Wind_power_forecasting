'''Supervised xRFM regression with competition-score model selection.'''

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from baram.config import PipelineConfig
from baram.metrics import ficr_aware_loss, target_score
from .base import RegressionModel


LOGGER = logging.getLogger('baram.pipeline')


class XRFMModel(RegressionModel):
    def __init__(self, config: PipelineConfig, iterations: int | None = None) -> None:
        self.config = config
        self.iterations = int(iterations or config.xrfm_iterations)
        self.best_iteration = self.iterations
        self.model: Any | None = None
        self.imputer = SimpleImputer(strategy='median')
        self.elapsed_seconds = 0.0
        self.device = 'cpu'
        self.has_temporal_validation = False
        self.n_threads = 1
        self.metric_fallback_count = 0
        self.training_history: list[dict[str, float | int | None]] = []
        self.metric_trace: list[tuple[float, float]] = []

    def _transform(self, X: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        values = (
            self.imputer.fit_transform(X) if fit else self.imputer.transform(X)
        )
        return pd.DataFrame(values, index=X.index, columns=X.columns)

    def fit(
        self, X: pd.DataFrame, y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> 'XRFMModel':
        from pytabkit import XRFM_D_Regressor
        from pytabkit.models.alg_interfaces.sub_split_interfaces import (
            SingleSplitWrapperAlgInterface,
        )
        from pytabkit.models.alg_interfaces.xrfm_interfaces import (
            xRFMSubSplitInterface,
        )
        from xrfm.rfm_src.metrics import Metric

        started = time.perf_counter()
        self.device = self.config.device or 'cpu'
        self.n_threads = (
            os.cpu_count() or 1
            if self.config.n_jobs < 1
            else self.config.n_jobs
        )
        X_train = self._transform(X, fit=True)
        has_validation = X_valid is not None and y_valid is not None
        self.has_temporal_validation = has_validation
        if has_validation:
            fit_X_val = self._transform(X_valid)
            fit_y_val = y_valid
        else:
            sample = min(512, len(X_train))
            fit_X_val = X_train.iloc[:sample].copy()
            fit_y_val = y.iloc[:sample].copy()

        y_mean = float(y.mean())
        y_scale = max(float(y.std(ddof=0)), 1e-12)
        configured_iterations = self.iterations
        owner = self

        class FICRAwareSelectionMetric(Metric):
            name = 'mse'
            display_name = 'FICR-aware loss'
            should_maximize = False
            task_types = ['reg']
            required_quantities = ['y_true_reg', 'y_pred']

            def _compute(self, **kwargs: Any) -> float:
                actual = kwargs['y_true_reg'].detach().cpu().numpy() * y_scale + y_mean
                prediction = kwargs['y_pred'].detach().cpu().numpy() * y_scale + y_mean
                valid = (
                    np.isfinite(actual)
                    & np.isfinite(prediction)
                    & (actual >= 0.10)
                )
                if not valid.any():
                    owner.metric_fallback_count += 1
                    finite = np.isfinite(actual) & np.isfinite(prediction)
                    if not finite.any():
                        return float('inf')
                    error = np.abs(
                        np.clip(prediction[finite], 0.0, 1.0) - actual[finite]
                    )
                    return float(error.mean())
                loss = ficr_aware_loss(
                    actual, prediction,
                    ficr_weight=owner.config.ficr_weight,
                    temperature=owner.config.ficr_temperature,
                )
                owner.metric_trace.append(
                    (loss, target_score(actual, prediction))
                )
                return loss

        class ConfigurableXRFMRegressor(XRFM_D_Regressor):
            def _create_alg_interface(self, n_cv: int) -> Any:
                model_config = self.get_config()
                model_config['rfm_iters'] = configured_iterations
                return SingleSplitWrapperAlgInterface([
                    xRFMSubSplitInterface(**model_config) for _ in range(n_cv)
                ])

        original_from_name = Metric.from_name

        def score_aware_metric(name: str) -> Metric:
            if name == 'mse':
                return FICRAwareSelectionMetric()
            return original_from_name(name)

        Metric.from_name = staticmethod(score_aware_metric)
        self.model = ConfigurableXRFMRegressor(
            device=self.device, random_state=self.config.seed,
            n_cv=1, n_refit=0, n_threads=self.n_threads,
            bandwidth=10.0, p_interp=1.0, exponent=1.0,
            reg=1e-3, iters=self.iterations, diag=True,
            bandwidth_mode='constant', kernel_type='l2',
            max_leaf_samples=self.config.xrfm_max_leaf_samples,
            val_metric_name='mse', early_stop_rfm=False,
            early_stop_multiplier=1.01,
            M_batch_size=self.config.xrfm_m_batch_size,
            verbosity=2,
        )
        try:
            self.model.fit(
                X_train,
                np.array(y.to_numpy(dtype=np.float32), copy=True),
                X_val=fit_X_val,
                y_val=np.array(
                    fit_y_val.to_numpy(dtype=np.float32), copy=True
                ),
            )
        finally:
            Metric.from_name = staticmethod(original_from_name)
        self.leaf_best_iterations = self._leaf_best_iterations()
        chunk_size = self.iterations + 1
        chunks = [
            self.metric_trace[start:start + chunk_size]
            for start in range(0, len(self.metric_trace), chunk_size)
            if len(self.metric_trace[start:start + chunk_size]) >= self.iterations
        ]
        self.training_history = [
            {
                'step': step,
                'train_loss': None,
                'validation_loss': float(np.mean([
                    chunk[step - 1][0] for chunk in chunks
                ])),
                'validation_score': float(np.mean([
                    chunk[step - 1][1] for chunk in chunks
                ])),
            }
            for step in range(1, self.iterations + 1)
        ] if chunks else []
        if has_validation and self.leaf_best_iterations:
            self.best_iteration = max(
                1, int(round(float(np.median(self.leaf_best_iterations))))
            )
        self.elapsed_seconds = time.perf_counter() - started
        LOGGER.info(
            'xRFM finished: max_iterations=%d, temporal_validation=%s, '
            'metric_fallbacks=%d',
            self.iterations, has_validation, self.metric_fallback_count,
        )
        return self

    def _leaf_best_iterations(self) -> list[int]:
        if self.model is None:
            return []
        interface = getattr(self.model, 'alg_interface_', None)
        sub_interfaces = getattr(interface, 'sub_split_interfaces', [])
        roots = [getattr(item, 'model_', None) for item in sub_interfaces]
        found: list[int] = []
        seen: set[int] = set()

        def visit(value: Any) -> None:
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            best_iter = getattr(value, 'best_iter', None)
            if best_iter is not None:
                found.append(int(best_iter))
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    if key != '_cache':
                        visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
            else:
                trees = getattr(value, 'trees', None)
                if trees is not None:
                    visit(trees)

        visit(roots)
        return found

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('xRFM must be fitted before predict().')
        return np.asarray(self.model.predict(self._transform(X)), dtype=float).reshape(-1)

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'xRFM-0.4.5', 'training': 'supervised-kernel-feature-learning',
            'max_iterations': self.iterations,
            'best_iteration': self.best_iteration,
            'leaf_best_iterations': getattr(self, 'leaf_best_iterations', []),
            'validation_metric': 'ficr-aware-loss',
            'loss_name': 'kernel-ridge-mse',
            'selection_metric': 'ficr-aware-loss',
            'training_history': self.training_history,
            'history_scope': 'mean-leaf-validation; train-loss-unavailable',
            'ficr_weight': self.config.ficr_weight,
            'ficr_temperature': self.config.ficr_temperature,
            'empty_ficr_leaf_metric': 'mae',
            'metric_fallback_count': self.metric_fallback_count,
            'temporal_validation': self.has_temporal_validation,
            'max_leaf_samples': self.config.xrfm_max_leaf_samples,
            'M_batch_size': self.config.xrfm_m_batch_size,
            'kernel_type': 'l2', 'device': self.device,
            'n_threads': self.n_threads,
            'elapsed_seconds': self.elapsed_seconds,
        }
