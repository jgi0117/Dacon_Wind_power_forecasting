"""CatBoost 학습 및 추론."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from baram.config import PipelineConfig
from baram.metrics import ficr_aware_grad_hess, ficr_aware_loss, target_score

from .base import RegressionModel


class FICRAwareObjective:
    def __init__(self, ficr_weight: float, temperature: float) -> None:
        self.ficr_weight = ficr_weight
        self.temperature = temperature

    def calc_ders_range(
        self, approxes: list[float], targets: list[float],
        weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        gradient, hessian = ficr_aware_grad_hess(
            np.asarray(targets), np.asarray(approxes),
            ficr_weight=self.ficr_weight, temperature=self.temperature,
        )
        if weights is not None:
            sample_weight = np.asarray(weights, dtype=float)
            gradient *= sample_weight
            hessian *= sample_weight
        return [(-float(g), -float(h)) for g, h in zip(gradient, hessian)]


class FICRAwareMetric:
    def __init__(self, ficr_weight: float, temperature: float) -> None:
        self.ficr_weight = ficr_weight
        self.temperature = temperature

    def is_max_optimal(self) -> bool:
        return False

    def evaluate(
        self, approxes: list[np.ndarray], target: np.ndarray,
        weight: np.ndarray | None,
    ) -> tuple[float, float]:
        loss = ficr_aware_loss(
            np.asarray(target), np.asarray(approxes[0]),
            ficr_weight=self.ficr_weight, temperature=self.temperature,
        )
        count = float(len(target))
        return loss * count, count

    def get_final_error(self, error: float, weight: float) -> float:
        return error / max(weight, 1e-12)


class CatBoostModel(RegressionModel):
    def __init__(self, config: PipelineConfig, iterations: int | None = None) -> None:
        self.config = config
        self.iterations = int(iterations or config.max_epochs)
        self.objective = FICRAwareObjective(
            config.ficr_weight, config.ficr_temperature
        )
        self.metric = FICRAwareMetric(
            config.ficr_weight, config.ficr_temperature
        )
        self.model = CatBoostRegressor(
            loss_function=self.objective,
            eval_metric=self.metric,
            iterations=self.iterations,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            random_seed=config.seed,
            thread_count=config.n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
        self.best_iteration = self.iterations
        self.elapsed_seconds = 0.0
        self.training_history: list[dict[str, float | int | None]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "CatBoostModel":
        started = time.perf_counter()
        kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            kwargs = {
                "eval_set": (X_valid, y_valid),
                "use_best_model": False,
            }
        self.model.fit(X, y, **kwargs)
        evals = self.model.get_evals_result()
        train_metrics = evals.get('learn', {})
        validation_metrics = evals.get('validation', {})
        train_values = next(iter(train_metrics.values()), [])
        validation_values = next(iter(validation_metrics.values()), [])
        self.training_history = [
            {
                'step': step,
                'train_loss': float(train_values[step - 1]),
                'validation_loss': float(validation_values[step - 1]),
                'validation_score': target_score(
                    y_valid.to_numpy(dtype=float),
                    self.model.predict(X_valid, ntree_end=step),
                ),
            }
            for step in range(1, min(len(train_values), len(validation_values)) + 1)
        ]
        if self.training_history:
            self.best_iteration = min(
                self.training_history, key=lambda row: float(row['validation_loss'])
            )['step']
        else:
            self.best_iteration = self.iterations
        self.elapsed_seconds = time.perf_counter() - started
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.model.predict(X, ntree_end=self.best_iteration), dtype=float
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "best_iteration": self.best_iteration,
            "elapsed_seconds": self.elapsed_seconds,
            'max_epochs': self.iterations,
            'loss_name': 'ficr-aware',
            'selection_metric': 'ficr-aware-loss',
            'ficr_weight': self.config.ficr_weight,
            'ficr_temperature': self.config.ficr_temperature,
            'training_history': self.training_history,
        }
