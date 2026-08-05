"""LightGBM 학습 및 추론."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from baram.config import PipelineConfig
from baram.metrics import ficr_aware_grad_hess, ficr_aware_loss, target_score

from .base import RegressionModel


class LightGBMModel(RegressionModel):
    def __init__(self, config: PipelineConfig, iterations: int | None = None) -> None:
        self.config = config
        self.iterations = int(iterations or config.max_epochs)
        self.model = LGBMRegressor(
            objective=self._objective,
            n_estimators=self.iterations,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=config.seed,
            n_jobs=config.n_jobs,
            verbosity=-1,
        )
        self.best_iteration = self.iterations
        self.base_prediction = 0.0
        self.elapsed_seconds = 0.0
        self.training_history: list[dict[str, float | int | None]] = []

    def _objective(
        self, actual: np.ndarray, prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return ficr_aware_grad_hess(
            actual, prediction,
            ficr_weight=self.config.ficr_weight,
            temperature=self.config.ficr_temperature,
        )

    def _metric(
        self, actual: np.ndarray, prediction: np.ndarray
    ) -> tuple[str, float, bool]:
        return (
            'ficr_aware_loss',
            ficr_aware_loss(
                actual, prediction,
                ficr_weight=self.config.ficr_weight,
                temperature=self.config.ficr_temperature,
            ),
            False,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "LightGBMModel":
        started = time.perf_counter()
        self.base_prediction = float(np.median(y.to_numpy(dtype=float)))
        kwargs: dict[str, Any] = {
            'init_score': np.full(len(X), self.base_prediction, dtype=float),
        }
        if X_valid is not None and y_valid is not None:
            kwargs.update({
                "eval_set": [(X, y), (X_valid, y_valid)],
                "eval_names": ["train", "validation"],
                "eval_metric": self._metric,
                'eval_init_score': [
                    np.full(len(X), self.base_prediction, dtype=float),
                    np.full(len(X_valid), self.base_prediction, dtype=float),
                ],
            })
        self.model.fit(X, y, **kwargs)
        evals = getattr(self.model, "evals_result_", {})
        train_values = evals.get("train", {}).get("ficr_aware_loss", [])
        validation_values = evals.get("validation", {}).get("ficr_aware_loss", [])
        self.training_history = [
            {
                "step": step,
                "train_loss": float(train_values[step - 1]),
                "validation_loss": float(validation_values[step - 1]),
                "validation_score": target_score(
                    y_valid.to_numpy(dtype=float),
                    self.base_prediction
                    + self.model.predict(X_valid, num_iteration=step),
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
            self.base_prediction
            + self.model.predict(X, num_iteration=self.best_iteration), dtype=float
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "best_iteration": self.best_iteration,
            "elapsed_seconds": self.elapsed_seconds,
            "max_epochs": self.iterations,
            "base_prediction": self.base_prediction,
            "loss_name": "ficr-aware",
            "selection_metric": "ficr-aware-loss",
            "ficr_weight": self.config.ficr_weight,
            "ficr_temperature": self.config.ficr_temperature,
            "training_history": self.training_history,
        }
