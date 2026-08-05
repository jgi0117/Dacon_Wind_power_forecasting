"""CatBoost 학습 및 추론."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from baram.config import PipelineConfig

from .base import RegressionModel


class CatBoostModel(RegressionModel):
    def __init__(self, config: PipelineConfig, iterations: int | None = None) -> None:
        self.iterations = iterations or 2000
        self.model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
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
                "early_stopping_rounds": 150,
                "use_best_model": True,
            }
        self.model.fit(X, y, **kwargs)
        raw_best = self.model.get_best_iteration()
        best = int(raw_best) if raw_best is not None else -1
        self.best_iteration = best + 1 if best >= 0 else self.iterations
        evals = self.model.get_evals_result()
        train_values = evals.get('learn', {}).get('MAE', [])
        validation_values = evals.get('validation', {}).get('MAE', [])
        self.training_history = [
            {
                'step': step,
                'train_loss': float(train_values[step - 1]),
                'validation_loss': float(validation_values[step - 1]),
                'validation_score': None,
            }
            for step in range(1, min(len(train_values), len(validation_values)) + 1)
        ]
        self.elapsed_seconds = time.perf_counter() - started
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)

    def metadata(self) -> dict[str, Any]:
        return {
            "best_iteration": self.best_iteration,
            "elapsed_seconds": self.elapsed_seconds,
            'loss_name': 'mae',
            'selection_metric': 'mae',
            'training_history': self.training_history,
        }
