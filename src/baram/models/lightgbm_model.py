"""LightGBM 학습 및 추론."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from baram.config import PipelineConfig

from .base import RegressionModel


class LightGBMModel(RegressionModel):
    def __init__(self, config: PipelineConfig, iterations: int | None = None) -> None:
        self.iterations = iterations or 2500
        self.model = LGBMRegressor(
            objective="l1",
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
        self.elapsed_seconds = 0.0
        self.training_history: list[dict[str, float | int | None]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "LightGBMModel":
        started = time.perf_counter()
        kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            kwargs = {
                "eval_set": [(X, y), (X_valid, y_valid)],
                "eval_names": ["train", "validation"],
                "eval_metric": "l1",
                "callbacks": [early_stopping(150, verbose=False)],
            }
        self.model.fit(X, y, **kwargs)
        self.best_iteration = int(
            getattr(self.model, "best_iteration_", 0) or self.iterations
        )
        evals = getattr(self.model, "evals_result_", {})
        train_values = evals.get("train", {}).get("l1", [])
        validation_values = evals.get("validation", {}).get("l1", [])
        self.training_history = [
            {
                "step": step,
                "train_loss": float(train_values[step - 1]),
                "validation_loss": float(validation_values[step - 1]),
                "validation_score": None,
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
            "loss_name": "mae",
            "selection_metric": "mae",
            "training_history": self.training_history,
        }
