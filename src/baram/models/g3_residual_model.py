"""Second-stage RealMLP correction for kpx_group_3.

The model learns:
    residual_cf = actual_group3_cf - base_group3_cf

It is intentionally separate from the base Teacher-Student model.  This keeps
the shared model intact and lets the experiment test whether systematic G3
calibration error can be corrected without changing G1/G2.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import CAPACITY_KWH


G3_TARGET = "kpx_group_3"


class Group3ResidualRealMLP:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        epochs: int | None = None,
    ) -> None:
        self.config = config
        self.epochs = int(epochs or config.group3_residual_epochs)
        self.model: Any | None = None
        self.elapsed_seconds = 0.0

    @staticmethod
    def build_features(
        X: pd.DataFrame,
        base_prediction_kw: pd.Series,
    ) -> pd.DataFrame:
        """Add leakage-free base prediction and calendar context."""
        prediction = pd.Series(
            base_prediction_kw,
            index=base_prediction_kw.index,
            dtype=float,
        ).reindex(X.index)

        if prediction.isna().any():
            missing = int(prediction.isna().sum())
            raise ValueError(
                f"G3 residual base prediction is missing for {missing} rows."
            )

        features = X.copy()
        base_cf = prediction / CAPACITY_KWH[G3_TARGET]

        features["g3_residual__base_prediction_cf"] = base_cf
        features["g3_residual__base_prediction_cf_sq"] = base_cf.square()

        index = pd.DatetimeIndex(features.index)
        month = index.month.to_numpy(dtype=float)
        hour = index.hour.to_numpy(dtype=float)
        dayofyear = index.dayofyear.to_numpy(dtype=float)

        features["g3_residual__month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        features["g3_residual__month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        features["g3_residual__hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        features["g3_residual__hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        features["g3_residual__doy_sin"] = np.sin(
            2.0 * np.pi * dayofyear / 366.0
        )
        features["g3_residual__doy_cos"] = np.cos(
            2.0 * np.pi * dayofyear / 366.0
        )

        return features

    @staticmethod
    def residual_target(
        actual_kw: pd.Series,
        base_prediction_kw: pd.Series,
    ) -> pd.Series:
        actual_cf = actual_kw.astype(float) / CAPACITY_KWH[G3_TARGET]
        base_cf = base_prediction_kw.astype(float) / CAPACITY_KWH[G3_TARGET]
        target = actual_cf - base_cf
        target.name = "g3_residual_cf"
        return target

    def fit(
        self,
        X: pd.DataFrame,
        base_prediction_kw: pd.Series,
        actual_kw: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        base_prediction_valid_kw: pd.Series | None = None,
        actual_valid_kw: pd.Series | None = None,
    ) -> "Group3ResidualRealMLP":
        from pytabkit import RealMLP_TD_Regressor

        started = time.perf_counter()

        train_features = self.build_features(X, base_prediction_kw)
        train_target = self.residual_target(
            actual_kw.reindex(X.index),
            base_prediction_kw.reindex(X.index),
        )

        valid_train = np.isfinite(train_target.to_numpy(dtype=float))
        train_features = train_features.loc[valid_train]
        train_target = train_target.loc[valid_train].to_frame()

        fit_kwargs: dict[str, Any] = {}

        if X_valid is not None:
            if base_prediction_valid_kw is None or actual_valid_kw is None:
                raise ValueError(
                    "Residual validation requires base prediction and actual y."
                )

            valid_features = self.build_features(
                X_valid,
                base_prediction_valid_kw,
            )
            valid_target = self.residual_target(
                actual_valid_kw.reindex(X_valid.index),
                base_prediction_valid_kw.reindex(X_valid.index),
            )
            valid_rows = np.isfinite(valid_target.to_numpy(dtype=float))
            valid_features = valid_features.loc[valid_rows]
            valid_target = valid_target.loc[valid_rows].to_frame()

            fit_kwargs = {
                "X_val": valid_features,
                "y_val": valid_target,
            }

        self.model = RealMLP_TD_Regressor(
            device=self.config.device or "cpu",
            random_state=self.config.seed,
            n_threads=self.config.n_jobs,
            n_epochs=self.epochs,
            batch_size=self.config.group3_residual_batch_size,
            n_cv=1,
            n_refit=0,
            n_ens=self.config.group3_residual_ensemble,
            normalize_output=True,
            lr=self.config.group3_residual_learning_rate,
            lr_sched="coslog4",
            hidden_sizes=[256, 256, 256],
            act="mish",
            p_drop=0.15,
            opt="adam",
            verbosity=2,
        )

        self.model.fit(
            train_features,
            train_target,
            **fit_kwargs,
        )

        self.elapsed_seconds = time.perf_counter() - started
        return self

    def predict_correction_cf(
        self,
        X: pd.DataFrame,
        base_prediction_kw: pd.Series,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Group3ResidualRealMLP must be fitted first.")

        features = self.build_features(X, base_prediction_kw)
        correction = np.asarray(
            self.model.predict(features),
            dtype=float,
        ).reshape(-1)

        limit = float(self.config.group3_residual_max_abs_correction)
        return np.clip(correction, -limit, limit)

    def correct_prediction_kw(
        self,
        X: pd.DataFrame,
        base_prediction_kw: pd.Series,
    ) -> pd.Series:
        base = base_prediction_kw.reindex(X.index).astype(float)
        base_cf = base.to_numpy(dtype=float) / CAPACITY_KWH[G3_TARGET]
        correction = self.predict_correction_cf(X, base)
        corrected_cf = np.clip(base_cf + correction, 0.0, 1.0)

        return pd.Series(
            corrected_cf * CAPACITY_KWH[G3_TARGET],
            index=X.index,
            name=G3_TARGET,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "Group3ResidualRealMLP",
            "target": G3_TARGET,
            "epochs": self.epochs,
            "ensemble": self.config.group3_residual_ensemble,
            "learning_rate": self.config.group3_residual_learning_rate,
            "batch_size": self.config.group3_residual_batch_size,
            "max_abs_correction_cf": (
                self.config.group3_residual_max_abs_correction
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "target_definition": "actual_cf - base_prediction_cf",
            "feature_definition": (
                "raw X + base G3 prediction + cyclic calendar features"
            ),
        }
