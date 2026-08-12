"""Second-stage RealMLP correction for kpx_group_3.

The model learns:
    residual_cf = actual_group3_cf - base_group3_cf

It is intentionally separate from the base Teacher-Student model.
This keeps the shared model intact and lets the experiment test whether
systematic G3 calibration error can be corrected without changing G1/G2.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import CAPACITY_KWH


LOGGER = logging.getLogger("baram.pipeline")

G3_TARGET = "kpx_group_3"

_RESIDUAL_TRAIN_LOSS = "baram_g3_residual_train_mse"
_RESIDUAL_VAL_LOSS = "baram_g3_residual_val_mse"


class Group3ResidualRealMLP:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        epochs: int | None = None,
    ) -> None:
        self.config = config

        self.epochs = int(
            epochs
            or config.group3_residual_epochs
        )

        self.model: Any | None = None
        self.elapsed_seconds = 0.0

        self.training_history: list[
            dict[str, float | int]
        ] = []

    @staticmethod
    def build_features(
        X: pd.DataFrame,
        base_prediction_kw: pd.Series,
    ) -> pd.DataFrame:
        """Build leakage-free G3 residual features."""

        prediction = pd.Series(
            base_prediction_kw,
            index=base_prediction_kw.index,
            dtype=float,
        ).reindex(X.index)

        if prediction.isna().any():
            missing = int(
                prediction.isna().sum()
            )

            raise ValueError(
                "G3 residual base prediction "
                f"is missing for {missing} rows."
            )

        features = X.copy()

        base_cf = (
            prediction
            / CAPACITY_KWH[G3_TARGET]
        )

        features[
            "g3_residual__base_prediction_cf"
        ] = base_cf

        # pandas 2.x / 3.x compatible
        features[
            "g3_residual__base_prediction_cf_sq"
        ] = base_cf ** 2

        index = pd.DatetimeIndex(
            features.index
        )

        month = index.month.to_numpy(
            dtype=float
        )

        hour = index.hour.to_numpy(
            dtype=float
        )

        dayofyear = (
            index.dayofyear.to_numpy(
                dtype=float
            )
        )

        features[
            "g3_residual__month_sin"
        ] = np.sin(
            2.0 * np.pi
            * month
            / 12.0
        )

        features[
            "g3_residual__month_cos"
        ] = np.cos(
            2.0 * np.pi
            * month
            / 12.0
        )

        features[
            "g3_residual__hour_sin"
        ] = np.sin(
            2.0 * np.pi
            * hour
            / 24.0
        )

        features[
            "g3_residual__hour_cos"
        ] = np.cos(
            2.0 * np.pi
            * hour
            / 24.0
        )

        features[
            "g3_residual__doy_sin"
        ] = np.sin(
            2.0 * np.pi
            * dayofyear
            / 366.0
        )

        features[
            "g3_residual__doy_cos"
        ] = np.cos(
            2.0 * np.pi
            * dayofyear
            / 366.0
        )

        return features

    @staticmethod
    def residual_target(
        actual_kw: pd.Series,
        base_prediction_kw: pd.Series,
    ) -> pd.Series:
        """Residual target in capacity-factor units."""

        actual_cf = (
            actual_kw.astype(float)
            / CAPACITY_KWH[G3_TARGET]
        )

        base_cf = (
            base_prediction_kw.astype(float)
            / CAPACITY_KWH[G3_TARGET]
        )

        target = (
            actual_cf
            - base_cf
        )

        target.name = (
            "g3_residual_cf"
        )

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

        from pytabkit import (
            RealMLP_TD_Regressor,
        )

        from pytabkit.models.training.metrics import (
            Metrics,
        )

        started = time.perf_counter()

        # =====================================================
        # Train data
        # =====================================================

        train_features = (
            self.build_features(
                X,
                base_prediction_kw,
            )
        )

        train_target = (
            self.residual_target(
                actual_kw.reindex(
                    X.index
                ),
                base_prediction_kw.reindex(
                    X.index
                ),
            )
        )

        valid_train = np.isfinite(
            train_target.to_numpy(
                dtype=float
            )
        )

        train_features = (
            train_features.loc[
                valid_train
            ]
        )

        train_target = (
            train_target.loc[
                valid_train
            ].to_frame()
        )

        if len(train_features) == 0:
            raise ValueError(
                "No finite rows remain for "
                "G3 residual training."
            )

        LOGGER.info(
            "G3 residual train prepared: "
            "rows=%d, features=%d",
            len(train_features),
            train_features.shape[1],
        )

        # =====================================================
        # Validation data
        # =====================================================

        fit_kwargs: dict[
            str,
            Any,
        ] = {}

        has_validation = (
            X_valid is not None
        )

        if has_validation:

            if (
                base_prediction_valid_kw
                is None
                or actual_valid_kw
                is None
            ):
                raise ValueError(
                    "Residual validation requires "
                    "base prediction and actual y."
                )

            valid_features = (
                self.build_features(
                    X_valid,
                    base_prediction_valid_kw,
                )
            )

            valid_target = (
                self.residual_target(
                    actual_valid_kw.reindex(
                        X_valid.index
                    ),
                    base_prediction_valid_kw.reindex(
                        X_valid.index
                    ),
                )
            )

            valid_rows = np.isfinite(
                valid_target.to_numpy(
                    dtype=float
                )
            )

            valid_features = (
                valid_features.loc[
                    valid_rows
                ]
            )

            valid_target = (
                valid_target.loc[
                    valid_rows
                ].to_frame()
            )

            if len(valid_features) == 0:
                raise ValueError(
                    "No finite rows remain for "
                    "G3 residual validation."
                )

            if (
                list(
                    train_features.columns
                )
                != list(
                    valid_features.columns
                )
            ):
                raise ValueError(
                    "G3 residual train/validation "
                    "feature schemas differ."
                )

            LOGGER.info(
                "G3 residual validation prepared: "
                "rows=%d, features=%d",
                len(valid_features),
                valid_features.shape[1],
            )

            fit_kwargs = {
                "X_val": valid_features,
                "y_val": valid_target,
            }

        # =====================================================
        # CPU threads
        # =====================================================

        n_threads = (
            os.cpu_count() or 1
            if self.config.n_jobs < 1
            else self.config.n_jobs
        )

        if n_threads < 1:
            n_threads = 1

        # =====================================================
        # Custom metric logging
        #
        # verbosity=0 suppresses PyTabKit's very long feature
        # output. Epoch losses are logged here instead.
        # =====================================================

        original_apply = Metrics.apply

        epoch_train_losses: list[
            float
        ] = []

        self.training_history = []

        def residual_metric_apply(
            y_pred: Any,
            actual: Any,
            metric_name: str,
        ) -> Any:
            import torch

            if metric_name not in {
                _RESIDUAL_TRAIN_LOSS,
                _RESIDUAL_VAL_LOSS,
            }:
                return original_apply(
                    y_pred,
                    actual,
                    metric_name,
                )

            prediction = y_pred
            target = actual

            if target.ndim < prediction.ndim:
                target = (
                    target.unsqueeze(0)
                    .expand_as(prediction)
                )

            elif prediction.ndim < target.ndim:
                prediction = (
                    prediction.unsqueeze(0)
                    .expand_as(target)
                )

            squared_error = (
                prediction
                - target
            ) ** 2

            loss = (
                squared_error.mean()
            )

            scalar_loss = float(
                loss.detach()
                .mean()
                .cpu()
            )

            if (
                metric_name
                == _RESIDUAL_TRAIN_LOSS
            ):
                epoch_train_losses.append(
                    scalar_loss
                )

                return loss

            # Validation metric is evaluated once
            # per epoch after train metric calls.
            train_loss = (
                float(
                    np.mean(
                        epoch_train_losses
                    )
                )
                if epoch_train_losses
                else float("nan")
            )

            epoch = (
                len(
                    self.training_history
                )
                + 1
            )

            history = {
                "epoch": epoch,
                "train_loss": (
                    train_loss
                ),
                "validation_loss": (
                    scalar_loss
                ),
            }

            self.training_history.append(
                history
            )

            LOGGER.info(
                "G3 Residual epoch "
                "%03d/%03d | "
                "train_loss=%.8f | "
                "val_loss=%.8f",
                epoch,
                self.epochs,
                train_loss,
                scalar_loss,
            )

            epoch_train_losses.clear()

            return loss

        Metrics.apply = staticmethod(
            residual_metric_apply
        )

        # =====================================================
        # Model
        # =====================================================

        LOGGER.info(
            "G3 residual RealMLP fit: "
            "epochs=%d, ensemble=%d, "
            "batch_size=%d, lr=%.5f, "
            "n_threads=%d",
            self.epochs,
            self.config.group3_residual_ensemble,
            self.config.group3_residual_batch_size,
            self.config.group3_residual_learning_rate,
            n_threads,
        )

        self.model = (
            RealMLP_TD_Regressor(
                device=(
                    self.config.device
                    or "cpu"
                ),
                random_state=(
                    self.config.seed
                ),
                n_threads=(
                    n_threads
                ),
                n_epochs=(
                    self.epochs
                ),
                batch_size=(
                    self.config
                    .group3_residual_batch_size
                ),
                n_cv=1,
                n_refit=0,
                n_ens=(
                    self.config
                    .group3_residual_ensemble
                ),
                normalize_output=True,
                lr=(
                    self.config
                    .group3_residual_learning_rate
                ),
                lr_sched="coslog4",
                hidden_sizes=[
                    256,
                    256,
                    256,
                ],
                act="mish",
                p_drop=0.15,
                opt="adam",

                # Custom loss names so that
                # residual_metric_apply() receives
                # per-epoch train/validation calls.
                train_metric_name=(
                    _RESIDUAL_TRAIN_LOSS
                ),
                val_metric_name=(
                    _RESIDUAL_VAL_LOSS
                ),

                use_early_stopping=False,

                # When no explicit validation set
                # exists (final residual refit),
                # train exactly self.epochs.
                stop_epoch=(
                    None
                    if has_validation
                    else self.epochs
                ),

                # IMPORTANT:
                # Suppress PyTabKit feature dump.
                # Epoch loss is logged manually above.
                verbosity=0,
            )
        )

        try:
            self.model.fit(
                train_features,
                train_target,
                **fit_kwargs,
            )

        finally:
            # Never leave global PyTabKit metric
            # monkeypatch active after this model.
            Metrics.apply = staticmethod(
                original_apply
            )

        self.elapsed_seconds = (
            time.perf_counter()
            - started
        )

        LOGGER.info(
            "G3 residual RealMLP fit complete: "
            "elapsed=%.1fs",
            self.elapsed_seconds,
        )

        return self

    def predict_correction_cf(
        self,
        X: pd.DataFrame,
        base_prediction_kw: pd.Series,
    ) -> np.ndarray:

        if self.model is None:
            raise RuntimeError(
                "Group3ResidualRealMLP "
                "must be fitted first."
            )

        features = (
            self.build_features(
                X,
                base_prediction_kw,
            )
        )

        correction = np.asarray(
            self.model.predict(
                features
            ),
            dtype=float,
        ).reshape(-1)

        if len(correction) != len(X):
            raise ValueError(
                "G3 residual prediction length "
                f"mismatch: "
                f"{len(correction)} "
                f"!= {len(X)}"
            )

        if not np.isfinite(
            correction
        ).all():
            raise ValueError(
                "G3 residual prediction "
                "contains NaN or infinite values."
            )

        limit = float(
            self.config
            .group3_residual_max_abs_correction
        )

        return np.clip(
            correction,
            -limit,
            limit,
        )

    def correct_prediction_kw(
        self,
        X: pd.DataFrame,
        base_prediction_kw: pd.Series,
    ) -> pd.Series:

        base = (
            base_prediction_kw
            .reindex(X.index)
            .astype(float)
        )

        if base.isna().any():
            missing = int(
                base.isna().sum()
            )

            raise ValueError(
                "Base G3 prediction contains "
                f"{missing} missing rows "
                "during residual correction."
            )

        base_cf = (
            base.to_numpy(
                dtype=float
            )
            / CAPACITY_KWH[
                G3_TARGET
            ]
        )

        correction = (
            self.predict_correction_cf(
                X,
                base,
            )
        )

        corrected_cf = np.clip(
            base_cf
            + correction,
            0.0,
            1.0,
        )

        return pd.Series(
            corrected_cf
            * CAPACITY_KWH[
                G3_TARGET
            ],
            index=X.index,
            name=G3_TARGET,
        )

    def metadata(
        self,
    ) -> dict[str, Any]:

        return {
            "model": (
                "Group3ResidualRealMLP"
            ),
            "target": (
                G3_TARGET
            ),
            "epochs": (
                self.epochs
            ),
            "ensemble": (
                self.config
                .group3_residual_ensemble
            ),
            "learning_rate": (
                self.config
                .group3_residual_learning_rate
            ),
            "batch_size": (
                self.config
                .group3_residual_batch_size
            ),
            "max_abs_correction_cf": (
                self.config
                .group3_residual_max_abs_correction
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "target_definition": (
                "actual_cf "
                "- base_prediction_cf"
            ),
            "feature_definition": (
                "raw X + base G3 prediction "
                "+ cyclic calendar features"
            ),
            "epoch_loss_history": (
                self.training_history
            ),
        }