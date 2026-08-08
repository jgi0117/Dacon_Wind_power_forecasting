"""Group-conditioned long-format wrapper around the existing RealMLPModel.

The existing ``RealMLPModel`` is intentionally left unchanged.  This wrapper
converts the original wide three-target problem into a long single-target
problem:

    weather/time + group/turbine metadata -> one capacity-factor prediction

The same RealMLP weights are shared across all groups, so Group 1/2 data can
teach the common weather-to-power relationship while Group 3 provides its own
UNISON/turbine-specific conditioning signal.

Crucially, the wrapped RealMLP still uses its original ``n_ens=8`` internal
ensemble and all existing PyTabKit preprocessing/training settings.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import TARGET_COLS

from .base import RegressionModel
from .group_metadata import (
    categorical_metadata_columns,
    load_group_metadata,
)
from .realmlp_model import RealMLPModel


LOGGER = logging.getLogger("baram.pipeline")
_SINGLE_TARGET_NAME = "capacity_factor"
_SELF_HISTORY_PREFIX = "teacher_target__self__lag_"
_GROUP_HISTORY_PREFIX = "teacher_target__"


def _group_history_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(_GROUP_HISTORY_PREFIX)
        and "__lag_" in column
        and not column.startswith(_SELF_HISTORY_PREFIX)
    ]


def _history_columns_for_target(frame: pd.DataFrame, target: str) -> list[str]:
    prefix = f"teacher_target__{target}__lag_"
    return sorted(column for column in frame.columns if column.startswith(prefix))


class GroupConditionedRealMLPModel(RegressionModel):
    """Wide-interface model backed by one shared long-format RealMLP."""

    def __init__(
        self,
        config: PipelineConfig,
        epochs: int | None = None,
    ) -> None:
        # The legacy reliability weighting assumes three simultaneous output
        # columns. Long-format training has one target column, so disable it
        # while preserving every other RealMLP setting, including n_ens=8.
        self.config = replace(config, group3_reliability_weighting=False)
        self.epochs = int(epochs or config.max_epochs)
        self.group_metadata = load_group_metadata(
            Path(self.config.data_dir) / "info.xlsx"
        )
        self.model: RealMLPModel | None = None
        self.elapsed_seconds = 0.0
        self.best_iteration = self.epochs
        self._teacher_history_mode = False
        self._fit_rows_by_target: dict[str, int] = {}

    @staticmethod
    def _require_wide_targets(
        y: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame:
        if not isinstance(y, pd.DataFrame):
            raise TypeError(
                "Group-conditioned RealMLP requires a DataFrame with "
                f"{TARGET_COLS}."
            )
        missing = set(TARGET_COLS) - set(y.columns)
        if missing:
            raise ValueError(
                f"Missing group target columns: {sorted(missing)}"
            )
        return y.loc[:, TARGET_COLS]

    def _decorate_group_frame(
        self,
        X: pd.DataFrame,
        target: str,
        *,
        require_complete_history: bool,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Attach one group's metadata and select only that group's y-history."""
        frame = X.copy()

        meta = self.group_metadata.loc[target]
        for column, value in meta.items():
            frame[column] = value

        all_group_history = _group_history_columns(frame)
        target_history = _history_columns_for_target(frame, target)

        history_ok = np.ones(len(frame), dtype=bool)
        if target_history:
            for source in target_history:
                lag = source.rsplit("__lag_", 1)[1]
                frame[f"{_SELF_HISTORY_PREFIX}{lag}"] = frame[source]
            history_ok = frame[target_history].notna().all(axis=1).to_numpy()

        # Never expose other groups' target history to the current long row.
        if all_group_history:
            frame = frame.drop(columns=all_group_history)

        if require_complete_history and target_history:
            frame = frame.loc[history_ok]
            selected = history_ok
        else:
            selected = np.ones(len(X), dtype=bool)

        return frame, selected

    @staticmethod
    def _normalise_categorical_columns(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in categorical_metadata_columns():
            if column in result.columns:
                result[column] = result[column].astype("category")
        return result

    def _to_long_training(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        blocks_X: list[pd.DataFrame] = []
        blocks_y: list[pd.DataFrame] = []
        rows_by_target: dict[str, int] = {}

        teacher_mode = bool(_group_history_columns(X))
        self._teacher_history_mode = self._teacher_history_mode or teacher_mode

        for target in TARGET_COLS:
            decorated, history_selected = self._decorate_group_frame(
                X,
                target,
                require_complete_history=teacher_mode,
            )

            target_values = y[target]
            if teacher_mode:
                # Align target values after history filtering.
                target_values = target_values.loc[decorated.index]

            observed = target_values.notna().to_numpy()
            decorated = decorated.loc[observed]
            target_values = target_values.loc[observed]

            if decorated.empty:
                rows_by_target[target] = 0
                continue

            blocks_X.append(decorated)
            blocks_y.append(
                pd.DataFrame(
                    {_SINGLE_TARGET_NAME: target_values.astype(float)},
                    index=decorated.index,
                )
            )
            rows_by_target[target] = int(len(decorated))

        if not blocks_X:
            raise ValueError("No observed long-format training rows remain.")

        long_X = pd.concat(blocks_X, axis=0)
        long_y = pd.concat(blocks_y, axis=0)
        long_X = self._normalise_categorical_columns(long_X)

        if not long_X.index.equals(long_y.index):
            raise RuntimeError("Long-format X/y indices are misaligned.")

        self._fit_rows_by_target = rows_by_target
        return long_X, long_y

    def _to_long_prediction(
        self,
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        blocks: list[pd.DataFrame] = []
        history_masks: dict[str, np.ndarray] = {}
        teacher_mode = bool(_group_history_columns(X))

        for target in TARGET_COLS:
            decorated, _ = self._decorate_group_frame(
                X,
                target,
                require_complete_history=False,
            )
            target_history = _history_columns_for_target(X, target)
            if target_history:
                history_masks[target] = (
                    X[target_history].notna().all(axis=1).to_numpy()
                )
            else:
                history_masks[target] = np.ones(len(X), dtype=bool)
            blocks.append(decorated)

        long_X = pd.concat(blocks, axis=0)
        return self._normalise_categorical_columns(long_X), history_masks

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> "GroupConditionedRealMLPModel":
        targets = self._require_wide_targets(y)
        long_X, long_y = self._to_long_training(X, targets)

        long_X_valid = None
        long_y_valid = None
        if X_valid is not None or y_valid is not None:
            if X_valid is None or y_valid is None:
                raise ValueError(
                    "X_valid and y_valid must be provided together."
                )
            valid_targets = self._require_wide_targets(y_valid)
            long_X_valid, long_y_valid = self._to_long_training(
                X_valid, valid_targets
            )

        LOGGER.info(
            "Group-conditioned RealMLP long fit: rows=%d, features=%d, "
            "rows_by_target=%s",
            len(long_X),
            long_X.shape[1],
            self._fit_rows_by_target,
        )

        self.model = RealMLPModel(self.config, epochs=self.epochs)
        self.model.fit(
            long_X,
            long_y,
            long_X_valid,
            long_y_valid,
        )
        self.elapsed_seconds = self.model.elapsed_seconds
        self.best_iteration = self.model.best_iteration
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "Group-conditioned RealMLP must be fitted before predict()."
            )

        long_X, history_masks = self._to_long_prediction(X)
        long_prediction = np.asarray(
            self.model.predict(long_X),
            dtype=float,
        ).reshape(-1)

        n_rows = len(X)
        expected = n_rows * len(TARGET_COLS)
        if len(long_prediction) != expected:
            raise ValueError(
                f"Long prediction length mismatch: "
                f"{len(long_prediction)} != {expected}"
            )

        wide = np.empty((n_rows, len(TARGET_COLS)), dtype=float)
        for group_index, target in enumerate(TARGET_COLS):
            start = group_index * n_rows
            end = start + n_rows
            wide[:, group_index] = long_prediction[start:end]

            # A privileged teacher must not emit a soft label for a group when
            # that group's own 12-hour target history is incomplete.
            if self._teacher_history_mode or bool(_group_history_columns(X)):
                mask = history_masks[target]
                wide[~mask, group_index] = np.nan

        return wide

    def metadata(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("metadata() requires a fitted model.")

        inner = self.model.metadata()
        return {
            **inner,
            "model": "GroupConditioned-" + str(inner.get("model", "RealMLP-TD")),
            "architecture": (
                "wide-to-long shared RealMLP; "
                "weather/time + group/turbine metadata -> one target"
            ),
            "group_conditioning": True,
            "long_single_target": True,
            "shared_across_groups": True,
            "internal_ensemble": inner.get("n_ens", 8),
            "best_iteration": self.best_iteration,
            "fit_rows_by_target": self._fit_rows_by_target,
            "teacher_history_mode": self._teacher_history_mode,
            "group_metadata": self.group_metadata.reset_index().to_dict(
                orient="records"
            ),
        }
