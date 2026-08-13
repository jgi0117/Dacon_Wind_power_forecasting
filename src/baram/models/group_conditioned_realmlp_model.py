"""Group-conditioned long-format wrapper around the existing RealMLPModel.

The existing ``RealMLPModel`` is intentionally left unchanged. This wrapper
converts the original wide three-target problem into a long single-target
problem:

    weather/time + group/turbine metadata -> one capacity-factor prediction

The same RealMLP weights are shared across all groups.

For privileged Teacher inference, only rows whose SAME-GROUP target history is
complete are sent into PyTabKit. This is important because PyTabKit RealMLP
does not allow NaN values in continuous columns. Groups whose own history is
incomplete simply receive NaN predictions, which are later ignored by the
distillation logic.
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
    numeric_metadata_columns,
    load_group_metadata,
)
from .realmlp_model import RealMLPModel


LOGGER = logging.getLogger("baram.pipeline")

_SINGLE_TARGET_NAME = "capacity_factor"
_SELF_HISTORY_PREFIX = "teacher_target__self__lag_"
_GROUP_HISTORY_PREFIX = "teacher_target__"
_LOSS_GROUP_CODE_COLUMN = "__baram_loss_group_code"


def _group_history_columns(frame: pd.DataFrame) -> list[str]:
    """Return privileged target-history source columns.

    Example:
        teacher_target__kpx_group_1__lag_01
    """
    return [
        column
        for column in frame.columns
        if column.startswith(_GROUP_HISTORY_PREFIX)
        and "__lag_" in column
        and not column.startswith(_SELF_HISTORY_PREFIX)
    ]


def _history_columns_for_target(
    frame: pd.DataFrame,
    target: str,
) -> list[str]:
    prefix = f"teacher_target__{target}__lag_"
    return sorted(
        column
        for column in frame.columns
        if column.startswith(prefix)
    )


class GroupConditionedRealMLPModel(RegressionModel):
    """Wide-interface model backed by one shared long-format RealMLP."""

    def __init__(
        self,
        config: PipelineConfig,
        epochs: int | None = None,
    ) -> None:
        # Legacy reliability weighting assumes three simultaneous output
        # columns. Long-format training has one target column.
        self.config = replace(
            config,
            group3_reliability_weighting=False,
        )

        self.epochs = int(
            epochs or config.max_epochs
        )

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
                "Group-conditioned RealMLP requires a "
                f"DataFrame with {TARGET_COLS}."
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
        """Attach group metadata and keep only the group's own target history.

        Returns
        -------
        frame:
            Decorated group-specific frame.

        selected:
            Boolean mask aligned to the original ``X`` rows. True means the
            row is present in ``frame``.
        """
        frame = X.copy()

        meta = self.group_metadata.loc[target]

        for column, value in meta.items():
            frame[column] = value

        all_group_history = _group_history_columns(
            frame
        )

        target_history = _history_columns_for_target(
            frame,
            target,
        )

        history_ok = np.ones(
            len(frame),
            dtype=bool,
        )

        if target_history:
            # Convert group-specific target history into one generic
            # "self" history representation.
            for source in target_history:
                lag = source.rsplit(
                    "__lag_",
                    1,
                )[1]

                frame[
                    f"{_SELF_HISTORY_PREFIX}{lag}"
                ] = frame[source]

            history_ok = (
                frame[target_history]
                .notna()
                .all(axis=1)
                .to_numpy()
            )

        # Do not expose any other group's target history to this group.
        if all_group_history:
            frame = frame.drop(
                columns=all_group_history
            )

        if (
            require_complete_history
            and target_history
        ):
            frame = frame.loc[history_ok]
            selected = history_ok
        else:
            selected = np.ones(
                len(X),
                dtype=bool,
            )

        return frame, selected

    @staticmethod
    def _normalise_columns(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare metadata columns for PyTabKit.

        Categorical metadata is cast to pandas category.

        RealMLP does not accept NaN in continuous columns. Fully missing
        optional numeric metadata columns are therefore removed; partially
        missing numeric metadata is median-imputed.
        """
        result = frame.copy()

        for column in categorical_metadata_columns():
            if column not in result.columns:
                continue

            result[column] = (
                result[column]
                .astype("string")
                .fillna("UNKNOWN")
                .astype("category")
            )

        for column in numeric_metadata_columns():
            if column not in result.columns:
                continue

            values = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            if values.isna().all():
                result = result.drop(
                    columns=[column]
                )
                continue

            if values.isna().any():
                values = values.fillna(
                    values.median()
                )

            result[column] = values.astype(float)

        return result

    @staticmethod
    def _assert_no_continuous_nan(
        frame: pd.DataFrame,
        *,
        context: str,
    ) -> None:
        """Fail with useful column names before entering PyTabKit."""
        continuous = frame.select_dtypes(
            exclude=[
                "category",
                "object",
                "string",
            ]
        )

        if continuous.empty:
            return

        counts = (
            continuous.isna()
            .sum()
        )

        counts = counts[
            counts > 0
        ]

        if not counts.empty:
            raise ValueError(
                f"NaN continuous columns before RealMLP "
                f"{context}:\n{counts.to_string()}"
            )

    def _to_long_training(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        blocks_X: list[pd.DataFrame] = []
        blocks_y: list[pd.DataFrame] = []

        rows_by_target: dict[str, int] = {}

        teacher_mode = bool(
            _group_history_columns(X)
        )

        self._teacher_history_mode = (
            self._teacher_history_mode
            or teacher_mode
        )

        for target in TARGET_COLS:
            decorated, _ = (
                self._decorate_group_frame(
                    X,
                    target,
                    require_complete_history=teacher_mode,
                )
            )

            target_values = y[target]

            if teacher_mode:
                # Align y after group-history filtering.
                target_values = target_values.loc[
                    decorated.index
                ]

            observed = (
                target_values
                .notna()
                .to_numpy()
            )

            decorated = decorated.loc[
                observed
            ]

            target_values = target_values.loc[
                observed
            ]

            if decorated.empty:
                rows_by_target[target] = 0
                continue

            # Loss-only marker. RealMLPModel removes this column before
            # PyTabKit sees the features; it is used only to reconstruct
            # kpx_group_1/2/3 inside the long-format FICR objective.
            decorated = decorated.copy()
            decorated[_LOSS_GROUP_CODE_COLUMN] = float(
                TARGET_COLS.index(target)
            )

            blocks_X.append(
                decorated
            )

            blocks_y.append(
                pd.DataFrame(
                    {
                        _SINGLE_TARGET_NAME:
                            target_values.astype(float)
                    },
                    index=decorated.index,
                )
            )

            rows_by_target[target] = int(
                len(decorated)
            )

        if not blocks_X:
            raise ValueError(
                "No observed long-format training "
                "rows remain."
            )

        long_X = pd.concat(
            blocks_X,
            axis=0,
        )

        long_y = pd.concat(
            blocks_y,
            axis=0,
        )

        long_X = self._normalise_columns(
            long_X
        )

        if not long_X.index.equals(
            long_y.index
        ):
            raise RuntimeError(
                "Long-format X/y indices are misaligned."
            )

        self._assert_no_continuous_nan(
            long_X,
            context="fit",
        )

        self._fit_rows_by_target = (
            rows_by_target
        )

        return long_X, long_y

    def _to_long_prediction(
        self,
        X: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        dict[str, np.ndarray],
        dict[str, int],
    ]:
        """Create RealMLP prediction rows.

        For a direct Student, every group/timestamp row is included.

        For a privileged Teacher, only SAME-GROUP complete-history rows are
        included. This prevents target-history NaNs from ever reaching
        PyTabKit.

        Returns
        -------
        long_X:
            Concatenated prediction rows actually sent to RealMLP.

        selected_masks:
            Original-row masks for each target.

        block_sizes:
            Number of prediction rows for each target, used to reconstruct the
            wide result.
        """
        blocks: list[pd.DataFrame] = []

        selected_masks: dict[
            str,
            np.ndarray,
        ] = {}

        block_sizes: dict[
            str,
            int,
        ] = {}

        teacher_mode = bool(
            _group_history_columns(X)
        )

        for target in TARGET_COLS:
            decorated, selected = (
                self._decorate_group_frame(
                    X,
                    target,
                    require_complete_history=teacher_mode,
                )
            )

            selected_masks[target] = selected
            block_sizes[target] = int(
                len(decorated)
            )

            if not decorated.empty:
                blocks.append(
                    decorated
                )

        if not blocks:
            # All groups may legitimately be unavailable at a privileged
            # timestamp. predict() will return an all-NaN wide array.
            empty = pd.DataFrame()
            return (
                empty,
                selected_masks,
                block_sizes,
            )

        long_X = pd.concat(
            blocks,
            axis=0,
        )

        long_X = self._normalise_columns(
            long_X
        )

        self._assert_no_continuous_nan(
            long_X,
            context="predict",
        )

        return (
            long_X,
            selected_masks,
            block_sizes,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> "GroupConditionedRealMLPModel":
        targets = self._require_wide_targets(
            y
        )

        long_X, long_y = (
            self._to_long_training(
                X,
                targets,
            )
        )

        long_X_valid = None
        long_y_valid = None

        if (
            X_valid is not None
            or y_valid is not None
        ):
            if (
                X_valid is None
                or y_valid is None
            ):
                raise ValueError(
                    "X_valid and y_valid must be "
                    "provided together."
                )

            valid_targets = (
                self._require_wide_targets(
                    y_valid
                )
            )

            long_X_valid, long_y_valid = (
                self._to_long_training(
                    X_valid,
                    valid_targets,
                )
            )

        LOGGER.info(
            "Group-conditioned RealMLP long fit: "
            "rows=%d, features=%d, "
            "rows_by_target=%s",
            len(long_X),
            long_X.shape[1],
            self._fit_rows_by_target,
        )

        self.model = RealMLPModel(
            self.config,
            epochs=self.epochs,
        )

        self.model.fit(
            long_X,
            long_y,
            long_X_valid,
            long_y_valid,
        )

        self.elapsed_seconds = (
            self.model.elapsed_seconds
        )

        self.best_iteration = (
            self.model.best_iteration
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "Group-conditioned RealMLP must "
                "be fitted before predict()."
            )

        (
            long_X,
            selected_masks,
            block_sizes,
        ) = self._to_long_prediction(X)

        n_rows = len(X)

        # Initialize all cells as unavailable. Direct Student inference will
        # fill every cell; privileged Teacher inference fills only groups with
        # complete own history.
        wide = np.full(
            (
                n_rows,
                len(TARGET_COLS),
            ),
            np.nan,
            dtype=float,
        )

        if long_X.empty:
            return wide

        long_prediction = np.asarray(
            self.model.predict(long_X),
            dtype=float,
        ).reshape(-1)

        expected = sum(
            block_sizes.values()
        )

        if len(long_prediction) != expected:
            raise ValueError(
                "Long prediction length mismatch: "
                f"{len(long_prediction)} != {expected}"
            )

        cursor = 0

        for group_index, target in enumerate(
            TARGET_COLS
        ):
            size = block_sizes[target]
            selected = selected_masks[target]

            if size == 0:
                continue

            block_prediction = (
                long_prediction[
                    cursor:cursor + size
                ]
            )

            cursor += size

            selected_count = int(
                np.count_nonzero(selected)
            )

            if selected_count != size:
                raise RuntimeError(
                    f"{target}: prediction selection "
                    f"mismatch: mask={selected_count}, "
                    f"block={size}"
                )

            wide[
                selected,
                group_index,
            ] = block_prediction

        if cursor != len(long_prediction):
            raise RuntimeError(
                "Not all long predictions were consumed: "
                f"{cursor} / {len(long_prediction)}"
            )

        return wide

    def metadata(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(
                "metadata() requires a fitted model."
            )

        inner = self.model.metadata()

        return {
            **inner,
            "model": (
                "GroupConditioned-"
                + str(
                    inner.get(
                        "model",
                        "RealMLP-TD",
                    )
                )
            ),
            "architecture": (
                "wide-to-long shared RealMLP; "
                "weather/time + group/turbine metadata "
                "-> one target"
            ),
            "group_conditioning": True,
            "long_single_target": True,
            "shared_across_groups": True,
            "internal_ensemble": inner.get(
                "n_ens",
                8,
            ),
            "best_iteration": (
                self.best_iteration
            ),
            "fit_rows_by_target": (
                self._fit_rows_by_target
            ),
            "teacher_history_mode": (
                self._teacher_history_mode
            ),
            "teacher_prediction_history_filter": (
                "same-group-complete-history-before-pytabkit"
            ),
            "group_metadata": (
                self.group_metadata
                .reset_index()
                .to_dict(
                    orient="records"
                )
            ),
        }