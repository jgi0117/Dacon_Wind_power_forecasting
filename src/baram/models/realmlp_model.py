'''PyTabKit shared-trunk multi-output RealMLP for Version 4.'''

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import (
    TARGET_COLS,
    activity_loss_torch,
    ficr_boundary_consistency_loss_torch,
    ficr_aware_loss_torch,
    long_group_competition_components_torch,
    long_group_ficr_aware_loss_torch,
    relu_ficr_aware_loss_torch,
    temporal_group_dro_ficr_loss_torch,
)
from baram.reliability import group3_cross_fitted_reliability
from .base import RegressionModel


LOGGER = logging.getLogger('baram.pipeline')
_TRAIN_FICR_LOSS = 'baram_train_ficr_aware_loss'
_VAL_FICR_LOSS = 'baram_val_ficr_aware_loss'
_MISSING_TARGET = -1.0
_ACTIVITY_CODES_PER_BLOCK = 3
_LOSS_GROUP_CODE_COLUMN = '__baram_loss_group_code'
_LOSS_SELECTION_PERIOD_COLUMN = '__baram_loss_selection_period'
_ACTIVITY_CODES_PER_GROUP = 3
_ACTIVITY_CODES_PER_SELECTION_PERIOD = 9
_REALMLP_TD_REG_PARAMS = {
    'hidden_sizes': [256] * 3,
    'max_one_hot_cat_size': 9,
    'embedding_size': 8,
    'weight_param': 'ntk',
    'weight_init_mode': 'std',
    'bias_init_mode': 'he+5',
    'bias_lr_factor': 0.1,
    'act': 'mish',
    'use_parametric_act': True,
    'act_lr_factor': 0.1,
    'wd': 2e-2,
    'wd_sched': 'flat_cos',
    'bias_wd_factor': 0.0,
    'block_str': 'w-b-a-d',
    'p_drop': 0.15,
    'p_drop_sched': 'flat_cos',
    'add_front_scale': True,
    'scale_lr_factor': 6.0,
    'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
    'num_emb_type': 'pbld',
    'plr_sigma': 0.1,
    'plr_hidden_1': 16,
    'plr_hidden_2': 4,
    'plr_lr_factor': 0.1,
    # NaN-masked multi-task targets make data-derived clamp bounds invalid.
    # Predictions are clipped to [0, 1] when restored to generation units.
    'clamp_output': False,
    'opt': 'adam',
    'sq_mom': 0.95,
}


def _quarter_block_ids(index: pd.Index):
    '''Return compact chronological quarter IDs for loss-only metadata.'''
    timestamps = pd.DatetimeIndex(index)
    quarter_keys = timestamps.year.astype(np.int64) * 4 + timestamps.quarter - 1
    unique_keys = sorted(np.unique(quarter_keys).tolist())
    key_to_id = {key: block_id for block_id, key in enumerate(unique_keys)}
    ids = np.asarray(
        [key_to_id[int(key)] for key in quarter_keys], dtype=np.float32
    )
    labels = {
        block_id: f'{key // 4}Q{key % 4 + 1}'
        for key, block_id in key_to_id.items()
    }
    return ids, labels


def _pack_activity_block_metadata(activity, block_ids):
    '''Pack block ID into the first activity label without adding an output.'''
    packed = np.array(activity, dtype=np.float32, copy=True)
    first_code = np.where(
        packed[:, 0] < 0.0, 2.0, packed[:, 0]
    ).astype(np.float32)
    packed[:, 0] = _ACTIVITY_CODES_PER_BLOCK * block_ids + first_code
    return packed


def _unpack_activity_block_metadata(packed):
    '''Decode original activity labels and temporal IDs inside the loss.'''
    import torch

    encoded = packed[..., 0]
    block_ids = torch.div(
        encoded, _ACTIVITY_CODES_PER_BLOCK, rounding_mode='floor'
    )
    first_code = torch.remainder(encoded, _ACTIVITY_CODES_PER_BLOCK)
    activity = packed.clone()
    activity[..., 0] = torch.where(
        first_code == 2.0,
        torch.full_like(first_code, _MISSING_TARGET),
        first_code,
    )
    return activity, block_ids


def _pack_activity_group_metadata(activity, group_codes):
    """Pack long-format G1/G2/G3 row identity into activity labels."""
    packed = np.array(activity, dtype=np.float32, copy=True)
    if packed.ndim != 2 or packed.shape[1] != 1:
        raise ValueError(
            "Group loss metadata requires one long-format target column."
        )
    codes = np.asarray(group_codes, dtype=np.float32).reshape(-1)
    if len(codes) != len(packed):
        raise ValueError(
            f"Group-code length mismatch: {len(codes)} != {len(packed)}"
        )
    if not np.isin(codes, [0.0, 1.0, 2.0]).all():
        raise ValueError("Group loss codes must be 0, 1, or 2.")
    activity_code = np.where(
        packed[:, 0] < 0.0, 2.0, packed[:, 0]
    ).astype(np.float32)
    packed[:, 0] = _ACTIVITY_CODES_PER_GROUP * codes + activity_code
    return packed


def _unpack_activity_group_metadata(packed):
    """Decode activity labels and long-format group IDs inside the loss."""
    import torch

    encoded = packed[..., 0]
    group_ids = torch.div(
        encoded, _ACTIVITY_CODES_PER_GROUP, rounding_mode='floor'
    ).to(dtype=torch.long)
    activity_code = torch.remainder(encoded, _ACTIVITY_CODES_PER_GROUP)
    activity = packed.clone()
    activity[..., 0] = torch.where(
        activity_code == 2.0,
        torch.full_like(activity_code, _MISSING_TARGET),
        activity_code,
    )
    return activity, group_ids



def _pack_activity_selection_period_metadata(activity, period_codes):
    """Pack Student-selection chronological period IDs after group metadata."""
    packed = np.array(activity, dtype=np.float32, copy=True)
    if packed.ndim != 2 or packed.shape[1] != 1:
        raise ValueError(
            "Selection-period metadata requires one long-format target column."
        )
    codes = np.asarray(period_codes, dtype=np.float32).reshape(-1)
    if len(codes) != len(packed):
        raise ValueError(
            f"Selection-period length mismatch: {len(codes)} != {len(packed)}"
        )
    if (codes < 0).any():
        raise ValueError("Selection-period codes must be non-negative.")
    packed[:, 0] = (
        _ACTIVITY_CODES_PER_SELECTION_PERIOD * codes + packed[:, 0]
    )
    return packed


def _unpack_activity_selection_period_metadata(packed):
    """Decode Student-selection period IDs before group/activity metadata."""
    import torch

    encoded = packed[..., 0]
    period_ids = torch.div(
        encoded,
        _ACTIVITY_CODES_PER_SELECTION_PERIOD,
        rounding_mode='floor',
    ).to(dtype=torch.long)
    remainder = torch.remainder(
        encoded, _ACTIVITY_CODES_PER_SELECTION_PERIOD
    )
    activity = packed.clone()
    activity[..., 0] = remainder
    return activity, period_ids

def _pack_reliability_metadata(activity, reliability):
    '''Pack continuous per-target weights into activity target fractions.'''
    integer_codes = np.where(activity < 0.0, 2.0, activity).astype(np.float32)
    return integer_codes + 0.1 * np.asarray(reliability, dtype=np.float32)


def _unpack_reliability_metadata(packed):
    '''Recover activity codes and continuous reliability weights in the loss.'''
    import torch

    integer_codes = torch.floor(packed + 1e-5)
    reliability = ((packed - integer_codes) * 10.0).clamp(0.0, 1.0)
    return integer_codes, reliability


def _competition_score_loss(y_pred: Any, y: Any) -> Any:
    import torch

    prediction = y_pred
    actual = y
    if actual.ndim < prediction.ndim:
        actual = actual.unsqueeze(0).expand_as(prediction)
    elif prediction.ndim < actual.ndim:
        prediction = prediction.unsqueeze(0).expand_as(actual)
    if actual.ndim < 2:
        actual = actual.reshape(-1, 1)
        prediction = prediction.reshape(-1, 1)
    valid = torch.isfinite(actual) & torch.isfinite(prediction) & (actual >= 0.10)
    safe_actual = torch.where(valid, actual, torch.zeros_like(actual))
    safe_prediction = torch.where(
        valid, prediction.clamp(0.0, 1.0), torch.zeros_like(prediction)
    )
    error = (safe_prediction - safe_actual).abs()
    valid_float = valid.to(error.dtype)
    count = valid_float.sum(dim=-2)
    safe_count = count.clamp_min(1.0)
    nmae = (error * valid_float).sum(dim=-2) / safe_count
    unit_price = torch.where(
        error <= 0.06, 4.0, torch.where(error <= 0.08, 3.0, 0.0)
    )
    numerator = (safe_actual * unit_price * valid_float).sum(dim=-2)
    denominator = (safe_actual * 4.0 * valid_float).sum(dim=-2)
    ficr = numerator / denominator.clamp_min(1e-12)
    group_score = 0.5 * (1.0 - nmae) + 0.5 * ficr
    valid_groups = count > 0
    valid_group_count = valid_groups.sum(dim=-1).clamp_min(1)
    score = (
        torch.where(valid_groups, group_score, torch.zeros_like(group_score))
        .sum(dim=-1)
        / valid_group_count
    )
    return -score


def _competition_components(y_pred: Any, y: Any) -> tuple[Any, Any]:
    '''Return exact group-averaged NMAE and FICR tensors.'''
    import torch

    prediction = y_pred
    actual = y
    if actual.ndim < prediction.ndim:
        actual = actual.unsqueeze(0).expand_as(prediction)
    elif prediction.ndim < actual.ndim:
        prediction = prediction.unsqueeze(0).expand_as(actual)
    valid = torch.isfinite(actual) & torch.isfinite(prediction) & (actual >= 0.10)
    safe_actual = torch.where(valid, actual, torch.zeros_like(actual))
    safe_prediction = torch.where(
        valid, prediction.clamp(0.0, 1.0), torch.zeros_like(prediction)
    )
    error = (safe_prediction - safe_actual).abs()
    valid_float = valid.to(error.dtype)
    count = valid_float.sum(dim=-2)
    nmae = (error * valid_float).sum(dim=-2) / count.clamp_min(1.0)
    unit_price = torch.where(
        error <= 0.06, 4.0, torch.where(error <= 0.08, 3.0, 0.0)
    )
    ficr = (safe_actual * unit_price * valid_float).sum(dim=-2) / (
        safe_actual * 4.0 * valid_float
    ).sum(dim=-2).clamp_min(1e-12)
    valid_groups = count > 0
    divisor = valid_groups.sum(dim=-1).clamp_min(1)
    mean_nmae = torch.where(valid_groups, nmae, 0.0).sum(dim=-1) / divisor
    mean_ficr = torch.where(valid_groups, ficr, 0.0).sum(dim=-1) / divisor
    return mean_nmae, mean_ficr


class RealMLPModel(RegressionModel):
    def __init__(self, config: PipelineConfig, epochs: int | None = None) -> None:
        self.config = config
        self.epochs = int(epochs or config.max_epochs)
        self.best_iteration = self.epochs
        self.model: Any | None = None
        self.elapsed_seconds = 0.0
        self.device = 'cpu'
        self.n_threads = 1
        self.early_stopping_enabled = False
        self.training_history: list[dict[str, Any]] = []
        self.target_names = list(TARGET_COLS)
        self.temporal_block_weights: dict[str, float] = {}
        self.group3_reliability_metadata: dict[str, Any] = {}

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> 'RealMLPModel':
        from pytabkit import RealMLP_TD_Regressor
        from pytabkit.models.training.metrics import Metrics

        started = time.perf_counter()
        self.device = self.config.device or 'cpu'
        self.n_threads = (
            os.cpu_count() or 1
            if self.config.n_jobs < 1
            else self.config.n_jobs
        )
        if self.n_threads < 1:
            self.n_threads = 1

        has_validation = X_valid is not None and y_valid is not None

        X = X.copy()
        train_group_codes: np.ndarray | None = None
        valid_group_codes: np.ndarray | None = None
        valid_selection_period_codes: np.ndarray | None = None

        if _LOSS_SELECTION_PERIOD_COLUMN in X.columns:
            # Selection-period metadata is validation-only by design.
            X.pop(_LOSS_SELECTION_PERIOD_COLUMN)

        if _LOSS_GROUP_CODE_COLUMN in X.columns:
            train_group_codes = pd.to_numeric(
                X.pop(_LOSS_GROUP_CODE_COLUMN), errors='raise'
            ).to_numpy(dtype=np.int64)
            if self.config.temporal_group_dro:
                raise ValueError(
                    "Long-format worst-group FICR cannot be combined with "
                    "temporal_group_dro."
                )
            if self.config.group3_reliability_weighting:
                raise ValueError(
                    "Long-format worst-group FICR cannot be combined with "
                    "group3_reliability_weighting."
                )

        if X_valid is not None:
            X_valid = X_valid.copy()
            if _LOSS_SELECTION_PERIOD_COLUMN in X_valid.columns:
                valid_selection_period_codes = pd.to_numeric(
                    X_valid.pop(_LOSS_SELECTION_PERIOD_COLUMN), errors='raise'
                ).to_numpy(dtype=np.int64)
            if _LOSS_GROUP_CODE_COLUMN in X_valid.columns:
                valid_group_codes = pd.to_numeric(
                    X_valid.pop(_LOSS_GROUP_CODE_COLUMN), errors='raise'
                ).to_numpy(dtype=np.int64)

        if has_validation and (
            (train_group_codes is None) != (valid_group_codes is None)
        ):
            raise ValueError(
                "Train/validation group-loss markers must be provided together."
            )

        original_apply = Metrics.apply
        epoch_train_losses: list[float] = []
        epoch_objective_train_losses: list[float] = []
        epoch_activity_train_losses: list[float] = []
        epoch_boundary_consistency_losses: list[float] = []
        epoch_group_train_losses: dict[str, list[float]] = {}
        if isinstance(y, pd.DataFrame):
            self.target_names = [str(column) for column in y.columns]
        else:
            self.target_names = [str(y.name or TARGET_COLS[0])]
        epoch_group_train_losses = {target: [] for target in self.target_names}
        block_ids, temporal_block_labels = _quarter_block_ids(X.index)
        dro_log_weights = {key: 0.0 for key in temporal_block_labels}
        epoch_temporal_losses = {
            key: [] for key in temporal_block_labels
        }

        def normalized_dro_weights() -> dict[int, float]:
            maximum = max(dro_log_weights.values())
            unscaled = {
                key: float(np.exp(value - maximum))
                for key, value in dro_log_weights.items()
            }
            total = sum(unscaled.values())
            return {key: value / total for key, value in unscaled.items()}

        def capacity_objective(
            actual: Any,
            prediction: Any,
            reliability: Any | None = None,
            group_ids: Any | None = None,
        ) -> Any:
            if group_ids is not None:
                if self.config.ficr_loss != 'sigmoid':
                    raise ValueError(
                        "Long-format worst-group FICR currently requires "
                        "--ficr-loss sigmoid."
                    )
                return long_group_ficr_aware_loss_torch(
                    actual, prediction, group_ids,
                    ficr_weight=self.config.ficr_weight,
                    temperature=self.config.ficr_temperature,
                    n_groups=len(TARGET_COLS),
                )
            if self.config.ficr_loss == 'relu':
                return relu_ficr_aware_loss_torch(
                    actual, prediction,
                    ficr_weight=self.config.ficr_weight,
                    margin=self.config.ficr_relu_margin,
                )
            return ficr_aware_loss_torch(
                actual, prediction,
                ficr_weight=self.config.ficr_weight,
                temperature=self.config.ficr_temperature,
                sample_weight=reliability,
            )

        def score_aware_apply(y_pred: Any, actual: Any, metric_name: str) -> Any:
            import torch

            if metric_name in {_TRAIN_FICR_LOSS, _VAL_FICR_LOSS}:
                n_targets = len(self.target_names)
                actual_capacity = actual[..., :n_targets]
                predicted_capacity = y_pred[..., :n_targets]
                actual_activity = actual[..., n_targets:2 * n_targets]
                predicted_activity = y_pred[..., n_targets:2 * n_targets]
                selection_period_ids = None
                if (
                    metric_name == _VAL_FICR_LOSS
                    and valid_selection_period_codes is not None
                ):
                    actual_activity, selection_period_ids = (
                        _unpack_activity_selection_period_metadata(actual_activity)
                    )
                loss_group_ids = None
                if train_group_codes is not None:
                    actual_activity, loss_group_ids = (
                        _unpack_activity_group_metadata(actual_activity)
                    )
                reliability = None
                if self.config.group3_reliability_weighting:
                    actual_activity, reliability = (
                        _unpack_reliability_metadata(actual_activity)
                    )
                actual_blocks = None
                if self.config.temporal_group_dro:
                    actual_activity, actual_blocks = (
                        _unpack_activity_block_metadata(actual_activity)
                    )
                actual_activity = torch.where(
                    actual_activity == 2.0,
                    torch.full_like(actual_activity, _MISSING_TARGET),
                    actual_activity,
                )
                temporal_losses = {}
                use_dro = (
                    metric_name == _TRAIN_FICR_LOSS
                    and self.config.temporal_group_dro
                    and self.config.ficr_loss == 'sigmoid'
                )
                if use_dro:
                    capacity_loss, temporal_losses = (
                        temporal_group_dro_ficr_loss_torch(
                            actual_capacity,
                            predicted_capacity,
                            actual_blocks,
                            normalized_dro_weights(),
                            ficr_weight=self.config.ficr_weight,
                            temperature=self.config.ficr_temperature,
                        )
                    )
                else:
                    capacity_loss = capacity_objective(
                        actual_capacity, predicted_capacity, reliability, loss_group_ids
                    )
                activity_loss = activity_loss_torch(
                    actual_activity, predicted_activity
                )
                boundary_consistency_loss = capacity_loss * 0.0
                if (
                    metric_name == _TRAIN_FICR_LOSS
                    and self.config.ficr_boundary_consistency_weight > 0.0
                ):
                    boundary_consistency_loss = (
                        ficr_boundary_consistency_loss_torch(
                            actual_capacity,
                            predicted_capacity,
                            temperature=self.config.ficr_temperature,
                        )
                    )
                value = (
                    capacity_loss
                    + self.config.activity_loss_weight * activity_loss
                    + self.config.ficr_boundary_consistency_weight
                    * boundary_consistency_loss
                    if metric_name == _TRAIN_FICR_LOSS
                    else capacity_loss
                )
                mean_loss = float(value.detach().mean().cpu())
                mean_capacity_loss = float(capacity_loss.detach().mean().cpu())
                mean_activity_loss = float(activity_loss.detach().mean().cpu())
                mean_boundary_loss = float(
                    boundary_consistency_loss.detach().mean().cpu()
                )
                if loss_group_ids is None:
                    group_losses = {
                        target: float(
                            capacity_objective(
                                actual_capacity[..., index:index + 1],
                                predicted_capacity[..., index:index + 1],
                                None if reliability is None else reliability[
                                    ..., index:index + 1
                                ],
                            ).detach().mean().cpu()
                        )
                        for index, target in enumerate(self.target_names)
                    }
                    long_group_components = None
                else:
                    long_group_components = long_group_competition_components_torch(
                        actual_capacity, predicted_capacity, loss_group_ids,
                        n_groups=len(TARGET_COLS),
                    )
                    group_losses = {}
                    for group_code, target in enumerate(TARGET_COLS):
                        group_nmae, group_ficr = long_group_components[group_code]
                        group_losses[target] = float(
                            ((1.0 - self.config.ficr_weight) * group_nmae
                             + self.config.ficr_weight * (1.0 - group_ficr))
                            .detach().mean().cpu()
                        )
                if metric_name == _TRAIN_FICR_LOSS:
                    epoch_train_losses.append(mean_capacity_loss)
                    epoch_objective_train_losses.append(mean_loss)
                    epoch_activity_train_losses.append(mean_activity_loss)
                    epoch_boundary_consistency_losses.append(
                        mean_boundary_loss
                    )
                    for target, loss in group_losses.items():
                        if target in epoch_group_train_losses:
                            epoch_group_train_losses[target].append(loss)
                    for block_id, block_loss in temporal_losses.items():
                        scalar_loss = float(block_loss.detach().mean().cpu())
                        epoch_temporal_losses[block_id].append(scalar_loss)
                    return value
                if loss_group_ids is None:
                    score_value = _competition_score_loss(
                        predicted_capacity, actual_capacity
                    )
                    validation_nmae, validation_ficr = _competition_components(
                        predicted_capacity, actual_capacity
                    )
                else:
                    assert long_group_components is not None
                    group_nmaes = torch.stack(
                        [long_group_components[i][0] for i in range(len(TARGET_COLS))],
                        dim=-1,
                    )
                    group_ficrs = torch.stack(
                        [long_group_components[i][1] for i in range(len(TARGET_COLS))],
                        dim=-1,
                    )
                    validation_nmae = group_nmaes.mean(dim=-1)
                    validation_ficr = group_ficrs.mean(dim=-1)
                    score_value = -(
                        0.5 * (1.0 - validation_nmae) + 0.5 * validation_ficr
                    )
                history: dict[str, Any] = {
                    'step': len(self.training_history) + 1,
                    'train_loss': (
                        float(np.mean(epoch_train_losses))
                        if epoch_train_losses else None
                    ),
                    'validation_loss': mean_loss,
                    'validation_score': -float(
                        score_value.detach().mean().cpu()
                    ),
                    'validation_nmae': float(
                        validation_nmae.detach().mean().cpu()
                    ),
                    'validation_ficr': float(
                        validation_ficr.detach().mean().cpu()
                    ),
                    'training_objective_loss': (
                        float(np.mean(epoch_objective_train_losses))
                        if epoch_objective_train_losses else None
                    ),
                    'activity_train_loss': (
                        float(np.mean(epoch_activity_train_losses))
                        if epoch_activity_train_losses else None
                    ),
                    'activity_validation_loss': mean_activity_loss,
                    'boundary_consistency_train_loss': (
                        float(np.mean(epoch_boundary_consistency_losses))
                        if epoch_boundary_consistency_losses else None
                    ),
                    'boundary_consistency_validation_loss': None,
                }
                if loss_group_ids is None:
                    for index, target in enumerate(self.target_names):
                        train_losses = epoch_group_train_losses[target]
                        history[f'{target}__train_loss'] = (
                            float(np.mean(train_losses)) if train_losses else None
                        )
                        history[f'{target}__validation_loss'] = group_losses[target]
                        group_score = _competition_score_loss(
                            predicted_capacity[..., index:index + 1],
                            actual_capacity[..., index:index + 1],
                        )
                        group_nmae, group_ficr = _competition_components(
                            predicted_capacity[..., index:index + 1],
                            actual_capacity[..., index:index + 1],
                        )
                        history[f'{target}__validation_score'] = -float(
                            group_score.detach().mean().cpu()
                        )
                        history[f'{target}__validation_nmae'] = float(
                            group_nmae.detach().mean().cpu()
                        )
                        history[f'{target}__validation_ficr'] = float(
                            group_ficr.detach().mean().cpu()
                        )
                else:
                    assert long_group_components is not None
                    for group_code, target in enumerate(TARGET_COLS):
                        group_nmae, group_ficr = long_group_components[group_code]
                        group_score = 0.5 * (1.0 - group_nmae) + 0.5 * group_ficr
                        history[f'{target}__train_loss'] = None
                        history[f'{target}__validation_loss'] = group_losses[target]
                        history[f'{target}__validation_score'] = float(
                            group_score.detach().mean().cpu()
                        )
                        history[f'{target}__validation_nmae'] = float(
                            group_nmae.detach().mean().cpu()
                        )
                        history[f'{target}__validation_ficr'] = float(
                            group_ficr.detach().mean().cpu()
                        )

                if selection_period_ids is not None:
                    period_codes = sorted(
                        int(value)
                        for value in torch.unique(selection_period_ids).detach().cpu().tolist()
                    )
                    for period_code in period_codes:
                        period_mask = selection_period_ids == period_code
                        period_actual = torch.where(
                            period_mask.unsqueeze(-1),
                            actual_capacity,
                            torch.full_like(actual_capacity, _MISSING_TARGET),
                        )
                        if loss_group_ids is not None:
                            period_components = (
                                long_group_competition_components_torch(
                                    period_actual,
                                    predicted_capacity,
                                    loss_group_ids,
                                    n_groups=len(TARGET_COLS),
                                )
                            )
                            period_group_nmae = torch.stack(
                                [period_components[i][0] for i in range(len(TARGET_COLS))],
                                dim=-1,
                            )
                            period_group_ficr = torch.stack(
                                [period_components[i][1] for i in range(len(TARGET_COLS))],
                                dim=-1,
                            )
                            period_nmae = period_group_nmae.mean(dim=-1)
                            period_ficr = period_group_ficr.mean(dim=-1)
                            period_score = (
                                0.5 * (1.0 - period_nmae)
                                + 0.5 * period_ficr
                            )
                        else:
                            period_nmae, period_ficr = _competition_components(
                                predicted_capacity, period_actual
                            )
                            period_score = (
                                0.5 * (1.0 - period_nmae)
                                + 0.5 * period_ficr
                            )
                        prefix = f'selection_period_{period_code + 1}'
                        history[f'{prefix}__validation_score'] = float(
                            period_score.detach().mean().cpu()
                        )
                        history[f'{prefix}__validation_nmae'] = float(
                            period_nmae.detach().mean().cpu()
                        )
                        history[f'{prefix}__validation_ficr'] = float(
                            period_ficr.detach().mean().cpu()
                        )

                if self.config.temporal_group_dro:
                    for block_id, losses in epoch_temporal_losses.items():
                        if losses:
                            dro_log_weights[block_id] += (
                                self.config.temporal_group_dro_eta
                                * float(np.mean(losses))
                            )
                    offset = max(dro_log_weights.values())
                    for block_id in dro_log_weights:
                        dro_log_weights[block_id] -= offset
                if self.config.temporal_group_dro:
                    dro_weights = normalized_dro_weights()
                    for block_id, label in temporal_block_labels.items():
                        losses = epoch_temporal_losses[block_id]
                        history[f'temporal_{label}__train_ficr_loss'] = (
                            float(np.mean(losses)) if losses else None
                        )
                        history[f'temporal_{label}__dro_weight'] = (
                            dro_weights[block_id]
                        )
                self.training_history.append(history)
                LOGGER.info(
                    'RealMLP epoch %03d/%03d | train_loss=%s | val_loss=%.8f | '
                    'score=%.6f | NMAE=%.6f | FICR=%.6f',
                    history['step'], self.epochs,
                    f"{history['train_loss']:.8f}" if history['train_loss'] is not None else 'N/A',
                    float(history['validation_loss']),
                    float(history['validation_score']),
                    float(history['validation_nmae']),
                    float(history['validation_ficr']),
                )
                epoch_train_losses.clear()
                epoch_objective_train_losses.clear()
                epoch_activity_train_losses.clear()
                epoch_boundary_consistency_losses.clear()
                for losses in epoch_group_train_losses.values():
                    losses.clear()
                for losses in epoch_temporal_losses.values():
                    losses.clear()
                return value
            return original_apply(y_pred, actual, metric_name)

        Metrics.apply = staticmethod(score_aware_apply)
        self.early_stopping_enabled = False
        fallback_rows = min(512, len(X))
        fit_X_val = X_valid if has_validation else X.iloc[-fallback_rows:]
        fit_y_val = y_valid if has_validation else y.iloc[-fallback_rows:]
        fit_group_codes_val = (
            valid_group_codes if has_validation
            else (None if train_group_codes is None else train_group_codes[-fallback_rows:])
        )
        capacity_y = np.array(y.to_numpy(dtype=np.float32), copy=True)
        capacity_y_val = np.array(
            fit_y_val.to_numpy(dtype=np.float32), copy=True
        )
        observed_y = np.isfinite(capacity_y)
        observed_y_val = np.isfinite(capacity_y_val)
        activity_y = np.where(
            observed_y, capacity_y >= 0.10, _MISSING_TARGET
        ).astype(np.float32)
        activity_y_val = np.where(
            observed_y_val, capacity_y_val >= 0.10, _MISSING_TARGET
        ).astype(np.float32)
        if self.config.group3_reliability_weighting:
            reliability_y, self.group3_reliability_metadata = (
                group3_cross_fitted_reliability(
                    X,
                    y,
                    min_weight=self.config.group3_reliability_min_weight,
                    seed=self.config.seed,
                )
            )
            LOGGER.info(
                'Group 3 cross-fitted reliability: %s',
                self.group3_reliability_metadata,
            )
        else:
            reliability_y = np.ones_like(capacity_y, dtype=np.float32)
            self.group3_reliability_metadata = {
                'enabled': False, 'reason': 'disabled by configuration'
            }
        reliability_y_val = np.ones_like(capacity_y_val, dtype=np.float32)
        capacity_y[~observed_y] = _MISSING_TARGET
        capacity_y_val[~observed_y_val] = _MISSING_TARGET
        validation_block_ids, _ = _quarter_block_ids(fit_X_val.index)
        if train_group_codes is not None:
            activity_y = _pack_activity_group_metadata(activity_y, train_group_codes)
            if fit_group_codes_val is None:
                raise ValueError('Validation group codes are missing.')
            activity_y_val = _pack_activity_group_metadata(
                activity_y_val, fit_group_codes_val
            )
        if valid_selection_period_codes is not None:
            if not has_validation:
                raise ValueError(
                    'Selection-period metadata requires explicit validation data.'
                )
            activity_y_val = _pack_activity_selection_period_metadata(
                activity_y_val, valid_selection_period_codes
            )
        packed_activity_y = (
            _pack_activity_block_metadata(activity_y, block_ids)
            if self.config.temporal_group_dro else activity_y
        )
        packed_activity_y_val = (
            _pack_activity_block_metadata(activity_y_val, validation_block_ids)
            if self.config.temporal_group_dro else activity_y_val
        )
        if self.config.group3_reliability_weighting:
            packed_activity_y = _pack_reliability_metadata(
                packed_activity_y, reliability_y
            )
            packed_activity_y_val = _pack_reliability_metadata(
                packed_activity_y_val, reliability_y_val
            )
        fit_y = np.concatenate([capacity_y, packed_activity_y], axis=1)
        fit_y_val_array = np.concatenate(
            [capacity_y_val, packed_activity_y_val], axis=1
        )
        self.model = RealMLP_TD_Regressor(
            device=self.device, random_state=self.config.seed,
            n_threads=self.n_threads, n_epochs=self.epochs,
            batch_size=self.config.batch_size, n_cv=1, n_refit=0,
            n_ens=8, normalize_output=False,
            lr=self.config.learning_rate, lr_sched='coslog4',
            **_REALMLP_TD_REG_PARAMS,
            train_metric_name=_TRAIN_FICR_LOSS,
            val_metric_name=_VAL_FICR_LOSS,
            use_early_stopping=False,
            stop_epoch=None if has_validation else self.epochs,
            verbosity=0,
        )

        try:
            self.model.fit(
                X, fit_y, X_val=fit_X_val, y_val=fit_y_val_array
            )
        finally:
            Metrics.apply = staticmethod(original_apply)
        if self.config.temporal_group_dro:
            final_dro_weights = normalized_dro_weights()
            self.temporal_block_weights = {
                temporal_block_labels[key]: final_dro_weights[key]
                for key in temporal_block_labels
            }
        else:
            self.temporal_block_weights = {}
        self.best_iteration = self._read_best_iteration()
        self.elapsed_seconds = time.perf_counter() - started
        LOGGER.info('RealMLP best epoch=%d', self.best_iteration)
        return self

    def _read_best_iteration(self) -> int:
        candidates = [
            getattr(self.model, 'fit_params_', None),
            getattr(getattr(self.model, 'alg_interface_', None), 'fit_params', None),
        ]
        for candidate in candidates:
            if isinstance(candidate, list) and candidate:
                candidate = candidate[0]
            if isinstance(candidate, dict):
                value = candidate.get('stop_epoch', candidate.get('best_epoch'))
                if isinstance(value, dict):
                    value = value.get(_VAL_FICR_LOSS)
                if value is not None:
                    return max(1, int(value))
        return self.epochs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('RealMLP must be fitted before predict().')
        prediction = np.asarray(self.model.predict(X), dtype=float)
        prediction = prediction.reshape(len(X), 2 * len(self.target_names))
        return prediction[:, :len(self.target_names)]

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'RealMLP-TD', 'training': 'supervised-gradient',
            'architecture': 'shared-trunk-capacity-and-activity-heads',
            'targets': self.target_names,
            'max_epochs': self.epochs, 'best_iteration': self.best_iteration,
            'validation_metric': 'ficr-aware-loss', 'n_ens': 8,
            'loss_name': f'{self.config.ficr_loss}-ficr-aware',
            'long_group_ficr_regularization': True,
            'long_group_marker_column': _LOSS_GROUP_CODE_COLUMN,
            'selection_metric': f'{self.config.ficr_loss}-ficr-aware-loss',
            'ficr_weight': self.config.ficr_weight,
            'ficr_temperature': self.config.ficr_temperature,
            'ficr_loss': self.config.ficr_loss,
            'ficr_relu_margin': self.config.ficr_relu_margin,
            'ficr_relu_thresholds': [
                0.06 - self.config.ficr_relu_margin,
                0.08 - self.config.ficr_relu_margin,
            ],
            'ficr_boundary_consistency_weight': (
                self.config.ficr_boundary_consistency_weight
            ),
            'ficr_boundary_consistency': 'internal-ensemble-soft-reward-variance',
            'temporal_group_dro': self.config.temporal_group_dro,
            'temporal_group_dro_eta': self.config.temporal_group_dro_eta,
            'temporal_blocks': 'calendar-quarter',
            'temporal_block_weights': self.temporal_block_weights,
            'temporal_metadata_transport': 'packed-first-activity-target',
            'group3_reliability_weighting': (
                self.config.group3_reliability_weighting
            ),
            'group3_reliability_min_weight': (
                self.config.group3_reliability_min_weight
            ),
            'group3_reliability': self.group3_reliability_metadata,
            'group3_reliability_scope': 'capacity-loss-only',
            'group3_reliability_validation_weights': 'all-one',
            'activity_loss_weight': self.config.activity_loss_weight,
            'activity_threshold': 0.10,
            'learning_rate': self.config.learning_rate,
            'lr_schedule': 'coslog4',
            'dropout': 0.15,
            'dropout_schedule': 'flat_cos',
            'weight_decay': 2e-2,
            'weight_decay_schedule': 'flat_cos',
            'optimizer': 'adam',
            'squared_momentum': 0.95,
            'weight_parameterization': 'ntk',
            'target_normalization': False,
            'target_masking': 'missing=-1; capacity>=0.10; activity=all-observed',
            'early_stopping': self.early_stopping_enabled,
            'training_history': self.training_history,
            'device': self.device, 'n_threads': self.n_threads,
            'elapsed_seconds': self.elapsed_seconds,
        }