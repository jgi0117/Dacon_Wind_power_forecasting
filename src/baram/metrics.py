"""대회 공식 평가 산식과 발전량 정규화 유틸리티."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}


def _validate_frames(answer: pd.DataFrame, prediction: pd.DataFrame) -> None:
    missing_answer = set(TARGET_COLS) - set(answer.columns)
    missing_prediction = set(TARGET_COLS) - set(prediction.columns)
    if missing_answer:
        raise ValueError(f"Missing answer columns: {sorted(missing_answer)}")
    if missing_prediction:
        raise ValueError(f"Missing prediction columns: {sorted(missing_prediction)}")
    if len(answer) != len(prediction):
        raise ValueError(
            f"Answer/prediction row mismatch: {len(answer)} != {len(prediction)}"
        )


def align_by_time(
    answer: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    answer_time_col: str = "kst_dtm",
    prediction_time_col: str = "forecast_kst_dtm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly align answer and prediction frames by timestamp."""
    answer = answer.copy()
    prediction = prediction.copy()
    answer[answer_time_col] = pd.to_datetime(answer[answer_time_col])
    prediction[prediction_time_col] = pd.to_datetime(
        prediction[prediction_time_col]
    )
    if answer[answer_time_col].duplicated().any():
        raise ValueError("Answer timestamps contain duplicates.")
    if prediction[prediction_time_col].duplicated().any():
        raise ValueError("Prediction timestamps contain duplicates.")
    answer = answer.set_index(answer_time_col).sort_index()
    prediction = prediction.set_index(prediction_time_col).sort_index()
    if not answer.index.equals(prediction.index):
        raise ValueError(
            "Answer/prediction timestamps differ: "
            f"answer_only={len(answer.index.difference(prediction.index))}, "
            f"prediction_only={len(prediction.index.difference(answer.index))}"
        )
    return answer[TARGET_COLS], prediction[TARGET_COLS]


def capacity_factor(y: pd.Series, target: str) -> pd.Series:
    return y.astype(float) / CAPACITY_KWH[target]


def restore_generation(prediction: np.ndarray, target: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    return np.clip(prediction, 0.0, 1.0) * CAPACITY_KWH[target]


def target_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    '''Competition score for one capacity-factor target (higher is better).'''
    actual = np.asarray(actual, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual >= 0.10)
    if not valid.any():
        raise ValueError('No rows are eligible for the target score.')
    actual_valid = actual[valid]
    error = np.abs(np.clip(prediction[valid], 0.0, 1.0) - actual_valid)
    nmae = float(error.mean())
    unit_price = np.select(
        [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
    )
    denominator = float(np.sum(actual_valid * 4.0))
    ficr = 0.0 if denominator <= 0.0 else float(
        np.sum(actual_valid * unit_price) / denominator
    )
    return 0.5 * (1.0 - nmae) + 0.5 * ficr


def ficr_aware_loss_torch(
    actual: Any,
    prediction: Any,
    *,
    ficr_weight: float = 0.75,
    temperature: float = 0.01,
    sample_weight: Any | None = None,
) -> Any:
    '''Masked soft-FICR loss with equal weighting across target groups.'''
    import torch

    if actual.ndim < prediction.ndim:
        actual = actual.unsqueeze(0).expand_as(prediction)
    elif prediction.ndim < actual.ndim:
        prediction = prediction.unsqueeze(0).expand_as(actual)
    if actual.shape != prediction.shape:
        actual = torch.broadcast_to(actual, prediction.shape)
    if sample_weight is None:
        sample_weight = torch.ones_like(actual)
    elif sample_weight.ndim < actual.ndim:
        sample_weight = sample_weight.unsqueeze(0).expand_as(actual)
    elif sample_weight.shape != actual.shape:
        sample_weight = torch.broadcast_to(sample_weight, actual.shape)

    if actual.ndim < 2:
        actual = actual.reshape(-1, 1)
        prediction = prediction.reshape(-1, 1)

    valid = torch.isfinite(actual) & torch.isfinite(prediction) & (actual >= 0.10)
    safe_actual = torch.where(valid, actual, torch.zeros_like(actual))
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    smooth_error = torch.sqrt((safe_prediction - safe_actual).square() + 1e-8)
    valid_float = valid.to(smooth_error.dtype)
    effective_weight = valid_float * sample_weight.clamp_min(0.0)
    sample_dim = -2
    weight_sum = effective_weight.sum(dim=sample_dim)
    safe_weight_sum = weight_sum.clamp_min(1e-12)
    mae = (
        (smooth_error * effective_weight).sum(dim=sample_dim)
        / safe_weight_sum
    )
    sigmoid_6 = torch.sigmoid((0.06 - smooth_error) / temperature)
    sigmoid_8 = torch.sigmoid((0.08 - smooth_error) / temperature)
    soft_reward = (3.0 * sigmoid_8 + sigmoid_6) / 4.0
    actual_weight = safe_actual * effective_weight
    soft_ficr = (soft_reward * actual_weight).sum(dim=sample_dim) / (
        actual_weight.sum(dim=sample_dim).clamp_min(1e-12)
    )
    group_loss = (
        (1.0 - ficr_weight) * mae + ficr_weight * (1.0 - soft_ficr)
    )
    valid_groups = weight_sum > 0
    valid_group_count = valid_groups.sum(dim=-1).clamp_min(1)
    return (
        torch.where(valid_groups, group_loss, torch.zeros_like(group_loss))
        .sum(dim=-1)
        / valid_group_count
    )


def relu_ficr_aware_loss_torch(
    actual, prediction, ficr_weight=0.75, margin=0.005,
):
    '''Masked conservative ReLU-FICR loss with global smooth MAE.'''
    import torch
    import torch.nn.functional as functional

    if actual.ndim < prediction.ndim:
        actual = actual.unsqueeze(0).expand_as(prediction)
    elif prediction.ndim < actual.ndim:
        prediction = prediction.unsqueeze(0).expand_as(actual)
    if actual.shape != prediction.shape:
        actual = torch.broadcast_to(actual, prediction.shape)
    if actual.ndim < 2:
        actual = actual.reshape(-1, 1)
        prediction = prediction.reshape(-1, 1)

    valid = torch.isfinite(actual) & torch.isfinite(prediction) & (actual >= 0.10)
    safe_actual = torch.where(valid, actual, torch.zeros_like(actual))
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    smooth_error = torch.sqrt((safe_prediction - safe_actual).square() + 1e-8)
    valid_float = valid.to(smooth_error.dtype)
    count = valid_float.sum(dim=-2)
    safe_count = count.clamp_min(1.0)
    mae = (smooth_error * valid_float).sum(dim=-2) / safe_count
    mean_actual = (
        (safe_actual * valid_float).sum(dim=-2) / safe_count
    ).clamp_min(1e-12)
    normalized_weight = safe_actual / mean_actual.unsqueeze(-2)
    hinge_6 = functional.relu(smooth_error - (0.06 - margin))
    hinge_8 = functional.relu(smooth_error - (0.08 - margin))
    relu_penalty = 0.25 * hinge_6 + 0.75 * hinge_8
    weighted_penalty = (
        relu_penalty * normalized_weight * valid_float
    ).sum(dim=-2) / safe_count
    group_loss = (
        (1.0 - ficr_weight) * mae + ficr_weight * weighted_penalty
    )
    valid_groups = count > 0
    return (
        torch.where(valid_groups, group_loss, torch.zeros_like(group_loss))
        .sum(dim=-1)
        / valid_groups.sum(dim=-1).clamp_min(1)
    )


def temporal_group_dro_ficr_loss_torch(
    actual, prediction, block_ids, block_weights,
    ficr_weight=0.75, temperature=0.01,
):
    '''Combine global MAE with GroupDRO-weighted temporal soft-FICR.'''
    import torch

    ids = block_ids
    while ids.ndim > 1:
        ids = ids.squeeze(-1) if ids.shape[-1] == 1 else ids[0]
    ids = ids.to(dtype=torch.long)
    global_mae = ficr_aware_loss_torch(
        actual, prediction, ficr_weight=0.0, temperature=temperature
    )
    block_losses = {}
    for block_id in sorted(block_weights):
        selected = ids == block_id
        if bool(selected.any().item()):
            block_losses[block_id] = ficr_aware_loss_torch(
                actual[..., selected, :], prediction[..., selected, :],
                ficr_weight=1.0, temperature=temperature,
            )
    if not block_losses:
        fallback = ficr_aware_loss_torch(
            actual, prediction, ficr_weight=ficr_weight,
            temperature=temperature,
        )
        return fallback, {}
    weight_sum = sum(block_weights[key] for key in block_losses)
    robust_ficr = sum(
        block_losses[key] * (block_weights[key] / weight_sum)
        for key in block_losses
    )
    loss = (1.0 - ficr_weight) * global_mae + ficr_weight * robust_ficr
    return loss, block_losses


def ficr_boundary_consistency_loss_torch(
    actual, prediction, temperature=0.01,
):
    '''Penalize ensemble disagreement in the soft 6%/8% FICR decisions.'''
    import torch

    if prediction.ndim < 3 or prediction.shape[0] < 2:
        return ficr_aware_loss_torch(
            actual, prediction, ficr_weight=0.0, temperature=temperature
        ) * 0.0
    if actual.ndim < prediction.ndim:
        actual = actual.unsqueeze(0).expand_as(prediction)
    elif actual.shape != prediction.shape:
        actual = torch.broadcast_to(actual, prediction.shape)
    valid = torch.isfinite(actual) & torch.isfinite(prediction) & (actual >= 0.10)
    safe_actual = torch.where(valid, actual, torch.zeros_like(actual))
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    error = (safe_prediction - safe_actual).abs()
    soft_6 = torch.sigmoid((0.06 - error) / temperature)
    soft_8 = torch.sigmoid((0.08 - error) / temperature)
    soft_reward = (3.0 * soft_8 + soft_6) / 4.0
    reward_variance = soft_reward.var(dim=0, unbiased=False)

    mean_error = error.mean(dim=0)
    mean_6 = torch.sigmoid((0.06 - mean_error) / temperature)
    mean_8 = torch.sigmoid((0.08 - mean_error) / temperature)
    attention = torch.maximum(
        4.0 * mean_6 * (1.0 - mean_6),
        4.0 * mean_8 * (1.0 - mean_8),
    ).detach()
    base_actual = safe_actual[0]
    base_valid = valid.all(dim=0).to(reward_variance.dtype)
    boundary_weight = base_actual * attention * base_valid
    denominator = boundary_weight.sum(dim=-2)
    group_loss = (
        reward_variance * boundary_weight
    ).sum(dim=-2) / denominator.clamp_min(1e-12)
    valid_groups = denominator > 0.0
    scalar = torch.where(
        valid_groups, group_loss, torch.zeros_like(group_loss)
    ).sum(dim=-1) / valid_groups.sum(dim=-1).clamp_min(1)
    return scalar.expand(prediction.shape[0])


def activity_loss_torch(actual: Any, logits: Any) -> Any:
    '''Masked binary loss for whether an observed capacity factor is >= 10%.'''
    import torch
    import torch.nn.functional as functional

    if actual.ndim < logits.ndim:
        actual = actual.unsqueeze(0).expand_as(logits)
    elif logits.ndim < actual.ndim:
        logits = logits.unsqueeze(0).expand_as(actual)
    if actual.shape != logits.shape:
        actual = torch.broadcast_to(actual, logits.shape)
    if actual.ndim < 2:
        actual = actual.reshape(-1, 1)
        logits = logits.reshape(-1, 1)
    observed = torch.isfinite(actual) & (actual >= 0.0)
    labels = (actual >= 0.10).to(logits.dtype)
    safe_logits = torch.where(observed, logits, torch.zeros_like(logits))
    element_loss = functional.binary_cross_entropy_with_logits(
        safe_logits, labels, reduction='none'
    )
    observed_float = observed.to(element_loss.dtype)
    count = observed_float.sum(dim=-2)
    group_loss = (
        (element_loss * observed_float).sum(dim=-2) / count.clamp_min(1.0)
    )
    valid_groups = count > 0
    return (
        torch.where(valid_groups, group_loss, torch.zeros_like(group_loss))
        .sum(dim=-1)
        / valid_groups.sum(dim=-1).clamp_min(1)
    )


def metric_report(answer: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    _validate_frames(answer, prediction)
    groups: dict[str, dict[str, float | int]] = {}
    nmaes: list[float] = []
    ficrs: list[float] = []
    for target in TARGET_COLS:
        actual = answer[target].to_numpy(dtype=float)
        forecast = prediction[target].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[target]
        valid = np.isfinite(actual) & (actual >= capacity * 0.10)
        if not valid.any():
            raise ValueError(f"{target}: 평가 가능한 정답 행이 없습니다.")
        if not np.isfinite(forecast[valid]).all():
            raise ValueError(f"{target}: 평가 대상 예측에 NaN/inf가 있습니다.")
        actual_valid = actual[valid]
        error_rate = np.abs(forecast[valid] - actual_valid) / capacity
        nmae = float(error_rate.mean())
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0
        )
        ficr = float(np.sum(actual_valid * unit_price) / np.sum(actual_valid * 4.0))
        nmaes.append(nmae)
        ficrs.append(ficr)
        groups[target] = {
            "n_valid": int(valid.sum()),
            "nmae": nmae,
            "one_minus_nmae": 1.0 - nmae,
            "ficr": ficr,
            "within_6pct_rate": float(np.mean(error_rate <= 0.06)),
            "between_6_and_8pct_rate": float(
                np.mean((error_rate > 0.06) & (error_rate <= 0.08))
            ),
            "over_8pct_rate": float(np.mean(error_rate > 0.08)),
        }
    one_minus_nmae = 1.0 - float(np.mean(nmaes))
    ficr = float(np.mean(ficrs))
    return {
        "score": 0.5 * one_minus_nmae + 0.5 * ficr,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "groups": groups,
    }


def metric(
    answer: pd.DataFrame, prediction: pd.DataFrame
) -> tuple[float, float, float]:
    """Return ``(score, one_minus_nmae, ficr)``."""
    report = metric_report(answer, prediction)
    return report["score"], report["one_minus_nmae"], report["ficr"]


def report_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Convert the per-target section of a metric report to a DataFrame."""
    frame = pd.DataFrame.from_dict(report["groups"], orient="index")
    frame.index.name = "group"
    return frame


def evaluate_complete_rows(
    answer: pd.DataFrame, prediction: pd.DataFrame
) -> dict[str, Any]:
    usable = answer[TARGET_COLS].notna().all(axis=1)
    if not usable.any():
        raise ValueError("세 그룹 정답이 모두 있는 평가 행이 없습니다.")
    return metric_report(answer.loc[usable, TARGET_COLS], prediction.loc[usable, TARGET_COLS])


def flatten_report(model_name: str, report: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "score": report["score"],
        "one_minus_nmae": report["one_minus_nmae"],
        "ficr": report["ficr"],
    }
    for target, group in report["groups"].items():
        row[f"{target}__nmae"] = group["nmae"]
        row[f"{target}__ficr"] = group["ficr"]
    return row
