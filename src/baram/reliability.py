'''Cross-fitted target reliability weights for the partially observed group 3.'''

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


GROUP3_RELIABILITY_FEATURES = (
    'gfs__ws100__mean',
    'gfs__ws100__std',
    'gfs__surface_0_gust__mean',
    'ldaps__ws50_mid__mean',
    'ldaps__ws50_mid__std',
    'time__hour_sin',
    'time__hour_cos',
    'time__doy_sin',
    'time__doy_cos',
)


def group3_cross_fitted_reliability(
    features: pd.DataFrame,
    capacity_targets: pd.DataFrame,
    *,
    min_weight: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    '''Return per-target weights using leave-one-quarter-out group 3 residuals.'''
    from sklearn.ensemble import GradientBoostingRegressor

    missing = set(GROUP3_RELIABILITY_FEATURES) - set(features.columns)
    if missing:
        raise ValueError(
            f'Missing group 3 reliability features: {sorted(missing)}'
        )
    targets = capacity_targets.to_numpy(dtype=np.float32)
    weights = np.ones_like(targets, dtype=np.float32)
    if capacity_targets.shape[1] < 3:
        return weights, {'enabled': False, 'reason': 'group 3 unavailable'}

    observed = capacity_targets.iloc[:, 2].notna().to_numpy()
    observed_positions = np.flatnonzero(observed)
    if len(observed_positions) < 256:
        return weights, {'enabled': False, 'reason': 'insufficient group 3 rows'}
    observed_index = pd.DatetimeIndex(features.index[observed])
    block_keys = (
        observed_index.year.astype(np.int64) * 4
        + observed_index.quarter.astype(np.int64)
        - 1
    ).to_numpy()
    unique_blocks = np.unique(block_keys)
    if len(unique_blocks) < 2:
        return weights, {'enabled': False, 'reason': 'insufficient quarters'}

    compact = features.loc[
        observed, list(GROUP3_RELIABILITY_FEATURES)
    ].astype(float)
    compact = compact.replace([np.inf, -np.inf], np.nan)
    actual = capacity_targets.iloc[observed_positions, 2].clip(0.0, 1.0)
    expected = np.full(len(actual), np.nan, dtype=float)
    for block in unique_blocks:
        train_mask = block_keys != block
        holdout_mask = block_keys == block
        train_x = compact.iloc[train_mask]
        holdout_x = compact.iloc[holdout_mask]
        medians = train_x.median(axis=0).fillna(0.0)
        train_x = train_x.fillna(medians)
        holdout_x = holdout_x.fillna(medians)
        detector = GradientBoostingRegressor(
            loss='huber',
            learning_rate=0.05,
            n_estimators=100,
            max_depth=2,
            min_samples_leaf=48,
            random_state=seed,
        )
        detector.fit(train_x, actual.iloc[train_mask])
        expected[holdout_mask] = detector.predict(holdout_x)
    expected = np.clip(expected, 0.0, 1.0)

    residual = actual.to_numpy(dtype=float) - expected
    residual_median = float(np.median(residual))
    residual_mad = float(
        1.4826 * np.median(np.abs(residual - residual_median))
    )
    if not np.isfinite(residual_mad) or residual_mad <= 1e-8:
        return weights, {'enabled': False, 'reason': 'degenerate residual scale'}
    proxy = capacity_targets.iloc[observed_positions, :2].mean(
        axis=1, skipna=True
    ).to_numpy(dtype=float)
    negative_z = (residual_median - residual) / residual_mad
    severity = np.clip((negative_z - 1.5) / 1.5, 0.0, 1.0)
    site_specific_low = (
        (expected >= 0.30)
        & np.isfinite(proxy)
        & ((proxy - actual.to_numpy(dtype=float)) > 0.20)
    )
    reliability = np.where(
        site_specific_low,
        1.0 - (1.0 - min_weight) * severity,
        1.0,
    ).astype(np.float32)
    weights[observed_positions, 2] = reliability
    eligible = actual.to_numpy(dtype=float) >= 0.10
    return weights, {
        'enabled': True,
        'method': 'leave-one-quarter-out-huber-power-curve',
        'n_observed_group3': int(len(actual)),
        'n_downweighted': int(np.sum(reliability < 0.999)),
        'n_eligible_downweighted': int(
            np.sum((reliability < 0.999) & eligible)
        ),
        'mean_weight': float(np.mean(reliability)),
        'eligible_mean_weight': float(np.mean(reliability[eligible])),
        'min_weight': float(np.min(reliability)),
        'residual_median': residual_median,
        'residual_mad': residual_mad,
        'quarters': [
            f'{int(key) // 4}Q{int(key) % 4 + 1}' for key in unique_blocks
        ],
    }
