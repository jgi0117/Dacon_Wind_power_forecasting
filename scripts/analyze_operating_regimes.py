'''Audit whether similar NWP wind conditions map to multiple power regimes.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baram.metrics import CAPACITY_KWH, TARGET_COLS


WIND_FEATURES = (
    'gfs__ws100__mean',
    'gfs__ws80__mean',
    'ldaps__ws50_mid__mean',
    'ldaps__ws10__mean',
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _conditional_bins(
    X: pd.DataFrame, y: pd.DataFrame, bin_width: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in WIND_FEATURES:
        wind_bin = np.floor(X[feature] / bin_width) * bin_width
        for target in TARGET_COLS:
            frame = pd.DataFrame({
                'wind_bin': wind_bin,
                'capacity_factor': y[target] / CAPACITY_KWH[target],
            }).dropna()
            for value, group in frame.groupby('wind_bin', sort=True):
                rows.append({
                    'wind_feature': feature,
                    'target': target,
                    'wind_bin': float(value),
                    'n': int(len(group)),
                    'mean': float(group['capacity_factor'].mean()),
                    'q10': float(group['capacity_factor'].quantile(0.10)),
                    'q50': float(group['capacity_factor'].quantile(0.50)),
                    'q90': float(group['capacity_factor'].quantile(0.90)),
                    'inactive_rate': float((group['capacity_factor'] < 0.10).mean()),
                    'normal_rate': float((group['capacity_factor'] > 0.40).mean()),
                })
    result = pd.DataFrame(rows)
    result['q10_q90_spread'] = result['q90'] - result['q10']
    result['ambiguous'] = (
        (result['n'] >= 100)
        & (result['inactive_rate'] >= 0.10)
        & (result['normal_rate'] >= 0.10)
    )
    return result


def _summary(
    X: pd.DataFrame, y: pd.DataFrame, bins: pd.DataFrame
) -> dict[str, object]:
    conditional: list[dict[str, object]] = []
    for (feature, target), group in bins.groupby(['wind_feature', 'target']):
        ambiguous_rows = int(group.loc[group['ambiguous'], 'n'].sum())
        observed_rows = int(y[target].notna().sum())
        high_wind = X[feature] >= X[feature].quantile(0.75)
        factors = y.loc[high_wind, target] / CAPACITY_KWH[target]
        factors = factors.dropna()
        conditional.append({
            'wind_feature': feature,
            'target': target,
            'observed_rows': observed_rows,
            'ambiguous_bins': int(group['ambiguous'].sum()),
            'ambiguous_rows': ambiguous_rows,
            'ambiguous_row_fraction': ambiguous_rows / observed_rows,
            'median_q10_q90_spread': float(group['q10_q90_spread'].median()),
            'top_quartile_wind_low_output_rate': float((factors < 0.10).mean()),
            'top_quartile_wind_normal_output_rate': float((factors > 0.40).mean()),
        })
    factors = pd.DataFrame({
        target: y[target] / CAPACITY_KWH[target] for target in TARGET_COLS
    }).dropna()
    levels = pd.DataFrame(
        np.where(factors < 0.10, 0, np.where(factors < 0.40, 1, 2)),
        index=factors.index,
        columns=factors.columns,
    )
    return {
        'definition': {
            'ambiguous_bin': 'n>=100 and CF<0.10 rate>=0.10 and CF>0.40 rate>=0.10',
            'warning': 'Ambiguity is a diagnostic signal, not a curtailment label.',
        },
        'conditional_wind_summary': conditional,
        'complete_target_rows': int(len(factors)),
        'mixed_group_level_rate': float((levels.nunique(axis=1) > 1).mean()),
        'cross_group_cf_range_over_0p4_rate': float(
            ((factors.max(axis=1) - factors.min(axis=1)) > 0.40).mean()
        ),
        'target_correlations': factors.corr().to_dict(),
    }


def _plot(X: pd.DataFrame, y: pd.DataFrame, destination: Path) -> None:
    features = ('gfs__ws100__mean', 'ldaps__ws50_mid__mean')
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for row, feature in enumerate(features):
        source = feature.partition('__')[0]
        for column, target in enumerate(TARGET_COLS):
            axis = axes[row, column]
            factors = y[target] / CAPACITY_KWH[target]
            valid = factors.notna() & X[feature].notna()
            axis.hexbin(
                X.loc[valid, feature], factors.loc[valid],
                gridsize=45, mincnt=1, bins='log', cmap='viridis',
            )
            axis.axhline(0.10, color='tab:red', linestyle='--', linewidth=1)
            axis.set_title(f'{target} / {source}')
            axis.set_xlabel('forecast wind speed')
            if column == 0:
                axis.set_ylabel('capacity factor')
            axis.set_ylim(-0.02, 1.05)
    figure.suptitle('Conditional wind-power density (diagnostic only)')
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, default=Path('artifacts'))
    parser.add_argument(
        '--output-dir', type=Path, default=Path('reports/v5/regime_audit')
    )
    parser.add_argument('--wind-bin-width', type=float, default=0.5)
    args = parser.parse_args()
    if args.wind_bin_width <= 0:
        parser.error('--wind-bin-width must be positive.')
    X = pd.read_pickle(args.artifacts_dir / 'X_train.pkl')
    y = pd.read_pickle(args.artifacts_dir / 'y_train.pkl')[TARGET_COLS]
    missing = set(WIND_FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f'Missing wind audit features: {sorted(missing)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bins = _conditional_bins(X, y, args.wind_bin_width)
    bins.to_csv(args.output_dir / 'conditional_wind_bins.csv', index=False)
    summary = _summary(X, y, bins)
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding='utf-8',
    )
    _plot(X, y, args.output_dir / 'conditional_wind_power.png')
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == '__main__':
    main()
