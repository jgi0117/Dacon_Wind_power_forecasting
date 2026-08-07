'''Consolidate per-model outputs into one tracked baseline report.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ('realmlp',)
TARGETS = ('kpx_group_1', 'kpx_group_2', 'kpx_group_3')
DISPLAY_NAMES = {
    'realmlp': 'RealMLP',
}
LOSS_NAMES = {
    'realmlp': 'sigmoid-ficr-aware-monthly-random',
}


def _report_path(runs_dir: Path, model: str) -> Path:
    return runs_dir / model / 'run_report.json'


def _monthly_path(runs_dir: Path, model: str) -> Path:
    return runs_dir / model / 'evaluation_results_by_month.csv'


def _load_reports(runs_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        path = _report_path(runs_dir, model)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        if model in payload.get('evaluation_reports', {}):
            reports[model] = payload
    return reports


def _preserved_dacon_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    old = pd.read_csv(path)
    if 'model' not in old.columns:
        return {}
    metric_columns = (
        'dacon_score', 'dacon_one_minus_nmae', 'dacon_ficr',
    )
    preserved: dict[str, dict[str, float]] = {}
    for row in old.set_index('model').itertuples():
        values = {
            column: float(getattr(row, column))
            for column in metric_columns
            if hasattr(row, column) and pd.notna(getattr(row, column))
        }
        if values:
            preserved[str(row.Index)] = values
    return preserved


def _result_frames(
    reports: dict[str, dict[str, Any]],
    dacon_metrics: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for model in MODELS:
        if model not in reports:
            continue
        payload = reports[model]
        report = payload['evaluation_reports'][model]
        leaderboard = dacon_metrics.get(model, {})
        results.append({
            'model': model,
            'display_name': DISPLAY_NAMES[model],
            'validation_score': report['score'],
            'one_minus_nmae': report['one_minus_nmae'],
            'ficr': report['ficr'],
            'dacon_score': leaderboard.get('dacon_score', np.nan),
            'dacon_one_minus_nmae': leaderboard.get(
                'dacon_one_minus_nmae', np.nan
            ),
            'dacon_ficr': leaderboard.get('dacon_ficr', np.nan),
        })
        for target in TARGETS:
            metric = report['groups'][target]
            groups.append({
                'model': model,
                'target': target,
                **metric,
            })
            metadata_by_target = payload.get('model_metadata', {}).get(model, {})
            metadata = metadata_by_target.get(
                'multitask', metadata_by_target.get(target, {})
            )
            rows_by_target = metadata.get('n_fit_rows_by_target', {})
            stacked_group3 = bool(metadata.get('stacking'))
            if stacked_group3 and target == 'kpx_group_3':
                best_iteration = metadata.get('group3_best_iteration')
                max_training_length = metadata.get('group3_epochs')
                elapsed_seconds = metadata.get('group3_elapsed_seconds')
            elif stacked_group3:
                best_iteration = metadata.get('stage1_best_iteration')
                max_training_length = metadata.get('stage1_epochs')
                elapsed_seconds = metadata.get('stage1_elapsed_seconds')
            else:
                best_iteration = metadata.get('best_iteration')
                max_training_length = metadata.get(
                    'max_epochs', metadata.get('max_iterations')
                )
                elapsed_seconds = metadata.get(
                    'elapsed_seconds_all_fits', metadata.get('elapsed_seconds')
                )
            training.append({
                'model': model,
                'target': target,
                'n_fit_rows': rows_by_target.get(
                    target, metadata.get('n_fit_rows')
                ),
                'best_iteration': best_iteration,
                'max_training_length': max_training_length,
                'loss_name': metadata.get(
                    'loss_name', LOSS_NAMES[model]
                ),
                'selection_metric': metadata.get(
                    'selection_metric',
                    'ficr-aware-loss',
                ),
                'elapsed_seconds': elapsed_seconds,
            })
    results_frame = pd.DataFrame(results).sort_values(
        'validation_score', ascending=False, ignore_index=True
    )
    results_frame.insert(0, 'rank', np.arange(1, len(results_frame) + 1))
    return results_frame, pd.DataFrame(groups), pd.DataFrame(training)


def _history_frame(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, payload in reports.items():
        metadata_by_target = payload.get('model_metadata', {}).get(model, {})
        multitask = metadata_by_target.get('multitask')
        if multitask:
            for item in multitask.get('training_history', []):
                aggregate_row = {
                    'model': model, 'target': 'all',
                    'step': item.get('step'),
                    'train_loss': item.get('train_loss'),
                    'validation_loss': item.get('validation_loss'),
                    'validation_score': item.get('validation_score'),
                    'validation_nmae': item.get('validation_nmae'),
                    'validation_ficr': item.get('validation_ficr'),
                    'training_objective_loss': item.get(
                        'training_objective_loss'
                    ),
                    'activity_train_loss': item.get('activity_train_loss'),
                    'activity_validation_loss': item.get(
                        'activity_validation_loss'
                    ),
                    'boundary_consistency_train_loss': item.get(
                        'boundary_consistency_train_loss'
                    ),
                    'boundary_consistency_validation_loss': item.get(
                        'boundary_consistency_validation_loss'
                    ),
                }
                aggregate_row.update({
                    key: value for key, value in item.items()
                    if (
                        key.startswith('temporal_')
                    )
                })
                rows.append(aggregate_row)
                for target in TARGETS:
                    rows.append({
                        'model': model, 'target': target,
                        'step': item.get('step'),
                        'train_loss': item.get(f'{target}__train_loss'),
                        'validation_loss': item.get(
                            f'{target}__validation_loss'
                        ),
                        'validation_score': item.get(
                            f'{target}__validation_score'
                        ),
                        'validation_nmae': item.get(
                            f'{target}__validation_nmae'
                        ),
                        'validation_ficr': item.get(
                            f'{target}__validation_ficr'
                        ),
                    })
            continue
        for target, metadata in metadata_by_target.items():
            for item in metadata.get('training_history', []):
                rows.append({'model': model, 'target': target, **item})
    columns = [
        'model', 'target', 'step', 'train_loss',
        'validation_loss', 'validation_score',
        'validation_nmae', 'validation_ficr',
        'training_objective_loss', 'activity_train_loss',
        'activity_validation_loss',
        'boundary_consistency_train_loss',
        'boundary_consistency_validation_loss',
    ]
    temporal_columns = sorted({
        key for row in rows for key in row
        if (
            key.startswith('temporal_')
        )
    })
    return pd.DataFrame(rows, columns=columns + temporal_columns)


def _monthly_frame(runs_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        path = _monthly_path(runs_dir, model)
        if path.is_file():
            frame = pd.read_csv(path)
            frames.append(frame.loc[frame['model'] == model])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_scores(results: pd.DataFrame, path: Path) -> None:
    ordered = results.sort_values('validation_score')
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(
        ordered['display_name'], ordered['validation_score'], color='#2878B5'
    )
    ax.bar_label(bars, fmt='%.4f', padding=4)
    if ordered['dacon_score'].notna().any():
        known = ordered['dacon_score'].notna()
        ax.scatter(
            ordered.loc[known, 'dacon_score'], ordered.loc[known, 'display_name'],
            marker='D', color='#D95319', label='DACON public score', zorder=3,
        )
        ax.legend()
    ax.set_xlabel('Score (higher is better)')
    ax.set_title('Version 5 original RealMLP chronological validation score')
    lower = max(0.0, float(ordered['validation_score'].min()) - 0.03)
    upper_values = [float(ordered['validation_score'].max())]
    if ordered['dacon_score'].notna().any():
        upper_values.append(float(ordered['dacon_score'].max()))
    ax.set_xlim(lower, min(1.0, max(upper_values) + 0.03))
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_dacon_components(results: pd.DataFrame, path: Path) -> None:
    available = results.dropna(
        subset=['dacon_one_minus_nmae', 'dacon_ficr']
    ).copy()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    if available.empty:
        ax.text(
            0.5, 0.5, 'DACON component scores have not been entered.',
            ha='center', va='center', transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        x = np.arange(len(available))
        width = 0.36
        mae_bars = ax.bar(
            x - width / 2, available['dacon_one_minus_nmae'], width,
            label='1-NMAE', color='#2878B5',
        )
        ficr_bars = ax.bar(
            x + width / 2, available['dacon_ficr'], width,
            label='FICR', color='#E07B39',
        )
        ax.bar_label(mae_bars, fmt='%.4f', padding=3, fontsize=8)
        ax.bar_label(ficr_bars, fmt='%.4f', padding=3, fontsize=8)
        ax.set_xticks(x, available['display_name'])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('DACON component score (higher is better)')
        ax.set_title('DACON public score components')
        ax.grid(axis='y', alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_histories(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_targets = ('all', *TARGETS)
    colors = dict(zip(
        plotted_targets,
        ('#222222', '#2878B5', '#E07B39', '#4E9F3D'),
        strict=True,
    ))
    model_history = history.loc[history['model'] == 'realmlp']
    for target in plotted_targets:
        target_history = model_history.loc[model_history['target'] == target]
        if target_history.empty:
            continue
        if target_history['train_loss'].notna().any():
            ax.plot(
                target_history['step'], target_history['train_loss'],
                linestyle='--', color=colors[target], alpha=0.75,
                label=f'{target} train',
            )
        if target_history['validation_loss'].notna().any():
            ax.plot(
                target_history['step'], target_history['validation_loss'],
                color=colors[target], label=f'{target} validation',
            )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Capacity objective loss')
    ax.set_title('Version 5 monthly-balanced validation history')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_validation_components(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'all': '#222222', 'kpx_group_1': '#2878B5',
        'kpx_group_2': '#E07B39', 'kpx_group_3': '#4E9F3D',
    }
    for target, color in colors.items():
        selected = history.loc[history['target'] == target]
        if selected.empty or not selected['validation_ficr'].notna().any():
            continue
        ax.plot(
            selected['step'], selected['validation_ficr'],
            color=color, label=target,
        )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Exact validation FICR')
    ax.set_title('Monthly-balanced random validation: exact FICR')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(results: pd.DataFrame) -> str:
    lines = [
        '| Rank | Model | Validation score | DACON score | DACON 1-NMAE | DACON FICR |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for row in results.itertuples(index=False):
        dacon = '-' if pd.isna(row.dacon_score) else f'{row.dacon_score:.6f}'
        dacon_mae = (
            '-' if pd.isna(row.dacon_one_minus_nmae)
            else f'{row.dacon_one_minus_nmae:.6f}'
        )
        dacon_ficr = '-' if pd.isna(row.dacon_ficr) else f'{row.dacon_ficr:.6f}'
        lines.append(
            f'| {row.rank} | {row.display_name} | {row.validation_score:.6f} '
            f'| {dacon} | {dacon_mae} | {dacon_ficr} |'
        )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--runs-dir',
        type=Path,
        default=Path(
            'model_outputs/v5/monthly_random_sigmoid_lr_0p02/runs'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/v5/monthly_random_sigmoid_lr_0p02'),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    reports = _load_reports(runs_dir)
    if not reports:
        raise FileNotFoundError('No completed model run reports were found.')

    results_path = output / 'results.csv'
    results, groups, training = _result_frames(
        reports, _preserved_dacon_metrics(results_path)
    )
    history = _history_frame(reports)
    monthly = _monthly_frame(runs_dir)
    results.to_csv(results_path, index=False, encoding='utf-8')
    groups.to_csv(output / 'group_metrics.csv', index=False, encoding='utf-8')
    training.to_csv(output / 'training_summary.csv', index=False, encoding='utf-8')
    history_path = output / 'training_history.csv'
    history_figure = figures / 'training_curves.png'
    ficr_figure = figures / 'validation_ficr.png'
    realmlp_history = history.loc[history['model'] == 'realmlp']
    ficr_available = (
        not realmlp_history.empty
        and realmlp_history['validation_ficr'].notna().any()
    )
    if history.empty:
        history_path.unlink(missing_ok=True)
    else:
        history.to_csv(history_path, index=False, encoding='utf-8')
    if realmlp_history.empty:
        history_figure.unlink(missing_ok=True)
    else:
        _plot_histories(realmlp_history, history_figure)
    if not ficr_available:
        ficr_figure.unlink(missing_ok=True)
    else:
        _plot_validation_components(realmlp_history, ficr_figure)
    monthly.to_csv(output / 'monthly_metrics.csv', index=False, encoding='utf-8')
    _plot_scores(results, figures / 'score_comparison.png')
    _plot_dacon_components(results, figures / 'dacon_components.png')
    history_section = (
        '\n\n![Training curves](figures/training_curves.png)\n\n'
        if not realmlp_history.empty else '\n\n'
    )
    if ficr_available:
        history_section += (
            '![Exact validation FICR](figures/validation_ficr.png)\n\n'
        )
    (output / 'RESULTS.md').write_text(
        '# Version 5 monthly-balanced random validation results\n\n'
        + _markdown_table(results)
        + '\n\n![Validation score](figures/score_comparison.png)\n\n'
        + '![DACON components](figures/dacon_components.png)\n\n'
        + history_section
        + 'Entered DACON score components are preserved when this report is rebuilt.\n',
        encoding='utf-8',
    )
    print(f'Consolidated {len(results)} models into {output}')


if __name__ == '__main__':
    main()
