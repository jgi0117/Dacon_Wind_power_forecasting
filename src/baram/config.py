'''Shared immutable pipeline configuration and CLI arguments.'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import (
    DEFAULT_MODEL_NAMES,
    SUPPORTED_MODEL_NAMES,
)


@dataclass(frozen=True)
class PipelineConfig:
    artifacts_dir: Path = Path('artifacts')
    data_dir: Path = Path('data')
    output_dir: Path = Path('model_outputs')
    models: tuple[str, ...] = DEFAULT_MODEL_NAMES
    validation_start: str = '2024-01-01 01:00:00'
    iteration_selection_end: str = '2024-04-01 01:00:00'
    comparison_start: str = '2024-10-01 01:00:00'
    seed: int = 42
    n_jobs: int = -1
    max_epochs: int = 200
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-5
    batch_size: int = 256
    learning_rate: float = 0.02
    ficr_weight: float = 0.75
    ficr_temperature: float = 0.01
    ficr_loss: str = 'sigmoid'
    ficr_relu_margin: float = 0.005
    temporal_group_dro: bool = False
    temporal_group_dro_eta: float = 0.05
    group3_reliability_weighting: bool = False
    group3_reliability_min_weight: float = 0.2
    group3_stacking: bool = False
    temporal_prediction_correction: bool = True
    correction_validation_start: str = '2024-07-01 01:00:00'
    temporal_oof_year: int = 2024
    ficr_boundary_consistency_weight: float = 0.0
    activity_loss_weight: float = 0.15
    device: str | None = None
    evaluation_only: bool = False


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description='Evaluate selected models on temporal splits and create submissions.'
    )
    parser.add_argument('--artifacts-dir', type=Path, default=Path('artifacts'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument(
        '--models',
        nargs='+',
        choices=SUPPORTED_MODEL_NAMES + ('all',),
        default=None,
        help='Train the direct plus temporal-correction RealMLP.',
    )
    parser.add_argument('--validation-start', default='2024-01-01 01:00:00')
    parser.add_argument('--iteration-selection-end', default='2024-04-01 01:00:00')
    parser.add_argument('--comparison-start', default='2024-10-01 01:00:00')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--ficr-weight', type=float, default=0.75)
    parser.add_argument('--ficr-temperature', type=float, default=0.01)
    parser.add_argument(
        '--ficr-loss', choices=('relu', 'sigmoid'), default='sigmoid'
    )
    parser.add_argument('--ficr-relu-margin', type=float, default=0.005)
    parser.add_argument(
        '--temporal-group-dro',
        dest='temporal_group_dro',
        action='store_true',
        help='Enable the optional calendar-quarter GroupDRO FICR loss.',
    )
    parser.add_argument('--temporal-group-dro-eta', type=float, default=0.05)
    parser.add_argument(
        '--enable-group3-reliability-weighting',
        dest='group3_reliability_weighting',
        action='store_true',
        help='Enable the legacy reliability-weighted group 3 loss.',
    )
    parser.add_argument(
        '--group3-reliability-min-weight', type=float, default=0.2
    )
    parser.add_argument(
        '--disable-group3-stacking',
        dest='group3_stacking',
        action='store_false',
        default=False,
        help='Disable the temporal-OOF private group 3 RealMLP stage.',
    )
    parser.add_argument(
        '--disable-temporal-prediction-correction',
        dest='temporal_prediction_correction',
        action='store_false',
        default=True,
    )
    parser.add_argument(
        '--correction-validation-start', default='2024-07-01 01:00:00'
    )
    parser.add_argument('--temporal-oof-year', type=int, default=2024)
    parser.add_argument(
        '--ficr-boundary-consistency-weight', type=float, default=0.0
    )
    parser.add_argument('--activity-loss-weight', type=float, default=0.15)
    parser.add_argument('--device', default=None, help='cpu, cuda, or omit for auto')
    parser.add_argument('--evaluation-only', action='store_true')
    values = vars(parser.parse_args())
    positive_options = (
        'max_epochs', 'early_stopping_patience', 'batch_size',
    )
    for option in positive_options:
        if values[option] < 1:
            parser.error(f'--{option.replace('_', '-')} must be at least 1.')
    if values['early_stopping_min_delta'] < 0.0:
        parser.error('--early-stopping-min-delta must be non-negative.')
    if values['learning_rate'] <= 0.0:
        parser.error('--learning-rate must be positive.')
    if not 0.0 <= values['ficr_weight'] <= 1.0:
        parser.error('--ficr-weight must be between 0 and 1.')
    if values['ficr_temperature'] <= 0.0:
        parser.error('--ficr-temperature must be positive.')
    if not 0.0 <= values['ficr_relu_margin'] < 0.06:
        parser.error('--ficr-relu-margin must be in [0, 0.06).')
    if values['temporal_group_dro_eta'] < 0.0:
        parser.error('--temporal-group-dro-eta must be non-negative.')
    if not 0.0 < values['group3_reliability_min_weight'] <= 1.0:
        parser.error('--group3-reliability-min-weight must be in (0, 1].')
    if values['ficr_boundary_consistency_weight'] < 0.0:
        parser.error('--ficr-boundary-consistency-weight must be non-negative.')
    if values['temporal_group_dro'] and values['ficr_loss'] != 'sigmoid':
        parser.error('--temporal-group-dro requires --ficr-loss sigmoid.')
    if values['temporal_group_dro'] and values['group3_reliability_weighting']:
        parser.error(
            '--temporal-group-dro cannot be combined with group 3 reliability; '
            'do not enable group 3 reliability weighting.'
        )
    if values['group3_stacking'] and values['group3_reliability_weighting']:
        parser.error(
            'group 3 stacking replaces reliability weighting; do not enable both.'
        )
    if values['group3_stacking'] and values['ficr_loss'] != 'sigmoid':
        parser.error('group 3 stacking uses the original sigmoid FICR loss.')
    if values['group3_stacking'] and values['temporal_group_dro']:
        parser.error('group 3 stacking cannot be combined with Temporal GroupDRO.')
    if (
        values['group3_stacking']
        and values['ficr_boundary_consistency_weight'] > 0.0
    ):
        parser.error(
            'group 3 stacking cannot be combined with boundary consistency.'
        )
    if values['group3_reliability_weighting'] and values['ficr_loss'] != 'sigmoid':
        parser.error(
            'group 3 reliability requires --ficr-loss sigmoid.'
        )
    if values['activity_loss_weight'] < 0.0:
        parser.error('--activity-loss-weight must be non-negative.')
    if values['group3_stacking']:
        parser.error('temporal correction cannot use group 3 stacking.')
    correction_start = pd.Timestamp(values['correction_validation_start'])
    if not (
        pd.Timestamp(values['iteration_selection_end'])
        < correction_start
        < pd.Timestamp(values['comparison_start'])
    ):
        parser.error(
            '--iteration-selection-end < --correction-validation-start '
            '< --comparison-start is required.'
        )
    requested_models = values.pop('models')
    if requested_models is None:
        models = DEFAULT_MODEL_NAMES
    elif 'all' in requested_models:
        if requested_models != ['all']:
            parser.error('--models all cannot be combined with another model name.')
        models = SUPPORTED_MODEL_NAMES
    else:
        models = tuple(dict.fromkeys(requested_models))
    output_dir = values.pop('output_dir')
    if output_dir is None:
        output_dir = (
            Path('model_outputs/v5/temporal_oof_correction_lr_0p02/runs/realmlp')
            if requested_models is None
            else Path('model_outputs/v5/temporal_oof_correction_lr_0p02/runs') / '_'.join(models)
        )
    return PipelineConfig(models=models, output_dir=output_dir, **values)
