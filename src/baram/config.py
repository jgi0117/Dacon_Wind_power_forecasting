'''Shared immutable pipeline configuration and CLI arguments.'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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
    comparison_start: str = '2024-07-01 01:00:00'
    seed: int = 42
    n_jobs: int = -1
    max_epochs: int = 200
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-5
    batch_size: int = 256
    learning_rate: float = 0.02
    ficr_weight: float = 0.75
    ficr_temperature: float = 0.01
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
        help='Version 4 uses one multi-task RealMLP.',
    )
    parser.add_argument('--validation-start', default='2024-01-01 01:00:00')
    parser.add_argument('--iteration-selection-end', default='2024-04-01 01:00:00')
    parser.add_argument('--comparison-start', default='2024-07-01 01:00:00')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--ficr-weight', type=float, default=0.75)
    parser.add_argument('--ficr-temperature', type=float, default=0.01)
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
            Path('model_outputs/v4/multitask_lr_0p02/runs/realmlp')
            if requested_models is None
            else Path('model_outputs/v4/multitask_lr_0p02/runs') / '_'.join(models)
        )
    return PipelineConfig(models=models, output_dir=output_dir, **values)
