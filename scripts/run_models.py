'''Run Version 5 experiments with the original RealMLP structure.'''

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUPPORTED_MODELS = ('realmlp',)
ALIASES = {'all': SUPPORTED_MODELS}


def _python_path(repo_root: Path) -> Path:
    windows = repo_root / '.venv313' / 'Scripts' / 'python.exe'
    posix = repo_root / '.venv313' / 'bin' / 'python'
    return windows if windows.exists() else posix


def _expand_models(requested: list[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    for name in requested:
        expanded.extend(ALIASES.get(name, (name,)))
    return tuple(dict.fromkeys(expanded))


def _run_complete(run_dir: Path, model: str, evaluation_only: bool) -> bool:
    required = [
        run_dir / 'validation_predictions.csv',
        run_dir / 'evaluation_results.csv',
        run_dir / 'run_report.json',
    ]
    if not evaluation_only:
        required.append(run_dir / f'submission_{model}.csv')
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _experiment_name(pipeline_args: list[str]) -> str:
    learning_rate = '0.02'
    ficr_loss = 'sigmoid'
    relu_margin = '0.005'
    group_dro = False
    group3_reliability = False
    group3_stacking = False
    consistency_weight = '0.0'
    for index, argument in enumerate(pipeline_args):
        if argument == '--learning-rate' and index + 1 < len(pipeline_args):
            learning_rate = pipeline_args[index + 1]
        elif argument.startswith('--learning-rate='):
            learning_rate = argument.split('=', 1)[1]
        elif argument == '--ficr-loss' and index + 1 < len(pipeline_args):
            ficr_loss = pipeline_args[index + 1]
        elif argument.startswith('--ficr-loss='):
            ficr_loss = argument.split('=', 1)[1]
        elif (
            argument == '--ficr-relu-margin'
            and index + 1 < len(pipeline_args)
        ):
            relu_margin = pipeline_args[index + 1]
        elif argument.startswith('--ficr-relu-margin='):
            relu_margin = argument.split('=', 1)[1]
        elif argument == '--temporal-group-dro':
            group_dro = True
        elif argument == '--enable-group3-reliability-weighting':
            group3_reliability = True
        elif argument == '--disable-group3-stacking':
            group3_stacking = False
        elif (
            argument == '--ficr-boundary-consistency-weight'
            and index + 1 < len(pipeline_args)
        ):
            consistency_weight = pipeline_args[index + 1]
        elif argument.startswith('--ficr-boundary-consistency-weight='):
            consistency_weight = argument.split('=', 1)[1]
    normalized = format(float(learning_rate), 'g').replace('.', 'p')
    normalized_consistency = format(
        float(consistency_weight), 'g'
    ).replace('.', 'p')
    normalized_margin = format(float(relu_margin), 'g').replace('.', 'p')
    if not group3_stacking and ficr_loss == 'sigmoid':
        prefix = 'temporal_oof_correction'
    elif group3_stacking and ficr_loss == 'sigmoid':
        prefix = 'group3_stacked_sigmoid'
    elif group3_reliability and ficr_loss == 'sigmoid':
        prefix = 'group3_reliability_sigmoid'
    elif ficr_loss == 'relu':
        prefix = f'relu_ficr_margin_{normalized_margin}'
    elif group_dro:
        prefix = 'temporal_group_dro'
    elif float(consistency_weight) > 0.0:
        prefix = f'ficr_boundary_consistency_w_{normalized_consistency}'
    else:
        prefix = 'realmlp_activity'
    return f'{prefix}_lr_{normalized}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train Version 5 with the original RealMLP structure.'
    )
    parser.add_argument(
        '--models', nargs='+', required=True,
        choices=SUPPORTED_MODELS + tuple(ALIASES),
        help='Model names, or all.',
    )
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--runs-dir', type=Path, default=None)
    parser.add_argument('--report-dir', type=Path, default=None)
    parser.add_argument('--reuse-completed', action='store_true')
    parser.add_argument('--evaluation-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--pipeline-args', nargs=argparse.REMAINDER, default=(),
        help='Additional run_pipeline.py options; this must be the final option.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    models = _expand_models(args.models)
    interpreter = _python_path(repo_root)
    if not interpreter.is_file():
        raise FileNotFoundError(
            f'Missing Python 3.13 environment: {interpreter}. '
            'Run scripts/setup_env.ps1 first.'
        )

    extra_args = list(args.pipeline_args)
    if extra_args[:1] == ['--']:
        extra_args = extra_args[1:]
    experiment_name = _experiment_name(extra_args)
    runs_dir_arg = args.runs_dir or (
        Path('model_outputs/v5') / experiment_name / 'runs'
    )
    report_dir_arg = args.report_dir or Path('reports/v5') / experiment_name
    runs_dir = (repo_root / runs_dir_arg).resolve()

    for model in models:
        run_dir = runs_dir / model
        if args.reuse_completed and _run_complete(
            run_dir, model, args.evaluation_only
        ):
            print(f'Reuse completed run: {model} -> {run_dir}', flush=True)
            continue
        command = [
            str(interpreter),
            str(repo_root / 'run_pipeline.py'),
            '--models', model,
            '--device', args.device,
            '--output-dir', str(run_dir),
            '--seed', str(args.seed),
        ]
        if args.evaluation_only:
            command.append('--evaluation-only')
        command.extend(extra_args)
        print('> ' + subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    report_command = [
        str(interpreter), str(repo_root / 'scripts' / 'build_report.py'),
        '--runs-dir', str(runs_dir),
        '--output-dir', str((repo_root / report_dir_arg).resolve()),
    ]
    print('> ' + subprocess.list2cmdline(report_command), flush=True)
    if not args.dry_run:
        subprocess.run(report_command, check=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f'Command failed with exit code {exc.returncode}.', file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
