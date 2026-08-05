'''Run selected models independently in the project Python environment.'''

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUPPORTED_MODELS = (
    'lightgbm', 'catboost', 'tabm', 'realmlp',
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train selected models independently in one environment.'
    )
    parser.add_argument(
        '--models', nargs='+', required=True,
        choices=SUPPORTED_MODELS + tuple(ALIASES),
        help='Model names, or all.',
    )
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument(
        '--runs-dir', type=Path, default=Path('model_outputs/v2/runs')
    )
    parser.add_argument('--report-dir', type=Path, default=Path('reports/v2'))
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
    runs_dir = (repo_root / args.runs_dir).resolve()
    interpreter = _python_path(repo_root)
    if not interpreter.is_file():
        raise FileNotFoundError(
            f'Missing Python 3.13 environment: {interpreter}. '
            'Run scripts/setup_env.ps1 first.'
        )

    extra_args = list(args.pipeline_args)
    if extra_args[:1] == ['--']:
        extra_args = extra_args[1:]

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
        '--output-dir', str((repo_root / args.report_dir).resolve()),
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
