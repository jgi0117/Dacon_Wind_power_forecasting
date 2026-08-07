'''Visualize the conservative ReLU-hinge surrogate for FICR.'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_figure(output: Path, margin: float) -> None:
    error = np.linspace(0.0, 0.15, 1501)
    train_6 = 0.06 - margin
    train_8 = 0.08 - margin
    hinge_6 = np.maximum(error - train_6, 0.0)
    hinge_8 = np.maximum(error - train_8, 0.0)
    weighted_6 = 0.25 * hinge_6
    weighted_8 = 0.75 * hinge_8
    relu_ficr = weighted_6 + weighted_8
    mae_component = 0.25 * error
    ficr_component = 0.75 * relu_ficr
    capacity_loss = mae_component + ficr_component
    relu_gradient = np.where(
        error <= train_6, 0.0,
        np.where(error <= train_8, 0.25, 1.0),
    )
    gradient = 0.25 + 0.75 * relu_gradient
    exact_penalty = np.select(
        [error <= 0.06, error <= 0.08], [0.0, 0.25], default=1.0
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(error, hinge_6, color='#E07B39', label='ReLU(error - 0.055)')
    axes[0].plot(error, hinge_8, color='#4E9F3D', label='ReLU(error - 0.075)')
    axes[0].set_title('1. Conservative ReLU hinges')
    axes[0].set_ylabel('Raw hinge loss')

    axes[1].plot(error, mae_component, '--', color='#E07B39', label='0.25 x MAE')
    axes[1].plot(error, ficr_component, '--', color='#4E9F3D', label='0.75 x ReLU FICR')
    axes[1].plot(error, capacity_loss, color='#2878B5', linewidth=2.2, label='Final capacity loss')
    axes[1].set_title('2. Final capacity loss')
    axes[1].set_ylabel('Loss')

    axes[2].step(
        error, gradient, where='post', color='#C43C39', linewidth=2.2,
        label='d(loss) / d(error)',
    )
    axes[2].step(
        error, exact_penalty, where='post', color='#222222', linestyle='--',
        label='Exact FICR penalty (reference)',
    )
    axes[2].set_title('3. Total gradient: 0.25 / 0.4375 / 1.0')
    axes[2].set_ylabel('Gradient / reference penalty')

    for axis in axes:
        axis.set_xlabel('Absolute capacity-factor error')
        axis.axvline(train_6, color='#E07B39', linestyle=':', linewidth=1.2)
        axis.axvline(0.06, color='#E07B39', linestyle='-', linewidth=0.8, alpha=0.5)
        axis.axvline(train_8, color='#4E9F3D', linestyle=':', linewidth=1.2)
        axis.axvline(0.08, color='#4E9F3D', linestyle='-', linewidth=0.8, alpha=0.5)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    fig.suptitle(
        'Capacity loss = 0.25 MAE + 0.75 ['
        '0.25 ReLU(e-0.055) + 0.75 ReLU(e-0.075)]'
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', type=Path,
        default=Path('docs/figures/relu_ficr_loss_mechanism.png'),
    )
    parser.add_argument('--margin', type=float, default=0.005)
    args = parser.parse_args()
    build_figure(args.output, args.margin)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
