'''Visualize the FICR surrogate and boundary-consistency mechanism.'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _soft_components(error: np.ndarray, temperature: float):
    soft_6 = _sigmoid((0.06 - error) / temperature)
    soft_8 = _sigmoid((0.08 - error) / temperature)
    reward = (soft_6 + 3.0 * soft_8) / 4.0
    attention = np.maximum(
        4.0 * soft_6 * (1.0 - soft_6),
        4.0 * soft_8 * (1.0 - soft_8),
    )
    return soft_6, soft_8, reward, attention


def build_figure(
    output: Path, temperature: float, consistency_weight: float
) -> None:
    error = np.linspace(0.0, 0.14, 1001)
    soft_6, soft_8, reward, attention = _soft_components(
        error, temperature
    )
    exact_reward = np.select(
        [error <= 0.06, error <= 0.08], [1.0, 0.75], default=0.0
    )

    disagreement = np.linspace(0.0, 0.05, 501)
    center_error = 0.07
    error_a = center_error - disagreement
    error_b = center_error + disagreement
    reward_a = _soft_components(error_a, temperature)[2]
    reward_b = _soft_components(error_b, temperature)[2]
    reward_variance = 0.25 * (reward_a - reward_b) ** 2

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].step(
        error, exact_reward, where='post', color='#222222',
        linewidth=2.0, label='Exact FICR reward',
    )
    axes[0].plot(error, reward, color='#2878B5', label='Soft reward')
    axes[0].plot(error, soft_6, '--', color='#E07B39', label='soft 6%')
    axes[0].plot(error, soft_8, '--', color='#4E9F3D', label='soft 8%')
    axes[0].set_title('1. Exact and differentiable FICR reward')
    axes[0].set_ylabel('Reward')

    axes[1].plot(
        error, attention, color='#9C4DCC', linewidth=2.0,
        label='Boundary attention',
    )
    axes[1].fill_between(error, 0.0, attention, color='#9C4DCC', alpha=0.18)
    axes[1].set_title('2. Detached boundary attention')
    axes[1].set_ylabel('Relative sample weight')

    axes[2].plot(
        disagreement, reward_variance, color='#C43C39',
        label='Reward variance',
    )
    axes[2].plot(
        disagreement, consistency_weight * reward_variance,
        '--', color='#222222',
        label=f'Weighted contribution (x{consistency_weight:g})',
    )
    axes[2].set_title('3. Penalty for ensemble disagreement')
    axes[2].set_xlabel('Half-gap between two errors')
    axes[2].set_ylabel('Consistency loss')
    axes[2].legend(fontsize=8)

    for axis in axes[:2]:
        axis.set_xlabel('Absolute capacity-factor error')
        axis.axvline(0.06, color='#666666', linestyle=':', linewidth=1.0)
        axis.axvline(0.08, color='#666666', linestyle=':', linewidth=1.0)
        axis.legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle(
        'FICR boundary consistency loss '
        f'(temperature={temperature:g}, weight={consistency_weight:g})'
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', type=Path,
        default=Path('docs/figures/ficr_boundary_consistency_mechanism.png'),
    )
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--consistency-weight', type=float, default=0.5)
    args = parser.parse_args()
    build_figure(args.output, args.temperature, args.consistency_weight)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
