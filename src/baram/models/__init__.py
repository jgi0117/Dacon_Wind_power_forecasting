'''Lazy model registry for independent optional-dependency execution.'''

from __future__ import annotations

from baram.config import PipelineConfig
from baram.constants import SUPPORTED_MODEL_NAMES

from .base import RegressionModel


MODEL_NAMES = SUPPORTED_MODEL_NAMES


def build_model(
    name: str,
    config: PipelineConfig,
    *,
    iterations: int | None = None,
) -> RegressionModel:
    if name == 'realmlp':
        from .realmlp_model import RealMLPModel
        return RealMLPModel(config, epochs=iterations)
    raise ValueError(f'Unsupported model: {name}')


__all__ = ['MODEL_NAMES', 'RegressionModel', 'build_model']
