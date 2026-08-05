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
    if name == 'lightgbm':
        from .lightgbm_model import LightGBMModel

        return LightGBMModel(config, iterations=iterations)
    if name == 'catboost':
        from .catboost_model import CatBoostModel

        return CatBoostModel(config, iterations=iterations)
    if name == 'tabm':
        from .tabm_model import TabMModel
        return TabMModel(config, epochs=iterations)
    if name == 'realmlp':
        from .realmlp_model import RealMLPModel
        return RealMLPModel(config, epochs=iterations)
    if name == 'xrfm':
        from .xrfm_model import XRFMModel
        return XRFMModel(config, iterations=iterations)
    raise ValueError(f'Unsupported model: {name}')


__all__ = ['MODEL_NAMES', 'RegressionModel', 'build_model']
